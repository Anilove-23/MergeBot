"""Disk indexes and memory-mapped TF-IDF postings; no all-target DataFrame."""
from contextlib import closing
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sparse_dot_topn import api as sparse_api
from tqdm.auto import tqdm

from .disk_cache import atomic_json

INDEX_VERSION = 2


def mapped_topn(left, right, top_k, threads):
    """Use the pinned library's native kernel without its copying Python wrapper.

    sparse-dot-topn 1.2.0 calls .astype() on ALL right-hand indices even when
    their dtype already matches. That duplicates a multi-GB mapped index.
    This adapter calls the same compiled algorithm with the original buffers.
    """
    if left.nnz == 0 or right.nnz == 0 or right.shape[1] == 0:
        return csr_matrix((left.shape[0], right.shape[1]), dtype=np.float32)
    dtype = right.indices.dtype
    arguments = dict(top_n=min(top_k, right.shape[1]), nrows=left.shape[0], ncols=right.shape[1], threshold=0.,
                     A_data=left.data, A_indptr=left.indptr.astype(dtype, copy=False),
                     A_indices=left.indices.astype(dtype, copy=False), B_data=right.data,
                     B_indptr=right.indptr, B_indices=right.indices)
    if sparse_api._core._has_openmp_support:
        result = sparse_api._core.sp_matmul_topn_sorted_mt(**arguments, n_threads=max(1, threads))
    else:
        result = sparse_api._core.sp_matmul_topn_sorted(**arguments, density=1.)
    return csr_matrix(result, shape=(left.shape[0], right.shape[1]))


def connect(path):
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA cache_size=-32768")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _get(connection, key, default=0):
    row = connection.execute("SELECT value FROM progress WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def _put(connection, key, value):
    connection.execute("INSERT OR REPLACE INTO progress VALUES (?,?)", (key, json.dumps(value)))


def build_index(prepared, split, root, cache_root):
    identity = [INDEX_VERSION, version('scikit-learn'), *[prepared[f"{split}_source{i}"].directory.name for i in [1, 2, 3]]]
    if split == "train":
        from .disk_cache import source_digest
        identity.append(source_digest(Path(root) / "train/train_ground_truth.tsv", Path(cache_root)))
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
    directory = Path(cache_root) / "indexes" / key
    directory.mkdir(parents=True, exist_ok=True)
    database = directory / "index.sqlite"
    from filelock import FileLock
    with FileLock(str(directory / "writer.lock")), closing(connect(database)) as db:
        db.executescript("""
          CREATE TABLE IF NOT EXISTS progress(key TEXT PRIMARY KEY,value TEXT);
          CREATE TABLE IF NOT EXISTS records(rid INTEGER PRIMARY KEY,source INTEGER,entity_id TEXT UNIQUE,name TEXT,payload TEXT);
          CREATE INDEX IF NOT EXISTS exact_name ON records(source,name);
          CREATE INDEX IF NOT EXISTS name_frequency ON records(name);
          CREATE TABLE IF NOT EXISTS tokens(source INTEGER,token TEXT,rid INTEGER,PRIMARY KEY(source,token,rid)) WITHOUT ROWID;
          CREATE TABLE IF NOT EXISTS numbers(source INTEGER,token TEXT,rid INTEGER,PRIMARY KEY(source,token,rid)) WITHOUT ROWID;
          CREATE TABLE IF NOT EXISTS token_counts(source INTEGER,token TEXT,count INTEGER,PRIMARY KEY(source,token)) WITHOUT ROWID;
          CREATE TABLE IF NOT EXISTS terms(source INTEGER,field TEXT,term TEXT,df INTEGER,tf INTEGER,PRIMARY KEY(source,field,term)) WITHOUT ROWID;
          CREATE TABLE IF NOT EXISTS queries(qidx INTEGER PRIMARY KEY,entity_id TEXT UNIQUE,country TEXT);
          CREATE TABLE IF NOT EXISTS truth(entity_id TEXT PRIMARY KEY REFERENCES queries(entity_id),matches TEXT);
          CREATE TABLE IF NOT EXISTS owners(target TEXT PRIMARY KEY REFERENCES records(entity_id),entity_id TEXT REFERENCES queries(entity_id));
        """)
        if _get(db, "complete"):
            tqdm.write(f"INDEX CACHE HIT {split}: {directory}")
            return DiskIndex(directory, prepared, split)
        offset = 0
        for source in [2, 3]:
            cached = prepared[f"{split}_source{source}"]
            done = _get(db, f"source{source}")
            row_start = 0
            for part_no, frame in enumerate(tqdm(cached.iter_batches(batch_rows=cached.manifest['identity']['chunk_rows']),
                                                desc=f"Index {split}/source{source}", total=len(cached.manifest['parts']))):
                if part_no < done:
                    row_start += len(frame)
                    continue
                # Each SQLite transaction is a resumable source chunk.
                with db:
                    for local, record in enumerate(frame.to_dict("records")):
                        rid = offset + row_start + local
                        record["numbers"] = sorted(record["numbers"])
                        db.execute("INSERT INTO records VALUES (?,?,?,?,?)", (rid, source, record['entity_id'], record['name'], json.dumps(record, ensure_ascii=False)))
                        db.executemany("INSERT INTO tokens VALUES (?,?,?)", ((source, token, rid) for token in set(record['name_folded'].split())))
                        db.executemany("INSERT INTO numbers VALUES (?,?,?)", ((source, token, rid) for token in record['numbers']))
                    for field in ['name', 'address']:
                        vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 4))
                        try:
                            matrix = vectorizer.fit_transform(frame[f'{field}_folded'])
                        except ValueError as exc:
                            if 'empty vocabulary' in str(exc):
                                continue
                            raise
                        dfs = np.asarray(matrix.getnnz(axis=0)).ravel()
                        tfs = np.asarray(matrix.sum(axis=0)).ravel()
                        db.executemany("INSERT INTO terms VALUES (?,?,?,?,?) ON CONFLICT(source,field,term) DO UPDATE SET df=df+excluded.df,tf=tf+excluded.tf",
                                       ((source, field, term, int(df), int(tf)) for term, df, tf in zip(vectorizer.get_feature_names_out(), dfs, tfs)))
                    _put(db, f"source{source}", part_no + 1)
                row_start += len(frame)
            offset += cached.rows
        if not _get(db, 'token_counts'):
            tqdm.write(f'Aggregate {split} token frequencies on disk')
            with db:
                db.execute('INSERT OR REPLACE INTO token_counts SELECT source,token,COUNT(*) FROM tokens GROUP BY source,token')
                _put(db, 'token_counts', True)
        done = _get(db, "queries")
        start = 0
        cached = prepared[f"{split}_source1"]
        for i, frame in enumerate(tqdm(cached.iter_batches(batch_rows=cached.manifest['identity']['chunk_rows'], columns=['entity_id', 'country']),
                                       desc=f"Index {split} queries", total=len(cached.manifest['parts']))):
            if i >= done:
                with db:
                    db.executemany("INSERT INTO queries VALUES (?,?,?)", ((start + j, q, c) for j, (q, c) in enumerate(frame.itertuples(index=False, name=None))))
                    _put(db, 'queries', i + 1)
            start += len(frame)
        if split == 'train':
            done = _get(db, 'truth')
            reader = pd.read_csv(Path(root) / 'train/train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False, chunksize=10_000)
            for i, frame in enumerate(tqdm(reader, desc='Index training labels', unit='chunk')):
                if list(frame.columns) != ['source1_entity_id', 'matched_entity_ids']:
                    raise ValueError('Unexpected ground-truth schema')
                if i < done:
                    continue
                with db:
                    for qid, text in frame.itertuples(index=False, name=None):
                        matches = sorted(set(filter(None, text.split(','))))
                        db.execute('INSERT INTO truth VALUES (?,?)', (qid, json.dumps(matches)))
                        db.executemany('INSERT INTO owners VALUES (?,?)', ((target, qid) for target in matches))
                    _put(db, 'truth', i + 1)
            if db.execute('SELECT COUNT(*) FROM queries').fetchone()[0] != db.execute('SELECT COUNT(*) FROM truth').fetchone()[0]:
                raise ValueError('Training ground truth must cover every query')
        with db:
            _put(db, 'complete', True)
    return DiskIndex(directory, prepared, split)


class DiskIndex:
    def __init__(self, directory, prepared, split):
        self.directory = Path(directory)
        self.prepared, self.split = prepared, split
        self.db = connect(self.directory / 'index.sqlite')
        self.vectors = {}
        self.matrices = {}

    def vectorizer(self, field, source=None, max_features=200_000):
        key = (field, source, max_features)
        if key in self.vectors:
            return self.vectors[key]
        sql = 'SELECT term,SUM(df),SUM(tf) FROM terms WHERE field=?'
        params = [field]
        if source is not None:
            sql += ' AND source=?'
            params.append(source)
        sql += ' GROUP BY term ORDER BY SUM(tf) DESC,term LIMIT ?'
        rows = self.db.execute(sql, (*params, max_features)).fetchall()
        rows.sort(key=lambda r: r[0])
        if not rows:
            self.vectors[key] = (None, np.array([], dtype=np.int64))
            return self.vectors[key]
        counts = np.array([r[1] for r in rows], dtype=np.int64)
        sources = [source] if source else [2, 3]
        n = sum(self.prepared[f'{self.split}_source{s}'].rows for s in sources)
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), dtype=np.float32,
                              vocabulary={r[0]: i for i, r in enumerate(rows)})
        vec.fit([''])
        vec.idf_ = (np.log((1 + n) / (1 + counts)) + 1).astype(np.float32)
        self.vectors[key] = (vec, counts)
        return vec, counts

    def matrix(self, field, source, max_features):
        key = (field, source, max_features)
        if key in self.matrices:
            return self.matrices[key]
        vec, counts = self.vectorizer(field, source, max_features)
        if vec is None:
            self.matrices[key] = None
            return None
        directory = self.directory / f'matrix-{field}-{source}-{max_features}'
        directory.mkdir(exist_ok=True)
        marker = directory / 'complete.json'
        state_path = directory / 'state.joblib'
        source_cache = self.prepared[f'{self.split}_source{source}']
        from filelock import FileLock
        with FileLock(str(directory / 'writer.lock')):
            if not marker.exists():
                if state_path.exists():
                    state = joblib.load(state_path)
                    values = np.load(directory / 'values.npy', mmap_mode='r+')
                    indices = np.load(directory / 'indices.npy', mmap_mode='r+')
                    ptr = np.load(directory / 'indptr.npy', mmap_mode='r')
                else:
                    ptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
                    # SciPy uses 32-bit pointers when dimensions and nnz permit it.
                    dtype = np.int32 if int(ptr[-1]) < 2**31 and source_cache.rows < 2**31 else np.int64
                    ptr = ptr.astype(dtype)
                    np.save(directory / 'indptr.npy', ptr)
                    values = np.lib.format.open_memmap(directory / 'values.npy', mode='w+', dtype=np.float32, shape=(int(ptr[-1]),))
                    indices = np.lib.format.open_memmap(directory / 'indices.npy', mode='w+', dtype=dtype, shape=(int(ptr[-1]),))
                    state = {'done': 0, 'rows': 0, 'positions': np.zeros(len(counts), dtype=np.int64)}
                for part_no, frame in enumerate(tqdm(source_cache.iter_batches(batch_rows=source_cache.manifest['identity']['chunk_rows'], columns=[f'{field}_folded']),
                                                     desc=f'Build mapped {self.split}/S{source} {field}', total=len(source_cache.manifest['parts']))):
                    if part_no < state['done']:
                        continue
                    chunk = vec.transform(frame[f'{field}_folded']).tocoo()
                    order = np.argsort(chunk.col, kind='stable')
                    cols = chunk.col[order]
                    local_counts = np.bincount(cols, minlength=len(counts))
                    starts = np.cumsum(local_counts) - local_counts
                    within = np.arange(len(cols)) - np.repeat(starts[local_counts > 0], local_counts[local_counts > 0])
                    destination = ptr[cols] + state['positions'][cols] + within
                    values[destination] = chunk.data[order]
                    indices[destination] = chunk.row[order] + state['rows']
                    values.flush()
                    indices.flush()
                    state = {'done': part_no + 1, 'rows': state['rows'] + len(frame), 'positions': state['positions'] + local_counts}
                    joblib.dump(state, directory / 'state.tmp')
                    os.replace(directory / 'state.tmp', state_path)
                if not np.array_equal(state['positions'], counts):
                    raise ValueError('TF-IDF postings do not agree with document frequencies')
                atomic_json(marker, {'rows': source_cache.rows, 'terms': len(counts), 'nnz': int(ptr[-1])})
                del values, indices
        # sparse-dot-topn's native binding requests writable buffers even for
        # read-only input. Copy-on-write mappings satisfy it without copying the
        # full index or allowing mutations to alter the checkpoint files.
        values = np.load(directory / 'values.npy', mmap_mode='c')
        indices = np.load(directory / 'indices.npy', mmap_mode='c')
        ptr = np.load(directory / 'indptr.npy', mmap_mode='c')
        matrix = csr_matrix((values, indices, ptr), shape=(len(counts), source_cache.rows), copy=False)
        if not np.shares_memory(matrix.data, values) or not np.shares_memory(matrix.indices, indices):
            raise RuntimeError('Sparse library copied the disk index; refusing an unbounded allocation')
        self.matrices[key] = matrix
        return matrix

    def retrieve(self, queries, config):
        hits = [dict() for _ in range(len(queries))]
        limit = config.get('max_candidates_per_query', 10_000)

        def add(q, rid, score=0., rank=0):
            item = hits[q].setdefault(int(rid), [0., 0])
            item[0] = max(item[0], float(score))
            if rank and (not item[1] or rank < item[1]):
                item[1] = int(rank)
            if len(hits[q]) > limit:
                raise MemoryError(f"Query {queries.iloc[q].entity_id} exceeds {limit} candidates. No candidates were silently dropped; inspect this common-name block.")

        offset = 0
        for source in [2, 3]:
            for q, record in enumerate(queries.itertuples()):
                if record.name:
                    for rid, in self.db.execute('SELECT rid FROM records WHERE source=? AND name=?', (source, record.name)):
                        add(q, rid, 1., 1)
                tokens = sorted(set(record.name_folded.split()))
                rare = []
                for token in tokens:
                    frequency = self.db.execute('SELECT count FROM token_counts WHERE source=? AND token=?', (source, token)).fetchone()
                    count = frequency[0] if frequency else 0
                    if 0 < count <= config['rare_max_df']:
                        rare.append((count, token))
                for _, token in sorted(rare)[:config['rare_tokens']]:
                    for rid, in self.db.execute('SELECT rid FROM tokens WHERE source=? AND token=?', (source, token)):
                        add(q, rid)
                for number in sorted(record.numbers):
                    for token in tokens:
                        for rid, in self.db.execute('SELECT n.rid FROM numbers n JOIN tokens t ON n.source=t.source AND n.rid=t.rid WHERE n.source=? AND n.token=? AND t.token=?', (source, number, token)):
                            add(q, rid)
            for field in ['name', 'address']:
                vec, _ = self.vectorizer(field, source, config.get('tfidf_max_features', 200_000))
                matrix = self.matrix(field, source, config.get('tfidf_max_features', 200_000))
                if matrix is None:
                    continue
                top = mapped_topn(vec.transform(queries[f'{field}_folded']), matrix, config['top_k'], config['threads'])
                for q in range(top.shape[0]):
                    begin, end = top.indptr[q:q+2]
                    for rank, ix in enumerate(range(begin, end), 1):
                        add(q, int(top.indices[ix]) + offset, top.data[ix], rank)
            offset += self.prepared[f'{self.split}_source{source}'].rows
        return pd.DataFrame([(q, rid, score, rank) for q, match in enumerate(hits) for rid, (score, rank) in sorted(match.items())],
                            columns=['q', 'rid', 'retrieval_score', 'rank']).astype({'q': 'int64', 'rid': 'int64'})

    def records(self, rids):
        rows = []
        ordered = sorted(set(int(r) for r in rids))
        for start in range(0, len(ordered), 500):
            batch = ordered[start:start+500]
            rows.extend(self.db.execute(f"SELECT rid,payload FROM records WHERE rid IN ({','.join('?' for _ in batch)})", batch).fetchall())
        rows.sort(key=lambda r: r[0])
        frame = pd.DataFrame([json.loads(payload) for _, payload in rows])
        if len(frame):
            frame['numbers'] = frame.numbers.map(frozenset)
        return frame, {rid: i for i, (rid, _) in enumerate(rows)}

    def truth(self, ids):
        result = {}
        for q in ids:
            row = self.db.execute('SELECT matches FROM truth WHERE entity_id=?', (q,)).fetchone()
            result[q] = set(json.loads(row[0])) if row else set()
        return result

    def close(self):
        self.matrices.clear()
        self.vectors.clear()
        self.db.close()
