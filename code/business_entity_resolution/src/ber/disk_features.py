"""Resumable pair/feature partitions and disk-backed model input arrays."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

from filelock import FileLock
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from .disk_cache import atomic_json, safe_replace
from .features import FEATURE_COLUMNS, build_features

FEATURE_VERSION = 1


class FeatureStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())

    def parts(self):
        for number in range(self.manifest['done']):
            yield json.loads((self.directory / f'part-{number:06d}.json').read_text())

    def frames(self, columns=None, batch_rows=100_000):
        for part in self.parts():
            parquet = pq.ParquetFile(self.directory / part['file'])
            for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns, use_threads=False):
                yield batch.to_pandas()

    def query_metadata(self, part):
        return json.loads((self.directory / part['queries']).read_text())

    def blocking_report(self, target_rows=0):
        total_truth = total_found = 0
        counts = []
        for part in self.parts():
            for q in self.query_metadata(part):
                total_truth += q['truth_count']
                total_found += q['found']
                counts.append(q['candidate_count'])
        counts = np.array(counts) if counts else np.array([0])
        queries = len(counts)
        pairs = self.manifest['rows']
        return {
            'candidate_recall': total_found / total_truth if total_truth else 1.0,
            'true_pairs': total_truth,
            'retrieved_true_pairs': total_found,
            'pairs': pairs,
            'mean': float(counts.mean()),
            'median': float(np.median(counts)),
            'p95': float(np.percentile(counts, 95)),
            'reduction_ratio': 1 - pairs / max(1, queries * target_rows) if target_rows else 1.0
        }


def prepare_features(index, config, cache_root):
    relevant = {k: config.get(k) for k in ['seed', 'folds', 'top_k', 'rare_max_df', 'rare_tokens',
                                          'disk_query_batch_rows', 'pair_feature_batch_rows', 'tfidf_max_features',
                                          'max_candidates_per_query', 'max_block_size', 'block_token_max_df']}
    relevant['train_query_limit'] = config.get('train_query_limit') if index.split == 'train' else None
    key = hashlib.sha256(json.dumps([FEATURE_VERSION, index.directory.name, relevant,
                                    version('rapidfuzz'), version('sparse-dot-topn')], sort_keys=True).encode()).hexdigest()[:24]
    directory = Path(cache_root) / 'features' / key
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'manifest.json'
    cached_queries = index.prepared[f'{index.split}_source1']
    with FileLock(str(directory / 'writer.lock')):
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
            'done': 0, 'rows': 0, 'queries': 0, 'complete': False, 'features': FEATURE_COLUMNS, 'config': relevant}
        if manifest['complete']:
            tqdm.write(f'FEATURE CACHE HIT {index.split}: {manifest["rows"]:,} candidate pairs')
            return FeatureStore(directory)
        fold_path = directory / 'query_folds.npy'
        if not fold_path.exists():
            fold_array = np.lib.format.open_memmap(directory / 'query_folds.tmp.npy', mode='w+', dtype=np.uint8, shape=(cached_queries.rows,))
            if index.split == 'train':
                for fold, (_, valid) in enumerate(KFold(n_splits=config['folds'], shuffle=True, random_state=config['seed']).split(np.empty((cached_queries.rows, 0)))):
                    fold_array[valid] = fold
            else:
                fold_array[:] = 0
            fold_array.flush()
            del fold_array
            safe_replace(directory / 'query_folds.tmp.npy', fold_path)
        fold_array = np.load(fold_path, mmap_mode='r')
        selection = None
        limit = relevant['train_query_limit']
        if limit is not None and limit < cached_queries.rows:
            selection = np.zeros(cached_queries.rows, dtype=bool)
            selection[np.random.default_rng(config['seed']).choice(cached_queries.rows, size=limit, replace=False)] = True
        global_vecs = {field: index.vectorizer(field, None, config.get('tfidf_max_features', 200_000))[0] for field in ['name', 'address']}
        schema = pa.schema([('qid', pa.string()), ('target_id', pa.string()), ('qidx', pa.int64()),
                            ('label', pa.uint8()), ('fold', pa.uint8()), *[(n, pa.float32()) for n in FEATURE_COLUMNS]])
        atomic_json(manifest_path, manifest)
        read_batch = config.get('disk_query_batch_rows', 64) if selection is None else min(10_000, cached_queries.manifest['identity']['chunk_rows'])
        sub_batch = config.get('disk_query_batch_rows', 64)
        with tqdm(total=cached_queries.rows, initial=manifest['queries'], desc=f'{index.split} candidates/features', unit='query') as progress:
            for chunk_frame in cached_queries.iter_batches(batch_rows=read_batch, start_row=manifest['queries']):
                chunk_frame = chunk_frame.reset_index(drop=True)
                start = manifest['queries']
                scanned = len(chunk_frame)
                query_indices = np.arange(start, start + scanned)
                if selection is not None:
                    keep = selection[query_indices]
                    selected_frame = chunk_frame.loc[keep].reset_index(drop=True)
                    selected_indices = query_indices[keep]
                else:
                    selected_frame = chunk_frame
                    selected_indices = query_indices
                if selected_frame.empty:
                    manifest['queries'] = start + scanned
                    atomic_json(manifest_path, manifest)
                    progress.update(scanned)
                    continue
                for sub_start in range(0, len(selected_frame), sub_batch):
                    sub_end = min(len(selected_frame), sub_start + sub_batch)
                    queries = selected_frame.iloc[sub_start:sub_end].reset_index(drop=True)
                    sub_qindices = selected_indices[sub_start:sub_end]
                    number = manifest['done']
                    pairs = index.retrieve(queries, config)
                    counts = pairs.q.value_counts()
                    truth = index.truth(queries.entity_id)
                    found = np.zeros(len(queries), dtype=np.int64)
                    query_matrices = {field: vec.transform(queries[f'{field}_folded'])
                                      for field, vec in global_vecs.items() if vec is not None}
                    frequency = {name: index.db.execute('SELECT COUNT(*) FROM records WHERE name=?', (name,)).fetchone()[0] for name in set(queries.name)}
                    filename = f'part-{number:06d}.parquet'
                    temporary = directory / f'{filename}.tmp'
                    with pq.ParquetWriter(temporary, schema=schema, compression='zstd') as writer:
                        for begin in range(0, len(pairs), config.get('pair_feature_batch_rows', 20_000)):
                            part = pairs.iloc[begin:begin + config.get('pair_feature_batch_rows', 20_000)].copy().reset_index(drop=True)
                            targets, mapping = index.records(part.rid)
                            part['t'] = part.rid.map(mapping).astype(np.int64)
                            features = build_features(queries, targets, part, config, vectorizers=global_vecs,
                                                      name_frequencies=frequency, candidate_counts=counts, query_matrices=query_matrices)
                            qids = queries.entity_id.to_numpy()[part.q]
                            tids = targets.entity_id.to_numpy()[part.t]
                            labels = np.array([t in truth[q] for q, t in zip(qids, tids)], dtype=np.uint8)
                            np.add.at(found, part.q.to_numpy(), labels)
                            features.insert(0, 'fold', np.asarray(fold_array[sub_qindices[part.q.to_numpy()]]))
                            features.insert(0, 'label', labels)
                            features.insert(0, 'qidx', sub_qindices[part.q.to_numpy()])
                            features.insert(0, 'target_id', tids)
                            features.insert(0, 'qid', qids)
                            writer.write_table(pa.Table.from_pandas(features, schema=schema, preserve_index=False))
                    safe_replace(temporary, directory / filename)
                    query_file = f'queries-{number:06d}.json'
                    query_meta = [{'qid': row.entity_id, 'qidx': int(sub_qindices[i]), 'country': row.country,
                                   'truth_count': len(truth[row.entity_id]), 'found': int(found[i]),
                                   'candidate_count': int(counts.get(i, 0)), 'fold': int(fold_array[sub_qindices[i]])}
                                  for i, row in enumerate(queries.itertuples())]
                    atomic_json(directory / query_file, query_meta)
                    spec = {'file': filename, 'queries': query_file, 'rows': len(pairs), 'query_count': len(queries), 'qstart': int(sub_qindices[0]), 'qstop': int(sub_qindices[-1]) + 1}
                    atomic_json(directory / f'part-{number:06d}.json', spec)
                    manifest.update(done=number + 1, rows=manifest['rows'] + len(pairs))
                manifest['queries'] = start + scanned
                atomic_json(manifest_path, manifest)
                progress.update(scanned)
        manifest['complete'] = True
        atomic_json(manifest_path, manifest)
    return FeatureStore(directory)


class DiskMatrix:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.x = np.load(self.directory / 'x.npy', mmap_mode='r')
        self.y = np.load(self.directory / 'y.npy', mmap_mode='r')
        self.weight = np.load(self.directory / 'weight.npy', mmap_mode='r')
        self.fold = np.load(self.directory / 'fold.npy', mmap_mode='r')

    def __len__(self):
        return len(self.y)


def prepare_matrix(store, config):
    # The hard-negative weight affects model input, not retrieval or raw features.
    weight = config.get('hard_negative_weight', 2.)
    directory = store.directory / f'matrix-weight-{weight:g}'
    directory.mkdir(exist_ok=True)
    state_path = directory / 'state.json'
    with FileLock(str(directory / 'writer.lock')):
        state = json.loads(state_path.read_text()) if state_path.exists() else {'done': 0, 'rows': 0, 'complete': False}
        if state['complete']:
            return DiskMatrix(directory)
        count, columns = store.manifest['rows'], len(FEATURE_COLUMNS)
        arrays = {}
        for name, dtype, shape in [('x', np.float32, (count, columns)), ('y', np.uint8, (count,)),
                                   ('weight', np.float32, (count,)), ('fold', np.uint8, (count,))]:
            path = directory / f'{name}.npy'
            arrays[name] = np.load(path, mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
        for number, part in enumerate(tqdm(store.parts(), total=store.manifest['done'], desc='Write disk model matrix', unit='chunk')):
            if number < state['done']:
                continue
            cursor = state['rows']
            for batch in pq.ParquetFile(store.directory / part['file']).iter_batches(batch_size=config.get('training_batch_rows', 100_000),
                         columns=[*FEATURE_COLUMNS, 'label', 'fold'], use_threads=False):
                frame = batch.to_pandas()
                stop = cursor + len(frame)
                arrays['x'][cursor:stop] = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
                arrays['y'][cursor:stop] = frame.label.to_numpy()
                arrays['fold'][cursor:stop] = frame.fold.to_numpy()
                arrays['weight'][cursor:stop] = np.where(frame.label.eq(0) & frame.name_similarity.ge(.75), weight, 1.)
                cursor = stop
            for array in arrays.values():
                array.flush()
            state.update(done=number + 1, rows=cursor)
            atomic_json(state_path, state)
        state['complete'] = True
        atomic_json(state_path, state)
        del arrays
    return DiskMatrix(directory)
