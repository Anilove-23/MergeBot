from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from tqdm.auto import tqdm


def vectors(queries, targets, field):
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), dtype=np.float32)
    try:
        right = vectorizer.fit_transform(targets[field])
        left = vectorizer.transform(queries[field])
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):
            raise
        left, right = csr_matrix((len(queries), 1)), csr_matrix((len(targets), 1))
    return left, right


def generate(queries, targets, config, full=True):
    """Union independent source-specific retrievals without a dense Cartesian matrix."""
    rows = []
    for prefix in tqdm(["S2-", "S3-"], desc="Retrieve candidates", leave=False):
        positions = np.flatnonzero(targets.entity_id.str.startswith(prefix).to_numpy())
        right = targets.iloc[positions].reset_index(drop=True)
        if right.empty:
            continue
        hits = [dict() for _ in range(len(queries))]

        def add(q, r, score=0.0, rank=0):
            existing = hits[q].setdefault(int(r), [0.0, 0])
            existing[0] = max(existing[0], float(score))
            if rank and (not existing[1] or rank < existing[1]):
                existing[1] = rank

        exact, token_index, number_index = defaultdict(list), defaultdict(set), defaultdict(set)
        for r, record in enumerate(right.itertuples()):
            if record.name:
                exact[record.name].append(r)
            if full:
                for token in set(record.name_folded.split()):
                    token_index[token].add(r)
                for number in record.numbers:
                    number_index[number].add(r)
        for q, record in enumerate(queries.itertuples()):
            for r in exact.get(record.name, []):
                add(q, r, 1.0, 1)
            if full:
                tokens = set(record.name_folded.split())
                rare = sorted((t for t in tokens if 0 < len(token_index[t]) <= config["rare_max_df"]),
                              key=lambda t: (len(token_index[t]), t))[:config["rare_tokens"]]
                for token in rare:
                    for r in sorted(token_index[token]):
                        add(q, r)
                token_hits = set().union(*(token_index[t] for t in tokens))
                number_hits = set().union(*(number_index[n] for n in record.numbers))
                for r in sorted(token_hits & number_hits):
                    add(q, r)
        for field in (["name_folded", "address_folded"] if full else ["name_folded"]):
            left_matrix, right_matrix = vectors(queries, right, field)
            for start in tqdm(range(0, len(queries), config["batch_size"]),
                              desc=f"Sparse top-k {prefix}{field}", leave=False):
                top = sp_matmul_topn(left_matrix[start:start + config["batch_size"]],
                                    right_matrix.T.tocsr(), top_n=min(config["top_k"], len(right)),
                                    threshold=0.0, sort=True, n_threads=config["threads"])
                for local in range(top.shape[0]):
                    begin, end = top.indptr[local:local + 2]
                    for rank, offset in enumerate(range(begin, end), 1):
                        add(start + local, top.indices[offset], top.data[offset], rank)
        for q, candidates in enumerate(hits):
            for r, (score, rank) in sorted(candidates.items()):
                rows.append((q, int(positions[r]), score, rank))
    pairs = pd.DataFrame(rows, columns=["q", "t", "retrieval_score", "rank"])
    return pairs.astype({"q": "int64", "t": "int64"})


def blocking_report(queries, targets, pairs, truth):
    retrieved = defaultdict(set)
    for q, t in pairs[["q", "t"]].itertuples(index=False, name=None):
        retrieved[queries.iloc[q].entity_id].add(targets.iloc[t].entity_id)
    total = sum(len(truth[q]) for q in queries.entity_id)
    found = sum(len(truth[q] & retrieved[q]) for q in queries.entity_id)
    counts = np.array([len(retrieved[q]) for q in queries.entity_id])
    return {"candidate_recall": found / total if total else 1.0,
            "true_pairs": total, "retrieved_true_pairs": found, "pairs": len(pairs),
            "mean": float(counts.mean()), "median": float(np.median(counts)),
            "p95": float(np.percentile(counts, 95)),
            "reduction_ratio": 1 - len(pairs) / max(1, len(queries) * len(targets))}
