import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from tqdm.auto import tqdm

from .candidates import vectors

BASE_FEATURES = ["name_similarity", "address_similarity", "country_agreement"]
FEATURE_COLUMNS = [f'{field}_{suffix}' for field in ['name', 'address']
                   for suffix in ['similarity', 'exact', 'token_overlap', 'missing_left', 'missing_right',
                                  'length_left', 'length_right', 'cosine']] + [
    'country_agreement', 'country_conflict', 'core_exact', 'number_agreement', 'number_conflict',
    'generic_name_frequency', 'retrieval_score', 'candidate_rank', 'candidate_count']


def overlap(a, b):
    return len(a & b) / len(a | b) if a or b else 0.0


def build_features(queries, targets, pairs, config, vectorizers=None, name_frequencies=None, candidate_counts=None, query_matrices=None):
    left = queries.iloc[pairs.q.to_numpy()].reset_index(drop=True)
    right = targets.iloc[pairs.t.to_numpy()].reset_index(drop=True)
    values = {}
    for field in tqdm(["name", "address"], desc="Pair features", leave=False):
        valid = left[field].ne("") & right[field].ne("")
        values[f"{field}_similarity"] = process.cpdist(
            left[field].tolist(), right[field].tolist(), scorer=fuzz.ratio,
            workers=config["threads"], dtype=np.float32) / 100 * valid.to_numpy()
        values[f"{field}_exact"] = (left[field].eq(right[field]) & valid).astype(float)
        tokens = {text: set(text.split()) for text in pd.concat([queries[field], targets[field]]).unique()}
        values[f"{field}_token_overlap"] = [overlap(tokens[a], tokens[b])
                                            for a, b in zip(left[field], right[field])]
        values[f"{field}_missing_left"] = left[field].eq("").astype(float)
        values[f"{field}_missing_right"] = right[field].eq("").astype(float)
        values[f"{field}_length_left"] = left[field].str.len()
        values[f"{field}_length_right"] = right[field].str.len()
        if vectorizers is None:
            lm, rm = vectors(queries, targets, f"{field}_folded")
        elif vectorizers[field] is None:
            from scipy.sparse import csr_matrix
            lm, rm = csr_matrix((len(queries), 1)), csr_matrix((len(targets), 1))
        else:
            lm = query_matrices[field] if query_matrices is not None else vectorizers[field].transform(queries[f"{field}_folded"])
            rm = vectorizers[field].transform(targets[f"{field}_folded"])
        cosines = []
        for start in range(0, len(pairs), config["batch_size"]):
            part = pairs.iloc[start:start + config["batch_size"]]
            cosines.extend(np.asarray(lm[part.q].multiply(rm[part.t]).sum(axis=1)).ravel())
        values[f"{field}_cosine"] = cosines
    values["country_agreement"] = (left.country_norm.eq(right.country_norm) & left.country_norm.ne("")).astype(float)
    values["country_conflict"] = (left.country_norm.ne(right.country_norm) & left.country_norm.ne("") & right.country_norm.ne("")).astype(float)
    values["core_exact"] = (left.name_core.eq(right.name_core) & left.name_core.ne("")).astype(float)
    values["number_agreement"] = [float(bool(a & b)) for a, b in zip(left.numbers, right.numbers)]
    values["number_conflict"] = [float(bool(a and b and not a & b)) for a, b in zip(left.numbers, right.numbers)]
    values["generic_name_frequency"] = left.name.map(targets.name.value_counts() if name_frequencies is None else name_frequencies).fillna(0)
    values["retrieval_score"] = pairs.retrieval_score.to_numpy()
    values["candidate_rank"] = pairs["rank"].to_numpy()
    values["candidate_count"] = pairs.q.map(pairs.q.value_counts() if candidate_counts is None else candidate_counts).to_numpy()
    return pd.DataFrame(values).astype(np.float32)


def labels(queries, targets, pairs, truth):
    qids, tids = queries.entity_id.to_numpy(), targets.entity_id.to_numpy()
    return np.array([int(tids[t] in truth[qids[q]]) for q, t in pairs[["q", "t"]].itertuples(index=False, name=None)])
