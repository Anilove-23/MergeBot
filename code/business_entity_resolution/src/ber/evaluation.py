import numpy as np
from sklearn.model_selection import KFold
from tqdm.auto import tqdm


def entity_score(expected, actual, beta=0.5):
    if not expected and not actual:
        return 1.0
    tp = len(expected & actual)
    denominator = (1 + beta ** 2) * tp + beta ** 2 * len(expected - actual) + len(actual - expected)
    return (1 + beta ** 2) * tp / denominator if denominator else 0.0


def assemble(queries, targets, pairs, scores, threshold):
    result = {q: set() for q in queries.entity_id}
    qids, tids = queries.entity_id.to_numpy(), targets.entity_id.to_numpy()
    for q, t in pairs.loc[np.asarray(scores) >= threshold, ["q", "t"]].itertuples(index=False, name=None):
        result[qids[q]].add(tids[t])
    return result


def metrics(queries, truth, predictions):
    ids = queries.entity_id.tolist()
    singleton = [q for q in ids if not truth[q]]
    fp = sum(len(predictions[q] - truth[q]) for q in ids)
    predicted = sum(len(predictions[q]) for q in ids)
    return {
        "macro_f0.5": float(np.mean([entity_score(truth[q], predictions[q]) for q in ids])),
        "macro_f1": float(np.mean([entity_score(truth[q], predictions[q], 1) for q in ids])),
        "singleton_accuracy": float(np.mean([not predictions[q] for q in singleton])) if singleton else None,
        "false_discovery_rate": fp / predicted if predicted else 0.0,
        "entities_with_false_positive_rate": sum(bool(predictions[q] - truth[q]) for q in ids) / max(1, len(ids)),
        "false_positive_pairs": fp,
        "per_country": {str(country): float(np.mean([entity_score(truth[q], predictions[q]) for q in group.entity_id]))
                        for country, group in queries.groupby("country")},
    }


def splits(queries, config):
    if len(queries) < config["folds"]:
        raise ValueError("Need at least one query per fold")
    return list(KFold(n_splits=config["folds"], shuffle=True, random_state=config["seed"]).split(queries))


def tune(queries, targets, pairs, scores, truth):
    best = (-1.0, 0.5)
    # Include >1 to allow rejecting every candidate; prefer stricter thresholds on ties.
    for threshold in tqdm(np.r_[np.linspace(0, 1, 101), 1.000001], desc="Tune macro F0.5", leave=False):
        prediction = assemble(queries, targets, pairs, scores, threshold)
        score = np.mean([entity_score(truth[q], prediction[q]) for q in queries.entity_id])
        best = max(best, (float(score), float(threshold)))
    return best[1]
