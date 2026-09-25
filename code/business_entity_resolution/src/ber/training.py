import numpy as np
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .evaluation import assemble, metrics, splits, tune


def fit_model(kind, features, y, config, hard=False):
    if not len(y):
        raise ValueError("No training candidates; increase the sample or retrieval top_k")
    if len(np.unique(y)) < 2:
        model = DummyClassifier(strategy="constant", constant=int(y[0]))
        return model.fit(features, y)
    weight = np.ones(len(y))
    if hard:
        weight[(y == 0) & (features.name_similarity.to_numpy() >= .75)] = config["hard_negative_weight"]
    if kind == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=config["seed"]))
        model.fit(features, y, logisticregression__sample_weight=weight)
    else:
        model = LGBMClassifier(n_estimators=config["n_estimators"], num_leaves=config["num_leaves"],
                               learning_rate=config["learning_rate"], random_state=config["seed"],
                               n_jobs=config["threads"], verbosity=-1, deterministic=True,
                               force_col_wise=True, min_child_samples=10)
        with tqdm(total=config["n_estimators"], desc="LightGBM iterations", leave=False) as bar:
            model.fit(features, y, sample_weight=weight, callbacks=[lambda env: bar.update(1)])
    return model


def probability(model, features):
    if features.empty:
        return np.empty(0)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(features))
    return model.predict_proba(features)[:, classes.index(1)]


def cross_validate(kind, queries, targets, pairs, features, y, truth, config, hard=False):
    scores = np.zeros(len(pairs))
    fold_assignment = {}
    for fold, (train, valid) in enumerate(tqdm(splits(queries, config), desc=f"OOF {kind}")):
        train_mask, valid_mask = pairs.q.isin(train).to_numpy(), pairs.q.isin(valid).to_numpy()
        model = fit_model(kind, features.loc[train_mask], y[train_mask], config, hard)
        scores[valid_mask] = probability(model, features.loc[valid_mask])
        fold_assignment.update({queries.iloc[int(i)].entity_id: fold for i in valid})
    threshold = tune(queries, targets, pairs, scores, truth)
    predictions = assemble(queries, targets, pairs, scores, threshold)
    report = metrics(queries, truth, predictions)
    report["threshold"] = threshold
    # Pair-level FPR is distinct from false-discovery rate.
    report["candidate_false_positive_rate"] = float(((scores >= threshold) & (y == 0)).sum() / max(1, (y == 0).sum()))
    return scores, predictions, report, fold_assignment


def country_stress(queries, targets, pairs, features, y, truth, config, threshold):
    reports = {}
    for origin, destination in tqdm([("US", "India"), ("India", "US")], desc="Country holdouts"):
        train = pairs.q.isin(queries.index[queries.country.eq(origin)]).to_numpy()
        valid = pairs.q.isin(queries.index[queries.country.eq(destination)]).to_numpy()
        qvalid = queries[queries.country.eq(destination)]
        if not train.any() or qvalid.empty:
            reports[f"{origin}->{destination}"] = {"skipped": "Insufficient sampled country records"}
            continue
        model = fit_model("lightgbm", features.loc[train], y[train], config, hard=True)
        selected = pairs.loc[valid]
        predictions = assemble(queries, targets, selected, probability(model, features.loc[valid]), threshold)
        reports[f"{origin}->{destination}"] = metrics(qvalid, truth, predictions)
    return reports
