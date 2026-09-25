import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import psutil
from tqdm.auto import tqdm

from .candidates import blocking_report, generate
from .evaluation import assemble
from .export import export, package, validate
from .features import BASE_FEATURES, build_features, labels
from .loading import audit, load_data, write_fixture
from .normalization import preprocess
from .training import country_stress, cross_validate, fit_model, probability


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


@contextmanager
def stage(name, timings):
    tqdm.write(f"\n{name}")
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = {"seconds": round(time.perf_counter() - started, 3),
                         "rss_mb_at_end": round(psutil.Process().memory_info().rss / 1024 ** 2, 2)}
        tqdm.write(f"{name}: {timings[name]['seconds']:.2f}s")


def run(args):
    project = Path(__file__).resolve().parents[2]
    config = json.loads(Path(args.config or project / "config/default.json").read_text())
    if args.smoke:
        config.update(top_k=5, n_estimators=40)
    root, destination = Path(args.data).resolve(), Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    reports, artifacts = destination / "reports", destination / "artifacts"
    reports.mkdir(exist_ok=True)
    artifacts.mkdir(exist_ok=True)
    timings = {}
    memory = joblib.Memory(destination / "cache", verbose=0)
    cached_preprocess = memory.cache(preprocess)
    cached_generate = memory.cache(generate)
    cached_features = memory.cache(build_features)
    write_json(artifacts / "config.json", config)
    write_json(reports / "run.json", {"python": platform.python_version(), "smoke": args.smoke,
                                     "data": str(root), "config": config})
    with stage("Load and audit", timings):
        data, truth = load_data(root, args.smoke)
        write_json(reports / "audit.json", audit(data, truth))
        if args.smoke:
            root = destination / "fixture"
            write_fixture(data, truth, root)
        (reports / "data_quality.md").write_text(
            "# Data quality\n\n" + ("Audit covers only the small execution fixture.\n\n" if args.smoke else "") +
            "All selected IDs, source prefixes, schemas, and ground-truth membership validated.\n\n"
            "See audit.json for missing fields, countries, counts, and singleton distribution.\n", encoding="utf-8")
    with stage("Normalize", timings):
        queries = cached_preprocess(data["train"][0])
        targets = cached_preprocess(pd.concat(data["train"][1:], ignore_index=True))
        test_queries = cached_preprocess(data["test"][0])
        test_targets = cached_preprocess(pd.concat(data["test"][1:], ignore_index=True))
        queries.head(20).drop(columns="numbers").to_csv(reports / "normalization_examples.tsv", sep="\t", index=False)
    experiments = []
    oof_predictions = {}
    with stage("Quick baseline comparison", timings):
        base_pairs = cached_generate(queries, targets, config, full=False)
        base_features = cached_features(queries, targets, base_pairs, config)[BASE_FEATURES]
        base_y = labels(queries, targets, base_pairs, truth)
        write_json(reports / "baseline_blocking.json", blocking_report(queries, targets, base_pairs, truth))
        for kind in ["logistic", "lightgbm"]:
            scores, predictions, report, folds = cross_validate(kind, queries, targets, base_pairs,
                                                               base_features, base_y, truth, config)
            experiments.append({"model": f"baseline_{kind}", **report})
            oof_predictions[f"baseline_{kind}"] = predictions
            joblib.dump(fit_model(kind, base_features, base_y, config), artifacts / f"baseline_{kind}.joblib")
        baseline_winner = max(experiments, key=lambda e: e["macro_f0.5"])["model"]
        tqdm.write(f"Baseline checkpoint winner: {baseline_winner}")
    with stage("Full candidate retrieval and features", timings):
        pairs = cached_generate(queries, targets, config, full=True)
        features = cached_features(queries, targets, pairs, config)
        y = labels(queries, targets, pairs, truth)
        blocking = blocking_report(queries, targets, pairs, truth)
        write_json(reports / "blocking.json", blocking)
        if blocking["candidate_recall"] < .99:
            tqdm.write("Candidate recall is below the plan's 99% target; inspect blocking.json before a full run.")
    with stage("Main model cross-validation", timings):
        scores, predictions, report, folds = cross_validate("lightgbm", queries, targets, pairs, features,
                                                           y, truth, config, hard=True)
        experiments.append({"model": "full_lightgbm", **report})
        oof_predictions["full_lightgbm"] = predictions
        write_json(reports / "folds.json", folds)
        write_json(reports / "experiments.json", experiments)
        pd.json_normalize(experiments).to_csv(reports / "experiments.tsv", sep="\t", index=False)
        error = pairs.copy()
        error["query_id"] = queries.entity_id.to_numpy()[pairs.q]
        error["target_id"] = targets.entity_id.to_numpy()[pairs.t]
        error["truth"] = y
        error["probability"] = scores
        error["predicted"] = scores >= report["threshold"]
        error["number_conflict"] = features.number_conflict.to_numpy()
        error[error.truth.ne(error.predicted.astype(int))].to_csv(reports / "pair_errors.tsv", sep="\t", index=False)
        false_positive = error[error.truth.eq(0) & error.predicted]
        false_negative = error[error.truth.eq(1) & ~error.predicted]
        write_json(reports / "error_analysis.json", {
            "model": "full_lightgbm",
            "false_positives_by_number_conflict": false_positive.number_conflict.value_counts().to_dict(),
            "false_negatives_by_retrieval_rank": false_negative["rank"].value_counts().to_dict(),
            "blocking_misses": blocking["true_pairs"] - blocking["retrieved_true_pairs"],
        })
        missing = [(q, t) for q in truth for t in sorted(truth[q] - predictions[q])]
        pd.DataFrame(missing, columns=["query_id", "missed_target_id"]).to_csv(reports / "false_negatives.tsv", sep="\t", index=False)
        write_json(reports / "country_stress.json", country_stress(queries, targets, pairs, features, y,
                                                                   truth, config, report["threshold"]))
        model = fit_model("lightgbm", features, y, config, hard=True)
        joblib.dump(model, artifacts / "full_lightgbm.joblib")
        # Keep the shipped model MIT licensed; logistic remains an explicit comparison.
        winner = max((e for e in experiments if "lightgbm" in e["model"]), key=lambda e: e["macro_f0.5"])
        use_full = winner["model"] == "full_lightgbm"
        chosen_features = list(features.columns) if use_full else BASE_FEATURES
        policy = {"model": winner["model"], "threshold": winner["threshold"], "global_fallback": True,
                  "features": chosen_features, "full_candidates": use_full,
                  "calibration": "entity-level out-of-fold macro F0.5; selection score is not an unbiased final evaluation"}
        write_json(artifacts / "threshold_policy.json", policy)
        if not use_full:
            model = joblib.load(artifacts / f"{winner['model']}.joblib")
        joblib.dump(model, artifacts / "selected_model.joblib")
        pd.DataFrame({"entity_id": list(folds), "fold": list(folds.values())}).to_csv(reports / "entity_folds.tsv", sep="\t", index=False)
    validator = Path(args.validator) if args.validator else project / "utils/validate_submission.py"
    with stage("Validate OOF output format", timings):
        validation_dir = destination / "validation_sources"
        validation_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(data["train"], 1):
            frame.to_csv(validation_dir / f"test_source{i}.tsv", sep="\t", index=False)
        validation_output = destination / "validation_output"
        export(queries, targets, pairs if use_full else base_pairs,
               oof_predictions[winner["model"]], validation_output)
        validate(validator, validation_output, validation_dir, reports / "validation_validator.log")
    with stage("Predict and validate", timings):
        test_pairs = cached_generate(test_queries, test_targets, config, full=use_full)
        test_features = cached_features(test_queries, test_targets, test_pairs, config)[chosen_features]
        test_scores = probability(model, test_features)
        final = assemble(test_queries, test_targets, test_pairs, test_scores, policy["threshold"])
        export(test_queries, test_targets, test_pairs, final, destination / "output")
        result = validate(validator, destination / "output", root / "test", reports / "validator.log")
        tqdm.write(result.strip())
    write_json(reports / "timings.json", timings)
    (destination / "Documentation_template.md").write_text(
        "# Business Entity Resolution\n\n"
        "## Methodology\n\nOffline sparse blocking followed by a pair classifier. Raw text is retained alongside Unicode, "
        "accent-folded, and suffix-stripped variants. No external business data or network services are used.\n\n"
        "## Candidate generation\n\nExact name, sparse character TF-IDF name/address top-k, rare name tokens, "
        "and name-token plus address-number blocks. Sources 2 and 3 are retrieved separately; countries are soft features.\n\n"
        "## Models and validation\n\nLogistic regression and LightGBM compete on three baseline features; full LightGBM adds "
        "contradiction, missingness, lexical and retrieval features with hard-negative weights. Source 1 entities stay in "
        "one fold, including all their candidate pairs. Global thresholds maximize entity-level OOF macro F0.5, "
        "including empty predictions. No ID features or transitive closure.\n\n"
        f"Selected model: {winner['model']}; threshold: {winner['threshold']}; "
        f"OOF selection macro F0.5: {winner['macro_f0.5']:.6f}.\n\n"
        f"Full blocking recall: {blocking['candidate_recall']:.6f}.\n\n"
        + ("**SMOKE ONLY:** The reduced target pool is artificially easy. These scores do not estimate competition performance. "
           "Outputs cover only fixture test queries and must not be submitted.\n\n" if args.smoke else "") +
        "Threshold/model selection reuses OOF predictions; reported scores are selection estimates. Country stress tests "
        "use the already calibrated threshold and are diagnostic, not an independent estimate of France accuracy.\n\n"
        "## Reproduction and licenses\n\nSee code/business_entity_resolution/README.md and pinned requirements.txt. "
        "LightGBM is MIT licensed; scikit-learn is BSD-3-Clause. Only LightGBM models are eligible for final selection "
        "to meet the competition's model license rule; logistic regression is retained for comparison. "
        "Both are small non-neural models, far below 8 billion parameters. The supplied validator passed. "
        "A full-data performance run and full-data clean-room reproduction have not been performed.\n", encoding="utf-8")
    archive = package(project, destination)
    tqdm.write(f"Done: {destination}\nPackage: {archive}")


def main():
    parser = argparse.ArgumentParser(description="Offline entity resolution; explicitly select smoke or full mode")
    parser.add_argument("--data", required=True, help="Directory containing train/ and test/")
    parser.add_argument("--output", required=True, help="Run directory for outputs, reports, cache and artifacts")
    parser.add_argument("--config")
    parser.add_argument("--validator")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", type=int, metavar="QUERIES", help="Small fixture only; e.g. 80")
    mode.add_argument("--full", action="store_true", help="Explicitly enable full-dataset execution")
    args = parser.parse_args()
    if args.smoke is not None and args.smoke < 12:
        parser.error("--smoke must be at least 12 for entity-level folds")
    run(args)


if __name__ == "__main__":
    main()
