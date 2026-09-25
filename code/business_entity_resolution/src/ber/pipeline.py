import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import shutil
import time

import joblib
import numpy as np
import pandas as pd
import psutil
from tqdm.auto import tqdm

from .candidates import blocking_report, generate
from .disk_cache import MemoryBudgetError, check_training_memory, checkpoint_report, prepare_dataset
from .evaluation import assemble
from .export import export, package, validate
from .features import BASE_FEATURES, build_features, labels
from .loading import audit, load_data, write_fixture
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
    mode = getattr(args, 'execution_mode', 'memory')
    if not args.prepare_only and (mode == 'disk' or (mode == 'auto' and not args.smoke)):
        from .disk_pipeline import run as run_disk
        return run_disk(args)
    project = Path(__file__).resolve().parents[2]
    config = json.loads(Path(args.config or project / "config/default.json").read_text())
    boosted_kinds = config.get("boosted_models", ["lightgbm", "xgboost"])
    if not boosted_kinds or len(set(boosted_kinds)) != len(boosted_kinds) or not set(boosted_kinds) <= {"lightgbm", "xgboost"}:
        raise ValueError("boosted_models must contain distinct lightgbm and/or xgboost entries")
    if args.smoke:
        config.update(top_k=5, n_estimators=40)
    root, destination = Path(args.data).resolve(), Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    reports, artifacts = destination / "reports", destination / "artifacts"
    reports.mkdir(exist_ok=True)
    artifacts.mkdir(exist_ok=True)
    timings = {}
    memory = joblib.Memory(destination / "cache", verbose=0)
    cached_generate = memory.cache(generate)
    cached_features = memory.cache(build_features)
    write_json(artifacts / "config.json", config)
    write_json(reports / "run.json", {"python": platform.python_version(), "smoke": args.smoke,
                                     "data": str(root), "config": config,
                                     "normalization_cache": str(Path(args.cache_dir).resolve())})
    prepared = None
    if not args.smoke:
        # Crucially, no full CSV/DataFrame load happens before disk preparation
        # and the downstream memory gate. A killed preparation resumes by chunk.
        with stage("Prepare normalization checkpoints on disk", timings):
            prepared = prepare_dataset(root, args.cache_dir, args.chunk_rows)
            write_json(reports / "normalization_cache.json", checkpoint_report(prepared))
        if args.prepare_only:
            write_json(reports / "timings.json", timings)
            tqdm.write(f"Preparation complete. Reusable checkpoints: {Path(args.cache_dir).resolve()}")
            return
        capacity = check_training_memory(prepared, config, args.memory_budget_gb)
        write_json(reports / "memory_estimate.json", capacity)
        if not capacity["fits_estimate"]:
            write_json(reports / "timings.json", timings)
            raise MemoryBudgetError(
                f"Normalization is safely checkpointed on disk, but downstream retrieval/training is not yet out-of-core. "
                f"Estimated {capacity['estimated_downstream_bytes'] / 2**30:.1f} GiB exceeds the "
                f"{capacity['budget_bytes'] / 2**30:.1f} GiB budget. No full DataFrames were loaded. "
                "Use a small --smoke run for training; --prepare-only can prepare/cache all files without model training. "
                "See reports/memory_estimate.json. Increasing the budget does not reduce actual RAM usage."
            )
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
        if prepared is None:
            prepared = prepare_dataset(root, args.cache_dir, args.chunk_rows)
            write_json(reports / "normalization_cache.json", checkpoint_report(prepared))
        if args.prepare_only:
            write_json(reports / "timings.json", timings)
            tqdm.write(f"Preparation complete. Reusable checkpoints: {Path(args.cache_dir).resolve()}")
            return
        queries = prepared["train_source1"].load()
        targets = pd.concat([prepared[f"train_source{i}"].load() for i in [2, 3]], ignore_index=True)
        queries.head(20).drop(columns="numbers").to_csv(reports / "normalization_examples.tsv", sep="\t", index=False)
    # Raw TSV frames are not needed during fitting; the fixture/source files are
    # already on disk. Also defer all normalized test frames until prediction.
    del data
    experiments = []
    oof_predictions = {}
    with stage("Quick baseline comparison", timings):
        base_pairs = cached_generate(queries, targets, config, full=False)
        base_features = cached_features(queries, targets, base_pairs, config)[BASE_FEATURES]
        base_y = labels(queries, targets, base_pairs, truth)
        write_json(reports / "baseline_blocking.json", blocking_report(queries, targets, base_pairs, truth))
        for kind in ["logistic", *boosted_kinds]:
            started = time.perf_counter()
            scores, predictions, report, folds = cross_validate(kind, queries, targets, base_pairs,
                                                               base_features, base_y, truth, config)
            report["oof_seconds"] = round(time.perf_counter() - started, 4)
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
        backend_reports = {}
        for kind in boosted_kinds:
            full_name = f"full_{kind}"
            started = time.perf_counter()
            scores, predictions, report, folds = cross_validate(kind, queries, targets, pairs, features,
                                                               y, truth, config, hard=True)
            report["oof_seconds"] = round(time.perf_counter() - started, 4)
            oof_predictions[full_name] = predictions
            model_reports = reports / kind
            model_reports.mkdir(exist_ok=True)
            write_json(model_reports / "folds.json", folds)
            error = pairs.copy()
            error["query_id"] = queries.entity_id.to_numpy()[pairs.q]
            error["target_id"] = targets.entity_id.to_numpy()[pairs.t]
            error["truth"] = y
            error["probability"] = scores
            error["predicted"] = scores >= report["threshold"]
            error["number_conflict"] = features.number_conflict.to_numpy()
            error.to_csv(model_reports / "oof_scores.tsv", sep="\t", index=False)
            error[error.truth.ne(error.predicted.astype(int))].to_csv(model_reports / "pair_errors.tsv", sep="\t", index=False)
            false_positive = error[error.truth.eq(0) & error.predicted]
            false_negative = error[error.truth.eq(1) & ~error.predicted]
            write_json(model_reports / "error_analysis.json", {
                "model": full_name,
                "false_positives_by_number_conflict": false_positive.number_conflict.value_counts().to_dict(),
                "false_negatives_by_retrieval_rank": false_negative["rank"].value_counts().to_dict(),
                "blocking_misses": blocking["true_pairs"] - blocking["retrieved_true_pairs"],
            })
            missing = [(q, t) for q in truth for t in sorted(truth[q] - predictions[q])]
            pd.DataFrame(missing, columns=["query_id", "missed_target_id"]).to_csv(model_reports / "false_negatives.tsv", sep="\t", index=False)
            write_json(model_reports / "country_stress.json", country_stress(queries, targets, pairs, features, y,
                       truth, {**config, "boosted_model": kind}, report["threshold"]))
            started = time.perf_counter()
            model = fit_model(kind, features, y, config, hard=True)
            report["final_fit_seconds"] = round(time.perf_counter() - started, 4)
            report["training_device"] = getattr(model, "training_device_", "cpu")
            joblib.dump(model, artifacts / f"{full_name}.joblib")
            write_json(artifacts / f"{full_name}_policy.json", {
                "model": full_name, "threshold": report["threshold"], "features": list(features.columns),
                "full_candidates": True, "global_fallback": True,
                "calibration": "entity-level out-of-fold macro F0.5",
            })
            if hasattr(model, "get_booster"):
                model.get_booster().save_model(artifacts / f"{full_name}.ubj")
            experiments.append({"model": full_name, **report})
            backend_reports[kind] = {"requested_device": config.get("device", "cpu") if kind == "xgboost" else "cpu",
                                     "confirmed_training_device": report["training_device"]}
        write_json(reports / "experiments.json", experiments)
        write_json(artifacts / "model_comparison.json", experiments)
        pd.json_normalize(experiments).to_csv(reports / "experiments.tsv", sep="\t", index=False)
        # Keep the shipped model MIT/Apache licensed; logistic is a comparison.
        winner = max((e for e in experiments if any(e["model"].endswith(k) for k in boosted_kinds)), key=lambda e: e["macro_f0.5"])
        use_full = winner["model"].startswith("full_")
        chosen_features = list(features.columns) if use_full else BASE_FEATURES
        policy = {"model": winner["model"], "threshold": winner["threshold"], "global_fallback": True,
                  "features": chosen_features, "full_candidates": use_full,
                  "calibration": "entity-level out-of-fold macro F0.5; selection score is not an unbiased final evaluation"}
        write_json(artifacts / "threshold_policy.json", policy)
        model = joblib.load(artifacts / f"{winner['model']}.joblib")
        joblib.dump(model, artifacts / "selected_model.joblib")
        write_json(reports / "training_backend.json", {"backends": backend_reports,
                   "selected_model": winner["model"],
                   "selected_model_training_device": getattr(model, "training_device_", "cpu"),
                   "preprocessing_and_retrieval_device": "cpu"})
        pd.DataFrame({"entity_id": list(folds), "fold": list(folds.values())}).to_csv(reports / "entity_folds.tsv", sep="\t", index=False)
    validator = Path(args.validator) if args.validator else project / "utils/validate_submission.py"
    with stage("Validate OOF output format", timings):
        validation_dir = destination / "validation_sources"
        validation_dir.mkdir(exist_ok=True)
        for i in range(1, 4):
            shutil.copyfile(root / "train" / f"train_source{i}.tsv", validation_dir / f"test_source{i}.tsv")
        validation_output = destination / "validation_output"
        export(queries, targets, pairs if use_full else base_pairs,
               oof_predictions[winner["model"]], validation_output)
        validate(validator, validation_output, validation_dir, reports / "validation_validator.log")
    with stage("Predict and validate", timings):
        test_queries = prepared["test_source1"].load()
        test_targets = pd.concat([prepared[f"test_source{i}"].load() for i in [2, 3]], ignore_index=True)
        # Score both full models on precisely the same held-out candidate rows.
        full_test_pairs = cached_generate(test_queries, test_targets, config, full=True)
        full_test_features = cached_features(test_queries, test_targets, full_test_pairs, config)
        for kind in boosted_kinds:
            comparison_model = joblib.load(artifacts / f"full_{kind}.joblib")
            threshold = next(e["threshold"] for e in experiments if e["model"] == f"full_{kind}")
            scores = probability(comparison_model, full_test_features)
            prediction = assemble(test_queries, test_targets, full_test_pairs, scores, threshold)
            model_output = destination / "model_outputs" / kind
            export(test_queries, test_targets, full_test_pairs, prediction, model_output)
            validate(validator, model_output, root / "test", reports / kind / "validator.log")
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
        f"## Models and validation\n\nLogistic regression and {', '.join(boosted_kinds)} compete on three baseline features; the full boosted models add "
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
        "LightGBM is MIT licensed; XGBoost is Apache-2.0 licensed; scikit-learn is BSD-3-Clause. Only boosted models are eligible for final selection "
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
    parser.add_argument("--cache-dir", default="runs/shared_cache",
                        help="Persistent normalization cache shared across output directories")
    parser.add_argument("--chunk-rows", type=int, default=10_000,
                        help="Maximum rows normalized at once and per checkpoint partition")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only build/resume bounded disk normalization checkpoints; do not train")
    parser.add_argument("--memory-budget-gb", type=float,
                        help="Downstream memory estimate budget in GiB (default: 70%% of available RAM)")
    parser.add_argument('--execution-mode', choices=['auto', 'memory', 'disk'], default='auto',
                        help='auto uses disk mode for full datasets and memory mode for smoke fixtures')
    parser.add_argument('--train-query-limit', type=int,
                        help='Disk mode: explicitly sample this many training entities, retaining the complete target pool')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", type=int, metavar="QUERIES", help="Small fixture only; e.g. 80")
    mode.add_argument("--full", action="store_true", help="Explicitly enable full-dataset execution")
    args = parser.parse_args()
    if args.smoke is not None and args.smoke < 12:
        parser.error("--smoke must be at least 12 for entity-level folds")
    if args.chunk_rows < 1:
        parser.error("--chunk-rows must be positive")
    if args.memory_budget_gb is not None and args.memory_budget_gb <= 0:
        parser.error("--memory-budget-gb must be positive")
    if args.train_query_limit is not None and args.train_query_limit < 12:
        parser.error('--train-query-limit must be at least 12')
    if args.train_query_limit is not None and (args.execution_mode == 'memory' or (args.execution_mode == 'auto' and args.smoke)):
        parser.error('--train-query-limit requires disk mode')
    try:
        run(args)
    except MemoryBudgetError as exc:
        parser.exit(2, f"\nMemory check stopped this run: {exc}\n")


if __name__ == "__main__":
    main()
