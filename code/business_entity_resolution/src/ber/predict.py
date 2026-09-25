"""Inference using a frozen model/config/threshold, without retraining."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import psutil
from tqdm.auto import tqdm

from .candidates import generate
from .disk_cache import prepare_source
from .evaluation import assemble
from .export import export, validate
from .features import build_features
from .training import probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=["selected", "full_lightgbm", "full_xgboost"], default="selected",
                        help="Use the selected winner or either retained full model")
    parser.add_argument("--cache-dir", default="runs/shared_cache")
    parser.add_argument("--chunk-rows", type=int, default=10_000)
    parser.add_argument("--memory-budget-gb", type=float)
    args = parser.parse_args()
    if args.chunk_rows <= 0 or (args.memory_budget_gb is not None and args.memory_budget_gb <= 0):
        parser.error("Chunk rows and memory budget must be positive")
    artifacts, root, output = Path(args.artifacts), Path(args.test_dir), Path(args.output)
    config = json.loads((artifacts / "config.json").read_text())
    policy_file = "threshold_policy.json" if args.model == "selected" else f"{args.model}_policy.json"
    model_file = "selected_model.joblib" if args.model == "selected" else f"{args.model}.joblib"
    policy = json.loads((artifacts / policy_file).read_text())
    model = joblib.load(artifacts / model_file)
    if policy.get('execution_mode') == 'disk':
        from .disk_pipeline import predict_saved
        return predict_saved(model, config, policy, root, output, args.cache_dir, args.chunk_rows)
    cached = [prepare_source(root / f"test_source{i}.tsv", args.cache_dir, args.chunk_rows, source=i)
              for i in range(1, 4)]
    required = (2 * sum(p["normalized_memory_bytes"] for item in cached for p in item.manifest["parts"])
                + cached[0].rows * 4 * config["top_k"] * 256)
    budget = args.memory_budget_gb * 2 ** 30 if args.memory_budget_gb is not None else psutil.virtual_memory().available * .7
    if required > budget:
        parser.exit(2, f"Normalization is checkpointed, but in-memory inference needs an estimated {required / 2**30:.1f} GiB, "
                       f"above the {budget / 2**30:.1f} GiB budget. Retrieval/scoring is not out-of-core.\n")
    queries = cached[0].load()
    targets = pd.concat([cached[1].load(), cached[2].load()], ignore_index=True)
    pairs = generate(queries, targets, config, full=policy["full_candidates"])
    features = build_features(queries, targets, pairs, config)[policy["features"]]
    predictions = assemble(queries, targets, pairs, probability(model, features), policy["threshold"])
    export(queries, targets, pairs, predictions, output)
    validator = Path(__file__).resolve().parents[2] / "utils/validate_submission.py"
    tqdm.write(validate(validator, output, root, output / "validator.log"))


if __name__ == "__main__":
    main()
