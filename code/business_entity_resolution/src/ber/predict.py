"""Inference using a frozen model/config/threshold, without retraining."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from tqdm.auto import tqdm

from .candidates import generate
from .evaluation import assemble
from .export import export, validate
from .features import build_features
from .loading import FIELDS, read_tsv
from .normalization import preprocess
from .training import probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    artifacts, root, output = Path(args.artifacts), Path(args.test_dir), Path(args.output)
    config = json.loads((artifacts / "config.json").read_text())
    policy = json.loads((artifacts / "threshold_policy.json").read_text())
    model = joblib.load(artifacts / "selected_model.joblib")
    frames = [read_tsv(root / f"test_source{i}.tsv", FIELDS)
              for i in tqdm(range(1, 4), desc="Load inference sources")]
    queries, targets = preprocess(frames[0]), preprocess(pd.concat(frames[1:], ignore_index=True))
    pairs = generate(queries, targets, config, full=policy["full_candidates"])
    features = build_features(queries, targets, pairs, config)[policy["features"]]
    predictions = assemble(queries, targets, pairs, probability(model, features), policy["threshold"])
    export(queries, targets, pairs, predictions, output)
    validator = Path(__file__).resolve().parents[2] / "utils/validate_submission.py"
    tqdm.write(validate(validator, output, root, output / "validator.log"))


if __name__ == "__main__":
    main()
