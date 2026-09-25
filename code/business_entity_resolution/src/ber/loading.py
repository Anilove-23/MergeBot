from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

FIELDS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path, columns):
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if list(frame.columns) != columns:
        raise ValueError(f"Unexpected schema in {path}: {list(frame.columns)}")
    if frame[columns[0]].eq("").any() or frame[columns[0]].duplicated().any():
        raise ValueError(f"Missing or duplicate IDs in {path}")
    return frame


def load_data(root, smoke_count=None):
    root = Path(root)
    data = {}
    for split in tqdm(["train", "test"], desc="Load TSV files"):
        data[split] = []
        for source in range(1, 4):
            path = root / split / f"{split}_source{source}.tsv"
            if smoke_count:
                if split == "train" and source > 1:
                    frame = stream_select(path, "entity_id", positives, smoke_count * 2)
                else:
                    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                                        nrows=smoke_count if source == 1 else smoke_count * 2)
                if list(frame.columns) != FIELDS or frame.entity_id.duplicated().any():
                    raise ValueError(f"Invalid sampled schema/IDs: {path}")
            else:
                frame = read_tsv(path, FIELDS)
            if not frame.entity_id.str.startswith(f"S{source}-").all():
                raise ValueError(f"Incorrect source prefix in {split}/source{source}")
            data[split].append(frame)
            if smoke_count and split == "train" and source == 1:
                gt = stream_select(root / "train/train_ground_truth.tsv", "source1_entity_id",
                                   set(frame.entity_id), 0)
                positives = set().union(*(set(filter(None, v.split(","))) for v in gt.matched_entity_ids))
    if not smoke_count:
        gt = read_tsv(root / "train/train_ground_truth.tsv",
                      ["source1_entity_id", "matched_entity_ids"])
    truth = {q: set(filter(None, ids.split(","))) for q, ids in gt.itertuples(index=False, name=None)}
    if set(truth) != set(data["train"][0].entity_id):
        raise ValueError("Ground truth must cover exactly the training queries")
    targets = set(pd.concat(data["train"][1:]).entity_id)
    owners = {}
    for query, matches in truth.items():
        if not matches <= targets:
            raise ValueError(f"Unknown target in ground truth for {query}")
        for target in matches:
            if target in owners:
                raise ValueError("Target belongs to multiple deduplicated Source 1 entities")
            owners[target] = query
    return data, truth


def stream_select(path, key, selected, distractors):
    pieces = []
    with tqdm(desc=f"Scan {Path(path).name}", unit=" rows", unit_scale=True) as bar:
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=50_000):
            if distractors:
                pieces.append(chunk.head(distractors))
                distractors = 0
            pieces.append(chunk[chunk[key].isin(selected)])
            bar.update(len(chunk))
    return pd.concat(pieces, ignore_index=True).drop_duplicates(key).reset_index(drop=True)


def audit(data, truth):
    result = {}
    for split, sources in data.items():
        for i, frame in enumerate(sources, 1):
            result[f"{split}_source{i}"] = {
                "rows": len(frame), "duplicate_ids": int(frame.entity_id.duplicated().sum()),
                "missing_fraction": frame.eq("").mean().to_dict(),
                "countries": frame.country.value_counts().to_dict(),
            }
    result["match_count_histogram"] = pd.Series([len(v) for v in truth.values()]).value_counts().to_dict()
    known = set(data["train"][0].country)
    result["unseen_test_countries"] = sorted(set(data["test"][0].country) - known)
    return result


def write_fixture(data, truth, root):
    root = Path(root)
    for split, frames in data.items():
        (root / split).mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames, 1):
            frame.to_csv(root / split / f"{split}_source{i}.tsv", sep="\t", index=False)
    pd.DataFrame([(q, ",".join(sorted(v))) for q, v in truth.items()],
                 columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
                     root / "train/train_ground_truth.tsv", sep="\t", index=False)
