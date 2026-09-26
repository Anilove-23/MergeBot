import json

import pandas as pd
import pytest

from ber import normalization
from ber.disk_cache import check_training_memory, checkpoint_report, prepare_source


def source(tmp_path, count=5):
    path = tmp_path / "train_source1.tsv"
    frame = pd.DataFrame({"entity_id": [f"S1-{i}" for i in range(count)],
                          "business_name": ["École GmbH", "東京 店", "", "Acme Inc", "Shop"][:count],
                          "business_address": ["12 Rue", "", "", "34 Road", "A1 4BC"][:count],
                          "country": ["France"] * count})
    frame.to_csv(path, sep="\t", index=False)
    return path, frame


def test_chunk_roundtrip_and_cross_run_hit(tmp_path, monkeypatch):
    path, raw = source(tmp_path)
    cached = prepare_source(path, tmp_path / "cache", chunk_rows=2, source=1)
    assert len(cached.manifest["parts"]) == 3
    assert cached.rows == 5
    pd.testing.assert_frame_equal(cached.load(), normalization.preprocess(raw, progress=False), check_dtype=False)
    batches = list(cached.iter_batches(batch_rows=1, columns=["entity_id", "name"]))
    assert all(len(frame) <= 1 and list(frame.columns) == ["entity_id", "name"] for frame in batches)

    def unexpected(*args, **kwargs):
        raise AssertionError("A complete checkpoint must not normalize again")

    monkeypatch.setattr(normalization, "preprocess", unexpected)
    reused = prepare_source(path, tmp_path / "cache", chunk_rows=2, source=1)
    assert reused.directory == cached.directory
    assert reused.rows == 5


def test_resume_skips_committed_partitions(tmp_path, monkeypatch):
    path, _ = source(tmp_path)
    real = normalization.preprocess
    calls = []

    def interrupt(frame, **kwargs):
        calls.append(frame.entity_id.tolist())
        if len(calls) == 2:
            raise RuntimeError("Simulated interruption")
        return real(frame, **kwargs)

    monkeypatch.setattr(normalization, "preprocess", interrupt)
    with pytest.raises(RuntimeError, match="Simulated"):
        prepare_source(path, tmp_path / "cache", chunk_rows=2, source=1)
    manifest = next((tmp_path / "cache/normalized").glob("*/manifest.json"))
    saved = json.loads(manifest.read_text())
    assert not saved["complete"] and len(saved["parts"]) == 1
    calls.clear()

    def track(frame, **kwargs):
        calls.append(frame.entity_id.tolist())
        return real(frame, **kwargs)

    monkeypatch.setattr(normalization, "preprocess", track)
    prepared = prepare_source(path, tmp_path / "cache", chunk_rows=2, source=1)
    assert prepared.manifest["complete"]
    assert calls == [["S1-2", "S1-3"], ["S1-4"]]


def test_corrupt_partition_rebuild_and_source_invalidation(tmp_path, monkeypatch):
    path, raw = source(tmp_path)
    cached = prepare_source(path, tmp_path / "cache", chunk_rows=2, source=1)
    (cached.directory / "part-000001.parquet").write_bytes(b"interrupted")
    repaired = prepare_source(path, tmp_path / "cache", chunk_rows=2, source=1)
    assert repaired.rows == 5
    pd.testing.assert_frame_equal(repaired.load(), normalization.preprocess(raw, progress=False), check_dtype=False)
    raw.loc[0, "business_name"] = "Updated business name"
    raw.to_csv(path, sep="\t", index=False)
    changed = prepare_source(path, tmp_path / "cache", chunk_rows=2, source=1)
    assert changed.directory != cached.directory
    assert changed.load().iloc[0].business_name == "Updated business name"


def test_duplicate_ids_across_chunks_fail(tmp_path):
    path, raw = source(tmp_path)
    raw.loc[4, "entity_id"] = "S1-0"
    raw.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="Duplicate entity IDs across"):
        prepare_source(path, tmp_path / "cache", chunk_rows=2, source=1)


def test_budget_estimate_does_not_materialize(tmp_path, monkeypatch):
    path, _ = source(tmp_path)
    cached = prepare_source(path, tmp_path / "cache", chunk_rows=2, source=1)
    prepared = {f"{split}_source{i}": cached for split in ["train", "test"] for i in [1, 2, 3]}
    details = check_training_memory(prepared, {"top_k": 30}, budget_gb=.000001)
    assert not details["fits_estimate"]
    assert checkpoint_report(prepared)["train_source1"]["rows"] == 5


def test_full_mode_memory_gate_precedes_raw_dataframe_load(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ber import pipeline
    from ber.disk_cache import MemoryBudgetError
    path, _ = source(tmp_path)
    cached = prepare_source(path, tmp_path / "cache", chunk_rows=2, source=1)
    prepared = {f"{split}_source{i}": cached for split in ["train", "test"] for i in [1, 2, 3]}
    monkeypatch.setattr(pipeline, "prepare_dataset", lambda *args: prepared)

    def forbidden(*args):
        raise AssertionError("Raw DataFrames must not load after failed memory estimate")

    monkeypatch.setattr(pipeline, "load_data", forbidden)
    args = SimpleNamespace(config=None, smoke=None, data=str(tmp_path), output=str(tmp_path / "output"),
                           cache_dir=str(tmp_path / "cache"), chunk_rows=2, prepare_only=False,
                           memory_budget_gb=.000001)
    with pytest.raises(MemoryBudgetError):
        pipeline.run(args)
    assert (tmp_path / "output/reports/normalization_cache.json").exists()
    assert not json.loads((tmp_path / "output/reports/memory_estimate.json").read_text())["fits_estimate"]


def test_safe_replace_retries_transient_permission_error(tmp_path, monkeypatch):
    import os
    from ber.disk_cache import safe_replace
    src, dst = tmp_path / "src.txt", tmp_path / "dst.txt"
    src.write_text("hello", encoding="utf-8")
    attempts = 0

    real_replace = os.replace

    def flaky_replace(s, d):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            err = PermissionError("Access is denied")
            err.winerror = 5
            raise err
        return real_replace(s, d)

    monkeypatch.setattr(os, "replace", flaky_replace)
    safe_replace(src, dst, max_retries=5, delay=0.01)
    assert attempts == 3
    assert dst.read_text(encoding="utf-8") == "hello"
    assert not src.exists()

