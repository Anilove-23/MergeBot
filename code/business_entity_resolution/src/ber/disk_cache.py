"""Bounded, resumable normalization checkpoints, shared between model runs.

Parquet holds raw and normalized columns together. Completed partitions are
immutable, and a manifest is committed atomically after each partition.
"""
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
import sqlite3

from filelock import FileLock
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from . import normalization
from .loading import FIELDS

CACHE_FORMAT = 1


class MemoryBudgetError(MemoryError):
    """An expected preflight refusal, rather than an allocation failure."""


def atomic_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def source_digest(path, cache_root):
    """Hash changed sources once; unchanged sources use their stored fingerprint."""
    directory = cache_root / "source_fingerprints"
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
    entry = directory / f"{key}.json"
    signature = _fingerprint(path)
    with FileLock(str(entry) + ".lock"):
        if entry.exists():
            cached = _read_json(entry)
            if cached["fingerprint"] == signature:
                return cached["sha256"]
        digest = hashlib.sha256()
        with path.open("rb") as handle, tqdm(total=signature["size"], unit="B", unit_scale=True,
                                               desc=f"Fingerprint {path.name}", leave=False) as bar:
            while block := handle.read(4 * 1024 * 1024):
                digest.update(block)
                bar.update(len(block))
        if _fingerprint(path) != signature:
            raise RuntimeError(f"Source changed while hashing: {path}")
        atomic_json(entry, {"fingerprint": signature, "sha256": digest.hexdigest()})
        return digest.hexdigest()


def _normalization_signature():
    return hashlib.sha256(inspect.getsource(normalization).encode()).hexdigest()


def _valid_part(directory, part):
    path = directory / part["file"]
    try:
        return path.stat().st_size == part["bytes"] and pq.ParquetFile(path).metadata.num_rows == part["rows"]
    except (OSError, pa.ArrowException):
        return False


class NormalizedFile:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = _read_json(self.directory / "manifest.json")

    @property
    def rows(self):
        return sum(p["rows"] for p in self.manifest["parts"])

    def iter_batches(self, batch_rows=10_000, columns=None, start_row=0):
        """Read only requested columns and at most batch_rows records at a time."""
        if not self.manifest["complete"]:
            raise RuntimeError("Checkpoint is incomplete; call prepare_source to resume it")
        if batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        cursor = 0
        for part in self.manifest["parts"]:
            if cursor + part['rows'] <= start_row:
                cursor += part['rows']
                continue
            parquet = pq.ParquetFile(self.directory / part["file"], memory_map=True)
            for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns, use_threads=False):
                frame = batch.to_pandas()
                end = cursor + len(frame)
                if end <= start_row:
                    cursor = end
                    continue
                if cursor < start_row:
                    frame = frame.iloc[start_row - cursor:].reset_index(drop=True)
                cursor = end
                if "numbers" in frame:
                    frame["numbers"] = frame.numbers.map(frozenset)
                yield frame

    def load(self):
        """Compatibility materialization; caller MUST pass the pipeline memory gate."""
        frames = list(self.iter_batches())
        if not frames:
            return normalization.preprocess(pd.DataFrame(columns=FIELDS), progress=False)
        return pd.concat(frames, ignore_index=True)


def _validate_unique_ids(directory, parts):
    # SQLite's unique index keeps the global ID set on disk, not in a Python set.
    database = directory / "validation.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA cache_size = -8192")
        connection.execute("DROP TABLE IF EXISTS ids")
        connection.execute("CREATE TABLE ids (entity_id TEXT PRIMARY KEY) WITHOUT ROWID")
        for part in tqdm(parts, desc="Validate checkpoint IDs", unit="chunk", leave=False):
            table = pq.read_table(directory / part["file"], columns=["entity_id"], use_threads=False)
            try:
                connection.executemany("INSERT INTO ids VALUES (?)", ((v,) for v in table.column(0).to_pylist()))
            except sqlite3.IntegrityError as exc:
                raise ValueError("Duplicate entity IDs across source chunks") from exc
            connection.commit()


def prepare_source(path, cache_root, chunk_rows=10_000, source=None):
    path, cache_root = Path(path).resolve(), Path(cache_root).resolve()
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    cache_root.mkdir(parents=True, exist_ok=True)
    signature = _fingerprint(path)
    identity = {"source_sha256": source_digest(path, cache_root), "normalizer": _normalization_signature(),
                "format": CACHE_FORMAT, "chunk_rows": chunk_rows, "source": source,
                "pandas": pd.__version__, "pyarrow": pa.__version__}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    directory = cache_root / "normalized" / f"{path.stem}-{key}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    with FileLock(str(directory / "writer.lock")):
        manifest = _read_json(manifest_path) if manifest_path.exists() else {
            "identity": identity, "complete": False, "parts": [], "source_name": path.name}
        if manifest["identity"] != identity:
            raise RuntimeError("Checkpoint identity mismatch")
        # Recover a missing/truncated partition and recompute only that suffix.
        valid = 0
        for part in manifest["parts"]:
            if not _valid_part(directory, part):
                break
            valid += 1
        if valid != len(manifest["parts"]):
            manifest["parts"] = manifest["parts"][:valid]
            manifest["complete"] = False
            atomic_json(manifest_path, manifest)
        if manifest["complete"]:
            tqdm.write(f"CACHE HIT {path.name}: {sum(p['rows'] for p in manifest['parts']):,} rows, no normalization")
            return NormalizedFile(directory)
        completed_rows = sum(p["rows"] for p in manifest["parts"])
        if completed_rows:
            tqdm.write(f"RESUME {path.name}: {valid} committed chunks / {completed_rows:,} rows; skipping their normalization")
        else:
            tqdm.write(f"CACHE MISS {path.name}: checkpoint every {chunk_rows:,} rows")
        atomic_json(manifest_path, manifest)
        reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk_rows)
        with reader, tqdm(desc=f"Normalize {path.name}", unit="row", unit_scale=True,
                          initial=completed_rows) as bar:
            # CSV parsing must scan earlier rows after an interruption, but completed
            # partitions are never normalized again. A complete hit skips CSV entirely.
            for index, chunk in enumerate(reader):
                if index < valid:
                    continue
                if list(chunk.columns) != FIELDS:
                    raise ValueError(f"Unexpected schema in {path}")
                if chunk.entity_id.eq("").any() or chunk.entity_id.duplicated().any():
                    raise ValueError(f"Missing or duplicate entity IDs in {path}")
                if source is not None and not chunk.entity_id.str.startswith(f"S{source}-").all():
                    raise ValueError(f"Incorrect source prefix in {path}")
                normalized = normalization.preprocess(chunk, progress=False)
                memory_bytes = int(normalized.memory_usage(index=True, deep=True).sum())
                raw_bytes = int(chunk.memory_usage(index=True, deep=True).sum())
                normalized["numbers"] = normalized.numbers.map(lambda v: sorted(v))
                table = pa.Table.from_pandas(normalized, preserve_index=False)
                # Empty lists must still have a stable string-element schema.
                number_index = table.schema.get_field_index("numbers")
                table = table.set_column(number_index, "numbers", pa.array(normalized.numbers.tolist(), type=pa.list_(pa.string())))
                filename = f"part-{index:06d}.parquet"
                temporary = directory / f"{filename}.tmp"
                pq.write_table(table, temporary, compression="zstd", row_group_size=chunk_rows)
                with temporary.open("rb+") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, directory / filename)
                part = {"file": filename, "rows": len(chunk), "bytes": (directory / filename).stat().st_size,
                        "normalized_memory_bytes": memory_bytes, "raw_memory_bytes": raw_bytes,
                        "missing_counts": {k: int(v) for k, v in chunk.eq("").sum().items()},
                        "countries": {str(k): int(v) for k, v in chunk.country.value_counts().items()}}
                manifest["parts"].append(part)
                atomic_json(manifest_path, manifest)
                bar.update(len(chunk))
                bar.set_postfix(chunks=len(manifest["parts"]), rss_MB=round(psutil.Process().memory_info().rss / 2 ** 20))
                del normalized, table, chunk
        if _fingerprint(path) != signature:
            # Never bless results produced while their source was being modified.
            manifest["parts"] = []
            atomic_json(manifest_path, manifest)
            raise RuntimeError(f"Source changed during normalization: {path}; rerun to create a new checkpoint")
        _validate_unique_ids(directory, manifest["parts"])
        manifest["complete"] = True
        atomic_json(manifest_path, manifest)
        return NormalizedFile(directory)


def prepare_dataset(root, cache_root, chunk_rows=10_000):
    prepared = {}
    for split in ["train", "test"]:
        for source in range(1, 4):
            name = f"{split}_source{source}"
            prepared[name] = prepare_source(Path(root) / split / f"{name}.tsv", cache_root, chunk_rows, source)
    return prepared


def checkpoint_report(prepared):
    result = {}
    for name, cached in prepared.items():
        countries, missing = Counter(), Counter()
        for part in cached.manifest["parts"]:
            countries.update(part["countries"])
            missing.update(part["missing_counts"])
        result[name] = {"rows": cached.rows, "chunks": len(cached.manifest["parts"]),
                        "directory": str(cached.directory), "complete": cached.manifest["complete"],
                        "disk_bytes": sum(p["bytes"] for p in cached.manifest["parts"]),
                        "normalized_memory_bytes": sum(p["normalized_memory_bytes"] for p in cached.manifest["parts"]),
                        "raw_memory_bytes": sum(p["raw_memory_bytes"] for p in cached.manifest["parts"]),
                        "countries": dict(countries), "missing_counts": dict(missing)}
    return result


def check_training_memory(prepared, config, budget_gb=None):
    """Fail before the legacy in-memory model stages can exhaust a laptop.

    This is deliberately an estimate, not a guarantee/OS allocation limit.
    Rare-token blocks and vocabulary size can add more memory than predicted.
    """
    report = checkpoint_report(prepared)
    normalized = sum(v["normalized_memory_bytes"] for v in report.values())
    raw = sum(v["raw_memory_bytes"] for v in report.values())
    queries = report["train_source1"]["rows"]
    # Two target sources, two top-k channels; features, indices, OOF masks/copies.
    candidate_estimate = queries * 4 * config["top_k"] * 256
    required = 2 * normalized + raw + candidate_estimate
    available = psutil.virtual_memory().available
    budget = int(budget_gb * 2 ** 30) if budget_gb is not None else int(available * .7)
    details = {"estimated_downstream_bytes": required, "budget_bytes": budget,
               "available_bytes": available, "normalized_bytes": normalized,
               "raw_bytes": raw, "candidate_estimate_bytes": candidate_estimate,
               "fits_estimate": required <= budget,
               "scope": "normalization is chunked; retrieval/features/model fitting still materialize data"}
    return details
