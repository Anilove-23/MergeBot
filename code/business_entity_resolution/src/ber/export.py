import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pandas as pd


def export(queries, targets, pairs, predictions, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    candidates = {q: set() for q in queries.entity_id}
    qids, tids = queries.entity_id.to_numpy(), targets.entity_id.to_numpy()
    valid = set(tids)
    for q, t in pairs[["q", "t"]].itertuples(index=False, name=None):
        candidates[qids[q]].add(tids[t])
    for q in qids:
        if not predictions[q] <= candidates[q] <= valid:
            raise ValueError(f"Output membership violation for {q}")
    for filename, column, values in [
        ("matching_results.tsv", "matched_entity_ids", predictions),
        ("candidate_pairs.tsv", "candidate_entity_ids", candidates),
    ]:
        pd.DataFrame([(q, ",".join(sorted(values[q]))) for q in sorted(qids)],
                     columns=["source1_entity_id", column]).to_csv(output / filename, sep="\t", index=False)


def validate(validator, output, test_dir, log):
    result = subprocess.run([sys.executable, "-X", "utf8", str(validator), "--matching", str(output / "matching_results.tsv"),
                             "--candidate", str(output / "candidate_pairs.tsv"), "--test-dir", str(test_dir), "--check-ids"],
                            capture_output=True, text=True, encoding="utf-8")
    Path(log).write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Submission validation failed: {result.stdout}\n{result.stderr}")
    return result.stdout


def package(project, run_dir):
    """Package code and validated outputs; fixture data remain outside the submission."""
    destination = run_dir / "submission.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        files = [project / name for name in ["README.md", "requirements.txt", "pyproject.toml"]]
        for directory in ["src", "config", "utils", "tests"]:
            files.extend((project / directory).rglob("*"))
        for path in sorted(files):
            if path.is_file() and not any(p in {"__pycache__", ".pytest_cache", "artifacts"} or p.endswith(".egg-info") for p in path.relative_to(project).parts):
                archive.write(path, Path("code/business_entity_resolution") / path.relative_to(project))
        for path in sorted((run_dir / "output").glob("*.tsv")):
            archive.write(path, Path("output") / path.name)
        for path in sorted((run_dir / "artifacts").glob("*")):
            archive.write(path, Path("code/business_entity_resolution/artifacts") / path.name)
        archive.write(run_dir / "Documentation_template.md", "Documentation_template.md")
    return destination
