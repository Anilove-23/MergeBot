# Implementation and small-sample verification

Implemented the supplied plan in `code/business_entity_resolution/` using Python
3.12.7. Installed pinned libraries with `uv pip`; a fresh environment installed all
24 dependencies using `--offline` and the local uv cache.

Implemented stages include strict TSV loading, field audit, Unicode normalization,
independent source retrieval, sparse TF-IDF top-k, token/number blocks, pair
features, logistic and LightGBM baseline comparison, full LightGBM, hard-negative
weights, entity-grouped three-fold predictions, macro F0.5 threshold selection,
country stress diagnostics, saved-model inference, TSV validation, and packaging.
Progress is visible through tqdm; expensive pure transformations use joblib caches.

The completed execution used only 80 training Source 1 entities and 80 test Source
1 entities. Training retained 284 Source 2 and 297 Source 3 records, including all
261 positive pairs and a small distractor sample. Test targets were limited to 160
records per source. The test queries included 12 France records. Original large
training TSVs were streamed only to collect the fixture's scattered labeled IDs.

Verification results:

- Seven correctness tests pass, including empty-set metrics, missing text, unseen
  countries, many-match output, entity folds, fixture reuse, and archive packaging.
- Full blocking retrieves 261/261 true pairs on this fixture.
- OOF selection macro F0.5: logistic baseline 0.913933; LightGBM baseline 0.917766;
  full LightGBM 0.978240. Selected global threshold: 0.75.
- Official supplied validator passes on OOF and test exports, with `--check-ids`.
- Fresh offline Python 3.12 environment and independently loaded saved model
  reproduce both TSV files byte-for-byte.

Artifacts, reports, fixture and smoke ZIP are under `runs/smoke/`. Reproduction
hashes are recorded in `runs/smoke/reports/reproduction.json`. Exact commands and
inference instructions are in the pipeline README.

These are execution checks on a deliberately small and easier candidate pool,
not competition performance estimates. The full dataset pipeline has not been
completed by the agent. Full-run runtime, memory and 99% blocking recall remain
unverified. The original memory execution retains full candidate/feature tables;
the disk extension below addresses that limitation. Threshold/model selection scores reuse OOF predictions;
country stress results also reuse the global threshold and are diagnostic only.
The smoke ZIP is not a leaderboard submission.

## GPU comparison extension

Both LightGBM and XGBoost are now retained and compared on identical candidate
rows, features, entity folds and hard-negative weights. Default configuration uses
CPU for both. `config/gpu.json` uses CPU LightGBM and CUDA XGBoost; the local
LightGBM wheel lacks its GPU tree learner. XGBoost 3.4.1 was available as a cached
Windows wheel and was installed with `uv pip --offline`, without downloading.
Actual CUDA training was confirmed on the RTX 3050 laptop GPU with 6 GB VRAM.

The same 80-entity smoke fixture gave full-model OOF selection macro F0.5 of
0.978240 for LightGBM and 0.980917 for XGBoost. Nine correctness tests pass. Both
models, independent frozen thresholds, OOF scores, errors and validated prediction
files are preserved in `runs/gpu_comparison/`; `reports/experiments.tsv` compares
metrics and fit timings. These tiny-sample results do not establish a general
quality advantage or GPU speedup. CPU/system RAM is still used for loading,
preprocessing, candidate generation and feature construction.

## Resumable disk preprocessing

The user's full run was stopped with permission after it reached approximately
29 GiB committed memory on a 16 GiB machine. Existing whole-frame caches remain
intact. Normalization now streams into reusable compressed Parquet partitions,
commits after each chunk, validates duplicate IDs with a disk-backed SQLite
index, and resumes after interruption. Unchanged files can be reused across
output directories and CPU/GPU/model settings without normalization. PyArrow and
filelock were installed offline from the local cache.

`--prepare-only` prepares disk checkpoints without training. Disk execution now
also uses SQLite retrieval indexes, mapped sparse TF-IDF postings and feature
arrays, partitioned candidate features and scores, streaming metrics/export,
LightGBM Sequence ingestion and XGBoost disk external-memory training. Completed
folds and final models are reusable across output directories. Disk execution
compares both full boosted models; extra minimal baselines and country holdout
experiments remain in memory execution.

Twenty-one tests pass, covering normalization interruption/resume, invalidation,
retrieval/feature parity, mapped-buffer preservation, streaming metrics and native
model checkpoint reloads. The 80-query disk run in `runs/disk_pipeline_verified`
retrieves 261/261 true pairs. OOF selection macro F0.5 is 0.972393 for CPU LightGBM
and 0.980322 for CUDA XGBoost, with thresholds 0.59 and 0.66. Both test exports pass
streaming and supplied official validation. These are small-sample diagnostics.
Another output directory reused all normalization, index, feature and trained-fold
checkpoints. Separately loaded LightGBM and XGBoost models reproduced both output
TSVs byte-for-byte; hashes are in the run's `reports/reproduction.json`.

No full-data training was restarted. Compressed LightGBM bins, labels, gradients
and GPU working buffers still need RAM/VRAM. Capacity estimates are not strict OS
limits, and full-dataset operation on 16 GB remains unverified. See the pipeline
README for preparation, resume, explicit query sampling and memory limits.
