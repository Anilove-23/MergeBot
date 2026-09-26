# Business entity resolution

Python 3.12 pipeline implementing the supplied plan with pandas, scikit-learn,
LightGBM, XGBoost, RapidFuzz, sparse-dot-topn, joblib caching, and terminal tqdm progress.
No external identity lookups or data augmentation are performed.

## Resumable normalization on disk

Normalization now uses a **shared Parquet cache**, separate from each model run's
output directory. Files are read and normalized in 10,000-row chunks by default.
Each compressed chunk and its manifest are committed atomically; stopping a run
does not discard already committed chunks. Changing the output directory, model,
GPU setting or model hyperparameters does not invalidate normalized data.

Prepare all input files without loading the complete data or starting training:

```powershell
.venv/Scripts/python.exe -m ber.pipeline --data dataset/student_resource/dataset --output runs/preparation --config code/business_entity_resolution/config/gpu.json --full --prepare-only --cache-dir runs/shared_cache --chunk-rows 10000
```

Rerun exactly that command to resume after interruption. Completed files print
`CACHE HIT ... no normalization`. Partially prepared files print `RESUME`, skip
the normalization of committed chunks and continue writing the rest. Resuming a
partial TSV still scans/parses earlier CSV rows to find the resume position; a
complete file hit skips TSV parsing entirely. The named file progress bar shows
rows, committed chunks and current process RSS instead of repeating an ambiguous
two-field progress bar.

The cache key includes source content SHA-256, normalization code, format/library
versions, source number and chunk size. Unchanged path/size/modification-time
fingerprints avoid rehashing raw files. An edited source or normalization code
gets a new cache entry. Keep chunk size unchanged to reuse the same partitions.
Missing/truncated partitions are rebuilt from the first invalid partition onward.
File locks prevent two writers from modifying the same checkpoint concurrently.
Old whole-DataFrame joblib caches remain untouched but cannot be safely treated
as chunk checkpoints; the first run with this implementation builds new files.

`runs/preparation/reports/normalization_cache.json` lists cache paths, rows,
chunks, disk size and estimated materialized memory. Raw columns and normalized
columns are stored together. Programmatic consumers can read only needed rows
and columns without materializing a full DataFrame:

```python
from ber.disk_cache import prepare_source
cached = prepare_source("dataset/student_resource/dataset/train/train_source1.tsv",
                        "runs/shared_cache", chunk_rows=10000, source=1)
for batch in cached.iter_batches(batch_rows=2000, columns=["entity_id", "name"]):
    # Process this batch, then release it before the next batch.
    print(len(batch))
```

## Disk-backed retrieval, training and inference

Index text preparation now uses two bounded worker processes while a single writer
commits SQLite chunks. `index_workers` (default 2, use 1 for serial execution) and
`index_cache_mb` (default 128 MiB for the writer) are configuration options. At most
one chunk per worker is queued; worker memory is additional to the database cache.
Token, number and term inserts are ordered by their database key within each
chunk to reduce random index-page access as the database grows.
Progress shows time waiting for prepared data (`wait`) and writing/committing it
(`write`). If writes dominate, adding workers is unlikely to help. Completed
source and sparse-matrix partitions are skipped before reading them on resume.
Feature batches reuse query TF-IDF vectors and tokenize unique records once.
One remaining chunk uses serial preparation to avoid process startup. A small
21,000-target benchmark before that startup optimization took 3.4 seconds serial
and 5.2 seconds with two workers; Windows startup outweighed the parallel gain.
Full-scale speedup has not been measured. Use `index_workers: 1` for tiny datasets
or when concurrent workers compete with other applications for memory or I/O.

These performance changes preserve existing index/feature checkpoint identities.
Changing `n_estimators`, `learning_rate`, `num_leaves`, `max_bin`, model device,
or index worker/cache settings does **not** rebuild normalization, indexes or pair
features. Model changes retrain models and rescore predictions. Changing retrieval
settings such as `top_k` rebuilds features but reuses the index; changing folds or
seed rebuilds the feature store because it contains fold assignments. Input,
normalization or chunk-size changes may require rebuilding earlier stages.

An already running Python process retains the implementation it imported. These
index changes apply on the next invocation; ordinary restart reuses committed
chunks with the same data, cache directory and chunk size. The ongoing full run
was not interrupted or restarted to apply these optimizations.

`--execution-mode auto` defaults to disk execution for `--full` and memory
execution for `--smoke`. To verify the disk path with the existing tiny fixture:

```powershell
.venv/Scripts/python.exe -m ber.pipeline --data runs/smoke/fixture --output runs/disk_check --config code/business_entity_resolution/config/gpu.json --smoke 80 --execution-mode disk --chunk-rows 50
```

Disk execution stores exact/token/number indexes and corpus frequencies in SQLite,
TF-IDF sparse postings and feature arrays in memory-mapped files, and candidate
features and scores in Parquet partitions. Retrieval, scoring, metrics and export
process chunks without collecting the complete candidate/feature table into RAM.
The pinned sparse-dot-topn native algorithm receives mapped buffers directly,
avoiding a full index-array copy in its public Python wrapper.

Shared checkpoints cover normalized chunks, index partitions, sparse postings,
candidate features, training arrays, completed folds/final models and scored
partitions. Rerun with the same data, configuration and chunk size to reuse them,
even with a different output directory. Training resumes at completed folds,
not individual boosting iterations. Prediction cache keys include model weights.

LightGBM reads features through its existing `Sequence` API, then retains compressed
bins in RAM. XGBoost uses native `DataIter`/`DMatrix` disk external-memory pages;
labels, gradients and working buffers still need RAM/VRAM. Memory-mapped pages are
managed by the operating system and may appear resident while in use. Training
checks estimates against 70% of available RAM and CUDA working memory against free
VRAM. `--memory-budget-gb` adjusts an estimate, not an OS-enforced memory limit.
Do not increase it merely to bypass an out-of-memory condition.

An explicit training query subset keeps **all target records** in retrieval:

```powershell
.venv/Scripts/python.exe -m ber.pipeline --data dataset/student_resource/dataset --output runs/disk_training --full --execution-mode disk --config code/business_entity_resolution/config/gpu.json --cache-dir runs/shared_cache --chunk-rows 10000 --train-query-limit 100000
```

Remove `--train-query-limit` to train on every query; no implicit sampling occurs.
This command was not run during verification. Full-scale memory and runtime are
unverified, and the subset size is not a guarantee of fitting RAM.

Defaults are 64 query rows, 20,000 feature pairs and 100,000 training rows per
batch, configured with `disk_query_batch_rows`, `pair_feature_batch_rows` and
`training_batch_rows`. `tfidf_max_features` caps each vocabulary at 200,000, which
can affect recall. `max_candidates_per_query` raises above 10,000 instead of
silently truncating candidates. Disk training defaults to `max_bin=64` and XGBoost
`disk_max_depth=5`; compare models within the same execution mode.

Disk execution compares both full boosted models. Minimal logistic/boosted
baselines and directional country holdouts remain available in memory execution;
disk execution reports per-country OOF metrics but omits those extra experiments.
`reports/disk_stores.json` points to shared stores, and each model's
`oof_scores.json` points to partitioned OOF scores. Streaming validation checks
all query rows, IDs, duplicates, scored candidate counts and match subsets. The
bundled official validator additionally runs on small test exports (at most
10,000 queries and 100,000 targets), avoiding its full-ID RAM load on larger runs.
Disk TSVs follow source query order.

Saved disk models automatically use disk inference:

```powershell
.venv/Scripts/python.exe -m ber.predict --artifacts runs/disk_check/artifacts --test-dir runs/disk_check/fixture/test --output runs/disk_prediction --model full_xgboost --cache-dir runs/shared_cache --chunk-rows 50
```

Use `--model full_lightgbm` for the other model. CUDA external-memory training was
verified on this Windows installation; missing HMM/async-pool support produces a
performance warning, so no speedup is claimed. The original memory path still
materializes candidate/features and uses a memory estimate gate before loading.

From the repository root in PowerShell:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r code/business_entity_resolution/requirements.txt
uv pip install --python .venv/Scripts/python.exe --no-deps --no-build-isolation -e code/business_entity_resolution
.venv/Scripts/python.exe -m pytest code/business_entity_resolution/tests -q
.venv/Scripts/python.exe -m ber.pipeline --data dataset/student_resource/dataset --output runs/smoke --smoke 80
```

`uv pip` reuses its local cache automatically. `--offline` can be added to install
commands when all pinned wheels are already cached. Dependencies are pinned,
including transitive packages; Python is constrained to 3.12.

Smoke mode reads the first N training and test queries, streams ground truth and
training targets to retain complete positive match sets, and includes 2N target
distractors per source. Only those small frames enter normalization, blocking,
training or inference. TSV scanning is necessary because matches are scattered
through the source files. Reports cover the fixture only. Reduced distractor
pools make smoke scores optimistic. Smoke outputs are **not a full submission**.
The fixture can be reused without scanning the original files:

```powershell
.venv/Scripts/python.exe -m ber.pipeline --data runs/smoke/fixture --output runs/reproduce --smoke 80
```

To reproduce inference with the saved model without retraining:

```powershell
.venv/Scripts/python.exe -m ber.predict --artifacts runs/smoke/artifacts --test-dir runs/smoke/fixture/test --output runs/smoke/saved_model_output
```

## Compare LightGBM CPU and XGBoost GPU

Both models are retained. The default configuration compares both on CPU. The GPU
configuration runs LightGBM on CPU and XGBoost on CUDA, with the same candidates,
full features, entity folds, hard-negative weights, and threshold search. Logistic
regression is also retained as a three-feature CPU baseline in memory execution.

XGBoost 3.4.1 was found in the local uv cache and installed without a download:

```powershell
uv pip install --offline --python .venv/Scripts/python.exe xgboost==3.4.1
.venv/Scripts/python.exe -m ber.pipeline --data runs/smoke/fixture --output runs/gpu_comparison --config code/business_entity_resolution/config/gpu.json --smoke 80
```

The NVIDIA RTX 3050 laptop GPU has 6 GB VRAM. The cached XGBoost Windows build was
confirmed to support CUDA; the installed LightGBM Windows build does not include
its GPU learner. GPU mode verifies XGBoost's actual training device and raises an
error if it silently falls back to CPU. `max_bin=64` and shallow leaf-limited trees
reduce GPU working memory. GPU training still needs CPU/system RAM for TSV data,
text normalization, sparse retrieval, feature construction and data transfers.
VRAM does not replace that system RAM. Full-data capacity and GPU speedup are not
established by the small smoke run; initialization can dominate tiny jobs.

Read `reports/experiments.tsv` for both baseline and full-model macro F0.5, singleton
accuracy, false positives, separately tuned thresholds, OOF time and final fit time.
`reports/lightgbm/` and `reports/xgboost/` preserve each model's OOF scores, folds,
errors, country diagnostics and validator logs. Both trained models are saved as
`artifacts/full_lightgbm.joblib` and `artifacts/full_xgboost.joblib`; XGBoost also has
a native `.ubj` model. `model_outputs/lightgbm/` and `model_outputs/xgboost/` contain
separate test predictions using the same full candidate set. `output/` contains
the automatically selected boosted model's predictions; neither competing model
is removed. `reports/training_backend.json` records the confirmed device.
Each full model has its own frozen `artifacts/full_<model>_policy.json` threshold
policy. Use `ber.predict --model full_lightgbm` or `--model full_xgboost` with the
comparison artifacts to run either model independently. Both policies and the
comparison metrics are included in the packaged artifacts.

This is a model-quality comparison, not an isolated CPU-vs-GPU speed benchmark:
the algorithms differ. Threshold/model selection scores are not unbiased holdout
estimates. On this fixture all numbers are smoke-test diagnostics only.

For an explicitly authorized full run, replace `--smoke 80` with `--full` and choose
a new output directory. No full-dataset training was restarted after the memory
fix. Full-scale capacity remains unverified. Memory execution retains complete
candidate/feature tables; disk execution uses the checkpoints described above.
Tune top_k and batch_size based on measured blocking recall.

Memory execution stages: schema/audit, conservative normalization, minimal baseline comparison
(exact/name top-k; three features; logistic, LightGBM and XGBoost), full retrieval/features,
entity-level OOF scoring, global macro-F0.5 threshold selection, country stress
tests, final model fit, TSV export, supplied validator, and submission ZIP.

The full and baseline boosted models compete for final selection. Logistic remains
a regression baseline; LightGBM is MIT licensed and XGBoost is Apache-2.0 licensed.
IDs are only keys and never predictors. Missing values never
count as exact agreement. Retrieval is not filtered by country. Full candidate
generation combines exact name, name/address char n-gram top-k, rare tokens, and
name tokens plus address-number agreement. Blocking recall is measured without
injecting missed labels into validation candidate sets.

Each Source 1 entity, including its complete candidate pair set, belongs to one
fold. All folds use the same unlabeled target retrieval corpus; no validation
labels train the model. IDF is fit on target text. Thresholds and model selection
use OOF predictions, so their reported score is a selection estimate rather than
an unbiased final holdout score. Country stress tests use the globally calibrated
threshold and are directional diagnostics only. No country-specific thresholds,
forced top-1 assignments, or transitive merges are used.

Run directories contain:

- `output/`: exact candidate lists fed to the selected model and accepted matches.
- `reports/`: audit, blocking recall/counts, OOF metrics, entity folds, pair errors,
  all false negatives (including blocking misses), country stress, stage timing
  and process RSS at stage completion, and the official validator log.
- `artifacts/`: fitted baseline/main/selected models, configuration, global policy.
- `cache/`: joblib content-based cache for retrieval and features.
- `runs/shared_cache/` (outside the run directory): normalized Parquet, source
  fingerprints, SQLite indexes, mapped postings/features, fold models and scores.
- `fixture/`: selected raw TSVs when running smoke mode.
- `submission.zip`: code, pinned environment, models, two outputs and methodology.

Only load trusted joblib artifacts. A smoke ZIP covers only the fixture and must
not be uploaded to the competition. Team identity must be added to the generated
methodology document before a real submission.

The standalone packaged code can be installed from its own directory with
`uv pip install -r requirements.txt` followed by `uv pip install --no-deps
--no-build-isolation -e .`. The validator is included under `utils/`.

Algorithm references: [sparse-dot-topn](https://github.com/ing-bank/sparse_dot_topn)
and [LightGBM classifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html).
GPU backend: [XGBoost GPU support](https://xgboost.readthedocs.io/en/stable/gpu/).
