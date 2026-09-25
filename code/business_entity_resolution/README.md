# Business entity resolution

Python 3.12 pipeline implementing the supplied plan with pandas, scikit-learn,
LightGBM, RapidFuzz, sparse-dot-topn, joblib caching, and terminal tqdm progress.
No external identity lookups or data augmentation are performed.

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

For an explicitly authorized full run, replace `--smoke 80` with `--full` and choose
a new output directory. No full-dataset training has been run. Full data may need
substantial memory: all final candidate pairs and features are currently held in
memory, even though sparse retrieval avoids the full Cartesian similarity matrix.
Tune top_k and batch_size in config/default.json based on measured blocking recall.

Stages: schema/audit, conservative normalization, minimal baseline comparison
(exact/name top-k; three features; logistic vs LightGBM), full retrieval/features,
entity-level OOF scoring, global macro-F0.5 threshold selection, country stress
tests, final model fit, TSV export, supplied validator, and submission ZIP.

The full and baseline LightGBM models compete for final selection. Logistic remains
a regression baseline; limiting final selection to LightGBM satisfies the MIT model
license constraint. IDs are only keys and never predictors. Missing values never
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
- `cache/`: joblib content-based cache for normalization, retrieval and features.
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
