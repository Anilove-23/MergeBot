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
run. Full-run runtime, memory and 99% blocking recall remain unverified. All final
candidate pairs/features are currently held in memory, so full-scale execution may
need substantial RAM. Threshold/model selection scores reuse OOF predictions;
country stress results also reuse the global threshold and are diagnostic only.
The smoke ZIP is not a leaderboard submission.
