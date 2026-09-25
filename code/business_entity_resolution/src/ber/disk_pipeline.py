"""Disk-backed end-to-end pipeline, with entity-level OOF model comparison."""
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import time

from filelock import FileLock
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .disk_cache import atomic_json, checkpoint_report, prepare_dataset
from .disk_features import prepare_features, prepare_matrix
from .disk_index import build_index
from .disk_training import TRAIN_VERSION, fit_disk
from .export import package, validate
from .features import FEATURE_COLUMNS
from .loading import load_data, write_fixture


def predict_saved(model, config, policy, root, output, cache_root, chunk_rows):
    from .disk_cache import prepare_source
    prepared = {f'test_source{i}': prepare_source(root / f'test_source{i}.tsv', cache_root, chunk_rows, source=i) for i in [1, 2, 3]}
    index = build_index(prepared, 'test', root.parent, cache_root)
    try:
        store = prepare_features(index, config, cache_root)
        scores = score_parts(store, [model], config, policy['model'].removeprefix('full_'), oof=False)
        export_scores(store, scores, policy['threshold'], output)
        stream_validate(index, store, output, output / 'validator.log')
        tqdm.write(f'PASS: streamed saved-model inference: {output}')
    finally:
        index.close()


def score_parts(store, models, config, kind, oof=True):
    # Test scores must also invalidate when training data/model weights change.
    key = hashlib.sha256(json.dumps([TRAIN_VERSION, config, kind, oof, joblib.hash(models)], sort_keys=True).encode()).hexdigest()[:20]
    directory = store.directory / 'scores' / key
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / 'state.json'
    with FileLock(str(directory / 'writer.lock')):
        state = json.loads(state_path.read_text()) if state_path.exists() else {'done': 0}
        for number, part in enumerate(tqdm(store.parts(), total=store.manifest['done'], desc=f'{kind} {"OOF" if oof else "test"} score chunks')):
            if number < state['done']:
                continue
            temporary = directory / f'{part["file"]}.tmp'
            schema = pa.schema([('qidx', pa.int64()), ('target_id', pa.string()), ('label', pa.uint8()), ('score', pa.float32())])
            with pq.ParquetWriter(temporary, schema, compression='zstd') as writer:
                for batch in pq.ParquetFile(store.directory / part['file']).iter_batches(batch_size=config.get('pair_feature_batch_rows', 20_000), use_threads=False):
                    frame = batch.to_pandas()
                    predictions = np.zeros(len(frame), dtype=np.float32)
                    if oof:
                        for fold, model in enumerate(models):
                            mask = frame.fold.eq(fold).to_numpy()
                            if mask.any():
                                predictions[mask] = model.predict_proba(frame.loc[mask])[:, 1]
                    else:
                        predictions = models[0].predict_proba(frame)[:, 1].astype(np.float32)
                    scored = frame[['qidx', 'target_id', 'label']].copy()
                    scored['score'] = predictions
                    writer.write_table(pa.Table.from_pandas(scored, schema=schema, preserve_index=False))
            os.replace(temporary, directory / part['file'])
            state['done'] = number + 1
            atomic_json(state_path, state)
    return directory


def evaluate(store, score_dir):
    thresholds = np.r_[np.linspace(0, 1, 101), 1.000001]
    f05, f1, fps, predicted, false_entities, singleton_correct = [np.zeros(len(thresholds), dtype=np.float64) for _ in range(6)]
    countries, country_counts = {}, Counter()
    n = singletons = negatives = true_total = found = 0
    count_histogram = Counter()
    for part in tqdm(store.parts(), total=store.manifest['done'], desc='Streaming macro F0.5 thresholds'):
        frame = pq.read_table(score_dir / part['file'], columns=['qidx', 'label', 'score']).to_pandas()
        groups = frame.groupby('qidx', sort=False).indices
        for query in store.query_metadata(part):
            subset = frame.iloc[groups.get(query['qidx'], [])]
            order = np.argsort(subset.score.to_numpy(), kind='stable')
            values = subset.score.to_numpy()[order]
            labels = subset.label.to_numpy()[order]
            prefix = np.r_[0, np.cumsum(labels, dtype=np.int64)]
            cutoff = np.searchsorted(values, thresholds, side='left')
            accepted = len(values) - cutoff
            tp = int(prefix[-1]) - prefix[cutoff]
            fp, fn = accepted - tp, query['truth_count'] - tp
            denom = 1.25 * tp + .25 * fn + fp
            scores = np.divide(1.25 * tp, denom, out=np.ones(len(thresholds)), where=denom != 0)
            denom1 = 2 * tp + fn + fp
            scores1 = np.divide(2 * tp, denom1, out=np.ones(len(thresholds)), where=denom1 != 0)
            f05 += scores
            f1 += scores1
            fps += fp
            predicted += accepted
            false_entities += fp > 0
            n += 1
            if query['truth_count'] == 0:
                singletons += 1
                singleton_correct += accepted == 0
            country = query['country']
            countries.setdefault(country, np.zeros(len(thresholds)))
            countries[country] += scores
            country_counts[country] += 1
            negatives += len(values) - int(prefix[-1])
            true_total += query['truth_count']
            found += query['found']
            count_histogram[query['candidate_count']] += 1
    if not n:
        raise ValueError('No training queries selected')
    winner = int(np.flatnonzero(f05 == f05.max())[-1])
    report = {'macro_f0.5': float(f05[winner] / n), 'macro_f1': float(f1[winner] / n),
              'threshold': float(thresholds[winner]), 'queries': n, 'false_positive_pairs': int(fps[winner]),
              'singleton_accuracy': float(singleton_correct[winner] / singletons) if singletons else None,
              'candidate_false_positive_rate': float(fps[winner] / max(1, negatives)),
              'false_discovery_rate': float(fps[winner] / max(1, predicted[winner])),
              'entities_with_false_positive_rate': float(false_entities[winner] / n),
              'per_country': {c: float(s[winner] / country_counts[c]) for c, s in countries.items()},
              'candidate_recall': found / true_total if true_total else 1.,
              'true_pairs': true_total, 'retrieved_true_pairs': found,
              'candidate_count_histogram': dict(count_histogram)}
    sweep = pd.DataFrame({'threshold': thresholds, 'macro_f0.5': f05/n, 'macro_f1': f1/n})
    return report, sweep


def export_scores(store, score_dir, threshold, output):
    output.mkdir(parents=True, exist_ok=True)
    matching_path, candidate_path = output / 'matching_results.tsv', output / 'candidate_pairs.tsv'
    with matching_path.with_suffix('.tsv.tmp').open('w', encoding='utf-8', newline='') as matching, candidate_path.with_suffix('.tsv.tmp').open('w', encoding='utf-8', newline='') as candidates:
        mw, cw = csv.writer(matching, delimiter='\t'), csv.writer(candidates, delimiter='\t')
        mw.writerow(['source1_entity_id', 'matched_entity_ids'])
        cw.writerow(['source1_entity_id', 'candidate_entity_ids'])
        for part in tqdm(store.parts(), total=store.manifest['done'], desc='Export TSV chunks'):
            frame = pq.read_table(score_dir / part['file'], columns=['qidx', 'target_id', 'score']).to_pandas()
            groups = frame.groupby('qidx', sort=False).indices
            for query in store.query_metadata(part):
                subset = frame.iloc[groups.get(query['qidx'], [])]
                mw.writerow([query['qid'], ','.join(sorted(subset.loc[subset.score.ge(threshold), 'target_id']))])
                cw.writerow([query['qid'], ','.join(sorted(subset.target_id))])
    os.replace(matching_path.with_suffix('.tsv.tmp'), matching_path)
    os.replace(candidate_path.with_suffix('.tsv.tmp'), candidate_path)


def write_errors(index, store, score_dir, threshold, directory):
    fp_conflicts, fn_ranks = Counter(), Counter()
    missed_by_blocking = 0
    with (directory / 'pair_errors.tsv').open('w', encoding='utf-8', newline='') as handle, (directory / 'false_negatives.tsv').open('w', encoding='utf-8', newline='') as missing_file:
        writer, missing_writer = csv.writer(handle, delimiter='\t'), csv.writer(missing_file, delimiter='\t')
        writer.writerow(['query_id', 'target_id', 'truth', 'probability', 'predicted', 'number_conflict', 'candidate_rank'])
        missing_writer.writerow(['query_id', 'target_id', 'reason'])
        for part in store.parts():
            scores = pq.read_table(score_dir / part['file']).to_pandas()
            context = pq.read_table(store.directory / part['file'], columns=['qid', 'number_conflict', 'candidate_rank']).to_pandas()
            predicted = scores.score.ge(threshold)
            error = predicted.ne(scores.label.astype(bool))
            for i in np.flatnonzero(error):
                row, fields = scores.iloc[i], context.iloc[i]
                writer.writerow([fields.qid, row.target_id, int(row.label), float(row.score), bool(predicted.iloc[i]), fields.number_conflict, fields.candidate_rank])
                if row.label:
                    fn_ranks[int(fields.candidate_rank)] += 1
                else:
                    fp_conflicts[int(fields.number_conflict)] += 1
            groups = scores.groupby('qidx', sort=False).indices
            metadata = store.query_metadata(part)
            truth = index.truth([q['qid'] for q in metadata])
            for query in metadata:
                subset = scores.iloc[groups.get(query['qidx'], [])]
                retrieved = set(subset.target_id)
                accepted = set(subset.loc[subset.score.ge(threshold), 'target_id'])
                for target in sorted(truth[query['qid']] - accepted):
                    reason = 'threshold' if target in retrieved else 'blocking'
                    missed_by_blocking += reason == 'blocking'
                    missing_writer.writerow([query['qid'], target, reason])
    atomic_json(directory / 'error_analysis.json', {'false_positives_by_number_conflict': dict(fp_conflicts),
                'false_negatives_by_candidate_rank': dict(fn_ranks), 'blocking_misses': missed_by_blocking})


def stream_validate(index, store, output, log):
    count = 0
    with (output / 'matching_results.tsv').open(encoding='utf-8', newline='') as mf, (output / 'candidate_pairs.tsv').open(encoding='utf-8', newline='') as cf:
        matches, candidates = csv.DictReader(mf, delimiter='\t'), csv.DictReader(cf, delimiter='\t')
        if matches.fieldnames != ['source1_entity_id', 'matched_entity_ids'] or candidates.fieldnames != ['source1_entity_id', 'candidate_entity_ids']:
            raise ValueError('Incorrect output headers')
        last_qidx = -1
        for part in store.parts():
            for query in store.query_metadata(part):
                m, c = next(matches, None), next(candidates, None)
                if m is None or c is None or m['source1_entity_id'] != query['qid'] or c['source1_entity_id'] != query['qid'] or query['qidx'] <= last_qidx:
                    raise ValueError('Missing, duplicate, out-of-order or unexpected query in output')
                last_qidx = query['qidx']
                ml = list(filter(None, m['matched_entity_ids'].split(',')))
                cl = list(filter(None, c['candidate_entity_ids'].split(',')))
                if len(set(ml)) != len(ml) or len(set(cl)) != len(cl) or not set(ml) <= set(cl):
                    raise ValueError('Duplicate match/candidate or match outside candidates')
                if len(cl) != query['candidate_count']:
                    raise ValueError('Candidate export differs from scored pair count')
                for start in range(0, len(cl), 500):
                    batch = cl[start:start+500]
                    existing = index.db.execute(f"SELECT COUNT(*) FROM records WHERE entity_id IN ({','.join('?' for _ in batch)})", batch).fetchone()[0]
                    if existing != len(batch) or any(not value.startswith(('S2-', 'S3-')) for value in batch):
                        raise ValueError('Unknown or invalid candidate target')
                count += 1
        if next(matches, None) is not None or next(candidates, None) is not None:
            raise ValueError('Extra output rows')
        if index.split == 'test' and count != index.prepared['test_source1'].rows:
            raise ValueError('Not every test query was exported')
    log.write_text(f'PASS: streaming validation, {count} queries, target IDs checked; exact scored candidate counts and match subsets checked.\n', encoding='utf-8')


def run(args):
    from .pipeline import stage
    project = Path(__file__).resolve().parents[2]
    config = json.loads(Path(args.config or project / 'config/default.json').read_text())
    for name, value in {'disk_query_batch_rows': 64, 'pair_feature_batch_rows': 20_000,
                        'training_batch_rows': 100_000, 'tfidf_max_features': 200_000,
                        'max_candidates_per_query': 10_000, 'disk_max_depth': 5, 'max_bin': 64}.items():
        config.setdefault(name, value)
    kinds = config.get('boosted_models', ['lightgbm', 'xgboost'])
    if not kinds or not set(kinds) <= {'lightgbm', 'xgboost'}:
        raise ValueError('Disk mode supports LightGBM and XGBoost')
    config.update(train_query_limit=args.train_query_limit, execution_mode='disk')
    if args.smoke:
        config.update(top_k=5, n_estimators=40)
    root, output = Path(args.data).resolve(), Path(args.output).resolve()
    artifacts, reports = output / 'artifacts', output / 'reports'
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(exist_ok=True)
    timings = {}
    atomic_json(artifacts / 'config.json', config)
    if args.smoke:
        data, truth = load_data(root, args.smoke)
        root = output / 'fixture'
        write_fixture(data, truth, root)
        del data, truth
    with stage('Prepare disk normalization', timings):
        prepared = prepare_dataset(root, args.cache_dir, args.chunk_rows)
        atomic_json(reports / 'normalization_cache.json', checkpoint_report(prepared))
    with stage('Build train indexes and feature checkpoints', timings):
        train_index = build_index(prepared, 'train', root, args.cache_dir)
        train_store = prepare_features(train_index, config, args.cache_dir)
        matrix = prepare_matrix(train_store, config)
    experiments = []
    for kind in kinds:
        model_reports = reports / kind
        model_reports.mkdir(exist_ok=True)
        with stage(f'{kind} disk OOF and final training', timings):
            start = time.perf_counter()
            models = [fit_disk(kind, matrix, config, fold, args.memory_budget_gb) for fold in range(config['folds'])]
            scores = score_parts(train_store, models, config, kind, oof=True)
            report, sweep = evaluate(train_store, scores)
            report.update(model=f'full_{kind}', oof_seconds=time.perf_counter()-start)
            sweep.to_csv(model_reports / 'threshold_sweep.tsv', sep='\t', index=False)
            atomic_json(model_reports / 'oof_scores.json', {'directory': str(scores), 'feature_store': str(train_store.directory)})
            write_errors(train_index, train_store, scores, report['threshold'], model_reports)
            del models
            start = time.perf_counter()
            model = fit_disk(kind, matrix, config, None, args.memory_budget_gb)
            report.update(final_fit_seconds=time.perf_counter()-start, training_device=model.training_device_)
            experiments.append(report)
            joblib.dump(model, artifacts / f'full_{kind}.joblib')
            if model.kind == 'xgboost':
                model.booster.save_model(artifacts / 'full_xgboost.ubj')
            elif model.kind == 'lightgbm':
                model.booster.save_model(str(artifacts / 'full_lightgbm.txt'))
            policy = {'model': f'full_{kind}', 'threshold': report['threshold'], 'features': FEATURE_COLUMNS,
                      'full_candidates': True, 'global_fallback': True, 'execution_mode': 'disk',
                      'calibration': 'entity-level out-of-fold macro F0.5; model/threshold selection estimate'}
            atomic_json(artifacts / f'full_{kind}_policy.json', policy)
            validation_output = output / 'validation_output' / kind
            export_scores(train_store, scores, report['threshold'], validation_output)
            stream_validate(train_index, train_store, validation_output, model_reports / 'validation_validator.log')
    atomic_json(reports / 'experiments.json', experiments)
    atomic_json(artifacts / 'model_comparison.json', experiments)
    pd.json_normalize([{k: v for k, v in e.items() if k != 'candidate_count_histogram'} for e in experiments]).to_csv(reports / 'experiments.tsv', sep='\t', index=False)
    winner = max(experiments, key=lambda e: e['macro_f0.5'])
    selected = joblib.load(artifacts / f'{winner["model"]}.joblib')
    joblib.dump(selected, artifacts / 'selected_model.joblib')
    policy = json.loads((artifacts / f'{winner["model"]}_policy.json').read_text())
    atomic_json(artifacts / 'threshold_policy.json', policy)
    train_index.close()
    del matrix, selected
    with stage('Stream test retrieval, prediction and validation', timings):
        test_index = build_index(prepared, 'test', root, args.cache_dir)
        test_store = prepare_features(test_index, config, args.cache_dir)
        for kind in kinds:
            model = joblib.load(artifacts / f'full_{kind}.joblib')
            scores = score_parts(test_store, [model], config, kind, oof=False)
            threshold = next(e['threshold'] for e in experiments if e['model'] == f'full_{kind}')
            model_output = output / 'model_outputs' / kind
            export_scores(test_store, scores, threshold, model_output)
            stream_validate(test_index, test_store, model_output, reports / kind / 'validator.log')
            # The bundled validator holds all IDs in RAM: run it only for small
            # exports; the streaming validator above covers all sizes.
            if prepared['test_source1'].rows <= 10_000 and sum(prepared[f'test_source{i}'].rows for i in [2, 3]) <= 100_000:
                validator = Path(args.validator) if args.validator else project / 'utils/validate_submission.py'
                validate(validator, model_output, root / 'test', reports / kind / 'official_validator.log')
            if f'full_{kind}' == winner['model']:
                import shutil
                (output / 'output').mkdir(exist_ok=True)
                for filename in ['matching_results.tsv', 'candidate_pairs.tsv']:
                    shutil.copyfile(model_output / filename, output / 'output' / filename)
        test_index.close()
    atomic_json(reports / 'timings.json', timings)
    atomic_json(reports / 'disk_stores.json', {'train_features': str(train_store.directory), 'test_features': str(test_store.directory),
                'normalization_cache': str(Path(args.cache_dir).resolve()), 'training_subset_limit': args.train_query_limit})
    atomic_json(reports / 'training_backend.json', {e['model']: e['training_device'] for e in experiments})
    (output / 'Documentation_template.md').write_text(
        '# Disk-backed entity resolution\n\n'
        'Raw and normalized records are checkpointed in Parquet; SQLite stores exact/token/number indexes and corpus document frequencies. '
        'Corpus-wide character TF-IDF uses a bounded vocabulary and memory-mapped sparse postings. Candidate features and OOF scores are partitioned on disk. '
        'LightGBM reads a Sequence over mapped feature arrays; its compressed bins remain resident. XGBoost uses native disk external-memory pages with '
        'CPU or explicitly verified CUDA training. Both models use the same entity folds, features, candidates and hard-negative weights. '
        'Thresholds optimize macro entity F0.5 including singletons and retrieval misses. Training entities can be explicitly sampled with train_query_limit; '
        'all target records remain in retrieval. No country filtering, external identity lookup, ID features, or transitive merging is used.\n\n'
        f'Selected: {winner["model"]}; OOF selection F0.5: {winner["macro_f0.5"]:.6f}; threshold: {winner["threshold"]}.\n\n'
        + ('SMOKE ONLY: 80-or-fewer/specified sampled queries and reduced target pool; not a competition-performance estimate or full submission.\n\n' if args.smoke else '') +
        'Disk mode compares both full feature boosted models. The separate memory-mode pipeline retains the minimal logistic/boosted baseline and country stress tests. '
        'Disk mode does not currently run those additional experiments. OOF model/threshold selection is not an unbiased final holdout estimate. '
        'LightGBM is MIT licensed and XGBoost Apache-2.0 licensed. Models are non-neural and far below 8 billion parameters. '
        'See README.md for exact commands, cache/resume behavior, memory estimates and validation.\n', encoding='utf-8')
    package(project, output)
    tqdm.write(f'Disk-backed pipeline complete: {output}')
