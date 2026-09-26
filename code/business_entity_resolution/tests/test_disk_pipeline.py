import json

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from ber.candidates import generate
from ber.disk_cache import prepare_dataset
from ber.disk_features import prepare_features, prepare_matrix
from ber.disk_index import build_index, mapped_topn
from ber.disk_pipeline import evaluate, export_scores, score_parts, stream_validate
from ber.disk_training import fit_disk
from ber.evaluation import entity_score
from ber.features import FEATURE_COLUMNS, build_features
from ber.loading import FIELDS


CONFIG = {'seed': 42, 'folds': 2, 'top_k': 2, 'rare_max_df': 10, 'rare_tokens': 2, 'threads': 1,
          'batch_size': 2, 'disk_query_batch_rows': 1, 'pair_feature_batch_rows': 2,
          'tfidf_max_features': 1000, 'training_batch_rows': 3, 'max_candidates_per_query': 100,
          'n_estimators': 3, 'num_leaves': 4, 'learning_rate': .1, 'hard_negative_weight': 2., 'device': 'cpu'}


@pytest.fixture
def indexed(tmp_path):
    root = tmp_path / 'data'
    rows = [
        [('S1-a', 'École', '12 Rue', 'France'), ('S1-b', 'Alpine', '34 Road', 'US'), ('S1-c', '', '', 'Unknown')],
        [('S2-a', 'Ecole', '12 Rue', 'France'), ('S2-b', 'Alpine', '34 Rd', 'US'), ('S2-c', 'Other', '99 Way', 'India')],
        [('S3-a', 'École', '12 Rue', 'Other'), ('S3-b', 'Elsewhere', '51 Avenue', 'France')],
    ]
    for split in ['train', 'test']:
        (root / split).mkdir(parents=True)
        for i, records in enumerate(rows, 1):
            pd.DataFrame(records, columns=FIELDS).to_csv(root / split / f'{split}_source{i}.tsv', sep='\t', index=False)
    pd.DataFrame([('S1-a', 'S2-a,S3-a'), ('S1-b', 'S2-b'), ('S1-c', '')], columns=['source1_entity_id', 'matched_entity_ids']).to_csv(root / 'train/train_ground_truth.tsv', sep='\t', index=False)
    cache = tmp_path / 'cache'
    prepared = prepare_dataset(root, cache, chunk_rows=2)
    index = build_index(prepared, 'train', root, cache, {'index_workers': 1})
    yield root, cache, prepared, index
    index.close()


def test_mapped_index_and_features_match_reference(indexed):
    _, cache, prepared, index = indexed
    queries = prepared['train_source1'].load()
    targets = pd.concat([prepared[f'train_source{i}'].load() for i in [2, 3]], ignore_index=True)
    matrix = index.matrix('name', 2, 1000)
    original = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), dtype=np.float32).fit_transform(prepared['train_source2'].load().name_folded)
    np.testing.assert_allclose(matrix.toarray(), original.T.toarray(), atol=1e-6)
    mapped = index.retrieve(queries, CONFIG).rename(columns={'rid': 't'}).sort_values(['q', 't']).reset_index(drop=True)
    reference = generate(queries, targets, CONFIG).sort_values(['q', 't']).reset_index(drop=True)
    pd.testing.assert_frame_equal(mapped[['q', 't']], reference[['q', 't']])
    vectors = {field: index.vectorizer(field, None, 1000)[0] for field in ['name', 'address']}
    a = build_features(queries, targets, reference, CONFIG)
    b = build_features(queries, targets, reference, CONFIG, vectorizers=vectors, name_frequencies=targets.name.value_counts())
    np.testing.assert_allclose(a, b, atol=1e-6)
    store = prepare_features(index, CONFIG, cache)
    frame = pd.concat(list(store.frames()), ignore_index=True)
    assert len(frame) == len(reference)
    assert 'qidx' not in FEATURE_COLUMNS and 'target_id' not in FEATURE_COLUMNS
    before = (store.directory / 'manifest.json').stat().st_mtime_ns
    assert prepare_features(index, CONFIG, cache).directory == store.directory
    assert (store.directory / 'manifest.json').stat().st_mtime_ns == before


def test_native_topn_does_not_copy_mapped_indices(indexed, monkeypatch):
    from ber.disk_index import sparse_api
    _, _, prepared, index = indexed
    matrix = index.matrix('name', 2, 1000)
    left = index.vectorizer('name', 2, 1000)[0].transform(prepared['train_source1'].load().name_folded)
    function_name = 'sp_matmul_topn_sorted_mt' if sparse_api._core._has_openmp_support else 'sp_matmul_topn_sorted'
    original = getattr(sparse_api._core, function_name)
    checked = []

    def check(**kwargs):
        assert kwargs['B_indices'] is matrix.indices
        assert kwargs['B_indptr'] is matrix.indptr
        checked.append(True)
        return original(**kwargs)

    monkeypatch.setattr(sparse_api._core, function_name, check)
    mapped_topn(left, matrix, 2, 1)
    assert checked


class SimilarityModel:
    def __init__(self, offset=0.):
        self.offset = offset

    def predict_proba(self, frame):
        score = np.clip(frame.name_similarity.to_numpy() + self.offset, 0, 1)
        return np.column_stack([1-score, score])


def test_streaming_metric_export_and_score_invalidation(indexed, tmp_path):
    _, cache, _, index = indexed
    store = prepare_features(index, CONFIG, cache)
    score_dir = score_parts(store, [SimilarityModel(), SimilarityModel()], CONFIG, 'test-model')
    report, sweep = evaluate(store, score_dir)
    destination = tmp_path / 'output'
    export_scores(store, score_dir, report['threshold'], destination)
    stream_validate(index, store, destination, destination / 'validator.log')
    output = pd.read_csv(destination / 'matching_results.tsv', sep='\t', keep_default_na=False)
    truth = index.truth(output.source1_entity_id)
    scores = [entity_score(truth[q], set(filter(None, ids.split(',')))) for q, ids in output.itertuples(index=False, name=None)]
    assert np.mean(scores) == pytest.approx(report['macro_f0.5'])
    assert output.loc[output.source1_entity_id.eq('S1-c'), 'matched_entity_ids'].iloc[0] == ''
    changed = score_parts(store, [SimilarityModel(-.3), SimilarityModel(-.3)], CONFIG, 'test-model')
    assert changed != score_dir


@pytest.mark.parametrize('kind', ['lightgbm', 'xgboost'])
def test_disk_training_save_and_checkpoint(indexed, kind):
    _, cache, _, index = indexed
    store = prepare_features(index, CONFIG, cache)
    matrix = prepare_matrix(store, CONFIG)
    assert isinstance(matrix.x, np.memmap)
    model = fit_disk(kind, matrix, CONFIG, budget_gb=1.)
    features = next(store.frames())
    prediction = model.predict_proba(features)
    checkpoints = list(matrix.directory.rglob('model.joblib'))
    reused = fit_disk(kind, matrix, {**CONFIG, 'index_workers': 2, 'index_cache_mb': 128}, budget_gb=1.)
    assert list(matrix.directory.rglob('model.joblib')) == checkpoints
    np.testing.assert_allclose(prediction, reused.predict_proba(features))
    assert prediction.shape == (len(features), 2)


def test_query_subset_preserves_target_pool(indexed):
    _, cache, _, index = indexed
    target_count = index.db.execute('SELECT COUNT(*) FROM records').fetchone()[0]
    store = prepare_features(index, {**CONFIG, 'train_query_limit': 2}, cache)
    selected = [q for part in store.parts() for q in store.query_metadata(part)]
    assert len(selected) == 2
    assert index.db.execute('SELECT COUNT(*) FROM records').fetchone()[0] == target_count


def test_parallel_index_resume_matches_serial(indexed, tmp_path, monkeypatch):
    import ber.disk_index as module
    from ber.disk_cache import NormalizedFile
    root, _, prepared, serial = indexed
    parallel_cache = tmp_path / 'parallel'
    original = module._prepared_chunks

    def interrupted(*args):
        stream = original(*args)
        try:
            yield next(stream)
            raise RuntimeError('simulated interruption after one committed chunk')
        finally:
            stream.close()

    monkeypatch.setattr(module, '_prepared_chunks', interrupted)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        build_index(prepared, 'train', root, parallel_cache, {'index_workers': 2})
    monkeypatch.setattr(module, '_prepared_chunks', original)
    read_starts = []
    original_batches = NormalizedFile.iter_batches

    def track(self, *args, **kwargs):
        read_starts.append((self.directory, kwargs.get('start_row', 0)))
        yield from original_batches(self, *args, **kwargs)

    monkeypatch.setattr(NormalizedFile, 'iter_batches', track)
    parallel = build_index(prepared, 'train', root, parallel_cache, {'index_workers': 2})
    try:
        for table in ['records', 'tokens', 'numbers', 'terms', 'token_counts', 'queries', 'truth', 'owners']:
            assert sorted(serial.db.execute(f'SELECT * FROM {table}').fetchall()) == sorted(parallel.db.execute(f'SELECT * FROM {table}').fetchall())
        assert (prepared['train_source2'].directory, 2) in read_starts
    finally:
        parallel.close()


def test_model_tuning_reuses_index_and_features(indexed, monkeypatch):
    import ber.disk_index as module
    root, cache, prepared, index = indexed
    store = prepare_features(index, CONFIG, cache)
    before = (store.directory / 'manifest.json').stat().st_mtime_ns
    changed = {**CONFIG, 'n_estimators': 100, 'learning_rate': .02, 'num_leaves': 31,
               'max_bin': 128, 'device': 'cuda:0', 'index_workers': 3, 'index_cache_mb': 64}

    def fail(*args):
        raise AssertionError('Model tuning must not rebuild indexing')

    monkeypatch.setattr(module, '_prepared_chunks', fail)
    reused = build_index(prepared, 'train', root, cache, changed)
    try:
        assert reused.directory == index.directory
        assert prepare_features(reused, changed, cache).directory == store.directory
        assert (store.directory / 'manifest.json').stat().st_mtime_ns == before
    finally:
        reused.close()
