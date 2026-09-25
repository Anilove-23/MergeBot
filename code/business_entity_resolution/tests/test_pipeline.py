import numpy as np
import pandas as pd
import pytest

from ber.candidates import generate
from ber.evaluation import assemble, entity_score, splits
from ber.export import export, package
from ber.features import build_features
from ber.normalization import normalize, preprocess
from ber.loading import stream_select
from ber.training import fit_model, probability


CONFIG = {"top_k": 2, "batch_size": 2, "threads": 1, "rare_max_df": 3, "rare_tokens": 2,
          "seed": 42, "folds": 2}


def frame(rows):
    return preprocess(pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"]))


def test_official_metric():
    assert entity_score(set(), set()) == 1
    assert entity_score({"a"}, set()) == 0
    assert entity_score(set(), {"a"}) == 0
    assert np.isclose(entity_score({"a", "b"}, {"a", "b", "c"}), 5 / 7)


def test_normalization():
    data = frame([("S1-a", "École GmbH", "12, Rue!", "France"),
                  ("S1-b", "東京 株式会社", "", "Japan")])
    assert data.iloc[0]["name"] == "école gmbh"
    assert data.iloc[0].name_folded == "ecole gmbh"
    assert data.iloc[0].name_core == "école"
    assert data.iloc[0].business_name == "École GmbH"
    assert normalize("東京 株式会社") == "東京 株式会社"


def test_retrieval_missingness_multimatch_and_export(tmp_path):
    queries = frame([("S1-a", "École", "12 Rue", "France"), ("S1-b", "", "", "Unknown")])
    targets = frame([("S2-a", "Ecole", "12 Rue", "France"), ("S3-a", "École", "12 Rue", "Other"),
                     ("S2-b", "", "", "Unknown")])
    pairs = generate(queries, targets, CONFIG)
    assert set(pairs[pairs.q.eq(0)].t) >= {0, 1}
    assert pairs[pairs.q.eq(1)].empty
    features = build_features(queries, targets, pairs, CONFIG)
    assert np.isfinite(features.to_numpy()).all()
    predictions = assemble(queries, targets, pairs, np.ones(len(pairs)), .5)
    assert predictions["S1-a"] >= {"S2-a", "S3-a"}
    assert predictions["S1-b"] == set()
    export(queries, targets, pairs, predictions, tmp_path)
    output = pd.read_csv(tmp_path / "matching_results.tsv", sep="\t", keep_default_na=False)
    assert len(output) == 2
    assert output.iloc[1].matched_entity_ids == ""
    artificial = pd.DataFrame([(1, 2, 0., 0)], columns=["q", "t", "retrieval_score", "rank"])
    absent = build_features(queries, targets, artificial, CONFIG)
    assert absent.name_exact.iloc[0] == absent.address_exact.iloc[0] == 0
    assert absent.name_similarity.iloc[0] == absent.address_similarity.iloc[0] == 0


def test_entity_folds():
    queries = pd.DataFrame({"entity_id": [f"S1-{i}" for i in range(8)]})
    held = []
    for train, valid in splits(queries, CONFIG):
        assert not set(train) & set(valid)
        held.extend(valid)
    assert sorted(held) == list(range(8))


def test_sample_reproduction_retains_distractors(tmp_path):
    path = tmp_path / "source.tsv"
    pd.DataFrame({"entity_id": ["a", "b", "c", "d", "e"]}).to_csv(path, sep="\t", index=False)
    first = stream_select(path, "entity_id", {"d", "e"}, 2)
    first.to_csv(path, sep="\t", index=False)
    second = stream_select(path, "entity_id", {"d", "e"}, 2)
    pd.testing.assert_frame_equal(first, second)


def test_empty_candidate_inference():
    queries = frame([("S1-a", "", "", "New Country")])
    targets = frame([("S2-a", "", "", "New Country")])
    pairs = generate(queries, targets, CONFIG)
    assert pairs.empty
    features = build_features(queries, targets, pairs, CONFIG)
    assert features.empty
    assert assemble(queries, targets, pairs, np.empty(0), .5) == {"S1-a": set()}


def test_repackage_has_no_duplicate_artifacts(tmp_path):
    import zipfile
    project, run = tmp_path / "project", tmp_path / "run"
    for root in [project, run]:
        (root / "artifacts").mkdir(parents=True)
        (root / "artifacts/model.txt").write_text("new" if root == run else "old")
    (run / "output").mkdir()
    (run / "Documentation_template.md").write_text("Smoke only")
    with zipfile.ZipFile(package(project, run)) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert archive.read("code/business_entity_resolution/artifacts/model.txt") == b"new"


@pytest.mark.parametrize("kind", ["lightgbm", "xgboost"])
def test_both_backends_save_and_reload(kind, tmp_path):
    import joblib
    features = pd.DataFrame(np.random.default_rng(42).normal(size=(100, 3)),
                            columns=["name_similarity", "address_similarity", "country_agreement"])
    y = (features.name_similarity > 0).astype(int).to_numpy()
    config = {**CONFIG, "n_estimators": 3, "num_leaves": 4, "learning_rate": .1,
              "hard_negative_weight": 2., "device": "cpu"}
    model = fit_model(kind, features, y, config, hard=True)
    before = probability(model, features)
    joblib.dump(model, tmp_path / f"{kind}.joblib")
    loaded = joblib.load(tmp_path / f"{kind}.joblib")
    np.testing.assert_allclose(before, probability(loaded, features))
    assert np.isfinite(before).all() and ((before >= 0) & (before <= 1)).all()
