"""Split hygiene, training/inference smoke tests, handling states."""
import pandas as pd
import pytest

from reteach import evaluate as ev
from reteach import model as m
from reteach import taxonomy as tx


@pytest.fixture(scope="module")
def split():
    labeled, contexts = ev.load_labeled()
    return m.split_dataset(labeled, contexts), labeled, contexts


def test_no_group_overlap_between_train_and_test(split):
    (dev, test), _, _ = split
    assert set(dev["group_id"]).isdisjoint(set(test["group_id"]))


def test_test_uses_only_held_out_contexts(split):
    (dev, test), _, contexts = split
    held_out = set(contexts.loc[contexts["context_split"] == "test", "context_id"])
    assert set(test["question_id"]) <= held_out
    assert set(dev["question_id"]).isdisjoint(held_out)


def test_every_class_has_enough_groups(split):
    (dev, test), _, _ = split
    for label in tx.TEXT_LABELS:
        assert dev.loc[dev["label"] == label, "group_id"].nunique() >= m.N_FOLDS
        assert test.loc[test["label"] == label, "group_id"].nunique() >= 3


def test_each_paraphrase_used_at_most_once(split):
    (dev, test), _, _ = split
    both = pd.concat([dev, test])
    stem = both["example_id"].str.rsplit("-", n=1).str[0]
    assert not stem.duplicated().any()


def test_split_is_deterministic(split):
    (dev, test), labeled, contexts = split
    dev2, test2 = m.split_dataset(labeled, contexts)
    assert dev["example_id"].tolist() == dev2["example_id"].tolist()
    assert test["example_id"].tolist() == test2["example_id"].tolist()


def test_choose_threshold_policy():
    table = pd.DataFrame({"threshold": [0.2, 0.3, 0.4], "coverage": [1, .5, .1],
                          "accepted": [10, 5, 1], "accepted_accuracy": [0.6, 0.92, 1.0]})
    assert m.choose_threshold(table, 0.9) == 0.3
    assert m.choose_threshold(table, 1.01) == 1.01   # nothing meets -> review everything


@pytest.fixture(scope="module")
def trained(split):
    (dev, _), _, _ = split
    return m.TrainedModel(m.train(dev, 1.0), threshold=0.0, meta={})


def test_train_and_classify_states(trained):
    texts = ["", "   ", "idk", "zzzz qqqq xxxx",
             "I added the numerators and the denominators together",
             "I added the numerators and the denominators together"]
    out = m.classify(trained, texts)
    assert out["state"].tolist()[:4] == [tx.STATE_BLANK, tx.STATE_BLANK, tx.STATE_INSUFFICIENT,
                                         tx.STATE_UNRECOGNIZED]
    assert out.loc[4, "state"] == tx.STATE_AUTO      # threshold 0 accepts everything
    assert out.loc[4, "predicted_label"] in tx.TEXT_LABELS
    assert 0 < out.loc[4, "score"] <= 1
    assert out.loc[0, "predicted_label"] == ""      # never forced into a category


def test_high_threshold_routes_to_review(trained):
    strict = m.TrainedModel(trained.pipeline, threshold=1.01, meta={})
    out = m.classify(strict, ["I added the numerators and the denominators together"])
    assert out.loc[0, "state"] == tx.STATE_NEEDS_REVIEW


def test_pipeline_contains_vectorizer(trained):
    assert list(trained.pipeline.named_steps) == ["tfidf", "clf"]


def test_evaluation_writes_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(m, "META_PATH", tmp_path / "meta.json")
    # save() binds default paths at definition time, so patch it explicitly too
    orig_save = m.save
    monkeypatch.setattr(m, "save", lambda model: orig_save(model, tmp_path / "model.joblib",
                                                           tmp_path / "meta.json"))
    metrics = ev.run_evaluation(out_dir=tmp_path)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "evaluation_report.md").read_text().startswith("# Explanation classifier")
    t = metrics["test"]
    assert metrics["dataset"]["group_overlap_dev_test"] == []
    assert 0 <= t["model"]["accuracy"] <= 1
    assert t["baseline_most_frequent"]["accuracy"] == pytest.approx(0.2)
    assert sum(sum(r) for r in t["confusion_matrix"]["matrix"]) == metrics["dataset"]["test"][
        "examples"]
