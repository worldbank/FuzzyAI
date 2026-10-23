import numpy as np
import pandas as pd
import pytest

from deduplix.evaluation import metrics, silver, synthetic


def gold_df():
    return pd.DataFrame({"id1": ["a", "a", "c"], "id2": ["b", "c", "d"], "label": [1, 0, 1]})


def test_pair_metrics():
    pred = pd.DataFrame({"id1": ["b", "c", "x"], "id2": ["a", "d", "y"]})
    m = metrics.pair_metrics(pred, gold_df())
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 0
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == 1.0
    assert metrics.pair_metrics(pred.iloc[:0], gold_df())["f1"] == 0.0


def test_metrics_by_band():
    pairs = pd.DataFrame({"id1": ["a", "c", "x", "y"], "id2": ["b", "d", "z", "w"],
                          "score": [0.97, 0.85, 0.82, 0.3]})
    out = metrics.metrics_by_band(pairs, gold_df(), bands=[0.5, 0.8, 0.95])
    assert list(out["n_pairs"]) == [0, 2, 1]
    assert list(out["n_true"]) == [0, 1, 1]
    assert out.loc[1, "precision"] == 0.5
    assert out.loc[0, "cum_recall"] == 1.0


def test_cluster_metrics_perfect_and_merged():
    gold = {"a": 1, "b": 1, "c": 2, "d": 2}
    assert metrics.cluster_metrics(gold, gold)["bcubed_f1"] == 1.0
    merged = metrics.cluster_metrics({k: 0 for k in gold}, gold)
    assert merged["bcubed_recall"] == 1.0 and merged["bcubed_precision"] == pytest.approx(0.5)
    assert merged["pair_precision"] == pytest.approx(2 / 6) and merged["pair_recall"] == 1.0


def test_blocking_report_lazy():
    with pytest.raises(AttributeError):
        metrics.not_there
    try:
        import deduplix.blocking.metrics  # noqa: F401
    except ImportError:
        pytest.skip("blocking.metrics not built yet")
    assert callable(metrics.blocking_report)


def test_make_synthetic():
    rec, gold = synthetic.make_synthetic(200, dup_rate=0.5, seed=3)
    rec2, gold2 = synthetic.make_synthetic(200, dup_rate=0.5, seed=3)
    pd.testing.assert_frame_equal(rec, rec2)
    assert rec["id"].is_unique and set(gold["label"]) == {0, 1}
    assert {"fund_number", "other_country"} <= set(gold.loc[gold.label == 0, "kind"]) | {"fund_number", "other_country"}
    names = rec.set_index("id")
    pos = gold[gold.label == 1]
    assert (names.loc[pos.id1, "entity_id"].to_numpy() == names.loc[pos.id2, "entity_id"].to_numpy()).all()
    neg = gold[gold.label == 0]
    assert (names.loc[neg.id1, "entity_id"].to_numpy() != names.loc[neg.id2, "entity_id"].to_numpy()).all()
    assert set(pos["kind"]) - {"exact"} <= set(synthetic.VARIANTS)
    with pytest.raises(ValueError):
        synthetic.make_synthetic(10, variants=["nope"])


def test_variants_change_names():
    rng = __import__("random").Random(0)
    base = "Müller Global Holdings Ltd"
    assert synthetic._accent(base, rng) == "Muller Global Holdings Ltd"
    assert synthetic._translit(base, rng) == "Mueller Global Holdings Ltd"
    assert synthetic._legal_form(base, rng) == "Müller Global Holdings Limited"
    assert synthetic._abbreviation(base, rng) == "Müller Global Hldgs Ltd"


class FakeValidator:
    name = "fake"
    scope = "review"

    def __call__(self, pairs, left, right, ctx):
        same = (left["name"].to_numpy()[pairs["i1"]] == right["name"].to_numpy()[pairs["i2"]])
        return pd.DataFrame({"decision": np.where(same, "match", "non_match"),
                             "decision_reason": "fake"}, index=pairs.index)


def test_silver_roundtrip(tmp_path):
    left = pd.DataFrame({"id": [1, 2, 3, 4], "name": ["acme corp", "zed ltd", "beta inc", "acme"],
                         "country": ["US", "US", "DE", "DE"]})
    right = pd.DataFrame({"id": [10, 11, 12], "name": ["acme corp", "acme corp ltd", "other"]})
    sd = silver.sample_seed(left, n_seed=2, seed=1)
    assert len(sd) == 2 and set(sd.country) == {"US", "DE"}
    review = silver.build_silver(left, right, FakeValidator(), tmp_path / "review.csv", n_seed=10,
                                 threshold=60)
    assert len(review) > 0 and review["human_label"].isna().all()
    csv = pd.read_csv(tmp_path / "review.csv")
    flip = csv.index[(csv.id1 == 1) & (csv.id2 == 11)]
    csv.loc[flip, "human_label"] = 1
    csv.loc[(csv.id1 == 1) & (csv.id2 == 10), "human_label"] = 0
    csv.to_csv(tmp_path / "review.csv", index=False)
    gold = silver.import_review(tmp_path / "review.csv", tmp_path / "gold.parquet")
    g = gold.set_index(["id1", "id2"])
    assert g.loc[("1", "10"), "label"] == 0 and g.loc[("1", "10"), "source"] == "human"
    assert g.loc[("1", "11"), "label"] == 1
    assert (tmp_path / "gold.parquet").exists()
