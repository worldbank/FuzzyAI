import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from deduplix.engine import registry  # noqa: E402
from deduplix.evaluation.train import train_scorer  # noqa: E402
from deduplix.exceptions import ConfigurationError  # noqa: E402
import deduplix.scoring  # noqa: E402,F401


def data(n=600, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.rand(n) < 0.4
    sim = np.clip(np.where(y, rng.normal(0.9, 0.08, n), rng.normal(0.5, 0.2, n)), 0, 1).astype("float32")
    sim[rng.rand(n) < 0.05] = np.nan
    des = np.where(y, "ok", rng.choice(["ok", "reject"], n, p=[0.5, 0.5]))
    pairs = pd.DataFrame({"i1": np.arange(n), "i2": np.arange(n) + n, "f_name": sim,
                          "f_des": pd.Categorical(des)})
    return pairs, y


@pytest.mark.parametrize("model", ["hist_gbdt", "logistic"])
def test_train_scorer_and_use_as_scorer(model):
    pairs, y = data()
    est, metrics = train_scorer(pairs, pd.Series(y.astype(int)), ["name", "f_des"], model=model, cv=3)
    assert {"band", "n", "precision", "mean_score"} <= set(metrics.columns)
    assert metrics["n"].sum() == len(pairs)
    hi = metrics[metrics["lo"] >= 0.9]
    assert hi["precision"].dropna().min() > 0.8
    scorer = registry.build("scorer", {"model": {"estimator": est}})
    s = scorer(pairs, None)
    assert s.between(0, 1).all() and s[y].mean() > s[~y].mean() + 0.3


def test_labels_forms_and_unlabelled():
    pairs, y = data(300)
    words = np.where(y, "match", "non_match").astype(object)
    words[:20] = None  # undecided rows dropped
    est, m = train_scorer(pairs, words, ["name"], cv=3, calibrate="sigmoid")
    assert m["n"].sum() == 280
    frame = pd.DataFrame({"i1": pairs.i1, "i2": pairs.i2, "label": y.astype(int)})
    est2, m2 = train_scorer(pairs, frame, ["name"], calibrate=None)
    assert m2["n"].sum() == 300


def test_train_scorer_errors():
    pairs, y = data(100)
    with pytest.raises(ConfigurationError):
        train_scorer(pairs, np.ones(100), ["name"])
    with pytest.raises(ConfigurationError):
        train_scorer(pairs, y, ["nope"])
    with pytest.raises(ConfigurationError):
        train_scorer(pairs, y, ["name"], model="xx")
