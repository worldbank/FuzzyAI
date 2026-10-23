import numpy as np
import pandas as pd
import pytest

import deduplix.compare  # noqa: F401
import deduplix.scoring  # noqa: F401
from deduplix.engine import registry
from deduplix.engine.context import RunContext
from deduplix.exceptions import ConfigurationError


def S(spec):
    return registry.build("scorer", spec)


def frame(**cols):
    n = len(next(iter(cols.values())))
    return pd.DataFrame({"i1": np.arange(n), "i2": np.arange(n), **cols})


def test_registered():
    names = {c["name"] for c in registry.list_components("scorer")}
    assert {"feature", "weighted", "grade_table", "fellegi_sunter", "model"} <= names


def test_feature():
    p = frame(f_name=np.array([0.5, np.nan, 1], dtype="float32"))
    s = S({"feature": {"name": "name"}})(p, None)
    assert s.iloc[0] == 0.5 and np.isnan(s.iloc[1]) and s.dtype == np.float32
    with pytest.raises(ConfigurationError):
        S({"feature": {"name": "zzz"}})(p, None)


def test_weighted_renormalize_and_zero():
    p = frame(f_a=[1.0, 1.0, np.nan, np.nan], f_b=[0.0, np.nan, 0.5, np.nan])
    s = S({"weighted": {"weights": {"a": 3, "f_b": 1}}})(p, None)
    assert s.iloc[0] == pytest.approx(0.75) and s.iloc[1] == pytest.approx(1.0)
    assert s.iloc[2] == pytest.approx(0.5) and np.isnan(s.iloc[3])
    z = S({"weighted": {"weights": {"a": 3, "b": 1}, "missing": "zero"}})(p, None)
    assert z.iloc[1] == pytest.approx(0.75) and z.iloc[3] == 0


def test_weighted_maps():
    p = frame(f_n=[1.0, 1.0, 1.0], f_lf=pd.Categorical(["same", "missing", "different"]))
    spec = {"weighted": {"weights": {"f_n": 0.5, "f_lf": 0.5},
                         "maps": {"f_lf": {"same": 1.0, "missing": 0.5, "different": 0.0}}}}
    assert S(spec)(p, None).round(3).tolist() == [1.0, 0.75, 0.5]
    bad = frame(f_n=[1.0], f_lf=["weird"])
    with pytest.raises(ConfigurationError):
        S(spec)(bad, None)
    with pytest.raises(ConfigurationError):
        S({"weighted": {"weights": {"f_lf": 1}}})(bad, None)


def test_grade_table_wildcards(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("grade_pattern,confidence\nAA*,10\nA**,8\n*B*,5\nZZZ,0\n", encoding="utf-8")
    p = frame(f_match_grade=["AAA", "AAF", "ABF", "FBA", "FFF", None, "AA"])
    s = S({"grade_table": {"path": str(path)}})(p, None)
    assert s.iloc[:5].tolist() == pytest.approx([1.0, 1.0, 0.8, 0.5, 0.0])
    assert np.isnan(s.iloc[5]) and s.iloc[6] == 0.0


def test_grade_table_bad(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("grade_pattern,confidence\nAA,11\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        S({"grade_table": {"path": str(path)}})


def make_fs_data(n=40000, prior=0.2, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.rand(n) < prior
    m = np.array([0.05, 0.15, 0.80])  # levels of f_a
    u = np.array([0.70, 0.25, 0.05])
    a = np.where(y, rng.choice(3, n, p=m), rng.choice(3, n, p=u))
    mb, ub = np.array([0.1, 0.9]), np.array([0.8, 0.2])  # f_b categorical
    b = np.where(y, rng.choice(2, n, p=mb), rng.choice(2, n, p=ub))
    df = frame(f_a=np.array([0.1, 0.8, 0.99])[a], f_b=np.array(["no", "yes"])[b])
    return df, y, (m, u, mb, ub)


def test_fellegi_sunter_em_recovers_params():
    df, y, (m, u, mb, ub) = make_fs_data()
    fs = S({"fellegi_sunter": {"levels": {"f_a": [0.5, 0.9], "f_b": None}, "estimate_u": "em",
                               "max_iter": 200, "tol": 1e-6}})
    score = fs(df, None)
    p = fs.params_
    assert np.abs(np.array(p["m"]["f_a"]) - m).max() < 0.05
    assert np.abs(np.array(p["u"]["f_a"]) - u).max() < 0.05
    assert abs(p["prior"] - 0.2) < 0.05
    assert score[y].mean() > 0.6 > score[~y].mean() and ((score >= 0) & (score <= 1)).all()


def test_fellegi_sunter_random_u_and_na():
    rng = np.random.RandomState(1)
    names = rng.choice(["alpha", "beta", "gamma", "delta"], 200)
    left = pd.DataFrame({"id": range(200), "name": names, "country": rng.choice(list("ab"), 200)})
    ctx = RunContext(left=left, right=left, mode="dedup")
    a, b = np.triu_indices(200, 1)
    sel = rng.choice(len(a), 5000, replace=False)
    pairs = pd.DataFrame({"i1": a[sel], "i2": b[sel]})
    comps = [{"exact": {"column": "country", "feature": "country"}}]
    comp = registry.build("comparator", comps[0])
    pairs = pd.concat([pairs, comp(pairs, left, left, ctx)], axis=1)
    fs = S({"fellegi_sunter": {"levels": {"country": None}, "comparators": comps, "u_sample_size": 5000}})
    score = fs(pairs, ctx)
    assert "fellegi_sunter" in ctx.artifacts and len(score) == len(pairs)
    # random pairs agree on country about half the time
    assert abs(ctx.artifacts["fellegi_sunter"]["u"]["f_country"][1] - 0.5) < 0.05
    # a missing feature contributes weight 0 -> prior probability
    pn = pairs.copy()
    pn["f_country"] = np.nan
    assert np.allclose(fs(pn, ctx), fs._prior, atol=1e-5)


def test_fellegi_sunter_bad():
    with pytest.raises(ConfigurationError):
        S({"fellegi_sunter": {}})
    with pytest.raises(ConfigurationError):
        S({"fellegi_sunter": {"levels": {"a": [1]}, "prior": 2}})


def test_model_scorer(tmp_path):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    X = pd.DataFrame({"f_a": [0.1, 0.2, 0.8, 0.9], "f_b": [0.0, 0.1, 0.9, 1.0]})
    clf = LogisticRegression().fit(X.values, [0, 0, 1, 1])
    p = frame(f_a=[0.05, 0.95], f_b=[0.0, 1.0])
    s = S({"model": {"estimator": clf, "features": ["a", "b"]}})(p, None)
    assert s.iloc[0] < 0.5 < s.iloc[1]
    import joblib

    joblib.dump(clf, tmp_path / "m.joblib")
    s2 = S({"model": {"estimator": str(tmp_path / "m.joblib"), "features": ["f_a", "f_b"]}})(p, None)
    assert np.allclose(s, s2)
    with pytest.raises(ConfigurationError):
        S({"model": {"estimator": clf, "features": ["zz"]}})(p, None)
