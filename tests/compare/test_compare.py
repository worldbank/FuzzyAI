import numpy as np
import pandas as pd
import pytest

import deduplix.compare  # noqa: F401
from deduplix.engine import registry
from deduplix.engine.context import RunContext
from deduplix.exceptions import ConfigurationError


def make(df, right=None):
    ctx = RunContext(left=df, right=right if right is not None else df,
                     mode="link" if right is not None else "dedup")
    return ctx


def allpairs(n, m=None):
    m = m or n
    a, b = np.divmod(np.arange(n * m), m)
    return pd.DataFrame({"i1": a.astype("int32"), "i2": b.astype("int32")})


def run(spec, df, pairs=None, right=None):
    ctx = make(df, right)
    pairs = allpairs(len(ctx.left), len(ctx.right)) if pairs is None else pairs
    comp = registry.build("comparator", spec)
    out = comp(pairs, ctx.left, ctx.right, ctx)
    assert out.index.equals(pairs.index)
    return out


def test_registered():
    names = {c["name"] for c in registry.list_components("comparator")}
    assert {"string_sim", "exact", "numeric_rel_diff", "designator_check", "legal_form_compat",
            "country_match", "token_idf_overlap", "grade", "match_grade"} <= names


@pytest.mark.parametrize("fn", ["ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio",
                                "WRatio", "QRatio", "jaro_winkler", "levenshtein_norm"])
def test_string_sim_functions(fn):
    df = pd.DataFrame({"name": ["acme corp", "acme corp", "zzz"]})
    out = run({"string_sim": {"fn": fn, "feature": "n"}}, df)
    v = out["f_n"]
    assert v.dtype == np.float32
    assert v.iloc[0] == pytest.approx(1.0)  # (0,0)
    assert (v.dropna() >= 0).all() and (v.dropna() <= 1).all()
    assert v.iloc[1] == pytest.approx(1.0)  # acme corp vs acme corp
    assert v.iloc[2] < 0.5


def test_string_sim_nulls_unicode_and_fallback():
    df = pd.DataFrame({"name": ["Zürich AG", None, "", "zürich ag"]})
    out = run({"string_sim": {"fn": "ratio", "feature": "n"}}, df)
    m = out["f_n"].to_numpy().reshape(4, 4)
    assert np.isnan(m[1]).all() and np.isnan(m[:, 1]).all()
    assert np.isnan(m[2]).all() and np.isnan(m[:, 2]).all()
    assert m[0, 0] == 1.0 and 0 < m[0, 3] < 1


def test_string_sim_callable_and_columns():
    left = pd.DataFrame({"name": ["a"], "name_norm": ["ab"]})
    right = pd.DataFrame({"name": ["x", "y"], "alt": ["ab", "zz"]})
    spec = registry.build("comparator", {"string_sim": {"fn": lambda a, b: float(a == b),
                                                        "right": "alt", "feature": "c"}})
    ctx = make(left, right)
    out = spec(allpairs(1, 2), ctx.left, ctx.right, ctx)
    assert out["f_c"].tolist() == [1.0, 0.0]


def test_string_sim_bad_args():
    with pytest.raises(ConfigurationError):
        registry.build("comparator", {"string_sim": {"fn": "nope"}})
    df = pd.DataFrame({"name": ["a"]})
    with pytest.raises(ConfigurationError):
        run({"string_sim": {"left": "missing"}}, df)


def test_exact_and_country():
    df = pd.DataFrame({"name": list("abcd"), "country": ["MX", "mx", None, "US"]})
    m = run({"exact": {"column": "country", "feature": "c"}}, df)["f_c"].to_numpy().reshape(4, 4)
    assert m[0, 1] == 0 and m[0, 0] == 1 and np.isnan(m[2]).all() and m[0, 3] == 0
    m = run({"country_match": {"feature": "c"}}, df)["f_c"].to_numpy().reshape(4, 4)
    assert m[0, 1] == 1 and np.isnan(m[:, 2]).all()


def test_country_distance():
    df = pd.DataFrame({"name": ["a", "b"], "country": ["X", "X"], "lat": [0.0, 0.0], "lon": [0.0, 1.0]})
    out = run({"country_match": {"lat": "lat", "lon": "lon", "feature": "g"}}, df)
    assert out["f_g_km"].iloc[1] == pytest.approx(111.2, abs=0.5)


def test_numeric_rel_diff():
    df = pd.DataFrame({"name": list("abcd"), "v": [100, 50, 0, None]})
    m = run({"numeric_rel_diff": {"column": "v", "feature": "v"}}, df)["f_v"].to_numpy().reshape(4, 4)
    assert m[0, 1] == pytest.approx(0.5) and m[0, 0] == 0
    assert np.isnan(m[2, 2]) and np.isnan(m[3]).all() and m[0, 2] == 1


def test_designator_check():
    df = pd.DataFrame({
        "name": list("abcdefg"),
        "designator_nums": ["4", "1", "", "4", "4", "", ""],
        "vintage_year": ["", "", "2019", "", "", "2019", "2020"],
        "designator_series": ["", "", "", "", "", "", ""],
        "designator_quals": ["", "", "", "", "feeder", "", ""],
    })
    pairs = pd.DataFrame({"i1": [0, 0, 0, 0, 5, 5], "i2": [1, 2, 3, 4, 6, 5]})
    out = run({"designator_check": {"feature": "designators"}}, df, pairs)["f_designators"]
    assert out.tolist() == ["reject", "review", "ok", "review", "reject", "ok"]
    assert out.dtype.name == "category"


def test_designator_check_needs_columns():
    with pytest.raises(ConfigurationError):
        run("designator_check", pd.DataFrame({"name": ["a", "b"]}))


def test_legal_form_compat():
    df = pd.DataFrame({"name": list("abcd"), "legal_form": ["LTD", "LIMITED", "INC", None]})
    out = run({"legal_form_compat": {"feature": "lf"}}, df)["f_lf"].to_numpy().reshape(4, 4)
    assert out[0, 0] == "same" and out[0, 1] == "equivalent" and out[0, 2] == "different"
    assert out[0, 3] == "missing" and out[3, 3] == "missing"


def test_token_idf_overlap():
    df = pd.DataFrame({"name_norm": ["abc fund", "abc fund", "abc holdings", None, "xyz"],
                       "name": list("abcde")})
    m = run("token_idf_overlap", df)["f_token_idf_overlap"].to_numpy().reshape(5, 5)
    assert m[0, 1] == pytest.approx(1.0) and 0 < m[0, 2] < 1 and m[0, 4] == 0
    assert np.isnan(m[3]).all()


def test_grade_and_match_grade():
    pairs = pd.DataFrame({"i1": [0] * 4, "i2": [0] * 4, "f_a": [0.99, 0.85, 0.2, np.nan],
                          "f_b": pd.Categorical(["same", "different", "missing", "same"])})
    ctx = make(pd.DataFrame({"name": ["x"]}))
    g = registry.build("comparator", {"grade": {"source": "a", "feature": "ga"}})
    assert g(pairs, ctx.left, ctx.right, ctx)["f_ga"].tolist() == ["A", "B", "F", "Z"]
    mg = registry.build("comparator", {"match_grade": {"features": ["a", "b"]}})
    assert mg(pairs, ctx.left, ctx.right, ctx)["f_match_grade"].tolist() == ["AA", "BF", "FZ", "ZA"]
    with pytest.raises(ConfigurationError):
        registry.build("comparator", {"match_grade": {"features": ["zz"]}})(pairs, ctx.left, ctx.right, ctx)


def test_yaml_style_pipeline_spec():
    from deduplix.engine.config import _comparator_spec

    spec = _comparator_spec("name", {"fn": "token_set_ratio", "left": "name_norm"})
    df = pd.DataFrame({"name": ["a b", "b a"]})
    assert "f_name" in run(spec, df).columns
    spec = _comparator_spec("designators", {"fn": "designator_check"})
    assert spec == {"designator_check": {"feature": "designators"}}
