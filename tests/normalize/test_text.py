import pandas as pd
import pytest

import deduplix.normalize  # noqa: F401  (registers normalizers)
from deduplix.engine import registry
from deduplix.engine.context import ColumnMap, RunContext


def run(spec, names, column="name", **ctx_kw):
    df = pd.DataFrame({"id": range(len(names)), column: names})
    ctx = RunContext(left=df, right=df, mode="dedup", columns=ColumnMap(name=column), **ctx_kw)
    return registry.build("normalizer", spec)(df, ctx), df


@pytest.mark.parametrize("spec,raw,expected", [
    ("strip_whitespace", "  a   b \t c ", "a b c"),
    ("lowercase", "ABC Corp", "abc corp"),
    ("fold_accents", "Société Générale Çà", "Societe Generale Ca"),
    ("fold_accents", "Müller", "Muller"),
    ("ampersand", "AT&T", "AT and T"),
    ("ampersand", "Smith & Sons", "Smith and Sons"),
    ({"remove_punctuation": {}}, "A.B.C., Inc.", "A B C Inc"),
    ({"remove_punctuation": {"pattern": r"[.,]"}}, "A-B, Inc.", "A-B Inc"),
    ("default_text", "  Société  AT&T (Mauritius), L.P. ", "societe at and t mauritius l p"),
    ("default_text", "ABC Fund IV (Mauritius), L.P.", "abc fund iv mauritius l p"),
    ("default_text", "Ünïcode & Co.", "unicode and co"),
    ({"expand_abbreviations": {}}, "intl mfg bros", "international manufacturing brothers"),
    ({"expand_abbreviations": {}}, "international", "international"),
    ({"expand_abbreviations": {}}, "mfgr intl", "mfgr international"),
    ({"expand_abbreviations": {"table": {"acme": "ACME Corp"}}}, "acme labs", "ACME Corp labs"),
    ({"remove_stopwords": {}}, "the bank of india", "bank india"),
    ({"remove_stopwords": {"languages": ["fr"]}}, "banque de france", "banque france"),
    ({"remove_stopwords": {}}, "the", "the"),
    ({"chain": {"steps": ["lowercase", "ampersand", "expand_abbreviations"]}}, "A&B Intl",
     "a and b international"),
])
def test_text_normalizers(spec, raw, expected):
    out, _ = run(spec, [raw])
    assert out["name_norm"].iloc[0] == expected


def test_originals_untouched_and_copy():
    out, df = run("default_text", ["Foo & Bar"])
    assert out is not df
    assert "name_norm" not in df.columns
    assert out["name"].tolist() == ["Foo & Bar"]


def test_missing_values_stay_missing():
    out, _ = run("default_text", ["Foo", None, "Bar"])
    assert pd.isna(out["name_norm"].iloc[1])
    assert out["name_norm"].tolist()[0] == "foo"


def test_columns_and_output_params():
    df = pd.DataFrame({"a": ["X Y"], "b": ["P Q"]})
    ctx = RunContext(left=df, right=df, mode="dedup")
    out = registry.build("normalizer", {"lowercase": {"columns": ["a", "b"]}})(df, ctx)
    assert out["a_norm"].iloc[0] == "x y" and out["b_norm"].iloc[0] == "p q"
    out = registry.build("normalizer", {"lowercase": {"columns": "a", "output": "low"}})(df, ctx)
    assert out["low"].iloc[0] == "x y"


def test_custom_name_column_and_chaining():
    df = pd.DataFrame({"org": ["  FOO & Bar  "]})
    ctx = RunContext(left=df, right=df, mode="dedup", columns=ColumnMap(name="org"))
    out = df
    for s in ("strip_whitespace", "lowercase", "ampersand"):
        out = registry.build("normalizer", s)(out, ctx)
    assert out["name_norm"].iloc[0] == "foo and bar"


def test_numeric_values_coerced():
    out, _ = run("lowercase", [123, "A"])
    assert out["name_norm"].tolist() == ["123", "a"]


def test_registered_and_spec_roundtrip():
    names = {c["name"] for c in registry.list_components("normalizer")}
    assert {"default_text", "name_parser", "lowercase", "transliterate", "fold_accents"} <= names
    obj = registry.build("normalizer", {"remove_punctuation": {"pattern": r"[.]"}})
    assert registry.spec_of(obj)["params"] == {"pattern": r"[.]"}


def test_transliterate_lazy():
    pytest.importorskip("anyascii")
    out, _ = run("transliterate", ["Москва Bank", "Plain"])
    assert out["name_norm"].tolist() == ["Moskva Bank", "Plain"]


def test_unknown_chain_step():
    with pytest.raises(Exception):
        registry.build("normalizer", {"chain": {"steps": ["nope"]}})
