import pandas as pd
import pytest

import deduplix.normalize  # noqa: F401
from deduplix.engine import registry
from deduplix.engine.context import ColumnMap, RunContext
from deduplix.normalize import parse_names

COLS = ["name_base", "legal_form", "legal_form_raw", "designator_nums", "designator_series",
        "vintage_year", "designator_quals", "geo_qualifier"]


def parse(name, country=None, **params):
    df = pd.DataFrame({"name": [name]})
    if country is not None:
        df["country"] = [country]
        params["country_column"] = "country"
    ctx = RunContext(left=df, right=df, mode="dedup", columns=ColumnMap())
    return registry.build("normalizer", {"name_parser": params})(df, ctx).iloc[0]


# name, country, base, legal_form, raw, nums, series, year, quals, geo
CASES = [
    ("ABC Fund IV, L.P.", None, "abc", "LP", "L.P.", "4", "", "", "", ""),
    ("ABC Fund IV (Mauritius), L.P.", None, "abc", "LP", "L.P.", "4", "", "", "", "MU"),
    ("Mix Capital Ltd", None, "mix", "LTD", "Ltd", "", "", "", "", ""),
    ("ABC Capital Management LLC", None, "abc", "LLC", "LLC", "", "", "", "management", ""),
    ("ABC Fund 2019", None, "abc", "", "", "", "", "2019", "", ""),
    ("Grupo Bimbo, S.A.B. de C.V.", "MX", "grupo bimbo", "SABDECV", "S.A.B. de C.V.", "", "", "", "", ""),
    ("Grupo Bimbo, S.A.B. de C.V.", None, "grupo bimbo", "SABDECV", "S.A.B. de C.V.", "", "", "", "", ""),
    ("Cemex S.A. de C.V.", "MX", "cemex", "SADECV", "S.A. de C.V.", "", "", "", "", ""),
    ("Limited Brands Inc", None, "limited brands", "INC", "Inc", "", "", "", "", ""),
    ("ABC Fund I", None, "abc", "", "", "1", "", "", "", ""),
    ("ABC Fund 1", None, "abc", "", "", "1", "", "", "", ""),
    ("ABC Fund II", None, "abc", "", "", "2", "", "", "", ""),
    ("ABC Fund IX", None, "abc", "", "", "9", "", "", "", ""),
    ("ABC Fund XXX", None, "abc", "", "", "30", "", "", "", ""),
    ("ABC Fund XXXI", None, "abc fund xxxi", "", "", "", "", "", "", ""),
    ("ABC Fund VII LP", None, "abc", "LP", "LP", "7", "", "", "", ""),
    ("ABC Series B", None, "abc", "", "", "", "B", "", "", ""),
    ("ABC Series B LLC", None, "abc", "LLC", "LLC", "", "B", "", "", ""),
    ("ABC Series V", None, "abc", "", "", "5", "", "", "", ""),
    ("ABC Class A Partners", None, "abc", "", "", "", "A", "", "", ""),
    ("ABC Tranche 3", None, "abc", "", "", "3", "", "", "", ""),
    ("ABC Fund 12", None, "abc", "", "", "12", "", "", "", ""),
    ("ABC Fund 1950", None, "abc", "", "", "", "", "1950", "", ""),
    ("ABC Fund 2035", None, "abc", "", "", "", "", "2035", "", ""),
    ("ABC Fund 2036", None, "abc", "", "", "2036", "", "", "", ""),
    ("ABC Fund 1949", None, "abc", "", "", "1949", "", "", "", ""),
    ("ABC Fund 12345", None, "abc fund 12345", "", "", "", "", "", "", ""),
    ("ABC Fund IV 2019 LP", None, "abc", "LP", "LP", "4", "", "2019", "", ""),
    ("Civil Engineering Co", None, "civil engineering", "CO", "Co", "", "", "", "", ""),
    ("Mix Holdings", None, "mix", "", "", "", "", "", "", ""),
    ("Vi Partners", None, "vi", "", "", "", "", "", "", ""),
    ("ABC Master Fund", None, "abc", "", "", "", "", "", "master", ""),
    ("ABC Feeder Fund II LP", None, "abc", "LP", "LP", "2", "", "", "feeder", ""),
    ("ABC Offshore Feeder Fund", None, "abc", "", "", "", "", "", "feeder|offshore", ""),
    ("ABC Co-Invest Fund", None, "abc", "", "", "", "", "", "coinvest", ""),
    ("ABC Coinvestment LP", None, "abc", "LP", "LP", "", "", "", "coinvest", ""),
    ("ABC General Partner LLC", None, "abc", "LLC", "LLC", "", "", "", "gp", ""),
    ("ABC GP Ltd", None, "abc", "LTD", "Ltd", "", "", "", "gp", ""),
    ("ABC Blocker Inc", None, "abc", "INC", "Inc", "", "", "", "blocker", ""),
    ("ABC AIV LP", None, "abc", "LP", "LP", "", "", "", "aiv", ""),
    ("ABC Sidecar Fund", None, "abc", "", "", "", "", "", "sidecar", ""),
    ("ABC Advisors LLC", None, "abc", "LLC", "LLC", "", "", "", "management", ""),
    ("ABC Onshore Parallel Fund", None, "abc", "", "", "", "", "", "onshore|parallel", ""),
    ("Acme GmbH", "DE", "acme", "GMBH", "GmbH", "", "", "", "", ""),
    ("Acme AG", None, "acme", "AG", "AG", "", "", "", "", ""),
    ("Acme S.A.R.L.", None, "acme", "SARL", "S.A.R.L.", "", "", "", "", ""),
    ("Acme Sàrl", "LU", "acme", "SARL", "Sàrl", "", "", "", "", ""),
    ("Acme S.p.A.", "IT", "acme", "SPA", "S.p.A.", "", "", "", "", ""),
    ("Acme S.r.l.", "IT", "acme", "SRL", "S.r.l.", "", "", "", "", ""),
    ("Acme B.V.", "NL", "acme", "BV", "B.V.", "", "", "", "", ""),
    ("Acme N.V.", "NL", "acme", "NV", "N.V.", "", "", "", "", ""),
    ("Acme AB", "SE", "acme", "AB", "AB", "", "", "", "", ""),
    ("Acme A/S", "DK", "acme", "AS", "A/S", "", "", "", "", ""),
    ("Acme Oy", "FI", "acme", "OY", "Oy", "", "", "", "", ""),
    ("Acme Pty Ltd", "AU", "acme", "PTY", "Pty Ltd", "", "", "", "", ""),
    ("Acme Pte. Ltd.", "SG", "acme", "PTE", "Pte. Ltd.", "", "", "", "", ""),
    ("Acme Sdn Bhd", "MY", "acme", "SDN", "Sdn Bhd", "", "", "", "", ""),
    ("Acme Berhad", "MY", "acme", "BHD", "Berhad", "", "", "", "", ""),
    ("Acme K.K.", "JP", "acme", "KK", "K.K.", "", "", "", "", ""),
    ("Acme Pvt. Ltd.", "IN", "acme", "PVT", "Pvt. Ltd.", "", "", "", "", ""),
    ("Acme Ltda", "BR", "acme", "LIMITADA", "Ltda", "", "", "", "", ""),
    ("Acme Eireli", "BR", "acme", "EIRELI", "Eireli", "", "", "", "", ""),
    ("Acme Public Limited Company", None, "acme", "PLC", "Public Limited Company", "", "", "", "", ""),
    ("Acme Corporation", None, "acme", "CORP", "Corporation", "", "", "", "", ""),
    ("Smith & Co", None, "smith and", "CO", "Co", "", "", "", "", ""),
    ("Acme SAS", "FR", "acme", "SAS", "SAS", "", "", "", "", ""),
    ("Acme KG", "DE", "acme", "KG", "KG", "", "", "", "", ""),
    ("Acme LLP", None, "acme", "LLP", "LLP", "", "", "", "", ""),
    ("Limited", None, "limited", "", "", "", "", "", "", ""),
    ("Acme (Cayman) LP", None, "acme", "LP", "LP", "", "", "", "", "KY"),
    ("Acme Fund (Luxembourg) S.C.A.", None, "acme fund sca", "", "", "", "", "", "", "LU"),
    ("Acme (formerly Beta) Ltd", None, "acme formerly beta", "LTD", "Ltd", "", "", "", "", ""),
    ("  ACME   fund   iv  ", None, "acme", "", "", "4", "", "", "", ""),
    ("Société Générale S.A.", "FR", "societe generale", "SA", "S.A.", "", "", "", "", ""),
]


@pytest.mark.parametrize("case", CASES, ids=[f"{i}-{c[0]}-{c[1]}" for i, c in enumerate(CASES)])
def test_parse(case):
    name, country, *expected = case
    row = parse(name, country)
    assert [row[c] for c in COLS] == expected


def test_country_restricts_legal_form():
    # SADECV is Mexico only; with a US country it is not a legal form
    assert parse("Acme S.A. de C.V.", "US")["legal_form"] == ""
    assert parse("Acme S.A. de C.V.", "MX")["legal_form"] == "SADECV"


def test_country_missing_matches_any():
    df = pd.DataFrame({"name": ["Acme SA de CV"], "country": [None]})
    ctx = RunContext(left=df, right=df, mode="dedup")
    out = registry.build("normalizer", {"name_parser": {"country_column": "country"}})(df, ctx)
    assert out["legal_form"].iloc[0] == "SADECV"


def test_roman_vs_numeric_equal():
    a, b = parse("ABC Fund I"), parse("ABC Fund 1")
    assert a["designator_nums"] == b["designator_nums"] == "1"


def test_creates_name_norm_and_keeps_existing():
    df = pd.DataFrame({"name": ["Foo & Bar Ltd"]})
    ctx = RunContext(left=df, right=df, mode="dedup")
    p = registry.build("normalizer", "name_parser")
    assert p(df, ctx)["name_norm"].iloc[0] == "foo and bar ltd"
    df2 = df.assign(name_norm=["keep me"])
    assert p(df2, ctx)["name_norm"].iloc[0] == "keep me"


def test_originals_not_overwritten_and_index_kept():
    df = pd.DataFrame({"name": ["A Ltd", "B LP"]}, index=[5, 9])
    ctx = RunContext(left=df, right=df, mode="dedup")
    out = registry.build("normalizer", "name_parser")(df, ctx)
    assert list(out.index) == [5, 9]
    assert out["name"].tolist() == ["A Ltd", "B LP"]
    assert "name_base" not in df.columns


def test_missing_name():
    r = parse_names(["Acme Ltd", None, ""])
    assert pd.isna(r["name_base"].iloc[1])
    assert r["legal_form"].tolist() == ["LTD", "", ""]


def test_user_tables_extend():
    tables = {"legal_forms": [{"code": "XYZ", "variants": ["xyzco"], "countries": []}],
              "designators": {"qualifiers": {"trust": ["trust"]}, "prefix_words": ["unit"]}}
    r = parse("Acme Unit B Trust Xyzco", tables=tables)
    assert r["legal_form"] == "XYZ"
    assert r["designator_quals"] == "trust"
    assert r["designator_series"] == "B"
    assert parse("Acme Ltd", tables=tables)["legal_form"] == "LTD"


def test_user_tables_replace():
    tables = {"legal_forms": [{"code": "XYZ", "variants": ["xyzco"], "countries": []}]}
    assert parse("Acme Ltd", tables=tables, merge="replace")["legal_form"] == ""
    assert parse("Acme Xyzco", tables=tables, merge="replace")["legal_form"] == "XYZ"


def test_bad_args():
    with pytest.raises(Exception):
        registry.build("normalizer", {"name_parser": {"merge": "bogus"}})
    with pytest.raises(Exception):
        registry.build("normalizer", {"name_parser": {"tables": {"nope": []}}})


def test_vectorised_many_rows():
    names = ["ABC Fund IV, L.P.", "Mix Capital Ltd", "Grupo Bimbo, S.A.B. de C.V."] * 500
    out = parse_names(names)
    assert len(out) == 1500
    assert out["legal_form"].iloc[:3].tolist() == ["LP", "LTD", "SABDECV"]
    assert out["designator_nums"].iloc[0] == "4"


def test_yaml_tables_load_and_packaged():
    from deduplix.normalize import load_table
    for t in ("legal_forms", "designators", "abbreviations", "stopwords"):
        assert load_table(t + ".yaml")
    codes = {f["code"] for f in load_table("legal_forms")}
    seed = ("LTD PLC LLC LLP LP INC CORP CO GMBH AG KG SA SAS SARL SADECV SABDECV SPA SRL BV NV AB AS OY "
            "PTY PTE BHD SDN KK PVT LIMITADA EIRELI").split()
    for c in seed:
        assert c in codes
