"""End-to-end checks with the real components against the 0.1.0 golden fixtures."""
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import deduplix
from deduplix import DeduplicationPipeline, EngineMatcher, FuzzyMatcher, Pipeline
from deduplix.engine import as_engine_validator, as_legacy_validator
from deduplix.validators import RuleBasedValidator
from scripts.make_golden import relabel

GOLD = Path(__file__).resolve().parent.parent / "golden"

CONFIG = {
    "normalize": ["strip_whitespace", "lowercase"],
    "blocking": [{"brute_force": {"scorer": "token_sort_ratio", "threshold": 85.0}}],
    "compare": {"name": {"fn": "token_sort_ratio", "left": "name_norm"}},
    "score": {"feature": {"name": "name"}},
    "route": {"threshold": {"accept_above": 0.85, "reject_below": 0.85}},
    "options": {"legacy_score_feature": "name"},
}


def sorted_pairs(df):
    return df.sort_values(["id1", "id2"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def companies():
    return pd.read_parquet(GOLD / "input_companies.parquet")


def test_top_level_exports():
    assert deduplix.__version__ == "0.2.0"
    for name in ("Pipeline", "PipelineResult", "register", "load_pipeline", "EngineMatcher"):
        assert name in deduplix.__all__


def test_engine_matcher_equals_golden_pairs(companies):
    res = EngineMatcher(CONFIG).find_matches(companies)
    expected = pd.read_parquet(GOLD / "pairs_token_sort_ratio_0.parquet")
    got = sorted_pairs(res.pairs)
    assert_frame_equal(got[expected.columns], expected, check_dtype=False, rtol=1e-4)


def test_legacy_pipeline_with_engine_matcher_matches_golden_groups(companies):
    res = DeduplicationPipeline(EngineMatcher(CONFIG), checkpoint=False).run(companies)
    legacy = DeduplicationPipeline(FuzzyMatcher(threshold=85.0), checkpoint=False).run(companies)
    assert res.statistics == pytest.approx(legacy.statistics)
    assert_frame_equal(relabel(res.entity_groups), relabel(legacy.entity_groups), check_dtype=False)


def test_full_engine_pipeline_to_legacy(companies):
    legacy = Pipeline.from_config(CONFIG).dedupe(companies).to_legacy()
    expected = DeduplicationPipeline(FuzzyMatcher(threshold=85.0), checkpoint=False).run(companies)
    assert legacy.statistics["duplicate_groups"] == expected.statistics["duplicate_groups"]
    assert legacy.statistics["duplicate_pairs"] == expected.statistics["duplicate_pairs"]
    assert list(legacy.entity_groups.columns) == ["entity_id", "entity_name", "group_id"]
    mine = legacy.entity_groups.set_index("entity_id")["group_id"]
    theirs = expected.entity_groups.set_index("entity_id")["group_id"]
    pd.testing.assert_series_equal((mine > 0), (theirs > 0), check_names=False)


def test_fuzzy_matcher_blocking_option_matches_cdist(companies):
    base = FuzzyMatcher(threshold=85.0).find_matches(companies).pairs
    blocked = FuzzyMatcher(threshold=85.0, blocking=[{"brute_force": {"threshold": 80.0}}]).find_matches(
        companies).pairs
    assert_frame_equal(sorted_pairs(blocked)[base.columns], sorted_pairs(base), check_dtype=False, rtol=1e-4)


def test_fuzzy_matcher_blocking_cross():
    df1 = pd.read_parquet(GOLD / "input_cross1.parquet")
    df2 = pd.read_parquet(GOLD / "input_cross2.parquet")
    args = (df1, df2, "company_id", "company_name", "entity_id", "name")
    base = FuzzyMatcher(threshold=85.0).find_cross_matches(*args).pairs
    blocked = FuzzyMatcher(threshold=85.0, blocking=[{"brute_force": {"threshold": 80.0}}]
                           ).find_cross_matches(*args).pairs
    assert len(base) == len(blocked) and set(zip(base.id1, base.id2)) == set(zip(blocked.id1, blocked.id2))


def test_validator_adapters_round_trip(companies):
    legacy = RuleBasedValidator(min_score=95.0)
    res = Pipeline.from_config({**CONFIG, "validate": []}).dedupe(companies)
    p = Pipeline.from_config(CONFIG)
    p.validators = [as_engine_validator(legacy, scope="all")]
    out = p.dedupe(companies)
    expected = (res.pairs["score"].astype("float64") * 100 >= 95.0).sum()
    assert (out.pairs["decision"] == "match").sum() == expected
    assert (out.pairs.loc[out.pairs["decision"] == "non_match", "decided_by"] == "RuleBasedValidator").all()

    back = as_legacy_validator(as_engine_validator(RuleBasedValidator(min_score=95.0), scope="all"))
    pairs = EngineMatcher(CONFIG).find_matches(companies)
    r = back.validate(pairs, original_df=companies)
    assert len(r.validated_pairs) == expected and len(r.removed_pairs) == len(pairs.pairs) - expected


def test_default_pipeline_runs(companies):
    res = Pipeline.default().dedupe(companies.head(200))
    assert res.stats["n_pairs"] > 0
    assert set(res.pairs["decision"].astype(str)) <= {"match", "non_match", "unvalidated"}
