"""Legacy outputs must match fixtures generated from 0.1.0 (scripts/make_golden.py)."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from deduplix import DeduplicationPipeline, FuzzyMatcher

G = Path(__file__).parent
ROOT = G.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.make_golden import relabel, sorted_pairs  # noqa: E402
from scripts.make_prompt_snapshot import cases  # noqa: E402


def load(name):
    return pd.read_parquet(G / f"{name}.parquet")


def same(actual, expected):
    assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                       check_dtype=False, rtol=1e-6)


@pytest.mark.parametrize("scorer", ["token_sort_ratio", "token_set_ratio", "WRatio"])
@pytest.mark.parametrize("punct", [0, 1])
def test_matcher_pairs(scorer, punct):
    m = FuzzyMatcher(threshold=85.0, scorer=scorer, remove_punctuation=bool(punct))
    same(sorted_pairs(m.find_matches(load("input_companies")).pairs), load(f"pairs_{scorer}_{punct}"))


def test_dedup_groups_and_pairs():
    res = DeduplicationPipeline(FuzzyMatcher(threshold=85.0), checkpoint=False).run(load("input_companies"))
    same(relabel(res.entity_groups), load("entity_groups"))
    same(sorted_pairs(res.duplicate_pairs), load("duplicate_pairs"))


def test_cross_multi_column():
    m = FuzzyMatcher(threshold=85.0)
    r = m.find_cross_matches(load("input_cross1"), load("input_cross2"), "company_id", "company_name",
                             "entity_id", "name", name_columns2=["name", "trading_name"])
    same(sorted_pairs(r.pairs), load("cross_multi"))


def test_cross_result_columns():
    res = DeduplicationPipeline(FuzzyMatcher(threshold=85.0), checkpoint=False).run_cross_dataset(
        load("input_cross1"), load("input_cross2"), "company_id", "company_name", "entity_id", "name")
    same(res.cross_matches.sort_values(["df1_id", "df2_id"]), load("cross_matches"))


def test_legacy_llm_prompts_unchanged():
    expected = json.loads((G / "legacy_prompts.json").read_text(encoding="utf-8"))
    assert dict(cases()) == expected
