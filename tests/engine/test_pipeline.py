import numpy as np
import pandas as pd
import pytest

from deduplix.checkpointing import FileCheckpointer
from deduplix.engine import Pipeline, ThresholdRouter, register
from deduplix.engine.hooks import CheckpointHook
from deduplix.engine.schema import DECISION, I1, I2, SCORE
from deduplix.exceptions import ConfigurationError, DataValidationError

from .conftest import AllPairsBlocker, FirstLetterBlocker, LenRatio, same_name


def pipe(**kw):
    base = dict(blockers=[AllPairsBlocker()], comparators=[same_name],
                router=ThresholdRouter(0.9, 0.4))
    base.update(kw)
    return Pipeline(**base)


def test_stage_order_and_stats(people):
    calls = []

    class H:
        def on_run_start(self, ctx): calls.append("start")
        def on_stage_start(self, stage, ctx, pairs): calls.append(stage)
        def on_stage_end(self, stage, ctx, pairs, seconds): pass
        def on_run_end(self, ctx, result): calls.append("end")
        def on_error(self, stage, ctx, exc): calls.append("error")

    res = pipe(hooks=[H()]).dedupe(people)
    assert calls == ["start", "normalize", "block", "compare", "score", "route", "validate", "cluster", "end"]
    assert [s["stage"] for s in res.stats["stages"]][:2] == ["normalize", "block"]
    assert res.stats["n_pairs"] == 6


def test_route_decisions_and_clusters(people):
    res = pipe().dedupe(people)
    p = res.pairs
    assert set(p.columns) >= {I1, I2, "blk_allpairs", "blk_score_allpairs", "f_same", SCORE, "route",
                              DECISION, "decision_prob", "decision_reason", "decided_by"}
    m = res.matches
    assert list(zip(m[I1], m[I2])) == [(0, 1)]  # Acme / ACME
    assert (p.loc[p[SCORE] < 0.4, DECISION] == "non_match").all()
    cid = res.records["cluster_id"].to_numpy()
    assert cid[0] == cid[1] and len(set(cid)) == 3
    assert (p[I1] < p[I2]).all()


def test_review_band_and_undecided_policy(people):
    router = ThresholdRouter(accept_above=1.1, reject_below=-1)  # everything is review
    res = pipe(router=router).dedupe(people)
    assert (res.pairs[DECISION] == "unvalidated").all() and len(res.review) == len(res.pairs)
    res = pipe(router=router, undecided="match").dedupe(people)
    assert (res.pairs[DECISION] == "match").all()
    assert (res.pairs["decided_by"] == "policy:match").all()
    assert len(set(res.records["cluster_id"])) == 1
    with pytest.raises(ConfigurationError):
        Pipeline(undecided="maybe")


def test_validator_scope_filtering(people):
    seen = {}

    def make(scope):
        def v(pairs, left, right, ctx):
            seen[scope] = len(pairs)
            return pd.DataFrame({"decision": "non_match", "decision_reason": "no"}, index=pairs.index)
        v.name, v.scope = f"v_{scope}", scope
        return v

    base = dict(router=ThresholdRouter(0.9, 0.0))  # accept exact, rest review
    res = pipe(validators=[make("review")], **base).dedupe(people)
    assert seen["review"] == 5  # one accepted pair is out of scope
    assert (res.matches[I1].tolist(), res.matches[I2].tolist()) == ([0], [1])
    res = pipe(validators=[make("accepted")], **base).dedupe(people)
    assert seen["accepted"] == 1 and len(res.matches) == 0  # validator overrides accepted pair
    assert (res.pairs.loc[res.pairs["decided_by"] == "v_accepted", "decision_reason"] == "no").all()
    pipe(validators=[make("all")], **base).dedupe(people)
    assert seen["all"] == 6


def test_validator_may_leave_rows_undecided(people):
    def first(pairs, left, right, ctx):
        return pd.DataFrame({"decision": ["match"]}, index=pairs.index[:1])

    first.name = "first"
    second_seen = []

    def second(pairs, left, right, ctx):
        second_seen.append(len(pairs))
        return pd.DataFrame({"decision": "non_match"}, index=pairs.index)

    res = pipe(router=ThresholdRouter(2, -1), validators=[first, second]).dedupe(people)
    assert second_seen == [5]
    assert (res.pairs["decided_by"] == "first").sum() == 1


def test_no_router_means_everything_review(people):
    res = Pipeline(blockers=[AllPairsBlocker()], comparators=[same_name]).dedupe(people)
    assert (res.pairs["route"] == "review").all()


def test_score_without_scorer(people):
    res = Pipeline(blockers=[AllPairsBlocker()], comparators=[same_name]).dedupe(people)
    assert res.pairs[SCORE].equals(res.pairs["f_same"].astype("float32"))
    with pytest.raises(ConfigurationError):
        Pipeline(blockers=[AllPairsBlocker()], comparators=[same_name, LenRatio()]).dedupe(people)


def test_column_collision_raises(people):
    with pytest.raises(ConfigurationError, match="duplicate columns"):
        Pipeline(blockers=[AllPairsBlocker()], comparators=[same_name, same_name]).dedupe(people)


def test_max_pairs_guard_names_biggest_blocker(people):
    with pytest.raises(ConfigurationError, match="contributed the most"):
        Pipeline(blockers=[AllPairsBlocker(), FirstLetterBlocker()], max_pairs=3).dedupe(people)
    with pytest.raises(ConfigurationError, match="all-pairs"):
        Pipeline(max_pairs=3).dedupe(people)


def test_no_blocker_uses_all_pairs(people):
    res = Pipeline(comparators=[same_name]).dedupe(people)
    assert len(res.pairs) == 6


def test_blocker_union_provenance(people):
    res = Pipeline(blockers=[AllPairsBlocker(), FirstLetterBlocker()], comparators=[same_name]).dedupe(people)
    p = res.pairs
    assert p["blk_allpairs"].all()
    assert p.loc[(p[I1] == 0) & (p[I2] == 1), "blk_firstletter"].iloc[0]
    assert not p.loc[(p[I1] == 0) & (p[I2] == 3), "blk_firstletter"].iloc[0]


def test_missing_columns_raise_validation_error_and_value_error(people):
    with pytest.raises(DataValidationError):
        pipe().dedupe(people, id_column="nope")
    with pytest.raises(ValueError):
        pipe().dedupe(people, name_column="nope")


def test_link_mode_and_legacy_conversion(people):
    right = pd.DataFrame({"rid": ["a", "b"], "rname": ["acme", "Zed"]})
    res = pipe(blockers=[AllPairsBlocker()], comparators=[same_name]).link(
        people, right, right_id_column="rid", right_name_column="rname")
    legacy = res.to_legacy()
    assert list(legacy.cross_matches.columns)[:6] == [
        "df1_id", "df1_name", "df2_id", "df2_name", "similarity_score", "validation_reason"]
    assert len(legacy.cross_matches) == 3  # Acme, ACME, Zed
    assert legacy.cross_matches["similarity_score"].eq(100).all()
    assert legacy.df2_metadata["id_col"] == "rid"


def test_dedup_legacy_numbering_and_columns(people):
    legacy = pipe().dedupe(people).to_legacy()
    g = legacy.entity_groups
    assert list(g.columns) == ["entity_id", "entity_name", "group_id"]
    assert g["group_id"].tolist() == [1, 1, 0, 0]  # smallest row position first, singletons 0
    assert list(legacy.duplicate_pairs.columns) == ["id1", "id2", "name1", "name2", "similarity_score"]
    assert legacy.statistics["duplicate_groups"] == 1 and legacy.statistics["singleton_entities"] == 2


def test_explain_and_save(people, tmp_path):
    res = pipe().dedupe(people)
    row = res.explain(10, 11)
    assert row["name1"] == "Acme" and row["decision"] == "match"
    assert res.explain(11, 10)["id1"] == 10
    with pytest.raises(KeyError):
        res.explain(10, 999)
    out = res.save(tmp_path / "out")
    assert {p.name for p in out.iterdir()} >= {"pairs.parquet", "records.parquet", "pipeline.yaml", "stats.json"}


def test_debug_validation_catches_bad_scores(people):
    def bad(pairs, ctx):
        return pd.Series(2.0, index=pairs.index)

    p = pipe(scorer=bad)
    with pytest.raises(ConfigurationError, match="score"):
        from deduplix.engine.context import ColumnMap, RunContext
        ctx = RunContext(left=people, right=people, mode="dedup", columns=ColumnMap(), debug=True)
        p._run(ctx)


def test_checkpoint_hook_invalidates_only_downstream(people, tmp_path):
    cp = FileCheckpointer(str(tmp_path))
    counts = {"cmp": 0}

    def counting(pairs, left, right, ctx):
        counts["cmp"] += 1
        return same_name(pairs, left, right, ctx)

    def build(accept):
        return Pipeline(blockers=[AllPairsBlocker()], comparators=[counting],
                        router=ThresholdRouter(accept, 0.4), hooks=[CheckpointHook(cp)])

    r1 = build(0.9).dedupe(people)
    assert counts["cmp"] == 1
    r2 = build(0.9).dedupe(people)
    assert counts["cmp"] == 1  # compare restored
    assert all(s["restored"] for s in r2.stats["stages"] if s["stage"] in ("block", "compare", "route"))
    pd.testing.assert_frame_equal(r1.pairs, r2.pairs)
    r3 = build(0.5).dedupe(people)  # different threshold: compare cached, route recomputed
    stages = {s["stage"]: s["restored"] for s in r3.stats["stages"]}
    assert stages["compare"] and not stages["route"]
    assert counts["cmp"] == 1
