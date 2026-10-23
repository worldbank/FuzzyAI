import numpy as np
import pandas as pd
import pytest

from deduplix.cluster import GuardedComponents, LouvainClusters, cluster_report
from deduplix.cluster.components import ConnectedComponents
from deduplix.engine import registry
from deduplix.engine.context import ColumnMap, RunContext


def make_pairs(edges):
    df = pd.DataFrame(edges, columns=["i1", "i2", "score"])
    df["decision"] = pd.Series(["match"] * len(df), dtype=object)
    df["decided_by"] = None
    return df


def ctx_for(n):
    return RunContext(left=pd.DataFrame({"id": range(n), "name": ["x"] * n}), right=None, mode="dedup")


def test_registered():
    assert registry.build("clusterer", {"guarded": {"max_size": 3}}).max_size == 3
    assert registry.build("clusterer", "louvain") is not None
    assert registry.build("consolidator", "rules") is not None
    assert registry.build("consolidator", "llm") is not None


def test_chain_splits_at_weak_edge():
    pairs = make_pairs([(0, 1, 0.95), (1, 2, 0.51), (2, 3, 0.93)])
    ctx = ctx_for(4)
    labels = GuardedComponents(max_size=2)(pairs, 4, ctx)
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[1] != labels[2]
    assert pairs.loc[1, "decision"] == "non_match"
    assert pairs.loc[1, "decided_by"] == "cluster.guarded"
    assert (pairs.loc[[0, 2], "decision"] == "match").all()
    assert (ctx.artifacts["cluster_flag"] == "").all()


def test_small_components_untouched():
    pairs = make_pairs([(0, 1, 0.9), (1, 2, 0.8)])
    labels = GuardedComponents(max_size=5)(pairs, 3, ctx_for(3))
    assert len(set(labels)) == 1 and (pairs["decision"] == "match").all()


def test_flag_strategy():
    pairs = make_pairs([(0, 1, 0.9), (1, 2, 0.8), (3, 4, 0.9)])
    ctx = ctx_for(5)
    labels = GuardedComponents(max_size=2, strategy="flag")(pairs, 5, ctx)
    assert len(set(labels[:3])) == 1
    assert list(ctx.artifacts["cluster_flag"]) == ["oversized"] * 3 + ["", ""]
    assert (pairs["decision"] == "match").all()


def test_min_edge_score():
    pairs = make_pairs([(0, 1, 0.9), (1, 2, 0.6)])
    labels = GuardedComponents(max_size=10, min_edge_score=0.7)(pairs, 3, ctx_for(3))
    assert labels[0] == labels[1] != labels[2]
    assert pairs.loc[1, "decision"] == "non_match"


def two_cliques():
    edges = [(a, b, 0.95) for a in range(4) for b in range(a + 1, 4)]
    edges += [(a, b, 0.95) for a in range(4, 8) for b in range(a + 1, 8)]
    edges += [(3, 4, 0.6)]
    return edges


@pytest.mark.parametrize("strategy", ["drop_weakest", "louvain"])
def test_two_cliques_split(strategy):
    pairs = make_pairs(two_cliques())
    labels = GuardedComponents(max_size=4, strategy=strategy, random_state=1)(pairs, 8, ctx_for(8))
    assert len(set(labels[:4])) == 1 and len(set(labels[4:])) == 1 and labels[0] != labels[4]
    assert pairs.iloc[-1]["decision"] == "non_match"


def test_louvain_reproducible():
    rng = np.random.RandomState(0)
    edges = [(i, j, float(rng.uniform(0.5, 1))) for i in range(30) for j in range(i + 1, 30)
             if rng.rand() < 0.15]
    out = [LouvainClusters(random_state=7)(make_pairs(edges), 30) for _ in range(3)]
    assert (out[0] == out[1]).all() and (out[1] == out[2]).all()


def test_louvain_clusterer_splits_cliques():
    labels = LouvainClusters(random_state=0)(make_pairs(two_cliques()), 8)
    assert labels[0] != labels[4] and len(set(labels[:4])) == 1


def test_guarded_matches_components_when_not_oversized():
    pairs = make_pairs([(0, 1, 0.9), (2, 3, 0.9)])
    a = GuardedComponents(max_size=10)(pairs, 5, ctx_for(5))
    b = ConnectedComponents()(pairs, 5)
    assert (a == b).all()


def test_cluster_report():
    pairs = make_pairs([(0, 1, 0.9), (1, 2, 0.8), (3, 4, 0.7)])
    labels = np.array([0, 0, 0, 1, 1, 2])
    rep = cluster_report(pairs, labels)
    row0 = rep[rep.cluster_id == 0].iloc[0]
    assert row0["size"] == 3 and row0["n_edges"] == 2
    assert row0["density"] == pytest.approx(2 / 3) and row0["min_score"] == pytest.approx(0.8)
    assert not row0["review"]
    assert 2 not in rep.cluster_id.values
    pairs2 = make_pairs([(0, 1, 0.9), (1, 2, 0.8)] + [(3, 4, 0.9), (4, 5, 0.9)])
    rep2 = cluster_report(pairs2, np.array([0, 0, 0, 0, 0, 0]))
    assert rep2.iloc[0]["review"]


def test_rules_consolidator():
    from deduplix.cluster.consolidators import RulesConsolidator

    matches = pd.DataFrame({"df1_id": [1, 1], "df1_name": ["a", "a"], "df2_id": [10, 11],
                            "df2_name": ["a1", "a2"], "similarity_score": [90.0, 95.0],
                            "validation_reason": ["", ""]})
    left = pd.DataFrame({"id": [1], "name": ["a"]})
    right = pd.DataFrame({"id": [10, 11], "name": ["a1", "a2"]})
    ctx = RunContext(left=left, right=right, mode="link", columns=ColumnMap())
    out = RulesConsolidator({"enabled": True})(matches, left, right, ctx)
    assert len(out) == 1
