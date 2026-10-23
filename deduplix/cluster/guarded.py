"""Size-guarded clustering: stop one bad edge from merging two groups."""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..exceptions import ConfigurationError
from ..engine.registry import register
from ..engine.schema import DECIDED_BY, DECISION, DECISION_REASON, I1, I2, MATCH, NON_MATCH, SCORE
from .components import ConnectedComponents

STRATEGIES = ("drop_weakest", "louvain", "flag")
DECIDER = "cluster.guarded"


def _component_sizes(labels: np.ndarray) -> np.ndarray:
    return np.bincount(labels)[labels]


def _split_drop_weakest(nodes: List[int], edges: List[Tuple[float, int, int, int]],
                        max_size: int) -> List[int]:
    """Row numbers of edges to remove so every part of ``nodes`` has <= max_size nodes."""
    import networkx as nx

    removed: List[int] = []
    stack = [(nodes, edges)]
    while stack:
        part_nodes, part_edges = stack.pop()
        if len(part_nodes) <= max_size:
            continue
        # ascending (score, row): the weakest edge goes first, ties broken by row
        ordered = sorted(part_edges, key=lambda e: (e[0], e[1]))
        g = nx.Graph()
        g.add_nodes_from(part_nodes)
        g.add_edges_from((e[2], e[3]) for e in ordered)
        alive = {(min(e[2], e[3]), max(e[2], e[3])): e for e in ordered}
        for e in ordered:
            key = (min(e[2], e[3]), max(e[2], e[3]))
            g.remove_edge(e[2], e[3])
            del alive[key]
            removed.append(e[1])
            comps = list(nx.connected_components(g))
            if len(comps) > 1:
                for comp in comps:
                    sub_edges = [v for (a, b), v in alive.items() if a in comp]
                    stack.append((sorted(comp), sub_edges))
                break
    return removed


def _split_louvain(nodes: List[int], edges: List[Tuple[float, int, int, int]], max_size: int,
                   resolution: float, seed: int) -> List[int]:
    """Edges to remove when oversized parts are split into Louvain communities."""
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    removed: List[int] = []
    stack = [(nodes, edges)]
    while stack:
        part_nodes, part_edges = stack.pop()
        if len(part_nodes) <= max_size:
            continue
        g = nx.Graph()
        g.add_nodes_from(part_nodes)
        for s, row, a, b in part_edges:
            g.add_edge(a, b, score=float(s), row=row)
        comms = louvain_communities(g, weight="score", resolution=resolution, seed=seed)
        if len(comms) <= 1:  # no structure found: fall back to the weakest-edge rule
            removed.extend(_split_drop_weakest(part_nodes, part_edges, max_size))
            continue
        where = {n: k for k, comm in enumerate(comms) for n in comm}
        groups: List[List[Tuple[float, int, int, int]]] = [[] for _ in comms]
        for e in part_edges:
            if where[e[2]] == where[e[3]]:
                groups[where[e[2]]].append(e)
            else:
                removed.append(e[1])
        for k, comm in enumerate(comms):
            stack.append((sorted(comm), groups[k]))
    return removed


@register("clusterer", "guarded")
class GuardedComponents:
    """Connected components with a size guard (drop_weakest, louvain or flag).

    Edges removed to split an oversized component are written back to ``pairs``
    as ``decision="non_match"``, ``decided_by="cluster.guarded"``. With
    ``strategy="flag"`` the component is kept and ``ctx.artifacts["cluster_flag"]``
    marks its records ``"oversized"``.
    """

    def __init__(self, max_size: int = 50, strategy: str = "drop_weakest",
                 min_edge_score: Optional[float] = None, resolution: float = 1.0,
                 random_state: Optional[int] = None) -> None:
        if strategy not in STRATEGIES:
            raise ConfigurationError(f"strategy must be one of {list(STRATEGIES)}, got '{strategy}'")
        if max_size < 2:
            raise ConfigurationError("max_size must be at least 2")
        self.max_size = max_size
        self.strategy = strategy
        self.min_edge_score = min_edge_score
        self.resolution = resolution
        self.random_state = random_state

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _mark(pairs: pd.DataFrame, rows: np.ndarray, reason: str) -> None:
        if len(rows) == 0:
            return
        dec = pairs[DECISION].astype(object).to_numpy().copy()
        dec[rows] = NON_MATCH
        pairs[DECISION] = dec
        by = (pairs[DECIDED_BY].astype(object).to_numpy().copy() if DECIDED_BY in pairs.columns
              else np.full(len(pairs), None, dtype=object))
        by[rows] = DECIDER
        pairs[DECIDED_BY] = by
        why = (pairs[DECISION_REASON].astype(object).to_numpy().copy()
               if DECISION_REASON in pairs.columns else np.full(len(pairs), None, dtype=object))
        why[rows] = reason
        pairs[DECISION_REASON] = why

    @staticmethod
    def _labels(pairs: pd.DataFrame, alive: np.ndarray, n_records: int) -> np.ndarray:
        sub = pd.DataFrame({I1: pairs[I1].to_numpy()[alive], I2: pairs[I2].to_numpy()[alive]})
        return ConnectedComponents()(sub, n_records)

    def __call__(self, pairs: pd.DataFrame, n_records: int, ctx: Any = None) -> np.ndarray:
        n = len(pairs)
        is_match = (pairs[DECISION] == MATCH).to_numpy() if DECISION in pairs.columns \
            else np.ones(n, dtype=bool)
        score = (pairs[SCORE].astype("float64").fillna(0.0).to_numpy() if SCORE in pairs.columns
                 else np.ones(n))
        if self.min_edge_score is not None:
            weak = np.flatnonzero(is_match & (score < self.min_edge_score))
            if DECISION in pairs.columns:
                self._mark(pairs, weak, f"edge score below min_edge_score={self.min_edge_score}")
            is_match = is_match.copy()
            is_match[weak] = False
        labels = self._labels(pairs, is_match, n_records)
        sizes = _component_sizes(labels)
        over_nodes = sizes > self.max_size

        if self.strategy == "flag":
            flag = np.where(over_nodes, "oversized", "").astype(object)
            if ctx is not None:
                ctx.artifacts["cluster_flag"] = flag
            return labels

        if over_nodes.any():
            i1, i2 = pairs[I1].to_numpy(), pairs[I2].to_numpy()
            rows = np.flatnonzero(is_match & over_nodes[i1])
            by_comp: dict = {}
            for r in rows:
                by_comp.setdefault(int(labels[i1[r]]), []).append(
                    (float(score[r]), int(r), int(i1[r]), int(i2[r])))
            seed = self.random_state if self.random_state is not None else getattr(
                ctx, "random_state", 0)
            removed: List[int] = []
            for lab, edges in sorted(by_comp.items()):
                nodes = np.flatnonzero(labels == lab).tolist()
                if self.strategy == "drop_weakest":
                    removed.extend(_split_drop_weakest(nodes, edges, self.max_size))
                else:
                    removed.extend(_split_louvain(nodes, edges, self.max_size, self.resolution, seed))
            rem = np.array(sorted(set(removed)), dtype=np.int64)
            self._mark(pairs, rem, f"edge removed to split cluster larger than {self.max_size}")
            is_match = is_match.copy()
            is_match[rem] = False
            labels = self._labels(pairs, is_match, n_records)
        if ctx is not None:
            ctx.artifacts["cluster_flag"] = np.full(n_records, "", dtype=object)
        return labels
