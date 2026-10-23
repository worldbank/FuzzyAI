"""Louvain community detection on the match graph."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..engine.registry import register
from ..engine.schema import I1, I2, SCORE
from .components import ConnectedComponents, match_edges


@register("clusterer", "louvain")
class LouvainClusters:
    """Split each connected component into Louvain communities (weight = score).

    Reproducible for a fixed ``random_state`` (defaults to ``ctx.random_state``).
    """

    def __init__(self, resolution: float = 1.0, random_state: Optional[int] = None) -> None:
        self.resolution = resolution
        self.random_state = random_state

    def __call__(self, pairs, n_records: int, ctx: Any = None) -> np.ndarray:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities

        seed = self.random_state if self.random_state is not None else getattr(ctx, "random_state", 0)
        edges = match_edges(pairs)
        base = ConnectedComponents()(edges, n_records)
        a = edges[I1].to_numpy(dtype=np.int64)
        b = edges[I2].to_numpy(dtype=np.int64)
        w = (edges[SCORE].astype("float64").fillna(0.0).to_numpy() if SCORE in edges.columns
             else np.ones(len(edges)))
        labels = base.copy()
        next_label = int(base.max()) + 1 if len(base) else 0
        sizes = np.bincount(base) if len(base) else np.array([], dtype=int)
        big = set(np.flatnonzero(sizes > 2).tolist())  # 1-2 nodes cannot split
        if not big:
            return labels
        g = {lab: nx.Graph() for lab in big}
        for x, y, s in zip(a, b, w):
            lab = int(base[x])
            if lab in g:
                g[lab].add_edge(int(x), int(y), score=float(s))
        for lab in sorted(big):
            comms = louvain_communities(g[lab], weight="score", resolution=self.resolution, seed=seed)
            comms = sorted((sorted(c) for c in comms), key=lambda c: c[0])
            for comm in comms[1:]:
                labels[comm] = next_label
                next_label += 1
        return labels
