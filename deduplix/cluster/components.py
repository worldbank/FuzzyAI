"""Connected-components clustering over match edges."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..engine.registry import register
from ..engine.schema import DECISION, I1, I2, MATCH


def match_edges(pairs: pd.DataFrame) -> pd.DataFrame:
    """Rows of the pairs table decided as match."""
    if DECISION in pairs.columns:
        return pairs[pairs[DECISION] == MATCH]
    return pairs


@register("clusterer", "components")
class ConnectedComponents:
    """Connected components of the match graph (scipy, networkx fallback).

    Returns one label per record; unmatched records get their own label.
    """

    def __call__(self, pairs: pd.DataFrame, n_records: int, ctx: Any = None) -> np.ndarray:
        edges = match_edges(pairs)
        a = edges[I1].to_numpy(dtype=np.int64)
        b = edges[I2].to_numpy(dtype=np.int64)
        try:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components

            adj = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n_records, n_records))
            _, labels = connected_components(adj, directed=False)
            return labels.astype(np.int64)
        except ImportError:  # pragma: no cover - scipy is optional
            import networkx as nx

            g = nx.Graph()
            g.add_nodes_from(range(n_records))
            g.add_edges_from(zip(a.tolist(), b.tolist()))
            labels = np.zeros(n_records, dtype=np.int64)
            for cid, comp in enumerate(nx.connected_components(g)):
                labels[list(comp)] = cid
            return labels
