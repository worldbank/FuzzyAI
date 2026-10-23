"""Per-cluster diagnostics for review."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..engine.schema import CLUSTER_ID, I1, I2, SCORE
from .components import match_edges


def cluster_report(pairs: pd.DataFrame, clusters: Any, density_threshold: float = 0.5) -> pd.DataFrame:
    """Size, match-edge density and minimum internal score per cluster (dedup mode).

    Parameters
    ----------
    pairs : pairs table (``i1``, ``i2``, optional ``decision`` and ``score``).
    clusters : one label per record (array, Series, or a frame with ``cluster_id``).
    density_threshold : clusters of 3+ records with density below this get ``review=True``.

    Returns
    -------
    DataFrame with ``cluster_id, size, n_edges, density, min_score, review``; singletons omitted.
    """
    if isinstance(clusters, pd.DataFrame):
        clusters = clusters[CLUSTER_ID]
    labels = np.asarray(clusters)
    sizes = pd.Series(labels).value_counts()
    sizes = sizes[sizes > 1]
    cols = ["cluster_id", "size", "n_edges", "density", "min_score", "review"]
    if sizes.empty:
        return pd.DataFrame({c: [] for c in cols})
    e = match_edges(pairs)
    a, b = e[I1].to_numpy(dtype=np.int64), e[I2].to_numpy(dtype=np.int64)
    same = labels[a] == labels[b]
    df = pd.DataFrame({"cluster_id": labels[a][same],
                       "score": (e[SCORE].to_numpy(dtype="float64")[same] if SCORE in e.columns
                                 else np.full(int(same.sum()), np.nan))})
    agg = df.groupby("cluster_id")["score"].agg(n_edges="size", min_score="min")
    out = pd.DataFrame({"cluster_id": sizes.index.to_numpy(), "size": sizes.to_numpy()})
    out = out.merge(agg, left_on="cluster_id", right_index=True, how="left")
    out["n_edges"] = out["n_edges"].fillna(0).astype("int64")
    out["density"] = out["n_edges"] / (out["size"] * (out["size"] - 1) / 2.0)
    out["review"] = (out["density"] < density_threshold) & (out["size"] > 2)
    return out.sort_values("cluster_id").reset_index(drop=True)[cols]
