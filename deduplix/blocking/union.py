"""Union of candidate sets from several blockers, with provenance flags."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..engine.schema import BLK_PREFIX, BLK_SCORE_PREFIX, I1, I2, index_dtype

Frames = Union[Dict[str, pd.DataFrame], Iterable[Tuple[str, pd.DataFrame]]]


def union_candidates(
    frames: Frames,
    n_right: int,
    mode: str = "dedup",
    n_left: Optional[int] = None,
    max_pairs_per_record: Optional[int] = None,
) -> pd.DataFrame:
    """Merge candidate frames (``i1``, ``i2`` [, ``score``]) into one pairs table.

    Each pair is encoded as ``i1 * n_right + i2`` (int64) and de-duplicated with
    ``np.unique``; provenance (``blk_<name>``) and the blocker's own score
    (``blk_score_<name>``) are scattered back, so no groupby over 10^8 rows is needed.
    In dedup mode pairs are canonicalised to ``i1 < i2`` and self pairs dropped.
    """
    items = list(frames.items()) if isinstance(frames, dict) else list(frames)
    n_left = n_right if n_left is None else n_left
    dt = index_dtype(n_left, n_right)
    nr = np.int64(max(n_right, 1))

    keys, prepared = [], []
    for name, df in items:
        a = df[I1].to_numpy(dtype=np.int64)
        b = df[I2].to_numpy(dtype=np.int64)
        score = df["score"].to_numpy(dtype=np.float32) if "score" in df.columns else None
        if mode == "dedup":
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            keep = lo != hi
            a, b = lo[keep], hi[keep]
            score = score[keep] if score is not None else None
        k = a * nr + b
        keys.append(k)
        prepared.append((name, k, score))

    if not keys or sum(len(k) for k in keys) == 0:
        out = pd.DataFrame({I1: np.array([], dtype=dt), I2: np.array([], dtype=dt)})
        for name, _, s in prepared:
            out[BLK_PREFIX + name] = np.array([], dtype=bool)
            if s is not None:
                out[BLK_SCORE_PREFIX + name] = np.array([], dtype=np.float32)
        return out

    uniq = np.unique(np.concatenate(keys))
    out = pd.DataFrame({I1: (uniq // nr).astype(dt), I2: (uniq % nr).astype(dt)})
    for name, k, score in prepared:
        pos = np.searchsorted(uniq, k)
        flag = np.zeros(len(uniq), dtype=bool)
        flag[pos] = True
        out[BLK_PREFIX + name] = flag
        if score is not None:
            col = np.full(len(uniq), np.nan, dtype=np.float32)
            order = np.argsort(score, kind="stable")  # ascending: the largest score is written last
            col[pos[order]] = score[order]
            out[BLK_SCORE_PREFIX + name] = col

    if max_pairs_per_record is not None:
        out = _cap_per_record(out, max_pairs_per_record)
    return out


def _cap_per_record(pairs: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep, per left record, the k pairs with the highest max blocker score."""
    score_cols = [c for c in pairs.columns if c.startswith(BLK_SCORE_PREFIX)]
    if not score_cols:
        return pairs
    best = pairs[score_cols].max(axis=1).fillna(0.0)
    rank = best.groupby(pairs[I1]).rank(method="first", ascending=False)
    return pairs[rank <= k].reset_index(drop=True)
