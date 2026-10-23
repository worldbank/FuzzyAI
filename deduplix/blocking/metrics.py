"""Blocking quality metrics: reduction ratio, pair completeness, pairs quality."""
from __future__ import annotations

from typing import Iterable, List, Set, Tuple

import numpy as np
import pandas as pd

from ..engine.schema import BLK_PREFIX, BLK_SCORE_PREFIX, I1, I2


def reduction_ratio(n_candidates: int, n_left: int, n_right: int, mode: str = "dedup") -> float:
    """1 - candidates / all possible pairs (n(n-1)/2 in dedup mode, n_left * n_right in link)."""
    total = n_left * (n_left - 1) // 2 if mode == "dedup" else n_left * n_right
    if total <= 0:
        return 1.0
    return 1.0 - float(n_candidates) / float(total)


def _gold_keys(gold, mode: str = "dedup") -> Set[Tuple[int, int]]:
    if isinstance(gold, pd.DataFrame):
        a, b = gold[I1].to_numpy(), gold[I2].to_numpy()
        pairs: Iterable = zip(a.tolist(), b.tolist())
    else:
        pairs = gold
    out = set()
    for a, b in pairs:
        a, b = int(a), int(b)
        out.add((min(a, b), max(a, b)) if mode == "dedup" else (a, b))
    return out


def _cand_keys(candidates: pd.DataFrame, mode: str) -> Set[Tuple[int, int]]:
    a = candidates[I1].to_numpy().astype(np.int64)
    b = candidates[I2].to_numpy().astype(np.int64)
    if mode == "dedup":
        a, b = np.minimum(a, b), np.maximum(a, b)
    return set(zip(a.tolist(), b.tolist()))


def pair_completeness(candidates: pd.DataFrame, gold, mode: str = "dedup") -> float:
    """Share of gold (true match) pairs present in the candidates. ``gold``: i1/i2 frame or pairs."""
    g = _gold_keys(gold, mode)
    if not g:
        return 1.0
    return len(g & _cand_keys(candidates, mode)) / len(g)


def pairs_quality(candidates: pd.DataFrame, gold, mode: str = "dedup") -> float:
    """Share of candidates that are gold pairs (precision of the candidate set)."""
    c = _cand_keys(candidates, mode)
    if not c:
        return 0.0
    return len(c & _gold_keys(gold, mode)) / len(c)


def blocking_report(candidates: pd.DataFrame, gold, mode: str = "dedup") -> pd.DataFrame:
    """One row per blocker (``blk_<name>`` columns) plus a ``union`` row.

    Columns: ``blocker, candidates, pair_completeness, pairs_quality, unique_true_matches``;
    the last counts true matches found only by that blocker (marginal recall).
    """
    g = _gold_keys(gold, mode)
    names: List[str] = [c[len(BLK_PREFIX):] for c in candidates.columns
                        if c.startswith(BLK_PREFIX) and not c.startswith(BLK_SCORE_PREFIX)]
    a = candidates[I1].to_numpy().astype(np.int64)
    b = candidates[I2].to_numpy().astype(np.int64)
    if mode == "dedup":
        a, b = np.minimum(a, b), np.maximum(a, b)
    is_gold = np.fromiter(((x, y) in g for x, y in zip(a.tolist(), b.tolist())),
                          dtype=bool, count=len(a))
    flags = {n: candidates[BLK_PREFIX + n].to_numpy().astype(bool) for n in names}
    n_flags = sum(f.astype(int) for f in flags.values()) if flags else np.zeros(len(a), int)
    rows = []
    for n, f in flags.items():
        found = int((f & is_gold).sum())
        rows.append({
            "blocker": n, "candidates": int(f.sum()),
            "pair_completeness": found / len(g) if g else 1.0,
            "pairs_quality": found / int(f.sum()) if f.sum() else 0.0,
            "unique_true_matches": int((f & is_gold & (n_flags == 1)).sum()),
        })
    total_found = int(is_gold.sum())
    rows.append({
        "blocker": "union", "candidates": len(candidates),
        "pair_completeness": total_found / len(g) if g else 1.0,
        "pairs_quality": total_found / len(candidates) if len(candidates) else 0.0,
        "unique_true_matches": 0,
    })
    return pd.DataFrame(rows)
