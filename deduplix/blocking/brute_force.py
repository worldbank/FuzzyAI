"""Brute-force blocker: the legacy ``rapidfuzz.process.cdist`` loop. Default below 20,000 records."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from ..engine.registry import register
from .base import BaseBlocker

_SCORERS = {
    "ratio": fuzz.ratio,
    "partial_ratio": fuzz.partial_ratio,
    "token_sort_ratio": fuzz.token_sort_ratio,
    "token_set_ratio": fuzz.token_set_ratio,
    "WRatio": fuzz.WRatio,
}


def _get_scorer(name: str):
    if name in _SCORERS:
        return _SCORERS[name], 1.0
    from rapidfuzz import distance as rfd

    extra = {"jaro": rfd.Jaro.normalized_similarity,
             "jaro_winkler": rfd.JaroWinkler.normalized_similarity}
    if name in extra:
        return extra[name], 100.0  # these return 0-1 in the legacy matcher's multiplier
    raise ValueError(f"Unknown scorer '{name}'. Choose from: {list(_SCORERS) + list(extra)}")


def _preprocess(text: str) -> str:
    """Legacy preprocessing: collapse whitespace, strip, lower case."""
    return re.sub(r"\s+", " ", text.strip()).lower()


@register("blocker", "brute_force")
class BruteForceBlocker(BaseBlocker):
    """All pairs with ``scorer >= threshold`` via batched ``cdist``; score is ``score / 100``."""

    name = "brute_force"

    def __init__(self, scorer: str = "token_sort_ratio", threshold: float = 80.0,
                 batch_size: int = 500, n_workers: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        _get_scorer(scorer)
        self.scorer = scorer
        self.threshold = threshold
        self.batch_size = batch_size
        self.n_workers = n_workers

    def _prepare(self, left, right, ctx) -> None:
        self._lt = [_preprocess(t) for t in self._texts(left, ctx, "left")]
        self._rt = self._lt if self._dedup else [
            _preprocess(t) for t in self._texts(right, ctx, "right")]

    def _block(self, left_idx, right_idx, ctx):
        fn, mult = _get_scorer(self.scorer)
        queries = [self._lt[g] for g in left_idx]
        choices = [self._rt[g] for g in right_idx]
        ok_l = np.array([bool(q) for q in queries])
        ok_r = np.array([bool(c) for c in choices])
        ii, jj, ss = [], [], []
        for start in range(0, len(queries), self.batch_size):
            end = min(start + self.batch_size, len(queries))
            mat = process.cdist(queries[start:end], choices, scorer=fn,
                                score_cutoff=self.threshold, workers=self.n_workers,
                                dtype=np.float32)
            rows, cols = np.nonzero(mat)
            vals = mat[rows, cols] * mult
            rows = rows + start
            keep = ok_l[rows] & ok_r[cols] & (vals >= self.threshold)
            if self._dedup:
                keep &= rows < cols
            ii.append(rows[keep])
            jj.append(cols[keep])
            ss.append(vals[keep] / 100.0)
        if not ii:
            e = np.array([], dtype=np.int64)
            return e, e, np.array([], dtype=np.float32)
        return np.concatenate(ii), np.concatenate(jj), np.concatenate(ss).astype(np.float32)
