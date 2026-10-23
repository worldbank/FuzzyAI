"""String-similarity comparator built on rapidfuzz."""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from ..engine.registry import register
from ..exceptions import ConfigurationError
from ._util import feature_column, pair_text, resolve_column


def _scorers():
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler, Levenshtein

    return {
        "ratio": (fuzz.ratio, 100.0),
        "partial_ratio": (fuzz.partial_ratio, 100.0),
        "token_sort_ratio": (fuzz.token_sort_ratio, 100.0),
        "token_set_ratio": (fuzz.token_set_ratio, 100.0),
        "WRatio": (fuzz.WRatio, 100.0),
        "QRatio": (fuzz.QRatio, 100.0),
        "jaro_winkler": (JaroWinkler.normalized_similarity, 1.0),
        "levenshtein_norm": (Levenshtein.normalized_similarity, 1.0),
    }


STRING_FUNCTIONS = ("ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio", "WRatio",
                    "QRatio", "jaro_winkler", "levenshtein_norm")


@register("comparator", "string_sim")
class StringSim:
    """Pairwise string similarity in [0, 1]; null or blank on either side gives NaN.

    Parameters
    ----------
    fn : str or callable
        One of ``ratio, partial_ratio, token_sort_ratio, token_set_ratio, WRatio, QRatio,
        jaro_winkler, levenshtein_norm`` or a callable ``(str, str) -> float``.
    left, right : str, optional
        Columns to compare (default ``name_norm``, falling back to the name column).
        ``right`` defaults to ``left``.
    workers : int
        Threads for ``rapidfuzz.process.cpdist`` (-1 = all cores).
    scale : float
        Divisor applied to the output of a custom callable (default 1).
    feature : str
        Output column is ``f_<feature>``.
    """

    def __init__(self, fn: Union[str, Callable[[str, str], float]] = "token_sort_ratio",
                 left: Optional[str] = None, right: Optional[str] = None, workers: int = -1,
                 scale: float = 1.0, feature: str = "string_sim"):
        if isinstance(fn, str) and fn not in STRING_FUNCTIONS:
            raise ConfigurationError(f"Unknown string function '{fn}'. Choose from {list(STRING_FUNCTIONS)}",
                                     config_key="fn", config_value=fn)
        if not isinstance(fn, str) and not callable(fn):
            raise ConfigurationError("fn must be a name or a callable", config_key="fn")
        self.fn, self.left, self.right = fn, left, right
        self.workers, self.scale, self.feature = workers, scale, feature

    def __call__(self, pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame, ctx) -> pd.DataFrame:
        lcol = resolve_column(left, self.left, ctx, "left")
        rcol = resolve_column(right, self.right or self.left, ctx, "right")
        a, b, ok = pair_text(pairs, left, right, lcol, rcol)
        out = np.full(len(pairs), np.nan, dtype=np.float32)
        idx = np.flatnonzero(ok)
        if len(idx):
            sa, sb = a[idx].tolist(), b[idx].tolist()
            out[idx] = self._score(sa, sb)
        return pd.DataFrame({feature_column(self.feature): out}, index=pairs.index)

    def _score(self, sa, sb) -> np.ndarray:
        if isinstance(self.fn, str):
            scorer, div = _scorers()[self.fn]
            try:
                from rapidfuzz.process import cpdist
            except ImportError:  # pragma: no cover - rapidfuzz < 3.6
                return np.asarray([scorer(x, y) for x, y in zip(sa, sb)], dtype=np.float32) / div
            res = cpdist(sa, sb, scorer=scorer, workers=self.workers, dtype=np.float32)
            return np.asarray(res, dtype=np.float32) / np.float32(div)
        res = np.fromiter((self.fn(x, y) for x, y in zip(sa, sb)), dtype=np.float32, count=len(sa))
        return res / np.float32(self.scale)
