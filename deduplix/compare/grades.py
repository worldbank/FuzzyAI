"""D&B-style match grades: A same, B similar, F different, Z missing."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..engine.registry import register
from ..exceptions import ConfigurationError
from ._util import feature_column

DEFAULT_BANDS = {"A": 0.95, "B": 0.80}
_WORD_GRADES = {"same": "A", "ok": "A", "equivalent": "A", "review": "B", "different": "F",
                "reject": "F", "missing": "Z"}


def to_grades(series: pd.Series, bands: Dict[str, float]) -> np.ndarray:
    """Object array of one-letter grades.

    Numbers use ``bands`` (below all bands -> F, NaN -> Z); strings map same/ok/equivalent -> A,
    review -> B, different/reject -> F, missing -> Z, and a single letter passes through.
    """
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        v = series.to_numpy(dtype="float64", na_value=np.nan)
        out = np.full(len(v), "F", dtype=object)
        for letter, th in sorted(bands.items(), key=lambda kv: kv[1]):  # highest band written last
            out[v >= th] = letter
        out[np.isnan(v)] = "Z"
        return out

    def one(x):
        if pd.isna(x):
            return "Z"
        s = str(x)
        return _WORD_GRADES.get(s.lower(), s.upper()[:1] or "Z")

    return series.astype(object).map(one).to_numpy(dtype=object)


def _check_bands(bands):
    bands = dict(bands or DEFAULT_BANDS)
    if not bands:
        raise ConfigurationError("bands must not be empty", config_key="bands")
    return bands


def _need(pairs, f):
    col = feature_column(f)
    if col not in pairs.columns:
        raise ConfigurationError(f"feature column '{col}' is not in the pairs table; "
                                 "declare its comparator earlier", config_key="source")
    return col


@register("comparator", "grade")
class Grade:
    """One-letter grade of an existing feature: A (same), B (similar), F (different), Z (missing).

    ``source`` is the feature to grade (``name`` or ``f_name``); the output is ``f_<feature>``.
    """

    def __init__(self, source: Optional[str] = None, bands: Optional[Dict[str, float]] = None,
                 feature: str = "grade"):
        self.source = source
        self.bands = _check_bands(bands)
        self.feature = feature

    def __call__(self, pairs, left, right, ctx):
        if self.source is None:
            raise ConfigurationError("grade needs a 'source' feature to grade", config_key="source")
        g = to_grades(pairs[_need(pairs, self.source)], self.bands)
        return pd.DataFrame({feature_column(self.feature): pd.Categorical(g)}, index=pairs.index)


@register("comparator", "match_grade")
class MatchGrade:
    """Concatenate the grades of several features into one string column ``f_match_grade``."""

    def __init__(self, features: Sequence[str] = (), bands: Optional[Dict[str, float]] = None,
                 feature: str = "match_grade"):
        if not features:
            raise ConfigurationError("match_grade needs a non-empty 'features' list", config_key="features")
        self.features = list(features)
        self.bands = _check_bands(bands)
        self.feature = feature

    def __call__(self, pairs, left, right, ctx):
        parts = [to_grades(pairs[_need(pairs, f)], self.bands).astype(str) for f in self.features]
        out = parts[0]
        for p in parts[1:]:
            out = np.char.add(out, p)
        return pd.DataFrame({feature_column(self.feature): out.astype(object)}, index=pairs.index)
