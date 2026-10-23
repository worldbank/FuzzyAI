"""Weighted average of features."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..compare._util import feature_column
from ..engine.registry import register
from ..exceptions import ConfigurationError


@register("scorer", "weighted")
class WeightedScorer:
    """``score = sum(w * f) / sum(w)`` over the features present for each pair.

    Parameters
    ----------
    weights : dict
        Feature name (``name`` or ``f_name``) to non-negative weight.
    maps : dict, optional
        For categorical features, category -> value, e.g. ``{f_legal_form: {same: 1.0}}``.
        An unmapped category raises ``ConfigurationError``.
    missing : {"renormalize", "zero"}
        Missing features are dropped from the average, or count as 0.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 maps: Optional[Dict[str, Dict[str, float]]] = None, missing: str = "renormalize"):
        if not weights:
            raise ConfigurationError("weighted scorer needs a non-empty 'weights'", config_key="weights")
        if missing not in ("renormalize", "zero"):
            raise ConfigurationError("missing must be 'renormalize' or 'zero'", config_key="missing",
                                     config_value=str(missing))
        self.weights = {feature_column(k): float(v) for k, v in weights.items()}
        if any(w < 0 for w in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ConfigurationError("weights must be non-negative with a positive sum", config_key="weights")
        self.maps = {feature_column(k): dict(v) for k, v in (maps or {}).items()}
        self.missing = missing

    def _values(self, pairs: pd.DataFrame, col: str) -> np.ndarray:
        if col not in pairs.columns:
            raise ConfigurationError(f"weighted scorer: '{col}' is not a column of the pairs table",
                                     config_key="weights", config_value=col)
        s = pairs[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            return s.to_numpy(dtype="float64", na_value=np.nan)
        mp = self.maps.get(col)
        if mp is None:
            raise ConfigurationError(f"Categorical feature '{col}' needs a map in 'maps'",
                                     config_key="maps", config_value=col)
        s = s.astype(object)
        bad = set(s.dropna().unique()) - set(mp)
        if bad:
            raise ConfigurationError(f"Unmapped categories {sorted(map(str, bad))} in feature '{col}'",
                                     config_key="maps", config_value=col)
        return pd.to_numeric(s.map(mp), errors="coerce").to_numpy(dtype="float64", na_value=np.nan)

    def __call__(self, pairs: pd.DataFrame, ctx=None) -> pd.Series:
        num = np.zeros(len(pairs))
        den = np.zeros(len(pairs))
        for col, w in self.weights.items():
            v = self._values(pairs, col)
            present = ~np.isnan(v)
            num += np.where(present, v, 0.0) * w
            den += np.where(present, w, 0.0)
        if self.missing == "zero":
            den = np.full(len(pairs), sum(self.weights.values()))
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(den > 0, num / den, np.nan)
        return pd.Series(out.astype(np.float32), index=pairs.index)
