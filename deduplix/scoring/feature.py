"""Scorer that returns one feature unchanged."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..compare._util import feature_column
from ..engine.registry import register
from ..exceptions import ConfigurationError


@register("scorer", "feature")
class FeatureScorer:
    """``score = f_<name>`` (used by the legacy path)."""

    def __init__(self, name: str = "name"):
        self.name = name

    def __call__(self, pairs: pd.DataFrame, ctx=None) -> pd.Series:
        col = feature_column(self.name)
        if col not in pairs.columns:
            raise ConfigurationError(f"feature scorer: '{col}' is not a column of the pairs table",
                                     config_key="name", config_value=self.name)
        s = pd.to_numeric(pairs[col], errors="coerce")
        return pd.Series(s.to_numpy(dtype="float32", na_value=np.nan), index=pairs.index)
