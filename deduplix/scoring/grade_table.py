"""Score from a lookup table of grade patterns."""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from ..compare._util import feature_column
from ..engine.registry import register
from ..exceptions import ConfigurationError


@register("scorer", "grade_table")
class GradeTableScorer:
    """Look up ``f_match_grade`` in a CSV ``grade_pattern, confidence`` (confidence 0-10).

    ``*`` matches any one letter; the first matching row wins; ``score = confidence / 10``.
    Patterns with no matching row get ``default``; a missing grade gives NaN.
    """

    def __init__(self, path: Optional[str] = None, column: str = "match_grade", default: float = 0.0,
                 table: Optional[pd.DataFrame] = None):
        if table is None:
            if path is None:
                raise ConfigurationError("grade_table needs a 'path'", config_key="path")
            table = pd.read_csv(path, dtype={"grade_pattern": str})
        missing = {"grade_pattern", "confidence"} - set(table.columns)
        if missing:
            raise ConfigurationError(f"grade table lacks columns {sorted(missing)}", config_key="path")
        conf = pd.to_numeric(table["confidence"], errors="coerce")
        if conf.isna().any() or (conf < 0).any() or (conf > 10).any():
            raise ConfigurationError("grade table confidence must be numbers in [0, 10]", config_key="path")
        self.path, self.column, self.default = path, column, default
        self.rules = [(re.compile("".join("." if ch == "*" else re.escape(ch) for ch in str(p).strip()),
                                  re.IGNORECASE), float(c) / 10.0)
                      for p, c in zip(table["grade_pattern"], conf)]

    def _lookup(self, pattern: str) -> float:
        for rx, val in self.rules:
            if rx.fullmatch(pattern):
                return val
        return float(self.default)

    def __call__(self, pairs: pd.DataFrame, ctx=None) -> pd.Series:
        col = feature_column(self.column)
        if col not in pairs.columns:
            raise ConfigurationError(f"grade_table: '{col}' is not a column of the pairs table",
                                     config_key="column", config_value=self.column)
        s = pairs[col].astype(object)
        mapping = {u: self._lookup(str(u)) for u in s.dropna().unique()}
        out = s.map(mapping).to_numpy(dtype="float64", na_value=np.nan)
        return pd.Series(out.astype(np.float32), index=pairs.index)
