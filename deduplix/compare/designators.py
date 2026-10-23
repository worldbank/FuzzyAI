"""Designator check: fund numbers, vintage years, series and qualifiers from the Phase 2 columns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.registry import register
from ..engine.schema import I1, I2
from ..exceptions import ConfigurationError
from ._util import feature_column, token_strings

DESIGNATOR_CATEGORIES = ["ok", "review", "reject"]
_COLUMNS = ("designator_nums", "vintage_year", "designator_series", "designator_quals")


@register("comparator", "designator_check")
class DesignatorCheck:
    """Category ok/review/reject from the name_parser designator columns.

    Reject: both sides have numbers, years or series and they differ. Review: qualifiers
    differ, or numbers on one side only. Otherwise ok. Absent columns count as empty.
    """

    def __init__(self, feature: str = "designator_check"):
        self.feature = feature

    def __call__(self, pairs, left, right, ctx):
        if not any(c in left.columns for c in _COLUMNS):
            raise ConfigurationError(
                "designator_check needs the name_parser columns " + ", ".join(_COLUMNS)
                + "; add the 'name_parser' normalizer", config_key="designator_check")
        i1, i2 = pairs[I1].to_numpy(), pairs[I2].to_numpy()
        g = {c: (token_strings(left, c)[i1], token_strings(right, c)[i2]) for c in _COLUMNS}

        def diff(a, b):
            return (a != "") & (b != "") & (a != b)

        reject = diff(*g["designator_nums"]) | diff(*g["vintage_year"]) | diff(*g["designator_series"])
        qa, qb = g["designator_quals"]
        na, nb = g["designator_nums"]
        review = (qa != qb) | ((na != "") ^ (nb != ""))
        codes = np.where(reject, 2, np.where(review, 1, 0)).astype(np.int8)
        cat = pd.Categorical.from_codes(codes, categories=DESIGNATOR_CATEGORIES)
        return pd.DataFrame({feature_column(self.feature): cat}, index=pairs.index)
