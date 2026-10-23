"""Shared helpers for comparators: column resolution, gathering and missing-value masks."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..engine.schema import FEATURE_PREFIX, I1, I2
from ..exceptions import ConfigurationError

DERIVED_NAME_COLUMNS = ("name_norm", "name_base")


def feature_column(feature: str) -> str:
    """``f_<feature>`` (idempotent)."""
    return feature if feature.startswith(FEATURE_PREFIX) else FEATURE_PREFIX + feature


def resolve_column(df: pd.DataFrame, col: Optional[str], ctx, side: str = "left") -> str:
    """Pick the column to read.

    ``None`` or a derived name column (``name_norm``/``name_base``) that is absent falls back to
    ``name_norm`` and then to the user's name column; any other missing column is an error.
    """
    if col is not None and col in df.columns:
        return col
    if col is None or col in DERIVED_NAME_COLUMNS:
        chain = ["name_norm", ctx.columns.name if side == "left" else ctx.columns.rname]
        for c in chain:
            if c in df.columns:
                return c
    raise ConfigurationError(f"Column '{col}' not found in the {side} table", config_key="column",
                             config_value=str(col))


def text_values(df: pd.DataFrame, col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Object array of strings and a validity mask (False for null / blank), one entry per record."""
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        valid = s.notna().to_numpy()
        vals = np.where(valid, s.astype(object).astype(str), "").astype(object)
        return vals, valid
    s = s.astype(object)
    notna = s.notna()
    vals = s.where(notna, "").astype(str).str.strip()
    valid = (notna & (vals != "")).to_numpy()
    return vals.to_numpy(dtype=object), valid


def pair_text(pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame, lcol: str, rcol: str):
    """Gathered left/right strings for each pair and the mask where both are present."""
    lv, lok = text_values(left, lcol)
    rv, rok = text_values(right, rcol)
    i1, i2 = pairs[I1].to_numpy(), pairs[I2].to_numpy()
    return lv[i1], rv[i2], lok[i1] & rok[i2]


def token_strings(df: pd.DataFrame, col: str) -> np.ndarray:
    """Per-record strings with "" for missing; whole numbers print without a decimal point."""
    if col not in df.columns:
        return np.full(len(df), "", dtype=object)
    vals, _ = text_values(df, col)
    return vals


def sides(left_spec: Optional[str], right_spec: Optional[str], column: Optional[str]):
    lcol = left_spec or column
    rcol = right_spec or lcol
    return lcol, rcol
