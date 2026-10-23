"""Pairs-table contract: column names, validation and legacy conversion."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..exceptions import ConfigurationError

I1 = "i1"
I2 = "i2"
BLK_PREFIX = "blk_"
BLK_SCORE_PREFIX = "blk_score_"
FEATURE_PREFIX = "f_"
SCORE = "score"
ROUTE = "route"
DECISION = "decision"
DECISION_PROB = "decision_prob"
DECISION_REASON = "decision_reason"
DECIDED_BY = "decided_by"
CLUSTER_ID = "cluster_id"

ACCEPT, REVIEW, REJECT = "accept", "review", "reject"
ROUTES = (ACCEPT, REVIEW, REJECT)
MATCH, NON_MATCH, UNVALIDATED = "match", "non_match", "unvalidated"
DECISIONS = (MATCH, NON_MATCH, UNVALIDATED)

LEGACY_PAIR_COLUMNS = ["id1", "id2", "name1", "name2", "similarity_score"]


def index_dtype(n_left: int, n_right: int):
    """int32 when both tables are under 2**31 rows, else int64."""
    return np.int32 if max(n_left, n_right) < 2**31 else np.int64


def empty_pairs(n_left: int = 0, n_right: int = 0) -> pd.DataFrame:
    dt = index_dtype(n_left, n_right)
    return pd.DataFrame({I1: np.array([], dtype=dt), I2: np.array([], dtype=dt)})


def validate_pairs(df: pd.DataFrame, stage: str) -> None:
    """Check structural invariants of the pairs table after ``stage`` (debug mode)."""
    if df.columns.duplicated().any():
        dup = list(df.columns[df.columns.duplicated()])
        raise ConfigurationError(f"Stage '{stage}' produced duplicate columns: {dup}")
    for col in (I1, I2):
        if col not in df.columns:
            raise ConfigurationError(f"Stage '{stage}': pairs table lacks column '{col}'")
        if not pd.api.types.is_integer_dtype(df[col]):
            raise ConfigurationError(f"Stage '{stage}': '{col}' must be integer, got {df[col].dtype}")
    if SCORE in df.columns and len(df):
        s = df[SCORE].dropna()
        if len(s) and (s.min() < -1e-6 or s.max() > 1 + 1e-6):
            raise ConfigurationError(f"Stage '{stage}': '{SCORE}' must lie in [0, 1]")
    if ROUTE in df.columns:
        bad = set(df[ROUTE].dropna().astype(str)) - set(ROUTES)
        if bad:
            raise ConfigurationError(f"Stage '{stage}': invalid route values {sorted(bad)}")


def to_legacy_pairs(
    pairs: pd.DataFrame,
    left: pd.DataFrame,
    right: pd.DataFrame,
    id_col: str,
    name_col: str,
    right_id_col: Optional[str] = None,
    right_name_col: Optional[str] = None,
    legacy_score_feature: Optional[str] = None,
) -> pd.DataFrame:
    """Build ``id1, id2, name1, name2, similarity_score`` (0-100) from the pairs table."""
    right_id_col = right_id_col or id_col
    right_name_col = right_name_col or name_col
    i1, i2 = pairs[I1].to_numpy(), pairs[I2].to_numpy()
    if legacy_score_feature:
        col = (legacy_score_feature if legacy_score_feature.startswith(FEATURE_PREFIX)
               else FEATURE_PREFIX + legacy_score_feature)
        if col not in pairs.columns:
            raise ConfigurationError(f"legacy_score_feature '{col}' is not a column of the pairs table")
        sim = pairs[col].astype("float64").to_numpy() * 100.0
    else:
        sim = pairs[SCORE].astype("float64").to_numpy() * 100.0
    return pd.DataFrame(
        {
            "id1": left[id_col].to_numpy()[i1],
            "id2": right[right_id_col].to_numpy()[i2],
            "name1": left[name_col].to_numpy()[i1],
            "name2": right[right_name_col].to_numpy()[i2],
            "similarity_score": sim,
        }
    )


def to_legacy_cross(
    pairs: pd.DataFrame,
    left: pd.DataFrame,
    right: pd.DataFrame,
    id_col: str,
    name_col: str,
    right_id_col: str,
    right_name_col: str,
    legacy_score_feature: Optional[str] = None,
) -> pd.DataFrame:
    """Legacy ``cross_matches``: df1_id, df1_name, df2_id, df2_name, similarity_score, validation_reason."""
    base = to_legacy_pairs(pairs, left, right, id_col, name_col, right_id_col, right_name_col,
                           legacy_score_feature)
    if DECISION_REASON in pairs.columns:
        reason = pairs[DECISION_REASON].fillna("").to_numpy()
    else:
        reason = np.full(len(pairs), "", dtype=object)
    out = pd.DataFrame(
        {
            "df1_id": base["id1"], "df1_name": base["name1"],
            "df2_id": base["id2"], "df2_name": base["name2"],
            "similarity_score": base["similarity_score"],
            "validation_reason": reason,
        }
    )
    for col in ("matched_column1", "matched_column2"):
        if col in pairs.columns:
            out[col] = pairs[col].to_numpy()
    return out
