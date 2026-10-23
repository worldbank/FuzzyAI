"""Scorer backed by any object with ``predict_proba`` (no sklearn import at module level)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..compare._util import feature_column
from ..engine.registry import register
from ..exceptions import ConfigurationError

MAX_CATEGORIES = 250


class FeatureModel:
    """An estimator plus the feature order and categorical vocabularies it was trained with.

    ``predict_proba`` takes a frame of ``f_`` columns; categorical columns are coded with the
    training vocabulary (unseen or missing -> NaN), numeric ones are cast to float.
    """

    def __init__(self, estimator: Any, features: Sequence[str], categories: Optional[Dict[str, List[str]]] = None):
        self.estimator = estimator
        self.features = list(features)
        self.categories = dict(categories or {})
        self.fill_nan = False  # replace NaN by 0 for estimators that cannot handle missing values

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        out = self._transform(X)
        return np.nan_to_num(out, nan=0.0) if self.fill_nan else out

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for f in self.features:
            s = X[f]
            if f in self.categories:
                codes = pd.Categorical(s.astype(object), categories=self.categories[f]).codes.astype("float64")
                codes[codes < 0] = np.nan
                cols.append(codes)
            else:
                cols.append(pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", na_value=np.nan))
        return np.column_stack(cols) if cols else np.empty((len(X), 0))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self.transform(X))


def _load(path: str) -> Any:
    try:
        import joblib

        return joblib.load(path)
    except ImportError:
        import pickle

        with open(path, "rb") as fh:
            return pickle.load(fh)


@register("scorer", "model")
class ModelScorer:
    """``score = estimator.predict_proba(features)[:, 1]``.

    ``estimator`` is a fitted object or a path to a joblib/pickle file. ``features`` (names with or
    without ``f_``) default to the estimator's own ``features`` or ``feature_names_in_``.
    """

    def __init__(self, estimator: Any = None, features: Optional[Sequence[str]] = None):
        if estimator is None:
            raise ConfigurationError("model scorer needs an 'estimator'", config_key="estimator")
        self.estimator_spec = estimator if isinstance(estimator, str) else None
        self.estimator = _load(estimator) if isinstance(estimator, str) else estimator
        if not hasattr(self.estimator, "predict_proba"):
            raise ConfigurationError("estimator must provide predict_proba", config_key="estimator")
        feats = features or getattr(self.estimator, "features", None) \
            or getattr(self.estimator, "feature_names_in_", None)
        if feats is None:
            raise ConfigurationError("model scorer needs 'features'", config_key="features")
        self.features = [feature_column(f) for f in feats]

    def __call__(self, pairs: pd.DataFrame, ctx=None) -> pd.Series:
        missing = [f for f in self.features if f not in pairs.columns]
        if missing:
            raise ConfigurationError(f"model scorer: missing feature columns {missing}", config_key="features")
        X = pairs[self.features]
        if not isinstance(self.estimator, FeatureModel):
            X = X.apply(lambda c: pd.to_numeric(c, errors="coerce"))
        proba = np.asarray(self.estimator.predict_proba(X))
        col = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
        return pd.Series(col.astype(np.float32), index=pairs.index)
