"""Train a calibrated classifier scorer from labelled pairs (needs scikit-learn)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..compare._util import feature_column
from ..exceptions import ConfigurationError
from ..scoring.model import MAX_CATEGORIES, FeatureModel

logger = logging.getLogger("deduplix.evaluation.train")

DEFAULT_BANDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
_LABEL_WORDS = {"match": 1, "non_match": 0, "true": 1, "false": 0, "yes": 1, "no": 0}


def _labels_to_array(pairs: pd.DataFrame, labels: Any) -> np.ndarray:
    """Float array aligned to ``pairs`` (NaN = unlabelled). Accepts a Series, array or a frame
    with ``i1, i2, label``."""
    if isinstance(labels, pd.DataFrame):
        col = "label" if "label" in labels.columns else "decision"
        key = pd.MultiIndex.from_arrays([labels["i1"], labels["i2"]])
        ser = pd.Series(labels[col].to_numpy(), index=key)
        ser = ser[~ser.index.duplicated()]
        raw = ser.reindex(pd.MultiIndex.from_arrays([pairs["i1"], pairs["i2"]])).to_numpy()
    elif isinstance(labels, pd.Series):
        if labels.index.equals(pairs.index) or len(labels) == len(pairs):
            raw = labels.to_numpy()
        else:
            raw = labels.reindex(pairs.index).to_numpy()
    else:
        raw = np.asarray(labels)
        if len(raw) != len(pairs):
            raise ConfigurationError("labels must have one entry per pair", config_key="labels")
    s = pd.Series(raw, dtype=object)
    s = s.map(lambda x: np.nan if pd.isna(x) else _LABEL_WORDS.get(str(x).lower(), x))
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)


def _band_table(y: np.ndarray, p: np.ndarray, bands: Sequence[float]) -> pd.DataFrame:
    rows = []
    edges = list(bands)
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = k == len(edges) - 2
        m = (p >= lo) & ((p <= hi) if last else (p < hi))
        n = int(m.sum())
        rows.append({"band": f"[{lo:.2f}, {hi:.2f}{']' if last else ')'}", "lo": lo, "hi": hi, "n": n,
                     "n_match": int(y[m].sum()) if n else 0,
                     "precision": float(y[m].mean()) if n else np.nan,
                     "mean_score": float(p[m].mean()) if n else np.nan})
    return pd.DataFrame(rows)


def train_scorer(pairs: pd.DataFrame, labels: Any, features: Sequence[str], model: Any = "hist_gbdt",
                 calibrate: Optional[str] = "isotonic", cv: int = 5, random_state: int = 0,
                 bands: Sequence[float] = DEFAULT_BANDS) -> Tuple[FeatureModel, pd.DataFrame]:
    """Fit a (calibrated) classifier on ``f_`` features of labelled pairs.

    Parameters
    ----------
    pairs : DataFrame
        Pairs table with the ``features`` columns (numeric or categorical; NA allowed).
    labels : Series, array or DataFrame
        1/0 (or ``match``/``non_match``) per pair, aligned to ``pairs``; or a frame with
        ``i1, i2, label``. Unlabelled (NaN / unvalidated) pairs are dropped.
    features : list of str
        Feature names, with or without ``f_``.
    model : {"hist_gbdt", "logistic"} or sklearn estimator
    calibrate : {"isotonic", "sigmoid", None}
    cv : int
        Folds for calibration and for the out-of-fold metrics (reduced when classes are small).

    Returns
    -------
    (FeatureModel, DataFrame)
        The fitted model (has ``predict_proba(frame_of_f_columns)``, usable by the ``model``
        scorer) and a table of precision per out-of-fold score band.
    """
    try:
        from sklearn.base import clone
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
    except ImportError as exc:  # pragma: no cover
        raise ImportError("train_scorer needs scikit-learn (pip install deduplix[eval])") from exc

    feats = [feature_column(f) for f in features]
    missing = [f for f in feats if f not in pairs.columns]
    if missing:
        raise ConfigurationError(f"train_scorer: missing feature columns {missing}", config_key="features")
    y_all = _labels_to_array(pairs, labels)
    keep = ~np.isnan(y_all)
    if keep.sum() == 0:
        raise ConfigurationError("train_scorer: no labelled pairs", config_key="labels")
    y = y_all[keep].astype(int)
    if set(np.unique(y)) - {0, 1} or len(np.unique(y)) < 2:
        raise ConfigurationError("train_scorer needs both classes (0 and 1) among the labels",
                                 config_key="labels")
    X = pairs.loc[:, feats].iloc[np.flatnonzero(keep)]

    categories: Dict[str, List[str]] = {}
    for f in feats:
        s = X[f]
        if not (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)):
            cats = sorted(s.dropna().astype(str).unique())
            if len(cats) > MAX_CATEGORIES:
                raise ConfigurationError(f"feature '{f}' has more than {MAX_CATEGORIES} categories")
            categories[f] = cats
    fm = FeatureModel(None, feats, categories)
    Xn = fm.transform(X)
    cat_mask = np.array([f in categories for f in feats])

    if isinstance(model, str):
        if model == "hist_gbdt":
            base = HistGradientBoostingClassifier(random_state=random_state,
                                                  categorical_features=cat_mask if cat_mask.any() else None)
        elif model == "logistic":
            base = LogisticRegression(max_iter=1000)
            fm.fill_nan = True
            Xn = fm.transform(X)
        else:
            raise ConfigurationError(f"Unknown model '{model}'", config_key="model", config_value=model)
    else:
        base = clone(model)

    counts = np.bincount(y)
    folds = int(min(cv, counts.min()))
    est: Any
    if calibrate and folds >= 2:
        if calibrate not in ("isotonic", "sigmoid"):
            raise ConfigurationError("calibrate must be 'isotonic', 'sigmoid' or None", config_key="calibrate")
        est = CalibratedClassifierCV(base, method=calibrate, cv=folds)
    else:
        if calibrate:
            logger.warning("too few examples of one class for calibration; fitting uncalibrated")
        est = base
    if folds >= 2:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        oof = cross_val_predict(clone(est), Xn, y, cv=splitter, method="predict_proba")[:, 1]
    else:
        logger.warning("too few examples for cross-validated metrics; using in-sample predictions")
        oof = clone(est).fit(Xn, y).predict_proba(Xn)[:, 1]
    est.fit(Xn, y)
    fm.estimator = est
    return fm, _band_table(y.astype(float), oof, bands)
