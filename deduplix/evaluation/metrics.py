"""Pair-level, band-level and cluster-level evaluation metrics.

Gold labels are a DataFrame ``id1, id2, label`` (1 = same entity).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_BANDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def __getattr__(name: str) -> Any:  # lazy re-export of the Phase 3 blocking report
    if name == "blocking_report":
        from ..blocking.metrics import blocking_report

        return blocking_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _keys(df: pd.DataFrame) -> pd.Series:
    """Order-independent string key per (id1, id2) pair."""
    a = df["id1"].astype(str).to_numpy()
    b = df["id2"].astype(str).to_numpy()
    swap = a > b
    lo, hi = np.where(swap, b, a), np.where(swap, a, b)
    return pd.Series(lo, index=df.index, dtype=object) + "|" + pd.Series(hi, index=df.index, dtype=object)


def _safe(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _prf(tp: int, n_pred: int, n_gold: int) -> Dict[str, float]:
    p, r = _safe(tp, n_pred), _safe(tp, n_gold)
    return {"precision": p, "recall": r, "f1": _safe(2 * p * r, p + r),
            "tp": int(tp), "fp": int(n_pred - tp), "fn": int(n_gold - tp)}


def _gold_positive_keys(gold: pd.DataFrame) -> set:
    pos = gold[gold["label"] == 1] if "label" in gold.columns else gold
    return set(_keys(pos)) if len(pos) else set()


def pair_metrics(pred: pd.DataFrame, gold: pd.DataFrame) -> Dict[str, float]:
    """Precision, recall and F1 of predicted pairs against gold positives.

    Pairs are unordered and counted once; predicted pairs absent from ``gold`` are
    false positives.
    """
    pk = set(_keys(pred)) if len(pred) else set()
    gk = _gold_positive_keys(gold)
    return _prf(len(pk & gk), len(pk), len(gk))


def metrics_by_band(pairs: pd.DataFrame, gold: pd.DataFrame, bands: Sequence[float] = DEFAULT_BANDS,
                    score_col: str = "score", top: Optional[float] = None) -> pd.DataFrame:
    """Precision per score band ``[lo, hi)``, to set router thresholds.

    The last band is open ended unless ``top`` is given (it then closes at ``top``).
    ``cum_precision`` and ``cum_recall`` are for ``score >= band_low``. Pairs absent
    from gold count as negatives.
    """
    gk = _gold_positive_keys(gold)
    df = pairs[["id1", "id2", score_col]].dropna(subset=[score_col])
    hit = _keys(df).isin(gk).to_numpy()
    s = df[score_col].to_numpy(dtype="float64")
    edges = list(bands) + [np.inf if top is None else top]
    rows = []
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last_closed = top is not None and k == len(edges) - 2
        in_band = (s >= lo) & ((s <= hi) if last_closed else (s < hi))
        above = s >= lo
        n, t = int(in_band.sum()), int(hit[in_band].sum())
        rows.append({"band_low": lo, "band_high": hi, "n_pairs": n, "n_true": t,
                     "precision": _safe(t, n),
                     "cum_precision": _safe(int(hit[above].sum()), int(above.sum())),
                     "cum_recall": _safe(len(set(_keys(df[above & hit]))), len(gk))})
    return pd.DataFrame(rows)


def _as_series(x: Any) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, dict):
        return pd.Series(x)
    return pd.Series(np.asarray(x))


def cluster_metrics(clusters: Any, gold_clusters: Any) -> Dict[str, float]:
    """B-cubed precision/recall/F1 and pairwise precision/recall/F1.

    Both arguments map record id -> cluster label (Series, dict, or equal-length arrays).
    Only records present in both are scored.
    """
    df = pd.concat([_as_series(clusters).rename("pred"), _as_series(gold_clusters).rename("gold")],
                   axis=1, join="inner")
    n = len(df)
    names = ["bcubed_precision", "bcubed_recall", "bcubed_f1", "pair_precision", "pair_recall", "pair_f1"]
    if n == 0:
        return dict.fromkeys(names, 0.0)
    cell_n = df.groupby(["pred", "gold"])["pred"].transform("size")
    psize = df.groupby("pred")["pred"].transform("size")
    gsize = df.groupby("gold")["gold"].transform("size")
    bp = float((cell_n / psize).sum() / n)
    br = float((cell_n / gsize).sum() / n)

    def n_pairs(k: np.ndarray) -> float:
        k = np.asarray(k, dtype="float64")
        return float((k * (k - 1) / 2).sum())

    tp = n_pairs(df.groupby(["pred", "gold"]).size().to_numpy())
    pp = _safe(tp, n_pairs(df["pred"].value_counts().to_numpy()))
    pr = _safe(tp, n_pairs(df["gold"].value_counts().to_numpy()))
    return dict(zip(names, [bp, br, _safe(2 * bp * br, bp + br), pp, pr, _safe(2 * pp * pr, pp + pr)]))
