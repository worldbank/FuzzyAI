"""Field comparators: exact, numeric, country, legal form and token IDF overlap."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..engine.registry import register
from ..engine.schema import I1, I2
from ._util import feature_column, pair_text, resolve_column, text_values


class _FieldComparator:
    """Base: ``column`` is the default for both ``left`` and ``right``."""

    feature = ""

    def _cols(self, left, right, ctx):
        lcol = resolve_column(left, self.left or self.column, ctx, "left")
        rcol = resolve_column(right, self.right or self.left or self.column, ctx, "right")
        return lcol, rcol

    def _frame(self, pairs, values) -> pd.DataFrame:
        return pd.DataFrame({feature_column(self.feature): np.asarray(values, dtype=np.float32)},
                            index=pairs.index)


@register("comparator", "exact")
class Exact(_FieldComparator):
    """1 when both values are equal, 0 when different, NaN when either is null or blank."""

    def __init__(self, column: Optional[str] = None, left: Optional[str] = None,
                 right: Optional[str] = None, feature: str = "exact"):
        self.column, self.left, self.right, self.feature = column, left, right, feature

    def __call__(self, pairs, left, right, ctx):
        lcol, rcol = self._cols(left, right, ctx)
        a, b, ok = pair_text(pairs, left, right, lcol, rcol)
        out = np.full(len(pairs), np.nan, dtype=np.float32)
        out[ok] = (a[ok] == b[ok]).astype(np.float32)
        return self._frame(pairs, out)


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


@register("comparator", "country_match")
class CountryMatch(_FieldComparator):
    """1 for the same country (case-insensitive), 0 for different, NaN when missing.

    With ``lat`` and ``lon`` columns it also emits ``f_<feature>_km`` (great-circle distance).
    """

    def __init__(self, column: str = "country", left: Optional[str] = None, right: Optional[str] = None,
                 lat: Optional[str] = None, lon: Optional[str] = None, feature: str = "country_match"):
        self.column, self.left, self.right = column, left, right
        self.lat, self.lon, self.feature = lat, lon, feature

    def __call__(self, pairs, left, right, ctx):
        lcol, rcol = self._cols(left, right, ctx)
        a, b, ok = pair_text(pairs, left, right, lcol, rcol)
        out = np.full(len(pairs), np.nan, dtype=np.float32)
        out[ok] = (np.char.upper(a[ok].astype(str)) == np.char.upper(b[ok].astype(str))).astype(np.float32)
        res = self._frame(pairs, out)
        if self.lat and self.lon:
            i1, i2 = pairs[I1].to_numpy(), pairs[I2].to_numpy()

            def g(df, c, idx):
                return pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")[idx]

            km = _haversine_km(g(left, self.lat, i1), g(left, self.lon, i1),
                               g(right, self.lat, i2), g(right, self.lon, i2))
            res[feature_column(self.feature) + "_km"] = km.astype(np.float32)
        return res


@register("comparator", "numeric_rel_diff")
class NumericRelDiff(_FieldComparator):
    """``|a - b| / max(|a|, |b|)``; NaN when either is null or both are 0."""

    def __init__(self, column: Optional[str] = None, left: Optional[str] = None,
                 right: Optional[str] = None, feature: str = "numeric_rel_diff"):
        self.column, self.left, self.right, self.feature = column, left, right, feature

    def __call__(self, pairs, left, right, ctx):
        lcol, rcol = self._cols(left, right, ctx)
        i1, i2 = pairs[I1].to_numpy(), pairs[I2].to_numpy()
        a = pd.to_numeric(left[lcol], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)[i1]
        b = pd.to_numeric(right[rcol], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)[i2]
        den = np.maximum(np.abs(a), np.abs(b))
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.abs(a - b) / den
        out[~np.isfinite(out)] = np.nan
        return self._frame(pairs, out)


LEGAL_FORM_CATEGORIES = ["same", "equivalent", "different", "missing"]


@register("comparator", "legal_form_compat")
class LegalFormCompat:
    """Category same, equivalent (same group in ``equivalent``), different or missing."""

    def __init__(self, equivalent: Sequence[Sequence[str]] = (("LTD", "LIMITED"),),
                 column: str = "legal_form", left: Optional[str] = None, right: Optional[str] = None,
                 feature: str = "legal_form_compat"):
        self.equivalent = [list(g) for g in equivalent]
        self.column, self.left, self.right, self.feature = column, left, right, feature
        self._group = {str(c).upper(): gi for gi, g in enumerate(self.equivalent) for c in g}

    def __call__(self, pairs, left, right, ctx):
        lcol = resolve_column(left, self.left or self.column, ctx, "left")
        rcol = resolve_column(right, self.right or self.left or self.column, ctx, "right")
        a, b, ok = pair_text(pairs, left, right, lcol, rcol)
        a = np.char.upper(a.astype(str))
        b = np.char.upper(b.astype(str))
        same = ok & (a == b)
        ga = pd.Series(a).map(self._group).to_numpy(dtype="float64", na_value=np.nan)
        gb = pd.Series(b).map(self._group).to_numpy(dtype="float64", na_value=np.nan)
        equiv = ok & ~same & (ga == gb)  # NaN == NaN is False
        codes = np.full(len(pairs), 3, dtype=np.int8)
        codes[ok] = 2
        codes[equiv] = 1
        codes[same] = 0
        cat = pd.Categorical.from_codes(codes, categories=LEGAL_FORM_CATEGORIES)
        return pd.DataFrame({feature_column(self.feature): cat}, index=pairs.index)


_TOKEN = re.compile(r"\w+", re.UNICODE)


@register("comparator", "token_idf_overlap")
class TokenIdfOverlap:
    """IDF-weighted Jaccard of the token sets, in [0, 1]; NaN when a side is blank.

    Reuses ``ctx.artifacts['token_index.idf']`` (token -> idf) when a token-index blocker stored
    it; otherwise IDF is computed from the records and cached in ``ctx.artifacts``.
    """

    def __init__(self, left: Optional[str] = None, right: Optional[str] = None,
                 feature: str = "token_idf_overlap"):
        self.left, self.right, self.feature = left, right, feature

    @staticmethod
    def _tokens(df: pd.DataFrame, col: str) -> List[Optional[frozenset]]:
        vals, ok = text_values(df, col)
        return [frozenset(_TOKEN.findall(v.lower())) if o else None for v, o in zip(vals, ok)]

    def _idf(self, ctx, tl, tr, same) -> Dict[str, float]:
        idf = ctx.artifacts.get("token_index.idf")
        if isinstance(idf, dict) and idf:
            return idf
        key = f"token_idf_overlap.idf.{self.left}.{self.right}"
        if key not in ctx.artifacts:
            docs = [t for t in (tl if same else tl + tr) if t is not None]
            n = max(len(docs), 1)
            cnt: Dict[str, int] = {}
            for t in docs:
                for tok in t:
                    cnt[tok] = cnt.get(tok, 0) + 1
            ctx.artifacts[key] = {tok: float(np.log((n + 1) / (c + 1)) + 1.0) for tok, c in cnt.items()}
        return ctx.artifacts[key]

    def __call__(self, pairs, left, right, ctx):
        lcol = resolve_column(left, self.left, ctx, "left")
        rcol = resolve_column(right, self.right or self.left, ctx, "right")
        same = right is left and lcol == rcol
        tl = self._tokens(left, lcol)
        tr = tl if same else self._tokens(right, rcol)
        idf = self._idf(ctx, tl, tr, same)
        default = max(idf.values()) if idf else 1.0

        def w(toks):
            return sum(idf.get(t, default) for t in toks)

        out = np.full(len(pairs), np.nan, dtype=np.float32)
        for k, (i, j) in enumerate(zip(pairs[I1].to_numpy().tolist(), pairs[I2].to_numpy().tolist())):
            a, b = tl[i], tr[j]
            if a is None or b is None:
                continue
            den = w(a | b)
            out[k] = w(a & b) / den if den > 0 else np.nan
        return pd.DataFrame({feature_column(self.feature): out}, index=pairs.index)
