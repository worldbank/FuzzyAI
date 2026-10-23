"""Split organisation names into base name, legal form and designators.

Output columns (originals are never overwritten; empty values are ``""``)::

    name_norm          default text normalization (created only if absent)
    name_base          aggressive base name used for blocking
    legal_form         canonical code (LP, SABDECV, ...)
    legal_form_raw     the legal form as written ("L.P.")
    designator_nums    sorted, pipe-joined fund numbers ("4" or "1|2")
    designator_series  sorted, pipe-joined series letters ("B")
    vintage_year       4-digit year inside ``year_range``
    designator_quals   sorted, pipe-joined qualifier keys (feeder, gp, management ...)
    geo_qualifier      ISO code of a parenthesised geography "(Mauritius)" -> MU
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..engine.registry import register
from .tables import load_table
from .text import DEFAULT_CHAIN, _CHAIN_STEPS, as_text, clean_for_parsing, clean_scalar, fold_series

OUTPUT_COLUMNS = (
    "name_base", "legal_form", "legal_form_raw", "designator_nums", "designator_series",
    "vintage_year", "designator_quals", "geo_qualifier",
)
_PAREN = re.compile(r"\(([^()]*)\)")
_ROLE_DESIGNATOR = ("num", "year", "series", "qual", "prefix")
_DEFAULT_SCALARS = {"roman_max": 30, "roman_trailing": True, "year_range": [1950, 2035], "series_letters": True}


def _to_roman(n: int) -> str:
    out = ""
    for val, sym in ((10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while n >= val:
            out += sym
            n -= val
    return out


def _merge_designators(base: Dict[str, Any], user: Dict[str, Any], merge: str) -> Dict[str, Any]:
    if merge == "replace":
        res: Dict[str, Any] = {"prefix_words": [], "qualifiers": {}, "geo": {}, **_DEFAULT_SCALARS}
        res.update(copy.deepcopy(user))
        return res
    res = copy.deepcopy(base)
    for key, val in user.items():
        if key == "prefix_words":
            res[key] = list(res.get(key, [])) + list(val)
        elif key in ("qualifiers", "geo"):
            cur = res.setdefault(key, {})
            for k, v in val.items():
                cur[k] = list(cur.get(k, [])) + list(v)
        else:
            res[key] = val
    return res


def _merge_legal_forms(base: List[dict], user: List[dict], merge: str) -> List[dict]:
    return copy.deepcopy(user) if merge == "replace" else copy.deepcopy(base) + copy.deepcopy(user)


class _Compiled:
    """Pre-compiled lookups for designators."""

    def __init__(self, d: Dict[str, Any]) -> None:
        self.prefix_words = {clean_scalar(w) for w in d.get("prefix_words", [])}
        self.roman = {_to_roman(i): i for i in range(1, int(d.get("roman_max", 30)) + 1)}
        self.roman_trailing = bool(d.get("roman_trailing", True))
        lo, hi = d.get("year_range", [1950, 2035])
        self.year_lo, self.year_hi = int(lo), int(hi)
        self.series_letters = bool(d.get("series_letters", True))
        quals: Dict[str, List[Tuple[str, ...]]] = {}
        for key, phrases in d.get("qualifiers", {}).items():
            for ph in phrases:
                toks = tuple(clean_scalar(ph).split())
                if toks:
                    quals.setdefault(toks[0], []).append((toks, key))
                joined = "".join(toks)  # "co-invest" also written "coinvest"
                if len(toks) > 1 and joined:
                    quals.setdefault(joined, []).append(((joined,), key))
        for lst in quals.values():
            lst.sort(key=lambda x: len(x[0]), reverse=True)
        self.quals = quals
        self.geo: Dict[str, str] = {}
        for code, names in d.get("geo", {}).items():
            for nm in names:
                self.geo[clean_scalar(nm)] = str(code).upper()

    def parse_tokens(self, toks: List[str]) -> Tuple[str, str, str, str, str]:
        """Return (base, nums, series, year, quals) for a token list."""
        n = len(toks)
        if n == 0:
            return "", "", "", "", ""
        roles: List[Optional[str]] = [None] * n
        nums: List[int] = []
        series: List[str] = []
        years: List[int] = []
        quals: set = set()
        # 4. qualifier phrases (multi-word first)
        i = 0
        while i < n:
            hit = False
            for phrase, key in self.quals.get(toks[i], ()):
                ln = len(phrase)
                if tuple(toks[i:i + ln]) == phrase:
                    for j in range(i, i + ln):
                        roles[j] = "qual"
                    quals.add(key)
                    i += ln
                    hit = True
                    break
            if not hit:
                i += 1
        for j, t in enumerate(toks):
            if roles[j] is not None:
                continue
            prev_prefix = j > 0 and toks[j - 1] in self.prefix_words
            if t.isdigit() and len(t) <= 4:  # 1. digits
                v = int(t)
                if len(t) == 4 and self.year_lo <= v <= self.year_hi:
                    years.append(v)
                    roles[j] = "year"
                else:
                    nums.append(v)
                    roles[j] = "num"
            elif t in self.roman and (prev_prefix or (self.roman_trailing and j == n - 1)):  # 2. roman
                nums.append(self.roman[t])
                roles[j] = "num"
            elif self.series_letters and prev_prefix and len(t) == 1 and t.isalpha():  # 3. series letter
                series.append(t.upper())
                roles[j] = "series"
            elif t in self.prefix_words:
                roles[j] = "prefix"
        end = n
        while end > 1 and roles[end - 1] in _ROLE_DESIGNATOR:
            end -= 1
        base = " ".join(toks[:end])
        return (
            base,
            "|".join(str(x) for x in sorted(set(nums))),
            "|".join(sorted(set(series))),
            str(years[0]) if years else "",
            "|".join(sorted(quals)),
        )


class _LegalForms:
    """Legal-form regexes, built per country on demand."""

    def __init__(self, forms: List[dict]) -> None:
        self.forms = []
        for f in forms:
            variants = sorted({clean_scalar(v) for v in f.get("variants", []) if clean_scalar(v)})
            self.forms.append({
                "code": str(f["code"]), "variants": variants,
                "countries": {str(c).upper() for c in f.get("countries", [])},
                "position": f.get("position", "suffix"),
            })
        self._cache: Dict[Optional[str], Tuple[Any, Any, Dict[str, str]]] = {}

    def for_country(self, country: Optional[str]):
        if country in self._cache:
            return self._cache[country]
        suffix_v: Dict[str, str] = {}
        prefix_v: Dict[str, str] = {}
        for f in self.forms:
            if country is not None and f["countries"] and country not in f["countries"]:
                continue
            for v in f["variants"]:
                if f["position"] in ("suffix", "any"):
                    suffix_v.setdefault(v, f["code"])
                if f["position"] in ("prefix", "any"):
                    prefix_v.setdefault(v, f["code"])

        def rx(vs: Dict[str, str], tmpl: str):
            if not vs:
                return None
            alts = "|".join(re.escape(v) for v in sorted(vs, key=lambda x: (-len(x), x)))
            return re.compile(tmpl.format(alts))

        res = (rx(suffix_v, r"\s({})$"), rx(prefix_v, r"^({})\s"), {**suffix_v, **prefix_v})
        self._cache[country] = res
        return res


def _raw_span(original: str, k: int, suffix: bool) -> str:
    """The original text of the legal form covering ``k`` cleaned tokens."""
    chunks = original.split()
    if not suffix:
        chunks = chunks[::-1]
    got, taken = 0, []
    for ch in reversed(chunks) if suffix else reversed(chunks):
        taken.append(ch)
        got += len(clean_scalar(ch).split())
        if got >= k:
            break
    taken = taken[::-1] if suffix else taken
    return " ".join(taken).strip(" ,;")


@register("normalizer", "name_parser")
class NameParser:
    """Parse names into base name, legal form, designators and qualifiers.

    Parameters
    ----------
    columns, output
        Source column (default ``ctx.columns.name``). ``output`` is the normalized-name column
        created if absent (default ``name_norm``).
    tables
        Optional ``{"legal_forms": [...], "designators": {...}}`` (values may be paths).
    merge
        ``"extend"`` (default) adds user tables to the packaged ones; ``"replace"`` uses only
        the user's tables for the keys given.
    country_column
        Column with ISO-2 country codes; limits legal forms to those valid in that country.
        Missing country matches any form.
    """

    def __init__(self, columns: Optional[str] = None, output: Optional[str] = None,
                 tables: Optional[Dict[str, Any]] = None, merge: str = "extend",
                 country_column: Optional[str] = None) -> None:
        if merge not in ("extend", "replace"):
            raise ValueError("merge must be 'extend' or 'replace'")
        self.columns = columns
        self.output = output
        self.tables = tables
        self.merge = merge
        self.country_column = country_column
        tables = tables or {}
        unknown = set(tables) - {"legal_forms", "designators"}
        if unknown:
            raise ValueError(f"Unknown table keys {sorted(unknown)}; use legal_forms / designators")
        forms = load_table("legal_forms")
        desig = load_table("designators")
        if "legal_forms" in tables:
            forms = _merge_legal_forms(forms, load_table(tables["legal_forms"]), merge)
        if "designators" in tables:
            desig = _merge_designators(desig, load_table(tables["designators"]), merge)
        self._forms = _LegalForms(forms)
        self._desig = _Compiled(desig)

    # -- legal forms (vectorised per country group) -------------------------
    def _extract_legal_forms(self, work: pd.Series, original: pd.Series, country: Optional[pd.Series]):
        n = len(work)
        code = np.full(n, "", dtype=object)
        raw = np.full(n, "", dtype=object)
        work = work.copy()
        if country is None:
            groups = {None: np.arange(n)}
        else:
            c = country.astype(object).where(country.notna(), None)
            c = c.map(lambda x: str(x).strip().upper() or None if x is not None else None)
            groups = {k: np.flatnonzero((c == k).to_numpy()) if k is not None else np.flatnonzero(c.isna().to_numpy())
                      for k in c.dropna().unique().tolist() + [None]}
        for ctry, pos in groups.items():
            if len(pos) == 0:
                continue
            suf, pre, vmap = self._forms.for_country(ctry)
            sub = work.iloc[pos]
            for pattern, is_suffix in ((suf, True), (pre, False)):
                if pattern is None:
                    continue
                todo = sub.index[(code[pos] == "")]
                if len(todo) == 0:
                    break
                m = sub.loc[todo].str.extract(pattern)[0]
                hit = m.notna()
                if not hit.any():
                    continue
                hit_idx = hit.index[hit.to_numpy()]
                variants = m[hit]
                sub.loc[hit_idx] = sub.loc[hit_idx].str.replace(pattern, "", n=1, regex=True)
                sub.loc[hit_idx] = sub.loc[hit_idx].str.strip()
                for ix, v in zip(hit_idx, variants):
                    code[ix] = vmap[v]
                    raw[ix] = _raw_span(original.iloc[ix], len(v.split()), is_suffix)
            work.iloc[pos] = sub.to_numpy()
        return work, code, raw

    def __call__(self, records: pd.DataFrame, ctx: Any) -> pd.DataFrame:
        out = records.copy()
        src = self.columns or (ctx.columns.name if ctx.columns.name in records.columns else ctx.columns.rname)
        if src not in records.columns:
            raise KeyError(f"Column '{src}' not found in records")
        norm_col = self.output or "name_norm"
        original = as_text(records[src]).reset_index(drop=True)
        missing = original.isna()
        text = original.fillna("")

        if norm_col not in out.columns:
            s = original
            for step in DEFAULT_CHAIN:
                s = _CHAIN_STEPS[step]().transform(s)
            out[norm_col] = s.to_numpy()

        # geography in parentheses
        folded = fold_series(text).str.lower()
        paren = folded.str.extract(_PAREN)[0]
        geo = paren.where(paren.notna(), None)
        geo_code = geo.map(lambda x: self._desig.geo.get(clean_scalar(x)) if x is not None else None)
        has_geo = geo_code.notna()
        folded = folded.copy()
        if has_geo.any():
            folded[has_geo] = folded[has_geo].str.replace(_PAREN, " ", n=1, regex=True)
        work = clean_for_parsing(folded)

        country = None
        if self.country_column:
            if self.country_column not in records.columns:
                raise KeyError(f"country_column '{self.country_column}' not found in records")
            country = records[self.country_column].reset_index(drop=True)

        work, code, raw = self._extract_legal_forms(work, text, country)

        parsed = [self._desig.parse_tokens(t) for t in work.str.split().tolist()]
        base, nums, series, year, quals = (list(x) for x in zip(*parsed)) if parsed else ([], [], [], [], [])
        base_s = pd.Series(base, dtype=object)
        base_s[missing.to_numpy()] = None
        idx = records.index
        out["name_base"] = pd.Series(base_s.to_numpy(), index=idx)
        out["legal_form"] = pd.Series(code, index=idx)
        out["legal_form_raw"] = pd.Series(raw, index=idx)
        out["designator_nums"] = pd.Series(nums, index=idx, dtype=object)
        out["designator_series"] = pd.Series(series, index=idx, dtype=object)
        out["vintage_year"] = pd.Series(year, index=idx, dtype=object)
        out["designator_quals"] = pd.Series(quals, index=idx, dtype=object)
        out["geo_qualifier"] = pd.Series(geo_code.fillna("").to_numpy(), index=idx, dtype=object)
        return out


def parse_names(names: Any, **kwargs: Any) -> pd.DataFrame:
    """Convenience: parse an iterable of names and return the output columns (no context needed)."""
    from ..engine.context import ColumnMap

    class _Ctx:
        columns = ColumnMap()

    df = pd.DataFrame({"name": list(names)})
    return NameParser(**kwargs)(df, _Ctx())
