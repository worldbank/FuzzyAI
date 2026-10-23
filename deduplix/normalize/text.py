"""Text normalizers: small, composable, vectorised string cleaners.

Every normalizer is a class registered under ``normalizer``; it is built with ``columns`` and
``output`` and called ``(records, ctx) -> DataFrame``. It returns a copy of ``records`` with the
new column(s) added and never overwrites an input column.

Column resolution
-----------------
* ``columns=None`` reads the name column of the context (``ctx.columns.name``, or
  ``ctx.columns.rname`` when the frame only has that one). If the ``output`` column already
  exists (an earlier normalizer in the pipeline wrote it) it is read instead, so
  ``[lowercase, remove_punctuation]`` chains.
* ``output=None`` writes ``name_norm`` for a single column, ``<column>_norm`` for several.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pandas as pd

from ..engine.registry import register
from .tables import load_table

DEFAULT_OUTPUT = "name_norm"
_WS = re.compile(r"\s+")
_DOT_INITIAL = re.compile(r"(?<!\w)(\w)\.")  # "s.a." -> "sa", "l.p." -> "lp"
_PUNCT = re.compile(r"[^\w\s]")

ColumnsArg = Union[None, str, Sequence[str]]


# ---------------------------------------------------------------------------
# Series-level primitives (also used by name_parser)
# ---------------------------------------------------------------------------
def as_text(s: pd.Series) -> pd.Series:
    """Object series of str with missing values kept missing."""
    s = s.astype(object)
    mask = s.notna()
    if mask.all():
        return s.astype(str).astype(object)
    out = s.copy()
    out[mask] = s[mask].astype(str)
    return out


def collapse_ws(s: pd.Series) -> pd.Series:
    return s.str.replace(_WS, " ", regex=True).str.strip()


def fold_accents_str(x: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", x) if not unicodedata.combining(c))


def fold_series(s: pd.Series) -> pd.Series:
    """Drop combining marks after NFKD; only non-ASCII rows are touched."""
    s = s.copy()
    mask = s.notna() & ~s.fillna("").str.isascii()
    if mask.any():
        s[mask] = s[mask].map(fold_accents_str)
    return s


def clean_for_parsing(s: pd.Series) -> pd.Series:
    """Fold, lower-case, ``&`` -> and, initials' dots removed, punctuation -> space."""
    s = fold_series(s).str.lower().str.replace("&", " and ", regex=False)
    s = s.str.replace(_DOT_INITIAL, r"\1", regex=True).str.replace(_PUNCT, " ", regex=True)
    return collapse_ws(s)


def clean_scalar(x: str) -> str:
    x = fold_accents_str(x).lower().replace("&", " and ")
    x = _PUNCT.sub(" ", _DOT_INITIAL.sub(r"\1", x))
    return _WS.sub(" ", x).strip()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class ColumnNormalizer:
    """Shared column handling. Subclasses implement :meth:`transform`."""

    def __init__(self, columns: ColumnsArg = None, output: ColumnsArg = None) -> None:
        self.columns = columns
        self.output = output

    def transform(self, s: pd.Series) -> pd.Series:  # pragma: no cover - abstract
        raise NotImplementedError

    def _plan(self, records: pd.DataFrame, ctx: Any) -> List[tuple]:
        """List of (source column, output column)."""
        outs = [self.output] if isinstance(self.output, str) else (list(self.output) if self.output else None)
        if self.columns is None:
            out = outs[0] if outs else DEFAULT_OUTPUT
            if out in records.columns:
                src = out
            else:
                src = ctx.columns.name if ctx.columns.name in records.columns else ctx.columns.rname
            return [(src, out)]
        cols = [self.columns] if isinstance(self.columns, str) else list(self.columns)
        if outs is None:
            outs = [DEFAULT_OUTPUT] if len(cols) == 1 else [f"{c}_norm" for c in cols]
        if len(outs) != len(cols):
            raise ValueError("`output` must have one entry per column in `columns`")
        return list(zip(cols, outs))

    def __call__(self, records: pd.DataFrame, ctx: Any) -> pd.DataFrame:
        out = records.copy()
        for src, dst in self._plan(records, ctx):
            if src not in records.columns:
                raise KeyError(f"Column '{src}' not found in records")
            out[dst] = self.transform(as_text(records[src]))
        return out


@register("normalizer", "strip_whitespace")
class StripWhitespace(ColumnNormalizer):
    """Trim and collapse runs of whitespace."""

    def transform(self, s: pd.Series) -> pd.Series:
        return collapse_ws(s)


@register("normalizer", "lowercase")
class Lowercase(ColumnNormalizer):
    """Lower-case (casefold-like via ``str.lower``)."""

    def transform(self, s: pd.Series) -> pd.Series:
        return s.str.lower()


@register("normalizer", "fold_accents")
class FoldAccents(ColumnNormalizer):
    """NFKD decomposition, then drop combining marks (no dependency)."""

    def transform(self, s: pd.Series) -> pd.Series:
        return fold_series(s)


@register("normalizer", "ampersand")
class Ampersand(ColumnNormalizer):
    """Replace ``&`` with ``and``."""

    def transform(self, s: pd.Series) -> pd.Series:
        return collapse_ws(s.str.replace("&", " and ", regex=False))


@register("normalizer", "remove_punctuation")
class RemovePunctuation(ColumnNormalizer):
    """Replace characters matching ``pattern`` with a space, then collapse whitespace."""

    def __init__(self, columns: ColumnsArg = None, output: ColumnsArg = None, pattern: str = r"[^\w\s]") -> None:
        super().__init__(columns, output)
        self.pattern = pattern
        self._re = re.compile(pattern)

    def transform(self, s: pd.Series) -> pd.Series:
        return collapse_ws(s.str.replace(self._re, " ", regex=True))


@register("normalizer", "transliterate")
class Transliterate(ColumnNormalizer):
    """Transliterate non-Latin scripts to ASCII with ``anyascii`` (extra ``[normalize]``)."""

    def transform(self, s: pd.Series) -> pd.Series:
        try:
            from anyascii import anyascii  # lazy: optional dependency
        except ImportError as exc:
            raise ImportError("transliterate needs `anyascii`; install deduplix[normalize]") from exc
        s = s.copy()
        mask = s.notna() & ~s.fillna("").str.isascii()
        if mask.any():
            s[mask] = s[mask].map(anyascii)
        return s


@register("normalizer", "expand_abbreviations")
class ExpandAbbreviations(ColumnNormalizer):
    """Whole-token replacement from a table (``intl`` -> ``international``).

    ``table`` is a packaged table name, a YAML path, or a dict.
    """

    def __init__(self, columns: ColumnsArg = None, output: ColumnsArg = None,
                 table: Union[str, dict] = "abbreviations.yaml") -> None:
        super().__init__(columns, output)
        self.table = table
        mapping = {str(k).lower(): str(v) for k, v in load_table(table).items()}
        self._map = mapping
        if mapping:
            alts = "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
            self._re: Optional[re.Pattern] = re.compile(rf"(?<!\S)(?:{alts})(?!\S)")
        else:
            self._re = None

    def transform(self, s: pd.Series) -> pd.Series:
        if self._re is None:
            return s
        return s.str.replace(self._re, lambda m: self._map[m.group(0)], regex=True)


@register("normalizer", "remove_stopwords")
class RemoveStopwords(ColumnNormalizer):
    """Drop whole-token stopwords of the given languages (keeps the name if all would go)."""

    def __init__(self, columns: ColumnsArg = None, output: ColumnsArg = None,
                 languages: Sequence[str] = ("en",), table: Union[str, dict] = "stopwords.yaml") -> None:
        super().__init__(columns, output)
        self.languages = list(languages)
        data = load_table(table)
        words = sorted({w for lang in self.languages for w in data.get(lang, [])}, key=len, reverse=True)
        self._re = re.compile(r"(?<!\S)(?:" + "|".join(re.escape(w) for w in words) + r")(?!\S)") if words else None

    def transform(self, s: pd.Series) -> pd.Series:
        if self._re is None:
            return s
        stripped = collapse_ws(s.str.replace(self._re, " ", regex=True))
        return stripped.where(stripped.fillna("").ne(""), s)


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------
DEFAULT_CHAIN = ("strip_whitespace", "fold_accents", "lowercase", "ampersand", "remove_punctuation")
_CHAIN_STEPS: Dict[str, Callable[[], ColumnNormalizer]] = {
    "strip_whitespace": StripWhitespace,
    "fold_accents": FoldAccents,
    "lowercase": Lowercase,
    "ampersand": Ampersand,
    "remove_punctuation": RemovePunctuation,
    "transliterate": Transliterate,
    "expand_abbreviations": ExpandAbbreviations,
    "remove_stopwords": RemoveStopwords,
}


class Chain(ColumnNormalizer):
    """Apply several steps in order on one column (or several), writing only the final output."""

    def __init__(self, columns: ColumnsArg = None, output: ColumnsArg = None,
                 steps: Sequence[str] = DEFAULT_CHAIN) -> None:
        super().__init__(columns, output)
        self.steps = list(steps)
        unknown = [x for x in self.steps if x not in _CHAIN_STEPS]
        if unknown:
            raise ValueError(f"Unknown chain steps {unknown}; choose from {sorted(_CHAIN_STEPS)}")
        self._objs = [_CHAIN_STEPS[x]() for x in self.steps]

    def transform(self, s: pd.Series) -> pd.Series:
        for step in self._objs:
            s = step.transform(s)
        return s


@register("normalizer", "default_text")
class DefaultText(Chain):
    """strip_whitespace, fold_accents, lowercase, ampersand, remove_punctuation -> ``name_norm``."""

    def __init__(self, columns: ColumnsArg = None, output: ColumnsArg = None) -> None:
        super().__init__(columns, output, DEFAULT_CHAIN)
        self.steps = list(DEFAULT_CHAIN)

    def _plan(self, records: pd.DataFrame, ctx: Any) -> List[tuple]:
        # always start from the raw name column, even if name_norm already exists
        plan = super()._plan(records, ctx)
        if self.columns is None:
            src = ctx.columns.name if ctx.columns.name in records.columns else ctx.columns.rname
            plan = [(src, plan[0][1])]
        return plan


@register("normalizer", "chain")
class ChainNormalizer(Chain):
    """Configurable chain: ``chain: {steps: [lowercase, expand_abbreviations]}``."""


def chain(*steps: str, columns: ColumnsArg = None, output: ColumnsArg = None) -> Chain:
    """Helper building a :class:`Chain` from step names."""
    return Chain(columns=columns, output=output, steps=steps or DEFAULT_CHAIN)
