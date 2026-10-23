"""Phonetic (Double Metaphone) key blocker. Latin-script and English-biased; never a default."""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import List

import numpy as np
import pandas as pd

from ..engine.registry import register
from .base import BaseBlocker, require

logger = logging.getLogger("deduplix.blocking.phonetic")

_TOKEN = re.compile(r"\w+", re.UNICODE)


@register("blocker", "phonetic")
class PhoneticBlocker(BaseBlocker):
    """Records sharing a Double Metaphone code of one of their ``n`` tokens.

    ``tokens="rarest"`` encodes the ``n`` lowest-frequency tokens of each record,
    ``"first"`` the first ``n``. Primary and secondary codes both become keys. Blocks larger
    than ``max_block_size`` are dropped and logged. Works for Latin script and is biased
    towards English pronunciation; use it as a complement, not as a default blocker.
    """

    name = "phonetic"

    def __init__(self, algorithm: str = "double_metaphone", tokens: str = "rarest", n: int = 2,
                 max_block_size: int = 500, **kwargs) -> None:
        super().__init__(**kwargs)
        if algorithm != "double_metaphone":
            raise ValueError("Only algorithm='double_metaphone' is supported")
        if tokens not in ("rarest", "first"):
            raise ValueError("tokens must be 'rarest' or 'first'")
        self.algorithm = algorithm
        self.tokens = tokens
        self.n = n
        self.max_block_size = max_block_size

    def _check_deps(self) -> None:
        require("metaphone", "phonetic")

    def _keys(self, texts: List[str], freq: Counter, cache: dict) -> pd.DataFrame:
        from metaphone import doublemetaphone

        pos, keys = [], []
        for p, t in enumerate(texts):
            toks = _TOKEN.findall(t.lower())
            if self.tokens == "rarest":
                order = sorted(range(len(toks)), key=lambda x: (freq[toks[x]], x))
                toks = [toks[x] for x in order[: self.n]]
            else:
                toks = toks[: self.n]
            seen = set()
            for tok in toks:
                if tok not in cache:
                    cache[tok] = tuple(c for c in doublemetaphone(tok) if c)
                for code in cache[tok]:
                    if code not in seen:
                        seen.add(code)
                        pos.append(p)
                        keys.append(code)
        return pd.DataFrame({"p": np.array(pos, dtype=np.int64), "k": keys})

    def _prepare(self, left, right, ctx) -> None:
        lt = self._texts(left, ctx, "left")
        rt = lt if self._dedup else self._texts(right, ctx, "right")
        freq: Counter = Counter()
        for t in rt:
            freq.update(_TOKEN.findall(t.lower()))
        if not self._dedup:
            for t in lt:
                freq.update(_TOKEN.findall(t.lower()))
        cache: dict = {}
        self._kl = self._keys(lt, freq, cache)
        self._kr = self._kl if self._dedup else self._keys(rt, freq, cache)

    def _block(self, left_idx, right_idx, ctx):
        def sub(df, idx):
            m = pd.Series(np.arange(len(idx)), index=idx)
            d = df[df["p"].isin(idx)]
            return pd.DataFrame({"i": m.loc[d["p"].to_numpy()].to_numpy(), "k": d["k"].to_numpy()})

        L, R = sub(self._kl, left_idx), sub(self._kr, right_idx)
        if L.empty or R.empty:
            e = np.array([], dtype=np.int64)
            return e, e, None
        size = R["k"].value_counts()
        if not self._dedup:
            size = size.add(L["k"].value_counts(), fill_value=0)
        big = set(size[size > self.max_block_size].index)
        if big:
            logger.warning("phonetic: dropped %d block(s) larger than %d, e.g. %s",
                           len(big), self.max_block_size, sorted(big)[:5])
            L, R = L[~L["k"].isin(big)], R[~R["k"].isin(big)]
        m = L.merge(R.rename(columns={"i": "j"}), on="k")[["i", "j"]].drop_duplicates()
        return m["i"].to_numpy(), m["j"].to_numpy(), None
