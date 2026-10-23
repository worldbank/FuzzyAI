"""Exact-key blocker: equal normalised key values form candidate pairs."""
from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..engine.registry import register
from .base import BaseBlocker, require

logger = logging.getLogger("deduplix.blocking.exact")


def norm_alnum(v) -> Optional[str]:
    s = re.sub(r"[^0-9A-Za-z]", "", str(v)).upper().lstrip("0")
    return s or None


def norm_lei(v) -> Optional[str]:
    """20-character LEI with valid ISO 17442 check digits (ISO 7064 MOD 97-10), else None."""
    s = re.sub(r"[^0-9A-Za-z]", "", str(v)).upper()
    if len(s) != 20:
        return None
    num = "".join(str(int(c, 36)) for c in s)
    return s if int(num) % 97 == 1 else None


_TLD = None


def norm_domain(v) -> Optional[str]:
    """Registered domain (tldextract), lower case, ``www`` removed."""
    global _TLD
    if _TLD is None:
        tldextract = require("tldextract", "normalize")
        _TLD = tldextract.TLDExtract(suffix_list_urls=())  # bundled snapshot, no network
    ext = _TLD(str(v).strip())
    dom = getattr(ext, "top_domain_under_public_suffix", None) or ext.registered_domain
    dom = (dom or "").lower()
    if dom.startswith("www."):
        dom = dom[4:]
    return dom or None


def norm_phone(v) -> Optional[str]:
    """E.164 when ``phonenumbers`` is installed, else ``+`` and the digits."""
    s = str(v).strip()
    try:
        import phonenumbers
    except ImportError:
        digits = re.sub(r"\D", "", s)
        return ("+" + digits) if digits else None
    try:
        num = phonenumbers.parse(s, None)
        if phonenumbers.is_possible_number(num):
            return phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.E164)
    except Exception:
        pass
    return None


NORMALIZERS: Dict[str, Callable] = {
    "alnum": norm_alnum, "lei": norm_lei, "domain": norm_domain, "phone": norm_phone,
}


def _missing(v) -> bool:
    return v is None or v is pd.NA or (isinstance(v, float) and v != v)


@register("blocker", "exact")
class ExactKeyBlocker(BaseBlocker):
    """Pairs of records sharing a normalised key value (id codes, LEI, domain, phone).

    Parameters
    ----------
    keys : list of str
        Columns to join on; each key contributes its own candidate pairs.
    normalizers : dict, optional
        ``{key: kind}`` with kind in ``alnum, lei, domain, phone`` (or a callable).
    max_key_freq : int
        Key values occurring more often than this on either side are skipped and logged.
    """

    name = "exact"

    def __init__(self, keys: Optional[List[str]] = None, normalizers: Optional[Dict] = None,
                 max_key_freq: int = 50, **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(keys, str):
            keys = [keys]
        if not keys:
            raise ValueError("ExactKeyBlocker needs at least one key column")
        self.keys = list(keys)
        self.normalizers = dict(normalizers or {})
        for kind in self.normalizers.values():
            if not callable(kind) and kind not in NORMALIZERS:
                raise ValueError(f"Unknown normalizer '{kind}'; choose from {sorted(NORMALIZERS)}")
        self.max_key_freq = max_key_freq
        self.skipped_keys: Dict[str, Dict[str, int]] = {}

    def _normalized(self, df: pd.DataFrame, key: str) -> np.ndarray:
        if key not in df.columns:
            raise KeyError(f"Blocker 'exact': key column '{key}' not found")
        kind = self.normalizers.get(key)
        fn = kind if callable(kind) else (NORMALIZERS[kind] if kind else None)
        cache: Dict = {}
        out = np.empty(len(df), dtype=object)
        for n, v in enumerate(df[key].tolist()):
            if _missing(v):
                out[n] = None
            elif fn is None:
                s = str(v)
                out[n] = s if s != "" else None
            else:
                if v not in cache:
                    cache[v] = fn(v)
                out[n] = cache[v]
        return out

    def _prepare(self, left, right, ctx) -> None:
        self._nl = {k: self._normalized(left, k) for k in self.keys}
        self._nr = self._nl if self._dedup else {k: self._normalized(right, k) for k in self.keys}
        self.skipped_keys = {}

    def _block(self, left_idx, right_idx, ctx):
        res_i, res_j = [], []
        for k in self.keys:
            L = pd.DataFrame({"i": np.arange(len(left_idx)), "k": self._nl[k][left_idx]}).dropna()
            R = pd.DataFrame({"j": np.arange(len(right_idx)), "k": self._nr[k][right_idx]}).dropna()
            if L.empty or R.empty:
                continue
            counts_r = R["k"].value_counts()
            counts_l = L["k"].value_counts()
            big = set(counts_r[counts_r > self.max_key_freq].index) | \
                set(counts_l[counts_l > self.max_key_freq].index)
            if big:
                self.skipped_keys.setdefault(k, {}).update(
                    {str(b): int(max(counts_r.get(b, 0), counts_l.get(b, 0))) for b in big})
                logger.warning("exact: key '%s': skipped %d value(s) with more than %d records, e.g. %s",
                               k, len(big), self.max_key_freq, sorted(map(str, big))[:5])
                L, R = L[~L["k"].isin(big)], R[~R["k"].isin(big)]
            m = L.merge(R, on="k")
            res_i.append(m["i"].to_numpy())
            res_j.append(m["j"].to_numpy())
        if not res_i:
            e = np.array([], dtype=np.int64)
            return e, e, np.array([], dtype=np.float32)
        i, j = np.concatenate(res_i), np.concatenate(res_j)
        return i, j, np.ones(len(i), dtype=np.float32)
