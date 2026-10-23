"""Fellegi-Sunter scorer: u from random record pairs, m (and prior) by EM on the candidates."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..compare._util import feature_column
from ..engine import registry
from ..engine.registry import register
from ..engine.schema import I1, I2
from ..exceptions import ConfigurationError

logger = logging.getLogger("deduplix.scoring.fellegi_sunter")
EPS = 1e-6
NA_LEVEL = -1


@register("scorer", "fellegi_sunter")
class FellegiSunter:
    """Posterior match probability from per-level match (m) and non-match (u) probabilities.

    Parameters
    ----------
    levels : dict
        Feature -> descending-or-any list of numeric thresholds (level = number of thresholds
        the value reaches), or ``None``/``[]`` for a categorical feature (levels = categories).
        Missing values form their own level and contribute weight 0.
    prior : float, optional
        P(match) among candidates; estimated by EM when ``None``.
    estimate_u : {"random_sample", "em"}
        ``random_sample`` featurises random record pairs with ``comparators`` (specs, or
        ``ctx.artifacts['comparators']``) to estimate u; without comparators it falls back to
        estimating u by EM on the candidates (with a warning).
    u_sample_size, max_iter, tol : sampling size and EM controls.

    Fitted ``m``/``u`` are stored in ``self.params_`` and ``ctx.artifacts['fellegi_sunter']``.
    """

    def __init__(self, levels: Optional[Dict[str, Optional[Sequence[float]]]] = None,
                 prior: Optional[float] = None, estimate_u: str = "random_sample",
                 u_sample_size: int = 1_000_000, max_iter: int = 50, tol: float = 1e-4,
                 comparators: Optional[Sequence[Any]] = None, random_state: int = 0):
        if not levels:
            raise ConfigurationError("fellegi_sunter needs 'levels'", config_key="levels")
        if estimate_u not in ("random_sample", "em"):
            raise ConfigurationError("estimate_u must be 'random_sample' or 'em'", config_key="estimate_u")
        if prior is not None and not 0 < prior < 1:
            raise ConfigurationError("prior must be in (0, 1)", config_key="prior")
        self.levels = {feature_column(k): (sorted(float(x) for x in v) if v else None)
                       for k, v in levels.items()}
        self.prior, self.estimate_u = prior, estimate_u
        self.u_sample_size, self.max_iter, self.tol = u_sample_size, max_iter, tol
        self.comparators = list(comparators) if comparators else None
        self.random_state = random_state
        self.params_: Optional[Dict[str, Any]] = None

    # -- encoding -----------------------------------------------------------------
    def _encode(self, pairs: pd.DataFrame, vocab: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        out = {}
        for col, ths in self.levels.items():
            if col not in pairs.columns:
                raise ConfigurationError(f"fellegi_sunter: '{col}' is not a column of the pairs table",
                                         config_key="levels", config_value=col)
            s = pairs[col]
            if ths is not None:
                v = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
                lv = np.searchsorted(np.asarray(ths), v, side="right")
                out[col] = np.where(np.isnan(v), NA_LEVEL, lv).astype(np.int64)
            else:
                codes = pd.Categorical(s.astype(object), categories=vocab[col]).codes
                out[col] = codes.astype(np.int64)  # -1 for NA or unseen
        return out

    def _n_levels(self, col: str, vocab) -> int:
        ths = self.levels[col]
        return len(ths) + 1 if ths is not None else max(len(vocab[col]), 1)

    @staticmethod
    def _counts(codes: np.ndarray, n: int, w: np.ndarray) -> np.ndarray:
        ok = codes >= 0
        return np.bincount(codes[ok], weights=w[ok], minlength=n)

    # -- fitting ------------------------------------------------------------------
    def _random_pairs(self, ctx) -> Optional[pd.DataFrame]:
        specs = self.comparators or ctx.artifacts.get("comparators")
        if not specs:
            return None
        rng = np.random.RandomState(self.random_state)
        nl, nr = len(ctx.left), len(ctx.right)
        n = int(min(self.u_sample_size, max(nl * nr - (nl if ctx.mode == "dedup" else 0), 0)))
        if n == 0:
            return None
        a = rng.randint(0, nl, size=n)
        b = rng.randint(0, nr, size=n)
        if ctx.mode == "dedup":
            keep = a != b
            a, b = a[keep], b[keep]
        pairs = pd.DataFrame({I1: a.astype(np.int64), I2: b.astype(np.int64)})
        frames = [pairs]
        for c in specs:
            comp = registry.build("comparator", c)
            frames.append(comp(pairs, ctx.left, ctx.right, ctx))
        return pd.concat(frames, axis=1)

    def fit(self, pairs: pd.DataFrame, ctx=None) -> "FellegiSunter":
        vocab = {c: sorted(pairs[c].dropna().astype(str).unique())
                 for c, t in self.levels.items() if t is None and c in pairs.columns}
        for c, t in self.levels.items():
            if t is None and c not in vocab:
                raise ConfigurationError(f"fellegi_sunter: '{c}' is not a column of the pairs table",
                                         config_key="levels", config_value=c)
        enc = self._encode(pairs, vocab)
        n_lv = {c: self._n_levels(c, vocab) for c in self.levels}
        u_fixed: Optional[Dict[str, np.ndarray]] = None
        if self.estimate_u == "random_sample":
            sample = self._random_pairs(ctx) if ctx is not None else None
            if sample is None or len(sample) == 0:
                logger.warning("fellegi_sunter: no comparators to featurise random pairs; "
                               "estimating u by EM on the candidates")
            else:
                senc = self._encode(sample, vocab)
                u_fixed = {}
                ones = np.ones(len(sample))
                for c in self.levels:
                    cnt = self._counts(senc[c], n_lv[c], ones) + 0.5
                    u_fixed[c] = cnt / cnt.sum()
        n = len(pairs)
        prior = self.prior if self.prior is not None else 0.1
        # initial guess: high levels are more likely among matches
        m, u = {}, {}
        for c in self.levels:
            k = n_lv[c]
            ramp = np.arange(1, k + 1, dtype="float64")
            m[c] = ramp / ramp.sum()
            u[c] = u_fixed[c] if u_fixed else ramp[::-1] / ramp.sum()
        ones = np.ones(n)
        for _ in range(self.max_iter):
            logodds = np.full(n, np.log(prior / (1 - prior)))
            for c in self.levels:
                lg = np.log(np.clip(m[c], EPS, 1)) - np.log(np.clip(u[c], EPS, 1))
                codes = enc[c]
                logodds += np.where(codes >= 0, lg[np.clip(codes, 0, None)], 0.0)
            post = 1.0 / (1.0 + np.exp(-np.clip(logodds, -50, 50)))
            delta = 0.0
            for c in self.levels:
                nm = self._counts(enc[c], n_lv[c], post) + EPS
                nm /= nm.sum()
                delta = max(delta, float(np.abs(nm - m[c]).max()))
                m[c] = nm
                if u_fixed is None:
                    nu = self._counts(enc[c], n_lv[c], 1 - post) + EPS
                    nu /= nu.sum()
                    delta = max(delta, float(np.abs(nu - u[c]).max()))
                    u[c] = nu
            if self.prior is None:
                new_prior = float(np.clip(post.mean(), EPS, 1 - EPS))
                delta = max(delta, abs(new_prior - prior))
                prior = new_prior
            if delta < self.tol:
                break
        self._vocab, self._m, self._u, self._prior = vocab, m, u, prior
        self.params_ = {"prior": prior, "m": {c: m[c].tolist() for c in m},
                        "u": {c: u[c].tolist() for c in u}, "categories": vocab,
                        "levels": {c: self.levels[c] for c in self.levels}}
        if ctx is not None:
            ctx.artifacts["fellegi_sunter"] = self.params_
        return self

    # -- scoring ------------------------------------------------------------------
    def match_weight(self, pairs: pd.DataFrame) -> np.ndarray:
        """log2 prior odds + sum of log2(m/u) over the non-missing features."""
        enc = self._encode(pairs, self._vocab)
        w = np.full(len(pairs), np.log2(self._prior / (1 - self._prior)))
        for c in self.levels:
            lg = np.log2(np.clip(self._m[c], EPS, 1)) - np.log2(np.clip(self._u[c], EPS, 1))
            codes = enc[c]
            w += np.where(codes >= 0, lg[np.clip(codes, 0, None)], 0.0)
        return w

    def __call__(self, pairs: pd.DataFrame, ctx=None) -> pd.Series:
        if self.params_ is None:
            self.fit(pairs, ctx)
        elif ctx is not None:
            ctx.artifacts.setdefault("fellegi_sunter", self.params_)
        w = self.match_weight(pairs)
        p = 1.0 / (1.0 + np.exp2(-np.clip(w, -60, 60)))
        return pd.Series(p.astype(np.float32), index=pairs.index)
