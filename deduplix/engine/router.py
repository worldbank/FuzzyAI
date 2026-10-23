"""Routers: split pairs into accept / review / reject."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..exceptions import ConfigurationError
from .registry import build, register
from .schema import ACCEPT, REJECT, REVIEW, ROUTES, SCORE

_OPS = ("equals", "in", "gte", "lt", "isna")


@register("router", "threshold")
class ThresholdRouter:
    """Route on a score column: >= accept_above accept, < reject_below reject, else review."""

    def __init__(self, accept_above: float = 0.95, reject_below: float = 0.60, on: str = SCORE):
        if reject_below > accept_above:
            raise ConfigurationError("reject_below must not exceed accept_above")
        self.accept_above, self.reject_below, self.on = accept_above, reject_below, on

    def __call__(self, pairs: pd.DataFrame, ctx: Any = None) -> pd.Series:
        s = pairs[self.on].astype("float64").to_numpy()
        out = np.full(len(pairs), REVIEW, dtype=object)
        out[s >= self.accept_above] = ACCEPT
        out[s < self.reject_below] = REJECT
        # a missing score cannot be decided automatically
        out[np.isnan(s)] = REVIEW
        return pd.Series(out, index=pairs.index, name="route")


def _rule_mask(rule: Union[Dict[str, Any], Callable], pairs: pd.DataFrame) -> np.ndarray:
    if callable(rule):
        return np.asarray(rule(pairs), dtype=bool)
    ops = [op for op in _OPS if op in rule]
    if "feature" not in rule or len(ops) != 1 or "route" not in rule:
        raise ConfigurationError(
            f"Rule {rule} needs 'feature', 'route' and exactly one of {list(_OPS)}")
    col = pairs[rule["feature"]] if rule["feature"] in pairs.columns else None
    if col is None:
        raise ConfigurationError(f"Rule refers to unknown column '{rule['feature']}'")
    op = ops[0]
    if op == "equals":
        mask = col.astype(object).eq(rule["equals"])
    elif op == "in":
        mask = col.astype(object).isin(list(rule["in"]))
    elif op == "gte":
        mask = col.astype("float64") >= rule["gte"]
    elif op == "lt":
        mask = col.astype("float64") < rule["lt"]
    else:  # isna
        mask = col.isna() if rule["isna"] else col.notna()
    return mask.fillna(False).to_numpy(dtype=bool)


@register("router", "rule")
class RuleRouter:
    """Declarative rules first (first match wins), then a fallback router.

    A YAML rule is ``{feature: f_x, equals|in|gte|lt|isna: value, route: reject}``;
    in Python a rule may instead be a callable ``pairs -> bool mask`` paired with a route
    via ``(callable, "reject")``. Nothing is evaluated from strings.
    """

    def __init__(self, rules: Sequence[Any] = (), then: Optional[Any] = None):
        self.rules: List[Any] = []
        for r in rules:
            if isinstance(r, tuple) and len(r) == 2 and callable(r[0]):
                fn, route = r
                r = {"callable": fn, "route": route}
            if r.get("route") not in ROUTES:
                raise ConfigurationError(f"Rule route must be one of {ROUTES}: {r}")
            self.rules.append(r)
        if isinstance(then, dict) and "threshold" not in then and len(then) != 1:
            then = {"threshold": then}  # {accept_above:.., reject_below:..} shorthand
        self.then = build("router", then if then is not None else "threshold")

    def __call__(self, pairs: pd.DataFrame, ctx: Any = None) -> pd.Series:
        out = self.then(pairs, ctx).astype(object).copy()
        decided = np.zeros(len(pairs), dtype=bool)
        for r in self.rules:
            mask = _rule_mask(r["callable"] if "callable" in r else r, pairs) & ~decided
            out[mask] = r["route"]
            decided |= mask
        return out.rename("route")
