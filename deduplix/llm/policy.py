"""Decision policy: Decision (or its absence) -> (decision, probability, reason)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from ..engine.schema import MATCH, NON_MATCH, UNVALIDATED
from ..exceptions import ConfigurationError
from .parsers import Decision

_FALLBACKS = {"unvalidated": UNVALIDATED, "match": MATCH, "non_match": NON_MATCH}
Outcome = Tuple[str, float, str]


@dataclass
class DecisionPolicy:
    """``mode="bool"`` uses ``is_duplicate``; ``"probability"`` thresholds ``confidence``
    (match when ``>= match_above``, non-match when ``<= non_match_below``, else undecided).
    ``on_failure`` / ``on_missing`` say what a failed batch / unanswered pair becomes."""

    mode: str = "bool"
    match_above: float = 0.7
    non_match_below: float = 0.3
    on_failure: str = "unvalidated"
    on_missing: str = "unvalidated"

    def __post_init__(self) -> None:
        if self.mode not in ("bool", "probability"):
            raise ConfigurationError(f"policy mode must be 'bool' or 'probability', got '{self.mode}'")
        for attr in ("on_failure", "on_missing"):
            if getattr(self, attr) not in _FALLBACKS:
                raise ConfigurationError(f"{attr} must be one of {list(_FALLBACKS)}, "
                                         f"got '{getattr(self, attr)}'")
        if self.non_match_below > self.match_above:
            raise ConfigurationError("non_match_below must not exceed match_above")

    def decide(self, d: Optional[Decision]) -> Outcome:
        if d is None:
            return self.missing()
        conf = math.nan if d.confidence is None else float(d.confidence)
        if self.mode == "bool":
            return (MATCH if d.is_duplicate else NON_MATCH), conf, d.reason
        if d.confidence is None:
            return self.missing()
        if conf >= self.match_above:
            return MATCH, conf, d.reason
        if conf <= self.non_match_below:
            return NON_MATCH, conf, d.reason
        return UNVALIDATED, conf, d.reason

    def missing(self) -> Outcome:
        return _FALLBACKS[self.on_missing], math.nan, "llm_missing: no answer for this pair"

    def failure(self, exc: BaseException) -> Outcome:
        return _FALLBACKS[self.on_failure], math.nan, f"llm_failed: {type(exc).__name__}"
