"""Turn a ProviderResponse into per-pair Decisions. ``pair_index`` is the local batch position."""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional

from pydantic import BaseModel

from ..engine.registry import register
from .providers.base import ProviderResponse

DECISIONS_SCHEMA = {
    "type": "object",
    "properties": {"decisions": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "pair_index": {"type": "integer"},
            "is_duplicate": {"type": "boolean"},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["pair_index", "is_duplicate"],
    }}},
    "required": ["decisions"],
}


class ParseError(ValueError):
    """The provider answer could not be understood."""


class Decision(BaseModel):
    pair_index: int
    is_duplicate: bool
    confidence: Optional[float] = None
    reason: str = ""


def extract_json(text: str) -> Any:
    """Fenced block, then first ``{...}``, then the whole text (legacy ``_extract_json``)."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return json.loads(fence.group(1))
    obj = re.search(r"\{.*\}", text, re.DOTALL)
    if obj:
        return json.loads(obj.group(0))
    return json.loads(text)


@register("parser", "json_decisions")
class JsonDecisionsParser:
    """Parse ``{"decisions": [{pair_index, is_duplicate, confidence?, reason?}]}``.

    Pairs absent from the answer, or with an unusable item, are simply not returned
    (undecided, never a match). Out-of-range and repeated indices are ignored.
    """

    schema = DECISIONS_SCHEMA

    def __call__(self, response: ProviderResponse, batch: list) -> List[Decision]:
        data = response.json
        if data is None:
            if not response.text:
                raise ParseError("empty response")
            try:
                data = extract_json(response.text)
            except ValueError as exc:
                raise ParseError(f"malformed JSON: {exc}") from exc
        items = data.get("decisions") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ParseError("no 'decisions' list in the answer")
        out: List[Decision] = []
        seen = set()
        for item in items:
            try:
                d = Decision(**item)
            except Exception:
                continue
            if 0 <= d.pair_index < len(batch) and d.pair_index not in seen:
                seen.add(d.pair_index)
                out.append(d)
        return out


@register("parser", "probabilities")
class ProbabilitiesParser:
    """Read ``response.probabilities`` of a classifier endpoint (one pair per call)."""

    schema = None
    single_pair = True

    def __init__(self, positive_label: str = "yes"):
        self.positive_label = positive_label

    def __call__(self, response: ProviderResponse, batch: list) -> List[Decision]:
        probs = response.probabilities
        if not probs:
            raise ParseError("response carries no probabilities")
        lowered = {str(k).lower(): float(v) for k, v in probs.items()}
        label = str(self.positive_label).lower()
        if label not in lowered:
            raise ParseError(f"positive label '{self.positive_label}' not in {sorted(lowered)}")
        p = lowered[label]
        return [Decision(pair_index=0, is_duplicate=p >= 0.5, confidence=p,
                         reason=f"p({self.positive_label})={p:.3f}")]
