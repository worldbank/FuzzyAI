"""Deterministic provider for tests: cycles through canned responses and records prompts."""
from __future__ import annotations

import json as _json
import threading
from typing import Any, List, Optional

from ...engine.registry import register
from .base import BaseProvider, ProviderResponse


def coerce_response(obj: Any) -> ProviderResponse:
    """str -> text, dict -> json (+ text), ProviderResponse unchanged."""
    if isinstance(obj, ProviderResponse):
        return obj
    if isinstance(obj, dict):
        return ProviderResponse(text=_json.dumps(obj), json=obj)
    return ProviderResponse(text=None if obj is None else str(obj))


@register("provider", "fake")
class FakeProvider(BaseProvider):
    """Cycle through ``responses``; an Exception instance (or class) is raised, a callable
    receives the messages. ``prompts`` holds the user prompt of every call."""

    name = "fake"

    def __init__(self, responses: Optional[list] = None, model: str = "fake-model", **kw: Any):
        kw.setdefault("max_retries", 0)
        super().__init__(**kw)
        self.responses = list(responses) if responses else [ProviderResponse(text="{}")]
        self.model = model
        self.calls: List[dict] = []
        self.prompts: List[str] = []
        self._i = 0
        self._lock = threading.Lock()

    def _complete(self, messages, *, schema, temperature, max_tokens):
        with self._lock:
            item = self.responses[self._i % len(self.responses)]
            self._i += 1
            self.calls.append({"messages": messages, "schema": schema,
                               "temperature": temperature, "max_tokens": max_tokens})
            users = [m["content"] for m in messages if m.get("role") == "user"]
            self.prompts.append(users[-1] if users else "")
        if isinstance(item, BaseException) or (isinstance(item, type)
                                               and issubclass(item, BaseException)):
            raise item
        if callable(item) and not isinstance(item, ProviderResponse):
            item = item(messages)
        resp = coerce_response(item)
        if not resp.usage:
            n_in = sum(len(str(m.get("content", "")).split()) for m in messages)
            n_out = len((resp.text or "").split())
            resp = ProviderResponse(resp.text, resp.json, resp.probabilities,
                                    {"input_tokens": n_in, "output_tokens": n_out}, resp.raw)
        return resp
