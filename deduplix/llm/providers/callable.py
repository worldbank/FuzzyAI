"""Wrap any Python function as a provider."""
from __future__ import annotations

import inspect
from typing import Any, Callable

from ...engine.registry import register
from .base import BaseProvider
from .fake import coerce_response


@register("provider", "callable")
class CallableProvider(BaseProvider):
    """``fn(messages, [schema=, temperature=, max_tokens=])`` -> str | dict | ProviderResponse.

    Only the keyword arguments ``fn`` declares (or all, with ``**kwargs``) are passed.
    """

    name = "callable"

    def __init__(self, fn: Callable[..., Any], model: str = "callable", **kw: Any):
        super().__init__(**kw)
        self.fn = fn
        self.model = model
        params = inspect.signature(fn).parameters
        self._var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
        self._accepted = set(params)

    def _complete(self, messages, *, schema, temperature, max_tokens):
        extra = {"schema": schema, "temperature": temperature, "max_tokens": max_tokens}
        if not self._var_kw:
            extra = {k: v for k, v in extra.items() if k in self._accepted}
        return coerce_response(self.fn(messages, **extra))
