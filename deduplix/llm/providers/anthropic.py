"""Anthropic Messages provider (SDK imported lazily); structured output via a forced tool."""
from __future__ import annotations

from typing import Any, Optional

from ...engine.registry import register
from .base import BaseProvider, ProviderResponse

TOOL_NAME = "record_decisions"


@register("provider", "anthropic")
class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, model: str, api_key: Optional[str] = None, max_tokens: int = 4096,
                 structured: bool = True, client: Any = None, **kw: Any):
        super().__init__(**kw)
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.structured = structured
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError("The anthropic provider needs `pip install anthropic`") from exc
            self._client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout,
                                               max_retries=0)
        return self._client

    def _complete(self, messages, *, schema, temperature, max_tokens):
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        kw: dict = dict(model=self.model, max_tokens=max_tokens or self.max_tokens,
                        temperature=temperature,
                        messages=[m for m in messages if m.get("role") != "system"])
        if system:
            kw["system"] = system
        if self.structured and schema:
            kw["tools"] = [{"name": TOOL_NAME, "description": "Record the decision for every pair.",
                            "input_schema": schema}]
            kw["tool_choice"] = {"type": "tool", "name": TOOL_NAME}
        resp = self._get_client().messages.create(**kw)
        texts, parsed = [], None
        for block in getattr(resp, "content", None) or []:
            if getattr(block, "type", None) == "tool_use" and parsed is None:
                parsed = dict(getattr(block, "input", None) or {})
            elif getattr(block, "text", None):
                texts.append(block.text)
        usage = getattr(resp, "usage", None)
        return ProviderResponse(
            text="\n".join(texts).strip() or None, json=parsed,
            usage={"input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                   "output_tokens": int(getattr(usage, "output_tokens", 0) or 0)},
            raw=resp)
