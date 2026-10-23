"""OpenAI Chat Completions provider (SDK imported lazily)."""
from __future__ import annotations

import json
from typing import Any, Optional

from ...engine.registry import register
from .base import BaseProvider, ProviderResponse


def chat_response(resp: Any, structured: bool) -> ProviderResponse:
    """Convert an OpenAI-style chat completion into a ProviderResponse."""
    text = resp.choices[0].message.content
    usage = getattr(resp, "usage", None)
    parsed = None
    if structured and text:
        try:
            val = json.loads(text)
            parsed = val if isinstance(val, dict) else None
        except ValueError:
            parsed = None
    return ProviderResponse(
        text=text, json=parsed,
        usage={"input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
               "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0)},
        raw=resp)


def chat_kwargs(model: str, messages, schema: Optional[dict], temperature: float,
                max_tokens: int, structured: bool) -> dict:
    kw = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if structured and schema:
        kw["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "decisions", "schema": schema, "strict": False}}
    return kw


@register("provider", "openai")
class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 structured: bool = True, client: Any = None, **kw: Any):
        super().__init__(**kw)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.structured = structured
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError("The openai provider needs `pip install openai>=1.0`") from exc
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url,
                                         timeout=self.timeout, max_retries=0)
        return self._client

    def _complete(self, messages, *, schema, temperature, max_tokens):
        resp = self._get_client().chat.completions.create(
            **chat_kwargs(self.model, messages, schema, temperature, max_tokens, self.structured))
        return chat_response(resp, self.structured)
