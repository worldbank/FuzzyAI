"""Databricks model serving: OpenAI-compatible chat, or raw ``invocations`` (classifiers)."""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Callable, Optional

from jinja2 import Environment, StrictUndefined

from ...engine.registry import register
from ...exceptions import ConfigurationError
from .base import BaseProvider, ProviderResponse
from .openai import chat_kwargs, chat_response

DEFAULT_REQUEST_TEMPLATE = '{"inputs": [{{ prompt | tojson }}]}'


def read_path(obj: Any, path: str) -> Any:
    """Read ``predictions[0].probabilities`` style paths."""
    for tok in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        obj = obj[int(tok[1:-1])] if tok.startswith("[") else obj[tok]
    return obj


def _urllib_post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return json.loads(r.read().decode("utf-8"))


@register("provider", "databricks")
class DatabricksProvider(BaseProvider):
    name = "databricks"

    def __init__(self, endpoint: str, host: Optional[str] = None, token: Optional[str] = None,
                 mode: str = "chat", structured: bool = False,
                 request_template: Optional[str] = None, response_path: Optional[str] = None,
                 client: Any = None, post: Optional[Callable[..., dict]] = None, **kw: Any):
        super().__init__(**kw)
        if mode not in ("chat", "invocations"):
            raise ConfigurationError(f"databricks mode must be 'chat' or 'invocations', got '{mode}'")
        self.endpoint = endpoint
        self.model = endpoint
        self.host = host
        self.token = token
        self.mode = mode
        self.structured = structured
        self.request_template = request_template or DEFAULT_REQUEST_TEMPLATE
        self.response_path = response_path
        self._client = client
        self._post = post or _urllib_post

    def _credentials(self):
        host = self.host or os.getenv("DATABRICKS_HOST")
        token = self.token or os.getenv("DATABRICKS_TOKEN") or os.getenv("DATABRICKS_API_KEY")
        if not host:
            raise ConfigurationError("Databricks host missing (host= or DATABRICKS_HOST)")
        if not token:
            raise ConfigurationError(
                "Databricks token missing (token=, DATABRICKS_TOKEN or DATABRICKS_API_KEY)")
        if not host.startswith("http"):
            host = "https://" + host
        return host.rstrip("/"), token

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError("databricks chat mode needs `pip install openai>=1.0`") from exc
            host, token = self._credentials()
            self._client = openai.OpenAI(api_key=token, base_url=f"{host}/serving-endpoints",
                                         timeout=self.timeout, max_retries=0)
        return self._client

    def _complete(self, messages, *, schema, temperature, max_tokens):
        if self.mode == "chat":
            resp = self._get_client().chat.completions.create(**chat_kwargs(
                self.endpoint, messages, schema, temperature, max_tokens, self.structured))
            return chat_response(resp, self.structured)
        host, token = self._credentials()
        prompt = [m["content"] for m in messages if m.get("role") == "user"][-1]
        env = Environment(undefined=StrictUndefined)
        body = json.loads(env.from_string(self.request_template).render(
            prompt=prompt, messages=messages))
        data = self._post(f"{host}/serving-endpoints/{self.endpoint}/invocations", body,
                          {"Authorization": f"Bearer {token}"}, self.timeout)
        out = read_path(data, self.response_path) if self.response_path else data
        if isinstance(out, dict) and out and all(isinstance(v, (int, float)) for v in out.values()):
            return ProviderResponse(text=json.dumps(out), probabilities=dict(out), raw=data)
        if isinstance(out, dict):
            return ProviderResponse(text=json.dumps(out), json=out, raw=data)
        return ProviderResponse(text=out if isinstance(out, str) else json.dumps(out), raw=data)
