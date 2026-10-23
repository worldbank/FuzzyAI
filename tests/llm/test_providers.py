from types import SimpleNamespace as NS

import pytest

import deduplix.llm  # noqa: F401  (registration)
from deduplix.engine import registry
from deduplix.exceptions import ConfigurationError
from deduplix.llm.providers import (AnthropicProvider, CallableProvider, DatabricksProvider,
                                    FakeProvider, OpenAIProvider)
from deduplix.llm.providers.databricks import read_path

MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]


class HttpError(Exception):
    def __init__(self, status_code):
        super().__init__(str(status_code))
        self.status_code = status_code


def test_fake_cycles_and_records():
    p = FakeProvider(["a", {"k": 1}])
    assert p.complete(MSGS).text == "a"
    assert p.complete(MSGS).json == {"k": 1}
    assert p.complete(MSGS).text == "a"
    assert p.prompts == ["hello"] * 3


def test_retry_backoff_full_jitter():
    sleeps = []
    p = FakeProvider([HttpError(429), HttpError(503), "ok"], max_retries=3,
                     sleep=sleeps.append, rng=lambda: 0.5)
    assert p.complete(MSGS).text == "ok"
    assert sleeps == [0.5, 1.0]


def test_backoff_cap_exhaustion_non_retryable():
    p = FakeProvider([HttpError(500)], max_retries=2, sleep=lambda s: None, rng=lambda: 1.0)
    assert p.backoff_delay(10) == 30.0
    with pytest.raises(HttpError):
        p.complete(MSGS)
    assert len(p.calls) == 3
    q = FakeProvider([HttpError(400), "x"], max_retries=3, sleep=lambda s: None)
    with pytest.raises(HttpError):
        q.complete(MSGS)
    assert len(q.calls) == 1
    t = FakeProvider([TimeoutError(), "x"], max_retries=1, sleep=lambda s: None)
    assert t.complete(MSGS).text == "x"


def test_rate_limit_allows_calls():
    p = FakeProvider(["a"], requests_per_minute=100, sleep=lambda s: None)
    for _ in range(5):
        p.complete(MSGS)
    assert p._limiter.get_statistics()["total_requests"] == 5


def test_callable_provider_passes_only_declared_kwargs():
    seen = {}

    def fn(messages, temperature):
        seen["t"] = temperature
        return '{"decisions": []}'

    r = CallableProvider(fn).complete(MSGS, temperature=0.3, schema={"a": 1})
    assert r.text == '{"decisions": []}' and seen["t"] == 0.3
    r = CallableProvider(lambda m, **kw: {"x": sorted(kw)}).complete(MSGS)
    assert r.json == {"x": ["max_tokens", "schema", "temperature"]}


def test_openai_provider_with_client_and_schema():
    captured = {}

    def create(**kw):
        captured.update(kw)
        return NS(choices=[NS(message=NS(content='{"decisions": []}'))],
                  usage=NS(prompt_tokens=7, completion_tokens=3))

    p = OpenAIProvider("gpt-x", client=NS(chat=NS(completions=NS(create=create))))
    r = p.complete(MSGS, schema={"type": "object"}, max_tokens=99)
    assert captured["response_format"]["type"] == "json_schema" and captured["max_tokens"] == 99
    assert r.json == {"decisions": []} and r.usage == {"input_tokens": 7, "output_tokens": 3}


def test_anthropic_provider_forced_tool():
    captured = {}

    def create(**kw):
        captured.update(kw)
        return NS(content=[NS(type="tool_use", input={"decisions": [1]})],
                  usage=NS(input_tokens=5, output_tokens=2))

    p = AnthropicProvider("claude-x", client=NS(messages=NS(create=create)))
    r = p.complete(MSGS, schema={"type": "object"}, max_tokens=2048)
    assert captured["tool_choice"] == {"type": "tool", "name": "record_decisions"}
    assert captured["system"] == "sys" and captured["max_tokens"] == 2048
    assert all(m["role"] != "system" for m in captured["messages"])
    assert r.json == {"decisions": [1]} and r.usage["input_tokens"] == 5


def test_databricks_invocations_probabilities():
    calls = []

    def post(url, payload, headers, timeout):
        calls.append((url, payload, headers))
        return {"predictions": [{"probabilities": {"yes": 0.9, "no": 0.1}}]}

    p = DatabricksProvider("semif", host="adb.example.net", token="tok", mode="invocations",
                           response_path="predictions[0].probabilities", post=post)
    r = p.complete(MSGS)
    assert r.probabilities == {"yes": 0.9, "no": 0.1}
    assert calls[0][0] == "https://adb.example.net/serving-endpoints/semif/invocations"
    assert calls[0][1] == {"inputs": ["hello"]}
    assert calls[0][2]["Authorization"] == "Bearer tok"


def test_databricks_missing_credentials(monkeypatch):
    for k in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    p = DatabricksProvider("e", mode="invocations", post=lambda *a: {})
    with pytest.raises(ConfigurationError):
        p.complete(MSGS)
    with pytest.raises(ConfigurationError):
        DatabricksProvider("e", mode="bogus")


def test_read_path():
    assert read_path({"a": [{"b": 5}]}, "a[0].b") == 5


def test_registry_build_from_yaml_style_spec():
    p = registry.build("provider", {"databricks": {"endpoint": "ep", "mode": "chat"}})
    assert isinstance(p, DatabricksProvider) and p.endpoint == "ep" and p.mode == "chat"
    assert registry.spec_of(p)["params"] == {"endpoint": "ep", "mode": "chat"}
    assert isinstance(registry.build("provider", {"fake": {"responses": ["x"]}}), FakeProvider)
    assert isinstance(registry.build("provider", {"openai": {"model": "m"}}), OpenAIProvider)
    assert isinstance(registry.build("provider", "fake"), FakeProvider)
