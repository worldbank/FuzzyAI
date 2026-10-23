import json
import math

import pandas as pd
import pytest

import deduplix.llm  # noqa: F401
from deduplix.engine import registry
from deduplix.engine.context import RunContext
from deduplix.llm import (Decision, DecisionPolicy, FakeProvider, LLMValidator, ParseError,
                          ProviderResponse, open_cache)
from deduplix.llm.parsers import JsonDecisionsParser, ProbabilitiesParser

LEFT = pd.DataFrame({"id": list("abcd"), "name": ["Acme Ltd", "ACME Limited", "Foo Fund I", "Foo Fund II"],
                     "country": ["US", "US", "GB", "GB"]})
PAIRS = pd.DataFrame({"i1": [0, 2, 0], "i2": [1, 3, 2]}, index=[5, 6, 7])


def ctx():
    return RunContext(left=LEFT, right=LEFT, mode="dedup")


def answer(*items):
    return {"decisions": [dict(pair_index=i, is_duplicate=d, confidence=c, reason=f"r{i}")
                          for i, d, c in items]}


def make(responses, **kw):
    kw.setdefault("cache", None)
    prov = FakeProvider(responses)
    return LLMValidator(prov, **kw), prov


def run(v, pairs=PAIRS):
    c = ctx()
    return v(pairs, c.left, c.right, c), c


def test_basic_decisions_usage_and_index():
    v, prov = make([answer((0, True, 0.9), (1, False, 0.2), (2, False, None))])
    out, c = run(v)
    assert list(out.index) == [5, 6, 7]
    assert list(out["decision"]) == ["match", "non_match", "non_match"]
    assert out["decision_reason"].tolist() == ["r0", "r1", "r2"]
    assert c.artifacts["usage"]["calls"] == 1 and c.artifacts["usage"]["input_tokens"] > 0
    assert "Pair 0: 'Acme Ltd' vs 'ACME Limited'" in prov.prompts[0]
    assert v.name == "llm" and v.scope == "review"


def test_partial_answer_is_undecided():
    v, _ = make([answer((0, True, None))])
    out, _ = run(v)
    assert list(out["decision"]) == ["match", "unvalidated", "unvalidated"]
    v2, _ = make([answer((0, True, None))], policy=DecisionPolicy(on_missing="non_match"))
    assert list(run(v2)[0]["decision"]) == ["match", "non_match", "non_match"]


def test_malformed_and_fenced_json():
    v, _ = make(["not json at all"])
    out, c = run(v)
    assert set(out["decision"]) == {"unvalidated"}
    assert out["decision_reason"].iloc[0] == "llm_failed: ParseError"
    assert c.artifacts["llm_failures"] == {"failed_batches": 1, "failed_pairs": 3}
    fenced = "Sure:\n```json\n" + json.dumps(answer((0, True, None), (1, True, None), (2, False, None))) + "\n```"
    v2, _ = make([fenced])
    assert list(run(v2)[0]["decision"]) == ["match", "match", "non_match"]


@pytest.mark.parametrize("mode,expected", [("unvalidated", "unvalidated"), ("match", "match"),
                                           ("non_match", "non_match")])
def test_failing_batch_on_failure(mode, expected):
    v, _ = make([RuntimeError("boom")], policy=DecisionPolicy(on_failure=mode))
    out, _ = run(v)
    assert set(out["decision"]) == {expected}
    assert all(r == "llm_failed: RuntimeError" for r in out["decision_reason"])


def test_probability_mode_thresholds():
    pol = DecisionPolicy(mode="probability", match_above=0.7, non_match_below=0.3)
    v, _ = make([answer((0, True, 0.95), (1, True, 0.5), (2, False, 0.1))], policy=pol)
    out, _ = run(v)
    assert list(out["decision"]) == ["match", "unvalidated", "non_match"]
    assert out["decision_prob"].iloc[1] == pytest.approx(0.5)


def test_batching_and_workers():
    pairs = pd.DataFrame({"i1": [0, 0, 2, 2], "i2": [1, 2, 3, 1]})
    resp = lambda m: answer((0, True, None), (1, False, None))  # noqa: E731
    v, prov = make([resp], batch_size=2, n_workers=2)
    out, c = run(v, pairs)
    assert len(prov.calls) == 2 and c.artifacts["usage"]["calls"] == 2
    assert list(out["decision"]) == ["match", "non_match", "match", "non_match"]


def test_cache_hits_and_threshold_tuning(tmp_path):
    cache = f"sqlite:///{(tmp_path / 'c.db').as_posix()}"
    v, prov = make([answer((0, True, 0.9), (1, False, 0.1), (2, True, 0.5))], cache=cache)
    out1, _ = run(v)
    assert len(prov.calls) == 1
    v2, prov2 = make(["unused"], cache=cache,
                     policy=DecisionPolicy(mode="probability", match_above=0.4))
    out2, c2 = run(v2)
    assert len(prov2.calls) == 0 and c2.artifacts["usage"]["calls"] == 0
    assert list(out2["decision"]) == ["match", "non_match", "match"]
    # different instructions -> different key -> a call is made
    v3, prov3 = make([answer((0, True, None))], cache=cache, instructions="strict")
    run(v3)
    assert len(prov3.calls) == 1
    assert len(open_cache(cache)) >= 4


def test_scope_and_empty_and_bad_args():
    v, _ = make(["x"], scope="all")
    assert v.scope == "all"
    out, _ = run(v, PAIRS.iloc[0:0])
    assert out.empty and list(out.columns) == ["decision", "decision_prob", "decision_reason"]
    with pytest.raises(Exception):
        LLMValidator(FakeProvider(), scope="nope", cache=None)


def test_registry_and_metadata_prompt():
    v = registry.build("validator", {"llm": {"provider": {"fake": {"responses": [answer((0, True, None))]}},
                                            "metadata_columns": ["country"], "cache": None}})
    assert isinstance(v, LLMValidator)
    out, _ = run(v)
    assert "country: US vs US" in v.provider.prompts[0]
    assert out["decision"].iloc[0] == "match"


def test_pipeline_integration_shape():
    # engine contract: decision values only match/non_match/unvalidated, float32 prob
    v, _ = make([answer((0, True, 0.9))])
    out, _ = run(v)
    assert set(out["decision"]) <= {"match", "non_match", "unvalidated"}
    assert str(out["decision_prob"].dtype) == "float32"


def test_parsers_direct():
    p = JsonDecisionsParser()
    batch = [{}, {}]
    r = ProviderResponse(json={"decisions": [{"pair_index": 0, "is_duplicate": True},
                                             {"pair_index": 0, "is_duplicate": False},
                                             {"pair_index": 9, "is_duplicate": True},
                                             {"pair_index": 1}]})
    assert [d.pair_index for d in p(r, batch)] == [0]
    with pytest.raises(ParseError):
        p(ProviderResponse(text=None), batch)
    pp = ProbabilitiesParser(positive_label="yes")
    d = pp(ProviderResponse(probabilities={"Yes": 0.8, "no": 0.2}), [{}])[0]
    assert d.confidence == pytest.approx(0.8) and d.is_duplicate
    with pytest.raises(ParseError):
        pp(ProviderResponse(probabilities={"a": 1.0}), [{}])
    assert isinstance(registry.build("parser", {"probabilities": {"positive_label": "dup"}}),
                      ProbabilitiesParser)


def test_probabilities_parser_forces_single_pair_batches():
    prov = FakeProvider([ProviderResponse(probabilities={"yes": 0.9, "no": 0.1})])
    v = LLMValidator(prov, parser={"probabilities": {}}, cache=None, batch_size=10,
                     policy=DecisionPolicy(mode="probability"))
    out, _ = run(v)
    assert len(prov.calls) == 3 and set(out["decision"]) == {"match"}


def test_policy_failure_and_validation():
    assert DecisionPolicy().decide(None)[0] == "unvalidated"
    d = DecisionPolicy().decide(Decision(pair_index=0, is_duplicate=True))
    assert d[0] == "match" and math.isnan(d[1])
    with pytest.raises(Exception):
        DecisionPolicy(on_failure="maybe")
