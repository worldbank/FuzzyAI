import warnings

import numpy as np
import pandas as pd
import pytest
import yaml

from deduplix.engine import Pipeline, RuleRouter, ThresholdRouter, load_pipeline, register
from deduplix.engine.config import config_json_schema
from deduplix.exceptions import ConfigurationError

from .conftest import AllPairsBlocker, LenRatio, same_name


def frame(**cols):
    return pd.DataFrame(cols)


def test_threshold_router_bands():
    r = ThresholdRouter(0.95, 0.60)
    out = r(frame(score=[0.95, 0.94, 0.60, 0.59, np.nan]))
    assert out.tolist() == ["accept", "review", "review", "reject", "review"]
    with pytest.raises(ConfigurationError):
        ThresholdRouter(0.5, 0.9)


def test_rule_router_declarative_rules_first():
    pairs = frame(score=[0.99, 0.99, 0.5, 0.7], f_d=["reject", "ok", "review", None])
    r = RuleRouter(
        rules=[{"feature": "f_d", "equals": "reject", "route": "reject"},
               {"feature": "f_d", "equals": "review", "route": "review"},
               {"feature": "f_d", "isna": True, "route": "reject"}],
        then={"threshold": {"accept_above": 0.95, "reject_below": 0.6}})
    assert r(pairs).tolist() == ["reject", "accept", "review", "reject"]


def test_rule_router_ops_and_callable():
    pairs = frame(score=[0.5, 0.5, 0.5], f_x=[1.0, 2.0, 3.0], f_c=["a", "b", "c"])
    r = RuleRouter(rules=[{"feature": "f_x", "gte": 3, "route": "accept"},
                          {"feature": "f_c", "in": ["a"], "route": "reject"},
                          (lambda p: p["f_x"] == 2.0, "accept")])
    assert r(pairs).tolist() == ["reject", "accept", "accept"]
    with pytest.raises(ConfigurationError):
        RuleRouter(rules=[{"feature": "f_x", "gte": 1, "route": "maybe"}])
    with pytest.raises(ConfigurationError):
        RuleRouter(rules=[{"feature": "f_x", "route": "accept"}])(pairs)
    with pytest.raises(ConfigurationError):
        RuleRouter(rules=[{"feature": "nope", "gte": 1, "route": "accept"}])(pairs)


def _register_test_components():
    register("blocker", "t_all")(AllPairsBlocker)
    register("comparator", "t_len")(LenRatio)

    @register("comparator", "t_same")
    def t_same(pairs, left, right, ctx, feature="same"):
        out = same_name(pairs, left, right, ctx)
        return out.rename(columns={"f_same": f"f_{feature}"})


CONFIG = {
    "blocking": ["t_all"],
    "compare": {"same": {"fn": "t_same"}, "len": {"fn": "t_len"}},
    "score": None,
    "route": {"rule": {"rules": [{"feature": "f_same", "equals": 0.0, "route": "reject"}],
                        "then": {"threshold": {"accept_above": 0.9, "reject_below": 0.4}}}},
    "cluster": "components",
    "options": {"max_pairs": 1000, "undecided": "unvalidated"},
}


def test_load_pipeline_runs_and_round_trips(people):
    _register_test_components()

    @register("scorer", "t_mean")
    def t_mean(pairs, ctx):
        return (pairs["f_same"] + pairs["f_len"]) / 2

    cfg = dict(CONFIG, score="t_mean")
    p = load_pipeline(cfg)
    res = p.dedupe(people)
    assert len(res.pairs) == 6 and (res.pairs["decision"] == "match").sum() >= 1
    out = p.to_config()
    assert out["compare"] == {"same": {"fn": "t_same"}, "len": {"fn": "t_len"}}
    assert out["blocking"] == ["t_all"] and out["score"] == "t_mean"
    again = load_pipeline(yaml.safe_load(yaml.safe_dump(out)))
    assert again.to_config() == out


def test_to_config_warns_for_python_objects():
    p = Pipeline(blockers=[AllPairsBlocker()], comparators=[same_name])
    with pytest.warns(UserWarning, match="Python object"):
        cfg = p.to_config()
    assert cfg["compare"]["comparator_0"] == "<python object>"
    with pytest.raises(ConfigurationError):
        load_pipeline({"blocking": ["<python object>"]})


def test_config_rejects_unknown_keys_and_bad_entries():
    with pytest.raises(ConfigurationError):
        load_pipeline({"blokcing": []})
    with pytest.raises(ConfigurationError):
        load_pipeline({"normalize": [{"a": {}, "b": {}}]})
    with pytest.raises(ConfigurationError):
        load_pipeline({"options": {"typo": 1}})
    with pytest.raises(ConfigurationError):
        load_pipeline("does_not_exist.yaml")


def test_env_interpolation_only_in_provider_blocks(monkeypatch):
    @register("validator", "t_llm")
    class V:
        def __init__(self, provider=None, note=None):
            self.provider, self.note = provider, note

    monkeypatch.setenv("T_ENDPOINT", "ep1")
    cfg = {"validate": [{"t_llm": {"provider": {"x": {"endpoint": "${T_ENDPOINT}"}}, "note": "${T_ENDPOINT}"}}]}
    v = load_pipeline(cfg).validators[0]
    assert v.provider == {"x": {"endpoint": "ep1"}} and v.note == "${T_ENDPOINT}"
    monkeypatch.delenv("T_ENDPOINT")
    with pytest.raises(ConfigurationError, match="T_ENDPOINT"):
        load_pipeline(cfg)
    v = load_pipeline(cfg, allow_missing_env=True).validators[0]
    assert v.provider["x"]["endpoint"] == "${T_ENDPOINT}"


def test_yaml_file_and_schema(tmp_path):
    _register_test_components()
    f = tmp_path / "p.yaml"
    f.write_text(yaml.safe_dump({"blocking": ["t_all"], "compare": {"same": {"fn": "t_same"}}}))
    assert len(Pipeline.from_config(str(f)).blockers) == 1
    assert "properties" in config_json_schema()
