import functools
import types

import pytest

from deduplix.engine import registry
from deduplix.engine.registry import build, list_components, register, spec_of
from deduplix.exceptions import ConfigurationError


def test_register_class_function_and_duplicate():
    @register("scorer", "t_cls")
    class S:
        def __init__(self, k=1):
            self.k = k

    @register("scorer", "t_fn")
    def fn(pairs, ctx, k=2):
        return k

    with pytest.raises(ConfigurationError):
        register("scorer", "t_cls")(S)
    register("scorer", "t_cls", overwrite=True)(S)
    assert build("scorer", "t_cls").k == 1
    assert build("scorer", {"t_cls": {"k": 5}}).k == 5
    assert build("scorer", "t_fn") is fn
    p = build("scorer", {"t_fn": {"k": 7}})
    assert isinstance(p, functools.partial) and p(None, None) == 7


def test_build_object_passthrough_and_errors():
    obj = lambda pairs, ctx: None  # noqa: E731
    assert build("scorer", obj) is obj
    with pytest.raises(ConfigurationError):
        build("scorer", "nope")
    with pytest.raises(ConfigurationError):
        build("scorer", {"a": {}, "b": {}})
    with pytest.raises(ConfigurationError):
        build("bogus", "x")


def test_bad_params_raise_configuration_error():
    @register("scorer", "t_strict")
    class S:
        def __init__(self, k=1):
            pass

    with pytest.raises(ConfigurationError):
        build("scorer", {"t_strict": {"unknown": 1}})


def test_constructor_args_are_remembered():
    @register("scorer", "t_spec")
    class S:
        def __init__(self, a=1, b=2):
            pass

    assert spec_of(S(b=5))["params"] == {"b": 5}
    assert spec_of(build("scorer", {"t_spec": {"a": 3}}))["params"] == {"a": 3}


def test_entry_points(monkeypatch):
    def fake_scorer(pairs, ctx):
        return 0.5

    ep = types.SimpleNamespace(name="ep_scorer", load=lambda: fake_scorer)
    monkeypatch.setattr(registry, "_ENTRY_POINTS_LOADED", False)
    monkeypatch.setattr(registry, "_iter_entry_points",
                        lambda group: [ep] if group == "deduplix.scorers" else [])
    assert build("scorer", "ep_scorer") is fake_scorer


def test_list_components():
    @register("scorer", "t_listed")
    class S:
        """First line.

        More."""

        def __init__(self, alpha=1):
            pass

    row = [r for r in list_components("scorer") if r["name"] == "t_listed"][0]
    assert row["doc"] == "First line." and row["parameters"] == ["alpha"] and row["kind"] == "scorer"
