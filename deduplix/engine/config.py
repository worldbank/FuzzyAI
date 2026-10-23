"""YAML/dict configuration for :class:`Pipeline` (pydantic v2, ``extra='forbid'``)."""
from __future__ import annotations

import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..exceptions import ConfigurationError
from . import registry
from .pipeline import Pipeline, _is_plain, describe

logger = logging.getLogger("deduplix.config")

Spec = Union[str, Dict[str, Any]]
PYTHON_OBJECT = "<python object>"
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")

    legacy_score_feature: Optional[str] = None
    max_pairs: int = 50_000_000
    undecided: str = "unvalidated"
    random_state: int = 0


class PipelineConfig(BaseModel):
    """Top-level pipeline document. List entries are ``name`` or ``{name: params}``."""

    model_config = ConfigDict(extra="forbid")

    normalize: List[Spec] = Field(default_factory=list)
    blocking: List[Spec] = Field(default_factory=list)
    compare: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    score: Optional[Spec] = None
    route: Optional[Spec] = None
    validate_: List[Spec] = Field(default_factory=list, alias="validate")
    cluster: Optional[Spec] = None
    consolidate: Optional[Spec] = None
    hooks: List[Spec] = Field(default_factory=list)
    options: Options = Field(default_factory=Options)

    @field_validator("normalize", "blocking", "validate_", "hooks", mode="after")
    @classmethod
    def _one_key(cls, v: List[Spec]) -> List[Spec]:
        for s in v:
            if isinstance(s, dict) and len(s) != 1:
                raise ValueError(f"entry must be a name or a one-key mapping, got keys {list(s)}")
            if s == PYTHON_OBJECT or (isinstance(s, dict) and PYTHON_OBJECT in s):
                raise ValueError("config refers to a Python object that cannot be rebuilt from text")
        return v


# ------------------------------------------------------------------ env interpolation
def _interpolate(obj: Any, allow_missing: bool) -> Any:
    if isinstance(obj, str):
        def sub(m: "re.Match[str]") -> str:
            val = os.environ.get(m.group(1))
            if val is None:
                if allow_missing:
                    return m.group(0)
                raise ConfigurationError(f"Environment variable {m.group(1)} is not set",
                                         config_key=m.group(1))
            return val

        return _ENV.sub(sub, obj)
    if isinstance(obj, dict):
        return {k: _interpolate(v, allow_missing) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v, allow_missing) for v in obj]
    return obj


def _walk_providers(obj: Any, allow_missing: bool) -> Any:
    """Substitute ``${VAR}`` only inside ``provider`` blocks (they carry secrets)."""
    if isinstance(obj, dict):
        return {k: (_interpolate(v, allow_missing) if k == "provider" else _walk_providers(v, allow_missing))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_providers(v, allow_missing) for v in obj]
    return obj


# ------------------------------------------------------------------ build / dump
def _comparator_spec(feature: str, body: Dict[str, Any]) -> Dict[str, Any]:
    body = dict(body)
    fn = body.pop("fn", None)
    if fn is None:
        raise ConfigurationError(f"compare.{feature} needs an 'fn'", config_key=f"compare.{feature}")
    registry._load_entry_points()
    if isinstance(fn, str) and fn in registry._REGISTRY["comparator"]:
        return {fn: {**body, "feature": feature}}
    return {"string_sim": {**body, "fn": fn, "feature": feature}}


def load_pipeline(source: Union[str, Path, Dict[str, Any]], allow_missing_env: bool = False) -> Pipeline:
    """Build a :class:`Pipeline` from a YAML path or a dict."""
    if isinstance(source, (str, Path)) and not isinstance(source, dict):
        p = Path(source)
        if not p.exists():
            raise ConfigurationError(f"Pipeline file not found: {p}", config_key="path", config_value=str(p))
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    else:
        raw = dict(source)
    raw = _walk_providers(raw, allow_missing_env)
    try:
        cfg = PipelineConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid pipeline config: {exc}") from exc
    comparators = [_comparator_spec(k, v) for k, v in cfg.compare.items()]
    return Pipeline(
        normalizers=cfg.normalize,
        blockers=cfg.blocking,
        comparators=comparators,
        scorer=cfg.score,
        router=cfg.route,
        validators=cfg.validate_,
        clusterer=cfg.cluster,
        consolidator=cfg.consolidate,
        hooks=cfg.hooks,
        **cfg.options.model_dump(),
    )


def _dump_list(items: List[Any], label: str) -> List[Any]:
    out = []
    for i, x in enumerate(items):
        d = describe(x)
        if isinstance(d, dict) and PYTHON_OBJECT in d:
            warnings.warn(f"{label}[{i}] is a Python object and is written as '{PYTHON_OBJECT}'")
            out.append(PYTHON_OBJECT)
        else:
            out.append(d)
    return out


def _dump_one(x: Any, label: str) -> Any:
    return None if x is None else _dump_list([x], label)[0]


def pipeline_to_config(p: Pipeline) -> Dict[str, Any]:
    """Resolved config; components not built from the registry become ``"<python object>"``."""
    compare: Dict[str, Any] = {}
    for i, c in enumerate(p.comparators):
        spec = registry.spec_of(c)
        if spec is None or not _is_plain(spec["params"]) or "feature" not in spec["params"]:
            warnings.warn(f"compare[{i}] is a Python object and is written as '{PYTHON_OBJECT}'")
            compare[f"comparator_{i}"] = PYTHON_OBJECT
            continue
        params = dict(spec["params"])
        feature = params.pop("feature")
        fn = params.pop("fn") if spec["name"] == "string_sim" else spec["name"]
        compare[feature] = {"fn": fn, **params}
    cfg: Dict[str, Any] = {
        "normalize": _dump_list(p.normalizers, "normalize"),
        "blocking": _dump_list(p.blockers, "blocking"),
        "compare": compare,
        "score": _dump_one(p.scorer, "score"),
        "route": _dump_one(p.router, "route"),
        "validate": _dump_list(p.validators, "validate"),
        "cluster": _dump_one(p.clusterer, "cluster"),
        "consolidate": _dump_one(p.consolidator, "consolidate"),
        "hooks": [h for h in _dump_list(p.hooks, "hooks") if h != "logging"],
        "options": {"legacy_score_feature": p.legacy_score_feature, "max_pairs": p.max_pairs,
                    "undecided": p.undecided, "random_state": p.random_state},
    }
    return {k: v for k, v in cfg.items() if v not in (None, [], {})}


def default_pipeline() -> Pipeline:
    """Default: default_text -> TF-IDF (or brute force) -> token_sort_ratio -> 0.95 / 0.60 router."""
    try:
        from ..blocking import tfidf  # noqa: F401  (registers 'tfidf_char')

        blockers: List[Spec] = ["tfidf_char"]
    except ImportError:
        warnings.warn("scikit-learn / sparse_dot_topn not installed; falling back to brute-force "
                      "blocking. Install deduplix[blocking] for large inputs.")
        blockers = [{"brute_force": {}}]
    return load_pipeline({
        "normalize": ["default_text"],
        "blocking": blockers,
        "compare": {"name": {"fn": "token_sort_ratio", "left": "name_norm"}},
        "score": {"feature": {"name": "name"}},
        "route": {"threshold": {"accept_above": 0.95, "reject_below": 0.60}},
    })


def config_json_schema() -> Dict[str, Any]:
    """JSON schema of the config document (used to generate the docs)."""
    return PipelineConfig.model_json_schema(by_alias=True)
