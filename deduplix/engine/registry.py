"""Component registry: register by name, build from a string, a one-key dict, or an object.

Conventions
-----------
* A registered **class** is instantiated with ``**params`` (no params -> defaults).
* A registered **function** with params becomes ``functools.partial(fn, **params)``;
  without params it is used as is.
* Any other object that is callable or has the protocol's methods is returned unchanged.
* Entry points in group ``deduplix.<kind>s`` are loaded lazily on the first ``build``.
"""
from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

from ..exceptions import ConfigurationError

logger = logging.getLogger("deduplix.registry")

KINDS = (
    "normalizer", "blocker", "comparator", "scorer", "router", "validator",
    "clusterer", "consolidator", "provider", "parser", "hook", "encoder",
)

_REGISTRY: Dict[str, Dict[str, Callable]] = {k: {} for k in KINDS}
_ENTRY_POINTS_LOADED = False
SPEC_ATTR = "__deduplix_spec__"


def _check_kind(kind: str) -> None:
    if kind not in _REGISTRY:
        raise ConfigurationError(f"Unknown component kind '{kind}'. Choose from {list(KINDS)}")


def register(kind: str, name: str, *, overwrite: bool = False) -> Callable:
    """Decorator registering a class or function as ``kind``/``name``."""
    _check_kind(kind)

    def deco(obj: Callable) -> Callable:
        if name in _REGISTRY[kind] and not overwrite:
            raise ConfigurationError(f"{kind} '{name}' is already registered; use overwrite=True")
        _REGISTRY[kind][name] = obj
        if inspect.isclass(obj):
            _record_init_spec(obj, kind, name)
        return obj

    return deco


def _record_init_spec(cls: type, kind: str, name: str) -> None:
    """Make instances remember their constructor arguments so they round-trip to YAML."""
    orig = cls.__init__
    if getattr(orig, "_deduplix_wrapped", False):
        return
    try:
        sig = inspect.signature(orig)
    except (TypeError, ValueError):  # pragma: no cover
        return

    @functools.wraps(orig)
    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        if type(self) is cls and SPEC_ATTR not in getattr(self, "__dict__", {}):
            try:
                bound = sig.bind(self, *args, **kwargs)
                params = {k: v for k, v in bound.arguments.items() if k != "self"}
                for pname, p in sig.parameters.items():  # flatten **kwargs
                    if p.kind is p.VAR_KEYWORD and pname in params:
                        params.update(params.pop(pname))
                    elif p.kind is p.VAR_POSITIONAL:
                        params.pop(pname, None)
                setattr(self, SPEC_ATTR, {"kind": kind, "name": name, "params": params})
            except (TypeError, AttributeError):
                pass

    __init__._deduplix_wrapped = True  # type: ignore[attr-defined]
    cls.__init__ = __init__  # type: ignore[method-assign]


def unregister(kind: str, name: str) -> None:
    _check_kind(kind)
    _REGISTRY[kind].pop(name, None)


def _iter_entry_points(group: str):
    from importlib import metadata

    eps = metadata.entry_points()
    if hasattr(eps, "select"):
        return list(eps.select(group=group))
    return list(eps.get(group, []))  # Python 3.9


def _load_entry_points() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    for kind in KINDS:
        for ep in _iter_entry_points(f"deduplix.{kind}s"):
            if ep.name in _REGISTRY[kind]:
                continue
            try:
                _REGISTRY[kind][ep.name] = ep.load()
            except Exception as exc:  # pragma: no cover - depends on third-party packages
                logger.warning("Could not load entry point %s/%s: %s", kind, ep.name, exc)


def _lookup(kind: str, name: str) -> Callable:
    if name not in _REGISTRY[kind]:
        _load_entry_points()
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        raise ConfigurationError(
            f"Unknown {kind} '{name}'. Registered: {sorted(_REGISTRY[kind])}",
            config_key=kind, config_value=name,
        ) from None


def _tag(obj: Any, kind: str, name: str, params: Dict[str, Any]) -> Any:
    try:
        setattr(obj, SPEC_ATTR, {"kind": kind, "name": name, "params": dict(params)})
    except (AttributeError, TypeError):  # builtins / slotted objects
        pass
    return obj


def build(kind: str, spec: Any) -> Any:
    """Turn ``spec`` into a component of ``kind``."""
    _check_kind(kind)
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, dict):
        if len(spec) != 1:
            raise ConfigurationError(
                f"A {kind} spec must be a name or a one-key mapping {{name: params}}, got keys {list(spec)}")
        name, params = next(iter(spec.items()))
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ConfigurationError(f"Parameters of {kind} '{name}' must be a mapping")
        if True in params:  # YAML 1.1 parses an unquoted ``on:`` key as boolean True
            params = {("on" if k is True else k): v for k, v in params.items()}
    else:
        return spec  # object satisfying the protocol: returned as is
    target = _lookup(kind, name)
    try:
        if inspect.isclass(target):
            return _tag(target(**params), kind, name, params)
        if params:
            return _tag(functools.partial(target, **params), kind, name, params)
        return _tag(target, kind, name, {})
    except TypeError as exc:
        raise ConfigurationError(f"Cannot build {kind} '{name}' with {params}: {exc}") from exc


def spec_of(obj: Any) -> Optional[Dict[str, Any]]:
    """The registry spec an object was built from, or None."""
    return getattr(obj, SPEC_ATTR, None)


def _first_line(obj: Callable) -> str:
    doc = inspect.getdoc(obj) or ""
    return doc.strip().splitlines()[0] if doc.strip() else ""


def list_components(kind: Optional[str] = None) -> List[Dict[str, Any]]:
    """Name, kind, first docstring line and parameters of every registered component."""
    _load_entry_points()
    kinds = [kind] if kind else list(KINDS)
    rows: List[Dict[str, Any]] = []
    for k in kinds:
        _check_kind(k)
        for name, obj in sorted(_REGISTRY[k].items()):
            target = obj.__init__ if inspect.isclass(obj) else obj
            try:
                params = [p for p in inspect.signature(target).parameters if p != "self"]
            except (TypeError, ValueError):
                params = []
            rows.append({"name": name, "kind": k, "doc": _first_line(obj), "parameters": params})
    return rows
