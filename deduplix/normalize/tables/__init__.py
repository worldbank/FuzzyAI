"""YAML lookup tables for normalization and name parsing."""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from typing import Any, Union

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=None)
def _read(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_table(table: Union[str, os.PathLike, dict, list]) -> Any:
    """Load a table by packaged name (``"legal_forms"``/``"legal_forms.yaml"``), file path, or
    pass an already-loaded dict/list through. Always returns a fresh copy."""
    if isinstance(table, (dict, list)):
        return copy.deepcopy(table)
    name = os.fspath(table)
    candidates = [name, os.path.join(_HERE, name), os.path.join(_HERE, name + ".yaml")]
    for cand in candidates:
        if os.path.isfile(cand):
            return copy.deepcopy(_read(os.path.abspath(cand)))
    raise FileNotFoundError(f"Table '{name}' not found (looked in {candidates})")
