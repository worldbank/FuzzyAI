"""Public API must only grow: legacy names, module paths and signatures stay (append-only)."""
import importlib
import json
from pathlib import Path

import pytest

import deduplix
from scripts.update_api_snapshot import build

SNAP = json.loads((Path(__file__).parent / "api_snapshot.json").read_text(encoding="utf-8"))


def test_legacy_exports_present():
    for name in SNAP["legacy_exports"]:
        assert name in deduplix.__all__
        assert hasattr(deduplix, name)


@pytest.mark.parametrize("mod", SNAP["modules"])
def test_module_paths_importable(mod):
    importlib.import_module(f"deduplix.{mod}")


@pytest.mark.parametrize("name", sorted(SNAP["signatures"]))
def test_signature_is_prefix_compatible(name):
    old = SNAP["signatures"][name]
    new = build()["signatures"][name]
    # existing parameters keep name, order and default; new ones may only be appended
    assert new[: len(old)] == old, f"{name}: {old} -> {new}"
