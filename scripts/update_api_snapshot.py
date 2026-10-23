"""Regenerate tests/compat/api_snapshot.json. Changes must only append parameters."""
from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODULES = ["core", "matchers", "validators", "consolidation", "llm_consolidation",
           "checkpointing", "exceptions", "validation", "security", "utils"]
LEGACY_NAMES = [
    "DeduplicationPipeline", "DeduplicationResult", "CrossDatasetResult", "FuzzyMatcher",
    "LLMValidator", "RuleBasedValidator", "load_config", "create_sample_config",
    "remove_duplicates_after_deduplication", "simple_remove_duplicates",
]


def sig(obj) -> list:
    try:
        s = inspect.signature(obj)
    except (TypeError, ValueError):
        return []
    return [[p.name, None if p.default is p.empty else repr(p.default)] for p in s.parameters.values()]


def build() -> dict:
    import deduplix

    snap = {"legacy_exports": LEGACY_NAMES, "modules": MODULES, "signatures": {}}
    for mod in MODULES:
        importlib.import_module(f"deduplix.{mod}")
    for name in LEGACY_NAMES:
        obj = getattr(deduplix, name)
        snap["signatures"][name] = sig(obj)
        if inspect.isclass(obj):
            for mname, m in inspect.getmembers(obj, inspect.isfunction):
                if not mname.startswith("_") or mname == "__init__":
                    snap["signatures"][f"{name}.{mname}"] = sig(m)
    for cname, modname in [("ConsolidationConfig", "consolidation"),
                           ("LLMConsolidationConfig", "llm_consolidation"),
                           ("MatchResult", "core"), ("ValidationResult", "core")]:
        obj = getattr(importlib.import_module(f"deduplix.{modname}"), cname)
        snap["signatures"][cname] = sig(obj)
    return snap


if __name__ == "__main__":
    out = ROOT / "tests" / "compat" / "api_snapshot.json"
    out.write_text(json.dumps(build(), indent=1, sort_keys=True), encoding="utf-8")
    print("wrote", out)
