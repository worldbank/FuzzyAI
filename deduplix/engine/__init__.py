"""Staged, pluggable deduplication/linkage engine."""
from . import router  # noqa: F401  (registers 'threshold' and 'rule')
from .config import load_pipeline
from .context import ColumnMap, RunContext
from .hooks import CheckpointHook, LoggingHook, MLflowHook, ProgressHook
from .pipeline import Pipeline, PipelineResult
from .registry import build, list_components, register
from .router import RuleRouter, ThresholdRouter

__all__ = [
    "Pipeline", "PipelineResult", "RunContext", "ColumnMap", "register", "build", "list_components",
    "load_pipeline", "ThresholdRouter", "RuleRouter", "LoggingHook", "ProgressHook",
    "CheckpointHook", "MLflowHook", "EngineMatcher", "as_engine_validator", "as_legacy_validator",
]

from .compat import EngineMatcher, as_engine_validator, as_legacy_validator  # noqa: E402

# importing the component packages registers their normalizers, blockers, comparators, ...
from .. import blocking, cluster, compare, llm, normalize, scoring  # noqa: E402,F401
