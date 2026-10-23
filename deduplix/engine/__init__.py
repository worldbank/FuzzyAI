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
    "CheckpointHook", "MLflowHook",
]

from .. import cluster  # noqa: E402,F401  (registers clusterers)
