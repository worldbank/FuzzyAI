"""Built-in hooks: logging, progress, checkpointing, MLflow."""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Optional

import pandas as pd

from .protocols import Hook
from .registry import register

logger = logging.getLogger("deduplix.hooks")

PAIR_STAGES = ("block", "compare", "score", "route", "validate")


@register("hook", "logging")
class LoggingHook(Hook):
    """Log rows in/out and seconds per stage."""

    def __init__(self, level: int = logging.INFO):
        self.level = level
        self._log = logging.getLogger("deduplix.engine")

    def on_run_start(self, ctx) -> None:
        self._log.log(self.level, "run start: mode=%s left=%d right=%d",
                      ctx.mode, len(ctx.left), len(ctx.right))

    def on_stage_end(self, stage, ctx, pairs, seconds) -> None:
        n = "-" if pairs is None else len(pairs)
        self._log.log(self.level, "stage %-10s rows=%s %.2fs", stage, n, seconds)

    def on_error(self, stage, ctx, exc) -> None:
        self._log.error("stage %s failed: %s: %s", stage, type(exc).__name__, exc)


@register("hook", "progress")
class ProgressHook(Hook):
    """tqdm bar advancing one step per stage."""

    def __init__(self, total_stages: int = 8):
        from tqdm import tqdm

        self._bar = tqdm(total=total_stages, desc="deduplix", unit="stage")

    def on_stage_end(self, stage, ctx, pairs, seconds) -> None:
        self._bar.set_postfix_str(stage)
        self._bar.update(1)

    def on_run_end(self, ctx, result) -> None:
        self._bar.close()


@register("hook", "checkpoint")
class CheckpointHook(Hook):
    """Cache pair-producing stages through a FileCheckpointer / DatabaseCheckpointer.

    The key is sha256 over the stage name, the resolved config of that stage and every
    upstream stage, and the data hash, so changing one threshold invalidates only that
    stage and the ones after it.
    """

    def __init__(self, checkpointer: Any):
        self.checkpointer = checkpointer

    @staticmethod
    def _key(stage: str, ctx) -> Optional[str]:
        configs: Dict[str, Any] = ctx.artifacts.get("stage_configs")
        if configs is None or stage not in configs:
            return None
        upstream = []
        for name, cfg in configs.items():
            upstream.append([name, cfg])
            if name == stage:
                break
        payload = json.dumps([upstream, ctx.artifacts.get("data_hash", "")], sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def on_stage_start(self, stage, ctx, pairs):
        if stage not in PAIR_STAGES:
            return None
        key = self._key(stage, ctx)
        if key is None:
            return None
        cached = self.checkpointer.load(f"engine_{stage}", key[:32])
        if cached is not None:
            logger.info("stage %s restored from checkpoint", stage)
        return cached

    def on_stage_end(self, stage, ctx, pairs, seconds) -> None:
        if stage not in PAIR_STAGES or pairs is None:
            return
        key = self._key(stage, ctx)
        if key is not None and not ctx.artifacts.get(f"restored.{stage}"):
            self.checkpointer.save(pairs, f"engine_{stage}", key[:32])


@register("hook", "mlflow")
class MLflowHook(Hook):
    """Log flattened config as params, stats as metrics and a 1,000-row pairs sample."""

    def __init__(self, experiment: Optional[str] = None):
        try:
            import mlflow  # noqa: F401
        except ImportError as exc:
            raise ImportError("MLflowHook needs mlflow: pip install mlflow") from exc
        self.experiment = experiment

    @staticmethod
    def _flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if isinstance(d, dict):
            for k, v in d.items():
                out.update(MLflowHook._flatten(v, f"{prefix}{k}."))
        elif isinstance(d, (list, tuple)):
            for i, v in enumerate(d):
                out.update(MLflowHook._flatten(v, f"{prefix}{i}."))
        else:
            out[prefix.rstrip(".")] = d
        return out

    def on_run_end(self, ctx, result) -> None:
        import mlflow

        if self.experiment:
            mlflow.set_experiment(self.experiment)
        with mlflow.start_run():
            for k, v in self._flatten(result.config).items():
                mlflow.log_param(k[:250], str(v)[:250])
            for st in result.stats.get("stages", []):
                mlflow.log_metric(f"{st['stage']}_seconds", st["seconds"])
                if st.get("rows_out") is not None:
                    mlflow.log_metric(f"{st['stage']}_rows", st["rows_out"])
            sample = result.pairs.head(1000)
            mlflow.log_table(sample, "pairs_sample.json") if hasattr(mlflow, "log_table") else None
