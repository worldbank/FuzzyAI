"""Consolidators wrapping the legacy rule-based and LLM engines (link mode)."""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from ..engine.registry import register


def _cfg(config: Any, cls: type) -> Any:
    if config is None:
        return cls()
    if isinstance(config, dict):
        return cls(**config)
    return config


@register("consolidator", "rules")
class RulesConsolidator:
    """Priority-rule consolidation (wraps ``ConsolidationEngine``)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config
        self._engine = None

    def __call__(self, matches: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame,
                 ctx: Any) -> pd.DataFrame:
        if self._engine is None:
            from ..consolidation import ConsolidationConfig, ConsolidationEngine

            cfg = _cfg(self.config, ConsolidationConfig)
            if isinstance(self.config, dict) and "enabled" not in self.config:
                cfg.enabled = True
            self._engine = ConsolidationEngine(cfg)
        c = ctx.columns
        return self._engine.consolidate(matches, left, right, c.id, c.rid, c.rname)


@register("consolidator", "llm")
class LLMConsolidator:
    """LLM consolidation (wraps ``LLMConsolidationEngine``); the engine is built on first call."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config
        self._engine = None

    def __call__(self, matches: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame,
                 ctx: Any) -> pd.DataFrame:
        if self._engine is None:
            from ..llm_consolidation import LLMConsolidationConfig, LLMConsolidationEngine

            self._engine = LLMConsolidationEngine(_cfg(self.config, LLMConsolidationConfig))
        c = ctx.columns
        return self._engine.consolidate(matches, left, right, c.id, c.rid, c.rname)
