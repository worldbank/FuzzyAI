"""Run context handed to every stage."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class ColumnMap:
    """Names of the user's columns in the records tables."""

    id: str = "id"
    name: str = "name"
    right_id: Optional[str] = None
    right_name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def rid(self) -> str:
        return self.right_id or self.id

    @property
    def rname(self) -> str:
        return self.right_name or self.name


@dataclass
class RunContext:
    """State shared by the stages of one run.

    Both tables carry a RangeIndex; in dedup mode ``right is left``. Fitted
    objects (blockers, IDF tables, fitted models) live in ``artifacts``.
    """

    left: pd.DataFrame
    right: pd.DataFrame
    mode: str  # "dedup" | "link"
    columns: ColumnMap = field(default_factory=ColumnMap)
    random_state: int = 0
    artifacts: Dict[str, Any] = field(default_factory=dict)
    checkpointer: Any = None
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("deduplix.engine"))
    debug: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("dedup", "link"):
            raise ValueError("mode must be 'dedup' or 'link'")
        self.left = self.left.reset_index(drop=True)
        self.right = self.left if self.mode == "dedup" else self.right.reset_index(drop=True)

    def set_left(self, df: pd.DataFrame) -> None:
        self.left = df
        if self.mode == "dedup":
            self.right = df
