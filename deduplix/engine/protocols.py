"""Stage protocols. Plain functions with these signatures are valid components."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from .context import RunContext


@runtime_checkable
class Normalizer(Protocol):
    def __call__(self, records: pd.DataFrame, ctx: "RunContext") -> pd.DataFrame:
        """Return ``records`` plus new columns (originals are never overwritten)."""


@runtime_checkable
class Blocker(Protocol):
    name: str

    def fit(self, right: pd.DataFrame, ctx: "RunContext") -> "Blocker": ...

    def candidates(self, left: pd.DataFrame, ctx: "RunContext") -> pd.DataFrame:
        """Return columns ``i1``, ``i2`` and optionally ``score``."""


@runtime_checkable
class Comparator(Protocol):
    def __call__(self, pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame,
                 ctx: "RunContext") -> pd.DataFrame:
        """Return only the new ``f_`` columns, with ``pairs``' index."""


class Scorer(Protocol):
    def __call__(self, pairs: pd.DataFrame, ctx: "RunContext") -> pd.Series:
        """Float scores in [0, 1]."""


class Router(Protocol):
    def __call__(self, pairs: pd.DataFrame, ctx: "RunContext") -> pd.Series:
        """Route (accept, review, reject) per pair."""


class Validator(Protocol):
    name: str
    scope: Literal["review", "all", "accepted"]

    def __call__(self, pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame,
                 ctx: "RunContext") -> pd.DataFrame:
        """Return ``decision``, ``decision_prob``, ``decision_reason`` for the rows it decided."""


class Clusterer(Protocol):
    def __call__(self, pairs: pd.DataFrame, n_records: int, ctx: "RunContext") -> np.ndarray:
        """Cluster id per record."""


class Consolidator(Protocol):
    def __call__(self, matches: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame,
                 ctx: "RunContext") -> pd.DataFrame:
        """Link mode: reduce the legacy cross-match frame to the chosen matches."""


class Hook:
    """Base class with no-op callbacks; subclass or duck-type."""

    def on_run_start(self, ctx: "RunContext") -> None: ...

    def on_stage_start(self, stage: str, ctx: "RunContext",
                       pairs: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Return a non-None frame to supply a cached result and skip the stage."""
        return None

    def on_stage_end(self, stage: str, ctx: "RunContext", pairs: Optional[pd.DataFrame],
                     seconds: float) -> None: ...

    def on_run_end(self, ctx: "RunContext", result: Any) -> None: ...

    def on_error(self, stage: str, ctx: "RunContext", exc: BaseException) -> None: ...
