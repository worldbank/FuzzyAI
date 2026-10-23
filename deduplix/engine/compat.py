"""Bridges between the 0.1.0 API and the engine: EngineMatcher and validator adapters."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core import Matcher, MatchResult, ValidationResult
from . import schema as S
from .context import ColumnMap, RunContext
from .pipeline import Pipeline

_POS = "_pair_index"


class EngineMatcher(Matcher):
    """Legacy ``Matcher`` interface backed by an engine :class:`Pipeline`.

    Runs normalize -> block -> compare -> score -> route and reports every pair that was not
    rejected as a potential duplicate, so a legacy validator can still follow::

        DeduplicationPipeline(matcher=EngineMatcher("pipeline.yaml"))
    """

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline if isinstance(pipeline, Pipeline) else Pipeline.from_config(pipeline)

    def _candidates(self, ctx: RunContext) -> pd.DataFrame:
        res = self.pipeline._run(ctx, stop_after="route")
        pairs = res.pairs
        pairs = pairs[pairs[S.ROUTE].astype(object) != S.REJECT]
        return pairs, res

    def find_matches(self, df: pd.DataFrame, **kwargs) -> MatchResult:
        ctx = RunContext(left=df, right=df, mode="dedup", columns=ColumnMap("id", "name"),
                         random_state=self.pipeline.random_state)
        pairs, res = self._candidates(ctx)
        legacy = S.to_legacy_pairs(pairs, res.records, res.records, "id", "name",
                                   legacy_score_feature=self.pipeline.legacy_score_feature)
        legacy = legacy.sort_values("similarity_score", ascending=False, kind="stable").reset_index(drop=True)
        return MatchResult(pairs=legacy, metadata={"matching_type": "engine", "matches_found": len(legacy),
                                                   "stats": res.stats})

    def find_cross_matches(self, df1: pd.DataFrame, df2: pd.DataFrame, id_column1: str = "id",
                           name_column1: str = "name", id_column2: str = "id", name_column2: str = "name",
                           name_columns1: Optional[List[str]] = None,
                           name_columns2: Optional[List[str]] = None, **kwargs) -> MatchResult:
        cols1 = name_columns1 or [name_column1]
        cols2 = name_columns2 or [name_column2]
        frames: List[pd.DataFrame] = []
        for c1 in cols1:
            for c2 in cols2:
                if c1 not in df1.columns or c2 not in df2.columns:
                    continue
                ctx = RunContext(left=df1, right=df2, mode="link",
                                 columns=ColumnMap(id_column1, c1, id_column2, c2),
                                 random_state=self.pipeline.random_state)
                pairs, res = self._candidates(ctx)
                legacy = S.to_legacy_pairs(pairs, res.records, res.right_records, id_column1, c1,
                                           id_column2, c2,
                                           legacy_score_feature=self.pipeline.legacy_score_feature)
                legacy["matched_column1"], legacy["matched_column2"] = c1, c2
                legacy["source1"], legacy["source2"] = "df1", "df2"
                frames.append(legacy)
        cols = ["id1", "id2", "name1", "name2", "similarity_score",
                "matched_column1", "matched_column2", "source1", "source2"]
        if not frames:
            return MatchResult(pairs=pd.DataFrame(columns=cols), metadata={"matching_type": "engine_cross"})
        out = (pd.concat(frames, ignore_index=True)
               .sort_values("similarity_score", ascending=False, kind="stable")
               .drop_duplicates(subset=["id1", "id2"]).reset_index(drop=True))
        return MatchResult(pairs=out, metadata={"matching_type": "engine_cross", "matches_found": len(out)})


# ---------------------------------------------------------------------- validator adapters
class _LegacyAsEngine:
    """Engine validator calling a legacy ``validate(MatchResult, original_df=...)``."""

    def __init__(self, legacy: Any, scope: str = "review"):
        self.legacy, self.scope = legacy, scope
        self.name = type(legacy).__name__

    def __call__(self, pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame, ctx: RunContext):
        c = ctx.columns
        legacy_pairs = S.to_legacy_pairs(pairs, left, right, c.id, c.name, c.rid, c.rname,
                                         ctx.artifacts.get("legacy_score_feature"))
        legacy_pairs[_POS] = pairs.index.to_numpy()
        original = left.rename(columns={c.id: "id", c.name: "name"})
        result = self.legacy.validate(MatchResult(pairs=legacy_pairs), original_df=original)
        out = pd.DataFrame(index=pairs.index, columns=[S.DECISION, S.DECISION_PROB, S.DECISION_REASON],
                           dtype=object)
        for frame, label in ((result.validated_pairs, S.MATCH), (result.removed_pairs, S.NON_MATCH)):
            if frame is None or frame.empty:
                continue
            idx = frame[_POS].to_numpy()
            out.loc[idx, S.DECISION] = label
            if "validation_reason" in frame.columns:
                out.loc[idx, S.DECISION_REASON] = frame["validation_reason"].to_numpy()
        out[S.DECISION_PROB] = np.nan
        return out


def as_engine_validator(legacy: Any, scope: str = "review") -> _LegacyAsEngine:
    """Wrap a 0.1.0 validator (``.validate``) so a :class:`Pipeline` can use it."""
    return _LegacyAsEngine(legacy, scope)


class _EngineAsLegacy:
    """Legacy-style validator (``.validate``) around an engine validator."""

    def __init__(self, engine_validator: Any):
        self.engine_validator = engine_validator

    def validate(self, match_result: MatchResult, original_df: Optional[pd.DataFrame] = None,
                 **kwargs) -> ValidationResult:
        legacy = match_result.pairs
        if legacy.empty:
            return ValidationResult(validated_pairs=pd.DataFrame(), removed_pairs=pd.DataFrame(),
                                    metadata={"message": "No pairs to validate"})
        if original_df is None:
            names = pd.concat([
                pd.DataFrame({"id": legacy["id1"], "name": legacy["name1"]}),
                pd.DataFrame({"id": legacy["id2"], "name": legacy["name2"]})]).drop_duplicates("id")
            original_df = names.reset_index(drop=True)
        original_df = original_df.reset_index(drop=True)
        pos = pd.Series(np.arange(len(original_df)), index=original_df["id"]).groupby(level=0).first()
        n = len(original_df)
        pairs = pd.DataFrame({
            S.I1: pos.reindex(legacy["id1"]).to_numpy().astype(np.int64),
            S.I2: pos.reindex(legacy["id2"]).to_numpy().astype(np.int64),
            S.SCORE: (legacy["similarity_score"].to_numpy(dtype=float) / 100.0).astype(np.float32),
        }, index=legacy.index)
        pairs[S.ROUTE] = pd.Categorical([S.REVIEW] * len(pairs), categories=list(S.ROUTES))
        pairs[S.DECISION] = None
        ctx = RunContext(left=original_df, right=original_df, mode="dedup", columns=ColumnMap("id", "name"))
        res = self.engine_validator(pairs, ctx.left, ctx.right, ctx)
        decision = res[S.DECISION].reindex(legacy.index)
        reason = (res[S.DECISION_REASON].reindex(legacy.index) if S.DECISION_REASON in res.columns
                  else pd.Series("", index=legacy.index))
        out = legacy.copy()
        out["validation_reason"] = reason.fillna("").to_numpy()
        removed = (decision == S.NON_MATCH).to_numpy()
        return ValidationResult(
            validated_pairs=out[~removed].reset_index(drop=True),
            removed_pairs=out[removed].reset_index(drop=True),
            metadata={"validator": getattr(self.engine_validator, "name", "engine"),
                      "validated_count": int((~removed).sum()), "removed_count": int(removed.sum())})


def as_legacy_validator(engine_validator: Any) -> _EngineAsLegacy:
    """Expose an engine validator through the legacy ``.validate`` interface."""
    return _EngineAsLegacy(engine_validator)


def is_engine_validator(obj: Any) -> bool:
    return callable(obj) and not hasattr(obj, "validate")
