"""The staged, pluggable pipeline and its result object."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..exceptions import ConfigurationError, DataValidationError
from . import schema as S
from .context import ColumnMap, RunContext
from .hooks import PAIR_STAGES, LoggingHook
from .registry import build, spec_of

logger = logging.getLogger("deduplix.pipeline")

STAGES = ("normalize", "block", "compare", "score", "route", "validate", "cluster", "consolidate")
_UNDECIDED_VALUES = (S.UNVALIDATED, S.MATCH, S.NON_MATCH)


def _is_plain(obj: Any) -> bool:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return True
    if isinstance(obj, (list, tuple)):
        return all(_is_plain(x) for x in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, str) and _is_plain(v) for k, v in obj.items())
    return False


def describe(obj: Any) -> Any:
    """Stable, JSON-able description of a component (used for config and cache keys)."""
    spec = spec_of(obj)
    if spec is not None and _is_plain(spec["params"]):
        return {spec["name"]: spec["params"]} if spec["params"] else spec["name"]
    mod = getattr(obj, "__module__", type(obj).__module__)
    qual = getattr(obj, "__qualname__", type(obj).__qualname__)
    attrs: Dict[str, Any] = {}
    if not hasattr(obj, "__qualname__") and hasattr(obj, "__dict__"):  # an instance, not a function
        attrs = {k: v for k, v in vars(obj).items() if not k.startswith("_") and _is_plain(v)}
    return {"<python object>": f"{mod}.{qual}", **({"attrs": attrs} if attrs else {})}


@dataclass
class PipelineResult:
    """Everything a run produced; ``pairs`` is the single source of truth for decisions."""

    records: pd.DataFrame
    pairs: pd.DataFrame
    stats: Dict[str, Any]
    config: Dict[str, Any]
    mode: str
    columns: ColumnMap
    right_records: Optional[pd.DataFrame] = None
    legacy_score_feature: Optional[str] = None
    consolidated: Optional[pd.DataFrame] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def matches(self) -> pd.DataFrame:
        return self.pairs[self.pairs[S.DECISION] == S.MATCH]

    @property
    def review(self) -> pd.DataFrame:
        """Pairs no stage could decide."""
        return self.pairs[self.pairs[S.DECISION] == S.UNVALIDATED]

    @property
    def clusters(self) -> pd.DataFrame:
        if S.CLUSTER_ID not in self.records.columns:
            return pd.DataFrame({S.CLUSTER_ID: pd.Series(dtype="int64"), "size": pd.Series(dtype="int64")})
        sizes = self.records[S.CLUSTER_ID].value_counts().rename_axis(S.CLUSTER_ID).reset_index(name="size")
        return sizes.sort_values(S.CLUSTER_ID).reset_index(drop=True)

    def explain(self, id1: Any, id2: Any) -> pd.Series:
        """One pair with every column, looked up by the user's record ids."""
        c = self.columns
        left, right = self.records, (self.right_records if self.right_records is not None else self.records)
        p1 = np.flatnonzero(left[c.id].to_numpy() == id1)
        p2 = np.flatnonzero(right[c.rid].to_numpy() == id2)
        for a in p1:
            for b in p2:
                for x, y in ((a, b), (b, a)) if self.mode == "dedup" else ((a, b),):
                    hit = self.pairs[(self.pairs[S.I1] == x) & (self.pairs[S.I2] == y)]
                    if len(hit):
                        row = hit.iloc[0].copy()
                        row["id1"], row["id2"] = left[c.id].iloc[x], right[c.rid].iloc[y]
                        row["name1"], row["name2"] = left[c.name].iloc[x], right[c.rname].iloc[y]
                        return row
        raise KeyError(f"No candidate pair for ({id1!r}, {id2!r})")

    def save(self, path: str) -> Path:
        """Write pairs.parquet, records.parquet, pipeline.yaml and stats.json."""
        import yaml

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.pairs.to_parquet(out / "pairs.parquet")
        self.records.to_parquet(out / "records.parquet")
        if self.right_records is not None:
            self.right_records.to_parquet(out / "right_records.parquet")
        (out / "pipeline.yaml").write_text(yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8")
        (out / "stats.json").write_text(json.dumps(self.stats, indent=2, default=str), encoding="utf-8")
        meta = {"mode": self.mode, "id": self.columns.id, "name": self.columns.name,
                "right_id": self.columns.right_id, "right_name": self.columns.right_name,
                "legacy_score_feature": self.legacy_score_feature}
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str) -> "PipelineResult":
        """Read a directory written by :meth:`save`."""
        import yaml

        d = Path(path)
        if not (d / "meta.json").exists() or not (d / "pairs.parquet").exists():
            raise FileNotFoundError(f"{d} does not contain engine results (pairs.parquet / meta.json)")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        right = d / "right_records.parquet"
        cfg_p, stats_p = d / "pipeline.yaml", d / "stats.json"
        return cls(
            records=pd.read_parquet(d / "records.parquet"),
            pairs=pd.read_parquet(d / "pairs.parquet"),
            stats=json.loads(stats_p.read_text(encoding="utf-8")) if stats_p.exists() else {},
            config=(yaml.safe_load(cfg_p.read_text(encoding="utf-8")) or {}) if cfg_p.exists() else {},
            mode=meta["mode"],
            columns=ColumnMap(id=meta["id"], name=meta["name"], right_id=meta.get("right_id"),
                              right_name=meta.get("right_name")),
            right_records=pd.read_parquet(right) if right.exists() else None,
            legacy_score_feature=meta.get("legacy_score_feature"),
        )

    def to_legacy(self):
        """Convert to ``DeduplicationResult`` (dedup) or ``CrossDatasetResult`` (link)."""
        from ..core import CrossDatasetResult, DeduplicationResult

        c = self.columns
        m = self.matches
        if self.mode == "dedup":
            left = self.records
            dup = S.to_legacy_pairs(m, left, left, c.id, c.name, legacy_score_feature=self.legacy_score_feature)
            dup = dup.sort_values("similarity_score", ascending=False, kind="stable").reset_index(drop=True)
            labels = left[S.CLUSTER_ID].to_numpy()
            sizes = pd.Series(labels).map(pd.Series(labels).value_counts()).to_numpy()
            first_pos = pd.Series(np.arange(len(labels))).groupby(labels).transform("min").to_numpy()
            group = np.zeros(len(labels), dtype=np.int64)
            multi = sizes > 1
            if multi.any():
                order = np.unique(first_pos[multi])  # sorted by smallest row position
                group[multi] = np.searchsorted(order, first_pos[multi]) + 1
            groups = pd.DataFrame({"entity_id": left[c.id].to_numpy(),
                                   "entity_name": left[c.name].to_numpy(), "group_id": group})
            n_groups = int(group.max()) if len(group) else 0
            in_groups = int(multi.sum())
            stats = {
                "total_entities": len(left),
                "duplicate_pairs": len(dup),
                "duplicate_groups": n_groups,
                "entities_with_duplicates": in_groups,
                "singleton_entities": len(left) - in_groups,
                "average_group_size": in_groups / n_groups if n_groups else 0,
                "largest_group_size": int(sizes[multi].max()) if in_groups else 0,
            }
            return DeduplicationResult(entity_groups=groups, duplicate_pairs=dup, statistics=stats)
        left, right = self.records, self.right_records
        cross = S.to_legacy_cross(m, left, right, c.id, c.name, c.rid, c.rname,
                                  legacy_score_feature=self.legacy_score_feature)
        stats = {
            "df1_total": len(left), "df2_total": len(right), "cross_matches": len(cross),
            "df1_matched_entities": cross["df1_id"].nunique(),
            "df2_matched_entities": cross["df2_id"].nunique(),
            "df1_unmatched": len(left) - cross["df1_id"].nunique(),
            "df2_unmatched": len(right) - cross["df2_id"].nunique(),
            "avg_similarity": cross["similarity_score"].mean() if len(cross) else 0,
        }
        return CrossDatasetResult(
            cross_matches=cross,
            df1_metadata={"total": len(left), "id_col": c.id, "name_col": c.name},
            df2_metadata={"total": len(right), "id_col": c.rid, "name_col": c.rname},
            statistics=stats,
        )


class Pipeline:
    """Normalize -> block -> compare -> score -> route -> validate -> cluster -> consolidate.

    Every argument accepts a registry name, a ``{name: params}`` mapping, a function or an
    object. ``blockers``, ``comparators``, ``normalizers``, ``validators`` and ``hooks`` are
    sequences; the rest are single components.
    """

    def __init__(
        self,
        normalizers: Sequence[Any] = (),
        blockers: Sequence[Any] = (),
        comparators: Sequence[Any] = (),
        scorer: Any = None,
        router: Any = None,
        validators: Sequence[Any] = (),
        clusterer: Any = None,
        consolidator: Any = None,
        hooks: Sequence[Any] = (),
        *,
        legacy_score_feature: Optional[str] = None,
        max_pairs: int = 50_000_000,
        undecided: str = S.UNVALIDATED,
        random_state: int = 0,
    ):
        if undecided not in _UNDECIDED_VALUES:
            raise ConfigurationError(f"undecided must be one of {_UNDECIDED_VALUES}", config_key="undecided",
                                     config_value=undecided)
        self.normalizers = [build("normalizer", s) for s in normalizers]
        self.blockers = [build("blocker", s) for s in blockers]
        self.comparators = [build("comparator", s) for s in comparators]
        self.scorer = build("scorer", scorer) if scorer is not None else None
        self.router = build("router", router) if router is not None else None
        self.validators = [build("validator", s) for s in validators]
        self.clusterer = build("clusterer", clusterer) if clusterer is not None else None
        self.consolidator = build("consolidator", consolidator) if consolidator is not None else None
        hooks = [build("hook", s) for s in hooks]
        if not any(isinstance(h, LoggingHook) for h in hooks):
            hooks.insert(0, LoggingHook())
        self.hooks = hooks
        self.legacy_score_feature = legacy_score_feature
        self.max_pairs = max_pairs
        self.undecided = undecided
        self.random_state = random_state

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_config(cls, path_or_dict: Any, **kwargs) -> "Pipeline":
        from .config import load_pipeline

        return load_pipeline(path_or_dict, **kwargs)

    @classmethod
    def default(cls) -> "Pipeline":
        """Name-similarity pipeline: TF-IDF blocking when installed, else brute force."""
        from .config import default_pipeline

        return default_pipeline()

    def to_config(self) -> Dict[str, Any]:
        from .config import pipeline_to_config

        return pipeline_to_config(self)

    # ------------------------------------------------------------------ public run API
    def dedupe(self, df: pd.DataFrame, id_column: str = "id", name_column: str = "name",
               **columns: Any) -> PipelineResult:
        self._check_columns(df, [id_column, name_column])
        cmap = ColumnMap(id=id_column, name=name_column, extra=columns)
        ctx = RunContext(left=df, right=df, mode="dedup", columns=cmap, random_state=self.random_state)
        return self._run(ctx)

    def link(self, left: pd.DataFrame, right: pd.DataFrame, id_column: str = "id", name_column: str = "name",
             right_id_column: Optional[str] = None, right_name_column: Optional[str] = None,
             **columns: Any) -> PipelineResult:
        self._check_columns(left, [id_column, name_column])
        self._check_columns(right, [right_id_column or id_column, right_name_column or name_column])
        cmap = ColumnMap(id=id_column, name=name_column, right_id=right_id_column,
                         right_name=right_name_column, extra=columns)
        ctx = RunContext(left=left, right=right, mode="link", columns=cmap, random_state=self.random_state)
        return self._run(ctx)

    @staticmethod
    def _check_columns(df: pd.DataFrame, cols: List[str]) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise DataValidationError(f"Required columns not found: {missing}",
                                      validation_errors=[f"missing column {c}" for c in missing])

    # ------------------------------------------------------------------ execution
    def stage_configs(self) -> Dict[str, Any]:
        """Resolved config per stage, in execution order (cache keys, ``to_config``)."""
        return {
            "normalize": [describe(x) for x in self.normalizers],
            "block": [describe(x) for x in self.blockers],
            "compare": [describe(x) for x in self.comparators],
            "score": describe(self.scorer) if self.scorer is not None else None,
            "route": describe(self.router) if self.router is not None else None,
            "validate": {"validators": [describe(x) for x in self.validators], "undecided": self.undecided},
            "cluster": describe(self.clusterer) if self.clusterer is not None else "components",
            "consolidate": describe(self.consolidator) if self.consolidator is not None else None,
        }

    @staticmethod
    def _data_hash(ctx: RunContext) -> str:
        parts = []
        for df, cols in ((ctx.left, [ctx.columns.id, ctx.columns.name]),
                         (ctx.right, [ctx.columns.rid, ctx.columns.rname])):
            h = pd.util.hash_pandas_object(df[cols].astype(str), index=False).sum()
            parts.append(f"{df.shape[0]}_{df.shape[1]}_{h}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def _stage(self, name: str, ctx: RunContext, fn: Callable[[], Any], pairs: Optional[pd.DataFrame],
               stats: List[Dict[str, Any]]) -> Any:
        cached = None
        for h in self.hooks:
            res = h.on_stage_start(name, ctx, pairs)
            if res is not None and cached is None and name in PAIR_STAGES:
                cached = res
        start = time.perf_counter()
        try:
            out = cached if cached is not None else fn()
        except BaseException as exc:
            for h in self.hooks:
                h.on_error(name, ctx, exc)
            raise
        secs = time.perf_counter() - start
        ctx.artifacts[f"restored.{name}"] = cached is not None
        is_pairs = isinstance(out, pd.DataFrame) and name in PAIR_STAGES
        if is_pairs and ctx.debug:
            S.validate_pairs(out, name)
        stats.append({
            "stage": name,
            "rows_in": None if pairs is None else len(pairs),
            "rows_out": len(out) if is_pairs else None,
            "seconds": secs,
            "restored": cached is not None,
        })
        for h in self.hooks:
            h.on_stage_end(name, ctx, out if is_pairs else None, secs)
        return out

    def _run(self, ctx: RunContext, stop_after: Optional[str] = None) -> PipelineResult:
        t_start = time.perf_counter()
        ctx.artifacts["stage_configs"] = self.stage_configs()
        ctx.artifacts["data_hash"] = self._data_hash(ctx)
        for h in self.hooks:
            h.on_run_start(ctx)
        stats: List[Dict[str, Any]] = []

        self._stage("normalize", ctx, lambda: self._normalize(ctx), None, stats)
        pairs = self._stage("block", ctx, lambda: self._block(ctx), None, stats)
        pairs = self._stage("compare", ctx, lambda: self._compare(ctx, pairs), pairs, stats)
        pairs = self._stage("score", ctx, lambda: self._score(ctx, pairs), pairs, stats)
        pairs = self._stage("route", ctx, lambda: self._route(ctx, pairs), pairs, stats)
        if stop_after == "route":
            return self._partial(ctx, pairs, stats, t_start)
        pairs = self._stage("validate", ctx, lambda: self._validate(ctx, pairs), pairs, stats)

        records, right_records, consolidated = ctx.left.copy(), None, None
        if ctx.mode == "dedup":
            labels = self._stage("cluster", ctx, lambda: self._cluster(ctx, pairs), pairs, stats)
            records[S.CLUSTER_ID] = labels
            flags = ctx.artifacts.get("cluster_flag")
            if flags is not None:
                records["cluster_flag"] = flags
        else:
            right_records = ctx.right.copy()
            if self.consolidator is not None:
                consolidated = self._stage(
                    "consolidate", ctx, lambda: self._consolidate(ctx, pairs), pairs, stats)

        config = self.to_config()
        result = PipelineResult(
            records=records, pairs=pairs, mode=ctx.mode, columns=ctx.columns, config=config,
            right_records=right_records, legacy_score_feature=self.legacy_score_feature,
            consolidated=consolidated,
            stats={
                "stages": stats,
                "total_seconds": time.perf_counter() - t_start,
                "n_left": len(ctx.left), "n_right": len(ctx.right), "n_pairs": len(pairs),
                "n_matches": int((pairs[S.DECISION] == S.MATCH).sum()),
                "usage": ctx.artifacts.get("usage", {}),
            },
        )
        for h in self.hooks:
            h.on_run_end(ctx, result)
        return result

    def _partial(self, ctx: RunContext, pairs: pd.DataFrame, stats: List[Dict[str, Any]],
                 t_start: float) -> PipelineResult:
        """Result of a run that stopped after routing (used by EngineMatcher)."""
        return PipelineResult(
            records=ctx.left.copy(), pairs=pairs, mode=ctx.mode, columns=ctx.columns,
            config=self.to_config(), right_records=ctx.right.copy() if ctx.mode == "link" else None,
            legacy_score_feature=self.legacy_score_feature,
            stats={"stages": stats, "total_seconds": time.perf_counter() - t_start,
                   "n_left": len(ctx.left), "n_right": len(ctx.right), "n_pairs": len(pairs),
                   "usage": ctx.artifacts.get("usage", {})})

    # ------------------------------------------------------------------ stages
    def _normalize(self, ctx: RunContext) -> None:
        def apply(df: pd.DataFrame) -> pd.DataFrame:
            for n in self.normalizers:
                out = n(df, ctx)
                if len(out) != len(df):
                    raise ConfigurationError(f"Normalizer {describe(n)} changed the number of rows")
                df = out.reset_index(drop=True)
            return df

        ctx.set_left(apply(ctx.left))
        if ctx.mode == "link":
            ctx.right = apply(ctx.right)

    def _block(self, ctx: RunContext) -> pd.DataFrame:
        n_left, n_right = len(ctx.left), len(ctx.right)
        if not self.blockers:
            total = n_left * (n_left - 1) // 2 if ctx.mode == "dedup" else n_left * n_right
            if total > self.max_pairs:
                raise ConfigurationError(
                    f"No blocker configured and all-pairs would produce {total:,} pairs "
                    f"(max_pairs={self.max_pairs:,}); add a blocker")
            dt = S.index_dtype(n_left, n_right)
            if ctx.mode == "dedup":
                a, b = np.triu_indices(n_left, k=1)
            else:
                a, b = np.divmod(np.arange(n_left * n_right, dtype=np.int64), max(n_right, 1))
            return pd.DataFrame({S.I1: a.astype(dt), S.I2: b.astype(dt)})
        frames, seen = [], {}
        for b in self.blockers:
            base = getattr(b, "name", None) or type(b).__name__.lower()
            seen[base] = seen.get(base, 0) + 1
            name = base if seen[base] == 1 else f"{base}_{seen[base]}"
            b.fit(ctx.right, ctx)
            cand = b.candidates(ctx.left, ctx)
            ctx.artifacts[f"blocker.{name}"] = b
            frames.append((name, cand))
        from ..blocking.union import union_candidates

        pairs = union_candidates(frames, n_right=n_right, mode=ctx.mode, n_left=n_left)
        if len(pairs) > self.max_pairs:
            biggest = max(frames, key=lambda f: len(f[1]))[0]
            raise ConfigurationError(
                f"Blocking produced {len(pairs):,} candidate pairs, above max_pairs={self.max_pairs:,}; "
                f"blocker '{biggest}' contributed the most ({len(dict(frames)[biggest]):,}). "
                "Tighten its parameters or raise max_pairs")
        return pairs

    def _compare(self, ctx: RunContext, pairs: pd.DataFrame) -> pd.DataFrame:
        new: List[pd.DataFrame] = []
        taken = set(pairs.columns)
        for c in self.comparators:
            feats = c(pairs, ctx.left, ctx.right, ctx)
            if len(feats) != len(pairs):
                raise ConfigurationError(f"Comparator {describe(c)} returned {len(feats)} rows for {len(pairs)} pairs")
            feats = feats.set_axis(pairs.index, axis=0)
            clash = taken.intersection(feats.columns)
            if clash:
                raise ConfigurationError(f"Comparator {describe(c)} produces duplicate columns: {sorted(clash)}")
            taken.update(feats.columns)
            new.append(feats)
        return pd.concat([pairs] + new, axis=1) if new else pairs

    def _score(self, ctx: RunContext, pairs: pd.DataFrame) -> pd.DataFrame:
        pairs = pairs.copy()
        if self.scorer is not None:
            s = self.scorer(pairs, ctx)
        else:
            feats = [c for c in pairs.columns if c.startswith(S.FEATURE_PREFIX)]
            if len(feats) != 1:
                raise ConfigurationError(
                    f"No scorer configured and {len(feats)} features found; "
                    "a scorer is required unless exactly one feature exists")
            s = pairs[feats[0]]
        pairs[S.SCORE] = np.asarray(pd.to_numeric(s, errors="coerce"), dtype=np.float32)
        return pairs

    def _route(self, ctx: RunContext, pairs: pd.DataFrame) -> pd.DataFrame:
        pairs = pairs.copy()
        n = len(pairs)
        route = (self.router(pairs, ctx).astype(object).to_numpy() if self.router is not None
                 else np.full(n, S.REVIEW, dtype=object))
        bad = set(route) - set(S.ROUTES)
        if bad:
            raise ConfigurationError(f"Router returned invalid routes: {sorted(map(str, bad))}")
        decision = np.full(n, None, dtype=object)
        decision[route == S.ACCEPT] = S.MATCH
        decision[route == S.REJECT] = S.NON_MATCH
        decided = decision != None  # noqa: E711
        by = np.full(n, None, dtype=object)
        by[decided] = "route"
        reason = np.full(n, None, dtype=object)
        reason[decided] = np.where(route[decided] == S.ACCEPT, "routed: accept", "routed: reject")
        pairs[S.ROUTE] = pd.Categorical(route, categories=list(S.ROUTES))
        pairs[S.DECISION] = decision
        pairs[S.DECISION_PROB] = np.full(n, np.nan, dtype=np.float32)
        pairs[S.DECISION_REASON] = reason
        pairs[S.DECIDED_BY] = by
        return pairs

    def _validate(self, ctx: RunContext, pairs: pd.DataFrame) -> pd.DataFrame:
        pairs = pairs.copy()
        for v in self.validators:
            scope = getattr(v, "scope", "review")
            if scope == "review":
                mask = pairs[S.DECISION].isna()
            elif scope == "accepted":
                mask = pairs[S.ROUTE].astype(object) == S.ACCEPT
            elif scope == "all":
                mask = pd.Series(True, index=pairs.index)
            else:
                raise ConfigurationError(f"Validator scope must be review, all or accepted, got '{scope}'")
            if not mask.any():
                continue
            sub = pairs[mask]
            res = v(sub, ctx.left, ctx.right, ctx)
            if res is None or len(res) == 0:
                continue
            res = res.loc[res.index.intersection(sub.index)]
            if S.DECISION not in res.columns:
                raise ConfigurationError(f"Validator {describe(v)} did not return a '{S.DECISION}' column")
            res = res[res[S.DECISION].isin([S.MATCH, S.NON_MATCH])]
            if res.empty:
                continue
            name = getattr(v, "name", None) or getattr(v, "__name__", type(v).__name__)
            idx = res.index
            pairs.loc[idx, S.DECISION] = res[S.DECISION].to_numpy()
            pairs.loc[idx, S.DECISION_PROB] = (res[S.DECISION_PROB].to_numpy(dtype=np.float32)
                                               if S.DECISION_PROB in res.columns else np.nan)
            pairs.loc[idx, S.DECISION_REASON] = (res[S.DECISION_REASON].to_numpy(dtype=object)
                                                 if S.DECISION_REASON in res.columns else None)
            pairs.loc[idx, S.DECIDED_BY] = name
        left_over = pairs[S.DECISION].isna()
        if left_over.any():
            pairs.loc[left_over, S.DECISION] = self.undecided
            if self.undecided != S.UNVALIDATED:
                pairs.loc[left_over, S.DECIDED_BY] = f"policy:{self.undecided}"
        pairs[S.DECISION] = pd.Categorical(pairs[S.DECISION], categories=list(S.DECISIONS))
        pairs[S.DECISION_PROB] = pairs[S.DECISION_PROB].astype(np.float32)
        return pairs

    def _cluster(self, ctx: RunContext, pairs: pd.DataFrame) -> np.ndarray:
        clusterer = self.clusterer
        if clusterer is None:
            from ..cluster.components import ConnectedComponents

            clusterer = ConnectedComponents()
        # the clusterer may write decisions back into ``pairs`` (guarded clustering)
        pairs[S.DECISION] = pairs[S.DECISION].astype(object)
        labels = np.asarray(clusterer(pairs, len(ctx.left), ctx), dtype=np.int64)
        pairs[S.DECISION] = pd.Categorical(pairs[S.DECISION], categories=list(S.DECISIONS))
        if len(labels) != len(ctx.left):
            raise ConfigurationError("Clusterer must return one label per record")
        return labels

    def _consolidate(self, ctx: RunContext, pairs: pd.DataFrame) -> pd.DataFrame:
        c = ctx.columns
        matches = pairs[pairs[S.DECISION] == S.MATCH]
        cross = S.to_legacy_cross(matches, ctx.left, ctx.right, c.id, c.name, c.rid, c.rname,
                                  self.legacy_score_feature)
        return self.consolidator(cross, ctx.left, ctx.right, ctx)
