"""Engine validator ``llm``: provider x prompt template x parser x decision policy.

Not the legacy ``deduplix.validators.LLMValidator`` (which keeps its own name and API).
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..engine import registry
from ..engine import schema as S
from ..engine.context import ColumnMap
from ..engine.registry import register
from ..exceptions import ConfigurationError
from .cache import DEFAULT_CACHE, cache_key, open_cache
from .parsers import Decision
from .policy import DecisionPolicy
from .templates import load_template

logger = logging.getLogger("deduplix.llm")

SYSTEM_PROMPT = "Identify duplicate entities. Reply with strict JSON only."
DEFAULT_PARSED_COLUMNS = ("legal_form", "designator_nums", "designator_series", "designator_quals")


def _py(v: Any) -> Any:
    """JSON-friendly python value; missing -> None."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v.item() if isinstance(v, np.generic) else v


@register("validator", "llm")
class LLMValidator:
    """Ask an LLM about candidate pairs. ``__call__`` returns ``decision``, ``decision_prob``
    and ``decision_reason`` for every input pair (same index); undecided pairs are
    ``unvalidated`` and the engine leaves them to its ``undecided`` policy.

    ``checkpoint_every_n_batches`` is accepted for config compatibility; with a cache,
    finished batches are already persisted and an interrupted run resumes for free.
    """

    name = "llm"

    def __init__(self, provider: Any, prompt: str = "pair_validation_v2.j2",
                 parser: Any = "json_decisions", policy: Any = None, batch_size: int = 10,
                 n_workers: int = 4, scope: str = "review",
                 metadata_columns: Sequence[str] = (), instructions: Optional[str] = None,
                 cache: Optional[str] = DEFAULT_CACHE, checkpoint_every_n_batches: int = 0,
                 include_features: Sequence[str] = (), parsed_columns: Sequence[str] = DEFAULT_PARSED_COLUMNS,
                 temperature: float = 0.0, max_tokens: int = 4096):
        if scope not in ("review", "all", "accepted"):
            raise ConfigurationError(f"scope must be review, all or accepted, got '{scope}'")
        if batch_size < 1 or n_workers < 1:
            raise ConfigurationError("batch_size and n_workers must be >= 1")
        self.provider = registry.build("provider", provider)
        self.template = load_template(prompt)
        self.parser = registry.build("parser", parser)
        if policy is None:
            policy = DecisionPolicy()
        elif isinstance(policy, dict):
            policy = DecisionPolicy(**policy)
        self.policy = policy
        self.batch_size = 1 if getattr(self.parser, "single_pair", False) else batch_size
        self.n_workers = n_workers
        self.scope = scope
        self.metadata_columns = list(metadata_columns)
        self.instructions = instructions
        self.cache = open_cache(cache)
        self.checkpoint_every_n_batches = checkpoint_every_n_batches
        self.include_features = list(include_features)
        self.parsed_columns = list(parsed_columns)
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------ items
    def _build_items(self, pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame,
                     columns: ColumnMap) -> List[dict]:
        extra = [c for c in self.metadata_columns + self.parsed_columns]

        def side(df: pd.DataFrame, positions: np.ndarray, name_col: str) -> List[dict]:
            cols = [c for c in dict.fromkeys(extra) if c in df.columns and c != name_col]
            sub = df[[name_col] + cols].iloc[positions]
            rows = sub.to_numpy(dtype=object).tolist()
            return [{"name": _py(r[0]), **{c: _py(v) for c, v in zip(cols, r[1:])}} for r in rows]

        lefts = side(left, pairs[S.I1].to_numpy(), columns.name)
        rights = side(right, pairs[S.I2].to_numpy(), columns.rname)
        feat_names = {c: c for c in pairs.columns if c.startswith(S.FEATURE_PREFIX)}
        wanted = []
        for f in self.include_features:
            col = f if f in feat_names else S.FEATURE_PREFIX + f
            if col in pairs.columns:
                wanted.append(col)
        legacy_sim = self.template.name.endswith("_v1.j2") and S.SCORE in pairs.columns
        items = []
        for k in range(len(pairs)):
            feats = {c: _py(pairs[c].iloc[k]) for c in wanted}
            if legacy_sim:
                feats["similarity_score"] = float(pairs[S.SCORE].iloc[k]) * 100.0
            items.append({"left": lefts[k], "right": rights[k], "features": feats})
        return items

    def _key(self, item: dict) -> str:
        return cache_key(self.provider.name, getattr(self.provider, "model", None),
                         self.template.name, self.template.version, self.instructions, item)

    # ------------------------------------------------------------------ batches
    def _run_batch(self, batch_items: List[dict], usage: dict, lock: threading.Lock):
        batch = [{"idx": j, **it} for j, it in enumerate(batch_items)]
        prompt = self.template.render(
            batch, instructions=self.instructions, schema=getattr(self.parser, "schema", None),
            metadata_columns=self.metadata_columns, parsed_columns=self.parsed_columns)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}]
        resp = self.provider.complete(messages, schema=getattr(self.parser, "schema", None),
                                      temperature=self.temperature, max_tokens=self.max_tokens)
        with lock:
            usage["calls"] += 1
            usage["input_tokens"] += int((resp.usage or {}).get("input_tokens", 0) or 0)
            usage["output_tokens"] += int((resp.usage or {}).get("output_tokens", 0) or 0)
        return self.parser(resp, batch)

    # ------------------------------------------------------------------ call
    def __call__(self, pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame,
                 ctx: Any = None) -> pd.DataFrame:
        out = pd.DataFrame({S.DECISION: pd.Series(S.UNVALIDATED, index=pairs.index, dtype=object),
                            S.DECISION_PROB: np.full(len(pairs), np.nan, dtype=np.float32),
                            S.DECISION_REASON: pd.Series(None, index=pairs.index, dtype=object)},
                           index=pairs.index)
        artifacts: Dict[str, Any] = ctx.artifacts if ctx is not None else {}
        usage = artifacts.setdefault("usage", {})
        for k in ("input_tokens", "output_tokens", "calls"):
            usage.setdefault(k, 0)
        failures = artifacts.setdefault("llm_failures", {"failed_batches": 0, "failed_pairs": 0})
        if len(pairs) == 0:
            return out
        columns = ctx.columns if ctx is not None else ColumnMap()
        items = self._build_items(pairs, left, right, columns)
        keys = [self._key(it) for it in items] if self.cache is not None else [None] * len(items)

        answers: Dict[int, Optional[Decision]] = {}  # position -> Decision (None = unanswered)
        failed: Dict[int, BaseException] = {}
        todo: List[int] = []
        for k in range(len(items)):
            hit = self.cache.get(keys[k]) if self.cache is not None else None
            if hit is not None:
                answers[k] = hit
            else:
                todo.append(k)

        batches = [todo[s:s + self.batch_size] for s in range(0, len(todo), self.batch_size)]
        lock = threading.Lock()

        def work(positions: List[int]):
            try:
                decisions = self._run_batch([items[p] for p in positions], usage, lock)
                return positions, decisions, None
            except Exception as exc:  # noqa: BLE001 - any failure is handled by the policy
                return positions, None, exc

        if batches:
            if self.n_workers > 1 and len(batches) > 1:
                with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
                    results = list(pool.map(work, batches))
            else:
                results = [work(b) for b in batches]
            for positions, decisions, exc in results:
                if exc is not None:
                    logger.warning("LLM batch of %d pairs failed: %s: %s", len(positions),
                                   type(exc).__name__, exc)
                    failures["failed_batches"] += 1
                    failures["failed_pairs"] += len(positions)
                    for p in positions:
                        failed[p] = exc
                    continue
                by_local = {d.pair_index: d for d in decisions}
                for j, p in enumerate(positions):
                    d = by_local.get(j)
                    answers[p] = d
                    if d is not None and self.cache is not None:
                        self.cache.set(keys[p], d)

        decisions_col, probs, reasons = [], [], []
        for k in range(len(items)):
            if k in failed:
                dec, prob, reason = self.policy.failure(failed[k])
            else:
                dec, prob, reason = self.policy.decide(answers.get(k))
            decisions_col.append(dec)
            probs.append(prob)
            reasons.append(reason)
        out[S.DECISION] = pd.Series(decisions_col, index=pairs.index, dtype=object)
        out[S.DECISION_PROB] = np.array(probs, dtype=np.float32)
        out[S.DECISION_REASON] = pd.Series(reasons, index=pairs.index, dtype=object)
        return out


__all__ = ["LLMValidator", "SYSTEM_PROMPT"]
