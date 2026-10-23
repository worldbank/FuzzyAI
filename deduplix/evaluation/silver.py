"""Silver-standard labels on real data: seed sample, brute force, LLM label, human review."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("deduplix.evaluation.silver")

REVIEW_COLUMNS = ["id1", "id2", "name1", "name2", "score", "llm_label", "reason", "human_label"]


def sample_seed(left: pd.DataFrame, n_seed: int = 5000, country_col: Optional[str] = "country",
                seed: int = 0) -> pd.DataFrame:
    """Sample ``n_seed`` left records, proportionally stratified by ``country_col``."""
    if len(left) <= n_seed:
        return left.copy()
    if not country_col or country_col not in left.columns:
        return left.sample(n=n_seed, random_state=seed)
    frac = n_seed / len(left)
    parts = [g.sample(n=max(1, int(round(len(g) * frac))), random_state=seed)
             for _, g in left.groupby(left[country_col].fillna("__missing__"), sort=True)]
    out = pd.concat(parts)
    if len(out) > n_seed:
        out = out.sample(n=n_seed, random_state=seed)
    return out.sort_index()


def brute_force_candidates(seed_df: pd.DataFrame, right: pd.DataFrame, id_col: str = "id",
                           name_col: str = "name", right_id_col: Optional[str] = None,
                           right_name_col: Optional[str] = None, threshold: float = 60.0,
                           exclude_same_id: bool = True, chunk: int = 500) -> pd.DataFrame:
    """All (seed, right) pairs with ``token_set_ratio >= threshold`` (rapidfuzz, no network)."""
    from rapidfuzz import fuzz, process

    rid, rname = right_id_col or id_col, right_name_col or name_col
    qn = seed_df[name_col].fillna("").astype(str).str.lower().tolist()
    rn = right[rname].fillna("").astype(str).str.lower().tolist()
    qid, rids = seed_df[id_col].to_numpy(), right[rid].to_numpy()
    qnames, rnames = seed_df[name_col].to_numpy(), right[rname].to_numpy()
    out = []
    for s in range(0, len(qn), chunk):
        sims = process.cdist(qn[s:s + chunk], rn, scorer=fuzz.token_set_ratio,
                             score_cutoff=threshold, dtype=np.float32, workers=1)
        a, b = np.nonzero(sims >= threshold)
        a = a + s
        keep = np.ones(len(a), dtype=bool)
        if exclude_same_id:
            keep &= qid[a] != rids[b]
        a, b = a[keep], b[keep]
        out.append(pd.DataFrame({"id1": qid[a], "id2": rids[b], "name1": qnames[a],
                                 "name2": rnames[b], "score": sims[a - s, b] if len(a) else []}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=REVIEW_COLUMNS[:5])


def llm_label(cands: pd.DataFrame, validator: Any, seed_df: pd.DataFrame, right: pd.DataFrame,
              id_col: str = "id", name_col: str = "name", right_id_col: Optional[str] = None,
              right_name_col: Optional[str] = None) -> pd.DataFrame:
    """Label candidates with any engine validator ``(pairs, left, right, ctx) -> frame``.

    Returns a review frame with ``llm_label`` (1/0, NaN if undecided), ``reason`` and an
    empty ``human_label``. The validator is called on positional indexes into the two
    (reset-index) tables.
    """
    from ..engine.context import ColumnMap, RunContext
    from ..engine.schema import DECISION, DECISION_REASON, I1, I2, MATCH, NON_MATCH

    left = seed_df.reset_index(drop=True)
    rt = right.reset_index(drop=True)
    rid = right_id_col or id_col
    lpos = pd.Series(np.arange(len(left)), index=left[id_col].to_numpy())
    rpos = pd.Series(np.arange(len(rt)), index=rt[rid].to_numpy())
    pairs = pd.DataFrame({I1: lpos[~lpos.index.duplicated()].reindex(cands["id1"]).to_numpy(),
                          I2: rpos[~rpos.index.duplicated()].reindex(cands["id2"]).to_numpy()})
    pairs["score"] = cands["score"].to_numpy() / 100.0 if cands["score"].max() > 1 else cands["score"].to_numpy()
    ctx = RunContext(left=left, right=rt, mode="link",
                     columns=ColumnMap(id=id_col, name=name_col, right_id=right_id_col,
                                       right_name=right_name_col))
    res = validator(pairs, ctx.left, ctx.right, ctx) if len(pairs) else None
    review = cands.copy().reset_index(drop=True)
    label = pd.Series(np.nan, index=review.index)
    reason = pd.Series("", index=review.index, dtype=object)
    if res is not None and len(res) and DECISION in res.columns:
        idx = res.index.to_numpy()
        dec = res[DECISION].astype(object)
        label.iloc[idx] = dec.map({MATCH: 1.0, NON_MATCH: 0.0}).to_numpy()
        if DECISION_REASON in res.columns:
            reason.iloc[idx] = res[DECISION_REASON].fillna("").to_numpy()
    review["llm_label"], review["reason"], review["human_label"] = label, reason, np.nan
    return review[REVIEW_COLUMNS]


def export_review(review: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Write ``review.csv`` (id1, id2, name1, name2, score, llm_label, reason, human_label)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(p, index=False, encoding="utf-8")
    return p


def import_review(csv: Union[str, Path, pd.DataFrame], out: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Merge human labels over LLM labels (human wins) into ``id1, id2, label, source``.

    Rows with neither label are dropped. When ``out`` is given the result is saved
    as parquet (``gold.parquet``).
    """
    df = csv.copy() if isinstance(csv, pd.DataFrame) else pd.read_csv(csv, dtype={"id1": object, "id2": object})
    for c in ("llm_label", "human_label"):
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    human = df["human_label"].notna()
    label = df["human_label"].where(human, df["llm_label"])
    gold = pd.DataFrame({"id1": df["id1"], "id2": df["id2"], "label": label,
                         "source": np.where(human, "human", "llm")})
    gold = gold[gold["label"].notna()].copy()
    gold["label"] = gold["label"].round().astype("int64")
    gold = gold.reset_index(drop=True)
    if out is not None:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        gold.to_parquet(p)
    return gold


def build_silver(left: pd.DataFrame, right: pd.DataFrame, validator: Any, review_path: Union[str, Path],
                 n_seed: int = 5000, threshold: float = 60.0, id_col: str = "id", name_col: str = "name",
                 country_col: Optional[str] = "country", seed: int = 0) -> pd.DataFrame:
    """Steps 1-3: sample, brute force, LLM label, export ``review.csv``; returns the review frame."""
    sd = sample_seed(left, n_seed, country_col, seed)
    cands = brute_force_candidates(sd, right, id_col, name_col, threshold=threshold)
    logger.info("silver: %d seed records, %d candidates", len(sd), len(cands))
    review = llm_label(cands, validator, sd, right, id_col, name_col)
    export_review(review, review_path)
    return review
