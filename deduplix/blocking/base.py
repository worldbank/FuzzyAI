"""Shared blocker machinery: partitioning, dedup handling, sparse top-n product."""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..engine.schema import I1, I2, index_dtype

logger = logging.getLogger("deduplix.blocking")

SCORE = "score"
NULL_PARTITIONS = ("compare_all", "skip")


def require(module: str, extra: str, pip_name: Optional[str] = None):
    """Import an optional dependency or raise ImportError with an install hint."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{module} is required for this blocker; install it with "
            f"`pip install {pip_name or module}` or `pip install deduplix[{extra}]`"
        ) from exc


def _is_null(values: pd.Series) -> np.ndarray:
    return values.isna().to_numpy()


def _empty3():
    return (np.array([], dtype=np.int64), np.array([], dtype=np.int64),
            np.array([], dtype=np.float32))


def sp_matmul_topn(A, B, top_n: int, threshold: float = 0.0, chunk_size: int = 10_000,
                   n_threads: int = -1, exclude_diagonal: bool = False
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise top-n of ``A @ B`` (sparse), returning ``(rows, cols, values)``.

    Uses ``sparse_dot_topn`` when installed; otherwise a chunked scipy product with a
    per-row top-n. Entries must be ``>= threshold`` (and ``> 0`` when threshold <= 0).
    With ``exclude_diagonal`` the entries ``(i, i)`` are removed before the top-n cut.
    """
    import scipy.sparse as sp

    A = sp.csr_matrix(A)
    B = sp.csr_matrix(B)
    n = A.shape[0]
    rows_out: List[np.ndarray] = []
    cols_out: List[np.ndarray] = []
    vals_out: List[np.ndarray] = []
    ask = top_n + 1 if exclude_diagonal else top_n
    try:  # pragma: no cover - depends on the optional package
        from sparse_dot_topn import sp_matmul_topn as _fast
    except Exception:
        _fast = None
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        chunk = A[start:stop]
        prod = None
        if _fast is not None:  # pragma: no cover
            try:
                prod = sp.csr_matrix(_fast(chunk, B, top_n=ask, threshold=float(threshold),
                                           n_threads=n_threads))
            except Exception as exc:
                logger.debug("sparse_dot_topn failed (%s); using scipy fallback", exc)
                prod = None
        if prod is None:
            prod = (chunk @ B).tocsr()
        prod.eliminate_zeros()
        indptr, indices, data = prod.indptr, prod.indices, prod.data
        for r in range(stop - start):
            lo, hi = indptr[r], indptr[r + 1]
            if lo == hi:
                continue
            c, v = indices[lo:hi], data[lo:hi]
            keep = (v >= threshold) if threshold > 0 else (v > 0)
            if exclude_diagonal:
                keep &= c != (start + r)
            c, v = c[keep], v[keep]
            if len(v) > top_n:
                sel = np.argpartition(-v, top_n - 1)[:top_n]
                c, v = c[sel], v[sel]
            if len(v):
                rows_out.append(np.full(len(v), start + r, dtype=np.int64))
                cols_out.append(c.astype(np.int64))
                vals_out.append(v.astype(np.float32))
    if not rows_out:
        return _empty3()
    return np.concatenate(rows_out), np.concatenate(cols_out), np.concatenate(vals_out)


class BaseBlocker:
    """Base class of all blockers.

    Subclasses implement ``_prepare(left, right, ctx)`` (optional, global state such as
    a vocabulary) and ``_block(left_idx, right_idx, ctx)`` returning local positions
    ``(i, j[, score])`` into ``left_idx`` / ``right_idx``. This class loops over
    partitions, maps positions back to global row positions, and in dedup mode keeps
    only ``i1 < i2`` (pairs are swapped first, self pairs dropped).

    Parameters
    ----------
    on : str, optional
        Column read by the blocker (default ``name_base``, else ``name_norm``, else the
        name column).
    partition_by : str, optional
        Only records with equal values of this column can be candidates.
    null_partition : {"compare_all", "skip"}
        Records with a null partition value are compared with every partition or skipped.
    top_k : int, optional
        Maximum candidates kept per left record (blocker-specific default).
    name : str, optional
        Provenance label; defaults to the registry name.
    """

    name = "base"
    default_top_k: Optional[int] = None
    _dedup = False

    def __init__(self, on: Optional[str] = None, partition_by: Optional[str] = None,
                 null_partition: str = "compare_all", top_k: Optional[int] = None,
                 name: Optional[str] = None) -> None:
        if null_partition not in NULL_PARTITIONS:
            raise ValueError(f"null_partition must be one of {NULL_PARTITIONS}")
        self.on = on
        self.partition_by = partition_by
        self.null_partition = null_partition
        self.top_k = top_k if top_k is not None else self.default_top_k
        if name:
            self.name = name
        self._right: Optional[pd.DataFrame] = None

    # ----- helpers -----
    def _column(self, df: pd.DataFrame, ctx, side: str = "left") -> str:
        if self.on:
            if self.on not in df.columns:
                raise KeyError(f"Blocker '{self.name}': column '{self.on}' not found")
            return self.on
        for c in ("name_base", "name_norm"):
            if c in df.columns:
                return c
        return ctx.columns.name if side == "left" else ctx.columns.rname

    def _texts(self, df: pd.DataFrame, ctx, side: str = "left") -> List[str]:
        col = self._column(df, ctx, side)
        return ["" if v is None or v is pd.NA or (isinstance(v, float) and v != v) else str(v)
                for v in df[col].tolist()]

    def _check_deps(self) -> None:
        """Raise ImportError early if an optional dependency is missing."""

    def _prepare(self, left: pd.DataFrame, right: pd.DataFrame, ctx) -> None:
        """Compute global state once per run (vocabularies, embeddings, ...)."""

    def _block(self, left_idx: np.ndarray, right_idx: np.ndarray, ctx):
        """Return local ``(i, j, score_or_None)`` for the given global row positions."""
        raise NotImplementedError

    # ----- protocol -----
    def fit(self, right: pd.DataFrame, ctx) -> "BaseBlocker":
        self._check_deps()
        self._right = right
        return self

    def candidates(self, left: pd.DataFrame, ctx) -> pd.DataFrame:
        right = self._right if self._right is not None else ctx.right
        dedup = ctx.mode == "dedup"
        self._check_deps()
        self._dedup = dedup
        self._prepare(left, right, ctx)
        dt = index_dtype(len(left), len(right))
        a_parts, b_parts, s_parts = [], [], []
        has_score = False
        for li, ri in self._groups(left, right):
            if len(li) == 0 or len(ri) == 0:
                continue
            res = self._block(li, ri, ctx)
            i, j = np.asarray(res[0], dtype=np.int64), np.asarray(res[1], dtype=np.int64)
            s = res[2] if len(res) > 2 else None
            if len(i) == 0:
                has_score = has_score or s is not None
                continue
            a_parts.append(li[i])
            b_parts.append(ri[j])
            if s is not None:
                has_score = True
                s_parts.append(np.asarray(s, dtype=np.float32))
            else:
                s_parts.append(np.full(len(i), np.nan, dtype=np.float32))
        if not a_parts:
            out = pd.DataFrame({I1: np.array([], dtype=dt), I2: np.array([], dtype=dt)})
            if has_score:
                out[SCORE] = np.array([], dtype=np.float32)
            return out
        a, b, s = np.concatenate(a_parts), np.concatenate(b_parts), np.concatenate(s_parts)
        if dedup:
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            keep = lo < hi
            a, b, s = lo[keep], hi[keep], s[keep]
        out = pd.DataFrame({I1: a, I2: b, SCORE: s})
        # a pair may arise from several partitions (null partitions) or from both sides
        out = out.sort_values(SCORE, ascending=False, kind="stable", na_position="last")
        out = out.drop_duplicates([I1, I2]).sort_values([I1, I2], kind="stable")
        out = out.reset_index(drop=True)
        out[I1] = out[I1].astype(dt)
        out[I2] = out[I2].astype(dt)
        if has_score:
            out[SCORE] = out[SCORE].astype(np.float32)
        else:
            out = out.drop(columns=[SCORE])
        return out

    def _groups(self, left: pd.DataFrame, right: pd.DataFrame
                ) -> List[Tuple[np.ndarray, np.ndarray]]:
        li_all = np.arange(len(left), dtype=np.int64)
        ri_all = np.arange(len(right), dtype=np.int64)
        if not self.partition_by:
            return [(li_all, ri_all)]
        for df in (left, right):
            if self.partition_by not in df.columns:
                raise KeyError(f"partition_by column '{self.partition_by}' not found")
        lp, rp = left[self.partition_by], right[self.partition_by]
        lnull, rnull = _is_null(lp), _is_null(rp)
        keep_null = self.null_partition == "compare_all"
        values = pd.unique(pd.concat([lp[~lnull], rp[~rnull]], ignore_index=True))
        lv = lp.to_numpy(dtype=object)
        rv = rp.to_numpy(dtype=object)
        groups = []
        for v in values:
            lm = (lv == v) & ~lnull
            rm = (rv == v) & ~rnull
            if keep_null:
                lm = lm | lnull
                rm = rm | rnull
            groups.append((li_all[lm], ri_all[rm]))
        if keep_null and len(values) == 0 and (lnull.any() or rnull.any()):
            groups.append((li_all[lnull], ri_all[rnull]))
        return groups

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
