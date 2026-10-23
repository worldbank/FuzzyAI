"""Inverted token index blocker scored by summed IDF of shared tokens."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, Optional, Set

import numpy as np

from ..engine.registry import register
from .base import BaseBlocker, require, sp_matmul_topn

logger = logging.getLogger("deduplix.blocking.token_index")


def load_stopwords(spec: Any) -> Set[str]:
    """Stopwords from a list, or a YAML file (list, or mapping of lists); missing file -> empty."""
    if spec is None:
        return set()
    if isinstance(spec, (list, tuple, set, frozenset)):
        return {str(s).lower() for s in spec}
    candidates = [str(spec)]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(here, "normalize", "tables", os.path.basename(str(spec))))
    for path in candidates:
        if os.path.isfile(path):
            import yaml

            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            words: Set[str] = set()

            def walk(x):
                if isinstance(x, str):
                    words.add(x.lower())
                elif isinstance(x, dict):
                    for v in x.values():
                        walk(v)
                elif isinstance(x, Iterable):
                    for v in x:
                        walk(v)

            walk(data)
            return words
    logger.info("token_index: stopwords file '%s' not found; no stopwords applied", spec)
    return set()


@register("blocker", "token_index")
class TokenIndexBlocker(BaseBlocker):
    """Candidates sharing rare tokens, scored by the sum of IDF over shared tokens.

    Tokens with document frequency above ``max_df`` (or ``max_df_ratio * N``) are removed
    from the vocabulary (a token in d records yields about d**2/2 pairs) and recorded in
    ``ctx.artifacts["token_index.dropped"]`` with their counts.
    """

    name = "token_index"
    default_top_k = 50

    def __init__(self, min_token_len: int = 2, max_df: int = 1000,
                 max_df_ratio: Optional[float] = None, stopwords: Any = "stopwords.yaml",
                 top_k: int = 50, min_score: float = 0.0, **kwargs) -> None:
        super().__init__(top_k=top_k, **kwargs)
        self.min_token_len = min_token_len
        self.max_df = max_df
        self.max_df_ratio = max_df_ratio
        self.stopwords = stopwords
        self.min_score = min_score

    def _check_deps(self) -> None:
        require("sklearn", "blocking", "scikit-learn")
        require("scipy", "blocking")

    def _prepare(self, left, right, ctx) -> None:
        import scipy.sparse as sp
        from sklearn.feature_extraction.text import CountVectorizer

        rt = self._texts(right, ctx, "right")
        stop = load_stopwords(self.stopwords)
        pattern = r"(?u)\b\w{%d,}\b" % max(1, self.min_token_len)
        vec = CountVectorizer(binary=True, token_pattern=pattern, lowercase=True,
                              stop_words=sorted(stop) or None, dtype=np.float32)
        try:
            Xr = vec.fit_transform(rt).tocsc()
        except ValueError:  # empty vocabulary
            self._empty = True
            return
        self._empty = False
        n = Xr.shape[0]
        df = np.diff(Xr.indptr)
        limit = self.max_df
        if self.max_df_ratio is not None:
            limit = min(limit, max(1, int(self.max_df_ratio * n)))
        keep = df <= limit
        vocab = vec.get_feature_names_out()
        dropped = {str(t): int(c) for t, c in zip(vocab[~keep], df[~keep])}
        art = ctx.artifacts.setdefault("token_index.dropped", {})
        art.update(dropped)
        if dropped:
            logger.warning("token_index: dropped %d token(s) with document frequency > %d", len(dropped), limit)
        self._keep = keep
        self._vec = vec
        self._idf = np.log(n / np.maximum(df, 1)).astype(np.float32)
        Xr = Xr.tocsr()[:, np.where(keep)[0]]
        self._idf = self._idf[keep]
        self._Xr = Xr.tocsr()
        if self._dedup:
            self._Xl = self._Xr
        else:
            Xl = vec.transform(self._texts(left, ctx, "left")).tocsr()
            self._Xl = Xl[:, np.where(keep)[0]].tocsr()
        self._Xl_w = (self._Xl @ sp.diags(self._idf)).tocsr()

    def _block(self, left_idx, right_idx, ctx):
        if self._empty:
            e = np.array([], dtype=np.int64)
            return e, e, np.array([], dtype=np.float32)
        A = self._Xl_w[left_idx]
        B = self._Xr[right_idx].T.tocsr()
        return sp_matmul_topn(A, B, top_n=int(self.top_k), threshold=self.min_score,
                              exclude_diagonal=self._dedup)
