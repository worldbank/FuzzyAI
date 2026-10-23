"""Character n-gram TF-IDF blocker with sparse top-n cosine search."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from ..engine.registry import register
from .base import BaseBlocker, require, sp_matmul_topn

logger = logging.getLogger("deduplix.blocking.tfidf")


@register("blocker", "tfidf_char")
class TfidfCharBlocker(BaseBlocker):
    """Top-n cosine neighbours over character n-gram TF-IDF vectors.

    The vectorizer is fitted on left and right together in link mode (IDF reflects both)
    and on left in dedup mode. The cosine goes to ``blk_score_tfidf_char``.
    """

    name = "tfidf_char"
    default_top_k = 20

    def __init__(self, analyzer: str = "char_wb", ngram_range: Tuple[int, int] = (3, 3),
                 min_df: int = 2, sublinear_tf: bool = True, dtype=np.float32,
                 top_n: int = 20, threshold: float = 0.5, chunk_size: int = 10_000,
                 n_threads: int = -1, **kwargs) -> None:
        kwargs.setdefault("top_k", top_n)
        super().__init__(**kwargs)
        self.analyzer = analyzer
        self.ngram_range = tuple(ngram_range)
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.dtype = dtype
        self.top_n = top_n
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.n_threads = n_threads

    def _check_deps(self) -> None:
        require("sklearn", "blocking", "scikit-learn")
        require("scipy", "blocking")

    def _prepare(self, left, right, ctx) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        lt = self._texts(left, ctx, "left")
        rt = lt if self._dedup else self._texts(right, ctx, "right")
        vec = TfidfVectorizer(analyzer=self.analyzer, ngram_range=self.ngram_range,
                              min_df=self.min_df, sublinear_tf=self.sublinear_tf,
                              dtype=np.dtype(self.dtype), lowercase=True)
        self._empty = False
        try:
            vec.fit(lt if self._dedup else lt + rt)
        except ValueError:  # empty vocabulary (e.g. min_df too high)
            self._empty = True
            return
        self._vec = vec
        self._Xr = vec.transform(rt).tocsr()
        self._Xl = self._Xr if self._dedup else vec.transform(lt).tocsr()

    def _block(self, left_idx, right_idx, ctx):
        if self._empty:
            e = np.array([], dtype=np.int64)
            return e, e, np.array([], dtype=np.float32)
        A = self._Xl[left_idx]
        B = self._Xr[right_idx].T.tocsr()  # kept as the right-hand matrix of every chunk
        return sp_matmul_topn(A, B, top_n=int(self.top_n), threshold=self.threshold,
                              chunk_size=self.chunk_size, n_threads=self.n_threads,
                              exclude_diagonal=self._dedup)
