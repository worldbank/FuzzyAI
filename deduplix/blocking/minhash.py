"""MinHash LSH blocker (datasketch)."""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from ..engine.registry import register
from .base import BaseBlocker, require

logger = logging.getLogger("deduplix.blocking.minhash")


@register("blocker", "minhash")
class MinHashLSHBlocker(BaseBlocker):
    """Jaccard-similar records via MinHash LSH over character or word shingles.

    Suited to incremental or online use (insert and query one record at a time); in batch,
    the TF-IDF blocker is faster. With ``verify=True`` the estimated Jaccard is recomputed
    for each candidate and pairs below ``threshold`` are dropped; it is the score.
    """

    name = "minhash"

    def __init__(self, shingle: str = "char", k: int = 3, num_perm: int = 128,
                 threshold: float = 0.5, seed: int = 1, verify: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        if shingle not in ("char", "word"):
            raise ValueError("shingle must be 'char' or 'word'")
        self.shingle = shingle
        self.k = k
        self.num_perm = num_perm
        self.threshold = threshold
        self.seed = seed
        self.verify = verify

    def _check_deps(self) -> None:
        require("datasketch", "minhash")

    def _shingles(self, text: str) -> List[bytes]:
        t = " ".join(text.lower().split())
        if not t:
            return []
        if self.shingle == "word":
            toks = t.split(" ")
            grams = [" ".join(toks[i:i + self.k]) for i in range(max(1, len(toks) - self.k + 1))]
        else:
            grams = [t[i:i + self.k] for i in range(max(1, len(t) - self.k + 1))]
        return [g.encode("utf-8") for g in set(grams)]

    def _signatures(self, texts: List[str]):
        from datasketch import LeanMinHash, MinHash

        out = []
        for t in texts:
            sh = self._shingles(t)
            if not sh:
                out.append(None)
                continue
            m = MinHash(num_perm=self.num_perm, seed=self.seed)
            m.update_batch(sh)
            out.append(LeanMinHash(m))
        return out

    def _prepare(self, left, right, ctx) -> None:
        lt = self._texts(left, ctx, "left")
        self._sl = self._signatures(lt)
        self._sr = self._sl if self._dedup else self._signatures(self._texts(right, ctx, "right"))

    def _block(self, left_idx, right_idx, ctx):
        from datasketch import MinHashLSH

        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        for j, g in enumerate(right_idx):
            sig = self._sr[g]
            if sig is not None:
                lsh.insert(str(j), sig)
        ii, jj, ss = [], [], []
        for i, g in enumerate(left_idx):
            sig = self._sl[g]
            if sig is None:
                continue
            for key in lsh.query(sig):
                j = int(key)
                est = float(sig.jaccard(self._sr[right_idx[j]]))
                if self.verify and est < self.threshold:
                    continue
                ii.append(i)
                jj.append(j)
                ss.append(est)
        return (np.array(ii, dtype=np.int64), np.array(jj, dtype=np.int64),
                np.array(ss, dtype=np.float32))
