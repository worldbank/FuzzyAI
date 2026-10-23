"""Blocking: candidate generation. Blocker implementations register themselves on import.

Optional dependencies (scikit-learn, scipy, metaphone, datasketch, sentence-transformers,
hnswlib, ...) are imported lazily when a blocker is fitted, so this package imports with the
core dependencies only.
"""
from . import brute_force, embedding, exact, minhash, phonetic, tfidf, token_index  # noqa: F401
from .base import BaseBlocker
from .brute_force import BruteForceBlocker
from .embedding import EmbeddingANNBlocker
from .exact import ExactKeyBlocker
from .metrics import blocking_report, pair_completeness, pairs_quality, reduction_ratio
from .minhash import MinHashLSHBlocker
from .phonetic import PhoneticBlocker
from .tfidf import TfidfCharBlocker
from .token_index import TokenIndexBlocker
from .union import union_candidates

__all__ = [
    "union_candidates", "BaseBlocker", "BruteForceBlocker", "EmbeddingANNBlocker",
    "ExactKeyBlocker", "MinHashLSHBlocker", "PhoneticBlocker", "TfidfCharBlocker",
    "TokenIndexBlocker", "blocking_report", "pair_completeness", "pairs_quality",
    "reduction_ratio",
]
