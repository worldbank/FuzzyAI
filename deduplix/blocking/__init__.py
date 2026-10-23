"""Blocking: candidate generation. Blocker implementations register themselves on import."""
from .union import union_candidates

__all__ = ["union_candidates"]
