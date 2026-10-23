"""Comparators: turn candidate pairs into ``f_<feature>`` columns."""
from . import designators, fields, grades, string  # noqa: F401  (register comparators)
from .designators import DesignatorCheck
from .fields import CountryMatch, Exact, LegalFormCompat, NumericRelDiff, TokenIdfOverlap
from .grades import Grade, MatchGrade
from .string import StringSim

__all__ = ["StringSim", "Exact", "NumericRelDiff", "DesignatorCheck", "LegalFormCompat",
           "CountryMatch", "TokenIdfOverlap", "Grade", "MatchGrade"]
