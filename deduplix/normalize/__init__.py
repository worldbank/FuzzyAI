"""Normalization and name parsing (importing this package registers the normalizers)."""
from __future__ import annotations

from . import name_parser, text  # noqa: F401  (registration side effects)
from .name_parser import NameParser, parse_names
from .tables import load_table
from .text import chain

__all__ = ["NameParser", "parse_names", "load_table", "chain"]
