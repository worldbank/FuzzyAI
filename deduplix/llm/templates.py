"""Jinja2 prompt templates: package files in ``llm/prompts`` or user paths."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, StrictUndefined

from ..exceptions import ConfigurationError

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

_ENV = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=False)


class PromptTemplate:
    """A compiled template with a ``name`` and a content-derived ``version``.

    Variables: ``pairs`` (list of ``{idx, left, right, features}``), ``instructions``,
    ``schema``, ``metadata_columns`` and ``parsed_columns``.
    """

    def __init__(self, source: str, name: str = "inline"):
        self.name = name
        self.source = source
        self.version = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
        self._template = _ENV.from_string(source)

    @classmethod
    def load(cls, template: str) -> "PromptTemplate":
        """Resolve a package template name (``pair_validation_v2.j2``) or a file path."""
        if isinstance(template, PromptTemplate):
            return template
        for cand in (PROMPT_DIR / template, PROMPT_DIR / (template + ".j2")):
            if cand.is_file():
                return cls(cand.read_text(encoding="utf-8"), name=cand.name)
        path = Path(template)
        if path.is_file():
            return cls(path.read_text(encoding="utf-8"), name=path.name)
        raise ConfigurationError(f"Prompt template '{template}' is neither a package template "
                                 f"({sorted(p.name for p in PROMPT_DIR.glob('*.j2'))}) nor a file")

    def render(self, pairs: list, instructions: Optional[str] = None, schema: Any = None,
               metadata_columns: Optional[list] = None, parsed_columns: Optional[list] = None,
               **extra: Any) -> str:
        ctx: Dict[str, Any] = dict(pairs=pairs, instructions=instructions, schema=schema,
                                   metadata_columns=metadata_columns or [],
                                   parsed_columns=parsed_columns or [])
        ctx.update(extra)
        return self._template.render(**ctx)


def load_template(template: Any) -> PromptTemplate:
    return PromptTemplate.load(template)
