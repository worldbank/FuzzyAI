"""LLM layer: providers, prompt templates, parsers, decision policy, cache and the ``llm``
engine validator. Importing this package registers its components.

``deduplix.llm.LLMValidator`` is the *engine* validator; it is deliberately not exported as
``deduplix.LLMValidator`` (the legacy class keeps that name).
"""
from . import parsers, providers, templates  # noqa: F401  (registration)
from .cache import SqliteCache, open_cache
from .parsers import Decision, JsonDecisionsParser, ParseError, ProbabilitiesParser
from .policy import DecisionPolicy
from .providers import (AnthropicProvider, BaseProvider, CallableProvider, DatabricksProvider,
                        FakeProvider, OpenAIProvider, ProviderResponse)
from .templates import PromptTemplate, load_template
from .validator import LLMValidator

__all__ = [
    "LLMValidator", "DecisionPolicy", "Decision", "ParseError", "JsonDecisionsParser",
    "ProbabilitiesParser", "ProviderResponse", "BaseProvider", "OpenAIProvider",
    "AnthropicProvider", "DatabricksProvider", "CallableProvider", "FakeProvider",
    "PromptTemplate", "load_template", "SqliteCache", "open_cache",
]
