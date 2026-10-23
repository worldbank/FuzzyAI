__version__ = "0.2.0"

from .core import DeduplicationPipeline, DeduplicationResult, CrossDatasetResult
from .matchers import FuzzyMatcher
from .validators import LLMValidator, RuleBasedValidator
from .engine import EngineMatcher, Pipeline, PipelineResult, load_pipeline, register
from .utils import (
    load_config,
    create_sample_config,
    remove_duplicates_after_deduplication,
    simple_remove_duplicates
)

__all__ = [
    'DeduplicationPipeline',
    'DeduplicationResult',
    'CrossDatasetResult',
    'FuzzyMatcher',
    'LLMValidator',
    'RuleBasedValidator',
    'load_config',
    'create_sample_config',
    'remove_duplicates_after_deduplication',
    'simple_remove_duplicates',
    'Pipeline',
    'PipelineResult',
    'register',
    'load_pipeline',
    'EngineMatcher',
]
