__version__ = "0.1.0"

from .core import DeduplicationPipeline, DeduplicationResult, CrossDatasetResult
from .matchers import FuzzyMatcher
from .validators import LLMValidator, RuleBasedValidator
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
    'simple_remove_duplicates'
]
