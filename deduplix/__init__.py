__version__ = "0.1.0"

from .core import DeduplicationPipeline, DeduplicationResult,CrossDatasetResult
from .matchers import FuzzyMatcher
from .validators import LLMValidator, RuleBasedValidator
from .utils import (
<<<<<<< HEAD
    load_config,
    create_sample_config,
=======
    load_config, 
    create_sample_config, 
>>>>>>> c032314305f8e2afa96c70afb0030f76ad8c3a64
    remove_duplicates_after_deduplication,
    simple_remove_duplicates
)

__all__ = [
    'DeduplicationPipeline',
    'DeduplicationResult',
    'CrossDatasetResult',
<<<<<<< HEAD
    'FuzzyMatcher',
=======
    'FuzzyMatcher', 
>>>>>>> c032314305f8e2afa96c70afb0030f76ad8c3a64
    'LLMValidator',
    'RuleBasedValidator',
    'load_config',
    'create_sample_config',
    'remove_duplicates_after_deduplication',
    'simple_remove_duplicates'
]
