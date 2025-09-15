
__version__ = "0.1.0"

from .core import DeduplicationPipeline, DeduplicationResult
from .matchers import FuzzyMatcher
from .validators import LLMValidator, RuleBasedValidator
from .utils import load_config, create_sample_config

__all__ = [
    'DeduplicationPipeline',
    'DeduplicationResult',
    'FuzzyMatcher', 
    'LLMValidator',
    'RuleBasedValidator',
    'load_config',
    'create_sample_config'
]