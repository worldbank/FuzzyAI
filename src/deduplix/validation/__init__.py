"""
Data validation module for deduplix.

Provides comprehensive input validation for deduplication operations including:
- DataFrame schema validation
- Data type checking and conversion
- Null/empty value handling
- Input sanitization for security
"""

from .validators import (
    validate_deduplication_input,
    validate_cross_dataset_input,
    sanitize_text_input
)
from .schemas import (
    EntityDataSchema,
    DeduplicationInputSchema,
    CrossDatasetInputSchema
)

__all__ = [
    'validate_deduplication_input',
    'validate_cross_dataset_input',
    'sanitize_text_input',
    'EntityDataSchema',
    'DeduplicationInputSchema',
    'CrossDatasetInputSchema'
]