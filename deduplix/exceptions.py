"""
Custom exception hierarchy for deduplix operations.

Provides specific exception types for different failure modes,
enabling better error handling and debugging.
"""

from typing import Optional, Dict, Any, List


class DeduplixError(Exception):
    """
    Base exception class for all deduplix operations.

    Provides common functionality for error reporting and context.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None
    ):
        """
        Initialize base deduplix error

        Parameters
        ----------
        message : str
            Human-readable error message
        error_code : Optional[str]
            Machine-readable error code for programmatic handling
        context : Optional[Dict[str, Any]]
            Additional context information about the error
        suggestions : Optional[List[str]]
            List of suggestions for resolving the error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.suggestions = suggestions or []

    def __str__(self) -> str:
        """Return formatted error message with context"""
        parts = [self.message]

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")

        if self.suggestions:
            suggestions_str = "; ".join(self.suggestions)
            parts.append(f"Suggestions: {suggestions_str}")

        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization"""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'context': self.context,
            'suggestions': self.suggestions
        }


class DataValidationError(DeduplixError):
    """
    Raised when input data fails validation.

    Indicates problems with data format, content, or structure
    that prevent processing from continuing.
    """

    def __init__(
        self,
        message: str,
        validation_errors: Optional[List[str]] = None,
        invalid_rows: Optional[List[int]] = None,
        **kwargs
    ):
        """
        Initialize data validation error

        Parameters
        ----------
        message : str
            Error message
        validation_errors : Optional[List[str]]
            Specific validation errors found
        invalid_rows : Optional[List[int]]
            Row indices that failed validation
        **kwargs
            Additional arguments passed to base class
        """
        super().__init__(message, **kwargs)
        self.validation_errors = validation_errors or []
        self.invalid_rows = invalid_rows or []

        # Add validation details to context
        if self.validation_errors:
            self.context['validation_errors'] = len(self.validation_errors)
        if self.invalid_rows:
            self.context['invalid_rows'] = len(self.invalid_rows)

        # Add common suggestions
        if not self.suggestions:
            self.suggestions = [
                "Check input data format and column names",
                "Ensure required columns are present and properly formatted",
                "Remove or fix invalid rows"
            ]


class MatchingError(DeduplixError):
    """
    Raised during entity matching operations.

    Indicates failures in similarity computation, indexing,
    or other matching-related processes.
    """

    def __init__(
        self,
        message: str,
        entity_count: Optional[int] = None,
        failed_pairs: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if entity_count is not None:
            self.context['entity_count'] = entity_count
        if failed_pairs is not None:
            self.context['failed_pairs'] = failed_pairs

        if not self.suggestions:
            self.suggestions = [
                "Check entity data quality and completeness",
                "Verify matching configuration parameters",
                "Consider reducing dataset size for troubleshooting"
            ]


class ValidationError(DeduplixError):
    """
    Raised during match validation (LLM or rule-based).

    Indicates failures in the validation stage of the pipeline.
    """

    def __init__(
        self,
        message: str,
        validation_stage: Optional[str] = None,
        failed_validations: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if validation_stage:
            self.context['validation_stage'] = validation_stage
        if failed_validations is not None:
            self.context['failed_validations'] = failed_validations

        if not self.suggestions:
            self.suggestions = [
                "Check validation configuration",
                "Verify API credentials if using LLM validation",
                "Review validation rules and thresholds"
            ]


class CheckpointError(DeduplixError):
    """
    Raised during checkpoint operations.

    Indicates failures in saving, loading, or managing
    checkpoint data.
    """

    def __init__(
        self,
        message: str,
        checkpoint_stage: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if checkpoint_stage:
            self.context['checkpoint_stage'] = checkpoint_stage
        if checkpoint_path:
            self.context['checkpoint_path'] = checkpoint_path

        if not self.suggestions:
            self.suggestions = [
                "Check disk space and write permissions",
                "Verify checkpoint directory exists and is accessible",
                "Consider disabling checkpointing if issues persist"
            ]


class ConfigurationError(DeduplixError):
    """
    Raised for configuration-related errors.

    Indicates problems with configuration files, parameters,
    or system setup.
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if config_key:
            self.context['config_key'] = config_key
        if config_value is not None:
            self.context['config_value'] = str(config_value)

        if not self.suggestions:
            self.suggestions = [
                "Check configuration file format and values",
                "Ensure all required parameters are provided",
                "Refer to documentation for valid configuration options"
            ]


class SecurityError(DeduplixError):
    """
    Raised for security-related violations.

    Indicates potential security threats, rate limit violations,
    or unsafe operations.
    """

    def __init__(
        self,
        message: str,
        security_violation_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if security_violation_type:
            self.context['violation_type'] = security_violation_type

        if not self.suggestions:
            self.suggestions = [
                "Review input data for malicious content",
                "Check rate limiting configuration",
                "Ensure API keys and credentials are secure"
            ]


class ResourceError(DeduplixError):
    """
    Raised when resource limits are exceeded.

    Indicates memory, processing time, or other resource
    constraints have been violated.
    """

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        current_usage: Optional[float] = None,
        limit: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if resource_type:
            self.context['resource_type'] = resource_type
        if current_usage is not None:
            self.context['current_usage'] = current_usage
        if limit is not None:
            self.context['limit'] = limit

        if not self.suggestions:
            self.suggestions = [
                "Reduce dataset size or batch size",
                "Increase resource limits if possible",
                "Use chunked processing for large datasets"
            ]


class ModelError(DeduplixError):
    """
    Raised for model-related errors (LLM, ML models).

    Indicates failures in model loading, inference,
    or configuration.
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if model_name:
            self.context['model_name'] = model_name
        if provider:
            self.context['provider'] = provider

        if not self.suggestions:
            self.suggestions = [
                "Check model name and provider configuration",
                "Verify API credentials and access permissions",
                "Ensure model is available and accessible"
            ]


class NetworkError(DeduplixError):
    """
    Raised for network-related failures.

    Indicates problems with API calls, network connectivity,
    or remote service availability.
    """

    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        retry_count: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if endpoint:
            self.context['endpoint'] = endpoint
        if status_code is not None:
            self.context['status_code'] = status_code
        if retry_count is not None:
            self.context['retry_count'] = retry_count

        if not self.suggestions:
            self.suggestions = [
                "Check network connectivity",
                "Verify API endpoint and credentials",
                "Consider increasing timeout values"
            ]


class DataIntegrityError(DeduplixError):
    """
    Raised when data integrity violations are detected.

    Indicates corruption, inconsistency, or unexpected
    changes in data during processing.
    """

    def __init__(
        self,
        message: str,
        expected_checksum: Optional[str] = None,
        actual_checksum: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if expected_checksum:
            self.context['expected_checksum'] = expected_checksum
        if actual_checksum:
            self.context['actual_checksum'] = actual_checksum

        if not self.suggestions:
            self.suggestions = [
                "Verify input data has not been modified",
                "Check for memory corruption or storage issues",
                "Re-run processing with fresh input data"
            ]


# Exception mapping for easy lookup
EXCEPTION_MAP = {
    'data_validation': DataValidationError,
    'matching': MatchingError,
    'validation': ValidationError,
    'checkpoint': CheckpointError,
    'configuration': ConfigurationError,
    'security': SecurityError,
    'resource': ResourceError,
    'model': ModelError,
    'network': NetworkError,
    'data_integrity': DataIntegrityError,
}


def create_error(
    error_type: str,
    message: str,
    **kwargs
) -> DeduplixError:
    """
    Create a specific exception type by name

    Parameters
    ----------
    error_type : str
        Type of error to create
    message : str
        Error message
    **kwargs
        Additional arguments for the exception

    Returns
    -------
    DeduplixError
        Specific exception instance

    Examples
    --------
    >>> error = create_error('data_validation', 'Invalid data format')
    >>> raise error
    """
    exception_class = EXCEPTION_MAP.get(error_type, DeduplixError)
    return exception_class(message, **kwargs)


def handle_and_reraise(
    func_name: str,
    original_exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    suggestions: Optional[List[str]] = None
) -> None:
    """
    Handle generic exceptions and reraise as deduplix exceptions

    Parameters
    ----------
    func_name : str
        Name of the function where error occurred
    original_exception : Exception
        Original exception that was caught
    context : Optional[Dict[str, Any]]
        Additional context information
    suggestions : Optional[List[str]]
        Suggestions for resolving the error

    Raises
    ------
    DeduplixError
        Appropriate deduplix exception based on original exception type
    """
    message = f"Error in {func_name}: {str(original_exception)}"
    error_context = context or {}
    error_context['original_error_type'] = type(original_exception).__name__

    # Map common exception types to deduplix exceptions
    if isinstance(original_exception, (ValueError, TypeError)):
        raise DataValidationError(
            message,
            context=error_context,
            suggestions=suggestions
        ) from original_exception
    elif isinstance(original_exception, FileNotFoundError):
        raise CheckpointError(
            message,
            context=error_context,
            suggestions=suggestions
        ) from original_exception
    elif isinstance(original_exception, MemoryError):
        raise ResourceError(
            message,
            resource_type='memory',
            context=error_context,
            suggestions=suggestions
        ) from original_exception
    elif isinstance(original_exception, TimeoutError):
        raise NetworkError(
            message,
            context=error_context,
            suggestions=suggestions
        ) from original_exception
    else:
        # Generic deduplix error for unknown types
        raise DeduplixError(
            message,
            context=error_context,
            suggestions=suggestions
        ) from original_exception