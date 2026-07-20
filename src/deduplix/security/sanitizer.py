"""
Security sanitizer for input validation and XSS prevention.

Provides comprehensive input sanitization for text data,
API keys, and other user-provided content.
"""

import re
import html
import unicodedata
from typing import Optional, List, Dict, Any, Union
import warnings
from pathlib import Path


class SecuritySanitizer:
    """
    Comprehensive security sanitizer for various input types.

    Provides methods to sanitize text inputs, validate API keys,
    and prevent common security vulnerabilities.
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize security sanitizer

        Parameters
        ----------
        strict_mode : bool
            If True, applies stricter sanitization rules
        """
        self.strict_mode = strict_mode

        # Dangerous patterns for XSS and injection prevention
        self.xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>.*?</iframe>',
            r'<object[^>]*>.*?</object>',
            r'<embed[^>]*>.*?</embed>',
            r'<link[^>]*>',
            r'<meta[^>]*>',
            r'<style[^>]*>.*?</style>',
        ]

        # SQL injection patterns
        self.sql_injection_patterns = [
            r'(\b(ALTER|CREATE|DELETE|DROP|EXEC(UTE)?|INSERT|SELECT|UNION|UPDATE)\b)',
            r'(;|\||&|\$|`)',
            r'(\'|"|\-\-|/\*|\*/)',
        ]

        # Path traversal patterns
        self.path_traversal_patterns = [
            r'\.\.',
            r'[/\\]+(etc|proc|sys|dev|tmp|var)[/\\]+',
            r'[/\\]+\.+[/\\]+',
        ]

        # Control character patterns (except allowed ones)
        self.control_char_pattern = re.compile(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]')

    def sanitize_text(
        self,
        text: str,
        max_length: int = 10000,
        allow_html: bool = False,
        remove_control_chars: bool = True,
        normalize_unicode: bool = True
    ) -> str:
        """
        Comprehensive text sanitization

        Parameters
        ----------
        text : str
            Text to sanitize
        max_length : int
            Maximum allowed length
        allow_html : bool
            Whether to allow HTML tags (they will be escaped)
        remove_control_chars : bool
            Whether to remove control characters
        normalize_unicode : bool
            Whether to normalize unicode characters

        Returns
        -------
        str
            Sanitized text

        Raises
        ------
        ValueError
            If text exceeds maximum length or contains dangerous content
        """
        if not isinstance(text, str):
            text = str(text)

        # Check length
        if len(text) > max_length:
            if self.strict_mode:
                raise ValueError(f"Text too long: {len(text)} characters (max: {max_length})")
            else:
                text = text[:max_length]
                warnings.warn(f"Text truncated to {max_length} characters", UserWarning)

        # Remove control characters
        if remove_control_chars:
            text = self.control_char_pattern.sub('', text)

        # Normalize Unicode
        if normalize_unicode:
            text = unicodedata.normalize('NFKC', text)

        # XSS prevention
        if not allow_html:
            for pattern in self.xss_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        else:
            # Escape HTML entities
            text = html.escape(text)

        # SQL injection prevention (basic)
        if self.strict_mode:
            for pattern in self.sql_injection_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    raise ValueError(f"Potentially dangerous SQL content detected")

        # Path traversal prevention
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                if self.strict_mode:
                    raise ValueError("Path traversal attempt detected")
                else:
                    text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def validate_api_key(
        self,
        api_key: str,
        expected_format: Optional[str] = None,
        min_length: int = 10,
        max_length: int = 256
    ) -> bool:
        """
        Validate API key format and security

        Parameters
        ----------
        api_key : str
            API key to validate
        expected_format : Optional[str]
            Expected format regex pattern
        min_length : int
            Minimum key length
        max_length : int
            Maximum key length

        Returns
        -------
        bool
            True if API key is valid format

        Raises
        ------
        ValueError
            If API key format is invalid
        """
        if not isinstance(api_key, str):
            raise ValueError("API key must be a string")

        if len(api_key) < min_length or len(api_key) > max_length:
            raise ValueError(f"API key length must be between {min_length} and {max_length} characters")

        # Check for suspicious patterns
        if any(char in api_key for char in [' ', '\n', '\r', '\t']):
            raise ValueError("API key contains whitespace characters")

        # Check for control characters
        if self.control_char_pattern.search(api_key):
            raise ValueError("API key contains control characters")

        # Check expected format if provided
        if expected_format and not re.match(expected_format, api_key):
            raise ValueError(f"API key does not match expected format")

        return True

    def sanitize_filename(self, filename: str, max_length: int = 255) -> str:
        """
        Sanitize filename for safe filesystem operations

        Parameters
        ----------
        filename : str
            Filename to sanitize
        max_length : int
            Maximum filename length

        Returns
        -------
        str
            Sanitized filename

        Raises
        ------
        ValueError
            If filename is invalid
        """
        if not isinstance(filename, str):
            filename = str(filename)

        # Remove path components
        filename = Path(filename).name

        # Remove dangerous characters
        dangerous_chars = r'<>:"/\\|?*\x00-\x1f'
        filename = re.sub(f'[{re.escape(dangerous_chars)}]', '_', filename)

        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')

        # Check length
        if len(filename) > max_length:
            if self.strict_mode:
                raise ValueError(f"Filename too long: {len(filename)} characters (max: {max_length})")
            else:
                # Preserve extension if possible
                path_obj = Path(filename)
                stem = path_obj.stem[:max_length - len(path_obj.suffix) - 1]
                filename = f"{stem}{path_obj.suffix}"

        # Ensure filename is not empty
        if not filename:
            raise ValueError("Filename cannot be empty after sanitization")

        # Check for reserved names (Windows)
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }

        if Path(filename).stem.upper() in reserved_names:
            filename = f"safe_{filename}"

        return filename

    def sanitize_config_value(
        self,
        value: Any,
        expected_type: type,
        allowed_values: Optional[List[Any]] = None,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None
    ) -> Any:
        """
        Sanitize configuration values

        Parameters
        ----------
        value : Any
            Configuration value to sanitize
        expected_type : type
            Expected type for the value
        allowed_values : Optional[List[Any]]
            List of allowed values (if restricted)
        min_value : Optional[Union[int, float]]
            Minimum value for numeric types
        max_value : Optional[Union[int, float]]
            Maximum value for numeric types

        Returns
        -------
        Any
            Sanitized configuration value

        Raises
        ------
        ValueError
            If value is invalid
        """
        # Type checking and conversion
        if not isinstance(value, expected_type):
            try:
                value = expected_type(value)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Cannot convert value to {expected_type.__name__}: {e}")

        # String sanitization
        if expected_type == str:
            value = self.sanitize_text(value, max_length=1000)

        # Range checking for numeric types
        if expected_type in (int, float):
            if min_value is not None and value < min_value:
                raise ValueError(f"Value {value} is below minimum {min_value}")
            if max_value is not None and value > max_value:
                raise ValueError(f"Value {value} is above maximum {max_value}")

        # Allowed values checking
        if allowed_values is not None and value not in allowed_values:
            raise ValueError(f"Value {value} not in allowed values: {allowed_values}")

        return value

    def sanitize_dataframe_columns(self, columns: List[str]) -> List[str]:
        """
        Sanitize DataFrame column names

        Parameters
        ----------
        columns : List[str]
            Column names to sanitize

        Returns
        -------
        List[str]
            Sanitized column names
        """
        sanitized = []

        for col in columns:
            if not isinstance(col, str):
                col = str(col)

            # Remove dangerous characters
            col = re.sub(r'[^\w\s\-_.]', '_', col)

            # Replace spaces with underscores
            col = re.sub(r'\s+', '_', col)

            # Remove leading/trailing underscores
            col = col.strip('_')

            # Ensure column name is not empty
            if not col:
                col = f"column_{len(sanitized)}"

            sanitized.append(col)

        return sanitized

    def check_for_secrets(self, text: str) -> List[Dict[str, str]]:
        """
        Check text for potential secrets or credentials

        Parameters
        ----------
        text : str
            Text to check for secrets

        Returns
        -------
        List[Dict[str, str]]
            List of potential secrets found with their types
        """
        secrets_found = []

        # Common secret patterns
        secret_patterns = {
            'api_key': r'api[_\-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9]{20,})',
            'password': r'password["\']?\s*[:=]\s*["\']?([^"\'\s]{8,})',
            'token': r'token["\']?\s*[:=]\s*["\']?([a-zA-Z0-9]{20,})',
            'secret': r'secret["\']?\s*[:=]\s*["\']?([a-zA-Z0-9]{16,})',
            'private_key': r'-----BEGIN [A-Z ]+PRIVATE KEY-----',
            'aws_access_key': r'AKIA[0-9A-Z]{16}',
            'aws_secret_key': r'aws[_\-]?secret[_\-]?access[_\-]?key',
        }

        for secret_type, pattern in secret_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                secrets_found.append({
                    'type': secret_type,
                    'value': match.group(1) if match.groups() else match.group(0),
                    'position': match.span()
                })

        return secrets_found

    def audit_log_entry(self, action: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create sanitized audit log entry

        Parameters
        ----------
        action : str
            Action being audited
        details : Dict[str, Any]
            Details about the action

        Returns
        -------
        Dict[str, Any]
            Sanitized audit log entry
        """
        import time

        # Sanitize action
        action = self.sanitize_text(action, max_length=100)

        # Sanitize details
        sanitized_details = {}
        for key, value in details.items():
            # Sanitize key
            key = self.sanitize_text(str(key), max_length=50)

            # Sanitize value based on type
            if isinstance(value, str):
                value = self.sanitize_text(value, max_length=500)
            elif isinstance(value, (int, float, bool)):
                pass  # Keep as-is
            else:
                value = self.sanitize_text(str(value), max_length=500)

            sanitized_details[key] = value

        return {
            'timestamp': time.time(),
            'action': action,
            'details': sanitized_details,
            'sanitized': True
        }