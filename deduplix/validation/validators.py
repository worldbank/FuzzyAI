"""
Data validation functions for deduplix operations.

Provides comprehensive validation for DataFrames and input parameters,
including data quality checks, type validation, and sanitization.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import re
import warnings
from pathlib import Path

from .schemas import (
    EntityDataSchema,
    DeduplicationInputSchema,
    CrossDatasetInputSchema,
    DataFrameValidationSchema
)


class DataValidationError(Exception):
    """Raised when data validation fails"""
    pass


class DataQualityWarning(UserWarning):
    """Warning for data quality issues that don't prevent processing"""
    pass


def sanitize_text_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize text input for security and consistency.

    Parameters
    ----------
    text : str
        Text to sanitize
    max_length : int
        Maximum allowed length for text

    Returns
    -------
    str
        Sanitized text

    Raises
    ------
    DataValidationError
        If text is invalid or too long
    """
    if not isinstance(text, str):
        text = str(text)

    # Check length
    if len(text) > max_length:
        raise DataValidationError(f"Text too long: {len(text)} characters (max: {max_length})")

    # Remove control characters except newline and tab
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Basic XSS prevention (remove potentially dangerous patterns)
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'vbscript:',
        r'on\w+\s*=',
    ]

    for pattern in dangerous_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    return text


def validate_dataframe_structure(
    df: pd.DataFrame,
    required_columns: List[str],
    min_rows: int = 1,
    max_rows: Optional[int] = None,
    null_policy: str = "warn"
) -> pd.DataFrame:
    """
    Validate DataFrame structure and basic content requirements.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : List[str]
        Columns that must exist in the DataFrame
    min_rows : int
        Minimum number of rows required
    max_rows : Optional[int]
        Maximum number of rows allowed
    null_policy : str
        How to handle null values: 'strict', 'warn', or 'ignore'

    Returns
    -------
    pd.DataFrame
        Validated DataFrame (may be modified for data quality)

    Raises
    ------
    DataValidationError
        If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected pandas DataFrame, got {type(df)}")

    if df.empty:
        raise DataValidationError("DataFrame is empty")

    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")

    # Check row count constraints
    if len(df) < min_rows:
        raise DataValidationError(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")

    if max_rows is not None and len(df) > max_rows:
        raise DataValidationError(f"DataFrame has {len(df)} rows, maximum allowed: {max_rows}")

    # Handle null values according to policy
    if null_policy != "ignore":
        null_counts = df[required_columns].isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]

        if len(columns_with_nulls) > 0:
            null_summary = {col: int(count) for col, count in columns_with_nulls.items()}

            if null_policy == "strict":
                raise DataValidationError(f"Null values found in required columns: {null_summary}")
            elif null_policy == "warn":
                warnings.warn(
                    f"Null values found in required columns: {null_summary}. "
                    "These rows will be filtered out during processing.",
                    DataQualityWarning
                )

    return df


def validate_entity_data(df: pd.DataFrame, id_column: str, name_column: str) -> pd.DataFrame:
    """
    Validate individual entity data using Pydantic schemas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing entity data
    id_column : str
        Name of the ID column
    name_column : str
        Name of the name column

    Returns
    -------
    pd.DataFrame
        Validated and cleaned DataFrame

    Raises
    ------
    DataValidationError
        If entity validation fails
    """
    validated_rows = []
    invalid_rows = []

    for idx, row in df.iterrows():
        try:
            # Extract and validate entity data
            entity_data = {
                'id': row[id_column],
                'name': row[name_column]
            }

            # Validate using Pydantic schema
            validated_entity = EntityDataSchema(**entity_data)

            # Update row with validated data
            row_dict = row.to_dict()
            row_dict[id_column] = validated_entity.id
            row_dict[name_column] = validated_entity.name
            validated_rows.append(row_dict)

        except Exception as e:
            invalid_rows.append({
                'row_index': idx,
                'id': row.get(id_column, 'N/A'),
                'name': row.get(name_column, 'N/A'),
                'error': str(e)
            })

    # Report validation results
    if invalid_rows:
        error_summary = f"Found {len(invalid_rows)} invalid entities out of {len(df)} total"

        if len(invalid_rows) == len(df):
            # All rows invalid - this is a critical error
            raise DataValidationError(f"{error_summary}. First error: {invalid_rows[0]['error']}")
        else:
            # Some rows invalid - warn and continue with valid rows
            warnings.warn(
                f"{error_summary}. Invalid rows will be excluded. "
                f"First error: {invalid_rows[0]['error']}",
                DataQualityWarning
            )

    if not validated_rows:
        raise DataValidationError("No valid entities remaining after validation")

    return pd.DataFrame(validated_rows)


def validate_deduplication_input(
    df: pd.DataFrame,
    id_column: str = "id",
    name_column: str = "name",
    additional_columns: Optional[List[str]] = None,
    threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Comprehensive validation for deduplication pipeline input.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with entities
    id_column : str
        Name of the ID column
    name_column : str
        Name of the name column
    additional_columns : Optional[List[str]]
        Additional columns to validate
    threshold : Optional[float]
        Similarity threshold to validate

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Validated DataFrame and validation metadata

    Raises
    ------
    DataValidationError
        If validation fails
    """
    validation_metadata = {
        'original_row_count': len(df),
        'validation_warnings': [],
        'columns_validated': [],
        'data_quality_score': 0.0
    }

    # Validate input parameters using Pydantic schema
    try:
        input_schema = DeduplicationInputSchema(
            entity_count=len(df),
            id_column=id_column,
            name_column=name_column,
            additional_columns=additional_columns,
            threshold=threshold
        )
    except Exception as e:
        raise DataValidationError(f"Invalid input parameters: {e}")

    # Validate DataFrame structure
    required_columns = [id_column, name_column]
    if additional_columns:
        required_columns.extend(additional_columns)

    df_validated = validate_dataframe_structure(
        df=df,
        required_columns=required_columns,
        min_rows=2,  # Need at least 2 entities for deduplication
        null_policy="warn"
    )

    validation_metadata['columns_validated'] = required_columns

    # Validate individual entity data
    df_validated = validate_entity_data(df_validated, id_column, name_column)

    # Additional data quality checks
    quality_score = _calculate_data_quality_score(df_validated, id_column, name_column)
    validation_metadata['data_quality_score'] = quality_score
    validation_metadata['final_row_count'] = len(df_validated)

    # Check for duplicate IDs
    duplicate_ids = df_validated[id_column].duplicated()
    if duplicate_ids.any():
        dup_count = duplicate_ids.sum()
        validation_metadata['validation_warnings'].append(f"Found {dup_count} duplicate IDs")
        warnings.warn(f"Found {dup_count} duplicate entity IDs", DataQualityWarning)

    # Sanitize text data
    df_validated = _sanitize_dataframe_text(df_validated, [name_column])

    return df_validated, validation_metadata


def validate_cross_dataset_input(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    id_column1: str = "id",
    name_column1: str = "name",
    id_column2: str = "id",
    name_column2: str = "name",
    name_columns1: Optional[List[str]] = None,
    name_columns2: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Validate input for cross-dataset matching.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames to validate for cross-dataset matching
    id_column1, id_column2 : str
        ID column names for each dataset
    name_column1, name_column2 : str
        Name column names for each dataset
    name_columns1, name_columns2 : Optional[List[str]]
        Multiple name columns for each dataset

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]
        Validated DataFrames and validation metadata

    Raises
    ------
    DataValidationError
        If validation fails
    """
    # Validate input parameters
    try:
        cross_schema = CrossDatasetInputSchema(
            df1_entity_count=len(df1),
            df2_entity_count=len(df2),
            id_column1=id_column1,
            name_column1=name_column1,
            id_column2=id_column2,
            name_column2=name_column2,
            name_columns1=name_columns1,
            name_columns2=name_columns2
        )
    except Exception as e:
        raise DataValidationError(f"Invalid cross-dataset parameters: {e}")

    validation_metadata = {
        'df1_original_rows': len(df1),
        'df2_original_rows': len(df2),
        'validation_warnings': [],
        'df1_quality_score': 0.0,
        'df2_quality_score': 0.0
    }

    # Validate each dataset individually
    df1_validated, df1_meta = validate_deduplication_input(
        df1, id_column1, name_column1
    )
    df2_validated, df2_meta = validate_deduplication_input(
        df2, id_column2, name_column2
    )

    # Merge metadata
    validation_metadata.update({
        'df1_final_rows': len(df1_validated),
        'df2_final_rows': len(df2_validated),
        'df1_quality_score': df1_meta['data_quality_score'],
        'df2_quality_score': df2_meta['data_quality_score'],
    })

    validation_metadata['validation_warnings'].extend(df1_meta['validation_warnings'])
    validation_metadata['validation_warnings'].extend(df2_meta['validation_warnings'])

    return df1_validated, df2_validated, validation_metadata


def _calculate_data_quality_score(df: pd.DataFrame, id_column: str, name_column: str) -> float:
    """
    Calculate a data quality score based on various metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to score
    id_column : str
        ID column name
    name_column : str
        Name column name

    Returns
    -------
    float
        Data quality score between 0.0 and 1.0
    """
    if df.empty:
        return 0.0

    score_components = []

    # 1. Completeness (non-null values)
    completeness = 1.0 - (df[[id_column, name_column]].isnull().sum().sum() / (len(df) * 2))
    score_components.append(completeness * 0.3)

    # 2. Uniqueness (unique IDs)
    uniqueness = df[id_column].nunique() / len(df) if len(df) > 0 else 0
    score_components.append(uniqueness * 0.3)

    # 3. Name quality (non-empty, reasonable length)
    name_series = df[name_column].dropna()
    if len(name_series) > 0:
        valid_names = name_series[
            (name_series.str.len() > 0) &
            (name_series.str.len() <= 1000) &
            (name_series.str.strip() != '')
        ]
        name_quality = len(valid_names) / len(name_series)
    else:
        name_quality = 0.0
    score_components.append(name_quality * 0.4)

    return sum(score_components)


def _sanitize_dataframe_text(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    """
    Sanitize text columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to sanitize
    text_columns : List[str]
        Columns containing text data to sanitize

    Returns
    -------
    pd.DataFrame
        DataFrame with sanitized text
    """
    df_sanitized = df.copy()

    for col in text_columns:
        if col in df_sanitized.columns:
            df_sanitized[col] = df_sanitized[col].apply(
                lambda x: sanitize_text_input(str(x)) if pd.notna(x) else x
            )

    return df_sanitized