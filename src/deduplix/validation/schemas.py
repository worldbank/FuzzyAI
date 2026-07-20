"""
Pydantic schemas for data validation in deduplix operations.

Defines validation schemas for:
- Entity data validation
- Deduplication input parameters
- Cross-dataset matching parameters
"""

from typing import List, Union, Optional, Any, Dict
from pydantic import BaseModel, Field, validator, root_validator
import pandas as pd


class EntityDataSchema(BaseModel):
    """Schema for validating individual entity data"""

    id: Union[str, int] = Field(..., description="Entity identifier")
    name: str = Field(..., min_length=1, max_length=10000, description="Entity name")

    @validator('name')
    def validate_name_not_empty(cls, v):
        """Ensure name is not empty after stripping whitespace"""
        if not v or not str(v).strip():
            raise ValueError("Entity name cannot be empty or whitespace-only")
        return str(v).strip()

    @validator('id')
    def validate_id_format(cls, v):
        """Validate ID format and ensure it's not empty"""
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError("Entity ID cannot be None or empty")

        # Convert to string for consistency but preserve original type info
        if isinstance(v, (int, float)):
            if pd.isna(v):
                raise ValueError("Entity ID cannot be NaN")

        return v


class DeduplicationInputSchema(BaseModel):
    """Schema for validating deduplication pipeline input parameters"""

    entity_count: int = Field(..., ge=2, description="Number of entities in dataset")
    id_column: str = Field(default="id", min_length=1, description="ID column name")
    name_column: str = Field(default="name", min_length=1, description="Name column name")
    additional_columns: Optional[List[str]] = Field(default=None, description="Additional columns for matching")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Similarity threshold")

    @validator('entity_count')
    def validate_minimum_entities(cls, v):
        """Ensure at least 2 entities for deduplication"""
        if v < 2:
            raise ValueError("At least 2 entities required for deduplication")
        return v

    @validator('additional_columns')
    def validate_additional_columns(cls, v):
        """Validate additional columns list"""
        if v is not None:
            # Remove empty strings and duplicates
            v = list(set([col.strip() for col in v if col and col.strip()]))
            return v if v else None
        return v


class CrossDatasetInputSchema(BaseModel):
    """Schema for validating cross-dataset matching input parameters"""

    df1_entity_count: int = Field(..., ge=1, description="Number of entities in first dataset")
    df2_entity_count: int = Field(..., ge=1, description="Number of entities in second dataset")
    id_column1: str = Field(..., min_length=1, description="ID column name in first dataset")
    name_column1: str = Field(..., min_length=1, description="Name column name in first dataset")
    id_column2: str = Field(..., min_length=1, description="ID column name in second dataset")
    name_column2: str = Field(..., min_length=1, description="Name column name in second dataset")
    name_columns1: Optional[List[str]] = Field(default=None, description="Multiple name columns for first dataset")
    name_columns2: Optional[List[str]] = Field(default=None, description="Multiple name columns for second dataset")

    @root_validator(skip_on_failure=True)
    def validate_minimum_entities_cross(cls, values):
        """Ensure both datasets have at least 1 entity"""
        df1_count = values.get('df1_entity_count', 0)
        df2_count = values.get('df2_entity_count', 0)

        if df1_count == 0 or df2_count == 0:
            raise ValueError("Both datasets must contain at least 1 entity for cross-dataset matching")

        return values

    @validator('name_columns1', 'name_columns2')
    def validate_name_columns(cls, v):
        """Validate multiple name columns"""
        if v is not None:
            # Remove empty strings and duplicates, maintain order
            cleaned = []
            seen = set()
            for col in v:
                if col and col.strip() and col.strip() not in seen:
                    cleaned_col = col.strip()
                    cleaned.append(cleaned_col)
                    seen.add(cleaned_col)
            return cleaned if cleaned else None
        return v


class DataFrameValidationSchema(BaseModel):
    """Schema for validating DataFrame structure and content"""

    class Config:
        arbitrary_types_allowed = True

    required_columns: List[str] = Field(..., description="Required columns that must exist")
    min_rows: int = Field(default=1, ge=0, description="Minimum number of rows")
    max_rows: Optional[int] = Field(default=None, ge=1, description="Maximum number of rows")
    allowed_dtypes: Optional[Dict[str, List[str]]] = Field(default=None, description="Allowed data types per column")
    null_policy: str = Field(default="warn", pattern="^(strict|warn|ignore)$", description="How to handle null values")

    @validator('required_columns')
    def validate_required_columns(cls, v):
        """Validate required columns list"""
        if not v:
            raise ValueError("At least one required column must be specified")

        # Remove duplicates while preserving order
        seen = set()
        unique_cols = []
        for col in v:
            if col not in seen:
                unique_cols.append(col)
                seen.add(col)

        return unique_cols

    @root_validator(skip_on_failure=True)
    def validate_row_constraints(cls, values):
        """Validate row count constraints"""
        min_rows = values.get('min_rows', 1)
        max_rows = values.get('max_rows')

        if max_rows is not None and min_rows > max_rows:
            raise ValueError("min_rows cannot be greater than max_rows")

        return values