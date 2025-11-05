"""
Tests for data validation functionality.
"""

import pytest
import pandas as pd
from pydantic import ValidationError as PydanticValidationError

from deduplix.validation import (
    validate_deduplication_input,
    validate_cross_dataset_input,
    EntityDataSchema,
    DeduplicationInputSchema,
    CrossDatasetInputSchema,
    sanitize_text_input
)
from deduplix.exceptions import DataValidationError


class TestEntityDataValidation:
    """Test entity data validation schemas"""

    def test_valid_entity_data(self):
        """Test validation of valid entity data"""
        valid_data = {
            'id': 'E123',
            'name': 'Apple Inc.',
            'country': 'USA',
            'industry': 'Technology'
        }

        # Should not raise exception
        entity = EntityDataSchema(**valid_data)
        assert entity.id == 'E123'
        assert entity.name == 'Apple Inc.'

    def test_invalid_entity_id(self):
        """Test validation with invalid entity ID"""
        invalid_data = {
            'id': '',  # Empty ID should fail
            'name': 'Apple Inc.'
        }

        with pytest.raises(PydanticValidationError) as exc_info:
            EntityDataSchema(**invalid_data)

        assert 'id' in str(exc_info.value)

    def test_invalid_entity_name(self):
        """Test validation with invalid entity name"""
        invalid_data = {
            'id': 'E123',
            'name': '   '  # Whitespace-only name should fail
        }

        with pytest.raises(PydanticValidationError) as exc_info:
            EntityDataSchema(**invalid_data)

        assert 'name' in str(exc_info.value)

    def test_name_length_validation(self):
        """Test name length validation"""
        # Very long name should fail
        long_name = 'A' * 1001  # Over 1000 character limit

        invalid_data = {
            'id': 'E123',
            'name': long_name
        }

        with pytest.raises(PydanticValidationError):
            EntityDataSchema(**invalid_data)

    def test_special_characters_in_name(self):
        """Test handling of special characters in names"""
        valid_data = {
            'id': 'E123',
            'name': 'AT&T Inc. (Français) #1'  # Should be allowed
        }

        entity = EntityDataSchema(**valid_data)
        assert entity.name == 'AT&T Inc. (Français) #1'


class TestDeduplicationInputValidation:
    """Test deduplication input validation"""

    def test_validate_valid_input(self, sample_entities_df):
        """Test validation of valid input"""
        validated_df, metadata = validate_deduplication_input(
            df=sample_entities_df,
            id_column='id',
            name_column='name'
        )

        # Should return cleaned dataframe
        assert isinstance(validated_df, pd.DataFrame)
        assert len(validated_df) <= len(sample_entities_df)

        # Check metadata
        assert 'original_row_count' in metadata
        assert 'final_row_count' in metadata
        assert 'data_quality_score' in metadata

    def test_validate_missing_id_column(self, sample_entities_df):
        """Test validation with missing ID column"""
        df_no_id = sample_entities_df.drop(columns=['id'])

        with pytest.raises(DataValidationError) as exc_info:
            validate_deduplication_input(
                df=df_no_id,
                id_column='id',
                name_column='name'
            )

        assert 'id' in str(exc_info.value).lower()

    def test_validate_missing_name_column(self, sample_entities_df):
        """Test validation with missing name column"""
        df_no_name = sample_entities_df.drop(columns=['name'])

        with pytest.raises(DataValidationError) as exc_info:
            validate_deduplication_input(
                df=df_no_name,
                id_column='id',
                name_column='name'
            )

        assert 'name' in str(exc_info.value).lower()

    def test_validate_empty_dataframe(self):
        """Test validation with empty dataframe"""
        empty_df = pd.DataFrame(columns=['id', 'name'])

        with pytest.raises(DataValidationError) as exc_info:
            validate_deduplication_input(
                df=empty_df,
                id_column='id',
                name_column='name'
            )

        assert 'empty' in str(exc_info.value).lower()

    def test_validate_duplicate_ids(self):
        """Test validation with duplicate IDs"""
        df_with_dupes = pd.DataFrame({
            'id': ['E1', 'E2', 'E1'],  # Duplicate ID
            'name': ['Apple', 'Microsoft', 'Google']
        })

        # Should remove duplicate IDs and warn
        validated_df, metadata = validate_deduplication_input(
            df=df_with_dupes,
            id_column='id',
            name_column='name'
        )

        assert len(validated_df) == 2  # One duplicate removed
        assert 'validation_warnings' in metadata
        assert len(metadata['validation_warnings']) > 0

    def test_validate_with_null_values(self):
        """Test validation with null values"""
        df_with_nulls = pd.DataFrame({
            'id': ['E1', 'E2', None, 'E4'],
            'name': ['Apple', None, 'Microsoft', 'Google']
        })

        validated_df, metadata = validate_deduplication_input(
            df=df_with_nulls,
            id_column='id',
            name_column='name'
        )

        # Should remove rows with null IDs or names
        assert len(validated_df) <= 2
        assert validated_df['id'].notna().all()
        assert validated_df['name'].notna().all()

    def test_validate_additional_columns(self, sample_entities_df):
        """Test validation with additional columns"""
        validated_df, metadata = validate_deduplication_input(
            df=sample_entities_df,
            id_column='id',
            name_column='name',
            additional_columns=['country', 'industry']
        )

        # Should include additional columns
        assert 'country' in validated_df.columns
        assert 'industry' in validated_df.columns

    def test_validate_missing_additional_columns(self, sample_entities_df):
        """Test validation with missing additional columns"""
        with pytest.raises(DataValidationError) as exc_info:
            validate_deduplication_input(
                df=sample_entities_df,
                id_column='id',
                name_column='name',
                additional_columns=['nonexistent_column']
            )

        assert 'nonexistent_column' in str(exc_info.value)


class TestCrossDatasetValidation:
    """Test cross-dataset input validation"""

    def test_validate_cross_dataset_valid(self, sample_cross_df1, sample_cross_df2):
        """Test valid cross-dataset validation"""
        df1_val, df2_val, metadata = validate_cross_dataset_input(
            df1=sample_cross_df1,
            df2=sample_cross_df2,
            id_column1='company_id',
            name_column1='company_name',
            id_column2='entity_id',
            name_column2='entity_name'
        )

        assert isinstance(df1_val, pd.DataFrame)
        assert isinstance(df2_val, pd.DataFrame)
        assert 'df1_final_rows' in metadata
        assert 'df2_final_rows' in metadata

    def test_validate_cross_dataset_with_name_columns(self):
        """Test cross-dataset validation with multiple name columns"""
        df1 = pd.DataFrame({
            'id': ['C1', 'C2'],
            'legal_name': ['Apple Inc.', 'Microsoft Corp'],
            'short_name': ['Apple', 'MSFT']
        })

        df2 = pd.DataFrame({
            'id': ['E1', 'E2'],
            'name': ['Apple Incorporated', 'Microsoft Corporation']
        })

        df1_val, df2_val, metadata = validate_cross_dataset_input(
            df1=df1, df2=df2,
            id_column1='id', name_column1='legal_name',
            id_column2='id', name_column2='name',
            name_columns1=['legal_name', 'short_name']
        )

        # Should validate successfully
        assert len(df1_val) == 2
        assert len(df2_val) == 2

    def test_validate_cross_dataset_missing_columns(self, sample_cross_df1):
        """Test cross-dataset validation with missing columns in df2"""
        df2_invalid = pd.DataFrame({
            'wrong_id': ['E1', 'E2'],
            'wrong_name': ['Apple', 'Microsoft']
        })

        with pytest.raises(DataValidationError) as exc_info:
            validate_cross_dataset_input(
                df1=sample_cross_df1,
                df2=df2_invalid,
                id_column1='company_id',
                name_column1='company_name',
                id_column2='entity_id',  # This column doesn't exist
                name_column2='entity_name'  # This column doesn't exist
            )

        assert 'entity_id' in str(exc_info.value) or 'entity_name' in str(exc_info.value)


class TestTextSanitization:
    """Test text sanitization functionality"""

    def test_sanitize_normal_text(self):
        """Test sanitizing normal text"""
        text = "Apple Inc. - Technology Company"
        sanitized = sanitize_text_input(text)
        assert sanitized == text

    def test_sanitize_html_content(self):
        """Test sanitizing HTML content"""
        text = "<script>alert('xss')</script>Apple Inc."
        sanitized = sanitize_text_input(text)
        # HTML tags should be removed
        assert "<script>" not in sanitized
        assert "Apple Inc." in sanitized

    def test_sanitize_control_characters(self):
        """Test sanitizing control characters"""
        text = "Apple\x00Inc.\x1f"  # Contains null and control chars
        sanitized = sanitize_text_input(text)
        # Control characters should be removed
        assert "\x00" not in sanitized
        assert "\x1f" not in sanitized
        assert "Apple" in sanitized and "Inc." in sanitized

    def test_sanitize_excessive_whitespace(self):
        """Test sanitizing excessive whitespace"""
        text = "Apple    Inc.   \n\n   Corporation"
        sanitized = sanitize_text_input(text)
        # Should normalize whitespace
        assert "Apple Inc. Corporation" == sanitized

    def test_sanitize_very_long_text(self):
        """Test sanitizing very long text"""
        long_text = "A" * 2000  # Very long text
        sanitized = sanitize_text_input(long_text, max_length=1000)
        # Should be truncated
        assert len(sanitized) <= 1000

    def test_sanitize_unicode_text(self):
        """Test sanitizing unicode text"""
        text = "Société Générale - François & Müller"
        sanitized = sanitize_text_input(text)
        # Unicode should be preserved
        assert "Société" in sanitized
        assert "François" in sanitized
        assert "Müller" in sanitized

    def test_sanitize_empty_text(self):
        """Test sanitizing empty or whitespace-only text"""
        empty_sanitized = sanitize_text_input("")
        whitespace_sanitized = sanitize_text_input("   \n\t  ")

        assert empty_sanitized == ""
        assert whitespace_sanitized == ""

    def test_sanitize_with_strict_mode(self):
        """Test sanitizing with strict mode"""
        text = "Apple & Sons (2023) - $1M revenue"
        sanitized = sanitize_text_input(text, strict_mode=True)
        # In strict mode, should be more aggressive
        assert sanitized is not None