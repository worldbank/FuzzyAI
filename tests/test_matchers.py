"""
Tests for matching functionality.
"""

import pytest
import pandas as pd
from unittest.mock import patch

from deduplix.matchers import FuzzyMatcher
from deduplix.core import MatchResult
from deduplix.exceptions import MatchingError


class TestFuzzyMatcher:
    """Test fuzzy matching functionality"""

    def test_matcher_creation_default(self):
        """Test basic matcher creation with defaults"""
        matcher = FuzzyMatcher()

        assert matcher.threshold == 80.0
        assert matcher.scorer == 'token_sort_ratio'
        assert matcher.n_workers == 4
        assert matcher.max_matches_per_entity == 100

    def test_matcher_creation_custom(self):
        """Test matcher creation with custom parameters"""
        matcher = FuzzyMatcher(
            threshold=85.0,
            scorer='ratio',
            n_workers=2,
            max_matches_per_entity=50
        )

        assert matcher.threshold == 85.0
        assert matcher.scorer == 'ratio'
        assert matcher.n_workers == 2
        assert matcher.max_matches_per_entity == 50

    def test_find_matches_basic(self, sample_entities_df):
        """Test basic matching functionality"""
        matcher = FuzzyMatcher(threshold=80.0)

        result = matcher.find_matches(sample_entities_df, id_column='id', name_column='name')

        assert isinstance(result, MatchResult)
        assert isinstance(result.pairs, pd.DataFrame)

        # Should find some matches (Apple Inc. vs Apple Incorporated, etc.)
        assert len(result.pairs) > 0

        # Check result structure
        expected_columns = ['id1', 'id2', 'name1', 'name2', 'similarity_score']
        for col in expected_columns:
            assert col in result.pairs.columns

    def test_find_matches_no_duplicates(self):
        """Test matching with dataset having no duplicates"""
        unique_df = pd.DataFrame({
            'id': ['E1', 'E2', 'E3'],
            'name': ['Apple Inc.', 'Microsoft Corp', 'Amazon LLC']
        })

        matcher = FuzzyMatcher(threshold=95.0)  # High threshold
        result = matcher.find_matches(unique_df, id_column='id', name_column='name')

        # Might have no matches or very few with high threshold
        assert isinstance(result, MatchResult)
        assert len(result.pairs) == 0 or len(result.pairs) < len(unique_df)

    def test_find_matches_with_additional_columns(self, sample_entities_df):
        """Test matching with additional columns for context"""
        matcher = FuzzyMatcher(threshold=80.0)

        result = matcher.find_matches(
            sample_entities_df,
            id_column='id',
            name_column='name',
            additional_columns=['country', 'industry']
        )

        # Should still work and might improve matching quality
        assert isinstance(result, MatchResult)

    def test_find_matches_empty_dataframe(self):
        """Test matching with empty dataframe"""
        empty_df = pd.DataFrame(columns=['id', 'name'])

        matcher = FuzzyMatcher()
        result = matcher.find_matches(empty_df, id_column='id', name_column='name')

        assert isinstance(result, MatchResult)
        assert len(result.pairs) == 0

    def test_find_matches_single_entity(self):
        """Test matching with single entity"""
        single_df = pd.DataFrame({
            'id': ['E1'],
            'name': ['Apple Inc.']
        })

        matcher = FuzzyMatcher()
        result = matcher.find_matches(single_df, id_column='id', name_column='name')

        assert isinstance(result, MatchResult)
        assert len(result.pairs) == 0  # Can't match with itself

    def test_different_scorers(self, sample_entities_df):
        """Test different fuzzy matching scorers"""
        scorers = ['ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio']

        for scorer in scorers:
            matcher = FuzzyMatcher(threshold=80.0, scorer=scorer)
            result = matcher.find_matches(sample_entities_df, id_column='id', name_column='name')

            assert isinstance(result, MatchResult)
            # All scorers should work, though results may vary

    def test_threshold_effect(self, sample_entities_df):
        """Test effect of different thresholds"""
        low_threshold_matcher = FuzzyMatcher(threshold=50.0)
        high_threshold_matcher = FuzzyMatcher(threshold=95.0)

        low_result = low_threshold_matcher.find_matches(sample_entities_df, id_column='id', name_column='name')
        high_result = high_threshold_matcher.find_matches(sample_entities_df, id_column='id', name_column='name')

        # Lower threshold should generally find more matches
        assert len(low_result.pairs) >= len(high_result.pairs)

    def test_max_matches_per_entity_limit(self):
        """Test max matches per entity limitation"""
        # Create dataset with many similar entities
        similar_df = pd.DataFrame({
            'id': [f'E{i}' for i in range(20)],
            'name': [f'Similar Company {i}' for i in range(20)]
        })

        matcher = FuzzyMatcher(threshold=60.0, max_matches_per_entity=5)
        result = matcher.find_matches(similar_df, id_column='id', name_column='name')

        # Check that no entity appears in more than max_matches_per_entity pairs
        if not result.pairs.empty:
            id1_counts = result.pairs['id1'].value_counts()
            id2_counts = result.pairs['id2'].value_counts()

            assert id1_counts.max() <= matcher.max_matches_per_entity
            assert id2_counts.max() <= matcher.max_matches_per_entity

    def test_find_cross_matches_basic(self, sample_cross_df1, sample_cross_df2):
        """Test cross-dataset matching"""
        matcher = FuzzyMatcher(threshold=80.0)

        result = matcher.find_cross_matches(
            df1=sample_cross_df1,
            df2=sample_cross_df2,
            id_column1='company_id',
            name_column1='company_name',
            id_column2='entity_id',
            name_column2='entity_name'
        )

        assert isinstance(result, MatchResult)
        assert isinstance(result.pairs, pd.DataFrame)

        # Check result structure for cross-matching
        if not result.pairs.empty:
            expected_columns = ['id1', 'id2', 'name1', 'name2', 'similarity_score']
            for col in expected_columns:
                assert col in result.pairs.columns

    def test_find_cross_matches_with_multiple_name_columns(self):
        """Test cross-dataset matching with multiple name columns"""
        df1 = pd.DataFrame({
            'id': ['C1', 'C2'],
            'legal_name': ['Apple Inc.', 'Microsoft Corporation'],
            'short_name': ['Apple', 'MSFT'],
            'trade_name': ['Apple Computer', 'Microsoft']
        })

        df2 = pd.DataFrame({
            'id': ['E1', 'E2', 'E3'],
            'entity_name': ['Apple Incorporated', 'Microsoft Corp', 'Google LLC']
        })

        matcher = FuzzyMatcher(threshold=80.0)

        result = matcher.find_cross_matches(
            df1=df1, df2=df2,
            id_column1='id', name_column1='legal_name',
            id_column2='id', name_column2='entity_name',
            name_columns1=['legal_name', 'short_name', 'trade_name']
        )

        assert isinstance(result, MatchResult)

        # Should find matches and include column information
        if not result.pairs.empty:
            # Check if matched_column1 is included (if implemented)
            # This depends on implementation details
            pass

    def test_find_cross_matches_no_matches(self):
        """Test cross-dataset matching with no matches"""
        df1 = pd.DataFrame({
            'id': ['C1', 'C2'],
            'name': ['Apple Inc.', 'Microsoft Corp']
        })

        df2 = pd.DataFrame({
            'id': ['E1', 'E2'],
            'name': ['Google LLC', 'Amazon Inc.']
        })

        matcher = FuzzyMatcher(threshold=95.0)  # High threshold

        result = matcher.find_cross_matches(
            df1=df1, df2=df2,
            id_column1='id', name_column1='name',
            id_column2='id', name_column2='name'
        )

        assert isinstance(result, MatchResult)
        assert len(result.pairs) == 0

    def test_parallel_processing(self, sample_entities_df):
        """Test parallel processing with different worker counts"""
        # Test with 1 worker (sequential)
        matcher_sequential = FuzzyMatcher(threshold=80.0, n_workers=1)
        result_sequential = matcher_sequential.find_matches(sample_entities_df, id_column='id', name_column='name')

        # Test with multiple workers
        matcher_parallel = FuzzyMatcher(threshold=80.0, n_workers=4)
        result_parallel = matcher_parallel.find_matches(sample_entities_df, id_column='id', name_column='name')

        # Results should be similar (order might differ)
        assert len(result_sequential.pairs) == len(result_parallel.pairs)

    def test_matching_with_special_characters(self):
        """Test matching with special characters in names"""
        special_df = pd.DataFrame({
            'id': ['E1', 'E2', 'E3', 'E4'],
            'name': [
                'AT&T Inc.',
                'A.T.&T. Incorporated',
                'McDonald\'s Corporation',
                'McDonalds Corp.'
            ]
        })

        matcher = FuzzyMatcher(threshold=80.0)
        result = matcher.find_matches(special_df, id_column='id', name_column='name')

        # Should handle special characters gracefully
        assert isinstance(result, MatchResult)
        # Should find matches between AT&T variants and McDonald's variants
        assert len(result.pairs) > 0

    def test_matching_with_unicode_characters(self):
        """Test matching with unicode characters"""
        unicode_df = pd.DataFrame({
            'id': ['E1', 'E2', 'E3', 'E4'],
            'name': [
                'Société Générale',
                'Societe Generale',
                'Müller & Co',
                'Mueller & Company'
            ]
        })

        matcher = FuzzyMatcher(threshold=75.0)
        result = matcher.find_matches(unicode_df, id_column='id', name_column='name')

        # Should handle unicode characters
        assert isinstance(result, MatchResult)

    def test_error_handling_missing_columns(self, sample_entities_df):
        """Test error handling with missing columns"""
        matcher = FuzzyMatcher()

        # Test missing ID column
        with pytest.raises(Exception):  # Specific exception depends on implementation
            matcher.find_matches(sample_entities_df, id_column='nonexistent_id', name_column='name')

        # Test missing name column
        with pytest.raises(Exception):
            matcher.find_matches(sample_entities_df, id_column='id', name_column='nonexistent_name')

    def test_metadata_in_result(self, sample_entities_df):
        """Test that result includes metadata"""
        matcher = FuzzyMatcher(threshold=85.0, scorer='token_sort_ratio')
        result = matcher.find_matches(sample_entities_df, id_column='id', name_column='name')

        # Check if metadata is included
        assert isinstance(result.metadata, dict)
        # Metadata content depends on implementation

    @pytest.mark.performance
    def test_performance_large_dataset(self):
        """Test performance with larger dataset"""
        from tests.conftest import generate_large_dataset

        # Generate larger dataset
        large_df = generate_large_dataset(n_entities=500)

        matcher = FuzzyMatcher(threshold=85.0, n_workers=4)

        import time
        start_time = time.time()
        result = matcher.find_matches(large_df, id_column='id', name_column='name')
        end_time = time.time()

        processing_time = end_time - start_time
        print(f"Processing time for {len(large_df)} entities: {processing_time:.2f} seconds")

        # Should complete in reasonable time
        assert processing_time < 60  # Less than 1 minute for 500 entities
        assert isinstance(result, MatchResult)

    def test_reproducibility(self, sample_entities_df):
        """Test that matching results are reproducible"""
        matcher1 = FuzzyMatcher(threshold=85.0, scorer='token_sort_ratio')
        matcher2 = FuzzyMatcher(threshold=85.0, scorer='token_sort_ratio')

        result1 = matcher1.find_matches(sample_entities_df, id_column='id', name_column='name')
        result2 = matcher2.find_matches(sample_entities_df, id_column='id', name_column='name')

        # Results should be identical
        assert len(result1.pairs) == len(result2.pairs)

        if not result1.pairs.empty and not result2.pairs.empty:
            # Sort both for comparison
            sorted_result1 = result1.pairs.sort_values(['id1', 'id2']).reset_index(drop=True)
            sorted_result2 = result2.pairs.sort_values(['id1', 'id2']).reset_index(drop=True)

            pd.testing.assert_frame_equal(sorted_result1, sorted_result2)