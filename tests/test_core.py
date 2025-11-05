"""
Tests for core deduplication functionality.
"""

import pytest
import pandas as pd
from pathlib import Path

from deduplix.core import (
    DeduplicationPipeline, MatchResult, ValidationResult, DeduplicationResult,
    CrossDatasetResult
)
from deduplix.exceptions import DataValidationError, ConfigurationError
from deduplix.matchers import FuzzyMatcher
from deduplix.validators import RuleBasedValidator


class TestMatchResult:
    """Test MatchResult dataclass"""

    def test_match_result_creation(self, sample_match_result):
        """Test basic MatchResult creation"""
        assert isinstance(sample_match_result.pairs, pd.DataFrame)
        assert len(sample_match_result.pairs) == 2
        assert 'id1' in sample_match_result.pairs.columns
        assert 'similarity_score' in sample_match_result.pairs.columns

    def test_match_result_metadata(self):
        """Test MatchResult with metadata"""
        pairs_df = pd.DataFrame({'id1': ['A'], 'id2': ['B'], 'similarity_score': [90.0]})
        result = MatchResult(pairs=pairs_df, metadata={'test': 'value'})
        assert result.metadata['test'] == 'value'


class TestValidationResult:
    """Test ValidationResult dataclass"""

    def test_validation_result_creation(self):
        """Test ValidationResult creation"""
        validated_df = pd.DataFrame({
            'id1': ['A'], 'id2': ['B'], 'similarity_score': [95.0],
            'validation_reason': 'Passed all rules'
        })
        removed_df = pd.DataFrame({
            'id1': ['C'], 'id2': ['D'], 'similarity_score': [75.0],
            'validation_reason': 'Score too low'
        })

        result = ValidationResult(
            validated_pairs=validated_df,
            removed_pairs=removed_df
        )

        assert len(result.validated_pairs) == 1
        assert len(result.removed_pairs) == 1


class TestDeduplicationResult:
    """Test DeduplicationResult dataclass"""

    @pytest.fixture
    def sample_dedup_result(self):
        """Sample deduplication result for testing"""
        entity_groups = pd.DataFrame({
            'entity_id': ['E1', 'E2', 'E3', 'E4'],
            'entity_name': ['Apple Inc.', 'Apple Corp', 'Microsoft', 'Google'],
            'group_id': [1, 1, 0, 0]  # E1,E2 are duplicates, E3,E4 are singletons
        })

        duplicate_pairs = pd.DataFrame({
            'id1': ['E1'], 'id2': ['E2'], 'similarity_score': [95.0]
        })

        statistics = {
            'total_entities': 4,
            'duplicate_groups': 1,
            'entities_with_duplicates': 2,
            'singleton_entities': 2
        }

        return DeduplicationResult(
            entity_groups=entity_groups,
            duplicate_pairs=duplicate_pairs,
            statistics=statistics
        )

    def test_get_group(self, sample_dedup_result):
        """Test getting entities in same group"""
        group = sample_dedup_result.get_group('E1')
        assert 'E1' in group and 'E2' in group
        assert len(group) == 2

        # Test singleton
        singleton_group = sample_dedup_result.get_group('E3')
        assert singleton_group == ['E3']

    def test_get_entities_to_keep_first(self, sample_dedup_result):
        """Test keeping first entity from each group"""
        to_keep = sample_dedup_result.get_entities_to_keep(keep_strategy='first')
        # Should keep singletons + first from duplicate group
        assert 'E3' in to_keep and 'E4' in to_keep  # singletons
        assert 'E1' in to_keep or 'E2' in to_keep  # one from duplicate group
        assert len(to_keep) == 3

    def test_remove_duplicates(self, sample_dedup_result, sample_entities_df):
        """Test removing duplicates from original dataframe"""
        # Adjust sample data to match our test
        original_df = pd.DataFrame({
            'id': ['E1', 'E2', 'E3', 'E4'],
            'name': ['Apple Inc.', 'Apple Corp', 'Microsoft', 'Google']
        })

        cleaned_df = sample_dedup_result.remove_duplicates(
            original_df, id_column='id', keep_strategy='first'
        )

        # Should have 3 entities (removed 1 duplicate)
        assert len(cleaned_df) == 3
        assert 'E3' in cleaned_df['id'].values  # singleton kept
        assert 'E4' in cleaned_df['id'].values  # singleton kept

    def test_save_load(self, sample_dedup_result, temp_dir):
        """Test saving and loading results"""
        save_path = temp_dir / "test_results"

        # Save
        sample_dedup_result.save(str(save_path))

        # Verify files exist
        assert (save_path / "entity_groups.csv").exists()
        assert (save_path / "duplicate_pairs.csv").exists()
        assert (save_path / "statistics.json").exists()

        # Load
        loaded_result = DeduplicationResult.load(str(save_path))

        # Verify content
        assert len(loaded_result.entity_groups) == len(sample_dedup_result.entity_groups)
        assert len(loaded_result.duplicate_pairs) == len(sample_dedup_result.duplicate_pairs)
        assert loaded_result.statistics['total_entities'] == 4


class TestDeduplicationPipeline:
    """Test DeduplicationPipeline class"""

    def test_pipeline_creation(self, fuzzy_matcher):
        """Test basic pipeline creation"""
        pipeline = DeduplicationPipeline(
            matcher=fuzzy_matcher,
            checkpoint=False
        )
        assert pipeline.matcher is not None
        assert pipeline.validator is None
        assert pipeline.checkpointer is None

    def test_pipeline_with_validator(self, fuzzy_matcher, rule_validator):
        """Test pipeline with validator"""
        pipeline = DeduplicationPipeline(
            matcher=fuzzy_matcher,
            validator=rule_validator,
            checkpoint=False
        )
        assert pipeline.validator is not None

    def test_pipeline_with_file_checkpointing(self, fuzzy_matcher, temp_dir):
        """Test pipeline with file checkpointing"""
        pipeline = DeduplicationPipeline(
            matcher=fuzzy_matcher,
            checkpoint=True,
            checkpoint_type="file",
            checkpoint_dir=str(temp_dir)
        )
        assert pipeline.checkpointer is not None
        assert pipeline.checkpoint_type == "file"

    def test_pipeline_with_database_checkpointing(self, fuzzy_matcher, temp_db_path):
        """Test pipeline with database checkpointing"""
        pipeline = DeduplicationPipeline(
            matcher=fuzzy_matcher,
            checkpoint=True,
            checkpoint_type="database",
            checkpoint_db_path=temp_db_path
        )
        assert pipeline.checkpointer is not None
        assert pipeline.checkpoint_type == "database"

    def test_pipeline_invalid_checkpoint_type(self, fuzzy_matcher):
        """Test pipeline with invalid checkpoint type"""
        with pytest.raises(ConfigurationError) as exc_info:
            DeduplicationPipeline(
                matcher=fuzzy_matcher,
                checkpoint=True,
                checkpoint_type="invalid"
            )
        assert "Unsupported checkpoint type" in str(exc_info.value)

    def test_compute_data_hash(self, dedup_pipeline, sample_entities_df):
        """Test data hash computation"""
        hash1 = dedup_pipeline._compute_data_hash(sample_entities_df)
        hash2 = dedup_pipeline._compute_data_hash(sample_entities_df)

        # Same data should produce same hash
        assert hash1 == hash2

        # Different data should produce different hash
        modified_df = sample_entities_df.copy()
        modified_df.loc[0, 'name'] = 'Different Name'
        hash3 = dedup_pipeline._compute_data_hash(modified_df)
        assert hash1 != hash3

    def test_remove_symmetric_pairs(self, dedup_pipeline):
        """Test removal of symmetric duplicate pairs"""
        pairs_df = pd.DataFrame({
            'id1': ['A', 'B', 'A'],
            'id2': ['B', 'A', 'C'],
            'name1': ['Apple', 'Beta', 'Apple'],
            'name2': ['Beta', 'Apple', 'Charlie'],
            'similarity_score': [95.0, 95.0, 88.0]
        })

        cleaned_pairs = dedup_pipeline._remove_symmetric_pairs(pairs_df)

        # Should remove (B,A) as it's symmetric to (A,B)
        assert len(cleaned_pairs) == 2

        # Verify we kept one version of the A-B pair
        ab_pairs = cleaned_pairs[
            ((cleaned_pairs['id1'] == 'A') & (cleaned_pairs['id2'] == 'B')) |
            ((cleaned_pairs['id1'] == 'B') & (cleaned_pairs['id2'] == 'A'))
        ]
        assert len(ab_pairs) == 1

    def test_run_basic_pipeline(self, sample_entities_df):
        """Test basic pipeline execution"""
        matcher = FuzzyMatcher(threshold=85.0)
        validator = RuleBasedValidator(min_score=90.0)

        pipeline = DeduplicationPipeline(
            matcher=matcher,
            validator=validator,
            checkpoint=False
        )

        result = pipeline.run(sample_entities_df, id_column='id', name_column='name')

        # Verify result structure
        assert isinstance(result, DeduplicationResult)
        assert 'total_entities' in result.statistics
        assert len(result.entity_groups) == len(sample_entities_df)

    def test_run_with_validation_errors(self, sample_entities_df):
        """Test pipeline with data validation errors"""
        matcher = FuzzyMatcher(threshold=85.0)
        pipeline = DeduplicationPipeline(matcher=matcher, checkpoint=False)

        # Test with missing ID column
        invalid_df = sample_entities_df.drop(columns=['id'])

        with pytest.raises(DataValidationError):
            pipeline.run(invalid_df, id_column='id', name_column='name')

    def test_run_cross_dataset(self, sample_cross_df1, sample_cross_df2):
        """Test cross-dataset matching"""
        matcher = FuzzyMatcher(threshold=80.0)
        pipeline = DeduplicationPipeline(matcher=matcher, checkpoint=False)

        result = pipeline.run_cross_dataset(
            df1=sample_cross_df1,
            df2=sample_cross_df2,
            id_column1='company_id',
            name_column1='company_name',
            id_column2='entity_id',
            name_column2='entity_name'
        )

        # Verify result structure
        assert isinstance(result, CrossDatasetResult)
        assert 'cross_matches' in result.statistics
        assert result.df1_metadata['id_col'] == 'company_id'
        assert result.df2_metadata['id_col'] == 'entity_id'


class TestCrossDatasetResult:
    """Test CrossDatasetResult functionality"""

    @pytest.fixture
    def sample_cross_result(self, sample_cross_df1, sample_cross_df2):
        """Sample cross-dataset result"""
        cross_matches = pd.DataFrame({
            'df1_id': ['C1', 'C2'],
            'df1_name': ['Apple Inc.', 'Microsoft Corp'],
            'df2_id': ['E1', 'E2'],
            'df2_name': ['Apple Incorporated', 'Microsoft Corporation'],
            'similarity_score': [95.0, 92.0]
        })

        return CrossDatasetResult(
            cross_matches=cross_matches,
            df1_metadata={'total': 3, 'id_col': 'company_id', 'name_col': 'company_name'},
            df2_metadata={'total': 4, 'id_col': 'entity_id', 'name_col': 'entity_name'},
            statistics={'cross_matches': 2, 'df1_matched_entities': 2}
        )

    def test_get_df1_matches(self, sample_cross_result):
        """Test getting matches for df1 entity"""
        matches = sample_cross_result.get_df1_matches('C1')
        assert len(matches) == 1
        assert matches[0]['df2_id'] == 'E1'

    def test_get_df2_matches(self, sample_cross_result):
        """Test getting matches for df2 entity"""
        matches = sample_cross_result.get_df2_matches('E2')
        assert len(matches) == 1
        assert matches[0]['df1_id'] == 'C2'

    def test_merge_datasets(self, sample_cross_result, sample_cross_df1, sample_cross_df2):
        """Test merging datasets based on matches"""
        merged = sample_cross_result.merge_datasets(
            df1=sample_cross_df1,
            df2=sample_cross_df2,
            how='inner'
        )

        # Should have 2 merged rows (matching pairs)
        assert len(merged) == 2
        assert 'similarity_score' in merged.columns

        # Verify merge correctness
        apple_row = merged[merged['company_id'] == 'C1'].iloc[0]
        assert apple_row['entity_id'] == 'E1'

    def test_save_load_cross_result(self, sample_cross_result, temp_dir):
        """Test saving and loading cross-dataset results"""
        save_path = temp_dir / "cross_results"

        # Save
        sample_cross_result.save(str(save_path))

        # Verify files
        assert (save_path / "cross_matches.csv").exists()
        assert (save_path / "metadata.json").exists()

        # Load
        loaded = CrossDatasetResult.load(str(save_path))

        # Verify
        assert len(loaded.cross_matches) == len(sample_cross_result.cross_matches)
        assert loaded.df1_metadata['id_col'] == 'company_id'