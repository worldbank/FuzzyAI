"""
Pytest configuration and shared fixtures for deduplix tests.
"""

import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from typing import Generator

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deduplix.core import DeduplicationPipeline, MatchResult, ValidationResult, DeduplicationResult
from deduplix.matchers import FuzzyMatcher
from deduplix.validators import RuleBasedValidator, LLMValidator
from deduplix.checkpointing import FileCheckpointer, DatabaseCheckpointer


@pytest.fixture
def sample_entities_df():
    """Create sample entity dataframe for testing"""
    return pd.DataFrame({
        'id': ['E1', 'E2', 'E3', 'E4', 'E5', 'E6'],
        'name': [
            'Apple Inc.',
            'Apple Incorporated',
            'Microsoft Corporation',
            'Microsoft Corp',
            'Google LLC',
            'Alphabet Inc.'
        ],
        'country': ['USA', 'USA', 'USA', 'USA', 'USA', 'USA'],
        'industry': ['Technology', 'Technology', 'Technology', 'Technology', 'Technology', 'Technology']
    })


@pytest.fixture
def sample_cross_df1():
    """First dataset for cross-dataset testing"""
    return pd.DataFrame({
        'company_id': ['C1', 'C2', 'C3'],
        'company_name': ['Apple Inc.', 'Microsoft Corp', 'Google LLC'],
        'sector': ['Tech', 'Tech', 'Tech']
    })


@pytest.fixture
def sample_cross_df2():
    """Second dataset for cross-dataset testing"""
    return pd.DataFrame({
        'entity_id': ['E1', 'E2', 'E3', 'E4'],
        'entity_name': ['Apple Incorporated', 'Microsoft Corporation', 'Alphabet Inc.', 'Tesla Inc.'],
        'type': ['Public', 'Public', 'Public', 'Public']
    })


@pytest.fixture
def sample_match_result():
    """Sample match result for testing"""
    pairs_df = pd.DataFrame({
        'id1': ['E1', 'E3'],
        'id2': ['E2', 'E4'],
        'name1': ['Apple Inc.', 'Microsoft Corporation'],
        'name2': ['Apple Incorporated', 'Microsoft Corp'],
        'similarity_score': [95.0, 92.0]
    })
    return MatchResult(pairs=pairs_df)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for testing"""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_db_path(temp_dir) -> str:
    """Temporary database path for testing"""
    return str(temp_dir / "test_checkpoints.db")


@pytest.fixture
def fuzzy_matcher():
    """Basic fuzzy matcher for testing"""
    return FuzzyMatcher(threshold=80.0)


@pytest.fixture
def rule_validator():
    """Basic rule-based validator for testing"""
    return RuleBasedValidator(min_score=85.0)


@pytest.fixture
def dedup_pipeline(fuzzy_matcher, rule_validator):
    """Basic deduplication pipeline for testing"""
    return DeduplicationPipeline(
        matcher=fuzzy_matcher,
        validator=rule_validator,
        checkpoint=False  # Disable checkpointing for basic tests
    )


@pytest.fixture
def file_checkpointer(temp_dir):
    """File-based checkpointer for testing"""
    return FileCheckpointer(checkpoint_dir=str(temp_dir))


@pytest.fixture
def db_checkpointer(temp_db_path):
    """Database checkpointer for testing"""
    return DatabaseCheckpointer(db_path=temp_db_path)


# Test data generators
def generate_large_dataset(n_entities: int = 1000) -> pd.DataFrame:
    """Generate large dataset for performance testing"""
    import random
    import string

    companies = []
    for i in range(n_entities):
        # Generate some duplicates intentionally
        if i % 10 == 0 and i > 0:
            # Create a duplicate of a previous company with slight variation
            base_idx = i - random.randint(1, min(9, i))
            base_name = f"Company_{base_idx}"
            name = f"{base_name} Inc." if random.random() > 0.5 else f"{base_name} Corp"
        else:
            name = f"Company_{i}"

        companies.append({
            'id': f'E{i}',
            'name': name,
            'country': random.choice(['USA', 'UK', 'Canada', 'Germany']),
            'industry': random.choice(['Technology', 'Finance', 'Healthcare', 'Manufacturing'])
        })

    return pd.DataFrame(companies)


# Mock API responses for LLM testing
MOCK_LLM_RESPONSES = {
    "duplicate_response": '''{
        "decisions": [
            {
                "pair_index": 0,
                "is_duplicate": true,
                "reason": "Same company with slight name variation"
            }
        ]
    }''',
    "not_duplicate_response": '''{
        "decisions": [
            {
                "pair_index": 0,
                "is_duplicate": false,
                "reason": "Different companies in same industry"
            }
        ]
    }'''
}


class MockLLMClient:
    """Mock LLM client for testing without API calls"""

    def __init__(self, response_type="duplicate"):
        self.response_type = response_type

    def chat_completions_create(self, **kwargs):
        """Mock OpenAI chat completion"""
        class MockChoice:
            def __init__(self, content):
                self.message = type('obj', (object,), {'content': content})

        class MockResponse:
            def __init__(self, content):
                self.choices = [MockChoice(content)]

        if self.response_type == "duplicate":
            return MockResponse(MOCK_LLM_RESPONSES["duplicate_response"])
        else:
            return MockResponse(MOCK_LLM_RESPONSES["not_duplicate_response"])


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    return MockLLMClient()