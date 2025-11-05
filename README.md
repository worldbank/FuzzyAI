# deduplix

A robust, deduplication library for Python with comprehensive validation and flexible checkpointing.

## Features

### Core Capabilities

- **Fast Fuzzy Matching**: Parallel processing with RapidFuzz for high-performance similarity computation
- **LLM Validation**: AI-powered validation with support for OpenAI, Anthropic, and Databricks models
- **Rules Based Validation: ** Function based validation
- **Cross-Dataset Matching**: Match entities between different datasets with intelligent consolidation
- **Simple Clustering**: Connected components algorithm for efficient duplicate grouping

### Features

- **Data Validation**: Comprehensive input validation with Pydantic schemas
- **Security**: Rate limiting, input sanitization, and resource monitoring
- **Flexible Checkpointing**: File-based or SQLite database checkpointing with compression
- **Error Handling**: Detailed error reporting with context and suggestions
- **CLI Tool**: Full-featured command-line interface

### Architecture

- **Modular Design**: Swap components (matchers, validators, checkpointers) as needed
- **Extensible**: Easy to add custom matchers and validation rules
- **Production Ready**: Built with enterprise deployment in mind

## Installation

```bash
pip install deduplix

# Install with all dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- pandas >= 1.3.0
- rapidfuzz >= 2.0.0
- networkx >= 2.6.0
- pydantic >= 2.0.0

Optional dependencies:

- openai >= 1.0.0 (for OpenAI LLM validation)
- anthropic >= 0.30.0 (for Anthropic LLM validation)
- langchain-community (for Databricks LLM validation)

## Quick Start

### Python API

```python
from deduplix import DeduplicationPipeline, FuzzyMatcher, RuleBasedValidator
import pandas as pd

# Load your data
df = pd.read_csv('companies.csv')

# Basic deduplication with file checkpointing
pipeline = DeduplicationPipeline(
    matcher=FuzzyMatcher(threshold=85.0),
    validator=RuleBasedValidator(min_score=90.0),  # Optional
    checkpoint=True,
    checkpoint_type="file"  # or "database"
)

# Run deduplication
result = pipeline.run(df, id_column='company_id', name_column='company_name')

# Access results
print(f"Found {result.statistics['duplicate_groups']} duplicate groups")
print(f"Entities with duplicates: {result.statistics['entities_with_duplicates']}")

# Remove duplicates from original data
cleaned_df = result.remove_duplicates(df, id_column='company_id')

# Save results
result.save("output/")
```

### Advanced Usage with Database Checkpointing

```python
from deduplix import DeduplicationPipeline, FuzzyMatcher, LLMValidator

# Enterprise setup with database checkpointing and LLM validation
pipeline = DeduplicationPipeline(
    matcher=FuzzyMatcher(threshold=80.0, n_workers=8),
    validator=LLMValidator(
        provider="openai",
        model="gpt-4o-mini",
        batch_size=10
    ),
    checkpoint=True,
    checkpoint_type="database",
    checkpoint_db_path="checkpoints.db",
    checkpoint_compress=True
)

# Process large dataset with automatic resume
result = pipeline.run(large_df, resume=True)
```

### Cross-Dataset Matching

```python
# Match entities between two datasets
cross_result = pipeline.run_cross_dataset(
    df1=companies_df,
    df2=vendors_df,
    id_column1='company_id',
    name_column1='legal_name',
    id_column2='vendor_id',
    name_column2='vendor_name'
)

# Get consolidated matches
consolidated = cross_result.consolidate_with_config(
    companies_df, vendors_df, consolidation_config
)

### Command Line

```bash
# Basic deduplication
deduplix run --input data.csv --output results/ --threshold 85

# With validation and custom config
deduplix run --input data.csv --output results/ --config config.yaml

# Cross-dataset matching
deduplix cross-match --df1 companies.csv --df2 vendors.csv --output matches/

# Create sample configuration
deduplix init
```

## Configuration

### Complete Configuration Example

```yaml
matching:
  threshold: 85.0
  scorer: token_sort_ratio
  max_matches_per_entity: 100
  n_workers: 4

validation:
  enabled: true
  type: rule  # or "llm"
  rules:
    min_score: 90.0
    metadata_rules:
      - column: country
        operation: exact
      - column: industry
        operation: fuzzy
        fuzzy_threshold: 80
  llm:
    provider: openai
    model: gpt-4o-mini
    batch_size: 10
    temperature: 0.1
    max_retries: 3

pipeline:
  checkpoint: true
  checkpoint_type: database  # or "file"
  checkpoint_db_path: checkpoints.db
  checkpoint_compress: true

security:
  enable_rate_limiting: true
  requests_per_minute: 60
  enable_input_sanitization: true
```

## Security Featuress:

- **Rate Limiting**: Configurable API call limits to prevent abuse
- **Input Sanitization**: Protection against XSS and injection attacks
- **Resource Monitoring**: Memory, thread, and processing time limits
- **Data Validation**: Comprehensive input validation with Pydantic schemas

## Performance

- **Parallel Processing**: Multi-threaded fuzzy matching for large datasets
- **Checkpointing**: Resume interrupted jobs without data loss
- **Memory Efficient**: Streaming processing for datasets larger than RAM
- **Optimized**: Handles 165k entities in ~40 minutes on standard hardware

## API Reference

### Core Classes

- `DeduplicationPipeline`: Main pipeline orchestrating the deduplication process
- `FuzzyMatcher`: High-performance fuzzy string matching
- `RuleBasedValidator`: Rule-based validation with metadata support
- `LLMValidator`: AI-powered validation using language models
- `FileCheckpointer`: File-based checkpointing system
- `DatabaseCheckpointer`: SQLite-based checkpointing with compression

## Error Handling

deduplix provides comprehensive error handling with specific exception types:

```python
from deduplix.exceptions import DataValidationError, MatchingError, CheckpointError

try:
    result = pipeline.run(df)
except DataValidationError as e:
    print(f"Data validation failed: {e}")
    print(f"Suggestions: {'; '.join(e.suggestions)}")
except MatchingError as e:
    print(f"Matching failed: {e}")
except CheckpointError as e:
    print(f"Checkpoint error: {e}")
```

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

## Changelog

## License

MIT
