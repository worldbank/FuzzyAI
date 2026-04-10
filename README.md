# Fuzzy AI

A comprehensive, production-ready entity deduplication and matching library for Python with LLM-powered validation and intelligent consolidation.

## Overview

The library provides a complete toolkit for:
- **Entity Deduplication**: Find and merge duplicate entities within a single dataset
- **Cross-Dataset Matching**: Match entities across two different datasets
- **LLM Validation**: Use AI to eliminate false positives with high accuracy
- **Smart Consolidation**: Rule-based or LLM-powered selection of best matches
- **Enterprise Scale**: Process large numbers of entities with checkpointing and resume capability

## Features

### Core Capabilities
- **Fast Fuzzy Matching**: Parallel processing with RapidFuzz for high-performance similarity computation
-  **Multi-Provider LLM Validation**: OpenAI, Anthropic Claude, Databricks (Llama models)
-  **Rules-Based Validation**: Function-based validation with metadata support
-  **Cross-Dataset Matching**: Match entities between different datasets with intelligent consolidation
-  **Intelligent Consolidation**: Rule-based or LLM-powered selection when multiple matches exist
-  **Flexible Checkpointing**: File-based or SQLite database checkpointing with compression
-  **Highly Configurable**: Extensive options for matching, validation, and consolidation
-  **Production Ready**: Handles large datasets (160K+ entities) with batch processing

### Advanced Features
- **Multi-column matching**: Try multiple name fields and keep best match
- **Metadata-aware validation**: Use additional columns (country, industry, etc.) in validation
- **Priority-based consolidation**: Select matches based on source system, data quality, etc.
- **LLM consolidation**: Let AI choose the best match considering all metadata
- **Custom rules**: Add your own validation and selection logic
- **Text preprocessing**: Configurable normalization (lowercase, punctuation removal, etc.)
- **Progress tracking**: Detailed progress bars and checkpoint saving
- **Security Features**: Rate limiting, input sanitization, and resource monitoring
- **Error Handling**: Detailed error reporting with context and suggestions

### Architecture
- **Modular Design**: Swap components (matchers, validators, checkpointers) as needed
- **Extensible**: Easy to add custom matchers and validation rules
- **Production Ready**: Built with enterprise deployment in mind

## Installation

```bash
# Basic installation
pip install fuzzy-ai

# With LLM validation support
pip install fuzzy-ai[llm]

# With all dependencies
pip install fuzzy-ai[all]

# Install with all dependencies from requirements
pip install -r requirements.txt
```

**Manual installation from source:**
```bash
git clone https://github.com/worldbank/fuzzy-ai.git
cd fuzzy-ai
pip install -e .
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

### 1. Basic Deduplication

Find duplicates within a single dataset:

```python
import pandas as pd
from deduplix import DeduplicationPipeline, FuzzyMatcher

# Load your data
df = pd.read_csv("companies.csv")

# Create pipeline
pipeline = DeduplicationPipeline(
    matcher=FuzzyMatcher(
        threshold=85.0,
        scorer='token_set_ratio'  # Best for company names
    )
)

# Run deduplication
result = pipeline.run(
    df,
    id_column='company_id',
    name_column='company_name'
)

# View results
print(f"Found {result.statistics['duplicate_groups']} duplicate groups")
print(f"Entities with duplicates: {result.statistics['entities_with_duplicates']}")

# Save results
result.save("output/")
```

### 2. Advanced Setup with Database Checkpointing

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
    checkpoint_type="database",  # or "file"
    checkpoint_db_path="checkpoints.db",
    checkpoint_compress=True
)

# Process large dataset with automatic resume
result = pipeline.run(large_df, resume=True)

# Remove duplicates from original data
cleaned_df = result.remove_duplicates(df, id_column='company_id', keep_strategy='first')
```

### 3. Cross-Dataset Matching

Match entities between two datasets:

```python
from deduplix import DeduplicationPipeline, FuzzyMatcher

# Load datasets
df1 = pd.read_csv("internal_companies.csv")
df2 = pd.read_csv("external_vendors.csv")

# Create pipeline
pipeline = DeduplicationPipeline(
    matcher=FuzzyMatcher(threshold=85.0, scorer='token_set_ratio')
)

# Find cross-dataset matches
result = pipeline.run_cross_dataset(
    df1, df2,
    id_column1='company_id',
    name_column1='company_name',
    id_column2='vendor_id',
    name_column2='vendor_name'
)

# View matches
print(f"Found {len(result.cross_matches)} cross-dataset matches")
print(f"DF1 matched: {result.statistics['df1_matched_entities']}/{result.statistics['df1_total']}")
print(f"DF2 matched: {result.statistics['df2_matched_entities']}/{result.statistics['df2_total']}")
```

### 4. Multi-Column Matching

Try multiple name fields and keep the best match:

```python
# Match using multiple column combinations
result = pipeline.run_cross_dataset(
    df1, df2,
    id_column1='company_id',
    name_columns1=['legal_name', 'short_name', 'dba_name'],  # Multiple columns
    id_column2='vendor_id',
    name_columns2=['vendor_name', 'trading_name']  # Multiple columns
)

# Results include which columns produced each match
print(result.cross_matches[['df1_name', 'df2_name', 'matched_column1', 'matched_column2', 'similarity_score']])
```

### 5. LLM Validation

Add AI-powered validation to eliminate false positives:

```python
from deduplix import DeduplicationPipeline, FuzzyMatcher, LLMValidator

# Create pipeline with LLM validation
pipeline = DeduplicationPipeline(
    matcher=FuzzyMatcher(threshold=75.0),  # Lower threshold, LLM will filter
    validator=LLMValidator(
        provider='openai',
        model='gpt-4o-mini',
        batch_size=10,
        n_workers=4
    )
)

# Run with validation
result = pipeline.run(df, id_column='id', name_column='name')

# LLM filters out false positives
print(f"Validated: {result.statistics['validated_pairs']} pairs")
```

**Supported LLM Providers:**
- **OpenAI**: `gpt-4o-mini`, `gpt-4o`, `gpt-4`, `gpt-3.5-turbo`
- **Anthropic**: `claude-sonnet-4`, `claude-opus-4`, `claude-3-5-sonnet-20241022`
- **Databricks**: Various LLM models (via serving endpoints)

### 6. Consolidation with Rules

When an entity has multiple matches, select the best one using rules:

```python
from deduplix.consolidation import ConsolidationConfig

# Priority-based consolidation
config = ConsolidationConfig(
    enabled=True,
    priority_column='source_system',
    priority_order=['SOURCE_A', 'SOURCE_B', 'SOURCE_C', 'SOURCE_D'],  # Highest to lowest
    priority_mode='strict',  # Always prefer higher priority
    metadata_columns={
        'df2': ['source_system', 'data_quality', 'certification_level']
    }
)

# Apply consolidation
consolidated = result.consolidate_with_config(df1, df2, config)

# Result: One row per df1 entity with main match + other candidates
print(consolidated[['company_name', 'main_vendor_name', 'main_source_system', 'other_candidates', 'total_matches']])
```

**Consolidation Modes:**
- **Strict**: Always prefer higher priority source, regardless of similarity score
- **Threshold**: Use priority only if scores are within threshold (e.g., 10 points)

### 7. LLM Consolidation

Let AI choose the best match based on all available metadata:

```python
from deduplix.llm_consolidation import LLMConsolidationConfig

# LLM-based consolidation
config = LLMConsolidationConfig(
    enabled=True,
    model='gpt-4o-mini',
    provider='openai',
    batch_size=5,
    n_workers=4,
    metadata_columns={
        'df1': ['entity_type', 'country'],
        'df2': ['source_system', 'data_quality', 'last_updated', 'certification_level']
    },
    instructions="""
    Select the best match based on:
    1. Prefer 'verified' or 'tier1' certification over unverified
    2. Prefer more recent last_updated dates
    3. Prefer SOURCE_A when quality is similar
    4. Consider similarity score as a secondary factor
    """,
    checkpoint_every_n_batches=5  # Save progress every 5 batches
)

# Apply LLM consolidation
consolidated = result.consolidate_with_llm(
    df1, df2, 
    config,
    checkpointer=pipeline.checkpointer,
    resume=True
)

# Result includes LLM reasoning
print(consolidated[['company_name', 'main_vendor_name', 'llm_reasoning', 'other_candidates']])
```

## Configuration

### Complete Configuration Example

```yaml
# config.yaml
matching:
  threshold: 85.0
  scorer: token_set_ratio  # token_set_ratio, token_sort_ratio, ratio, partial_ratio
  max_matches_per_entity: 100
  n_workers: 4
  preprocessing:
    lowercase: true
    strip_whitespace: true
    remove_punctuation: false

validation:
  enabled: true
  type: llm  # 'llm' or 'rules'
  
  # LLM validation settings
  llm:
    provider: openai  # 'openai', 'anthropic', 'databricks'
    model: gpt-4o-mini
    batch_size: 10
    n_workers: 4
    temperature: 0.0
    max_retries: 3
    checkpoint_every_n_batches: 5
    
    # Metadata-aware validation
    metadata_columns:
      - country
      - industry
      - entity_type
    
    custom_rules:
      custom_instructions: |
        Additional validation criteria:
        - Reject matches where country differs
        - Be conservative with cross-industry matches
  
  # Rule-based validation settings
  rules:
    min_score: 90.0
    metadata_rules:
      - column: country
        operation: exact
      - column: industry
        operation: fuzzy
        fuzzy_threshold: 80
      - column: revenue
        operation: inequality
        max_diff_percent: 50

consolidation:
  enabled: true
  priority_column: source_system
  priority_order:
    - SOURCE_A
    - SOURCE_B
    - SOURCE_C
    - SOURCE_D
  priority_mode: strict  # 'strict' or 'threshold'
  priority_threshold: 10.0
  metadata_columns:
    df2:
      - source_system
      - data_quality
      - certification_level
  keep_all_matches: true
  other_candidates_column: other_candidates

llm_consolidation:
  enabled: true
  provider: openai
  model: gpt-4o-mini
  batch_size: 5
  n_workers: 4
  checkpoint_every_n_batches: 5
  metadata_columns:
    df1:
      - entity_type
      - country
    df2:
      - source_system
      - data_quality
      - certification_level
  instructions: |
    Select the best match based on:
    1. Prefer 'verified' certification over unverified
    2. Prefer more recent data
    3. Prefer SOURCE_A when quality is similar

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

## Command Line Interface

### Basic Commands

```bash
# Basic deduplication
fuzzy-ai run -i companies.csv -o output/ --threshold 85

# With configuration file
fuzzy-ai run -i companies.csv -o output/ --config config.yaml

# With LLM validation
fuzzy-ai run -i companies.csv -o output/ --validate --validator llm

# Resume from checkpoint
fuzzy-ai run -i companies.csv -o output/ --resume

# Create sample config
fuzzy-ai init --output fuzzy_ai_config.yaml

# Analyze results
fuzzy-ai analyze output/ --format json

# Show duplicate group
fuzzy-ai show-group output/ 12345

# Remove duplicates from original data
fuzzy-ai remove -i companies.csv -r output/ -o cleaned.csv --strategy first

# Cross-dataset matching
fuzzy-ai cross-match --df1 companies.csv --df2 vendors.csv --output matches/
```

## Advanced Usage

### Custom Validation Rules

```python
from deduplix import LLMValidator

# Custom validation with metadata
validator = LLMValidator(
    provider='openai',
    model='gpt-4o-mini',
    metadata_columns=['country', 'industry', 'revenue'],
    custom_rules={
        'custom_instructions': """
        Additional criteria:
        - Reject if countries differ
        - Be strict with revenue differences >50%
        - Consider industry alignment
        """
    }
)
```

### Custom Consolidation Logic

```python
from deduplix.consolidation import ConsolidationConfig

# Custom selection function
def select_best_match(matches_df):
    """Custom logic to select best match"""
    # Prefer verified sources with high scores
    verified = matches_df[matches_df['data_quality'] == 'verified']
    if not verified.empty and verified['similarity_score'].max() > 90:
        return verified.loc[verified['similarity_score'].idxmax()]
    # Otherwise, highest score
    return matches_df.loc[matches_df['similarity_score'].idxmax()]

config = ConsolidationConfig(
    enabled=True,
    selection_function=select_best_match,
    metadata_columns={'df2': ['data_quality']}
)
```

### Databricks Configuration

```python
from deduplix import LLMValidator

# Using Databricks serving endpoints
validator = LLMValidator(
    provider='databricks',
    databricks_host='https://your-workspace.cloud.databricks.com',
    databricks_endpoint='llama-3-3-70b-endpoint',
    batch_size=8,  # Smaller batches for models with small context window
    n_workers=3,
    temperature=0.0
)

# Environment variables
# DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
# DATABRICKS_TOKEN=your-token-here
```

### Processing Very Large Datasets

```python
from deduplix import DeduplicationPipeline, FuzzyMatcher, LLMValidator

# Optimized for 160K+ entities
pipeline = DeduplicationPipeline(
    matcher=FuzzyMatcher(
        threshold=85.0,
        scorer='token_set_ratio',
        n_workers=8,  # More workers for large datasets
        max_matches_per_entity=100
    ),
    validator=LLMValidator(
        provider='databricks',
        model='llama-3-3-70b',
        batch_size=8,
        n_workers=3,
        checkpoint_every_n_batches=5  # Checkpoint frequently
    ),
    checkpoint=True,
    checkpoint_dir='.fuzzy_ai_checkpoints'
)

# Run with resume enabled
result = pipeline.run(
    large_df,
    id_column='id',
    name_column='name',
    resume=True  # Resume from last checkpoint if interrupted
)
```

### Export Results

```python
from deduplix.utils import save_to_excel_safe, save_multiple_sheets_safe

# Export single sheet
save_to_excel_safe(result.entity_groups, 'entity_groups.xlsx')

# Export multiple sheets
sheets = {
    'Entity_Groups': result.entity_groups,
    'Duplicate_Pairs': result.duplicate_pairs,
    'Statistics': pd.DataFrame([result.statistics])
}
save_multiple_sheets_safe(sheets, 'deduplication_results.xlsx')

# For cross-dataset results
sheets = {
    'Consolidated': consolidated_df,
    'All_Matches': result.cross_matches,
    'Summary': pd.DataFrame([result.statistics])
}
save_multiple_sheets_safe(sheets, 'cross_matching_results.xlsx')
```

## API Reference

### Core Classes

#### `DeduplicationPipeline`
Main orchestrator for deduplication workflows.

```python
pipeline = DeduplicationPipeline(
    matcher: Matcher,                    # Matching strategy
    validator: Optional[Validator],      # Optional validation
    checkpoint: bool = True,             # Enable checkpointing
    checkpoint_type: str = "file",       # "file" or "database"
    checkpoint_dir: str = '.fuzzy_ai_checkpoints',
    checkpoint_db_path: str = None,      # For database checkpointing
    checkpoint_compress: bool = False    # Compress checkpoints
)
```

**Methods:**
- `run(df, id_column, name_column)`: Deduplicate within dataset
- `run_cross_dataset(df1, df2, ...)`: Match across datasets

#### `FuzzyMatcher`
Fuzzy string matching with RapidFuzz.

```python
matcher = FuzzyMatcher(
    threshold: float = 80.0,             # Similarity threshold (0-100)
    scorer: str = 'ratio',               # Scoring algorithm
    max_matches_per_entity: int = 100,   # Limit matches per entity (optional)
    n_workers: int = 4,                  # Parallel workers
    lowercase: bool = True,              # Normalize case
    strip_whitespace: bool = True,       # Normalize whitespace
    remove_punctuation: bool = False     # Remove punctuation
)
```

**Scorers:**
- `token_set_ratio`: Best for company names (handles word order)
- `token_sort_ratio`: Good for names with different word order
- `ratio`: Exact character-level similarity
- `partial_ratio`: Substring matching
- `WRatio`: Weighted ratio (automatic selection)

#### `LLMValidator`
AI-powered validation with multiple providers.

```python
validator = LLMValidator(
    provider: str = 'openai',            # 'openai', 'anthropic', 'databricks'
    model: str = 'gpt-4o-mini',          # Model name
    batch_size: int = 10,                # Pairs per LLM call
    n_workers: int = 4,                  # Parallel workers
    temperature: float = 0.0,            # Sampling temperature
    max_retries: int = 3,                # Retry attempts
    metadata_columns: List[str] = None,  # Columns for validation context
    checkpoint_every_n_batches: int = 0  # Checkpoint frequency
)
```

#### `RuleBasedValidator`
Rule-based validation with metadata support.

```python
validator = RuleBasedValidator(
    min_score: float = 90.0,
    metadata_rules: List[Dict] = None,   # Metadata-based rules
    custom_rules: List[Callable] = None  # Custom validation functions
)
```

#### `ConsolidationConfig`
Rule-based consolidation configuration.

```python
config = ConsolidationConfig(
    enabled: bool = True,
    priority_column: str = 'source_system',
    priority_order: List[str] = [...],   # Highest to lowest priority
    priority_mode: str = 'strict',       # 'strict' or 'threshold'
    priority_threshold: float = 10.0,    # For threshold mode
    metadata_columns: Dict = None,       # Additional metadata
    filter_function: Callable = None,    # Custom filter
    selection_function: Callable = None  # Custom selection
)
```

#### `LLMConsolidationConfig`
LLM-based consolidation configuration.

```python
config = LLMConsolidationConfig(
    enabled: bool = True,
    provider: str = 'openai',
    model: str = 'gpt-4o-mini',
    batch_size: int = 5,
    n_workers: int = 4,
    metadata_columns: Dict = None,       # Metadata for LLM context
    instructions: str = None,            # Custom instructions
    selection_criteria: List[str] = None,
    checkpoint_every_n_batches: int = 5
)
```

### Result Objects

#### `DeduplicationResult`
Result from within-dataset deduplication.

**Attributes:**
- `entity_groups`: DataFrame with entity_id, entity_name, group_id
- `duplicate_pairs`: DataFrame with validated duplicate pairs
- `statistics`: Dictionary with stats

**Methods:**
- `get_group(entity_id)`: Get all entities in same group
- `save(path)`: Save results to directory
- `load(path)`: Load results from directory
- `remove_duplicates(df, keep_strategy)`: Remove duplicates from original data

#### `CrossDatasetResult`
Result from cross-dataset matching.

**Attributes:**
- `cross_matches`: DataFrame with matches
- `statistics`: Dictionary with stats

**Methods:**
- `get_df1_matches(df1_id)`: Get all df2 matches for df1 entity
- `get_df2_matches(df2_id)`: Get all df1 matches for df2 entity
- `consolidate_with_config(df1, df2, config)`: Apply rule-based consolidation
- `consolidate_with_llm(df1, df2, config)`: Apply LLM consolidation
- `merge_datasets(df1, df2)`: Merge datasets on matches

## Security Features

Fuzzy AI includes comprehensive security measures:

- **Rate Limiting**: Configurable API call limits to prevent abuse
- **Input Sanitization**: Protection against XSS and injection attacks
- **Resource Monitoring**: Memory, thread, and processing time limits
- **Data Validation**: Comprehensive input validation with Pydantic schemas

## Performance Tips

1. **Matching Threshold**: Start with 80-85 for initial matching, use LLM validation to filter
2. **Scorer Selection**: Use `token_set_ratio` for company names (best accuracy)
3. **Batch Sizes**: 
   - GPT-4: 10-15 pairs per batch
   - Llama: 8-12 pairs per batch (less reliable JSON parsing)
4. **Workers**: 
   - Matching: 4-8 workers
   - LLM validation: 3-4 workers (avoid rate limits)
5. **Checkpointing**: Enable for jobs >10 minutes, checkpoint every 5-10 batches
6. **Large Datasets**: Use `max_matches_per_entity=1000` to limit memory usage when there is high similarity
7. **Optimization**: Handles 165k entities in ~40 minutes on standard hardware

## Typical Workflows

### Workflow 1: Internal Deduplication with LLM Validation

```python
# 1. Find fuzzy matches
pipeline = DeduplicationPipeline(
    matcher=FuzzyMatcher(threshold=80.0),
    validator=LLMValidator(model='gpt-4o-mini'),
    checkpoint=True,
    checkpoint_type="database"
)

# 2. Run deduplication
result = pipeline.run(df, id_column='id', name_column='name')

# 3. Remove duplicates
cleaned_df = result.remove_duplicates(df, keep_strategy='first')

# 4. Export
result.save('output/')
```

### Workflow 2: Cross-Dataset Matching with Consolidation

```python
# 1. Match across datasets
result = pipeline.run_cross_dataset(
    df1, df2,
    name_columns1=['legal_name', 'short_name'],
    name_columns2=['vendor_name', 'dba']
)

# 2. Consolidate with rules
config = ConsolidationConfig(
    priority_column='source_system',
    priority_order=['SOURCE_A', 'SOURCE_B']
)
consolidated = result.consolidate_with_config(df1, df2, config)

# 3. Export
save_multiple_sheets_safe({
    'Consolidated': consolidated,
    'All_Matches': result.cross_matches
}, 'output.xlsx')
```

### Workflow 3: Large-Scale Processing with Checkpoints

```python
# 1. Configure for large dataset
pipeline = DeduplicationPipeline(
    matcher=FuzzyMatcher(threshold=85.0, n_workers=8),
    validator=LLMValidator(
        provider='databricks',
        model='llama-3-3-70b',
        batch_size=8,
        checkpoint_every_n_batches=5
    ),
    checkpoint=True,
    checkpoint_type="database",
    checkpoint_compress=True
)

# 2. Run with resume
result = pipeline.run_cross_dataset(
    df1, df2,
    resume=True  # Resume from last checkpoint if interrupted
)

# 3. LLM consolidation
config = LLMConsolidationConfig(
    model='gpt-4o-mini',
    batch_size=5,
    checkpoint_every_n_batches=5
)
consolidated = result.consolidate_with_llm(df1, df2, config, resume=True)
```

## Error Handling

Fuzzy AI provides comprehensive error handling with specific exception types:

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

## Troubleshooting

### Common Issues

**1. Memory errors with large datasets**
- Reduce `max_matches_per_entity`
- Increase `batch_size` in matching
- Use checkpointing and resume in batches
- Enable database checkpointing with compression

**2. LLM rate limits**
- Reduce `n_workers`
- Increase `batch_size`
- Add retries with `max_retries`
- Enable rate limiting in security config

**3. False positives in matching**
- Increase `threshold`
- Add LLM validation
- Use metadata-aware validation

**4. Checkpoint files corrupt**
- Delete checkpoint directory and restart
- Use `resume=False` to start fresh
- Try database checkpointing instead of file-based

**5. JSON parsing errors**
- Reduce `batch_size` (some models less reliable with large outputs)
- Reduce `temperature` (closer to 0)
- Add more explicit JSON formatting instructions

## Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY=your-key-here

# Anthropic
export ANTHROPIC_API_KEY=your-key-here

# Databricks
export DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
export DATABRICKS_TOKEN=your-token-here
# or
export DATABRICKS_API_KEY=your-token-here
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

See our contributing guidelines for details.

## Contact

📧 **Portfolio Intelligence Team** — [portfoliointelligence@worldbankgroup.org](mailto:portfoliointelligence@worldbankgroup.org)

## License

This project is licensed under the MIT License together with the World Bank IGO Rider. The Rider is purely procedural: it reserves all privileges and immunities enjoyed by the World Bank, without adding restrictions to the MIT permissions. Please review both files before using, distributing or contributing.

## Citation

If you use Fuzzy AI in your research, please cite:

```bibtex
@software{fuzzy_ai,
  title = {Fuzzy AI: Entity Deduplication and Matching with LLM Validation},
  author = {World Bank},
  year = {2024},
  url = {https://github.com/worldbank/fuzzy-ai}
}
```

## Support

- **Issues**: https://github.com/worldbank/fuzzy-ai/issues
- **Discussions**: https://github.com/worldbank/fuzzy-ai/discussions
- **Documentation**: https://github.com/worldbank/fuzzy-ai

## Changelog

### Version 0.1.0 (Current)
- Initial release
- Fuzzy matching with RapidFuzz
- LLM validation (OpenAI, Anthropic, Databricks)
- Rule-based validation with metadata support
- Cross-dataset matching with multi-column support
- Rule-based and LLM consolidation
- File-based and database checkpointing
- Security features (rate limiting, sanitization)
- Comprehensive error handling
- CLI tool
- Production-ready performance optimizations

---
