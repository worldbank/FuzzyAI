# deduplix

A simple, modular, and efficient entity deduplication library for Python.

## Features

- **Fast Fuzzy Matching**: Parallel processing with RapidFuzz
- **LLM Validation**: Optional AI-powered validation to remove false positives
- **Simple Clustering**: Connected components for grouping duplicates
- **Checkpointing**: Resume interrupted processing
- **Modular Design**: Swap components as needed
- **CLI Tool**: Command-line interface for easy use

## Installation

```bash
pip install deduplix

# For LLM validation support
pip install deduplix[llm]

# For Spark support
pip install deduplix[spark]
```

## Quick Start

### Python API

```python
from deduplix import DeduplicationPipeline, FuzzyMatcher, LLMValidator

# Create pipeline
pipeline = DeduplicationPipeline(
    matcher=FuzzyMatcher(threshold=85),
    validator=LLMValidator(model="gpt-4"),  # Optional
)

# Run deduplication
result = pipeline.run(df, id_column='vendor_id', name_column='company_name')

# Access results
print(f"Found {result.statistics['duplicate_groups']} duplicate groups")
result.save("output/")
```

### Command Line

```bash
# Basic usage
deduplix run -i data.csv -o output/ --threshold 85

# With LLM validation
deduplix run -i data.csv -o output/ --validate --config config.yaml

# Create sample config
deduplix init
```

## Configuration

Create a `config.yaml`:

```yaml
matching:
  threshold: 85.0
  scorer: token_sort_ratio
  max_matches_per_entity: 100
  n_workers: 4

validation:
  enabled: true
  type: llm
  llm:
    model: gpt-4
    batch_size: 10
    temperature: 0.1

pipeline:
  checkpoint: true
  checkpoint_dir: .deduplix_checkpoints
```

## License

MIT
**"""**
