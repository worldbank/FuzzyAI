# Migration guide: 0.1 to 0.2

Nothing is required. Every public name, signature, default and output column of 0.1 keeps working, and the saved files (`entity_groups.csv`, `duplicate_pairs.csv`, `statistics.json`, `cross_matches.csv`, `metadata.json`) are unchanged. New behaviour is opt-in. See the [changelog](https://github.com/worldbank/FuzzyAI/blob/master/CHANGELOG.md) for the four behaviour changes that do affect existing code.

## Step 0: upgrade and keep going

```bash
pip install -U deduplix
```

Existing `DeduplicationPipeline(...)` code runs as before. Checkpoints written by 0.1 do not resume once, because the data hash now includes the name column; rerun from scratch.

## Step 1: swap the matcher with `EngineMatcher`

Keep the 0.1 pipeline, results and validators; only the candidate search changes.

```python
import pandas as pd
from deduplix import DeduplicationPipeline, EngineMatcher

df = pd.DataFrame({"id": ["a", "b", "c"],
                   "name": ["Acme Fund LP", "Acme Fund L.P.", "Zeta Trust"]})

matcher = EngineMatcher({
    "normalize": ["default_text"],
    "blocking": [{"brute_force": {"threshold": 40}}],
    "compare": {"name": {"fn": "token_sort_ratio", "left": "name_norm"}},
    "score": {"feature": {"name": "name"}},
    "route": {"threshold": {"accept_above": 0.9, "reject_below": 0.5}},
})
result = DeduplicationPipeline(matcher=matcher).run(df, id_column="id", name_column="name")
print(result.duplicate_pairs)
```

`EngineMatcher` accepts a `Pipeline`, a YAML path or a dict. It reports every pair that was not rejected, so a legacy validator can still follow. `similarity_score` stays a 0 to 100 float.

## Step 2: use the full `Pipeline`

Move validation, clustering and consolidation into the engine too. Results become a `PipelineResult` with the pairs table; call `to_legacy()` if downstream code expects the 0.1 result classes.

```python
import pandas as pd
from deduplix import Pipeline

df = pd.DataFrame({"id": ["a", "b", "c"],
                   "name": ["Acme Fund LP", "Acme Fund L.P.", "Zeta Trust"]})
pipeline = Pipeline.default()
result = pipeline.dedupe(df, id_column="id", name_column="name")
legacy = result.to_legacy()
print(legacy.entity_groups)
```

Export a pipeline you built in Python with `pipeline.to_config()`, or write YAML by hand and load it with `Pipeline.from_config("pipeline.yaml")`.

## Also worth knowing

- `FuzzyMatcher.n_workers` now defaults to 4 (speed only).
- `get_group` on a singleton returns only the entity itself.
- Bad input raises `DataValidationError`, which is still a `ValueError`, so existing `except ValueError` blocks keep working.
- The legacy `LLMValidator` still treats failed batches as matches, now with a warning and counts. The engine's `llm` validator instead leaves failures `unvalidated` by default.
