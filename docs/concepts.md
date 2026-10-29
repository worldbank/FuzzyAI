# Concepts

## Stages

A `Pipeline` runs eight stages in order. Every stage is optional and swappable by name in YAML, by a registered function, or by a Python object.

| Stage | Config key | Python argument | What it does |
| --- | --- | --- | --- |
| normalize | `normalize` | `normalizers` | Adds columns such as `name_norm` or `name_base` to the records. |
| block | `blocking` | `blockers` | Proposes candidate pairs cheaply, so cost scales with candidates and not with n x m. Results of several blockers are unioned. |
| compare | `compare` | `comparators` | Adds one feature column `f_<feature>` per comparator. |
| score | `score` | `scorer` | Combines features into `score`. |
| route | `route` | `router` | Labels every pair `accept`, `review` or `reject`. |
| validate | `validate` | `validators` | Decides pairs left for review, for example with an LLM. |
| cluster | `cluster` | `clusterer` | Groups matched records into `cluster_id`. |
| consolidate | `consolidate` | `consolidator` | Picks the best match per record in linkage runs. |

Hooks (`hooks`) observe stage boundaries for logging, progress, checkpointing and MLflow.

```python
from deduplix import Pipeline

p = Pipeline.from_config({
    "normalize": ["default_text"],
    "blocking": ["tfidf_char"],
    "compare": {"name": {"fn": "token_set_ratio", "left": "name_norm"}},
    "score": {"feature": {"name": "name"}},
    "route": {"threshold": {"accept_above": 0.9, "reject_below": 0.6}},
})
```

Use `p.dedupe(df, id_column=..., name_column=...)` for one dataset and `p.link(left, right, ...)` for two.

## The pairs table

`PipelineResult.pairs` is the single source of truth for decisions; every decision is traceable to one row. Columns accumulate as the stages run:

| Column | Written by | Meaning |
| --- | --- | --- |
| `i1`, `i2` | blockers | Row positions into the left and right record tables (the same table when deduplicating). |
| `blk_<name>`, `blk_score_<name>` | blockers | Which blocker proposed the pair, and its score. |
| `f_<feature>` | comparators | One column per feature. May be numeric or categorical (for example `f_designators` is `ok`, `review` or `reject`). |
| `score` | scorer | Overall score, normally in 0 to 1. |
| `route` | router | `accept`, `review` or `reject`. |
| `decision`, `decision_prob`, `decision_reason`, `decided_by` | router and validators | `match`, `non_match` or `unvalidated`, with the stage that decided. |

Helpers on the result: `matches` (decision `match`), `review` (decision `unvalidated`), `clusters`, `explain(id1, id2)` (every column for one pair, looked up by your own ids), `save(path)` and `to_legacy()` (a 0.1 `DeduplicationResult` or `CrossDatasetResult`).

```python
import pandas as pd
from deduplix import Pipeline

df = pd.DataFrame({"id": [1, 2, 3],
                   "name": ["Acme Holdings Ltd", "ACME Holdings Limited", "Zeta Corp"]})
result = Pipeline.default().dedupe(df, id_column="id", name_column="name")
print(result.pairs.columns.tolist())
```

## Routing

A router turns scores and features into one of three routes:

- `accept`: a match without further checks.
- `reject`: a non-match without further checks.
- `review`: undecided; passed to the validators.

`threshold` routes on the score: at or above `accept_above` is accepted, below `reject_below` is rejected, anything between (and any missing score) goes to review. `rule` applies declarative rules first (first match wins) and falls back to another router:

```yaml
route:
  rule:
    rules:
      - {feature: f_designators, equals: reject, route: reject}
    then: {threshold: {accept_above: 0.95, reject_below: 0.60}}
```

A rule has `feature`, `route` and exactly one of `equals`, `in`, `gte`, `lt`, `isna`.

After validation, pairs that are still undecided follow `options.undecided` (default `unvalidated`, so they surface in `result.review`).
