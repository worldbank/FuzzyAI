# Cookbook

## Fund vehicles

Funds differ by number ("Fund I" vs "Fund II"), feeder or master role, and vintage, so a high name similarity is not enough. `name_parser` splits names into a base name, legal form and designators, and `designator_check` turns them into `ok`, `review` or `reject`. Save this as `funds.yaml`:

```yaml
normalize:
  - default_text
  - name_parser: {country_column: country}
blocking:
  - exact: {keys: [lei, registration_no]}
  - token_index: {max_df: 1000, top_k: 50}
compare:
  name: {fn: token_set_ratio, left: name_base}
  designators: {fn: designator_check}
  legal_form: {fn: legal_form_compat}
  country: {fn: exact, left: country}
score:
  weighted:
    weights: {f_name: 0.7, f_country: 0.2, f_legal_form: 0.1}
    maps: {f_legal_form: {same: 1.0, equivalent: 1.0, missing: 0.5, different: 0.0}}
route:
  rule:
    rules:
      - {feature: f_designators, equals: reject, route: reject}
      - {feature: f_designators, equals: review, route: review}
    then: {threshold: {accept_above: 0.95, reject_below: 0.60}}
cluster:
  guarded: {max_size: 50, strategy: drop_weakest}
```

```python
import pandas as pd
from deduplix import Pipeline

df = pd.DataFrame({
    "id": range(6),
    "name": ["Acme Growth Fund I LP", "Acme Growth Fund II LP", "ACME Growth Fund I, L.P.",
             "Beta Capital Partners SA", "Beta Capital Partners S.A.", "Zeta Ltd"],
    "country": ["US", "US", "US", "FR", "FR", "UK"],
    "lei": [None, None, None, "L1", "L1", None],
    "registration_no": [None] * 6,
})
result = Pipeline.from_config("funds.yaml").dedupe(df)
print(result.pairs[["i1", "i2", "f_name", "f_designators", "route", "decision"]])
```

Fund I and Fund II are rejected by the designator rule even though their names are nearly identical.

## Cross-dataset linkage with LEI

An `exact` blocker on the LEI proposes every pair that shares one, and a rule routes those pairs straight to `accept`. The remaining candidates come from name similarity.

```python
import pandas as pd
from deduplix import Pipeline

a = pd.DataFrame({"id": ["a1", "a2", "a3"],
                  "name": ["Acme Fund LP", "Beta Holdings", "Gamma Trust"],
                  "lei": ["L1", None, "L3"]})
b = pd.DataFrame({"id": ["b1", "b2", "b3"],
                  "name": ["ACME Fund L.P.", "Totally Different Name", "Gamma Trust Co"],
                  "lei": ["L1", "L3", None]})

link = Pipeline.from_config({
    "normalize": ["default_text"],
    "blocking": [{"exact": {"keys": ["lei"]}}, "tfidf_char"],
    "compare": {"name": {"fn": "token_set_ratio", "left": "name_norm"},
                "lei": {"fn": "exact", "left": "lei"}},
    "score": {"weighted": {"weights": {"f_name": 0.5, "f_lei": 0.5}}},
    "route": {"rule": {
        "rules": [{"feature": "f_lei", "equals": 1, "route": "accept"}],
        "then": {"threshold": {"accept_above": 0.9, "reject_below": 0.5}}}},
})
result = link.link(a, b, id_column="id", name_column="name")
print(result.to_legacy().cross_matches)
```

Here `a3` and `b2` are linked by LEI alone although the names differ. If you would rather send such conflicts to review, route on `f_lei` and `f_name` together. `Pipeline.link` takes `right_id_column` and `right_name_column` when the two tables use different column names.

## LLM through a Databricks endpoint

Pairs the router cannot decide go to the model. Set `DATABRICKS_HOST` and `DATABRICKS_TOKEN` (or pass `host` and `token` to the provider) and the endpoint name in `DEDUPLIX_ENDPOINT`; `${...}` is resolved from the environment when the config is loaded.

```yaml
normalize: [default_text]
blocking: [tfidf_char]
compare:
  name: {fn: token_set_ratio, left: name_norm}
score: {feature: {name: name}}
route: {threshold: {accept_above: 0.98, reject_below: 0.5}}
validate:
  - llm:
      provider: {databricks: {endpoint: "${DEDUPLIX_ENDPOINT}", mode: chat}}
      prompt: pair_validation_v2.j2
      policy: {mode: bool, on_failure: unvalidated}
      metadata_columns: [country]
```

`on_failure: unvalidated` means a failed or malformed batch leaves the pairs undecided (they appear in `result.review`) instead of silently counting as matches. Check the file without credentials or data:

```python
from deduplix import load_pipeline

pipeline = load_pipeline("llm.yaml", allow_missing_env=True)
print([type(v).__name__ for v in pipeline.validators])
```

To exercise the same pipeline offline, swap the provider for a `callable` that returns the decisions JSON; the prompt, parser and policy are unchanged:

```python
from deduplix import Pipeline
from deduplix.evaluation.synthetic import make_synthetic


def fake_llm(messages, **kwargs):
    return {"decisions": [{"pair_index": i, "is_duplicate": True, "confidence": 0.9, "reason": "same entity"}
                          for i in range(10)]}


df, _ = make_synthetic(30)
pipeline = Pipeline(
    normalizers=["default_text"],
    blockers=["tfidf_char"],
    comparators=[{"string_sim": {"fn": "token_set_ratio", "left": "name_norm", "feature": "name"}}],
    scorer={"feature": {"name": "name"}},
    router={"threshold": {"accept_above": 0.99, "reject_below": 0.5}},
    validators=[{"llm": {"provider": {"callable": {"fn": fake_llm}}, "cache": None}}],
)
result = pipeline.dedupe(df)
print(result.pairs["decided_by"].value_counts())
```

## Distilling LLM labels into a GBDT scorer

LLM validation is accurate but slow and costly. Label a sample once, train a gradient-boosted scorer on the comparator features, and let it route most pairs so the LLM only sees the hard ones. Needs scikit-learn.

1. Run the pipeline with an LLM validator on a sample and keep `decision` (`match` or `non_match`) as labels.
2. Train with `train_scorer`; it accepts the pairs table and labels (1/0, or `match`/`non_match`, unlabelled pairs dropped) and returns a calibrated model plus a precision-by-band table.
3. Use the model through the `model` scorer.

The example uses the gold labels of the synthetic generator in place of LLM labels; in practice take them from `result.pairs["decision"]`.

```python
import numpy as np
from deduplix import Pipeline
from deduplix.evaluation.synthetic import make_synthetic
from deduplix.evaluation.train import train_scorer

df, gold = make_synthetic(80)
base = Pipeline(
    normalizers=["default_text"],
    blockers=["tfidf_char"],
    comparators=[
        {"string_sim": {"fn": "token_set_ratio", "left": "name_norm", "feature": "name"}},
        {"string_sim": {"fn": "jaro_winkler", "left": "name_norm", "feature": "jw"}},
    ],
    scorer={"feature": {"name": "name"}},
    router={"threshold": {"accept_above": 0.9, "reject_below": 0.6}},
)
pairs = base.dedupe(df).pairs
ids = df["id"].to_numpy()
truth = {tuple(sorted(p)): l for *p, l in zip(gold.id1, gold.id2, gold.label)}
labels = np.array([truth.get(tuple(sorted((ids[i], ids[j]))), 0) for i, j in zip(pairs.i1, pairs.i2)])

model, bands = train_scorer(pairs, labels, features=["name", "jw"])
print(bands)

distilled = Pipeline(
    normalizers=["default_text"],
    blockers=["tfidf_char"],
    comparators=base.comparators,
    scorer={"model": {"estimator": model}},
    router={"threshold": {"accept_above": 0.9, "reject_below": 0.3}},
)
print(distilled.dedupe(df).matches.shape)
```

A fitted model is a Python object, so `to_config()` writes it as `<python object>`. Persist it with joblib and pass the file path as `estimator` to keep the pipeline expressible in YAML.
