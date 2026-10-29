# Write your own component

Any stage can be extended in three ways. All three register under a kind (`normalizer`, `blocker`, `comparator`, `scorer`, `router`, `validator`, `clusterer`, `consolidator`, `provider`, `parser`, `hook`, `encoder`) and a name, after which the name works in YAML and in Python.

## Conventions

- A registered **class** is instantiated with its params as keyword arguments: `{my_router: {label: review}}` calls `MyRouter(label="review")`. No params means defaults.
- A registered **function** with params becomes `functools.partial(fn, **params)`; without params it is used as is.
- **Comparators take `feature=`**. The output column is `f_<feature>`. In the `compare:` section the key is the feature name and is passed as `feature=` for you, so `compare: {first: {fn: same_first_token}}` writes `f_first`. When you list a comparator in Python without `feature`, the default of your function is used.
- Comparators receive `(pairs, left, right, ctx)` and return a DataFrame with the same index as `pairs`. Use `pairs["i1"]` and `pairs["i2"]` as row positions into `left` and `right`.
- Routers receive `(pairs, ctx)` and return a Series of `accept`, `review` or `reject`. Scorers receive `(pairs, ctx)` and return a Series.
- Re-registering an existing name raises an error unless you pass `overwrite=True`.

## A function

```python
import pandas as pd
from deduplix import Pipeline, register


@register("comparator", "same_first_token")
def same_first_token(pairs, left, right, ctx, feature="same_first_token"):
    a = left["name_norm"].str.split().str[0].to_numpy()[pairs["i1"].to_numpy()]
    b = right["name_norm"].str.split().str[0].to_numpy()[pairs["i2"].to_numpy()]
    return pd.DataFrame({"f_" + feature: (a == b).astype("float32")}, index=pairs.index)


df = pd.DataFrame({"id": [1, 2, 3],
                   "name": ["Acme Fund LP", "Acme Fund L.P.", "Zeta Trust"]})
pipeline = Pipeline.from_config({
    "normalize": ["default_text"],
    "blocking": [{"brute_force": {"threshold": 40}}],
    "compare": {"first": {"fn": "same_first_token"}},
    "score": {"feature": {"name": "first"}},
    "route": {"threshold": {"accept_above": 0.9, "reject_below": 0.5}},
})
print(pipeline.dedupe(df).pairs[["i1", "i2", "f_first", "route"]])
```

## A class

Use a class when the component has parameters or state.

```python
import pandas as pd
from deduplix import Pipeline, register


@register("router", "always_review")
class AlwaysReview:
    """Send every pair to the validators."""

    def __init__(self, label="review"):
        self.label = label

    def __call__(self, pairs, ctx):
        return pd.Series(self.label, index=pairs.index)


df = pd.DataFrame({"id": [1, 2], "name": ["Acme Fund LP", "Acme Fund L.P."]})
pipeline = Pipeline(
    normalizers=["default_text"],
    blockers=[{"brute_force": {"threshold": 40}}],
    comparators=[{"string_sim": {"fn": "ratio", "left": "name_norm", "feature": "name"}}],
    scorer={"feature": {"name": "name"}},
    router={"always_review": {"label": "review"}},
)
print(pipeline.dedupe(df).pairs["route"].tolist())
```

Registered classes remember their constructor arguments, so `pipeline.to_config()` can write them back to YAML. Plain Python objects (lambdas, fitted models) are written as `<python object>` with a warning and cannot be reloaded from text.

## An entry point

To ship components in a package, declare them in the group `deduplix.<kind>s` (for example `deduplix.comparators`). They are loaded lazily the first time a name is not found, so importing deduplix stays fast.

```toml
# pyproject.toml of your package
[project.entry-points."deduplix.comparators"]
same_first_token = "my_pkg.compare:same_first_token"

[project.entry-points."deduplix.routers"]
always_review = "my_pkg.routing:AlwaysReview"
```

Entry points are loaded as plain objects, so the module does not need to call `register` itself. After installing the package, `same_first_token` is usable in YAML like any built-in; check with `deduplix.engine.registry.list_components("comparator")`.
