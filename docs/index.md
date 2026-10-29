# deduplix

deduplix finds duplicate entities in one dataset and links entities across two datasets. Version 0.2 adds a staged, pluggable engine next to the 0.1 API, which keeps working unchanged.

```python
import pandas as pd
from deduplix import Pipeline

df = pd.DataFrame({"id": [1, 2], "name": ["Acme Holdings Ltd", "ACME Holdings Limited"]})
result = Pipeline.default().dedupe(df, id_column="id", name_column="name")
result.matches      # pairs decided as duplicates
result.review       # pairs no stage could decide
result.clusters     # cluster sizes
```

- [Concepts](concepts.md): stages, the pairs table, routing.
- [Config reference](config-reference.md) and [component catalogue](components.md): generated from the code.
- [Write your own component](custom-components.md).
- [Migration 0.1 to 0.2](migration.md): nothing is required.
- [Cookbook](cookbook.md): fund vehicles, LEI linkage, LLM validation, distilling LLM labels.

The generated pages are rebuilt with `python scripts/gen_docs.py`; build the site with `mkdocs build`.
