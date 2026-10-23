"""Store the prompts the legacy LLMValidator builds, so wording changes are deliberate."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deduplix.validators import LLMValidator  # noqa: E402


def make_validator(**kw):
    v = object.__new__(LLMValidator)  # skip client init: no network, no API key
    v.custom_rules = kw.get("custom_rules", {})
    v.metadata_columns = kw.get("metadata_columns", [])
    return v


BATCH = pd.DataFrame(
    {"id1": ["a", "b"], "id2": ["c", "d"], "name1": ["Acme Ltd", "Foo Fund IV"],
     "name2": ["ACME Limited", "Foo Fund I"], "similarity_score": [91.234, 88.0]},
    index=[10, 11],
)
ORIG = pd.DataFrame({"id": ["a", "b", "c", "d"], "name": list("wxyz"),
                     "country": ["US", "GB", "US", "GB"]})


def cases():
    yield "plain", make_validator()._create_prompt(BATCH)
    yield "metadata", make_validator(metadata_columns=["country", "missing"])._create_prompt(BATCH, ORIG)
    yield "custom", make_validator(custom_rules={"custom_instructions": "Fund numbers matter."})._create_prompt(BATCH)


if __name__ == "__main__":
    out = ROOT / "tests" / "golden" / "legacy_prompts.json"
    out.write_text(json.dumps(dict(cases()), indent=1), encoding="utf-8")
    print("wrote", out)
