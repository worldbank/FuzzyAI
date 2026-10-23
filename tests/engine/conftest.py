"""Small stand-in components so engine tests do not depend on later phases."""
import numpy as np
import pandas as pd
import pytest

from deduplix.engine import registry
from deduplix.engine.schema import I1, I2


class AllPairsBlocker:
    name = "allpairs"

    def fit(self, right, ctx):
        self.n = len(right)
        return self

    def candidates(self, left, ctx):
        a, b = np.triu_indices(len(left), 1) if ctx.mode == "dedup" else np.divmod(
            np.arange(len(left) * self.n), self.n)
        return pd.DataFrame({I1: a, I2: b, "score": np.ones(len(a), dtype=np.float32)})


class FirstLetterBlocker:
    name = "firstletter"

    def fit(self, right, ctx):
        self.keys = right[ctx.columns.rname].str[0].str.lower().to_numpy()
        return self

    def candidates(self, left, ctx):
        k = left[ctx.columns.name].str[0].str.lower().to_numpy()
        a, b = np.nonzero(k[:, None] == self.keys[None, :])
        return pd.DataFrame({I1: a, I2: b})


class LenRatio:
    """Comparator: f_len = min/max name length."""

    def __init__(self, feature="len"):
        self.feature = feature

    def __call__(self, pairs, left, right, ctx):
        n1 = left[ctx.columns.name].str.len().to_numpy()[pairs[I1]]
        n2 = right[ctx.columns.rname].str.len().to_numpy()[pairs[I2]]
        return pd.DataFrame({f"f_{self.feature}": (np.minimum(n1, n2) / np.maximum(n1, n2)).astype("float32")},
                            index=pairs.index)


def same_name(pairs, left, right, ctx):
    a = left[ctx.columns.name].to_numpy()[pairs[I1]]
    b = right[ctx.columns.rname].to_numpy()[pairs[I2]]
    return pd.DataFrame({"f_same": (np.char.lower(a.astype(str)) == np.char.lower(b.astype(str))).astype("float32")},
                        index=pairs.index)


@pytest.fixture
def people():
    return pd.DataFrame({"id": [10, 11, 12, 13], "name": ["Acme", "ACME", "Acne", "Zed"]})


@pytest.fixture(autouse=True)
def clean_registry():
    """Remove test registrations after each test."""
    before = {k: set(v) for k, v in registry._REGISTRY.items()}
    yield
    for k, names in registry._REGISTRY.items():
        for n in list(names):
            if n not in before[k]:
                del names[n]
