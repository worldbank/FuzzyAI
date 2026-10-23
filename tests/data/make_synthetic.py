"""Deterministic synthetic company data (seed 42) shared by golden tests and benchmarks."""
from __future__ import annotations

import random
from typing import Tuple

import pandas as pd

_STEMS = [
    "Alpha", "Borealis", "Cobalt", "Delta", "Evergreen", "Falcon", "Granite", "Horizon",
    "Iris", "Juniper", "Keystone", "Lumen", "Meridian", "Nova", "Orion", "Pinnacle",
    "Quartz", "Redwood", "Summit", "Titan", "Umbra", "Vertex", "Willow", "Zenith",
]
_SECTORS = [
    "Capital", "Holdings", "Energy", "Logistics", "Foods", "Mining", "Textiles",
    "Pharma", "Software", "Shipping", "Steel", "Telecom", "Agro", "Power", "Water",
]
_FORMS = ["Ltd", "Inc", "LLC", "Corp", "PLC", "GmbH", "SA", "Co"]
_COUNTRIES = ["US", "GB", "DE", "MX", "IN", "KE", "BR", "ZA"]


def _typo(name: str, rng: random.Random) -> str:
    if len(name) < 4:
        return name
    i = rng.randrange(1, len(name) - 1)
    return name[:i] + name[i + 1] + name[i] + name[i + 2:]


def _variant(name: str, rng: random.Random) -> str:
    kind = rng.choice(["typo", "form", "case", "punct", "order"])
    if kind == "typo":
        return _typo(name, rng)
    if kind == "form":
        parts = name.split()
        parts[-1] = rng.choice(_FORMS)
        return " ".join(parts)
    if kind == "case":
        return name.upper()
    if kind == "punct":
        return name.replace(" ", ", ", 1) + "."
    parts = name.split()
    return " ".join(reversed(parts[:-1])) + " " + parts[-1] if len(parts) > 2 else name


def _base_names(n: int, rng: random.Random) -> list:
    names, seen = [], set()
    while len(names) < n:
        nm = f"{rng.choice(_STEMS)} {rng.choice(_SECTORS)} {rng.choice(_STEMS)} {rng.choice(_FORMS)}"
        if nm not in seen:
            seen.add(nm)
            names.append(nm)
    return names


def make_companies(n: int = 2000, dup_rate: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Return ``n`` rows (``id``, ``name``, ``country``) with injected near-duplicates."""
    rng = random.Random(seed)
    n_dups = int(n * dup_rate)
    base = _base_names(n - n_dups, rng)
    rows = [(nm, rng.choice(_COUNTRIES)) for nm in base]
    for _ in range(n_dups):
        nm, c = rng.choice(rows[: len(base)])
        rows.append((_variant(nm, rng), c))
    rng.shuffle(rows)
    return pd.DataFrame(
        {"id": [f"C{i:05d}" for i in range(len(rows))],
         "name": [r[0] for r in rows], "country": [r[1] for r in rows]}
    )


def make_cross(n1: int = 1000, n2: int = 1500, overlap: float = 0.4,
               seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return two frames (``company_id``/``company_name``; ``entity_id``/``name``) sharing entities."""
    rng = random.Random(seed)
    names = _base_names(n1 + n2, rng)
    shared = int(min(n1, n2) * overlap)
    left = names[:n1]
    right = [_variant(x, rng) for x in left[:shared]] + names[n1: n1 + n2 - shared]
    rng.shuffle(right)
    df1 = pd.DataFrame({"company_id": [f"A{i:05d}" for i in range(n1)],
                        "company_name": left, "sector": "x"})
    df2 = pd.DataFrame({"entity_id": [f"B{i:05d}" for i in range(len(right))],
                        "name": right, "trading_name": [r.split()[0] for r in right]})
    return df1, df2
