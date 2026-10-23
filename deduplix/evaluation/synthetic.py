"""Synthetic company records with gold labels and hard negatives."""
from __future__ import annotations

import random
import unicodedata
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

LEGAL_FORMS = ["Ltd", "Inc", "Corp", "GmbH", "SA", "LLC", "AG"]
FORM_SWAP = {"Ltd": "Limited", "Limited": "Ltd", "Inc": "Corp", "Corp": "Inc", "GmbH": "AG",
             "AG": "GmbH", "SA": "S.A.", "LLC": "Ltd"}
ABBREV = {"International": "Intl", "Holdings": "Hldgs", "Technologies": "Tech", "Group": "Grp",
          "Management": "Mgmt", "Capital": "Cap", "Partners": "Ptrs", "Systems": "Sys",
          "Industries": "Ind", "Logistics": "Logist"}
# u-umlaut, o-umlaut, a-umlaut, sharp s, e-acute, e-grave, o-slash, ae-ligature
TRANSLIT = {"ü": "ue", "ö": "oe", "ä": "ae", "ß": "ss", "é": "e",
            "è": "e", "ø": "o", "æ": "ae"}
FIRST = ["Müller", "Schröder", "Nordic", "Alpine", "Orion", "Atlas", "Zenith", "Borealis",
         "Meridian", "Sovereign", "Cobalt", "Jünger", "Lefèvre", "Højgaard", "Granite",
         "Falcon", "Summit", "Vertex", "Harbor", "Crescent", "Delta", "Evergreen", "Pioneer",
         "Quantum", "Café", "Bär"]
SECOND = ["Global", "Pacific", "Atlantic", "Continental", "Premier", "Dynamic", "United", "Allied",
          "Strategic", "Northern", "Southern", "Central"]
NOUN = ["International", "Holdings", "Technologies", "Group", "Management", "Capital", "Partners",
        "Systems", "Industries", "Logistics", "Energy", "Foods"]
COUNTRIES = ["US", "GB", "DE", "FR", "CH", "DK", "NL", "JP", "BR", "IN"]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
VARIANTS = ["typo", "legal_form", "abbreviation", "word_order", "accent_removal", "transliteration"]
HARD_NEGATIVES = ["fund_number", "feeder_master", "manager_fund", "other_country"]


def _typo(s: str, rng: random.Random) -> str:
    idx = [i for i, c in enumerate(s) if c.isalpha()]
    if len(idx) < 4:
        return s
    i = rng.choice(idx[1:-1])
    op = rng.choice(["swap", "drop", "dup"])
    if op == "swap" and i + 1 < len(s):
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if op == "drop":
        return s[:i] + s[i + 1:]
    return s[:i] + s[i] + s[i:]


def _legal_form(s: str, rng: random.Random) -> str:
    toks = s.split()
    if toks[-1] in FORM_SWAP:
        toks[-1] = FORM_SWAP[toks[-1]]
        return " ".join(toks)
    return s + " Ltd"


def _abbreviation(s: str, rng: random.Random) -> str:
    return " ".join(ABBREV.get(t, t) for t in s.split())


def _word_order(s: str, rng: random.Random) -> str:
    toks = s.split()
    if len(toks) < 3:
        return s
    core = toks[:-1]
    return " ".join(core[1:] + core[:1] + [toks[-1]])


def _accent(s: str, rng: random.Random) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _translit(s: str, rng: random.Random) -> str:
    return "".join(TRANSLIT.get(c, c) for c in s)


_FUNCS: Dict[str, Callable[[str, random.Random], str]] = {
    "typo": _typo, "legal_form": _legal_form, "abbreviation": _abbreviation,
    "word_order": _word_order, "accent_removal": _accent, "transliteration": _translit,
}


def _base_name(rng: random.Random) -> str:
    return " ".join([rng.choice(FIRST), rng.choice(SECOND), rng.choice(NOUN), rng.choice(LEGAL_FORMS)])


def make_synthetic(n_entities: int, dup_rate: float = 0.3, variants: Optional[Sequence[str]] = None,
                   seed: int = 42, hard_negative_rate: float = 0.1
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate records and gold pairs.

    Returns
    -------
    records : ``id, name, country, entity_id``.
    gold : ``id1, id2, label, kind`` (``id1 < id2``). ``label=1`` for records of the same
        entity (``kind`` is the variant that made the pair); ``label=0`` for hard
        negatives: ``fund_number`` ("Fund I" vs "Fund IV"), ``feeder_master``,
        ``manager_fund`` and ``other_country`` (same name, different country).
    """
    variants = list(variants) if variants is not None else list(VARIANTS)
    bad = set(variants) - set(_FUNCS)
    if bad:
        raise ValueError("Unknown variants %s; choose from %s" % (sorted(bad), VARIANTS))
    rng = random.Random(seed)
    recs: List[Dict] = []
    neg: List[Dict] = []

    def add(name: str, country: str, ent: int, kind: Optional[str]) -> int:
        recs.append({"id": "S%06d" % len(recs), "name": name, "country": country,
                     "entity_id": ent, "_k": kind})
        return len(recs) - 1

    canon: Dict[int, int] = {}
    for e in range(n_entities):
        name, country = _base_name(rng), rng.choice(COUNTRIES)
        canon[e] = add(name, country, e, None)
        if rng.random() < dup_rate:
            for _ in range(rng.randint(1, 3)):
                kind = rng.choice(variants)
                v = _FUNCS[kind](name, rng)
                if v == name:
                    kind, v = "typo", _typo(name, rng)
                add(v, country, e, kind)

    next_ent = n_entities
    for _ in range(int(round(hard_negative_rate * n_entities))):
        base = recs[canon[rng.randrange(n_entities)]]
        toks = base["name"].split()
        core, form = " ".join(toks[:-1]), toks[-1]
        kind = rng.choice(HARD_NEGATIVES)
        if kind == "fund_number":
            a, b = rng.sample(range(len(ROMAN)), 2)
            r1 = add("%s Fund %s %s" % (core, ROMAN[a], form), base["country"], next_ent, None)
            r2 = add("%s Fund %s %s" % (core, ROMAN[b], form), base["country"], next_ent + 1, None)
            next_ent += 2
        else:
            r1 = recs.index(base)
            if kind == "feeder_master":
                nm, ctry = "%s Feeder Fund %s" % (core, form), base["country"]
            elif kind == "manager_fund":
                nm, ctry = "%s Opportunities Fund %s" % (core, form), base["country"]
            else:
                nm = base["name"]
                ctry = rng.choice([c for c in COUNTRIES if c != base["country"]])
            r2 = add(nm, ctry, next_ent, None)
            next_ent += 1
        neg.append({"a": r1, "b": r2, "label": 0, "kind": kind})

    records = pd.DataFrame(recs)
    rows: List[Dict] = []
    for _, idx in records.groupby("entity_id").indices.items():
        idx = sorted(int(i) for i in idx)
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                a, b = idx[x], idx[y]
                rows.append({"a": a, "b": b, "label": 1,
                             "kind": recs[b]["_k"] or recs[a]["_k"] or "exact"})
    rows += neg
    ids = records["id"].to_numpy()
    gold = pd.DataFrame({
        "id1": [ids[min(r["a"], r["b"])] for r in rows],
        "id2": [ids[max(r["a"], r["b"])] for r in rows],
        "label": [r["label"] for r in rows],
        "kind": [r["kind"] for r in rows],
    })
    return records.drop(columns="_k"), gold
