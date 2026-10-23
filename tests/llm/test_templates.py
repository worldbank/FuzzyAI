import json
import sys
from pathlib import Path

import pytest

from deduplix.llm.templates import PromptTemplate
from deduplix.exceptions import ConfigurationError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.make_prompt_snapshot import BATCH, ORIG  # noqa: E402

GOLDEN = json.loads((ROOT / "tests" / "golden" / "legacy_prompts.json").read_text(encoding="utf-8"))


def v1_pairs(metadata_columns=()):
    pairs = []
    for idx, row in BATCH.iterrows():
        left, right = {"name": row["name1"]}, {"name": row["name2"]}
        for col in metadata_columns:
            if col in ORIG.columns:
                left[col] = ORIG.loc[ORIG["id"] == row["id1"], col].iloc[0]
                right[col] = ORIG.loc[ORIG["id"] == row["id2"], col].iloc[0]
        pairs.append({"idx": idx, "left": left, "right": right,
                      "features": {"similarity_score": row["similarity_score"]}})
    return pairs


def test_v1_plain_byte_for_byte():
    t = PromptTemplate.load("pair_validation_v1.j2")
    assert t.render(v1_pairs()) == GOLDEN["plain"]


def test_v1_metadata_byte_for_byte():
    cols = ["country", "missing"]
    t = PromptTemplate.load("pair_validation_v1.j2")
    assert t.render(v1_pairs(cols), metadata_columns=cols) == GOLDEN["metadata"]


def test_v1_custom_instructions_byte_for_byte():
    t = PromptTemplate.load("pair_validation_v1.j2")
    assert t.render(v1_pairs(), instructions="Fund numbers matter.") == GOLDEN["custom"]


def test_v2_has_no_similarity_and_shows_fields():
    t = PromptTemplate.load("pair_validation_v2.j2")
    pairs = [{"idx": 0, "left": {"name": "Acme Ltd", "country": "US", "legal_form": "LTD"},
              "right": {"name": "ACME Limited", "country": "US", "legal_form": "LTD"},
              "features": {"f_name": 0.93}}]
    text = t.render(pairs, metadata_columns=["country"], parsed_columns=["legal_form"],
                    instructions="Be strict.")
    assert "similarity" not in text.lower()
    assert "Pair 0: 'Acme Ltd' vs 'ACME Limited'" in text
    assert "country: US vs US" in text and "legal_form: LTD vs LTD" in text
    assert "f_name: 0.93" in text and "- Be strict." in text and "confidence" in text


def test_strict_undefined_and_unknown_template(tmp_path):
    p = tmp_path / "t.j2"
    p.write_text("{{ nope }}", encoding="utf-8")
    with pytest.raises(Exception):
        PromptTemplate.load(str(p)).render([])
    with pytest.raises(ConfigurationError):
        PromptTemplate.load("does_not_exist.j2")
