"""Generate golden fixtures from the legacy code path. Run only against 0.1.0 behaviour.

    python scripts/make_golden.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "data"))

from make_synthetic import make_companies, make_cross  # noqa: E402

from deduplix import DeduplicationPipeline, FuzzyMatcher  # noqa: E402

OUT = ROOT / "tests" / "golden"
SCORERS = ["token_sort_ratio", "token_set_ratio", "WRatio"]


def sorted_pairs(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["id1", "id2"]).reset_index(drop=True) if not df.empty else df


def relabel(groups: pd.DataFrame) -> pd.DataFrame:
    """Relabel group ids by the smallest entity id in each group (0 stays 0)."""
    g = groups.copy()
    mins = g[g.group_id > 0].groupby("group_id").entity_id.min().sort_values()
    mapping = {gid: i + 1 for i, gid in enumerate(mins.index)}
    g["group_id"] = g.group_id.map(lambda x: mapping.get(x, 0))
    return g.sort_values("entity_id").reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = make_companies(300)
    df.to_parquet(OUT / "input_companies.parquet")
    df1, df2 = make_cross(120, 150)
    df1.to_parquet(OUT / "input_cross1.parquet")
    df2.to_parquet(OUT / "input_cross2.parquet")
    work = df.rename(columns={"name": "name"})
    for scorer in SCORERS:
        for punct in (False, True):
            m = FuzzyMatcher(threshold=85.0, scorer=scorer, remove_punctuation=punct)
            sorted_pairs(m.find_matches(work).pairs).to_parquet(OUT / f"pairs_{scorer}_{int(punct)}.parquet")
    m = FuzzyMatcher(threshold=85.0)
    res = DeduplicationPipeline(m, checkpoint=False).run(df)
    relabel(res.entity_groups).to_parquet(OUT / "entity_groups.parquet")
    sorted_pairs(res.duplicate_pairs).to_parquet(OUT / "duplicate_pairs.parquet")
    cross = m.find_cross_matches(df1, df2, "company_id", "company_name", "entity_id", "name",
                                 name_columns2=["name", "trading_name"])
    sorted_pairs(cross.pairs).to_parquet(OUT / "cross_multi.parquet")
    cres = DeduplicationPipeline(m, checkpoint=False).run_cross_dataset(
        df1, df2, "company_id", "company_name", "entity_id", "name")
    cres.cross_matches.sort_values(["df1_id", "df2_id"]).reset_index(drop=True).to_parquet(
        OUT / "cross_matches.parquet")
    print("golden written to", OUT)


if __name__ == "__main__":
    main()
