"""Benchmark legacy cdist against blocked engine pipelines on synthetic data.

    python benchmarks/run.py --sizes 10000 165000 --configs legacy tfidf tfidf_token_exact

Writes benchmarks/results/<date>.md and .json. Never run in CI (marked slow in tests).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "data"))

from make_synthetic import make_companies  # noqa: E402

from deduplix import FuzzyMatcher, Pipeline  # noqa: E402
from deduplix.engine import schema as S  # noqa: E402
from deduplix.evaluation.metrics import pair_metrics  # noqa: E402

THRESHOLD = 85.0


class PeakRSS:
    """Sample process RSS in a background thread (psutil)."""

    def __init__(self) -> None:
        import psutil

        self.proc, self.peak, self._stop = psutil.Process(), 0, threading.Event()

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                self.peak = max(self.peak, self.proc.memory_info().rss)
                time.sleep(0.05)

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def gold_pairs(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("entity")["id"].apply(list)
    rows = []
    for ids in g:
        ids = sorted(ids)
        rows += [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]
    out = pd.DataFrame(rows, columns=["id1", "id2"])
    out["label"] = 1
    return out


def engine_config(blockers: List[Any]) -> Dict[str, Any]:
    return {
        "normalize": ["default_text"],
        "blocking": blockers,
        "compare": {"name": {"fn": "token_sort_ratio", "left": "name_norm"}},
        "score": {"feature": {"name": "name"}},
        "route": {"threshold": {"accept_above": THRESHOLD / 100, "reject_below": THRESHOLD / 100}},
        "options": {"legacy_score_feature": "name"},
    }


CONFIGS: Dict[str, Any] = {
    "tfidf": [{"tfidf_char": {"top_n": 20, "threshold": 0.5}}],
    "tfidf_token_exact": [
        {"tfidf_char": {"top_n": 20, "threshold": 0.5}},
        {"token_index": {"max_df": 1000, "top_k": 50}},
    ],
    "tfidf_token_embedding": [
        {"tfidf_char": {"top_n": 20, "threshold": 0.5}},
        {"token_index": {"max_df": 1000, "top_k": 50}},
        {"embedding": {"top_k": 20, "min_cosine": 0.8}},
    ],
}


def run_legacy(df: pd.DataFrame) -> Dict[str, Any]:
    t = time.perf_counter()
    res = FuzzyMatcher(threshold=THRESHOLD, n_workers=-1).find_matches(df[["id", "name"]])
    return {"pairs": res.pairs[["id1", "id2"]], "seconds": {"match": time.perf_counter() - t},
            "candidates": len(df) * (len(df) - 1) // 2}


def run_engine(df: pd.DataFrame, blockers: List[Any]) -> Dict[str, Any]:
    pipe = Pipeline.from_config(engine_config(blockers))
    res = pipe.dedupe(df[["id", "name"]])
    m = res.matches
    ids = df["id"].to_numpy()
    pairs = pd.DataFrame({"id1": ids[m[S.I1].to_numpy()], "id2": ids[m[S.I2].to_numpy()]})
    return {"pairs": pairs, "seconds": {s["stage"]: s["seconds"] for s in res.stats["stages"]},
            "candidates": res.stats["n_pairs"], "usage": res.stats.get("usage", {})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10_000])
    ap.add_argument("--configs", nargs="+", default=["legacy", "tfidf", "tfidf_token_exact"])
    ap.add_argument("--max-legacy", type=int, default=50_000,
                    help="skip the O(n^2) legacy config above this size")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    for n in args.sizes:
        df = make_companies(n, with_entity=True)
        gold = gold_pairs(df)
        for name in args.configs:
            if name == "legacy" and n > args.max_legacy:
                print(f"skip legacy at n={n:,}")
                continue
            print(f"n={n:,} config={name}")
            with PeakRSS() as mem:
                out = run_legacy(df) if name == "legacy" else run_engine(df, CONFIGS[name])
            pm = pair_metrics(out["pairs"], gold)
            total_pairs = n * (n - 1) // 2
            rows.append({
                "n": n, "config": name, "seconds": out["seconds"],
                "total_seconds": float(sum(out["seconds"].values())),
                "peak_rss_mb": round(mem.peak / 2**20, 1),
                "candidates": int(out["candidates"]),
                "reduction_ratio": 1 - out["candidates"] / max(total_pairs, 1),
                "precision": pm["precision"], "recall": pm["recall"], "f1": pm["f1"],
                "llm_usage": out.get("usage", {}),
            })

    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    (out_dir / f"{stamp}.json").write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
    lines = ["| n | config | seconds | peak MB | candidates | reduction | P | R | F1 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['n']:,} | {r['config']} | {r['total_seconds']:.1f} | {r['peak_rss_mb']} | "
                     f"{r['candidates']:,} | {r['reduction_ratio']:.4f} | {r['precision']:.3f} | "
                     f"{r['recall']:.3f} | {r['f1']:.3f} |")
    (out_dir / f"{stamp}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
