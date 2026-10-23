"""Tests for the Phase 3 blockers."""
import logging

import numpy as np
import pandas as pd
import pytest

from deduplix.blocking import (
    BruteForceBlocker, ExactKeyBlocker, blocking_report, pair_completeness, pairs_quality,
    reduction_ratio, union_candidates,
)
from deduplix.engine import registry
from deduplix.engine.context import RunContext

NAMES = [
    "Acme Holdings Limited", "ACME Holdings Ltd", "Zenith Capital Partners",
    "Zenith Capital Partner", "Blue Ocean Fund", "Globex Corporation",
    "Globex Corp", "Northwind Traders", "Quantum Dynamics",
]
TRUE = {(0, 1), (2, 3), (5, 6)}


def make_df(country=None):
    df = pd.DataFrame({"id": range(len(NAMES)), "name": NAMES})
    if country is not None:
        df["country"] = country
    return df


def dedup_ctx(df):
    return RunContext(left=df, right=df, mode="dedup")


def run(blocker, df=None, right=None):
    df = make_df() if df is None else df
    ctx = RunContext(left=df, right=df if right is None else right,
                     mode="dedup" if right is None else "link")
    blocker.fit(ctx.right, ctx)
    return blocker.candidates(ctx.left, ctx), ctx


def keys(c):
    return set(zip(c["i1"].tolist(), c["i2"].tolist()))


def test_registered_names():
    for n in ["exact", "token_index", "tfidf_char", "phonetic", "minhash", "embedding", "brute_force"]:
        assert n in {r["name"] for r in registry.list_components("blocker")}


def test_import_without_optional_deps():
    import deduplix.blocking as b  # noqa: F401


# ---------------- brute force ----------------
def test_brute_force_finds_pairs_and_scores():
    c, _ = run(BruteForceBlocker(threshold=70, n_workers=1))
    assert TRUE <= keys(c)
    assert (c["i1"] < c["i2"]).all()
    assert c["score"].between(0.7, 1.0).all()


def test_brute_force_link_mode():
    left = make_df().iloc[:2]
    right = pd.DataFrame({"id": [1, 2], "name": ["acme holdings limited", "something else"]})
    c, _ = run(BruteForceBlocker(threshold=80, n_workers=1), left, right)
    assert keys(c) == {(0, 0), (1, 0)} or (0, 0) in keys(c)


def test_brute_force_partition_and_null_partition():
    df = make_df(["A", "B", "A", "A", "B", "A", "B", "A", "A"])
    c, _ = run(BruteForceBlocker(threshold=80, n_workers=1, partition_by="country"), df)
    assert (0, 1) not in keys(c)  # different countries
    assert (2, 3) in keys(c)
    df2 = df.copy()
    df2.loc[1, "country"] = None
    c2, _ = run(BruteForceBlocker(threshold=80, n_workers=1, partition_by="country"), df2)
    assert (0, 1) in keys(c2)  # compare_all
    c3, _ = run(BruteForceBlocker(threshold=80, n_workers=1, partition_by="country",
                                  null_partition="skip"), df2)
    assert (0, 1) not in keys(c3)


def test_deterministic():
    a, _ = run(BruteForceBlocker(threshold=70, n_workers=1))
    b, _ = run(BruteForceBlocker(threshold=70, n_workers=1))
    pd.testing.assert_frame_equal(a, b)


# ---------------- exact ----------------
def test_exact_keys_normalized_and_freq_cap(caplog):
    df = pd.DataFrame({
        "id": range(6), "name": list("abcdef"),
        "code": ["0012-ab", "12AB", "N/A", "N/A", "N/A", None],
    })
    b = ExactKeyBlocker(keys=["code"], normalizers={"code": "alnum"}, max_key_freq=2)
    with caplog.at_level(logging.WARNING, logger="deduplix.blocking.exact"):
        c, _ = run(b, df)
    assert keys(c) == {(0, 1)}
    assert "N/A" in caplog.text or "NA" in caplog.text
    assert b.skipped_keys["code"]


def test_exact_lei_validation():
    from deduplix.blocking.exact import norm_lei

    valid = "5493001KJTIIGC8Y1R12"  # a well-formed LEI (check digits 12)
    assert norm_lei(valid) == valid
    assert norm_lei(valid[:-1] + "3") is None
    assert norm_lei("short") is None


def test_exact_link_mode_and_partition():
    left = pd.DataFrame({"id": [1, 2], "name": ["x", "y"], "k": ["a", "b"], "c": ["US", "FR"]})
    right = pd.DataFrame({"id": [1, 2], "name": ["x", "y"], "k": ["b", "b"], "c": ["US", "FR"]})
    c, _ = run(ExactKeyBlocker(keys=["k"]), left, right)
    assert keys(c) == {(1, 0), (1, 1)}
    c, _ = run(ExactKeyBlocker(keys=["k"], partition_by="c"), left, right)
    assert keys(c) == {(1, 1)}


def test_exact_requires_keys():
    with pytest.raises(ValueError):
        ExactKeyBlocker(keys=[])


# ---------------- scipy/sklearn based ----------------
@pytest.mark.extras
def test_tfidf_char_dedup():
    pytest.importorskip("sklearn")
    from deduplix.blocking import TfidfCharBlocker

    c, _ = run(TfidfCharBlocker(min_df=1, threshold=0.5))
    assert {(0, 1), (2, 3), (5, 6)} <= keys(c)
    assert (c["i1"] < c["i2"]).all()
    assert c["score"].between(0.5, 1.0 + 1e-5).all()
    c2, _ = run(TfidfCharBlocker(min_df=1, threshold=0.5))
    pd.testing.assert_frame_equal(c, c2)


@pytest.mark.extras
def test_tfidf_char_link_and_partition():
    pytest.importorskip("sklearn")
    from deduplix.blocking import TfidfCharBlocker

    left = pd.DataFrame({"id": [1, 2], "name": ["Acme Holdings Limited", "Zenith Capital"],
                         "c": ["US", "FR"]})
    right = pd.DataFrame({"id": [1, 2], "name": ["ACME Holdings Ltd", "Zenith Capital Partners"],
                          "c": ["US", "US"]})
    c, _ = run(TfidfCharBlocker(min_df=1, threshold=0.3), left, right)
    assert {(0, 0), (1, 1)} <= keys(c)
    c, _ = run(TfidfCharBlocker(min_df=1, threshold=0.3, partition_by="c"), left, right)
    assert keys(c) == {(0, 0)}


@pytest.mark.extras
def test_token_index_idf_and_df_cap():
    pytest.importorskip("sklearn")
    from deduplix.blocking import TokenIndexBlocker

    names = ["Acme Fund", "Acme Fund Beta", "Globex Fund", "Zed Fund", "Quark Fund", "Omega Fund"]
    df = pd.DataFrame({"id": range(6), "name": names})
    b = TokenIndexBlocker(max_df=3, stopwords=None)
    c, ctx = run(b, df)
    assert keys(c) == {(0, 1)}  # "fund" (df=6) dropped, "acme" shared
    assert ctx.artifacts["token_index.dropped"]["fund"] == 6
    assert (c["i1"] < c["i2"]).all() and (c["score"] > 0).all()
    # without the cap the common token links many pairs
    c2, _ = run(TokenIndexBlocker(max_df=100, stopwords=None, top_k=50), df)
    assert (0, 1) in keys(c2)


@pytest.mark.extras
def test_token_index_partition():
    pytest.importorskip("sklearn")
    from deduplix.blocking import TokenIndexBlocker

    df = make_df(["A", "B", "A", "A", "B", "A", "B", "A", "A"])
    c, _ = run(TokenIndexBlocker(partition_by="country", stopwords=None), df)
    assert (0, 1) not in keys(c)
    assert (2, 3) in keys(c)


def test_missing_sklearn_message(monkeypatch):
    import importlib

    real = importlib.import_module

    def fake(name, *a, **k):
        if name == "sklearn":
            raise ImportError("no sklearn")
        return real(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake)
    from deduplix.blocking import TfidfCharBlocker

    with pytest.raises(ImportError, match="pip install"):
        TfidfCharBlocker().fit(make_df(), dedup_ctx(make_df()))


# ---------------- phonetic ----------------
@pytest.mark.extras
def test_phonetic_blocks():
    pytest.importorskip("metaphone")
    from deduplix.blocking import PhoneticBlocker

    df = pd.DataFrame({"id": range(4), "name": ["Smith Bank", "Smyth Bank", "Zorro Trust", "Other"]})
    c, _ = run(PhoneticBlocker(tokens="first", n=1), df)
    assert keys(c) == {(0, 1)}
    c, _ = run(PhoneticBlocker(tokens="first", n=1, max_block_size=1), df)
    assert len(c) == 0


@pytest.mark.extras
def test_phonetic_partition():
    pytest.importorskip("metaphone")
    from deduplix.blocking import PhoneticBlocker

    df = pd.DataFrame({"id": range(2), "name": ["Smith Bank", "Smyth Bank"], "c": ["A", "B"]})
    c, _ = run(PhoneticBlocker(tokens="first", n=1, partition_by="c"), df)
    assert len(c) == 0


# ---------------- minhash ----------------
@pytest.mark.extras
def test_minhash_finds_similar_and_is_deterministic():
    pytest.importorskip("datasketch")
    from deduplix.blocking import MinHashLSHBlocker

    c, _ = run(MinHashLSHBlocker(threshold=0.3, num_perm=128))
    assert (2, 3) in keys(c) and (0, 1) in keys(c)
    assert (c["i1"] < c["i2"]).all()
    c2, _ = run(MinHashLSHBlocker(threshold=0.3, num_perm=128))
    pd.testing.assert_frame_equal(c, c2)


# ---------------- embedding ----------------
def _hash_encoder(texts):
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for n, t in enumerate(texts):
        t = t.lower()
        for i in range(len(t) - 2):
            out[n, hash(t[i:i + 3]) % 64] += 1
    return out


def _stable_encoder(texts):
    import zlib

    out = np.zeros((len(texts), 64), dtype=np.float32)
    for n, t in enumerate(texts):
        t = t.lower()
        for i in range(len(t) - 2):
            out[n, zlib.crc32(t[i:i + 3].encode()) % 64] += 1
    return out


def test_embedding_exact_index_with_callable_and_cache(tmp_path):
    from deduplix.blocking import EmbeddingANNBlocker

    mk = lambda: EmbeddingANNBlocker(  # noqa: E731
        encoder=_stable_encoder, index="exact", text_template="{name}", prefix="",
        min_cosine=0.8, top_k=3, cache_dir=str(tmp_path))
    c, _ = run(mk())
    assert (2, 3) in keys(c) and (0, 1) in keys(c)
    assert (c["i1"] < c["i2"]).all()
    assert list(tmp_path.glob("*.npy"))
    calls = []
    b = mk()
    b.encoder = lambda texts: calls.append(1) or _stable_encoder(texts)
    c2, _ = run(b)
    assert not calls  # all embeddings came from the cache
    pd.testing.assert_frame_equal(c, c2)


def test_embedding_partition():
    from deduplix.blocking import EmbeddingANNBlocker

    df = make_df(["A", "B", "A", "A", "B", "A", "B", "A", "A"])
    c, _ = run(EmbeddingANNBlocker(encoder=_stable_encoder, index="exact", text_template="{name}",
                                   prefix="", min_cosine=0.8, partition_by="country"), df)
    assert (0, 1) not in keys(c) and (2, 3) in keys(c)


@pytest.mark.extras
def test_embedding_hnsw():
    pytest.importorskip("hnswlib")
    from deduplix.blocking import EmbeddingANNBlocker

    c, _ = run(EmbeddingANNBlocker(encoder=_stable_encoder, index="hnsw", text_template="{name}",
                                   prefix="", min_cosine=0.8))
    assert (2, 3) in keys(c)


# ---------------- union / metrics ----------------
def test_union_and_metrics_report():
    a = pd.DataFrame({"i1": [0, 2], "i2": [1, 3], "score": [0.9, 0.8]})
    b = pd.DataFrame({"i1": [0, 5], "i2": [1, 6]})
    u = union_candidates({"x": a, "y": b}, n_right=9, mode="dedup")
    gold = pd.DataFrame({"i1": [0, 2, 5, 7], "i2": [1, 3, 6, 8]})
    assert pair_completeness(u, gold) == 0.75
    assert pairs_quality(u, gold) == 1.0
    rep = blocking_report(u, gold).set_index("blocker")
    assert rep.loc["x", "candidates"] == 2 and rep.loc["y", "candidates"] == 2
    assert rep.loc["x", "unique_true_matches"] == 1  # (2, 3)
    assert rep.loc["y", "unique_true_matches"] == 1  # (5, 6)
    assert rep.loc["union", "candidates"] == 3


def test_reduction_ratio():
    assert reduction_ratio(10, 5, 5, "dedup") == 0.0
    assert reduction_ratio(0, 100, 100, "link") == 1.0
    assert reduction_ratio(500, 100, 100, "link") == pytest.approx(0.95)
