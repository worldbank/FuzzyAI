"""Embedding + approximate nearest neighbour blocker.

Failure mode to expect: names such as "National Bank of Kenya" / "National Bank of Egypt"
and "Fund I" / "Fund IV" sit close in embedding space. This blocker buys recall; precision
comes from later stages.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..engine.registry import register
from .base import BaseBlocker, require

logger = logging.getLogger("deduplix.blocking.embedding")


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


@register("encoder", "sentence_transformers")
class SentenceTransformersEncoder:
    """``encode(texts) -> float32 unit vectors`` with a sentence-transformers model."""

    def __init__(self, model: str = "intfloat/multilingual-e5-small", batch_size: int = 256,
                 device: Optional[str] = None) -> None:
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self._m = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._m is None:
            st = require("sentence_transformers", "embeddings", "sentence-transformers")
            self._m = st.SentenceTransformer(self.model, device=self.device)
        return _unit(self._m.encode(list(texts), batch_size=self.batch_size,
                                    show_progress_bar=False, convert_to_numpy=True))


@register("encoder", "databricks_embeddings")
class DatabricksEmbeddingsEncoder:
    """Embeddings from a Databricks model-serving endpoint (needs ``mlflow``)."""

    def __init__(self, endpoint: str, batch_size: int = 256) -> None:
        self.endpoint = endpoint
        self.batch_size = batch_size

    def encode(self, texts: List[str]) -> np.ndarray:
        deployments = require("mlflow.deployments", "embeddings", "mlflow")
        client = deployments.get_deploy_client("databricks")
        out = []
        for s in range(0, len(texts), self.batch_size):
            resp = client.predict(endpoint=self.endpoint,
                                  inputs={"input": list(texts[s:s + self.batch_size])})
            out.extend(d["embedding"] for d in resp["data"])
        return _unit(np.array(out, dtype=np.float32))


@register("encoder", "callable")
class CallableEncoder:
    """Wrap ``fn(list[str]) -> array`` as an encoder."""

    def __init__(self, fn: Callable) -> None:
        self.fn = fn

    def encode(self, texts: List[str]) -> np.ndarray:
        return _unit(self.fn(list(texts)))


class _Row(dict):
    def __missing__(self, key):
        return ""


@register("blocker", "embedding")
class EmbeddingANNBlocker(BaseBlocker):
    """Cosine nearest neighbours of text embeddings (hnswlib, faiss IVF-PQ, or exact numpy).

    Parameters
    ----------
    encoder : str or object
        ``sentence_transformers``, ``databricks_embeddings``, ``callable`` (give ``fn`` in
        ``encoder_params``), or any object with ``encode(list[str]) -> ndarray``.
    text_template : str
        ``str.format`` template over the record's columns (missing or null -> empty).
    index : {"hnsw", "faiss_ivfpq", "exact"}
        ANN index; ``exact`` is a dense numpy search for small data and tests.
    cache_dir : str, optional
        Embeddings are cached at ``cache_dir/<sha1(model + text)>.npy``.
    """

    name = "embedding"
    default_top_k = 20

    def __init__(self, encoder: Any = "sentence_transformers",
                 model: str = "intfloat/multilingual-e5-small",
                 text_template: str = "{name_base} {city} {country}", prefix: str = "query: ",
                 batch_size: int = 256, index: str = "hnsw",
                 hnsw: Optional[Dict[str, int]] = None, faiss: Optional[Dict[str, int]] = None,
                 top_k: int = 20, min_cosine: float = 0.80, cache_dir: Optional[str] = None,
                 encoder_params: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        super().__init__(top_k=top_k, **kwargs)
        if index not in ("hnsw", "faiss_ivfpq", "exact"):
            raise ValueError("index must be 'hnsw', 'faiss_ivfpq' or 'exact'")
        self.encoder = encoder
        self.model = model
        self.text_template = text_template
        self.prefix = prefix
        self.batch_size = batch_size
        self.index = index
        self.hnsw = {"M": 32, "ef_construction": 200, "ef": 100, **(hnsw or {})}
        self.faiss = {"nlist": 1024, "m": 16, "nbits": 8, **(faiss or {})}
        self.min_cosine = min_cosine
        self.cache_dir = cache_dir
        self.encoder_params = dict(encoder_params or {})

    def _check_deps(self) -> None:
        if self.index == "hnsw":
            require("hnswlib", "embeddings")
        elif self.index == "faiss_ivfpq":
            require("faiss", "embeddings", "faiss-cpu")

    def _encoder(self):
        if hasattr(self.encoder, "encode"):
            return self.encoder
        if callable(self.encoder) and not isinstance(self.encoder, str):
            return CallableEncoder(self.encoder)
        if self.encoder == "sentence_transformers":
            return SentenceTransformersEncoder(self.model, self.batch_size, **self.encoder_params)
        if self.encoder == "databricks_embeddings":
            return DatabricksEmbeddingsEncoder(batch_size=self.batch_size, **self.encoder_params)
        if self.encoder == "callable":
            return CallableEncoder(**self.encoder_params)
        raise ValueError(f"Unknown encoder '{self.encoder}'")

    def _texts_for(self, df, ctx) -> List[str]:
        cols = [c for c in df.columns]
        data = {c: df[c].tolist() for c in cols}
        if "name_base" not in data:
            alt = "name_norm" if "name_norm" in data else ctx.columns.name
            data.setdefault("name_base", data.get(alt, [""] * len(df)))
        out = []
        for n in range(len(df)):
            row = _Row()
            for c, vals in data.items():
                v = vals[n]
                row[c] = "" if v is None or v is pd_NA or (isinstance(v, float) and v != v) else v
            out.append(" ".join((self.prefix + self.text_template.format_map(row)).split()))
        return out

    def _embed(self, texts: List[str]) -> np.ndarray:
        enc = self._encoder()
        tag = self.model if isinstance(self.encoder, str) else type(enc).__name__
        paths = None
        vecs: List[Optional[np.ndarray]] = [None] * len(texts)
        todo = list(range(len(texts)))
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            paths = [os.path.join(self.cache_dir,
                                  hashlib.sha1((tag + t).encode("utf-8")).hexdigest() + ".npy")
                     for t in texts]
            todo = []
            for n, p in enumerate(paths):
                if os.path.isfile(p):
                    vecs[n] = np.load(p)
                else:
                    todo.append(n)
        if todo:
            uniq: Dict[str, int] = {}
            for n in todo:
                uniq.setdefault(texts[n], len(uniq))
            enc_out = _unit(enc.encode(list(uniq)))
            for n in todo:
                vecs[n] = enc_out[uniq[texts[n]]]
                if paths:
                    np.save(paths[n], vecs[n])
        return np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, 1), np.float32)

    def _prepare(self, left, right, ctx) -> None:
        self._El = self._embed(self._texts_for(left, ctx))
        self._Er = self._El if self._dedup else self._embed(self._texts_for(right, ctx))

    def _knn(self, Er: np.ndarray, El: np.ndarray, k: int):
        n, dim = Er.shape
        if self.index == "exact" or n <= 2:
            sims = El @ Er.T
            kk = min(k, n)
            idx = np.argsort(-sims, axis=1, kind="stable")[:, :kk]
            return idx, np.take_along_axis(sims, idx, axis=1)
        if self.index == "hnsw":
            import hnswlib

            h = self.hnsw
            ix = hnswlib.Index(space="cosine", dim=dim)
            ix.init_index(max_elements=n, ef_construction=h["ef_construction"], M=h["M"],
                          random_seed=int(getattr(self, "_seed", 0)))
            ix.add_items(Er, np.arange(n))
            ix.set_ef(max(h["ef"], k))
            labels, dist = ix.knn_query(El, k=min(k, n))
            return labels, 1.0 - dist
        import faiss

        f = self.faiss
        nlist = max(1, min(f["nlist"], n // 39 or 1))
        quant = faiss.IndexFlatIP(dim)
        ix = faiss.IndexIVFPQ(quant, dim, nlist, f["m"], f["nbits"], faiss.METRIC_INNER_PRODUCT)
        ix.train(Er)
        ix.add(Er)
        ix.nprobe = min(nlist, 16)
        sims, labels = ix.search(El, min(k, n))
        return labels, sims

    def _block(self, left_idx, right_idx, ctx):
        self._seed = ctx.random_state
        k = int(self.top_k) + (1 if self._dedup else 0)
        labels, sims = self._knn(self._Er[right_idx], self._El[left_idx], k)
        rows = np.repeat(np.arange(len(left_idx)), labels.shape[1])
        cols, sc = labels.reshape(-1).astype(np.int64), sims.reshape(-1).astype(np.float32)
        keep = (cols >= 0) & (sc >= self.min_cosine)
        if self._dedup:
            keep &= cols != rows
        return rows[keep], cols[keep], sc[keep]


try:  # pandas NA sentinel without importing pandas at module top twice
    from pandas import NA as pd_NA
except Exception:  # pragma: no cover
    pd_NA = object()
