"""SQLite decision cache keyed by a sha256 of everything that determines the answer."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from typing import Any, Optional

from ..exceptions import ConfigurationError
from .parsers import Decision

DEFAULT_CACHE = "sqlite:///.deduplix_llm_cache.db"


def cache_key(provider: str, model: Optional[str], template_name: str, template_version: str,
              instructions: Optional[str], content: Any) -> str:
    payload = json.dumps([provider, model, template_name, template_version, instructions or "",
                          content], sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SqliteCache:
    """Stores parsed Decisions (not policy outcomes), so threshold tuning reuses them."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        with self._lock, self._conn:
            self._conn.execute("CREATE TABLE IF NOT EXISTS decisions "
                               "(key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def get(self, key: str) -> Optional[Decision]:
        with self._lock:
            row = self._conn.execute("SELECT value FROM decisions WHERE key=?", (key,)).fetchone()
        return Decision(**json.loads(row[0])) if row else None

    def set(self, key: str, decision: Decision) -> None:
        value = json.dumps(decision.dict() if hasattr(decision, "dict") else decision.model_dump())
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO decisions (key, value) VALUES (?, ?)",
                               (key, value))

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def open_cache(spec: Any) -> Optional[SqliteCache]:
    """``None`` -> no cache; ``'sqlite:///path'`` (or a bare path) -> SqliteCache."""
    if spec is None or spec is False:
        return None
    if isinstance(spec, SqliteCache):
        return spec
    if isinstance(spec, str):
        path = spec[len("sqlite:///"):] if spec.startswith("sqlite:///") else spec
        if spec.startswith("sqlite:") and not spec.startswith("sqlite:///"):
            raise ConfigurationError(f"Unsupported cache '{spec}'; use 'sqlite:///path'")
        return SqliteCache(path or ":memory:")
    raise ConfigurationError(f"Unsupported cache {spec!r}")
