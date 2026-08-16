from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Mapping

import numpy as np

from core.symbolic.symbolic_dsl import evaluate_expression_numpy, expression_to_string
from core.symbolic.symbolic_gradient import differentiate_expression_wrt_feature

try:  # pragma: no cover - optional dependency
    import lmdb  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    lmdb = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ExpressionGraphCacheStats:
    value_hits: int
    value_misses: int
    derivative_hits: int
    derivative_misses: int
    value_entries: int
    derivative_entries: int
    backend: str
    namespace: str
    persistent_derivative_hits: int
    persistent_derivative_misses: int
    persistent_derivative_writes: int


class ExpressionGraphCache:
    """Reusable compute-graph cache for symbolic expression/gradient evaluation."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_value_entries: int = 20000,
        max_derivative_entries: int = 50000,
        backend: str = "memory",
        db_path: str = "",
        namespace: str = "global",
        persist_values: bool = False,
        sqlite_commit_every: int = 64,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_value_entries = max(1, int(max_value_entries))
        self.max_derivative_entries = max(1, int(max_derivative_entries))
        self.backend = self._normalize_backend(backend)
        self.db_path = str(db_path or "")
        self.namespace = str(namespace or "global")
        self.persist_values = bool(persist_values)
        self.sqlite_commit_every = max(1, int(sqlite_commit_every))

        self._value_cache: OrderedDict[tuple[str, str, tuple[tuple[str, float], ...]], np.ndarray] = OrderedDict()
        self._derivative_cache: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()

        self._value_hits = 0
        self._value_misses = 0
        self._derivative_hits = 0
        self._derivative_misses = 0
        self._persistent_derivative_hits = 0
        self._persistent_derivative_misses = 0
        self._persistent_derivative_writes = 0

        self._sqlite_conn: sqlite3.Connection | None = None
        self._sqlite_dirty_writes = 0
        self._lmdb_env: Any | None = None
        self._resolved_db_path: str = ""

        if self.enabled:
            self._init_persistent_backend()

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        key = str(backend or "memory").strip().lower()
        if key in {"memory", "mem", "none", "off"}:
            return "memory"
        if key in {"sqlite", "sqlite3"}:
            return "sqlite"
        if key == "lmdb":
            return "lmdb"
        raise ValueError(f"Unsupported graph cache backend: {backend}")

    @staticmethod
    def _default_db_path(backend: str) -> Path:
        root = Path(".mlblack_cache")
        if backend == "sqlite":
            return root / "expression_graph_cache.sqlite3"
        if backend == "lmdb":
            return root / "expression_graph_cache.lmdb"
        return root / "expression_graph_cache.cache"

    def _init_persistent_backend(self) -> None:
        if self.backend == "memory":
            return
        if self.backend == "sqlite":
            self._init_sqlite()
            return
        if self.backend == "lmdb":
            self._init_lmdb()
            return

    def _resolve_persistent_path(self) -> Path:
        if self.db_path:
            p = Path(self.db_path).expanduser()
        else:
            p = self._default_db_path(self.backend).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _init_sqlite(self) -> None:
        path = self._resolve_persistent_path()
        conn = sqlite3.connect(str(path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS derivative_cache (
                namespace TEXT NOT NULL,
                expr_key TEXT NOT NULL,
                feature_index INTEGER NOT NULL,
                derivative_expr_json TEXT NOT NULL,
                derivative_expr_key TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY(namespace, expr_key, feature_index)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_derivative_cache_expr
            ON derivative_cache(namespace, expr_key)
            """
        )
        conn.commit()
        self._sqlite_conn = conn
        self._resolved_db_path = str(path)
        self._sqlite_dirty_writes = 0

    def _init_lmdb(self) -> None:
        if lmdb is None:  # pragma: no cover - optional dependency
            raise RuntimeError("LMDB backend requested but 'lmdb' package is not installed")
        path = self._resolve_persistent_path()
        path.mkdir(parents=True, exist_ok=True)
        self._lmdb_env = lmdb.open(  # type: ignore[union-attr]
            str(path),
            map_size=512 * 1024 * 1024,
            subdir=True,
            max_dbs=1,
            readonly=False,
            lock=True,
            readahead=False,
            meminit=False,
        )
        self._resolved_db_path = str(path)

    def _persistent_derivative_key(self, expr_key: str, feature_index: int) -> bytes:
        return f"d|{self.namespace}|{expr_key}|{int(feature_index)}".encode("utf-8")

    def _load_derivative_from_persistent(self, *, expr_key: str, feature_index: int) -> dict[str, Any] | None:
        if not self.enabled or self.backend == "memory":
            return None

        if self.backend == "sqlite":
            conn = self._sqlite_conn
            if conn is None:
                return None
            row = conn.execute(
                """
                SELECT derivative_expr_json, derivative_expr_key
                FROM derivative_cache
                WHERE namespace = ? AND expr_key = ? AND feature_index = ?
                """,
                (self.namespace, str(expr_key), int(feature_index)),
            ).fetchone()
            if row is None:
                self._persistent_derivative_misses += 1
                return None
            try:
                expr = json.loads(str(row[0]))
                key = str(row[1]) if row[1] else self.expression_key(expr)
            except Exception:
                self._persistent_derivative_misses += 1
                return None
            self._persistent_derivative_hits += 1
            return {"expr": expr, "key": key}

        if self.backend == "lmdb":  # pragma: no cover - optional dependency
            env = self._lmdb_env
            if env is None:
                return None
            key_bytes = self._persistent_derivative_key(expr_key=str(expr_key), feature_index=int(feature_index))
            with env.begin(write=False) as txn:
                blob = txn.get(key_bytes)
            if blob is None:
                self._persistent_derivative_misses += 1
                return None
            try:
                item = json.loads(bytes(blob).decode("utf-8"))
                expr = item["expr"]
                dkey = str(item.get("key") or self.expression_key(expr))
            except Exception:
                self._persistent_derivative_misses += 1
                return None
            self._persistent_derivative_hits += 1
            return {"expr": expr, "key": dkey}

        return None

    def _store_derivative_to_persistent(
        self,
        *,
        expr_key: str,
        feature_index: int,
        item: Mapping[str, Any],
    ) -> None:
        if not self.enabled or self.backend == "memory":
            return

        if self.backend == "sqlite":
            conn = self._sqlite_conn
            if conn is None:
                return
            payload = json.dumps(dict(item["expr"]), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            conn.execute(
                """
                INSERT INTO derivative_cache (
                    namespace, expr_key, feature_index, derivative_expr_json, derivative_expr_key, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, expr_key, feature_index)
                DO UPDATE SET
                    derivative_expr_json = excluded.derivative_expr_json,
                    derivative_expr_key = excluded.derivative_expr_key,
                    updated_at = excluded.updated_at
                """,
                (
                    self.namespace,
                    str(expr_key),
                    int(feature_index),
                    payload,
                    str(item["key"]),
                    float(time.time()),
                ),
            )
            self._sqlite_dirty_writes += 1
            if self._sqlite_dirty_writes >= int(self.sqlite_commit_every):
                conn.commit()
                self._sqlite_dirty_writes = 0
            self._persistent_derivative_writes += 1
            return

        if self.backend == "lmdb":  # pragma: no cover - optional dependency
            env = self._lmdb_env
            if env is None:
                return
            key_bytes = self._persistent_derivative_key(expr_key=str(expr_key), feature_index=int(feature_index))
            payload = json.dumps(
                {
                    "expr": dict(item["expr"]),
                    "key": str(item["key"]),
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            with env.begin(write=True) as txn:
                txn.put(key_bytes, payload)
            self._persistent_derivative_writes += 1

    @staticmethod
    def expression_key(expr: Mapping[str, Any]) -> str:
        return expression_to_string(expr, param_values=None, precision=12)

    @staticmethod
    def _param_signature(param_values: Mapping[str, float] | None) -> tuple[tuple[str, float], ...]:
        if not param_values:
            return tuple()
        out: list[tuple[str, float]] = []
        for k, v in sorted(param_values.items(), key=lambda kv: str(kv[0])):
            out.append((str(k), float(v)))
        return tuple(out)

    @staticmethod
    def _batch_token(X: np.ndarray, *, batch_key: str | None = None) -> str:
        if batch_key is not None:
            return str(batch_key)
        arr = np.asarray(X)
        return f"id:{id(arr)}:{arr.shape}"

    def _touch_value(self, key: tuple[str, str, tuple[tuple[str, float], ...]], value: np.ndarray) -> None:
        self._value_cache[key] = value
        self._value_cache.move_to_end(key)
        while len(self._value_cache) > int(self.max_value_entries):
            self._value_cache.popitem(last=False)

    def _touch_derivative(self, key: tuple[str, int], value: dict[str, Any]) -> None:
        self._derivative_cache[key] = value
        self._derivative_cache.move_to_end(key)
        while len(self._derivative_cache) > int(self.max_derivative_entries):
            self._derivative_cache.popitem(last=False)

    def evaluate_expression(
        self,
        expr: Mapping[str, Any],
        X: np.ndarray,
        *,
        param_values: Mapping[str, float] | None = None,
        eps: float = 1e-6,
        expr_key: str | None = None,
        batch_key: str | None = None,
    ) -> np.ndarray:
        if not self.enabled:
            return evaluate_expression_numpy(expr, X, param_values=param_values, eps=float(eps))

        ekey = str(expr_key) if expr_key else self.expression_key(expr)
        token = self._batch_token(X, batch_key=batch_key)
        p_sig = self._param_signature(param_values)
        cache_key = (ekey, token, p_sig)

        cached = self._value_cache.get(cache_key)
        if cached is not None:
            self._value_hits += 1
            self._value_cache.move_to_end(cache_key)
            return cached

        self._value_misses += 1
        out = evaluate_expression_numpy(expr, X, param_values=param_values, eps=float(eps))
        value = np.asarray(out, dtype=float).reshape(-1)
        self._touch_value(cache_key, value)
        return value

    def differentiate_wrt_feature(
        self,
        expr: Mapping[str, Any],
        *,
        feature_index: int,
        eps: float = 1e-6,
        expr_key: str | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            d = differentiate_expression_wrt_feature(expr, feature_index=int(feature_index), eps=float(eps))
            return {"expr": d, "key": self.expression_key(d)}

        ekey = str(expr_key) if expr_key else self.expression_key(expr)
        key = (ekey, int(feature_index))
        cached = self._derivative_cache.get(key)
        if cached is not None:
            self._derivative_hits += 1
            self._derivative_cache.move_to_end(key)
            return cached

        persistent = self._load_derivative_from_persistent(expr_key=ekey, feature_index=int(feature_index))
        if persistent is not None:
            self._derivative_hits += 1
            self._touch_derivative(key, dict(persistent))
            return dict(persistent)

        self._derivative_misses += 1
        d_expr = differentiate_expression_wrt_feature(expr, feature_index=int(feature_index), eps=float(eps))
        item = {
            "expr": d_expr,
            "key": self.expression_key(d_expr),
        }
        self._touch_derivative(key, item)
        self._store_derivative_to_persistent(expr_key=ekey, feature_index=int(feature_index), item=item)
        return item

    def evaluate_gradient(
        self,
        expr: Mapping[str, Any],
        X: np.ndarray,
        *,
        feature_index: int,
        param_values: Mapping[str, float] | None = None,
        eps: float = 1e-6,
        expr_key: str | None = None,
        batch_key: str | None = None,
    ) -> np.ndarray:
        deriv = self.differentiate_wrt_feature(
            expr,
            feature_index=int(feature_index),
            eps=float(eps),
            expr_key=expr_key,
        )
        return self.evaluate_expression(
            deriv["expr"],
            X,
            param_values=param_values,
            eps=float(eps),
            expr_key=str(deriv["key"]),
            batch_key=batch_key,
        )

    def clear(self) -> None:
        self._value_cache.clear()
        self._derivative_cache.clear()

    def close(self) -> None:
        if self._sqlite_conn is not None:
            try:
                if self._sqlite_dirty_writes > 0:
                    self._sqlite_conn.commit()
                    self._sqlite_dirty_writes = 0
            finally:
                self._sqlite_conn.close()
                self._sqlite_conn = None
        if self._lmdb_env is not None:  # pragma: no cover - optional dependency
            self._lmdb_env.close()
            self._lmdb_env = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def stats(self) -> ExpressionGraphCacheStats:
        return ExpressionGraphCacheStats(
            value_hits=int(self._value_hits),
            value_misses=int(self._value_misses),
            derivative_hits=int(self._derivative_hits),
            derivative_misses=int(self._derivative_misses),
            value_entries=int(len(self._value_cache)),
            derivative_entries=int(len(self._derivative_cache)),
            backend=str(self.backend),
            namespace=str(self.namespace),
            persistent_derivative_hits=int(self._persistent_derivative_hits),
            persistent_derivative_misses=int(self._persistent_derivative_misses),
            persistent_derivative_writes=int(self._persistent_derivative_writes),
        )

    def snapshot(self) -> dict[str, Any]:
        s = self.stats()
        return {
            "value_hits": int(s.value_hits),
            "value_misses": int(s.value_misses),
            "derivative_hits": int(s.derivative_hits),
            "derivative_misses": int(s.derivative_misses),
            "value_entries": int(s.value_entries),
            "derivative_entries": int(s.derivative_entries),
            "backend": str(s.backend),
            "namespace": str(s.namespace),
            "db_path": str(self._resolved_db_path),
            "persistent_derivative_hits": int(s.persistent_derivative_hits),
            "persistent_derivative_misses": int(s.persistent_derivative_misses),
            "persistent_derivative_writes": int(s.persistent_derivative_writes),
        }


__all__ = [
    "ExpressionGraphCache",
    "ExpressionGraphCacheStats",
]
