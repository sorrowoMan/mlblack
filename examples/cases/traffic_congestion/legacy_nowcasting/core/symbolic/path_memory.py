from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlparse


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expr_hash(expr_key: str) -> str:
    return hashlib.sha1(str(expr_key).encode("utf-8")).hexdigest()


def default_path_memory_db() -> Path:
    return Path.home() / ".mlblack" / "symbolic_path_memory.sqlite3"


@dataclass(frozen=True)
class PathPrior:
    seen: int = 0
    success: int = 0
    failure: int = 0
    total_delta_rmse: float = 0.0
    total_selected_score: float = 0.0

    @property
    def outcomes(self) -> int:
        return int(max(0, int(self.success) + int(self.failure)))

    @property
    def accept_rate(self) -> float:
        n = int(self.outcomes)
        if n <= 0:
            return 0.5
        return float(self.success) / float(n)

    @property
    def avg_delta_rmse(self) -> float:
        n = int(self.outcomes)
        if n <= 0:
            return 0.0
        return float(self.total_delta_rmse) / float(n)

    @property
    def avg_selected_score(self) -> float:
        n = int(self.outcomes)
        if n <= 0:
            return 0.0
        return float(self.total_selected_score) / float(n)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "seen": int(self.seen),
            "success": int(self.success),
            "failure": int(self.failure),
            "outcomes": int(self.outcomes),
            "accept_rate": float(self.accept_rate),
            "avg_delta_rmse": float(self.avg_delta_rmse),
            "avg_selected_score": float(self.avg_selected_score),
        }


class SymbolicPathMemory:
    """Persistent path memory for symbolic structure search.

    Supported backends:
    - SQLite file path (default)
    - PostgreSQL DSN: `postgresql://...` or `postgres://...` (requires `psycopg`)
    - MySQL DSN: `mysql://...` (requires `pymysql`)
    """

    def __init__(self, *, db_path: str | None = None, namespace: str = "global") -> None:
        raw = str(db_path or "").strip()
        self.namespace = str(namespace or "global")
        self.backend = self._detect_backend(raw)
        self.path = raw
        self._conn: Any = None

        if self.backend == "sqlite":
            path = Path(raw).expanduser() if raw else default_path_memory_db()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.path = str(path)
            self._conn = sqlite3.connect(str(path))
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
        elif self.backend == "postgres":
            try:
                import psycopg
            except Exception as exc:
                raise ImportError("PostgreSQL backend requires `psycopg` package.") from exc
            if not raw:
                raise ValueError("PostgreSQL backend requires DSN like postgresql://user:pass@host:5432/db")
            self._conn = psycopg.connect(raw)
        elif self.backend == "mysql":
            try:
                import pymysql
            except Exception as exc:
                raise ImportError("MySQL backend requires `pymysql` package.") from exc
            if not raw:
                raise ValueError("MySQL backend requires DSN like mysql://user:pass@host:3306/db")
            self._conn = self._connect_mysql(raw, pymysql)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self._ensure_schema()

    @staticmethod
    def _detect_backend(raw: str) -> str:
        key = str(raw).strip().lower()
        if key.startswith("postgres://") or key.startswith("postgresql://"):
            return "postgres"
        if key.startswith("mysql://"):
            return "mysql"
        return "sqlite"

    @staticmethod
    def _connect_mysql(dsn: str, pymysql_mod: Any):
        parsed = urlparse(dsn)
        if parsed.scheme.lower() != "mysql":
            raise ValueError("MySQL DSN must start with mysql://")
        db_name = parsed.path.lstrip("/")
        if not db_name:
            raise ValueError("MySQL DSN must include database name, e.g. mysql://user:pass@host:3306/db")
        return pymysql_mod.connect(
            host=parsed.hostname or "127.0.0.1",
            port=int(parsed.port or 3306),
            user=unquote(parsed.username or ""),
            password=unquote(parsed.password or ""),
            database=db_name,
            charset="utf8mb4",
            autocommit=False,
        )

    @staticmethod
    def genome_signature(expr_keys: Sequence[str]) -> str:
        payload = "\x1f".join(str(v) for v in expr_keys)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def _execute(self, query: str, params: Sequence[Any] = ()) -> Any:
        q = str(query)
        if self.backend in {"postgres", "mysql"}:
            q = q.replace("?", "%s")
        cur = self._conn.cursor()
        cur.execute(q, tuple(params))
        return cur

    def _execute_nonquery(self, query: str, params: Sequence[Any] = ()) -> None:
        cur = self._execute(query, params)
        cur.close()

    def _query_one(self, query: str, params: Sequence[Any] = ()) -> tuple[Any, ...] | None:
        cur = self._execute(query, params)
        row = cur.fetchone()
        cur.close()
        return row

    def _commit(self) -> None:
        self._conn.commit()

    def _ensure_schema(self) -> None:
        if self.backend == "mysql":
            self._execute_nonquery(
                """
                CREATE TABLE IF NOT EXISTS expr_stats (
                    namespace VARCHAR(128) NOT NULL,
                    expr_hash CHAR(40) NOT NULL,
                    expr_key TEXT NOT NULL,
                    seen BIGINT NOT NULL DEFAULT 0,
                    success BIGINT NOT NULL DEFAULT 0,
                    failure BIGINT NOT NULL DEFAULT 0,
                    total_delta_rmse DOUBLE NOT NULL DEFAULT 0.0,
                    total_selected_score DOUBLE NOT NULL DEFAULT 0.0,
                    updated_at VARCHAR(64) NOT NULL,
                    PRIMARY KEY (namespace, expr_hash)
                )
                """
            )
            self._execute_nonquery(
                """
                CREATE TABLE IF NOT EXISTS edge_stats (
                    namespace VARCHAR(128) NOT NULL,
                    src_sig CHAR(40) NOT NULL,
                    op VARCHAR(32) NOT NULL,
                    expr_hash CHAR(40) NOT NULL,
                    expr_key TEXT NOT NULL,
                    dst_sig CHAR(40) NOT NULL,
                    seen BIGINT NOT NULL DEFAULT 0,
                    success BIGINT NOT NULL DEFAULT 0,
                    failure BIGINT NOT NULL DEFAULT 0,
                    total_delta_rmse DOUBLE NOT NULL DEFAULT 0.0,
                    updated_at VARCHAR(64) NOT NULL,
                    PRIMARY KEY (namespace, src_sig, op, expr_hash, dst_sig)
                )
                """
            )
            self._commit()
            return

        # SQLite / PostgreSQL
        self._execute_nonquery(
            """
            CREATE TABLE IF NOT EXISTS expr_stats (
                namespace TEXT NOT NULL,
                expr_hash TEXT NOT NULL,
                expr_key TEXT NOT NULL,
                seen INTEGER NOT NULL DEFAULT 0,
                success INTEGER NOT NULL DEFAULT 0,
                failure INTEGER NOT NULL DEFAULT 0,
                total_delta_rmse REAL NOT NULL DEFAULT 0.0,
                total_selected_score REAL NOT NULL DEFAULT 0.0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (namespace, expr_hash)
            )
            """
        )
        self._execute_nonquery(
            """
            CREATE TABLE IF NOT EXISTS edge_stats (
                namespace TEXT NOT NULL,
                src_sig TEXT NOT NULL,
                op TEXT NOT NULL,
                expr_hash TEXT NOT NULL,
                expr_key TEXT NOT NULL,
                dst_sig TEXT NOT NULL,
                seen INTEGER NOT NULL DEFAULT 0,
                success INTEGER NOT NULL DEFAULT 0,
                failure INTEGER NOT NULL DEFAULT 0,
                total_delta_rmse REAL NOT NULL DEFAULT 0.0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (namespace, src_sig, op, expr_hash, dst_sig)
            )
            """
        )
        self._commit()

    def get_expr_prior(self, expr_key: str) -> PathPrior:
        key = str(expr_key)
        h = _expr_hash(key)
        row = self._query_one(
            """
            SELECT seen, success, failure, total_delta_rmse, total_selected_score
            FROM expr_stats
            WHERE namespace = ? AND expr_hash = ?
            """,
            (self.namespace, h),
        )
        if row is None:
            return PathPrior()
        return PathPrior(
            seen=int(row[0]),
            success=int(row[1]),
            failure=int(row[2]),
            total_delta_rmse=float(row[3]),
            total_selected_score=float(row[4]),
        )

    def touch_expr(self, expr_key: str) -> None:
        key = str(expr_key)
        h = _expr_hash(key)
        now = _utc_now()
        if self.backend == "mysql":
            self._execute_nonquery(
                """
                INSERT INTO expr_stats (
                    namespace, expr_hash, expr_key, seen, success, failure,
                    total_delta_rmse, total_selected_score, updated_at
                )
                VALUES (?, ?, ?, 1, 0, 0, 0.0, 0.0, ?)
                ON DUPLICATE KEY UPDATE
                    seen = seen + 1,
                    updated_at = VALUES(updated_at)
                """,
                (self.namespace, h, key, now),
            )
        else:
            self._execute_nonquery(
                """
                INSERT INTO expr_stats (
                    namespace, expr_hash, expr_key, seen, success, failure,
                    total_delta_rmse, total_selected_score, updated_at
                )
                VALUES (?, ?, ?, 1, 0, 0, 0.0, 0.0, ?)
                ON CONFLICT(namespace, expr_hash) DO UPDATE SET
                    seen = seen + 1,
                    updated_at = excluded.updated_at
                """,
                (self.namespace, h, key, now),
            )
        self._commit()

    def record_expr_outcome(
        self,
        expr_key: str,
        *,
        selected_score: float,
        delta_rmse: float,
        success: bool,
    ) -> None:
        key = str(expr_key)
        h = _expr_hash(key)
        now = _utc_now()
        suc = 1 if bool(success) else 0
        fail = 0 if bool(success) else 1

        if self.backend == "mysql":
            self._execute_nonquery(
                """
                INSERT INTO expr_stats (
                    namespace, expr_hash, expr_key, seen, success, failure,
                    total_delta_rmse, total_selected_score, updated_at
                )
                VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
                ON DUPLICATE KEY UPDATE
                    seen = seen + 1,
                    success = success + VALUES(success),
                    failure = failure + VALUES(failure),
                    total_delta_rmse = total_delta_rmse + VALUES(total_delta_rmse),
                    total_selected_score = total_selected_score + VALUES(total_selected_score),
                    updated_at = VALUES(updated_at)
                """,
                (
                    self.namespace,
                    h,
                    key,
                    int(suc),
                    int(fail),
                    float(delta_rmse),
                    float(selected_score),
                    now,
                ),
            )
        else:
            self._execute_nonquery(
                """
                INSERT INTO expr_stats (
                    namespace, expr_hash, expr_key, seen, success, failure,
                    total_delta_rmse, total_selected_score, updated_at
                )
                VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, expr_hash) DO UPDATE SET
                    seen = seen + 1,
                    success = success + excluded.success,
                    failure = failure + excluded.failure,
                    total_delta_rmse = total_delta_rmse + excluded.total_delta_rmse,
                    total_selected_score = total_selected_score + excluded.total_selected_score,
                    updated_at = excluded.updated_at
                """,
                (
                    self.namespace,
                    h,
                    key,
                    int(suc),
                    int(fail),
                    float(delta_rmse),
                    float(selected_score),
                    now,
                ),
            )
        self._commit()

    def record_edge(
        self,
        *,
        src_sig: str,
        op: str,
        expr_key: str,
        dst_sig: str,
        delta_rmse: float,
        success: bool,
    ) -> None:
        key = str(expr_key)
        h = _expr_hash(key)
        now = _utc_now()
        suc = 1 if bool(success) else 0
        fail = 0 if bool(success) else 1

        if self.backend == "mysql":
            self._execute_nonquery(
                """
                INSERT INTO edge_stats (
                    namespace, src_sig, op, expr_hash, expr_key, dst_sig,
                    seen, success, failure, total_delta_rmse, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                ON DUPLICATE KEY UPDATE
                    seen = seen + 1,
                    success = success + VALUES(success),
                    failure = failure + VALUES(failure),
                    total_delta_rmse = total_delta_rmse + VALUES(total_delta_rmse),
                    updated_at = VALUES(updated_at)
                """,
                (
                    self.namespace,
                    str(src_sig),
                    str(op),
                    h,
                    key,
                    str(dst_sig),
                    int(suc),
                    int(fail),
                    float(delta_rmse),
                    now,
                ),
            )
        else:
            self._execute_nonquery(
                """
                INSERT INTO edge_stats (
                    namespace, src_sig, op, expr_hash, expr_key, dst_sig,
                    seen, success, failure, total_delta_rmse, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                ON CONFLICT(namespace, src_sig, op, expr_hash, dst_sig) DO UPDATE SET
                    seen = seen + 1,
                    success = success + excluded.success,
                    failure = failure + excluded.failure,
                    total_delta_rmse = total_delta_rmse + excluded.total_delta_rmse,
                    updated_at = excluded.updated_at
                """,
                (
                    self.namespace,
                    str(src_sig),
                    str(op),
                    h,
                    key,
                    str(dst_sig),
                    int(suc),
                    int(fail),
                    float(delta_rmse),
                    now,
                ),
            )
        self._commit()


__all__ = [
    "PathPrior",
    "SymbolicPathMemory",
    "default_path_memory_db",
]
