from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple


@dataclass
class ContextStore:
    """In-memory lightweight runtime key-value store."""

    _data: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        k = str(key).strip()
        if not k:
            raise ValueError("context key must not be empty")
        self._data[k] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(str(key), default)

    def pop(self, key: str, default: Any = None) -> Any:
        return self._data.pop(str(key), default)

    def has(self, key: str) -> bool:
        return str(key) in self._data

    def update(self, mapping: Mapping[str, Any]) -> None:
        for k, v in dict(mapping).items():
            self.set(str(k), v)

    def set_many(self, items: Iterable[tuple[str, Any]]) -> None:
        for k, v in items:
            self.set(str(k), v)

    def keys(self) -> Tuple[str, ...]:
        return tuple(sorted(self._data.keys()))

    def items(self) -> Tuple[tuple[str, Any], ...]:
        return tuple((k, self._data[k]) for k in self.keys())

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def clear(self) -> None:
        self._data.clear()


@dataclass
class SQLiteContextStore:
    """SQLite-backed lightweight runtime key-value store."""

    db_path: str
    namespace: str = "default"

    def __post_init__(self) -> None:
        p = Path(self.db_path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(p)
        self.namespace = str(self.namespace).strip() or "default"
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        sql = """
        CREATE TABLE IF NOT EXISTS context_kv (
            namespace TEXT NOT NULL,
            key TEXT NOT NULL,
            value_json TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY (namespace, key)
        );
        """
        with closing(self._connect()) as conn:
            conn.execute(sql)
            conn.commit()

    @staticmethod
    def _dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def _loads(text: str) -> Any:
        return json.loads(str(text))

    def set(self, key: str, value: Any) -> None:
        k = str(key).strip()
        if not k:
            raise ValueError("context key must not be empty")
        now = datetime.now(timezone.utc).isoformat()
        payload = self._dumps(value)
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO context_kv(namespace, key, value_json, updated_at_utc)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (self.namespace, k, payload, now),
            )
            conn.commit()

    def get(self, key: str, default: Any = None) -> Any:
        k = str(key)
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT value_json FROM context_kv WHERE namespace=? AND key=?",
                (self.namespace, k),
            ).fetchone()
        if row is None:
            return default
        return self._loads(str(row[0]))

    def pop(self, key: str, default: Any = None) -> Any:
        k = str(key)
        out = self.get(k, default=default)
        with closing(self._connect()) as conn:
            conn.execute(
                "DELETE FROM context_kv WHERE namespace=? AND key=?",
                (self.namespace, k),
            )
            conn.commit()
        return out

    def has(self, key: str) -> bool:
        k = str(key)
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT 1 FROM context_kv WHERE namespace=? AND key=? LIMIT 1",
                (self.namespace, k),
            ).fetchone()
        return bool(row is not None)

    def update(self, mapping: Mapping[str, Any]) -> None:
        for k, v in dict(mapping).items():
            self.set(str(k), v)

    def set_many(self, items: Iterable[tuple[str, Any]]) -> None:
        for k, v in items:
            self.set(str(k), v)

    def keys(self) -> Tuple[str, ...]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT key FROM context_kv WHERE namespace=? ORDER BY key ASC",
                (self.namespace,),
            ).fetchall()
        return tuple(str(r[0]) for r in rows)

    def items(self) -> Tuple[tuple[str, Any], ...]:
        out: list[tuple[str, Any]] = []
        for k in self.keys():
            out.append((k, self.get(k)))
        return tuple(out)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.items()}

    def clear(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "DELETE FROM context_kv WHERE namespace=?",
                (self.namespace,),
            )
            conn.commit()


__all__ = [
    "ContextStore",
    "SQLiteContextStore",
]
