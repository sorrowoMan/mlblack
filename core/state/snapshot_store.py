from __future__ import annotations

import json
import pickle
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple
from uuid import uuid4


@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: str
    kind: str
    created_at_utc: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class InMemorySnapshotStore:
    """In-memory snapshot backend for runtime payloads."""

    _payloads: Dict[str, Any] = field(default_factory=dict)
    _records: Dict[str, SnapshotRecord] = field(default_factory=dict)

    def write(
        self,
        payload: Any,
        *,
        kind: str = "generic",
        metadata: Mapping[str, Any] | None = None,
        snapshot_id: str | None = None,
    ) -> str:
        sid = str(snapshot_id).strip() if snapshot_id is not None else str(uuid4())
        if not sid:
            raise ValueError("snapshot_id must not be empty")
        if sid in self._payloads:
            raise KeyError(f"snapshot '{sid}' already exists")

        now = datetime.now(timezone.utc).isoformat()
        record = SnapshotRecord(
            snapshot_id=sid,
            kind=str(kind),
            created_at_utc=now,
            metadata=dict(metadata or {}),
        )
        self._payloads[sid] = payload
        self._records[sid] = record
        return sid

    def read(self, snapshot_id: str) -> Any:
        sid = str(snapshot_id)
        if sid not in self._payloads:
            raise KeyError(f"snapshot '{sid}' not found")
        return self._payloads[sid]

    def try_read(self, snapshot_id: str) -> Any | None:
        return self._payloads.get(str(snapshot_id))

    def has(self, snapshot_id: str) -> bool:
        return str(snapshot_id) in self._payloads

    def metadata(self, snapshot_id: str) -> Dict[str, Any]:
        sid = str(snapshot_id)
        rec = self._records.get(sid)
        if rec is None:
            raise KeyError(f"snapshot '{sid}' not found")
        return {
            "snapshot_id": rec.snapshot_id,
            "kind": rec.kind,
            "created_at_utc": rec.created_at_utc,
            "metadata": dict(rec.metadata),
        }

    def count(self) -> int:
        return int(len(self._payloads))

    def keys(self) -> Tuple[str, ...]:
        return tuple(self._payloads.keys())

    def describe(self) -> Tuple[Dict[str, Any], ...]:
        rows: list[Dict[str, Any]] = []
        for sid in self.keys():
            rows.append(self.metadata(sid))
        return tuple(rows)

    def clear(self) -> None:
        self._payloads.clear()
        self._records.clear()


@dataclass
class SQLiteSnapshotStore:
    """SQLite-backed snapshot store for runtime payloads."""

    db_path: str
    namespace: str = "default"
    fallback_unserializable: bool = True

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
        CREATE TABLE IF NOT EXISTS snapshots (
            namespace TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            payload_blob BLOB NOT NULL,
            PRIMARY KEY (namespace, snapshot_id)
        );
        """
        with closing(self._connect()) as conn:
            conn.execute(sql)
            conn.commit()

    @staticmethod
    def _json_dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def _json_loads(text: str) -> Any:
        return json.loads(str(text))

    def _serialize_payload(self, payload: Any) -> tuple[bytes, dict[str, Any]]:
        try:
            blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            return blob, {}
        except Exception:
            if not bool(self.fallback_unserializable):
                raise
            marker = {
                "__mlblack_unserializable__": True,
                "type": str(type(payload)),
                "repr": repr(payload),
            }
            blob = pickle.dumps(marker, protocol=pickle.HIGHEST_PROTOCOL)
            return blob, {"unserializable_payload": True}

    @staticmethod
    def _deserialize_payload(blob: bytes) -> Any:
        return pickle.loads(blob)

    def write(
        self,
        payload: Any,
        *,
        kind: str = "generic",
        metadata: Mapping[str, Any] | None = None,
        snapshot_id: str | None = None,
    ) -> str:
        sid = str(snapshot_id).strip() if snapshot_id is not None else str(uuid4())
        if not sid:
            raise ValueError("snapshot_id must not be empty")
        if self.has(sid):
            raise KeyError(f"snapshot '{sid}' already exists")

        now = datetime.now(timezone.utc).isoformat()
        blob, extra = self._serialize_payload(payload)
        raw_meta = dict(metadata or {})
        raw_meta.update(extra)
        metadata_json = self._json_dumps(raw_meta)

        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO snapshots(namespace, snapshot_id, kind, created_at_utc, metadata_json, payload_blob)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (self.namespace, sid, str(kind), now, metadata_json, blob),
            )
            conn.commit()
        return sid

    def read(self, snapshot_id: str) -> Any:
        sid = str(snapshot_id)
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT payload_blob FROM snapshots WHERE namespace=? AND snapshot_id=?",
                (self.namespace, sid),
            ).fetchone()
        if row is None:
            raise KeyError(f"snapshot '{sid}' not found")
        return self._deserialize_payload(bytes(row[0]))

    def try_read(self, snapshot_id: str) -> Any | None:
        sid = str(snapshot_id)
        if not self.has(sid):
            return None
        return self.read(sid)

    def has(self, snapshot_id: str) -> bool:
        sid = str(snapshot_id)
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT 1 FROM snapshots WHERE namespace=? AND snapshot_id=? LIMIT 1",
                (self.namespace, sid),
            ).fetchone()
        return bool(row is not None)

    def metadata(self, snapshot_id: str) -> Dict[str, Any]:
        sid = str(snapshot_id)
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT snapshot_id, kind, created_at_utc, metadata_json
                FROM snapshots
                WHERE namespace=? AND snapshot_id=?
                """,
                (self.namespace, sid),
            ).fetchone()
        if row is None:
            raise KeyError(f"snapshot '{sid}' not found")
        return {
            "snapshot_id": str(row[0]),
            "kind": str(row[1]),
            "created_at_utc": str(row[2]),
            "metadata": dict(self._json_loads(str(row[3]))),
        }

    def count(self) -> int:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COUNT(1) FROM snapshots WHERE namespace=?",
                (self.namespace,),
            ).fetchone()
        return int(row[0] if row else 0)

    def keys(self) -> Tuple[str, ...]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT snapshot_id
                FROM snapshots
                WHERE namespace=?
                ORDER BY created_at_utc ASC
                """,
                (self.namespace,),
            ).fetchall()
        return tuple(str(r[0]) for r in rows)

    def describe(self) -> Tuple[Dict[str, Any], ...]:
        rows: list[Dict[str, Any]] = []
        for sid in self.keys():
            rows.append(self.metadata(sid))
        return tuple(rows)

    def clear(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "DELETE FROM snapshots WHERE namespace=?",
                (self.namespace,),
            )
            conn.commit()


__all__ = [
    "SnapshotRecord",
    "InMemorySnapshotStore",
    "SQLiteSnapshotStore",
]
