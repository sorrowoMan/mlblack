from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlparse

from ..relations import build_relation_payload_index, relation_fields, relation_search_text
from ..registry import Catalog, CatalogEntry, get_catalog

try:  # optional catalog backend
    from psycopg import connect as _pg_connect
    from psycopg.rows import dict_row as _pg_dict_row
except Exception:  # pragma: no cover - optional dependency
    _pg_connect = None
    _pg_dict_row = None

_SCALAR_FIELDS: tuple[tuple[str, str], ...] = (
    ("base", "key"),
    ("base", "title"),
    ("base", "kind"),
    ("base", "import_path"),
    ("base", "module"),
    ("base", "symbol"),
    ("base", "summary"),
    ("base", "tags"),
    ("contract", "context_requires"),
    ("contract", "context_provides"),
    ("contract", "context_mutates"),
    ("contract", "context_cache"),
    ("contract", "requires_metrics"),
    ("contract", "unknown_context_keys"),
    ("contract", "unknown_metric_keys"),
    ("metadata", "architecture_path"),
    ("metadata", "catalog_source"),
    ("metadata", "module"),
    ("metadata", "symbol"),
)

DEFAULT_CATALOG_DB_PATH = Path(__file__).resolve().parents[2] / ".mlblack" / "catalog.sqlite"


@dataclass(frozen=True)
class CatalogDbSummary:
    profile: str
    total: int
    by_kind: Mapping[str, int]
    built_at_utc: str = ""
    db_path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "total": int(self.total),
            "by_kind": dict(self.by_kind),
            "built_at_utc": self.built_at_utc,
            "db_path": self.db_path,
        }


class SQLiteCatalogStore:
    """SQLite implementation of the mlblack catalog DB surface.

    The query side is intentionally DB-only: if a requested profile/database is
    missing, callers get an exception instead of silently reading the registry.
    """

    backend = "sqlite"

    def __init__(self, db_path: str | Path | None = None, *, readonly: bool = True) -> None:
        self.path = resolve_catalog_db_path(db_path)
        self.readonly = bool(readonly)

    def sync_catalog(self, catalog: Catalog, *, profile: str = "default") -> dict[str, Any]:
        entries = tuple(catalog.list())
        relation_payloads = build_relation_payload_index(entries)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect(readonly=False)) as conn:
            with conn:
                self._ensure_schema(conn)
                built_at = _utc_now_iso()
                summary = _summary_payload(entries, profile=profile)
                schema = _schema_payload(entries, profile=profile)
                conn.execute("DELETE FROM catalog_profiles WHERE profile = ?", (str(profile),))
                conn.execute("DELETE FROM catalog_entries WHERE profile = ?", (str(profile),))
                conn.execute("DELETE FROM catalog_scalars WHERE profile = ?", (str(profile),))
                conn.execute(
                    """
INSERT INTO catalog_profiles (profile, built_at_utc, total, summary_json, schema_json)
VALUES (?, ?, ?, ?, ?)
""",
                    (str(profile), built_at, len(entries), _json_dumps(summary), _json_dumps(schema)),
                )
                for entry in entries:
                    relations = relation_payloads.get(entry.key, {})
                    fields = _entry_fields(entry, relations=relations)
                    conn.execute(
                        """
INSERT INTO catalog_entries
(profile, key, kind, title, import_path, summary, tags_json, contract_json, metadata_json, fields_json, relations_json, search_text, built_at_utc)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
                        (
                            str(profile),
                            entry.key,
                            entry.kind,
                            entry.title,
                            entry.import_path,
                            entry.summary,
                            _json_dumps(list(entry.tags)),
                            _json_dumps(dict(entry.contract)),
                            _json_dumps(dict(entry.metadata)),
                            _json_dumps(fields),
                            _json_dumps(relations),
                            _entry_search_text(entry, relations=relations),
                            built_at,
                        ),
                    )
                    conn.executemany(
                        """
INSERT INTO catalog_scalars (profile, entry_key, scope, field_name, scalar_value)
VALUES (?, ?, ?, ?, ?)
""",
                        [
                            (str(profile), entry.key, row["scope"], row["field_name"], row["scalar_value"])
                            for row in _entry_scalar_rows(entry, relations=relations)
                        ],
                    )
        return {
            "backend": self.backend,
            "profile": str(profile),
            "db_path": str(self.path),
            "entries": len(entries),
            "scalars": sum(len(_entry_scalar_rows(entry, relations=relation_payloads.get(entry.key, {}))) for entry in entries),
            "relations": len(relation_payloads),
        }

    def has_profile(self, *, profile: str = "default") -> bool:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT 1 FROM catalog_profiles WHERE profile = ? LIMIT 1", (str(profile),)).fetchone()
            return bool(row)

    def list_catalog_entries(
        self,
        *,
        profile: str = "default",
        kind: str | None = None,
        tags: Sequence[str] | None = None,
        limit: int | None = None,
        field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    ) -> list[CatalogEntry]:
        return self._query_entries(
            profile=profile,
            kind=kind,
            tags=tags,
            limit=limit,
            field_filters=field_filters,
        )

    def search_catalog_entries(
        self,
        query: str,
        *,
        profile: str = "default",
        kind: str | None = None,
        tags: Sequence[str] | None = None,
        limit: int = 20,
        field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    ) -> list[CatalogEntry]:
        return self._query_entries(
            profile=profile,
            kind=kind,
            tags=tags,
            query=query,
            limit=limit,
            field_filters=field_filters,
        )

    def get_catalog_entry(self, key: str, *, profile: str = "default") -> CatalogEntry | None:
        entries = self._query_entries(profile=profile, field_filters={"key": str(key)}, limit=1)
        return entries[0] if entries else None

    def get_catalog_entry_relations(self, key: str, *, profile: str = "default") -> dict[str, Any]:
        with closing(self._connect()) as conn:
            self._require_profile(conn, profile=str(profile))
            row = conn.execute(
                "SELECT relations_json FROM catalog_entries WHERE profile = ? AND key = ?",
                (str(profile), str(key)),
            ).fetchone()
        return _json_mapping(row["relations_json"]) if row else {}

    def catalog_summary(self, *, profile: str = "default") -> CatalogDbSummary:
        with closing(self._connect()) as conn:
            self._require_profile(conn, profile=str(profile))
            row = conn.execute(
                "SELECT built_at_utc, total, summary_json FROM catalog_profiles WHERE profile = ?",
                (str(profile),),
            ).fetchone()
            payload = _json_loads(row["summary_json"] if row else "{}")
            by_kind = dict(payload.get("by_kind", {})) if isinstance(payload, Mapping) else {}
            return CatalogDbSummary(
                profile=str(profile),
                total=int(row["total"] if row else 0),
                by_kind=by_kind,
                built_at_utc=str(row["built_at_utc"] if row else ""),
                db_path=str(self.path),
            )

    def load_catalog(self, *, profile: str = "default") -> Catalog:
        return Catalog(self.list_catalog_entries(profile=profile, limit=None))

    def field_values(
        self,
        field_name: str,
        *,
        profile: str = "default",
        kind: str | None = None,
        limit: int = 100,
    ) -> tuple[dict[str, Any], ...]:
        field = str(field_name or "").strip()
        if not field:
            return tuple()
        entries = self.list_catalog_entries(profile=profile, kind=kind, limit=None)
        counter: Counter[str] = Counter()
        for entry in entries:
            for value in _entry_field_values(entry, field):
                counter[value] += 1
        rows = sorted(counter.items(), key=lambda item: (-item[1], item[0].lower()))
        return tuple({"value": value, "count": count} for value, count in rows[: max(0, int(limit))])

    def _query_entries(
        self,
        *,
        profile: str,
        kind: str | None = None,
        tags: Sequence[str] | None = None,
        query: str = "",
        limit: int | None = None,
        field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    ) -> list[CatalogEntry]:
        with closing(self._connect()) as conn:
            self._require_profile(conn, profile=str(profile))
            clauses = ["profile = ?"]
            params: list[Any] = [str(profile)]
            normalized_kind = str(kind or "").strip().lower()
            if normalized_kind:
                clauses.append("kind = ?")
                params.append(normalized_kind)
            for token in _query_tokens(query):
                clauses.append("LOWER(search_text) LIKE ?")
                params.append(f"%{token}%")
            sql = (
                "SELECT key, kind, title, import_path, summary, tags_json, contract_json, metadata_json "
                "FROM catalog_entries WHERE " + " AND ".join(clauses) + " ORDER BY kind ASC, key ASC"
            )
            rows = conn.execute(sql, tuple(params)).fetchall()
        entries = [_row_to_entry(row) for row in rows]
        entries = [entry for entry in entries if entry is not None]
        tag_filter = {str(tag).strip().lower() for tag in (tags or ()) if str(tag).strip()}
        if tag_filter:
            entries = [
                entry
                for entry in entries
                if tag_filter.issubset({str(tag).strip().lower() for tag in entry.tags})
            ]
        filters = _normalize_field_filters(field_filters)
        if filters:
            entries = [entry for entry in entries if _matches_field_filters(entry, filters)]
        if limit is not None:
            entries = entries[: max(0, int(limit))]
        return entries

    def _connect(self, *, readonly: bool | None = None) -> sqlite3.Connection:
        ro = self.readonly if readonly is None else bool(readonly)
        if ro:
            if not self.path.exists():
                raise FileNotFoundError(f"catalog DB not found: {self.path}")
            uri_path = quote(self.path.resolve().as_posix(), safe="/:")
            conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
        else:
            conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
CREATE TABLE IF NOT EXISTS catalog_profiles (
  profile TEXT PRIMARY KEY,
  built_at_utc TEXT NOT NULL,
  total INTEGER NOT NULL,
  summary_json TEXT NOT NULL,
  schema_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS catalog_entries (
  profile TEXT NOT NULL,
  key TEXT NOT NULL,
  kind TEXT NOT NULL,
  title TEXT NOT NULL,
  import_path TEXT NOT NULL,
  summary TEXT NOT NULL,
  tags_json TEXT NOT NULL,
  contract_json TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  fields_json TEXT NOT NULL,
  relations_json TEXT NOT NULL DEFAULT '{}',
  search_text TEXT NOT NULL,
  built_at_utc TEXT NOT NULL,
  PRIMARY KEY (profile, key)
);
CREATE INDEX IF NOT EXISTS idx_catalog_entries_kind ON catalog_entries(profile, kind);
CREATE TABLE IF NOT EXISTS catalog_scalars (
  profile TEXT NOT NULL,
  entry_key TEXT NOT NULL,
  scope TEXT NOT NULL,
  field_name TEXT NOT NULL,
  scalar_value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_catalog_scalars_profile ON catalog_scalars(profile);
CREATE INDEX IF NOT EXISTS idx_catalog_scalars_entry ON catalog_scalars(profile, entry_key);
CREATE INDEX IF NOT EXISTS idx_catalog_scalars_field ON catalog_scalars(profile, field_name);
"""
        )
        columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(catalog_entries)").fetchall()}
        if "relations_json" not in columns:
            conn.execute("ALTER TABLE catalog_entries ADD COLUMN relations_json TEXT NOT NULL DEFAULT '{}'")

    def _require_profile(self, conn: sqlite3.Connection, *, profile: str) -> None:
        row = conn.execute("SELECT 1 FROM catalog_profiles WHERE profile = ? LIMIT 1", (str(profile),)).fetchone()
        if not row:
            raise RuntimeError(f"catalog DB profile not materialized: {profile}")


class PostgresCatalogStore:
    """PostgreSQL implementation of the same DB-only catalog store surface."""

    backend = "postgresql"

    def __init__(self, db_url: str | None = None, *, readonly: bool = True) -> None:
        self.url = _resolve_postgres_url(db_url)
        self.readonly = bool(readonly)
        if _pg_connect is None:
            raise RuntimeError("PostgreSQL catalog backend requires optional dependency: psycopg.")

    def sync_catalog(self, catalog: Catalog, *, profile: str = "default") -> dict[str, Any]:
        if self.readonly:
            raise RuntimeError("PostgreSQL catalog store is readonly.")
        entries = tuple(catalog.list())
        relation_payloads = build_relation_payload_index(entries)
        with closing(self._connect()) as conn:
            with conn:
                self._ensure_schema(conn)
                built_at = _utc_now_iso()
                summary = _summary_payload(entries, profile=profile)
                schema = _schema_payload(entries, profile=profile)
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM catalog_profiles WHERE profile = %s", (str(profile),))
                    cur.execute("DELETE FROM catalog_entries WHERE profile = %s", (str(profile),))
                    cur.execute("DELETE FROM catalog_scalars WHERE profile = %s", (str(profile),))
                    cur.execute(
                        """
INSERT INTO catalog_profiles (profile, built_at_utc, total, summary_json, schema_json)
VALUES (%s, %s, %s, %s, %s)
""",
                        (str(profile), built_at, len(entries), _json_dumps(summary), _json_dumps(schema)),
                    )
                    for entry in entries:
                        relations = relation_payloads.get(entry.key, {})
                        fields = _entry_fields(entry, relations=relations)
                        cur.execute(
                            """
INSERT INTO catalog_entries
(profile, key, kind, title, import_path, summary, tags_json, contract_json, metadata_json, fields_json, relations_json, search_text, built_at_utc)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
""",
                            (
                                str(profile),
                                entry.key,
                                entry.kind,
                                entry.title,
                                entry.import_path,
                                entry.summary,
                                _json_dumps(list(entry.tags)),
                                _json_dumps(dict(entry.contract)),
                                _json_dumps(dict(entry.metadata)),
                                _json_dumps(fields),
                                _json_dumps(relations),
                                _entry_search_text(entry, relations=relations),
                                built_at,
                            ),
                        )
                        scalar_rows = [
                            (str(profile), entry.key, row["scope"], row["field_name"], row["scalar_value"])
                            for row in _entry_scalar_rows(entry, relations=relations)
                        ]
                        if scalar_rows:
                            cur.executemany(
                                """
INSERT INTO catalog_scalars (profile, entry_key, scope, field_name, scalar_value)
VALUES (%s, %s, %s, %s, %s)
""",
                                scalar_rows,
                            )
        return {
            "backend": self.backend,
            "profile": str(profile),
            "db_path": self.url,
            "entries": len(entries),
            "scalars": sum(len(_entry_scalar_rows(entry, relations=relation_payloads.get(entry.key, {}))) for entry in entries),
            "relations": len(relation_payloads),
        }

    def has_profile(self, *, profile: str = "default") -> bool:
        with closing(self._connect()) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM catalog_profiles WHERE profile = %s LIMIT 1", (str(profile),))
                return bool(cur.fetchone())

    def list_catalog_entries(
        self,
        *,
        profile: str = "default",
        kind: str | None = None,
        tags: Sequence[str] | None = None,
        limit: int | None = None,
        field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    ) -> list[CatalogEntry]:
        return self._query_entries(
            profile=profile,
            kind=kind,
            tags=tags,
            limit=limit,
            field_filters=field_filters,
        )

    def search_catalog_entries(
        self,
        query: str,
        *,
        profile: str = "default",
        kind: str | None = None,
        tags: Sequence[str] | None = None,
        limit: int = 20,
        field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    ) -> list[CatalogEntry]:
        return self._query_entries(
            profile=profile,
            kind=kind,
            tags=tags,
            query=query,
            limit=limit,
            field_filters=field_filters,
        )

    def get_catalog_entry(self, key: str, *, profile: str = "default") -> CatalogEntry | None:
        entries = self._query_entries(profile=profile, field_filters={"key": str(key)}, limit=1)
        return entries[0] if entries else None

    def get_catalog_entry_relations(self, key: str, *, profile: str = "default") -> dict[str, Any]:
        with closing(self._connect()) as conn:
            with conn.cursor() as cur:
                self._require_profile(conn, profile=str(profile))
                cur.execute(
                    "SELECT relations_json FROM catalog_entries WHERE profile = %s AND key = %s",
                    (str(profile), str(key)),
                )
                row = cur.fetchone()
        value = row.get("relations_json") if isinstance(row, Mapping) else (row[0] if row else "")
        return _json_mapping(value) if row else {}

    def catalog_summary(self, *, profile: str = "default") -> CatalogDbSummary:
        with closing(self._connect()) as conn:
            self._require_profile(conn, profile=str(profile))
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT built_at_utc, total, summary_json FROM catalog_profiles WHERE profile = %s",
                    (str(profile),),
                )
                row = cur.fetchone()
        payload = _json_loads(row["summary_json"] if row else "{}")
        by_kind = dict(payload.get("by_kind", {})) if isinstance(payload, Mapping) else {}
        return CatalogDbSummary(
            profile=str(profile),
            total=int(row["total"] if row else 0),
            by_kind=by_kind,
            built_at_utc=str(row["built_at_utc"] if row else ""),
            db_path=self.url,
        )

    def load_catalog(self, *, profile: str = "default") -> Catalog:
        return Catalog(self.list_catalog_entries(profile=profile, limit=None))

    def field_values(
        self,
        field_name: str,
        *,
        profile: str = "default",
        kind: str | None = None,
        limit: int = 100,
    ) -> tuple[dict[str, Any], ...]:
        field = str(field_name or "").strip()
        if not field:
            return tuple()
        entries = self.list_catalog_entries(profile=profile, kind=kind, limit=None)
        counter: Counter[str] = Counter()
        for entry in entries:
            for value in _entry_field_values(entry, field):
                counter[value] += 1
        rows = sorted(counter.items(), key=lambda item: (-item[1], item[0].lower()))
        return tuple({"value": value, "count": count} for value, count in rows[: max(0, int(limit))])

    def _query_entries(
        self,
        *,
        profile: str,
        kind: str | None = None,
        tags: Sequence[str] | None = None,
        query: str = "",
        limit: int | None = None,
        field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    ) -> list[CatalogEntry]:
        with closing(self._connect()) as conn:
            self._require_profile(conn, profile=str(profile))
            clauses = ["profile = %s"]
            params: list[Any] = [str(profile)]
            normalized_kind = str(kind or "").strip().lower()
            if normalized_kind:
                clauses.append("kind = %s")
                params.append(normalized_kind)
            for token in _query_tokens(query):
                clauses.append("LOWER(search_text) LIKE %s")
                params.append(f"%{token}%")
            sql = (
                "SELECT key, kind, title, import_path, summary, tags_json, contract_json, metadata_json "
                "FROM catalog_entries WHERE " + " AND ".join(clauses) + " ORDER BY kind ASC, key ASC"
            )
            if limit is not None:
                sql += f" LIMIT {max(0, int(limit))}"
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall() or []
        entries = [_mapping_to_entry(row) for row in rows]
        entries = [entry for entry in entries if entry is not None]
        tag_filter = {str(tag).strip().lower() for tag in (tags or ()) if str(tag).strip()}
        if tag_filter:
            entries = [
                entry
                for entry in entries
                if tag_filter.issubset({str(tag).strip().lower() for tag in entry.tags})
            ]
        filters = _normalize_field_filters(field_filters)
        if filters:
            entries = [entry for entry in entries if _matches_field_filters(entry, filters)]
        return entries

    def _connect(self):
        return _pg_connect(self.url, row_factory=_pg_dict_row)

    def _ensure_schema(self, conn: Any) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
CREATE TABLE IF NOT EXISTS catalog_profiles (
  profile TEXT PRIMARY KEY,
  built_at_utc TEXT NOT NULL,
  total INTEGER NOT NULL,
  summary_json TEXT NOT NULL,
  schema_json TEXT NOT NULL
)
"""
            )
            cur.execute(
                """
CREATE TABLE IF NOT EXISTS catalog_entries (
  profile TEXT NOT NULL,
  key TEXT NOT NULL,
  kind TEXT NOT NULL,
  title TEXT NOT NULL,
  import_path TEXT NOT NULL,
  summary TEXT NOT NULL,
  tags_json TEXT NOT NULL,
  contract_json TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  fields_json TEXT NOT NULL,
  relations_json TEXT NOT NULL DEFAULT '{}',
  search_text TEXT NOT NULL,
  built_at_utc TEXT NOT NULL,
  PRIMARY KEY (profile, key)
)
"""
            )
            cur.execute("ALTER TABLE catalog_entries ADD COLUMN IF NOT EXISTS relations_json TEXT NOT NULL DEFAULT '{}'")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mlblack_catalog_entries_kind ON catalog_entries(profile, kind)")
            cur.execute(
                """
CREATE TABLE IF NOT EXISTS catalog_scalars (
  profile TEXT NOT NULL,
  entry_key TEXT NOT NULL,
  scope TEXT NOT NULL,
  field_name TEXT NOT NULL,
  scalar_value TEXT NOT NULL
)
"""
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mlblack_catalog_scalars_profile ON catalog_scalars(profile)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mlblack_catalog_scalars_entry ON catalog_scalars(profile, entry_key)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mlblack_catalog_scalars_field ON catalog_scalars(profile, field_name)")

    def _require_profile(self, conn: Any, *, profile: str) -> None:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM catalog_profiles WHERE profile = %s LIMIT 1", (str(profile),))
            row = cur.fetchone()
        if not row:
            raise RuntimeError(f"catalog DB profile not materialized: {profile}")


def resolve_catalog_store(db_path: str | Path | None = None, *, readonly: bool = True) -> SQLiteCatalogStore | PostgresCatalogStore:
    target = _resolve_store_target(db_path)
    if _is_postgres_target(target):
        return PostgresCatalogStore(target, readonly=readonly)
    return SQLiteCatalogStore(target or None, readonly=readonly)


def materialize_catalog_db(
    db_path: str | Path | None = None,
    *,
    catalog: Catalog | None = None,
    profile: str = "default",
    refresh: bool = False,
) -> dict[str, Any]:
    cat = catalog or get_catalog(refresh=refresh)
    store = resolve_catalog_store(db_path, readonly=False)
    return store.sync_catalog(cat, profile=profile)


def load_catalog_db(db_path: str | Path | None = None, *, profile: str = "default") -> Catalog:
    return resolve_catalog_store(db_path, readonly=True).load_catalog(profile=profile)


def catalog_db_summary(db_path: str | Path | None = None, *, profile: str = "default") -> dict[str, Any]:
    return resolve_catalog_store(db_path, readonly=True).catalog_summary(profile=profile).as_dict()


def _resolve_store_target(db_path: str | Path | None = None) -> str:
    import tomllib
    explicit = str(db_path or "").strip()
    if explicit:
        return explicit
    env_url = os.environ.get("MLBLACK_CATALOG_DB_URL", "").strip()
    if env_url:
        return env_url
    env_path = os.environ.get("MLBLACK_CATALOG_DB_PATH", "").strip()
    if env_path:
        return env_path
    # check config file for postgresql
    config_path = Path(__file__).resolve().parent.parent / "db.toml"
    if config_path.exists():
        try:
            cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
            pg = cfg.get("postgresql") or cfg.get("postgres")
            if isinstance(pg, dict) and pg.get("enabled"):
                return (
                    f"postgresql://{pg.get('user','postgres')}:{pg.get('password','')}"
                    f"@{pg.get('host','localhost')}:{pg.get('port',5432)}/"
                    f"{pg.get('database','mlblack_catalog')}"
                )
        except Exception:
            pass
    return str(DEFAULT_CATALOG_DB_PATH)


def _is_postgres_target(value: str) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return False
    try:
        scheme = urlparse(raw).scheme.strip().lower()
    except Exception:
        return False
    return scheme in {"postgres", "postgresql", "postgresql+psycopg", "postgresql+psycopg2"}


def _resolve_postgres_url(db_url: str | None = None) -> str:
    target = str(db_url or os.environ.get("MLBLACK_CATALOG_DB_URL", "")).strip()
    if not target:
        raise RuntimeError("PostgreSQL catalog URL missing. Set MLBLACK_CATALOG_DB_URL or pass db_path.")
    parsed = urlparse(target)
    if parsed.scheme == "postgres":
        return "postgresql://" + target.split("://", 1)[1]
    if parsed.scheme in {"postgresql+psycopg", "postgresql+psycopg2"}:
        return "postgresql://" + target.split("://", 1)[1]
    if parsed.scheme != "postgresql":
        raise ValueError(f"Unsupported PostgreSQL catalog URL scheme: {parsed.scheme}")
    return target


def resolve_catalog_db_path(db_path: str | Path | None = None) -> Path:
    explicit = str(db_path or "").strip()
    if explicit:
        return _path_from_url_or_path(explicit)
    env_url = os.environ.get("MLBLACK_CATALOG_DB_URL", "").strip()
    if env_url:
        return _path_from_url_or_path(env_url)
    env_path = os.environ.get("MLBLACK_CATALOG_DB_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_CATALOG_DB_PATH


def _path_from_url_or_path(value: str) -> Path:
    if len(value) >= 3 and value[1:3] in {":\\", ":/"}:
        return Path(value).expanduser()
    parsed = urlparse(value)
    if parsed.scheme in {"", "file"}:
        if parsed.scheme == "file" and len(parsed.path) >= 3 and parsed.path[0] == "/" and parsed.path[2] == ":":
            return Path(parsed.path[1:]).expanduser()
        return Path(parsed.path if parsed.scheme == "file" else value).expanduser()
    if parsed.scheme == "sqlite":
        raw = value.removeprefix("sqlite:///")
        return Path(raw).expanduser()
    raise ValueError(f"Unsupported mlblack catalog DB URL scheme: {parsed.scheme}")


def _mapping_to_entry(row: Mapping[str, Any]) -> CatalogEntry | None:
    key = str(row.get("key", "") or "").strip()
    kind = str(row.get("kind", "") or "").strip().lower()
    import_path = str(row.get("import_path", "") or "").strip()
    if not key or not kind or not import_path:
        return None
    return CatalogEntry(
        key=key,
        title=str(row.get("title", "") or "").strip(),
        kind=kind,
        import_path=import_path,
        tags=_json_string_tuple(row.get("tags_json", "")),
        summary=str(row.get("summary", "") or "").strip(),
        contract=_json_mapping(row.get("contract_json", "")),
        metadata=_json_mapping(row.get("metadata_json", "")),
    )


def _row_to_entry(row: sqlite3.Row) -> CatalogEntry | None:
    key = str(row["key"] or "").strip()
    kind = str(row["kind"] or "").strip().lower()
    import_path = str(row["import_path"] or "").strip()
    if not key or not kind or not import_path:
        return None
    return CatalogEntry(
        key=key,
        title=str(row["title"] or "").strip(),
        kind=kind,
        import_path=import_path,
        tags=_json_string_tuple(row["tags_json"]),
        summary=str(row["summary"] or "").strip(),
        contract=_json_mapping(row["contract_json"]),
        metadata=_json_mapping(row["metadata_json"]),
    )


def _entry_fields(entry: CatalogEntry, *, relations: Mapping[str, Any] | None = None) -> dict[str, Any]:
    module, _, symbol = entry.import_path.partition(":")
    return {
        "key": entry.key,
        "title": entry.title,
        "kind": entry.kind,
        "import_path": entry.import_path,
        "module": module,
        "symbol": symbol,
        "tags": list(entry.tags),
        "summary": entry.summary,
        "contract": dict(entry.contract),
        "metadata": dict(entry.metadata),
        "relations": dict(relations or {}),
    }


def _entry_scalar_rows(entry: CatalogEntry, *, relations: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for scope, field_name in _SCALAR_FIELDS:
        for value in _entry_field_values(entry, field_name):
            token = (scope, field_name, value.lower())
            if token in seen:
                continue
            seen.add(token)
            rows.append({"scope": scope, "field_name": field_name, "scalar_value": value})
    for key, value in entry.contract.items():
        for scalar in _flatten_strings(value):
            token = ("contract", str(key), scalar.lower())
            if token not in seen:
                seen.add(token)
                rows.append({"scope": "contract", "field_name": str(key), "scalar_value": scalar})
    for key, value in entry.metadata.items():
        for scalar in _flatten_strings(value):
            token = ("metadata", str(key), scalar.lower())
            if token not in seen:
                seen.add(token)
                rows.append({"scope": "metadata", "field_name": str(key), "scalar_value": scalar})
    for key, value in dict(relations or {}).items():
        for scalar in _flatten_strings(value):
            token = ("relations", str(key), scalar.lower())
            if token not in seen:
                seen.add(token)
                rows.append({"scope": "relations", "field_name": str(key), "scalar_value": scalar})
    return rows


def _entry_field_values(entry: CatalogEntry, field_name: str) -> tuple[str, ...]:
    field = str(field_name or "").strip()
    module, _, symbol = entry.import_path.partition(":")
    if field == "key":
        return (entry.key,)
    if field in {"title", "name"}:
        return (entry.title,)
    if field == "kind":
        return (entry.kind,)
    if field == "import_path":
        return (entry.import_path,)
    if field == "module":
        return (module,) if module else tuple()
    if field == "symbol":
        return (symbol,) if symbol else tuple()
    if field == "summary":
        return (entry.summary,) if entry.summary else tuple()
    if field == "tags":
        return tuple(str(tag) for tag in entry.tags)
    relation_value = relation_fields(entry).get(field)
    if relation_value:
        return tuple(relation_value)
    if field in entry.contract:
        return _flatten_strings(entry.contract[field])
    if field in entry.metadata:
        return _flatten_strings(entry.metadata[field])
    return tuple()


def _matches_field_filters(entry: CatalogEntry, filters: Mapping[str, tuple[str, ...]]) -> bool:
    for field_name, expected_values in filters.items():
        current = {value.lower() for value in _entry_field_values(entry, field_name)}
        expected = {str(value).strip().lower() for value in expected_values}
        if not current or current.isdisjoint(expected):
            return False
    return True


def _normalize_field_filters(
    field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None,
) -> dict[str, tuple[str, ...]]:
    if not field_filters:
        return {}
    items = field_filters.items() if isinstance(field_filters, Mapping) else field_filters
    out: dict[str, tuple[str, ...]] = {}
    for field_name, value in items:
        values = _flatten_strings(value)
        if values:
            out[str(field_name)] = values
    return out


def _entry_search_text(entry: CatalogEntry, *, relations: Mapping[str, Any] | None = None) -> str:
    tokens = [
        entry.key,
        entry.title,
        entry.kind,
        entry.import_path,
        entry.summary,
        *[str(tag) for tag in entry.tags],
        *_flatten_strings(entry.contract),
        *_flatten_strings(entry.metadata),
        relation_search_text(relations or {}),
    ]
    return " ".join(str(token).strip().lower() for token in tokens if str(token).strip())


def _query_tokens(query: str) -> tuple[str, ...]:
    return tuple(token for token in str(query or "").strip().lower().split() if token)


def _summary_payload(entries: Sequence[CatalogEntry], *, profile: str) -> dict[str, Any]:
    by_kind = Counter(entry.kind for entry in entries)
    return {
        "profile": str(profile),
        "total": len(entries),
        "by_kind": dict(sorted(by_kind.items())),
        "tags": sorted({str(tag) for entry in entries for tag in entry.tags}),
    }


def _schema_payload(entries: Sequence[CatalogEntry], *, profile: str) -> dict[str, Any]:
    kinds = tuple(sorted({entry.kind for entry in entries}))
    return {
        "profile": str(profile),
        "kinds": kinds,
        "fields": (
            "key",
            "title",
            "kind",
            "import_path",
            "tags",
            "summary",
            "contract",
            "metadata",
            "relations",
            "context_requires",
            "context_provides",
            "context_mutates",
            "requires_metrics",
            "artifact_requires",
            "artifact_provides",
            "phase_in",
            "phase_out",
        ),
        "field_groups": {
            "base": ("key", "title", "kind", "import_path", "tags", "summary"),
            "contract": tuple(sorted({key for entry in entries for key in entry.contract.keys()})),
            "metadata": tuple(sorted({key for entry in entries for key in entry.metadata.keys()})),
            "relations": ("neighbors", "field_refs", "flow", "usage", "relation_cards"),
        },
    }


def _flatten_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else tuple()
    if isinstance(value, Mapping):
        out: list[str] = []
        for key, item in value.items():
            key_text = str(key).strip()
            if key_text:
                out.append(key_text)
            out.extend(_flatten_strings(item))
        return tuple(out)
    if isinstance(value, (list, tuple, set, frozenset)):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_strings(item))
        return tuple(out)
    text = str(value).strip()
    return (text,) if text else tuple()


def _json_string_tuple(value: Any) -> tuple[str, ...]:
    loaded = _json_loads(value)
    if isinstance(loaded, list):
        return tuple(str(item).strip() for item in loaded if str(item).strip())
    return tuple()


def _json_mapping(value: Any) -> dict[str, Any]:
    loaded = _json_loads(value)
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _json_loads(value: Any) -> Any:
    try:
        return json.loads(str(value or ""))
    except Exception:
        return {}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "CatalogDbSummary",
    "DEFAULT_CATALOG_DB_PATH",
    "PostgresCatalogStore",
    "SQLiteCatalogStore",
    "catalog_db_summary",
    "load_catalog_db",
    "materialize_catalog_db",
    "resolve_catalog_db_path",
    "resolve_catalog_store",
]
