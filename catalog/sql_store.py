from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote_plus

from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    create_engine,
    delete,
    exists,
    func,
    select,
)
from sqlalchemy.engine import Connection, Engine, Row
from sqlalchemy.engine.url import URL, make_url
from sqlalchemy.exc import NoSuchModuleError

try:  # py>=3.11
    import tomllib as _toml
except Exception:  # pragma: no cover
    try:  # py<3.11
        import tomli as _toml  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        _toml = None

from .registry import (
    CatalogEntry,
    _BASE_ENTRY_FIELDS,
    _FIELD_ALIASES,
    _UI_FACET_FIELDS_BY_KIND,
    _normalize_field_filters,
    _normalize_field_name,
    _normalize_kind,
    _normalize_profile,
    catalog_schema,
    catalog_summary,
    list_entries,
)

_METADATA = MetaData()

_CATALOG_PROFILES = Table(
    "catalog_profiles",
    _METADATA,
    Column("profile", String(255), primary_key=True),
    Column("built_at_utc", String(64), nullable=False),
    Column("total", Integer, nullable=False),
    Column("summary_json", Text, nullable=False),
    Column("schema_json", Text, nullable=False),
)

_CATALOG_ENTRIES = Table(
    "catalog_entries",
    _METADATA,
    Column("profile", String(255), primary_key=True),
    Column("key", String(255), primary_key=True),
    Column("kind", String(64), nullable=False, index=True),
    Column("name", String(255), nullable=False),
    Column("source", String(255), nullable=False),
    Column("path", Text, nullable=True),
    Column("summary", Text, nullable=False),
    Column("tags_json", Text, nullable=False),
    Column("metadata_json", Text, nullable=False),
    Column("fields_json", Text, nullable=False),
    Column("relations_json", Text, nullable=False),
    Column("search_text", Text, nullable=False),
    Column("built_at_utc", String(64), nullable=False),
)

_CATALOG_SCALARS = Table(
    "catalog_scalars",
    _METADATA,
    Column("profile", String(255), nullable=False, index=True),
    Column("entry_key", String(255), nullable=False, index=True),
    Column("scope", String(32), nullable=False),
    Column("field_name", String(255), nullable=False, index=True),
    # Keep scalar_value unindexed so the schema stays portable across
    # sqlite/postgresql/mysql. MySQL rejects TEXT indexes without a prefix
    # length, and catalog scale is small enough that field_name/profile
    # indexes are sufficient for current query patterns.
    Column("scalar_value", Text, nullable=False),
)

_CATALOG_RELATION_EDGES = Table(
    "catalog_relation_edges",
    _METADATA,
    Column("profile", String(255), nullable=False, index=True),
    Column("source_key", String(255), nullable=False, index=True),
    Column("source_kind", String(64), nullable=False, index=True),
    Column("relation_name", String(255), nullable=False, index=True),
    Column("target_key", String(255), nullable=False, index=True),
    Column("target_kind", String(64), nullable=True, index=True),
    Column("target_name", String(255), nullable=True),
    Column("target_missing", Integer, nullable=False, default=0),
)

_CATALOG_RELATION_KEYS = Table(
    "catalog_relation_keys",
    _METADATA,
    Column("profile", String(255), nullable=False, index=True),
    Column("source_kind", String(64), nullable=False, index=True),
    Column("relation_name", String(255), nullable=False, index=True),
    Column("relation_value", String(255), nullable=False, index=True),
    Column("edge_count", Integer, nullable=False),
    Column("source_count", Integer, nullable=False),
    Column("target_count", Integer, nullable=False),
    Column("source_keys_json", Text, nullable=False),
    Column("source_kinds_json", Text, nullable=False),
    Column("target_keys_json", Text, nullable=False),
    Column("target_kinds_json", Text, nullable=False),
)


@dataclass(frozen=True)
class CatalogDbTarget:
    raw_target: str
    engine_url: str
    safe_label: str
    backend: str
    driver: str
    filesystem_path: str | None


@dataclass(frozen=True)
class CatalogDbResolvedConfig:
    target: str
    readonly: bool
    mode: str
    source: str
    config_path: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_loads(text: str) -> Any:
    return json.loads(str(text))


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _normalize_mode(raw: str | None) -> str:
    key = str(raw or "").strip().lower()
    if key in {"only", "prefer", "off"}:
        return key
    if key == "disabled":
        return "off"
    return "prefer"


def _read_catalog_db_config_file() -> tuple[dict[str, Any], str | None]:
    if _toml is None:
        return {}, None
    env_path = os.environ.get("MLBLACK_CATALOG_DB_CONFIG", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "catalog" / "db.toml")
    candidates.append(Path(__file__).resolve().parent / "db.toml")
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        try:
            data = _toml.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if isinstance(data, dict):
            return dict(data), str(path.resolve())
    return {}, None


def _catalog_db_block(data: Mapping[str, Any]) -> dict[str, Any]:
    block = data.get("catalog_db")
    if isinstance(block, dict):
        return dict(block)
    catalog_block = data.get("catalog")
    if isinstance(catalog_block, dict):
        nested = catalog_block.get("db")
        if isinstance(nested, dict):
            return dict(nested)
    return {}


def _build_db_target_from_block(block: Mapping[str, Any]) -> str:
    raw_url = str(block.get("url", block.get("db_url", "")) or "").strip()
    if raw_url:
        return raw_url

    backend = str(block.get("backend", "sqlite") or "sqlite").strip().lower()
    if backend in {"sqlite", "sqlite3"}:
        raw_path = str(block.get("path", block.get("db_path", block.get("database", ""))) or "").strip()
        if not raw_path:
            raise ValueError("catalog_db config requires 'url' or sqlite 'path'/'db_path'")
        return raw_path

    host = str(block.get("host", "127.0.0.1") or "127.0.0.1").strip() or "127.0.0.1"
    database = str(block.get("database", block.get("db", "")) or "").strip()
    user = str(block.get("user", block.get("username", "")) or "").strip()
    password = str(block.get("password", "") or "").strip()
    if not database:
        raise ValueError("catalog_db config requires 'database' when using a network SQL backend without full 'url'")

    if backend in {"postgres", "postgresql"}:
        driver = str(block.get("driver", "postgresql+psycopg") or "postgresql+psycopg").strip()
        port = int(block.get("port", 5432))
    elif backend == "mysql":
        driver = str(block.get("driver", "mysql+pymysql") or "mysql+pymysql").strip()
        port = int(block.get("port", 3306))
    else:
        raise ValueError(f"unsupported catalog_db backend '{backend}'")

    auth = ""
    if user:
        auth = quote_plus(user)
        if password:
            auth += f":{quote_plus(password)}"
        auth += "@"
    return f"{driver}://{auth}{host}:{port}/{database}"


def _flatten_scalars(value: Any) -> Iterable[str]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip().lower()
        return (text,) if text else ()
    if isinstance(value, (bool, int, float)):
        return (str(value).strip().lower(),)
    if isinstance(value, Path):
        return (value.as_posix().strip().lower(),)
    if isinstance(value, Mapping):
        out: list[str] = []
        for key, inner in value.items():
            out.extend(_flatten_scalars(key))
            out.extend(_flatten_scalars(inner))
        return tuple(out)
    if isinstance(value, (tuple, list, set, frozenset)):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_scalars(item))
        return tuple(out)
    text = str(value).strip().lower()
    return (text,) if text else ()


def _field_aliases(name: str) -> tuple[str, ...]:
    key = _normalize_field_name(name)
    aliases = _FIELD_ALIASES.get(key)
    if aliases is not None:
        return aliases
    if key.endswith("s") and key[:-1]:
        return (key, key[:-1])
    return (key,)


def _entry_search_text(entry: CatalogEntry) -> str:
    chunks = [entry.key, entry.kind, entry.name, entry.source, entry.summary]
    if entry.path:
        chunks.append(entry.path)
    chunks.extend(str(tag) for tag in entry.tags)
    chunks.extend(_flatten_scalars(entry.metadata))
    chunks.extend(_flatten_scalars(entry.fields))
    chunks.extend(_flatten_scalars(entry.relations))
    return " ".join(str(chunk) for chunk in chunks if str(chunk).strip()).lower()


def _entry_scalar_rows(entry: CatalogEntry) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    base_fields = {
        "id": entry.key,
        "key": entry.key,
        "kind": entry.kind,
        "name": entry.name,
        "source": entry.source,
        "path": entry.path,
        "tags": entry.tags,
        "summary": entry.summary,
    }
    for field_name, value in base_fields.items():
        for scalar in _flatten_scalars(value):
            rows.append({"scope": "base", "field_name": str(field_name), "scalar_value": scalar})

    for field_name, value in dict(entry.fields).items():
        for scalar in _flatten_scalars(value):
            rows.append({"scope": "field", "field_name": str(field_name), "scalar_value": scalar})

    for relation_name, value in dict(entry.relations).items():
        for scalar in _flatten_scalars(value):
            rows.append({"scope": "relation", "field_name": str(relation_name), "scalar_value": scalar})

    return rows


def _entry_relation_edge_rows(
    entry: CatalogEntry,
    *,
    profile: str,
    entry_by_key: Mapping[str, CatalogEntry],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relation_name, relation_value in dict(entry.relations).items():
        rel_name = str(relation_name).strip()
        if not rel_name:
            continue
        seen: set[str] = set()
        for target_value in _flatten_scalars(relation_value):
            target_key = str(target_value).strip().lower()
            if not target_key or target_key in seen:
                continue
            seen.add(target_key)
            target_entry = entry_by_key.get(target_key)
            rows.append(
                {
                    "profile": str(profile),
                    "source_key": str(entry.key),
                    "source_kind": str(entry.kind),
                    "relation_name": rel_name,
                    "target_key": target_key,
                    "target_kind": None if target_entry is None else str(target_entry.kind),
                    "target_name": None if target_entry is None else str(target_entry.name),
                    "target_missing": 1 if target_entry is None else 0,
                }
            )
    return rows


def _relation_key_rows_from_edges(edges: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for edge in edges:
        profile = str(edge.get("profile", "") or "").strip()
        source_kind = str(edge.get("source_kind", "") or "").strip()
        relation_name = str(edge.get("relation_name", "") or "").strip()
        relation_value = str(edge.get("target_key", "") or edge.get("relation_value", "") or "").strip()
        if not profile or not source_kind or not relation_name or not relation_value:
            continue
        token = (profile, source_kind, relation_name, relation_value)
        row = grouped.setdefault(
            token,
            {
                "profile": profile,
                "source_kind": source_kind,
                "relation_name": relation_name,
                "relation_value": relation_value,
                "edge_count": 0,
                "source_keys": set(),
                "source_kinds": set(),
                "target_keys": set(),
                "target_kinds": set(),
            },
        )
        row["edge_count"] = int(row.get("edge_count", 0) or 0) + 1
        source_key = str(edge.get("source_key", "") or "").strip()
        target_key = str(edge.get("target_key", "") or "").strip()
        target_kind = str(edge.get("target_kind", "") or "").strip()
        if source_key:
            row["source_keys"].add(source_key)
        if source_kind:
            row["source_kinds"].add(source_kind)
        if target_key:
            row["target_keys"].add(target_key)
        if target_kind:
            row["target_kinds"].add(target_kind)

    rows: list[dict[str, Any]] = []
    for profile, source_kind, relation_name, relation_value in sorted(
        grouped.keys(),
        key=lambda item: (item[0], item[1], item[2], item[3]),
    ):
        row = grouped[(profile, source_kind, relation_name, relation_value)]
        source_keys = tuple(sorted(str(item) for item in row.get("source_keys", set())))
        source_kinds = tuple(sorted(str(item) for item in row.get("source_kinds", set())))
        target_keys = tuple(sorted(str(item) for item in row.get("target_keys", set())))
        target_kinds = tuple(sorted(str(item) for item in row.get("target_kinds", set())))
        rows.append(
            {
                "profile": profile,
                "source_kind": source_kind,
                "relation_name": relation_name,
                "relation_value": relation_value,
                "edge_count": int(row.get("edge_count", 0) or 0),
                "source_count": len(source_keys),
                "target_count": len(target_keys),
                "source_keys_json": _json_dumps(list(source_keys)),
                "source_kinds_json": _json_dumps(list(source_kinds)),
                "target_keys_json": _json_dumps(list(target_keys)),
                "target_kinds_json": _json_dumps(list(target_kinds)),
            }
        )
    return rows


def _row_to_entry(row: Row[Any] | Mapping[str, Any]) -> CatalogEntry:
    payload = row._mapping if isinstance(row, Row) else row
    return CatalogEntry(
        key=str(payload["key"]),
        kind=str(payload["kind"]),
        name=str(payload["name"]),
        source=str(payload["source"]),
        path=None if payload["path"] is None else str(payload["path"]),
        tags=tuple(_json_loads(str(payload["tags_json"]))),
        summary=str(payload["summary"]),
        metadata=dict(_json_loads(str(payload["metadata_json"]))),
        fields=dict(_json_loads(str(payload["fields_json"]))),
        relations=dict(_json_loads(str(payload["relations_json"]))),
    )


def _normalize_sqlite_url(url: URL) -> CatalogDbTarget:
    database = str(url.database or "").strip()
    if not database or database == ":memory:":
        normalized = url if url.drivername else url.set(drivername="sqlite+pysqlite")
        drivername = normalized.drivername if "+" in normalized.drivername else "sqlite+pysqlite"
        normalized = normalized.set(drivername=drivername)
        return CatalogDbTarget(
            raw_target=normalized.render_as_string(hide_password=False),
            engine_url=normalized.render_as_string(hide_password=False),
            safe_label=normalized.render_as_string(hide_password=True),
            backend="sqlite",
            driver=drivername,
            filesystem_path=None,
        )

    db_path = Path(database).expanduser()
    if not db_path.is_absolute():
        db_path = db_path.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    drivername = url.drivername if "+" in url.drivername else "sqlite+pysqlite"
    normalized = url.set(drivername=drivername, database=db_path.as_posix())
    return CatalogDbTarget(
        raw_target=normalized.render_as_string(hide_password=False),
        engine_url=normalized.render_as_string(hide_password=False),
        safe_label=normalized.render_as_string(hide_password=True),
        backend="sqlite",
        driver=drivername,
        filesystem_path=str(db_path),
    )


def _normalize_db_target(db_path: str) -> CatalogDbTarget:
    raw = str(db_path or "").strip()
    if not raw:
        raise ValueError("catalog db target must be a non-empty sqlite path or SQLAlchemy URL")

    if "://" not in raw and not raw.lower().startswith("sqlite:"):
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        engine_url = f"sqlite+pysqlite:///{path.as_posix()}"
        return CatalogDbTarget(
            raw_target=str(path),
            engine_url=engine_url,
            safe_label=engine_url,
            backend="sqlite",
            driver="sqlite+pysqlite",
            filesystem_path=str(path),
        )

    try:
        url = make_url(raw)
    except Exception as exc:
        raise ValueError(f"invalid catalog db target: {raw}") from exc

    drivername = str(url.drivername or "").strip().lower()
    if drivername == "postgres":
        url = url.set(drivername="postgresql+psycopg")
    elif drivername == "postgresql":
        url = url.set(drivername="postgresql+psycopg")
    elif drivername == "mysql":
        url = url.set(drivername="mysql+pymysql")
    elif drivername == "sqlite":
        return _normalize_sqlite_url(url)

    backend = str(url.get_backend_name()).strip().lower()
    if backend == "sqlite":
        return _normalize_sqlite_url(url)
    if backend not in {"postgresql", "mysql"}:
        raise ValueError(
            "catalog db target must be a sqlite path or a SQLAlchemy URL for sqlite/postgresql/mysql"
        )

    normalized_url = url.render_as_string(hide_password=False)
    safe_label = url.render_as_string(hide_password=True)
    return CatalogDbTarget(
        raw_target=raw,
        engine_url=normalized_url,
        safe_label=safe_label,
        backend=backend,
        driver=str(url.drivername),
        filesystem_path=None,
    )


def catalog_db_config_mode() -> str:
    env_mode = os.environ.get("MLBLACK_CATALOG_DB_MODE", "").strip()
    if env_mode:
        return _normalize_mode(env_mode)
    data, _ = _read_catalog_db_config_file()
    block = _catalog_db_block(data)
    return _normalize_mode(block.get("mode"))


def catalog_db_config_enabled() -> bool:
    env_url = os.environ.get("MLBLACK_CATALOG_DB_URL", "").strip()
    if env_url:
        return True
    data, _ = _read_catalog_db_config_file()
    block = _catalog_db_block(data)
    return bool(block.get("enabled", False))


def catalog_db_resolved_config() -> CatalogDbResolvedConfig | None:
    env_url = os.environ.get("MLBLACK_CATALOG_DB_URL", "").strip()
    if env_url:
        return CatalogDbResolvedConfig(
            target=env_url,
            readonly=_truthy_env("MLBLACK_CATALOG_DB_READONLY"),
            mode=catalog_db_config_mode(),
            source="env",
            config_path=None,
        )

    data, config_path = _read_catalog_db_config_file()
    block = _catalog_db_block(data)
    if not bool(block.get("enabled", False)):
        return None

    target = _build_db_target_from_block(block)
    return CatalogDbResolvedConfig(
        target=target,
        readonly=bool(block.get("readonly", False)),
        mode=catalog_db_config_mode(),
        source="file",
        config_path=config_path,
    )


def catalog_db_config_info() -> dict[str, Any]:
    resolved = catalog_db_resolved_config()
    payload = {
        "enabled": bool(resolved is not None),
        "mode": catalog_db_config_mode(),
    }
    if resolved is None:
        payload.update(
            {
                "readonly": False,
                "source": None,
                "config_path": None,
                "db_target": None,
                "db_path": None,
                "db_backend": None,
                "db_driver": None,
                "filesystem_path": None,
                "is_file_backed": False,
            }
        )
        return payload

    target_info = catalog_db_target_info(resolved.target)
    return {
        **payload,
        "readonly": bool(resolved.readonly),
        "source": str(resolved.source),
        "config_path": resolved.config_path,
        **target_info,
    }


def catalog_db_target_info(db_path: str) -> dict[str, Any]:
    target = _normalize_db_target(db_path)
    return {
        "db_path": target.filesystem_path or target.safe_label,
        "db_target": target.safe_label,
        "db_backend": target.backend,
        "db_driver": target.driver,
        "filesystem_path": target.filesystem_path,
        "is_file_backed": bool(target.filesystem_path),
    }


def _missing_driver_hint(target: CatalogDbTarget) -> str:
    if target.backend == "postgresql":
        return "Install `psycopg[binary]` or another SQLAlchemy PostgreSQL driver."
    if target.backend == "mysql":
        return "Install `pymysql` or another SQLAlchemy MySQL driver."
    return "Install the required SQLAlchemy driver for this backend."


def _create_engine_for_target(target: CatalogDbTarget) -> Engine:
    try:
        return create_engine(target.engine_url)
    except (ImportError, ModuleNotFoundError, NoSuchModuleError) as exc:
        raise ImportError(
            f"catalog db backend '{target.backend}' is unavailable for {target.safe_label}. {_missing_driver_hint(target)}"
        ) from exc


def _open_engine(db_path: str) -> tuple[CatalogDbTarget, Engine]:
    target = _normalize_db_target(db_path)
    return target, _create_engine_for_target(target)


def _ensure_schema(conn: Connection) -> None:
    _METADATA.create_all(conn)


def _empty_schema_payload(profile: str, kind: str | None) -> dict[str, Any]:
    if kind is not None:
        return {
            "profile": str(profile),
            "kind": str(kind),
            "base_fields": list(_BASE_ENTRY_FIELDS),
            "fields": [],
            "relations": [],
            "count": 0,
        }
    return {
        "profile": str(profile),
        "base_fields": list(_BASE_ENTRY_FIELDS),
        "kinds": {},
    }


def _schema_for_kind(stored_schema: Mapping[str, Any], profile: str, kind: str | None) -> dict[str, Any]:
    if kind is None:
        return dict(stored_schema)
    kind_key = _normalize_kind(kind) or ""
    bucket = dict(dict(stored_schema).get("kinds", {})).get(kind_key, {})
    return {
        "profile": str(profile),
        "kind": kind_key,
        "base_fields": list(_BASE_ENTRY_FIELDS),
        "fields": list(bucket.get("fields", [])),
        "relations": list(bucket.get("relations", [])),
        "count": int(bucket.get("count", 0)),
    }


def _profile_row(conn: Connection, profile: str) -> Mapping[str, Any] | None:
    row = conn.execute(
        select(
            _CATALOG_PROFILES.c.profile,
            _CATALOG_PROFILES.c.built_at_utc,
            _CATALOG_PROFILES.c.total,
            _CATALOG_PROFILES.c.summary_json,
            _CATALOG_PROFILES.c.schema_json,
        ).where(_CATALOG_PROFILES.c.profile == str(profile))
    ).mappings().first()
    return None if row is None else dict(row)


def _entry_select_stmt():
    return select(
        _CATALOG_ENTRIES.c.profile,
        _CATALOG_ENTRIES.c.key,
        _CATALOG_ENTRIES.c.kind,
        _CATALOG_ENTRIES.c.name,
        _CATALOG_ENTRIES.c.source,
        _CATALOG_ENTRIES.c.path,
        _CATALOG_ENTRIES.c.summary,
        _CATALOG_ENTRIES.c.tags_json,
        _CATALOG_ENTRIES.c.metadata_json,
        _CATALOG_ENTRIES.c.fields_json,
        _CATALOG_ENTRIES.c.relations_json,
    )


def _entry_row_by_lower_key(conn: Connection, profile: str, target_key: str) -> Mapping[str, Any] | None:
    row = conn.execute(
        _entry_select_stmt().where(
            and_(
                _CATALOG_ENTRIES.c.profile == str(profile),
                func.lower(_CATALOG_ENTRIES.c.key) == str(target_key),
            )
        )
    ).mappings().first()
    return None if row is None else dict(row)


def _entry_rows_by_lower_keys(
    conn: Connection,
    profile: str,
    target_keys: Sequence[str],
) -> dict[str, CatalogEntry]:
    lowered = tuple(str(value).strip().lower() for value in target_keys if str(value).strip())
    if not lowered:
        return {}
    rows = conn.execute(
        _entry_select_stmt().where(
            and_(
                _CATALOG_ENTRIES.c.profile == str(profile),
                func.lower(_CATALOG_ENTRIES.c.key).in_(lowered),
            )
        )
    ).mappings().all()
    return {
        str(row["key"]).strip().lower(): _row_to_entry(dict(row))
        for row in rows
    }


def _build_entry_query(
    *,
    profile: str,
    kind: str | None,
    field_filters: Sequence[tuple[str, str]],
    query: str | None = None,
    limit: int | None = None,
):
    stmt = select(
        _CATALOG_ENTRIES.c.profile,
        _CATALOG_ENTRIES.c.key,
        _CATALOG_ENTRIES.c.kind,
        _CATALOG_ENTRIES.c.name,
        _CATALOG_ENTRIES.c.source,
        _CATALOG_ENTRIES.c.path,
        _CATALOG_ENTRIES.c.summary,
        _CATALOG_ENTRIES.c.tags_json,
        _CATALOG_ENTRIES.c.metadata_json,
        _CATALOG_ENTRIES.c.fields_json,
        _CATALOG_ENTRIES.c.relations_json,
    ).where(_CATALOG_ENTRIES.c.profile == str(profile))

    if kind is not None:
        stmt = stmt.where(_CATALOG_ENTRIES.c.kind == str(kind))

    for field_name, expected_value in field_filters:
        aliases = tuple(_field_aliases(field_name))
        exists_stmt = (
            select(_CATALOG_SCALARS.c.entry_key)
            .where(
                and_(
                    _CATALOG_SCALARS.c.profile == _CATALOG_ENTRIES.c.profile,
                    _CATALOG_SCALARS.c.entry_key == _CATALOG_ENTRIES.c.key,
                    _CATALOG_SCALARS.c.field_name.in_(aliases),
                    _CATALOG_SCALARS.c.scalar_value == str(expected_value),
                )
            )
            .limit(1)
        )
        stmt = stmt.where(exists(exists_stmt))

    normalized_query = str(query or "").strip().lower()
    if normalized_query:
        stmt = stmt.where(_CATALOG_ENTRIES.c.search_text.contains(normalized_query))

    stmt = stmt.order_by(_CATALOG_ENTRIES.c.kind.asc(), _CATALOG_ENTRIES.c.key.asc())
    if limit is not None:
        stmt = stmt.limit(max(0, int(limit)))
    return stmt


def materialize_catalog_db(
    db_path: str,
    *,
    profile: str = "default",
) -> dict[str, Any]:
    profile_key = _normalize_profile(profile)
    target, engine = _open_engine(db_path)
    built_at = _utc_now_iso()
    items = list_entries(profile=profile_key, limit=None)
    summary = catalog_summary(profile=profile_key)
    schema = catalog_schema(profile=profile_key)

    try:
        with engine.begin() as conn:
            _ensure_schema(conn)
            conn.execute(delete(_CATALOG_PROFILES).where(_CATALOG_PROFILES.c.profile == str(profile_key)))
            conn.execute(delete(_CATALOG_ENTRIES).where(_CATALOG_ENTRIES.c.profile == str(profile_key)))
            conn.execute(delete(_CATALOG_SCALARS).where(_CATALOG_SCALARS.c.profile == str(profile_key)))
            conn.execute(delete(_CATALOG_RELATION_EDGES).where(_CATALOG_RELATION_EDGES.c.profile == str(profile_key)))
            conn.execute(delete(_CATALOG_RELATION_KEYS).where(_CATALOG_RELATION_KEYS.c.profile == str(profile_key)))

            conn.execute(
                _CATALOG_PROFILES.insert(),
                [
                    {
                        "profile": str(profile_key),
                        "built_at_utc": built_at,
                        "total": int(len(items)),
                        "summary_json": _json_dumps(summary),
                        "schema_json": _json_dumps(schema),
                    }
                ],
            )

            entry_rows = []
            scalar_rows = []
            relation_edge_rows = []
            relation_key_rows = []
            entry_by_key = {str(entry.key).strip().lower(): entry for entry in items}
            for entry in items:
                entry_rows.append(
                    {
                        "profile": str(profile_key),
                        "key": entry.key,
                        "kind": entry.kind,
                        "name": entry.name,
                        "source": entry.source,
                        "path": entry.path,
                        "summary": entry.summary,
                        "tags_json": _json_dumps(tuple(entry.tags)),
                        "metadata_json": _json_dumps(dict(entry.metadata)),
                        "fields_json": _json_dumps(dict(entry.fields)),
                        "relations_json": _json_dumps(dict(entry.relations)),
                        "search_text": _entry_search_text(entry),
                        "built_at_utc": built_at,
                    }
                )
                for scalar_row in _entry_scalar_rows(entry):
                    scalar_rows.append(
                        {
                            "profile": str(profile_key),
                            "entry_key": entry.key,
                            **scalar_row,
                        }
                    )
                relation_edge_rows.extend(
                    _entry_relation_edge_rows(
                        entry,
                        profile=str(profile_key),
                        entry_by_key=entry_by_key,
                    )
                )
            relation_key_rows = _relation_key_rows_from_edges(relation_edge_rows)

            if entry_rows:
                conn.execute(_CATALOG_ENTRIES.insert(), entry_rows)
            if scalar_rows:
                conn.execute(_CATALOG_SCALARS.insert(), scalar_rows)
            if relation_edge_rows:
                conn.execute(_CATALOG_RELATION_EDGES.insert(), relation_edge_rows)
            if relation_key_rows:
                conn.execute(_CATALOG_RELATION_KEYS.insert(), relation_key_rows)
    finally:
        engine.dispose()

    return {
        "db_path": target.filesystem_path or target.safe_label,
        "db_target": target.safe_label,
        "db_backend": target.backend,
        "db_driver": target.driver,
        "filesystem_path": target.filesystem_path,
        "profile": str(profile_key),
        "built_at_utc": built_at,
        "total": int(len(items)),
        "kinds": dict(summary.get("by_kind", {})),
        "relation_edges": int(len(relation_edge_rows)),
        "relation_keys": int(len(relation_key_rows)),
    }


def materialize_catalog_sqlite(
    db_path: str,
    *,
    profile: str = "default",
) -> dict[str, Any]:
    return materialize_catalog_db(db_path, profile=profile)


def catalog_db_summary(
    db_path: str,
    *,
    profile: str | None = None,
) -> dict[str, Any]:
    target, engine = _open_engine(db_path)
    try:
        with engine.connect() as conn:
            _ensure_schema(conn)
            if profile is not None:
                profile_key = _normalize_profile(profile)
                row = _profile_row(conn, profile_key)
                if row is None:
                    return {
                        "db_path": target.filesystem_path or target.safe_label,
                        "db_target": target.safe_label,
                        "db_backend": target.backend,
                        "db_driver": target.driver,
                        "filesystem_path": target.filesystem_path,
                        "profile": str(profile_key),
                        "materialized": False,
                    }
                return {
                    "db_path": target.filesystem_path or target.safe_label,
                    "db_target": target.safe_label,
                    "db_backend": target.backend,
                    "db_driver": target.driver,
                    "filesystem_path": target.filesystem_path,
                    "profile": str(row["profile"]),
                    "materialized": True,
                    "built_at_utc": str(row["built_at_utc"]),
                    "total": int(row["total"]),
                    "summary": dict(_json_loads(str(row["summary_json"]))),
                    "schema": dict(_json_loads(str(row["schema_json"]))),
                }

            rows = conn.execute(
                select(
                    _CATALOG_PROFILES.c.profile,
                    _CATALOG_PROFILES.c.built_at_utc,
                    _CATALOG_PROFILES.c.total,
                    _CATALOG_PROFILES.c.summary_json,
                ).order_by(_CATALOG_PROFILES.c.profile.asc())
            ).mappings().all()
    finally:
        engine.dispose()

    return {
        "db_path": target.filesystem_path or target.safe_label,
        "db_target": target.safe_label,
        "db_backend": target.backend,
        "db_driver": target.driver,
        "filesystem_path": target.filesystem_path,
        "profiles": [
            {
                "profile": str(row["profile"]),
                "built_at_utc": str(row["built_at_utc"]),
                "total": int(row["total"]),
                "summary": dict(_json_loads(str(row["summary_json"]))),
            }
            for row in rows
        ],
    }


def catalog_db_schema(
    db_path: str,
    *,
    profile: str = "default",
    kind: str | None = None,
) -> dict[str, Any]:
    profile_key = _normalize_profile(profile)
    kind_key = _normalize_kind(kind)
    target, engine = _open_engine(db_path)
    try:
        with engine.connect() as conn:
            _ensure_schema(conn)
            row = _profile_row(conn, profile_key)
    finally:
        engine.dispose()
    if row is None:
        return _empty_schema_payload(profile_key, kind_key)
    stored_schema = dict(_json_loads(str(row["schema_json"])))
    return _schema_for_kind(stored_schema, profile_key, kind_key)


def catalog_db_show_entry(
    db_path: str,
    key: str,
    *,
    profile: str = "default",
) -> CatalogEntry | None:
    profile_key = _normalize_profile(profile)
    target_key = str(key).strip().lower()
    if not target_key:
        return None
    _, engine = _open_engine(db_path)
    try:
        with engine.connect() as conn:
            _ensure_schema(conn)
            row = _entry_row_by_lower_key(conn, profile_key, target_key)
    finally:
        engine.dispose()
    if row is None:
        return None
    return _row_to_entry(row)


def catalog_db_list_entries(
    db_path: str,
    *,
    profile: str = "default",
    kind: str | None = None,
    limit: int | None = None,
    field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
) -> tuple[CatalogEntry, ...]:
    profile_key = _normalize_profile(profile)
    kind_key = _normalize_kind(kind)
    filters = _normalize_field_filters(field_filters)
    _, engine = _open_engine(db_path)
    try:
        with engine.connect() as conn:
            _ensure_schema(conn)
            rows = conn.execute(
                _build_entry_query(
                    profile=profile_key,
                    kind=kind_key,
                    field_filters=filters,
                    query=None,
                    limit=limit,
                )
            ).mappings().all()
    finally:
        engine.dispose()
    return tuple(_row_to_entry(dict(row)) for row in rows)


def catalog_db_search_entries(
    db_path: str,
    query: str,
    *,
    profile: str = "default",
    kind: str | None = None,
    limit: int = 20,
    field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
) -> tuple[CatalogEntry, ...]:
    normalized_query = str(query).strip().lower()
    if not normalized_query:
        return catalog_db_list_entries(
            db_path,
            profile=profile,
            kind=kind,
            limit=limit,
            field_filters=field_filters,
        )

    profile_key = _normalize_profile(profile)
    kind_key = _normalize_kind(kind)
    filters = _normalize_field_filters(field_filters)
    _, engine = _open_engine(db_path)
    try:
        with engine.connect() as conn:
            _ensure_schema(conn)
            rows = conn.execute(
                _build_entry_query(
                    profile=profile_key,
                    kind=kind_key,
                    field_filters=filters,
                    query=normalized_query,
                    limit=limit,
                )
            ).mappings().all()
    finally:
        engine.dispose()
    return tuple(_row_to_entry(dict(row)) for row in rows)


def catalog_db_field_values(
    db_path: str,
    field_name: str,
    *,
    profile: str = "default",
    kind: str | None = None,
    limit: int | None = None,
) -> tuple[str, ...]:
    profile_key = _normalize_profile(profile)
    kind_key = _normalize_kind(kind)
    aliases = tuple(_field_aliases(field_name))
    _, engine = _open_engine(db_path)
    try:
        with engine.connect() as conn:
            _ensure_schema(conn)
            stmt = (
                select(_CATALOG_SCALARS.c.scalar_value.label("value"))
                .select_from(
                    _CATALOG_SCALARS.join(
                        _CATALOG_ENTRIES,
                        and_(
                            _CATALOG_ENTRIES.c.profile == _CATALOG_SCALARS.c.profile,
                            _CATALOG_ENTRIES.c.key == _CATALOG_SCALARS.c.entry_key,
                        ),
                    )
                )
                .where(
                    and_(
                        _CATALOG_SCALARS.c.profile == str(profile_key),
                        _CATALOG_SCALARS.c.scope != "relation",
                        _CATALOG_SCALARS.c.field_name.in_(aliases),
                    )
                )
                .distinct()
                .order_by(_CATALOG_SCALARS.c.scalar_value.asc())
            )
            if kind_key is not None:
                stmt = stmt.where(_CATALOG_ENTRIES.c.kind == str(kind_key))
            if limit is not None:
                stmt = stmt.limit(max(0, int(limit)))
            rows = conn.execute(stmt).mappings().all()
    finally:
        engine.dispose()
    return tuple(str(row["value"]) for row in rows)


def catalog_db_relation_edges(
    db_path: str,
    *,
    profile: str = "default",
    kind: str | None = None,
    relation_name: str | None = None,
    target_kind: str | None = None,
    source_key: str | None = None,
    limit: int | None = 200,
) -> tuple[dict[str, Any], ...]:
    profile_key = _normalize_profile(profile)
    kind_key = _normalize_kind(kind)
    relation_key = str(relation_name or "").strip()
    target_kind_key = _normalize_kind(target_kind)
    source_key_value = str(source_key or "").strip().lower()
    _, engine = _open_engine(db_path)
    try:
        with engine.connect() as conn:
            _ensure_schema(conn)
            stmt = select(
                _CATALOG_RELATION_EDGES.c.profile,
                _CATALOG_RELATION_EDGES.c.source_key,
                _CATALOG_RELATION_EDGES.c.source_kind,
                _CATALOG_RELATION_EDGES.c.relation_name,
                _CATALOG_RELATION_EDGES.c.target_key,
                _CATALOG_RELATION_EDGES.c.target_kind,
                _CATALOG_RELATION_EDGES.c.target_name,
                _CATALOG_RELATION_EDGES.c.target_missing,
            ).where(_CATALOG_RELATION_EDGES.c.profile == str(profile_key))
            if kind_key is not None:
                stmt = stmt.where(_CATALOG_RELATION_EDGES.c.source_kind == str(kind_key))
            if relation_key:
                stmt = stmt.where(_CATALOG_RELATION_EDGES.c.relation_name == relation_key)
            if target_kind_key is not None:
                stmt = stmt.where(_CATALOG_RELATION_EDGES.c.target_kind == str(target_kind_key))
            if source_key_value:
                stmt = stmt.where(func.lower(_CATALOG_RELATION_EDGES.c.source_key) == source_key_value)
            stmt = stmt.order_by(
                _CATALOG_RELATION_EDGES.c.relation_name.asc(),
                _CATALOG_RELATION_EDGES.c.source_key.asc(),
                _CATALOG_RELATION_EDGES.c.target_key.asc(),
            )
            if limit is not None:
                stmt = stmt.limit(max(0, int(limit)))
            rows = conn.execute(stmt).mappings().all()
    finally:
        engine.dispose()
    return tuple(
        {
            "profile": str(row["profile"]),
            "source_key": str(row["source_key"]),
            "source_kind": str(row["source_kind"]),
            "relation_name": str(row["relation_name"]),
            "relation_value": str(row["target_key"]),
            "target_key": str(row["target_key"]),
            "target_kind": None if row["target_kind"] is None else str(row["target_kind"]),
            "target_name": None if row["target_name"] is None else str(row["target_name"]),
            "target_missing": bool(int(row["target_missing"] or 0)),
        }
        for row in rows
    )


def _catalog_db_relation_keys_from_edges(
    db_path: str,
    *,
    profile: str,
    kind: str | None,
    relation_name: str | None,
    limit: int | None,
) -> tuple[dict[str, Any], ...]:
    edges = catalog_db_relation_edges(
        db_path,
        profile=profile,
        kind=kind,
        relation_name=relation_name,
        target_kind=None,
        source_key=None,
        limit=None,
    )
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in edges:
        rel_name = str(edge.get("relation_name", "") or "").strip()
        rel_value = str(edge.get("relation_value", "") or "").strip()
        token = (rel_name, rel_value)
        row = grouped.setdefault(
            token,
            {
                "profile": str(profile),
                "relation_name": rel_name,
                "relation_value": rel_value,
                "edge_count": 0,
                "source_keys": set(),
                "source_kinds": set(),
                "target_keys": set(),
                "target_kinds": set(),
            },
        )
        row["edge_count"] = int(row.get("edge_count", 0) or 0) + 1
        source_key_value = str(edge.get("source_key", "") or "").strip()
        source_kind_value = str(edge.get("source_kind", "") or "").strip()
        target_key_value = str(edge.get("target_key", "") or "").strip()
        target_kind_value = str(edge.get("target_kind", "") or "").strip()
        if source_key_value:
            row["source_keys"].add(source_key_value)
        if source_kind_value:
            row["source_kinds"].add(source_kind_value)
        if target_key_value:
            row["target_keys"].add(target_key_value)
        if target_kind_value:
            row["target_kinds"].add(target_kind_value)

    rows: list[dict[str, Any]] = []
    for rel_name, rel_value in sorted(grouped.keys(), key=lambda item: (item[0], item[1])):
        row = grouped[(rel_name, rel_value)]
        source_keys = tuple(sorted(str(item) for item in row.get("source_keys", set())))
        source_kinds = tuple(sorted(str(item) for item in row.get("source_kinds", set())))
        target_keys = tuple(sorted(str(item) for item in row.get("target_keys", set())))
        target_kinds = tuple(sorted(str(item) for item in row.get("target_kinds", set())))
        rows.append(
            {
                "profile": str(profile),
                "relation_name": rel_name,
                "relation_value": rel_value,
                "edge_count": int(row.get("edge_count", 0) or 0),
                "source_count": len(source_keys),
                "target_count": len(target_keys),
                "source_keys": list(source_keys),
                "source_kinds": list(source_kinds),
                "target_keys": list(target_keys),
                "target_kinds": list(target_kinds),
            }
        )
    if limit is None:
        return tuple(rows)
    return tuple(rows[: max(0, int(limit))])


def catalog_db_relation_keys(
    db_path: str,
    *,
    profile: str = "default",
    kind: str | None = None,
    relation_name: str | None = None,
    limit: int | None = 200,
) -> tuple[dict[str, Any], ...]:
    profile_key = _normalize_profile(profile)
    kind_key = _normalize_kind(kind)
    relation_key = str(relation_name or "").strip()
    _, engine = _open_engine(db_path)
    try:
        with engine.connect() as conn:
            _ensure_schema(conn)
            stmt = select(
                _CATALOG_RELATION_KEYS.c.profile,
                _CATALOG_RELATION_KEYS.c.source_kind,
                _CATALOG_RELATION_KEYS.c.relation_name,
                _CATALOG_RELATION_KEYS.c.relation_value,
                _CATALOG_RELATION_KEYS.c.edge_count,
                _CATALOG_RELATION_KEYS.c.source_count,
                _CATALOG_RELATION_KEYS.c.target_count,
                _CATALOG_RELATION_KEYS.c.source_keys_json,
                _CATALOG_RELATION_KEYS.c.source_kinds_json,
                _CATALOG_RELATION_KEYS.c.target_keys_json,
                _CATALOG_RELATION_KEYS.c.target_kinds_json,
            ).where(_CATALOG_RELATION_KEYS.c.profile == str(profile_key))
            if kind_key is not None:
                stmt = stmt.where(_CATALOG_RELATION_KEYS.c.source_kind == str(kind_key))
            if relation_key:
                stmt = stmt.where(_CATALOG_RELATION_KEYS.c.relation_name == relation_key)
            stmt = stmt.order_by(
                _CATALOG_RELATION_KEYS.c.relation_name.asc(),
                _CATALOG_RELATION_KEYS.c.relation_value.asc(),
                _CATALOG_RELATION_KEYS.c.source_kind.asc(),
            )
            stored_rows = conn.execute(stmt).mappings().all()
    finally:
        engine.dispose()

    if not stored_rows:
        return _catalog_db_relation_keys_from_edges(
            db_path,
            profile=profile_key,
            kind=kind_key,
            relation_name=relation_key or None,
            limit=limit,
        )

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in stored_rows:
        rel_name = str(row["relation_name"])
        rel_value = str(row["relation_value"])
        token = (rel_name, rel_value)
        bucket = grouped.setdefault(
            token,
            {
                "profile": str(profile_key),
                "relation_name": rel_name,
                "relation_value": rel_value,
                "edge_count": 0,
                "source_keys": set(),
                "source_kinds": set(),
                "target_keys": set(),
                "target_kinds": set(),
            },
        )
        bucket["edge_count"] = int(bucket.get("edge_count", 0) or 0) + int(row["edge_count"] or 0)
        bucket["source_keys"].update(_json_loads(str(row["source_keys_json"])))
        bucket["source_kinds"].update(_json_loads(str(row["source_kinds_json"])))
        bucket["target_keys"].update(_json_loads(str(row["target_keys_json"])))
        bucket["target_kinds"].update(_json_loads(str(row["target_kinds_json"])))

    rows: list[dict[str, Any]] = []
    for rel_name, rel_value in sorted(grouped.keys(), key=lambda item: (item[0], item[1])):
        bucket = grouped[(rel_name, rel_value)]
        source_keys = tuple(sorted(str(item) for item in bucket.get("source_keys", set())))
        source_kinds = tuple(sorted(str(item) for item in bucket.get("source_kinds", set())))
        target_keys = tuple(sorted(str(item) for item in bucket.get("target_keys", set())))
        target_kinds = tuple(sorted(str(item) for item in bucket.get("target_kinds", set())))
        rows.append(
            {
                "profile": str(profile_key),
                "relation_name": rel_name,
                "relation_value": rel_value,
                "edge_count": int(bucket.get("edge_count", 0) or 0),
                "source_count": len(source_keys),
                "target_count": len(target_keys),
                "source_keys": list(source_keys),
                "source_kinds": list(source_kinds),
                "target_keys": list(target_keys),
                "target_kinds": list(target_kinds),
            }
        )
    if limit is None:
        return tuple(rows)
    return tuple(rows[: max(0, int(limit))])


def catalog_db_neighbors(
    db_path: str,
    key: str,
    *,
    profile: str = "default",
) -> dict[str, Any]:
    profile_key = _normalize_profile(profile)
    target_key = str(key).strip().lower()
    if not target_key:
        return {
            "profile": str(profile_key),
            "key": str(key),
            "entry": None,
            "neighbors": {},
        }

    entry = catalog_db_show_entry(db_path, str(key), profile=profile_key)
    if entry is None:
        return {
            "profile": str(profile_key),
            "key": str(key),
            "entry": None,
            "neighbors": {},
            "relation_keys": [],
        }
    edge_rows = catalog_db_relation_edges(
        db_path,
        profile=profile_key,
        source_key=str(entry.key),
        limit=None,
    )
    neighbor_payload: dict[str, list[dict[str, Any]]] = {}
    relation_keys: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in edge_rows:
        grouped.setdefault(str(row.get("relation_name", "")), []).append(dict(row))
    for relation_name, rows in sorted(grouped.items()):
        output_rows: list[dict[str, Any]] = []
        for row in rows:
            output_rows.append(
                {
                    "key": str(row.get("target_key") or row.get("relation_value") or ""),
                    "kind": row.get("target_kind"),
                    "name": row.get("target_name"),
                    "summary": None,
                    "missing": bool(row.get("target_missing", False)),
                }
            )
        neighbor_payload[str(relation_name)] = output_rows
        relation_keys.append(
            {
                "relation_name": str(relation_name),
                "relation_value_count": len(output_rows),
                "target_keys": [str(item.get("key") or "") for item in output_rows if str(item.get("key") or "").strip()],
                "target_kinds": sorted({str(item.get("kind") or "").strip() for item in output_rows if str(item.get("kind") or "").strip()}),
            }
        )

    return {
        "profile": str(profile_key),
        "key": entry.key,
        "entry": entry.to_dict(),
        "neighbors": neighbor_payload,
        "relation_keys": relation_keys,
    }


def catalog_db_facets(
    db_path: str,
    *,
    profile: str = "default",
    kind: str | None = None,
    query: str | None = None,
    field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    fields: Sequence[str] | None = None,
    limit_per_field: int = 25,
) -> dict[str, Any]:
    profile_key = _normalize_profile(profile)
    kind_key = _normalize_kind(kind)
    filters = _normalize_field_filters(field_filters)
    normalized_query = str(query or "").strip().lower()

    schema = catalog_db_schema(db_path, profile=profile_key, kind=kind_key)
    target_fields = tuple(str(value) for value in (fields or schema.get("fields", [])) if str(value).strip())

    _, engine = _open_engine(db_path)
    try:
        with engine.connect() as conn:
            _ensure_schema(conn)
            selected_rows = conn.execute(
                _build_entry_query(
                    profile=profile_key,
                    kind=kind_key,
                    field_filters=filters,
                    query=normalized_query or None,
                    limit=None,
                )
            ).mappings().all()
            selected_keys = tuple(str(row["key"]) for row in selected_rows)

            facets: dict[str, list[dict[str, Any]]] = {}
            if not selected_keys:
                for field_name in target_fields:
                    facets[str(field_name)] = []
            else:
                for field_name in target_fields:
                    aliases = tuple(_field_aliases(field_name))
                    count_stmt = (
                        select(
                            _CATALOG_SCALARS.c.scalar_value.label("value"),
                            func.count(func.distinct(_CATALOG_SCALARS.c.entry_key)).label("count"),
                        )
                        .where(
                            and_(
                                _CATALOG_SCALARS.c.profile == str(profile_key),
                                _CATALOG_SCALARS.c.scope != "relation",
                                _CATALOG_SCALARS.c.entry_key.in_(selected_keys),
                                _CATALOG_SCALARS.c.field_name.in_(aliases),
                            )
                        )
                        .group_by(_CATALOG_SCALARS.c.scalar_value)
                        .order_by(func.count(func.distinct(_CATALOG_SCALARS.c.entry_key)).desc(), _CATALOG_SCALARS.c.scalar_value.asc())
                        .limit(max(0, int(limit_per_field)))
                    )
                    facet_rows = conn.execute(count_stmt).mappings().all()
                    facets[str(field_name)] = [
                        {"value": str(row["value"]), "count": int(row["count"])}
                        for row in facet_rows
                    ]
    finally:
        engine.dispose()

    return {
        "profile": str(profile_key),
        "kind": kind_key,
        "query": str(query or ""),
        "filters": [{"field": str(name), "value": str(value)} for name, value in filters],
        "total": int(len(selected_keys)),
        "facets": facets,
    }


def catalog_db_ui_snapshot(
    db_path: str,
    *,
    profile: str = "default",
    kind: str | None = None,
    query: str | None = None,
    field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    limit: int = 200,
    selected_key: str | None = None,
) -> dict[str, Any]:
    profile_key = _normalize_profile(profile)
    kind_key = _normalize_kind(kind)
    filters = _normalize_field_filters(field_filters)
    normalized_query = str(query or "").strip()

    if normalized_query:
        items = catalog_db_search_entries(
            db_path,
            normalized_query,
            profile=profile_key,
            kind=kind_key,
            limit=limit,
            field_filters=filters,
        )
    else:
        items = catalog_db_list_entries(
            db_path,
            profile=profile_key,
            kind=kind_key,
            limit=limit,
            field_filters=filters,
        )

    schema = catalog_db_schema(db_path, profile=profile_key, kind=kind_key)
    facet_fields = tuple(_UI_FACET_FIELDS_BY_KIND.get(kind_key or "", ())) or tuple(schema.get("fields", []))
    facets = catalog_db_facets(
        db_path,
        profile=profile_key,
        kind=kind_key,
        query=normalized_query,
        field_filters=filters,
        fields=facet_fields,
        limit_per_field=20,
    )

    summary_wrapper = catalog_db_summary(db_path, profile=profile_key)
    summary_payload = dict(summary_wrapper.get("summary", {})) if bool(summary_wrapper.get("materialized")) else {
        "profile": str(profile_key),
        "total": 0,
        "by_kind": {},
    }

    selected_entry = None
    neighbors = None
    if selected_key:
        selected_entry = catalog_db_show_entry(db_path, str(selected_key), profile=profile_key)
        neighbors = catalog_db_neighbors(db_path, str(selected_key), profile=profile_key)

    return {
        "profile": str(profile_key),
        "kind": kind_key,
        "query": str(query or ""),
        "filters": [{"field": str(name), "value": str(value)} for name, value in filters],
        "summary": summary_payload,
        "schema": schema,
        "facets": facets,
        "items": [entry.to_dict() for entry in items],
        "selected": selected_entry.to_dict() if selected_entry is not None else None,
        "neighbors": neighbors,
    }


__all__ = [
    "CatalogDbTarget",
    "CatalogDbResolvedConfig",
    "catalog_db_config_enabled",
    "catalog_db_config_mode",
    "catalog_db_config_info",
    "catalog_db_resolved_config",
    "catalog_db_target_info",
    "materialize_catalog_db",
    "materialize_catalog_sqlite",
    "catalog_db_summary",
    "catalog_db_schema",
    "catalog_db_show_entry",
    "catalog_db_list_entries",
    "catalog_db_search_entries",
    "catalog_db_field_values",
    "catalog_db_relation_edges",
    "catalog_db_relation_keys",
    "catalog_db_neighbors",
    "catalog_db_facets",
    "catalog_db_ui_snapshot",
]
