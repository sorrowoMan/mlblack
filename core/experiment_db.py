from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence
from urllib.parse import quote_plus, urlparse

try:
    import tomllib as _toml
except Exception:  # pragma: no cover
    try:
        import tomli as _toml  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        _toml = None

try:
    from psycopg import connect as _pg_connect
    from psycopg.rows import dict_row as _pg_dict_row
except Exception:  # pragma: no cover
    _pg_connect = None
    _pg_dict_row = None


_DEFAULT_SQLITE_TARGET = "runs/experiments.sqlite3"
_POSTGRES_SCHEMES = {"postgres", "postgresql", "postgresql+psycopg", "postgresql+psycopg2"}
_SQLITE_SCHEMES = {"sqlite", "sqlite3", "sqlite+pysqlite"}


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_mode(value: str | None) -> str:
    key = str(value or "").strip().lower()
    if key in {"only", "prefer", "off"}:
        return key
    if key == "disabled":
        return "off"
    return "prefer"


def _url_scheme(raw: str | None) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    try:
        return str(urlparse(text).scheme or "").strip().lower()
    except Exception:
        return ""


def _is_postgres_target(raw: str | None) -> bool:
    return _url_scheme(raw) in _POSTGRES_SCHEMES


def _is_sqlite_url(raw: str | None) -> bool:
    return _url_scheme(raw) in _SQLITE_SCHEMES


def _safe_target_label(target: str) -> str:
    text = str(target or "").strip()
    if not text or not _is_postgres_target(text):
        return text
    parsed = urlparse(text)
    host = parsed.hostname or "localhost"
    port = f":{parsed.port}" if parsed.port else ""
    user = quote_plus(parsed.username or "postgres")
    auth = f"{user}:***@" if parsed.username else ""
    return f"{parsed.scheme}://{auth}{host}{port}{parsed.path or ''}"


def _read_toml_file(path: Path) -> dict[str, Any]:
    if _toml is None or not path.exists() or not path.is_file():
        return {}
    try:
        payload = _toml.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _read_experiment_db_config_file() -> tuple[dict[str, Any], str | None]:
    env_path = str(os.environ.get("MLBLACK_EXPERIMENT_DB_CONFIG", "") or "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "experiment" / "db.toml")
    candidates.append(Path(__file__).resolve().parents[1] / "experiment" / "db.toml")
    for path in candidates:
        payload = _read_toml_file(path)
        if payload:
            return payload, str(path.resolve())
    return {}, None


def _experiment_db_block(data: Mapping[str, Any]) -> dict[str, Any]:
    block = data.get("experiment_db")
    if isinstance(block, dict):
        return dict(block)
    experiment_block = data.get("experiment")
    if isinstance(experiment_block, dict):
        nested = experiment_block.get("db")
        if isinstance(nested, dict):
            return dict(nested)
    return {}


def _build_target_from_block(block: Mapping[str, Any]) -> str:
    raw_url = str(block.get("url", block.get("db_url", "")) or "").strip()
    if raw_url:
        return raw_url

    backend = str(block.get("backend", "sqlite") or "sqlite").strip().lower()
    if backend in {"sqlite", "sqlite3"}:
        raw_path = str(block.get("path", block.get("db_path", block.get("database", ""))) or "").strip()
        if not raw_path:
            raise ValueError("experiment_db config requires 'url' or sqlite 'path'/'db_path'")
        return raw_path

    host = str(block.get("host", "127.0.0.1") or "127.0.0.1").strip() or "127.0.0.1"
    database = str(block.get("database", block.get("db", "")) or "").strip()
    user = str(block.get("user", block.get("username", "")) or "").strip()
    password = str(block.get("password", "") or "").strip()
    if not database:
        raise ValueError("experiment_db config requires 'database' for a network backend without full 'url'")

    if backend in {"postgres", "postgresql"}:
        driver = str(block.get("driver", "postgresql+psycopg") or "postgresql+psycopg").strip()
        port = int(block.get("port", 5432))
    else:
        raise ValueError(f"unsupported experiment_db backend '{backend}'")

    auth = ""
    if user:
        auth = quote_plus(user)
        if password:
            auth += f":{quote_plus(password)}"
        auth += "@"
    return f"{driver}://{auth}{host}:{port}/{database}"


@dataclass(frozen=True)
class ExperimentDbResolvedConfig:
    target: str
    source: str
    config_path: str | None
    mode: str
    readonly: bool


def experiment_db_resolved_config() -> ExperimentDbResolvedConfig | None:
    env_target = str(os.environ.get("MLBLACK_EXPERIMENT_DB_URL", "") or "").strip()
    if env_target:
        return ExperimentDbResolvedConfig(
            target=env_target,
            source="env",
            config_path=None,
            mode=_normalize_mode(os.environ.get("MLBLACK_EXPERIMENT_DB_MODE")),
            readonly=_truthy_env("MLBLACK_EXPERIMENT_DB_READONLY"),
        )

    data, config_path = _read_experiment_db_config_file()
    block = _experiment_db_block(data)
    if bool(block.get("enabled", False)):
        return ExperimentDbResolvedConfig(
            target=_build_target_from_block(block),
            source="file",
            config_path=config_path,
            mode=_normalize_mode(os.environ.get("MLBLACK_EXPERIMENT_DB_MODE") or str(block.get("mode", ""))),
            readonly=bool(block.get("readonly", False)) or _truthy_env("MLBLACK_EXPERIMENT_DB_READONLY"),
        )

    try:
        from catalog.sql_store import catalog_db_resolved_config

        catalog_cfg = catalog_db_resolved_config()
    except Exception:
        catalog_cfg = None
    if catalog_cfg is not None and _is_postgres_target(catalog_cfg.target):
        return ExperimentDbResolvedConfig(
            target=str(catalog_cfg.target),
            source="catalog_fallback",
            config_path=str(catalog_cfg.config_path) if catalog_cfg.config_path else None,
            mode=_normalize_mode(os.environ.get("MLBLACK_EXPERIMENT_DB_MODE") or str(catalog_cfg.mode)),
            readonly=bool(catalog_cfg.readonly) or _truthy_env("MLBLACK_EXPERIMENT_DB_READONLY"),
        )

    return None


def resolve_experiment_db_target(explicit_target: str | None = None) -> str:
    explicit = str(explicit_target or "").strip()
    if explicit:
        return explicit
    resolved = experiment_db_resolved_config()
    if resolved is not None:
        return resolved.target
    return _DEFAULT_SQLITE_TARGET


def experiment_db_config_info() -> dict[str, Any]:
    resolved = experiment_db_resolved_config()
    target = resolve_experiment_db_target()
    return {
        "enabled": bool(resolved is not None),
        "mode": "prefer" if resolved is None else resolved.mode,
        "source": None if resolved is None else resolved.source,
        "config_path": None if resolved is None else resolved.config_path,
        "readonly": False if resolved is None else bool(resolved.readonly),
        "db_target": _safe_target_label(target),
        "db_backend": "postgresql" if _is_postgres_target(target) else "sqlite",
        "filesystem_path": None if _is_postgres_target(target) or _is_sqlite_url(target) else str(Path(target).expanduser().resolve()),
        "is_file_backed": not _is_postgres_target(target),
    }


@dataclass(frozen=True)
class ExperimentDbTarget:
    raw_target: str
    backend: str
    safe_label: str
    filesystem_path: str | None


def normalize_experiment_db_target(target: str | None) -> ExperimentDbTarget:
    raw = resolve_experiment_db_target(target)
    if _is_postgres_target(raw):
        return ExperimentDbTarget(raw_target=raw, backend="postgresql", safe_label=_safe_target_label(raw), filesystem_path=None)
    if _is_sqlite_url(raw):
        parsed = urlparse(raw)
        path = str(parsed.path or "").lstrip("/") or _DEFAULT_SQLITE_TARGET
        resolved_path = str(Path(path).expanduser().resolve())
        return ExperimentDbTarget(raw_target=resolved_path, backend="sqlite", safe_label=resolved_path, filesystem_path=resolved_path)
    resolved_path = str(Path(raw).expanduser().resolve())
    return ExperimentDbTarget(raw_target=resolved_path, backend="sqlite", safe_label=resolved_path, filesystem_path=resolved_path)


class ExperimentDbConnection:
    def __init__(self, target: ExperimentDbTarget) -> None:
        self.target = target
        self.backend = target.backend
        if self.backend == "postgresql":
            if _pg_connect is None:
                raise RuntimeError("PostgreSQL driver missing: install psycopg.")
            parsed = urlparse(target.raw_target)
            self._conn = _pg_connect(
                host=parsed.hostname or "127.0.0.1",
                port=int(parsed.port or 5432),
                user=parsed.username or "postgres",
                password=parsed.password or "",
                dbname=(parsed.path or "").lstrip("/") or "postgres",
                row_factory=_pg_dict_row,
            )
        else:
            db_path = str(target.filesystem_path or target.raw_target)
            Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            self._conn = conn

    def _adapt_sql(self, sql: str) -> str:
        if self.backend != "postgresql":
            return sql
        return str(sql).replace("?", "%s")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        return self._conn.execute(self._adapt_sql(sql), tuple(params))

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ExperimentDbConnection":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if exc_type is not None:
                self.rollback()
        finally:
            self.close()


def open_experiment_db(target: str | None = None) -> ExperimentDbConnection:
    return ExperimentDbConnection(normalize_experiment_db_target(target))


def table_columns(conn: ExperimentDbConnection, table_name: str) -> set[str]:
    if conn.backend == "postgresql":
        rows = conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema() AND table_name = %s
            """,
            (str(table_name),),
        ).fetchall()
        return {str((row.get("column_name") if isinstance(row, Mapping) else row[0])) for row in rows}
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def ensure_table_columns(conn: ExperimentDbConnection, table_name: str, columns: Mapping[str, str]) -> None:
    existing = table_columns(conn, table_name)
    for name, sql_type in dict(columns).items():
        if str(name) in existing:
            continue
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {name} {sql_type}")


def table_exists(conn: ExperimentDbConnection, table_name: str) -> bool:
    if conn.backend == "postgresql":
        row = conn.execute("SELECT to_regclass(%s) AS table_ref", (str(table_name),)).fetchone()
        if row is None:
            return False
        if isinstance(row, Mapping):
            return bool(row.get("table_ref"))
        return bool(row[0])
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (str(table_name),),
    ).fetchone()
    return row is not None


def table_count(conn: ExperimentDbConnection, table_name: str) -> int:
    if not table_exists(conn, table_name):
        return 0
    row = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    if row is None:
        return 0
    if isinstance(row, Mapping):
        return int(next(iter(row.values())))
    return int(row[0])


def first_column_texts(rows: Sequence[Any]) -> list[str]:
    out: list[str] = []
    for row in rows:
        if isinstance(row, Mapping):
            value = next(iter(row.values()), None)
        else:
            value = row[0] if row else None
        text = str(value).strip() if value is not None else ""
        if text:
            out.append(text)
    return out


def decode_row(row: Any, *, json_fields: Sequence[str], json_loader: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        payload = {str(key): value for key, value in dict(row).items()}
    else:
        payload = {str(key): row[key] for key in row.keys()}
    for field_name in tuple(json_fields):
        payload[str(field_name)] = json_loader(payload.get(str(field_name)))
    return payload


__all__ = [
    "ExperimentDbConnection",
    "ExperimentDbResolvedConfig",
    "ExperimentDbTarget",
    "decode_row",
    "ensure_table_columns",
    "experiment_db_config_info",
    "experiment_db_resolved_config",
    "first_column_texts",
    "normalize_experiment_db_target",
    "open_experiment_db",
    "resolve_experiment_db_target",
    "table_columns",
    "table_count",
    "table_exists",
]
