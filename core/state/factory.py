from __future__ import annotations

from typing import Any, Mapping

from .context_store import ContextStore, SQLiteContextStore
from .snapshot_store import InMemorySnapshotStore, SQLiteSnapshotStore


def _normalize_backend_name(backend: str | None) -> str:
    return str(backend or "memory").strip().lower()


def create_context_store(*, backend: str = "memory", **kwargs: Any) -> Any:
    name = _normalize_backend_name(backend)
    if name in {"memory", "in_memory", "in-memory"}:
        return ContextStore()
    if name in {"sqlite", "sqlite3"}:
        db_path = str(kwargs.pop("db_path", "")).strip()
        if not db_path:
            raise ValueError("sqlite context backend requires db_path")
        namespace = str(kwargs.pop("namespace", "default"))
        if kwargs:
            _ = dict(kwargs)
        return SQLiteContextStore(db_path=db_path, namespace=namespace)
    raise ValueError(f"Unsupported context backend '{backend}'")


def create_snapshot_store(*, backend: str = "memory", **kwargs: Any) -> Any:
    name = _normalize_backend_name(backend)
    if name in {"memory", "in_memory", "in-memory"}:
        return InMemorySnapshotStore()
    if name in {"sqlite", "sqlite3"}:
        db_path = str(kwargs.pop("db_path", "")).strip()
        if not db_path:
            raise ValueError("sqlite snapshot backend requires db_path")
        namespace = str(kwargs.pop("namespace", "default"))
        fallback_unserializable = bool(kwargs.pop("fallback_unserializable", True))
        if kwargs:
            _ = dict(kwargs)
        return SQLiteSnapshotStore(
            db_path=db_path,
            namespace=namespace,
            fallback_unserializable=fallback_unserializable,
        )
    raise ValueError(f"Unsupported snapshot backend '{backend}'")


def create_state_pair(
    *,
    context: Mapping[str, Any] | None = None,
    snapshot: Mapping[str, Any] | None = None,
) -> tuple[Any, Any]:
    context_cfg = dict(context or {})
    snapshot_cfg = dict(snapshot or {})

    context_backend = str(context_cfg.pop("backend", "memory"))
    snapshot_backend = str(snapshot_cfg.pop("backend", "memory"))

    context_store = create_context_store(backend=context_backend, **context_cfg)
    snapshot_store = create_snapshot_store(backend=snapshot_backend, **snapshot_cfg)
    return context_store, snapshot_store


__all__ = [
    "create_context_store",
    "create_snapshot_store",
    "create_state_pair",
]

