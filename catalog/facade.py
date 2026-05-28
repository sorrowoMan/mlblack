from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .query import CatalogQuery, query_catalog
from .relations import build_entry_relation_payload
from .registry import CatalogEntry
from .store import catalog_db_summary, load_catalog_db, resolve_catalog_store


def catalog_summary(*, db_path: str | None = None, profile: str = "default") -> dict[str, Any]:
    return catalog_db_summary(db_path, profile=profile)


def catalog_schema(*, db_path: str | None = None, profile: str = "default") -> dict[str, Any]:
    catalog = load_catalog_db(db_path, profile=profile)
    entries = catalog.list()
    contract_keys = sorted({key for entry in entries for key in dict(entry.contract).keys()})
    metadata_keys = sorted({key for entry in entries for key in dict(entry.metadata).keys()})
    return {
        "profile": profile,
        "kinds": tuple(sorted({entry.kind for entry in entries})),
        "tags": tuple(sorted({str(tag) for entry in entries for tag in entry.tags})),
        "fields": (
            "key",
            "title",
            "kind",
            "import_path",
            "tags",
            "summary",
            "context_requires",
            "context_provides",
            "context_mutates",
            "requires_metrics",
            "artifact_requires",
            "artifact_provides",
            "phase_in",
            "phase_out",
        ),
        "contract_keys": tuple(contract_keys),
        "metadata_keys": tuple(metadata_keys),
    }


def list_entries(
    *,
    db_path: str | None = None,
    profile: str = "default",
    kind: str | None = None,
    tags: Sequence[str] = (),
    limit: int = 100,
) -> tuple[dict[str, Any], ...]:
    result = query_catalog(
        CatalogQuery(kind=kind, tags=tuple(tags), limit=limit, profile=profile, db_path=db_path)
    )
    return tuple(_entry_payload(entry) for entry in result.entries)


def search_entries(
    query: str,
    *,
    db_path: str | None = None,
    profile: str = "default",
    kind: str | None = None,
    tags: Sequence[str] = (),
    fields: Mapping[str, Any] | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    result = query_catalog(
        CatalogQuery(
            kind=kind,
            query=query,
            tags=tuple(tags),
            fields=dict(fields or {}),
            limit=limit,
            profile=profile,
            db_path=db_path,
        )
    )
    return result.as_dict()


def show_entry(key: str, *, db_path: str | None = None, profile: str = "default") -> dict[str, Any]:
    store = resolve_catalog_store(db_path, readonly=True)
    entry = store.get_catalog_entry(key, profile=profile)
    if entry is None:
        raise KeyError(f"catalog entry not found: {key}")
    entries = load_catalog_db(db_path, profile=profile).list()
    return {
        "entry": entry.as_dict(),
        "relations": build_entry_relation_payload(entry, all_entries=entries),
    }


def catalog_neighbors(key: str, *, db_path: str | None = None, profile: str = "default") -> dict[str, Any]:
    return dict(show_entry(key, db_path=db_path, profile=profile)["relations"]["neighbors"])


def catalog_flow(key: str, *, db_path: str | None = None, profile: str = "default") -> dict[str, Any]:
    return dict(show_entry(key, db_path=db_path, profile=profile)["relations"]["flow"])


def catalog_usage_matrix(*, db_path: str | None = None, profile: str = "default") -> dict[str, Any]:
    entries = load_catalog_db(db_path, profile=profile).list()
    by_kind = Counter(entry.kind for entry in entries)
    by_stage = Counter()
    rows = []
    for entry in entries:
        payload = build_entry_relation_payload(entry, all_entries=entries)
        stage = str(payload.get("flow", {}).get("current_stage", "") or "")
        by_stage[stage] += 1
        rows.append(
            {
                "key": entry.key,
                "kind": entry.kind,
                "stage": stage,
                "upstream": len(payload["neighbors"].get("context_upstream", ())),
                "downstream": len(payload["neighbors"].get("context_downstream", ())),
                "companions": len(payload["neighbors"].get("role_companions", ())),
            }
        )
    return {"by_kind": dict(by_kind), "by_stage": dict(by_stage), "rows": tuple(rows)}


def _entry_payload(entry: CatalogEntry) -> dict[str, Any]:
    return {
        "key": entry.key,
        "title": entry.title,
        "kind": entry.kind,
        "summary": entry.summary,
        "tags": tuple(entry.tags),
    }


__all__ = [
    "catalog_flow",
    "catalog_neighbors",
    "catalog_schema",
    "catalog_summary",
    "catalog_usage_matrix",
    "list_entries",
    "search_entries",
    "show_entry",
]
