from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

from .registry import Catalog, CatalogEntry
from .store import resolve_catalog_store


@dataclass(frozen=True)
class CatalogQuery:
    kind: str | None = None
    query: str = ""
    tags: Sequence[str] = tuple()
    fields: Mapping[str, Any] = field(default_factory=dict)
    limit: int = 100
    profile: str = "default"
    source: str = "db"
    db_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "query": self.query,
            "tags": list(self.tags),
            "fields": dict(self.fields),
            "limit": int(self.limit),
            "profile": self.profile,
            "source": self.source,
            "db_path": self.db_path,
        }


@dataclass(frozen=True)
class CatalogQueryResult:
    query: CatalogQuery
    entries: Sequence[CatalogEntry]
    facets: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    deep_link: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.as_dict(),
            "entries": [item.as_dict() for item in self.entries],
            "facets": {str(k): dict(v) for k, v in self.facets.items()},
            "deep_link": self.deep_link,
        }


def query_catalog(
    query: CatalogQuery | Mapping[str, Any] | None = None,
    *,
    catalog: Catalog | None = None,
    db_path: str | None = None,
    source: str | None = None,
    profile: str | None = None,
) -> CatalogQueryResult:
    q = query if isinstance(query, CatalogQuery) else CatalogQuery(**dict(query or {}))
    effective = CatalogQuery(
        kind=q.kind,
        query=q.query,
        tags=tuple(q.tags),
        fields=dict(q.fields),
        limit=q.limit,
        profile=profile or q.profile,
        source="db",
        db_path=db_path or q.db_path,
    )
    if catalog is not None:
        entries = _query_catalog_object(catalog, effective)
    else:
        entries = _query_db_entries(effective)
    facets = build_catalog_facets(entries)
    return CatalogQueryResult(
        query=effective,
        entries=tuple(entries),
        facets=facets,
        deep_link=build_catalog_deep_link(effective),
    )


def query_catalog_db(
    query: CatalogQuery | Mapping[str, Any] | None = None,
    *,
    db_path: str | None = None,
    profile: str = "default",
) -> CatalogQueryResult:
    q = query if isinstance(query, CatalogQuery) else CatalogQuery(**dict(query or {}))
    return query_catalog(
        CatalogQuery(
            kind=q.kind,
            query=q.query,
            tags=tuple(q.tags),
            fields=dict(q.fields),
            limit=q.limit,
            profile=profile or q.profile,
            source="db",
            db_path=db_path or q.db_path,
        )
    )


def build_catalog_facets(entries: Sequence[CatalogEntry]) -> dict[str, dict[str, int]]:
    facets: dict[str, dict[str, int]] = {"kind": {}, "tags": {}}
    for item in entries:
        facets["kind"][item.kind] = facets["kind"].get(item.kind, 0) + 1
        for tag in item.tags:
            value = str(tag)
            facets["tags"][value] = facets["tags"].get(value, 0) + 1
    return facets


def build_catalog_deep_link(query: CatalogQuery, *, base: str = "mlblack://catalog") -> str:
    params = {"q": query.query, "limit": int(query.limit), "source": query.source, "profile": query.profile}
    if query.kind:
        params["kind"] = query.kind
    if query.tags:
        params["tags"] = ",".join(str(tag) for tag in query.tags)
    if query.db_path:
        params["db_path"] = query.db_path
    for key, value in query.fields.items():
        params[f"field.{key}"] = str(value)
    return f"{base}?{urlencode(params)}"


def _entry_field(entry: CatalogEntry, key: str) -> Any:
    if hasattr(entry, key):
        return getattr(entry, key)
    if key in entry.metadata:
        return entry.metadata[key]
    if key in entry.contract:
        return entry.contract[key]
    return None


def _query_catalog_object(catalog: Catalog, query: CatalogQuery) -> list[CatalogEntry]:
    entries = list(catalog.search(query.query, kind=query.kind, limit=max(1, int(query.limit))))
    tag_filter = {str(tag) for tag in query.tags}
    if tag_filter:
        entries = [item for item in entries if tag_filter.issubset({str(tag) for tag in item.tags})]
    for key, expected in dict(query.fields).items():
        entries = [item for item in entries if _entry_field(item, key) == expected]
    return entries


def _query_db_entries(query: CatalogQuery) -> list[CatalogEntry]:
    store = resolve_catalog_store(query.db_path, readonly=True)
    if str(query.query or "").strip():
        return store.search_catalog_entries(
            query.query,
            profile=query.profile,
            kind=query.kind,
            tags=query.tags,
            limit=max(1, int(query.limit)),
            field_filters=query.fields,
        )
    return store.list_catalog_entries(
        profile=query.profile,
        kind=query.kind,
        tags=query.tags,
        limit=max(1, int(query.limit)),
        field_filters=query.fields,
    )
