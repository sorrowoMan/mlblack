from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

from .registry import Catalog, CatalogEntry, get_catalog


@dataclass(frozen=True)
class CatalogQuery:
    kind: str | None = None
    query: str = ""
    tags: Sequence[str] = tuple()
    fields: Mapping[str, Any] = field(default_factory=dict)
    limit: int = 100

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "query": self.query,
            "tags": list(self.tags),
            "fields": dict(self.fields),
            "limit": int(self.limit),
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


def query_catalog(query: CatalogQuery | Mapping[str, Any] | None = None, *, catalog: Catalog | None = None) -> CatalogQueryResult:
    q = query if isinstance(query, CatalogQuery) else CatalogQuery(**dict(query or {}))
    cat = catalog or get_catalog()
    entries = list(cat.search(q.query, kind=q.kind, limit=max(1, int(q.limit))))
    tag_filter = {str(tag) for tag in q.tags}
    if tag_filter:
        entries = [item for item in entries if tag_filter.issubset({str(tag) for tag in item.tags})]
    for key, expected in dict(q.fields).items():
        entries = [item for item in entries if _entry_field(item, key) == expected]
    facets = build_catalog_facets(entries)
    return CatalogQueryResult(query=q, entries=tuple(entries), facets=facets, deep_link=build_catalog_deep_link(q))


def build_catalog_facets(entries: Sequence[CatalogEntry]) -> dict[str, dict[str, int]]:
    facets: dict[str, dict[str, int]] = {"kind": {}, "tags": {}}
    for item in entries:
        facets["kind"][item.kind] = facets["kind"].get(item.kind, 0) + 1
        for tag in item.tags:
            value = str(tag)
            facets["tags"][value] = facets["tags"].get(value, 0) + 1
    return facets


def build_catalog_deep_link(query: CatalogQuery, *, base: str = "mlblack://catalog") -> str:
    params = {"q": query.query, "limit": int(query.limit)}
    if query.kind:
        params["kind"] = query.kind
    if query.tags:
        params["tags"] = ",".join(str(tag) for tag in query.tags)
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
