from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import registry as _registry
from .project_catalog import find_project_root, load_project_entries
from .registry import (
    CatalogEntry,
    _BASE_ENTRY_FIELDS,
    _UI_FACET_FIELDS_BY_KIND,
    _entry_field_values,
    _flatten_scalars,
    _jsonable,
    _normalize_field_filters,
    _normalize_kind,
    _normalize_profile,
    _search_text,
)
from .sql_store import (
    catalog_db_config_enabled,
    catalog_db_config_info,
    catalog_db_config_mode,
    catalog_db_resolved_config,
    catalog_db_target_info,
)


@dataclass(frozen=True)
class _CatalogReadRoute:
    profile: str
    effective_source: str
    source_mode: str
    db_target: str | None = None
    db_target_info: Mapping[str, Any] | None = None
    db_materialized: bool | None = None
    db_readonly: bool = False
    db_source: str | None = None
    db_error: str | None = None
    explicit_db_path: bool = False
    config_enabled: bool = False
    config_path: str | None = None


@dataclass(frozen=True)
class _CatalogContext:
    profile: str
    scope: str
    entries: tuple[CatalogEntry, ...]
    project_root: str | None
    project_found: bool
    include_global: bool
    effective_source: str
    framework_source: str
    route: _CatalogReadRoute | None = None


def _normalize_mode(raw: str | None) -> str:
    key = str(raw or "").strip().lower()
    if key in {"only", "prefer", "off"}:
        return key
    if key == "disabled":
        return "off"
    return "prefer"


def _normalize_scope(raw: str | None) -> str:
    return "project" if str(raw or "").strip().lower() == "project" else "framework"


def _resolve_read_route(
    *,
    profile: str,
    db_path: str | None = None,
    source_mode: str | None = None,
) -> _CatalogReadRoute:
    profile_key = _normalize_profile(profile)
    explicit_db_path = bool(str(db_path or "").strip())
    mode = _normalize_mode(source_mode if source_mode is not None else catalog_db_config_mode())
    config = catalog_db_resolved_config()
    config_enabled = bool(config is not None or catalog_db_config_enabled())

    if explicit_db_path:
        target = str(db_path).strip()
        target_info = catalog_db_target_info(target)
        from .sql_store import catalog_db_summary

        summary = catalog_db_summary(target, profile=profile_key)
        materialized = bool(summary.get("materialized"))
        if not materialized:
            raise RuntimeError(
                f"catalog DB target '{target_info.get('db_target')}' has no materialized profile '{profile_key}'"
            )
        return _CatalogReadRoute(
            profile=profile_key,
            effective_source="db",
            source_mode="only",
            db_target=target,
            db_target_info=target_info,
            db_materialized=True,
            explicit_db_path=True,
            config_enabled=config_enabled,
            config_path=None if config is None else config.config_path,
            db_source="explicit",
        )

    if mode == "off":
        target_info = None
        target = None
        readonly = False
        source = None
        config_path = None
        if config is not None:
            target = str(config.target)
            target_info = catalog_db_target_info(target)
            readonly = bool(config.readonly)
            source = str(config.source)
            config_path = config.config_path
        return _CatalogReadRoute(
            profile=profile_key,
            effective_source="registry",
            source_mode=mode,
            config_enabled=config_enabled,
            config_path=config_path,
            db_target=target,
            db_target_info=target_info,
            db_readonly=readonly,
            db_source=source,
        )

    if config is None:
        if mode == "only":
            raise RuntimeError("catalog DB mode is 'only' but no catalog DB target is configured")
        return _CatalogReadRoute(
            profile=profile_key,
            effective_source="registry",
            source_mode=mode,
            config_enabled=False,
            config_path=None,
        )

    target_info = catalog_db_target_info(config.target)
    from .sql_store import catalog_db_summary

    try:
        summary = catalog_db_summary(config.target, profile=profile_key)
        materialized = bool(summary.get("materialized"))
    except Exception as exc:
        if mode == "prefer":
            return _CatalogReadRoute(
                profile=profile_key,
                effective_source="registry",
                source_mode=mode,
                db_target=config.target,
                db_target_info=target_info,
                db_materialized=None,
                db_readonly=bool(config.readonly),
                db_source=str(config.source),
                db_error=str(exc),
                config_enabled=True,
                config_path=config.config_path,
            )
        raise RuntimeError(
            f"catalog DB target '{target_info.get('db_target')}' is unavailable in mode '{mode}': {exc}"
        ) from exc

    if not materialized:
        if mode == "prefer":
            return _CatalogReadRoute(
                profile=profile_key,
                effective_source="registry",
                source_mode=mode,
                db_target=config.target,
                db_target_info=target_info,
                db_materialized=False,
                db_readonly=bool(config.readonly),
                db_source=str(config.source),
                config_enabled=True,
                config_path=config.config_path,
            )
        raise RuntimeError(
            f"catalog DB target '{target_info.get('db_target')}' has no materialized profile '{profile_key}'"
        )

    return _CatalogReadRoute(
        profile=profile_key,
        effective_source="db",
        source_mode=mode,
        db_target=config.target,
        db_target_info=target_info,
        db_materialized=True,
        db_readonly=bool(config.readonly),
        db_source=str(config.source),
        config_enabled=True,
        config_path=config.config_path,
    )


def _framework_entries(route: _CatalogReadRoute) -> tuple[CatalogEntry, ...]:
    if route.effective_source == "db":
        from .sql_store import catalog_db_list_entries

        assert route.db_target is not None
        return tuple(catalog_db_list_entries(route.db_target, profile=route.profile, limit=None))
    return tuple(_registry.list_entries(profile=route.profile, limit=None))


def _matches_field_filters(entry: CatalogEntry, filters: Sequence[tuple[str, str]]) -> bool:
    for field_name, expected_value in filters:
        values = {
            scalar
            for value in _entry_field_values(entry, field_name, include_relations=False)
            for scalar in _flatten_scalars(value)
            if scalar
        }
        if expected_value not in values:
            return False
    return True


def _filter_entries(
    entries: Iterable[CatalogEntry],
    *,
    kind: str | None,
    field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None,
) -> tuple[CatalogEntry, ...]:
    normalized_kind = _normalize_kind(kind)
    filters = _normalize_field_filters(field_filters)
    out: list[CatalogEntry] = []
    for entry in entries:
        if normalized_kind is not None and entry.kind != normalized_kind:
            continue
        if filters and not _matches_field_filters(entry, filters):
            continue
        out.append(entry)
    return tuple(out)


def _search_entries_in_collection(
    entries: Iterable[CatalogEntry],
    query: str,
    *,
    limit: int,
) -> tuple[CatalogEntry, ...]:
    def _normalized_chunks(values: Sequence[Any]) -> tuple[str, ...]:
        out: list[str] = []
        for value in values:
            text = str(value or "").strip().lower()
            if text:
                out.append(text)
        return tuple(out)

    def _searchable_chunks(entry: CatalogEntry) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        fields = dict(entry.fields or {})
        primary = _normalized_chunks(
            (
                entry.key,
                entry.name,
                fields.get("title_zh"),
                fields.get("title"),
            )
        )
        secondary_values: list[Any] = [
            entry.summary,
            fields.get("summary_zh"),
            entry.kind,
            entry.path,
            *tuple(entry.tags),
            *tuple(fields.get("use_when_zh", ()) or ()),
            *tuple(fields.get("use_when", ()) or ()),
        ]
        secondary = _normalized_chunks(secondary_values)
        field_values: list[Any] = []
        for field_name in (
            "family",
            "preset",
            "trainer",
            "head",
            "runtime_backend",
            "parameter_backend",
            "status",
            "surface_status",
            "preset_kind",
            "component",
            "component_surface",
            "component_kind",
            "provider",
            "provider_surface",
            "plane",
            "plugin",
            "plugin_surface",
            "lifecycle_plane",
            "bias",
            "pipeline",
            "numericizer",
            "mount_point",
        ):
            if field_name not in fields:
                continue
            field_values.extend(_flatten_scalars(fields.get(field_name)))
        tertiary = _normalized_chunks(field_values)
        return primary, secondary, tertiary

    def _match_score(chunks: Sequence[str], query_text: str) -> tuple[int, int] | None:
        best: tuple[int, int] | None = None
        for chunk in chunks:
            if chunk == query_text:
                score = (0, 0)
            else:
                index = chunk.find(query_text)
                if index < 0:
                    continue
                score = (1 if index == 0 else 2, int(index))
            if best is None or score < best:
                best = score
        return best

    q = str(query or "").strip().lower()
    if not q:
        rows = tuple(entries)
        return rows[: max(0, int(limit))]
    scored: list[tuple[tuple[int, int, int, str, str], CatalogEntry]] = []
    for entry in entries:
        primary, secondary, tertiary = _searchable_chunks(entry)
        score: tuple[int, int, int, str, str] | None = None
        for zone_index, chunks in enumerate((primary, secondary, tertiary)):
            zone_score = _match_score(chunks, q)
            if zone_score is None:
                continue
            score = (int(zone_index), int(zone_score[0]), int(zone_score[1]), str(entry.kind), str(entry.key))
            break
        if score is not None:
            scored.append((score, entry))
    scored.sort(key=lambda item: item[0])
    return tuple(entry for _, entry in scored[: max(0, int(limit))])


def _show_entry_in_collection(entries: Iterable[CatalogEntry], key: str) -> CatalogEntry | None:
    target = str(key or "").strip().lower()
    if not target:
        return None
    for entry in entries:
        if entry.key.lower() == target:
            return entry
    return None


def _schema_from_entries(
    entries: Sequence[CatalogEntry],
    *,
    profile: str,
    scope: str,
    project_root: str | None,
    project_found: bool,
    include_global: bool,
    kind: str | None,
) -> dict[str, Any]:
    by_kind: dict[str, dict[str, set[str]]] = {}
    for entry in entries:
        bucket = by_kind.setdefault(entry.kind, {"fields": set(), "relations": set()})
        bucket["fields"].update(str(name) for name in entry.fields.keys())
        bucket["relations"].update(str(name) for name in entry.relations.keys())

    kind_names = tuple(sorted(by_kind.keys()))
    all_fields = tuple(sorted({name for payload in by_kind.values() for name in payload["fields"]}))
    all_relations = tuple(sorted({name for payload in by_kind.values() for name in payload["relations"]}))
    normalized_kind = _normalize_kind(kind)

    payload: dict[str, Any] = {
        "profile": profile,
        "scope": scope,
        "project_root": project_root,
        "project_found": project_found,
        "include_global": include_global,
        "kind": normalized_kind,
        "kinds": kind_names,
        "base_fields": list(_BASE_ENTRY_FIELDS),
        "field_groups": {
            "base": list(_BASE_ENTRY_FIELDS),
            "fields": list(all_fields),
            "relations": list(all_relations),
        },
        "search_fields": ("all",),
        "counts": {"entries": len(entries)},
    }

    if normalized_kind is not None:
        kind_payload = by_kind.get(normalized_kind, {"fields": set(), "relations": set()})
        payload.update(
            {
                "fields": sorted(kind_payload["fields"]),
                "relations": sorted(kind_payload["relations"]),
                "count": len([entry for entry in entries if entry.kind == normalized_kind]),
            }
        )
        return payload

    payload.update(
        {
            "fields": sorted(all_fields),
            "relations": sorted(all_relations),
            "count": len(entries),
            "kind_details": {
                entry_kind: {
                    "fields": sorted(kind_payload["fields"]),
                    "relations": sorted(kind_payload["relations"]),
                    "count": len([entry for entry in entries if entry.kind == entry_kind]),
                }
                for entry_kind, kind_payload in sorted(by_kind.items(), key=lambda item: item[0])
            },
        }
    )
    return payload


def _neighbors_from_entries(entries: Sequence[CatalogEntry], key: str, *, profile: str) -> dict[str, Any]:
    entry = _show_entry_in_collection(entries, key)
    if entry is None:
        return {
            "profile": profile,
            "key": str(key),
            "entry": None,
            "neighbors": {},
        }

    neighbor_payload: dict[str, list[dict[str, Any]]] = {}
    for relation_name, relation_value in dict(entry.relations).items():
        rows: list[dict[str, Any]] = []
        for candidate_key in _flatten_scalars(relation_value):
            target = _show_entry_in_collection(entries, candidate_key)
            if target is None:
                rows.append({"key": str(candidate_key), "kind": None, "name": None, "missing": True})
                continue
            rows.append(
                {
                    "key": target.key,
                    "kind": target.kind,
                    "name": target.name,
                    "summary": target.summary,
                    "fields": _jsonable(dict(target.fields)),
                }
            )
        neighbor_payload[str(relation_name)] = rows

    return {
        "profile": profile,
        "key": entry.key,
        "entry": entry.to_dict(),
        "neighbors": neighbor_payload,
    }


def _load_catalog_context(
    *,
    profile: str = "default",
    scope: str | None = None,
    project_path: str | Path | None = None,
    include_global: bool = False,
    db_path: str | None = None,
    source_mode: str | None = None,
) -> _CatalogContext:
    normalized_profile = _normalize_profile(profile)
    normalized_scope = _normalize_scope(scope)

    if normalized_scope == "framework":
        route = _resolve_read_route(profile=normalized_profile, db_path=db_path, source_mode=source_mode)
        return _CatalogContext(
            profile=normalized_profile,
            scope="framework",
            entries=_framework_entries(route),
            project_root=None,
            project_found=True,
            include_global=False,
            effective_source=route.effective_source,
            framework_source=route.effective_source,
            route=route,
        )

    root = find_project_root(project_path)
    if root is None:
        return _CatalogContext(
            profile=normalized_profile,
            scope="project",
            entries=tuple(),
            project_root=None,
            project_found=False,
            include_global=bool(include_global),
            effective_source="project-missing",
            framework_source="registry",
            route=None,
        )

    local_entries = tuple(load_project_entries(root))
    if not include_global:
        return _CatalogContext(
            profile=normalized_profile,
            scope="project",
            entries=local_entries,
            project_root=str(root),
            project_found=True,
            include_global=False,
            effective_source="project",
            framework_source="registry",
            route=None,
        )

    route = _resolve_read_route(profile=normalized_profile, db_path=db_path, source_mode=source_mode)
    global_entries = _framework_entries(route)
    local_keys = {entry.key.lower() for entry in local_entries}
    merged = list(local_entries) + [entry for entry in global_entries if entry.key.lower() not in local_keys]
    return _CatalogContext(
        profile=normalized_profile,
        scope="project",
        entries=tuple(merged),
        project_root=str(root),
        project_found=True,
        include_global=True,
        effective_source=f"project+{route.effective_source}",
        framework_source=route.effective_source,
        route=route,
    )


def catalog_source_info(
    *,
    profile: str = "default",
    scope: str | None = None,
    project_path: str | Path | None = None,
    include_global: bool = False,
    db_path: str | None = None,
    source_mode: str | None = None,
) -> dict[str, Any]:
    context = _load_catalog_context(
        profile=profile,
        scope=scope,
        project_path=project_path,
        include_global=include_global,
        db_path=db_path,
        source_mode=source_mode,
    )
    info = {
        "profile": context.profile,
        "scope": context.scope,
        "project_root": context.project_root,
        "project_found": context.project_found,
        "include_global": context.include_global,
        "effective_source": context.effective_source,
        "framework_source": context.framework_source,
    }
    route = context.route
    if route is not None:
        info.update(
            {
                "source_mode": route.source_mode,
                "config_enabled": bool(route.config_enabled),
                "config_path": route.config_path,
                "explicit_db_path": bool(route.explicit_db_path),
                "db_materialized": route.db_materialized,
                "db_readonly": bool(route.db_readonly),
                "db_source": route.db_source,
                "db_error": route.db_error,
            }
        )
        info.update(dict(route.db_target_info or {}))
    else:
        info.update(
            {
                "source_mode": _normalize_mode(source_mode if source_mode is not None else catalog_db_config_mode()),
                "config_enabled": bool(catalog_db_config_enabled()),
                "config_path": catalog_db_config_info().get("config_path"),
                "explicit_db_path": bool(str(db_path or "").strip()),
                "db_materialized": None,
                "db_readonly": False,
                "db_source": None,
                "db_error": None,
            }
        )
    return info


def list_entries(
    *,
    profile: str = "default",
    scope: str | None = None,
    project_path: str | Path | None = None,
    include_global: bool = False,
    kind: str | None = None,
    limit: int | None = None,
    field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    db_path: str | None = None,
    source_mode: str | None = None,
) -> tuple[CatalogEntry, ...]:
    context = _load_catalog_context(
        profile=profile,
        scope=scope,
        project_path=project_path,
        include_global=include_global,
        db_path=db_path,
        source_mode=source_mode,
    )
    entries = _filter_entries(context.entries, kind=kind, field_filters=field_filters)
    if limit is None:
        return entries
    return entries[: max(0, int(limit))]


def search_entries(
    query: str,
    *,
    profile: str = "default",
    scope: str | None = None,
    project_path: str | Path | None = None,
    include_global: bool = False,
    kind: str | None = None,
    limit: int = 20,
    field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    db_path: str | None = None,
    source_mode: str | None = None,
) -> tuple[CatalogEntry, ...]:
    context = _load_catalog_context(
        profile=profile,
        scope=scope,
        project_path=project_path,
        include_global=include_global,
        db_path=db_path,
        source_mode=source_mode,
    )
    entries = _filter_entries(context.entries, kind=kind, field_filters=field_filters)
    return _search_entries_in_collection(entries, query, limit=limit)


def show_entry(
    key: str,
    *,
    profile: str = "default",
    scope: str | None = None,
    project_path: str | Path | None = None,
    include_global: bool = False,
    db_path: str | None = None,
    source_mode: str | None = None,
) -> CatalogEntry | None:
    context = _load_catalog_context(
        profile=profile,
        scope=scope,
        project_path=project_path,
        include_global=include_global,
        db_path=db_path,
        source_mode=source_mode,
    )
    return _show_entry_in_collection(context.entries, key)


def catalog_summary(
    *,
    profile: str = "default",
    scope: str | None = None,
    project_path: str | Path | None = None,
    include_global: bool = False,
    db_path: str | None = None,
    source_mode: str | None = None,
) -> dict[str, Any]:
    context = _load_catalog_context(
        profile=profile,
        scope=scope,
        project_path=project_path,
        include_global=include_global,
        db_path=db_path,
        source_mode=source_mode,
    )
    by_kind: dict[str, int] = {}
    for entry in context.entries:
        by_kind[entry.kind] = int(by_kind.get(entry.kind, 0) + 1)
    return {
        "profile": context.profile,
        "scope": context.scope,
        "project_root": context.project_root,
        "project_found": context.project_found,
        "include_global": context.include_global,
        "effective_source": context.effective_source,
        "total": int(len(context.entries)),
        "by_kind": dict(sorted(by_kind.items(), key=lambda item: item[0])),
    }


def field_values(
    field_name: str,
    *,
    profile: str = "default",
    scope: str | None = None,
    project_path: str | Path | None = None,
    include_global: bool = False,
    kind: str | None = None,
    limit: int | None = None,
    db_path: str | None = None,
    source_mode: str | None = None,
) -> tuple[str, ...]:
    context = _load_catalog_context(
        profile=profile,
        scope=scope,
        project_path=project_path,
        include_global=include_global,
        db_path=db_path,
        source_mode=source_mode,
    )
    entries = _filter_entries(context.entries, kind=kind, field_filters=None)
    values: set[str] = set()
    for entry in entries:
        for value in _entry_field_values(entry, str(field_name), include_relations=False):
            for scalar in _flatten_scalars(value):
                if scalar:
                    values.add(scalar)
    ordered = tuple(sorted(values))
    if limit is not None:
        return ordered[: max(0, int(limit))]
    return ordered


def catalog_schema(
    *,
    profile: str = "default",
    scope: str | None = None,
    project_path: str | Path | None = None,
    include_global: bool = False,
    kind: str | None = None,
    db_path: str | None = None,
    source_mode: str | None = None,
) -> dict[str, Any]:
    context = _load_catalog_context(
        profile=profile,
        scope=scope,
        project_path=project_path,
        include_global=include_global,
        db_path=db_path,
        source_mode=source_mode,
    )
    return _schema_from_entries(
        context.entries,
        profile=context.profile,
        scope=context.scope,
        project_root=context.project_root,
        project_found=context.project_found,
        include_global=context.include_global,
        kind=kind,
    )


def catalog_neighbors(
    key: str,
    *,
    profile: str = "default",
    scope: str | None = None,
    project_path: str | Path | None = None,
    include_global: bool = False,
    db_path: str | None = None,
    source_mode: str | None = None,
) -> dict[str, Any]:
    context = _load_catalog_context(
        profile=profile,
        scope=scope,
        project_path=project_path,
        include_global=include_global,
        db_path=db_path,
        source_mode=source_mode,
    )
    payload = _neighbors_from_entries(context.entries, key, profile=context.profile)
    payload.update(
        {
            "scope": context.scope,
            "project_root": context.project_root,
            "project_found": context.project_found,
            "include_global": context.include_global,
            "effective_source": context.effective_source,
        }
    )
    return payload


def catalog_facets(
    *,
    profile: str = "default",
    scope: str | None = None,
    project_path: str | Path | None = None,
    include_global: bool = False,
    kind: str | None = None,
    query: str | None = None,
    field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    fields: Sequence[str] | None = None,
    limit_per_field: int = 25,
    db_path: str | None = None,
    source_mode: str | None = None,
) -> dict[str, Any]:
    context = _load_catalog_context(
        profile=profile,
        scope=scope,
        project_path=project_path,
        include_global=include_global,
        db_path=db_path,
        source_mode=source_mode,
    )
    filters = _normalize_field_filters(field_filters)
    entries = _filter_entries(context.entries, kind=kind, field_filters=filters)
    if query and str(query).strip():
        entries = _search_entries_in_collection(entries, str(query), limit=10_000)

    schema = _schema_from_entries(
        context.entries,
        profile=context.profile,
        scope=context.scope,
        project_root=context.project_root,
        project_found=context.project_found,
        include_global=context.include_global,
        kind=kind,
    )
    kind_key = _normalize_kind(kind) or ""
    target_fields = tuple(str(value) for value in (fields or _UI_FACET_FIELDS_BY_KIND.get(kind_key, ())) if str(value).strip())
    if not target_fields:
        target_fields = tuple(str(value) for value in schema.get("fields", ()) if str(value).strip())

    facets: dict[str, list[dict[str, Any]]] = {}
    for field_name in target_fields:
        counts: dict[str, int] = {}
        for entry in entries:
            seen: set[str] = set()
            for value in _entry_field_values(entry, field_name, include_relations=False):
                for scalar in _flatten_scalars(value):
                    if not scalar or scalar in seen:
                        continue
                    seen.add(scalar)
                    counts[scalar] = int(counts.get(scalar, 0) + 1)
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        facets[field_name] = [
            {"value": str(value), "count": int(count)}
            for value, count in ordered[: max(0, int(limit_per_field))]
        ]

    return {
        "profile": context.profile,
        "scope": context.scope,
        "project_root": context.project_root,
        "project_found": context.project_found,
        "include_global": context.include_global,
        "kind": _normalize_kind(kind),
        "query": str(query or ""),
        "filters": [{"field": str(name), "value": str(value)} for name, value in filters],
        "total": int(len(entries)),
        "facets": facets,
    }


def catalog_ui_snapshot(
    *,
    profile: str = "default",
    scope: str | None = None,
    project_path: str | Path | None = None,
    include_global: bool = False,
    kind: str | None = None,
    query: str | None = None,
    field_filters: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    limit: int = 200,
    selected_key: str | None = None,
    db_path: str | None = None,
    source_mode: str | None = None,
) -> dict[str, Any]:
    context = _load_catalog_context(
        profile=profile,
        scope=scope,
        project_path=project_path,
        include_global=include_global,
        db_path=db_path,
        source_mode=source_mode,
    )
    filters = _normalize_field_filters(field_filters)
    items = _filter_entries(context.entries, kind=kind, field_filters=filters)
    if query and str(query).strip():
        items = _search_entries_in_collection(items, str(query), limit=limit)
    else:
        items = items[: max(0, int(limit))]

    selected = _show_entry_in_collection(context.entries, str(selected_key or "")) if selected_key else None
    return {
        "source": catalog_source_info(
            profile=context.profile,
            scope=context.scope,
            project_path=context.project_root,
            include_global=context.include_global,
            db_path=db_path,
            source_mode=source_mode,
        ),
        "summary": catalog_summary(
            profile=context.profile,
            scope=context.scope,
            project_path=context.project_root,
            include_global=context.include_global,
            db_path=db_path,
            source_mode=source_mode,
        ),
        "schema": catalog_schema(
            profile=context.profile,
            scope=context.scope,
            project_path=context.project_root,
            include_global=context.include_global,
            kind=kind,
            db_path=db_path,
            source_mode=source_mode,
        ),
        "facets": catalog_facets(
            profile=context.profile,
            scope=context.scope,
            project_path=context.project_root,
            include_global=context.include_global,
            kind=kind,
            query=query,
            field_filters=filters,
            db_path=db_path,
            source_mode=source_mode,
        ),
        "items": [entry.to_dict() for entry in items],
        "selected": None if selected is None else selected.to_dict(),
        "neighbors": None if selected is None else catalog_neighbors(
            selected.key,
            profile=context.profile,
            scope=context.scope,
            project_path=context.project_root,
            include_global=context.include_global,
            db_path=db_path,
            source_mode=source_mode,
        ),
    }


__all__ = [
    "CatalogEntry",
    "catalog_source_info",
    "catalog_db_config_info",
    "catalog_db_config_enabled",
    "catalog_db_config_mode",
    "list_entries",
    "search_entries",
    "show_entry",
    "catalog_summary",
    "field_values",
    "catalog_schema",
    "catalog_neighbors",
    "catalog_facets",
    "catalog_ui_snapshot",
]
