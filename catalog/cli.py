from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .dashboard import launch_catalog_dashboard
from .facade import (
    catalog_facets,
    catalog_neighbors,
    catalog_schema,
    catalog_source_info,
    catalog_summary,
    catalog_ui_snapshot,
    field_values,
    list_entries,
    search_entries,
    show_entry,
)
from .sql_store import (
    catalog_db_config_info,
    catalog_db_facets,
    catalog_db_field_values,
    catalog_db_list_entries,
    catalog_db_neighbors,
    catalog_db_relation_edges,
    catalog_db_relation_keys,
    catalog_db_resolved_config,
    catalog_db_search_entries,
    catalog_db_show_entry,
    catalog_db_summary,
    catalog_db_target_info,
    catalog_db_ui_snapshot,
    materialize_catalog_db,
)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _parse_field_filters(values: list[str] | None) -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    for raw in values or []:
        text = str(raw).strip()
        if not text:
            continue
        key, sep, value = text.partition("=")
        if not sep:
            raise ValueError(f"field filter must use key=value form, got: {raw}")
        if not key.strip() or not value.strip():
            raise ValueError(f"field filter must use non-empty key=value form, got: {raw}")
        out.append((key.strip(), value.strip()))
    return tuple(out)


def _field_hints(item: Any) -> str:
    fields = dict(getattr(item, "fields", {}) or {})
    chunks: list[str] = []
    for key in ("family", "preset", "head", "status", "runtime_backend"):
        if key not in fields:
            continue
        value = fields.get(key)
        if value in (None, "", (), [], {}):
            continue
        if isinstance(value, (tuple, list)):
            rendered = ",".join(str(v) for v in value)
        else:
            rendered = str(value)
        chunks.append(f"{key}={rendered}")
    return " " + " ".join(chunks) if chunks else ""


def _catalog_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if hasattr(args, "scope") and args.scope is not None:
        payload["scope"] = str(args.scope)
    if hasattr(args, "project_path") and args.project_path is not None:
        payload["project_path"] = str(args.project_path)
    if hasattr(args, "include_global"):
        payload["include_global"] = bool(args.include_global)
    if hasattr(args, "db_path") and args.db_path is not None:
        payload["db_path"] = str(args.db_path)
    if hasattr(args, "source_mode") and args.source_mode is not None:
        payload["source_mode"] = str(args.source_mode)
    return payload


def _resolve_db_target_arg(args: argparse.Namespace, *, require_write: bool = False) -> str:
    raw = str(getattr(args, "db_path", "") or "").strip()
    if raw:
        return raw

    resolved = catalog_db_resolved_config()
    if resolved is None:
        raise RuntimeError(
            "no catalog DB target configured. Set --db-path, MLBLACK_CATALOG_DB_URL, or catalog/db.toml."
        )
    if require_write and bool(resolved.readonly):
        raise RuntimeError(
            "catalog DB config is readonly. Pass --db-path explicitly or disable readonly before materializing."
        )
    return str(resolved.target)


def _cmd_list(args: argparse.Namespace) -> int:
    filters = _parse_field_filters(args.field)
    items = list_entries(
        profile=args.profile,
        kind=args.kind,
        limit=args.limit,
        field_filters=filters,
        **_catalog_kwargs(args),
    )
    if args.format == "json":
        _print_json([x.to_dict() for x in items])
        return 0

    for item in items:
        p = f" path={item.path}" if item.path else ""
        print(f"{item.key} kind={item.kind} source={item.source}{p}{_field_hints(item)}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    filters = _parse_field_filters(args.field)
    items = search_entries(
        args.query,
        profile=args.profile,
        kind=args.kind,
        limit=args.limit,
        field_filters=filters,
        **_catalog_kwargs(args),
    )
    if args.format == "json":
        _print_json([x.to_dict() for x in items])
        return 0

    for item in items:
        p = f" path={item.path}" if item.path else ""
        print(f"{item.key} kind={item.kind} source={item.source}{p}{_field_hints(item)}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    item = show_entry(args.key, profile=args.profile, **_catalog_kwargs(args))
    if item is None:
        print(f"not found: {args.key}")
        return 1
    _print_json(item.to_dict())
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    _print_json(catalog_summary(profile=args.profile, **_catalog_kwargs(args)))
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    _print_json(catalog_schema(profile=args.profile, kind=args.kind, **_catalog_kwargs(args)))
    return 0


def _cmd_values(args: argparse.Namespace) -> int:
    payload = {
        "profile": str(args.profile),
        "kind": args.kind,
        "field": str(args.field_name),
        "values": list(
            field_values(
                args.field_name,
                profile=args.profile,
                kind=args.kind,
                limit=args.limit,
                **_catalog_kwargs(args),
            )
        ),
    }
    _print_json(payload)
    return 0


def _cmd_neighbors(args: argparse.Namespace) -> int:
    _print_json(catalog_neighbors(args.key, profile=args.profile, **_catalog_kwargs(args)))
    return 0


def _cmd_facets(args: argparse.Namespace) -> int:
    filters = _parse_field_filters(args.field)
    _print_json(
        catalog_facets(
            profile=args.profile,
            kind=args.kind,
            query=args.query,
            field_filters=filters,
            fields=args.facet_field or None,
            limit_per_field=args.limit_per_field,
            **_catalog_kwargs(args),
        )
    )
    return 0


def _cmd_snapshot(args: argparse.Namespace) -> int:
    filters = _parse_field_filters(args.field)
    _print_json(
        catalog_ui_snapshot(
            profile=args.profile,
            kind=args.kind,
            query=args.query,
            field_filters=filters,
            limit=args.limit,
            selected_key=args.selected,
            **_catalog_kwargs(args),
        )
    )
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    return int(
        launch_catalog_dashboard(
            profile=str(args.profile),
            scope=str(args.scope),
            kind=str(args.kind),
            query=str(args.query or ""),
            project_path=None if args.project_path is None else str(args.project_path),
            include_global=bool(args.include_global),
            db_path=None if args.db_path is None else str(args.db_path),
            source_mode=None if args.source_mode is None else str(args.source_mode),
            column_mode=str(args.column_mode),
            page_size=int(args.page_size),
            results_collapse=str(args.results_collapse),
            host=str(args.host) if args.host else None,
            port=int(args.port) if args.port is not None else None,
            headless=bool(args.headless),
        )
    )


def _cmd_db_materialize(args: argparse.Namespace) -> int:
    _print_json(materialize_catalog_db(_resolve_db_target_arg(args, require_write=True), profile=str(args.profile)))
    return 0


def _cmd_db_summary(args: argparse.Namespace) -> int:
    _print_json(
        catalog_db_summary(
            _resolve_db_target_arg(args),
            profile=None if args.profile is None else str(args.profile),
        )
    )
    return 0


def _cmd_db_target(args: argparse.Namespace) -> int:
    raw = str(getattr(args, "db_path", "") or "").strip()
    if raw:
        payload = catalog_db_target_info(raw)
    else:
        payload = catalog_db_config_info()
    _print_json(payload)
    return 0


def _cmd_db_show(args: argparse.Namespace) -> int:
    item = catalog_db_show_entry(
        _resolve_db_target_arg(args),
        str(args.key),
        profile=str(args.profile),
    )
    if item is None:
        print(f"not found in catalog db: {args.key}")
        return 1
    _print_json(item.to_dict())
    return 0


def _cmd_db_list(args: argparse.Namespace) -> int:
    filters = _parse_field_filters(args.field)
    items = catalog_db_list_entries(
        _resolve_db_target_arg(args),
        profile=str(args.profile),
        kind=args.kind,
        limit=args.limit,
        field_filters=filters,
    )
    if args.format == "json":
        _print_json([x.to_dict() for x in items])
        return 0

    for item in items:
        p = f" path={item.path}" if item.path else ""
        print(f"{item.key} kind={item.kind} source={item.source}{p}{_field_hints(item)}")
    return 0


def _cmd_db_search(args: argparse.Namespace) -> int:
    filters = _parse_field_filters(args.field)
    items = catalog_db_search_entries(
        _resolve_db_target_arg(args),
        str(args.query),
        profile=str(args.profile),
        kind=args.kind,
        limit=args.limit,
        field_filters=filters,
    )
    if args.format == "json":
        _print_json([x.to_dict() for x in items])
        return 0

    for item in items:
        p = f" path={item.path}" if item.path else ""
        print(f"{item.key} kind={item.kind} source={item.source}{p}{_field_hints(item)}")
    return 0


def _cmd_db_facets(args: argparse.Namespace) -> int:
    filters = _parse_field_filters(args.field)
    _print_json(
        catalog_db_facets(
            _resolve_db_target_arg(args),
            profile=str(args.profile),
            kind=args.kind,
            query=str(args.query or ""),
            field_filters=filters,
            fields=args.facet_field or None,
            limit_per_field=args.limit_per_field,
        )
    )
    return 0


def _cmd_db_values(args: argparse.Namespace) -> int:
    target = _resolve_db_target_arg(args)
    payload = {
        "db_path": target,
        "profile": str(args.profile),
        "kind": args.kind,
        "field": str(args.field_name),
        "values": list(
            catalog_db_field_values(
                target,
                str(args.field_name),
                profile=str(args.profile),
                kind=args.kind,
                limit=args.limit,
            )
        ),
    }
    _print_json(payload)
    return 0


def _cmd_db_neighbors(args: argparse.Namespace) -> int:
    _print_json(
        catalog_db_neighbors(
            _resolve_db_target_arg(args),
            str(args.key),
            profile=str(args.profile),
        )
    )
    return 0


def _cmd_db_relation_edges(args: argparse.Namespace) -> int:
    _print_json(
        list(
            catalog_db_relation_edges(
                _resolve_db_target_arg(args),
                profile=str(args.profile),
                kind=args.kind,
                relation_name=args.relation_name,
                target_kind=args.target_kind,
                source_key=args.source_key,
                limit=args.limit,
            )
        )
    )
    return 0


def _cmd_db_relation_keys(args: argparse.Namespace) -> int:
    _print_json(
        list(
            catalog_db_relation_keys(
                _resolve_db_target_arg(args),
                profile=str(args.profile),
                kind=args.kind,
                relation_name=args.relation_name,
                limit=args.limit,
            )
        )
    )
    return 0


def _cmd_db_snapshot(args: argparse.Namespace) -> int:
    filters = _parse_field_filters(args.field)
    _print_json(
        catalog_db_ui_snapshot(
            _resolve_db_target_arg(args),
            profile=str(args.profile),
            kind=args.kind,
            query=str(args.query or ""),
            field_filters=filters,
            limit=args.limit,
            selected_key=args.selected,
        )
    )
    return 0


def _cmd_source(args: argparse.Namespace) -> int:
    _print_json(
        catalog_source_info(
            profile=str(args.profile),
            db_path=None if args.db_path is None else str(args.db_path),
            source_mode=None if args.source_mode is None else str(args.source_mode),
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="mlblack catalog CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_source_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--db-path",
            type=str,
            default=None,
            help="Optional catalog DB target. Accepts a sqlite file path or a SQLAlchemy URL.",
        )
        p.add_argument(
            "--source-mode",
            type=str,
            default=None,
            choices=["prefer", "only", "off"],
            help="Optional source routing override. Defaults to MLBLACK_CATALOG_DB_MODE or catalog/db.toml mode.",
        )

    def add_scope_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--scope", type=str, default="framework", choices=["framework", "project"])
        p.add_argument("--project-path", type=str, default=None, help="Optional project root or child path for project scope.")
        p.add_argument(
            "--include-global",
            action="store_true",
            help="When scope=project, merge framework catalog entries into the project-local view.",
        )

    p_list = sub.add_parser("list", help="List catalog entries")
    p_list.add_argument("--profile", type=str, default="default")
    p_list.add_argument("--kind", type=str, default=None)
    p_list.add_argument("--field", action="append", default=[], help="Structured field filter in key=value form.")
    p_list.add_argument("--limit", type=int, default=200)
    p_list.add_argument("--format", type=str, default="text", choices=["text", "json"])
    add_scope_args(p_list)
    add_source_args(p_list)
    p_list.set_defaults(_fn=_cmd_list)

    p_search = sub.add_parser("search", help="Search catalog entries")
    p_search.add_argument("query", type=str)
    p_search.add_argument("--profile", type=str, default="default")
    p_search.add_argument("--kind", type=str, default=None)
    p_search.add_argument("--field", action="append", default=[], help="Structured field filter in key=value form.")
    p_search.add_argument("--limit", type=int, default=20)
    p_search.add_argument("--format", type=str, default="text", choices=["text", "json"])
    add_scope_args(p_search)
    add_source_args(p_search)
    p_search.set_defaults(_fn=_cmd_search)

    p_show = sub.add_parser("show", help="Show one catalog entry")
    p_show.add_argument("key", type=str)
    p_show.add_argument("--profile", type=str, default="default")
    add_scope_args(p_show)
    add_source_args(p_show)
    p_show.set_defaults(_fn=_cmd_show)

    p_summary = sub.add_parser("summary", help="Show catalog summary")
    p_summary.add_argument("--profile", type=str, default="default")
    add_scope_args(p_summary)
    add_source_args(p_summary)
    p_summary.set_defaults(_fn=_cmd_summary)

    p_schema = sub.add_parser("schema", help="Show structured field schema for one kind or the whole catalog")
    p_schema.add_argument("--profile", type=str, default="default")
    p_schema.add_argument("--kind", type=str, default=None)
    add_scope_args(p_schema)
    add_source_args(p_schema)
    p_schema.set_defaults(_fn=_cmd_schema)

    p_values = sub.add_parser("values", help="List distinct values for one structured field")
    p_values.add_argument("field_name", type=str)
    p_values.add_argument("--profile", type=str, default="default")
    p_values.add_argument("--kind", type=str, default=None)
    p_values.add_argument("--limit", type=int, default=200)
    add_scope_args(p_values)
    add_source_args(p_values)
    p_values.set_defaults(_fn=_cmd_values)

    p_neighbors = sub.add_parser("neighbors", help="Show relation-linked neighbor entries for one catalog entry")
    p_neighbors.add_argument("key", type=str)
    p_neighbors.add_argument("--profile", type=str, default="default")
    add_scope_args(p_neighbors)
    add_source_args(p_neighbors)
    p_neighbors.set_defaults(_fn=_cmd_neighbors)

    p_facets = sub.add_parser("facets", help="Show facet counts for structured catalog fields")
    p_facets.add_argument("--profile", type=str, default="default")
    p_facets.add_argument("--kind", type=str, default=None)
    p_facets.add_argument("--query", type=str, default="")
    p_facets.add_argument("--field", action="append", default=[], help="Structured field filter in key=value form.")
    p_facets.add_argument(
        "--facet-field",
        action="append",
        default=[],
        help="Restrict facet output to one or more field names. Defaults to all fields for the chosen kind.",
    )
    p_facets.add_argument("--limit-per-field", type=int, default=25)
    add_scope_args(p_facets)
    add_source_args(p_facets)
    p_facets.set_defaults(_fn=_cmd_facets)

    p_snapshot = sub.add_parser("snapshot", help="Export one UI-friendly catalog snapshot payload")
    p_snapshot.add_argument("--profile", type=str, default="default")
    p_snapshot.add_argument("--kind", type=str, default=None)
    p_snapshot.add_argument("--query", type=str, default="")
    p_snapshot.add_argument("--field", action="append", default=[], help="Structured field filter in key=value form.")
    p_snapshot.add_argument("--selected", type=str, default=None, help="Optional selected entry key for neighbor expansion.")
    p_snapshot.add_argument("--limit", type=int, default=200)
    add_scope_args(p_snapshot)
    add_source_args(p_snapshot)
    p_snapshot.set_defaults(_fn=_cmd_snapshot)

    p_source = sub.add_parser("source", help="Show effective catalog source routing after env/config/mode resolution")
    p_source.add_argument("--profile", type=str, default="default")
    add_scope_args(p_source)
    add_source_args(p_source)
    p_source.set_defaults(_fn=_cmd_source)

    p_ui = sub.add_parser(
        "ui",
        help="Launch the standalone catalog page",
        description="Launch the standalone catalog page for structured family/preset/head/component/provider/plugin browsing.",
    )
    p_ui.add_argument("--profile", type=str, default="framework-core")
    p_ui.add_argument("--scope", type=str, default="framework", choices=["framework", "project"])
    p_ui.add_argument("--kind", type=str, default="preset", choices=["family", "preset", "head", "component", "provider", "plugin"])
    p_ui.add_argument("--query", type=str, default="")
    p_ui.add_argument("--project-path", type=str, default=None, help="Optional project root or child path for project scope.")
    p_ui.add_argument("--include-global", action="store_true", help="When scope=project, merge framework entries into the project view.")
    p_ui.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Optional catalog DB target. Accepts a sqlite file path or a SQLAlchemy URL.",
    )
    p_ui.add_argument(
        "--source-mode",
        type=str,
        default=None,
        choices=["prefer", "only", "off"],
        help="Optional source routing override. Defaults to MLBLACK_CATALOG_DB_MODE or catalog/db.toml mode.",
    )
    p_ui.add_argument(
        "--column-mode",
        type=str,
        default="standard",
        choices=["compact", "standard", "full"],
        help="Initial results table column scheme.",
    )
    p_ui.add_argument("--page-size", type=int, default=50, help="Initial results table visible window size.")
    p_ui.add_argument(
        "--results-collapse",
        type=str,
        default="expanded",
        choices=["expanded", "collapsed"],
        help="Initial results section collapse mode.",
    )
    p_ui.add_argument("--host", type=str, default=None, help="Optional Streamlit server address.")
    p_ui.add_argument("--port", type=int, default=None, help="Optional Streamlit server port.")
    p_ui.add_argument("--headless", action="store_true", help="Launch Streamlit without opening a browser window.")
    p_ui.set_defaults(_fn=_cmd_ui)

    p_db = sub.add_parser("db", help="Materialize or inspect SQL-backed catalog storage")
    sub_db = p_db.add_subparsers(dest="db_cmd", required=True)

    p_db_materialize = sub_db.add_parser("materialize", help="Materialize structured catalog into a SQL catalog store")
    p_db_materialize.add_argument("--db-path", required=False, help="Catalog DB target: sqlite file path or SQLAlchemy URL.")
    p_db_materialize.add_argument("--profile", type=str, default="framework-core")
    p_db_materialize.set_defaults(_fn=_cmd_db_materialize)

    p_db_target = sub_db.add_parser("target", help="Inspect one catalog DB target or the configured catalog DB protocol")
    p_db_target.add_argument("--db-path", required=False, help="Catalog DB target: sqlite file path or SQLAlchemy URL.")
    p_db_target.set_defaults(_fn=_cmd_db_target)

    p_db_summary = sub_db.add_parser("summary", help="Show summary for one materialized catalog DB")
    p_db_summary.add_argument("--db-path", required=False, help="Catalog DB target: sqlite file path or SQLAlchemy URL.")
    p_db_summary.add_argument("--profile", type=str, default=None, help="Optional profile filter. Defaults to all profiles.")
    p_db_summary.set_defaults(_fn=_cmd_db_summary)

    p_db_show = sub_db.add_parser("show", help="Show one materialized catalog entry from SQL storage")
    p_db_show.add_argument("key", type=str)
    p_db_show.add_argument("--db-path", required=False, help="Catalog DB target: sqlite file path or SQLAlchemy URL.")
    p_db_show.add_argument("--profile", type=str, default="framework-core")
    p_db_show.set_defaults(_fn=_cmd_db_show)

    p_db_list = sub_db.add_parser("list", help="List materialized catalog entries from SQL storage")
    p_db_list.add_argument("--db-path", required=False, help="Catalog DB target: sqlite file path or SQLAlchemy URL.")
    p_db_list.add_argument("--profile", type=str, default="framework-core")
    p_db_list.add_argument("--kind", type=str, default=None)
    p_db_list.add_argument("--field", action="append", default=[], help="Structured field filter in key=value form.")
    p_db_list.add_argument("--limit", type=int, default=200)
    p_db_list.add_argument("--format", type=str, default="text", choices=["text", "json"])
    p_db_list.set_defaults(_fn=_cmd_db_list)

    p_db_search = sub_db.add_parser("search", help="Search materialized catalog entries from SQL storage")
    p_db_search.add_argument("query", type=str)
    p_db_search.add_argument("--db-path", required=False, help="Catalog DB target: sqlite file path or SQLAlchemy URL.")
    p_db_search.add_argument("--profile", type=str, default="framework-core")
    p_db_search.add_argument("--kind", type=str, default=None)
    p_db_search.add_argument("--field", action="append", default=[], help="Structured field filter in key=value form.")
    p_db_search.add_argument("--limit", type=int, default=20)
    p_db_search.add_argument("--format", type=str, default="text", choices=["text", "json"])
    p_db_search.set_defaults(_fn=_cmd_db_search)

    p_db_facets = sub_db.add_parser("facets", help="Show facet counts from materialized SQL catalog")
    p_db_facets.add_argument("--db-path", required=False, help="Catalog DB target: sqlite file path or SQLAlchemy URL.")
    p_db_facets.add_argument("--profile", type=str, default="framework-core")
    p_db_facets.add_argument("--kind", type=str, default=None)
    p_db_facets.add_argument("--query", type=str, default="")
    p_db_facets.add_argument("--field", action="append", default=[], help="Structured field filter in key=value form.")
    p_db_facets.add_argument(
        "--facet-field",
        action="append",
        default=[],
        help="Restrict facet output to one or more field names. Defaults to all fields for the chosen kind.",
    )
    p_db_facets.add_argument("--limit-per-field", type=int, default=25)
    p_db_facets.set_defaults(_fn=_cmd_db_facets)

    p_db_values = sub_db.add_parser("values", help="List distinct structured field values from materialized SQL catalog")
    p_db_values.add_argument("field_name", type=str)
    p_db_values.add_argument("--db-path", required=False, help="Catalog DB target: sqlite file path or SQLAlchemy URL.")
    p_db_values.add_argument("--profile", type=str, default="framework-core")
    p_db_values.add_argument("--kind", type=str, default=None)
    p_db_values.add_argument("--limit", type=int, default=200)
    p_db_values.set_defaults(_fn=_cmd_db_values)

    p_db_neighbors = sub_db.add_parser("neighbors", help="Show relation-linked neighbor entries from materialized SQL catalog")
    p_db_neighbors.add_argument("key", type=str)
    p_db_neighbors.add_argument("--db-path", required=False, help="Catalog DB target: sqlite file path or SQLAlchemy URL.")
    p_db_neighbors.add_argument("--profile", type=str, default="framework-core")
    p_db_neighbors.set_defaults(_fn=_cmd_db_neighbors)

    p_db_relation_edges = sub_db.add_parser("relation-edges", help="List materialized relation edges from SQL catalog")
    p_db_relation_edges.add_argument("--db-path", required=False, help="Catalog DB target: sqlite file path or SQLAlchemy URL.")
    p_db_relation_edges.add_argument("--profile", type=str, default="framework-core")
    p_db_relation_edges.add_argument("--kind", type=str, default=None)
    p_db_relation_edges.add_argument("--relation-name", type=str, default=None)
    p_db_relation_edges.add_argument("--target-kind", type=str, default=None)
    p_db_relation_edges.add_argument("--source-key", type=str, default=None)
    p_db_relation_edges.add_argument("--limit", type=int, default=200)
    p_db_relation_edges.set_defaults(_fn=_cmd_db_relation_edges)

    p_db_relation_keys = sub_db.add_parser("relation-keys", help="List aggregated relation key groups from SQL catalog")
    p_db_relation_keys.add_argument("--db-path", required=False, help="Catalog DB target: sqlite file path or SQLAlchemy URL.")
    p_db_relation_keys.add_argument("--profile", type=str, default="framework-core")
    p_db_relation_keys.add_argument("--kind", type=str, default=None)
    p_db_relation_keys.add_argument("--relation-name", type=str, default=None)
    p_db_relation_keys.add_argument("--limit", type=int, default=200)
    p_db_relation_keys.set_defaults(_fn=_cmd_db_relation_keys)

    p_db_snapshot = sub_db.add_parser("snapshot", help="Export one UI-friendly snapshot from materialized SQL catalog")
    p_db_snapshot.add_argument("--db-path", required=False, help="Catalog DB target: sqlite file path or SQLAlchemy URL.")
    p_db_snapshot.add_argument("--profile", type=str, default="framework-core")
    p_db_snapshot.add_argument("--kind", type=str, default=None)
    p_db_snapshot.add_argument("--query", type=str, default="")
    p_db_snapshot.add_argument("--field", action="append", default=[], help="Structured field filter in key=value form.")
    p_db_snapshot.add_argument("--selected", type=str, default=None, help="Optional selected entry key for neighbor expansion.")
    p_db_snapshot.add_argument("--limit", type=int, default=200)
    p_db_snapshot.set_defaults(_fn=_cmd_db_snapshot)

    args = parser.parse_args(argv)
    fn = getattr(args, "_fn")
    try:
        return int(fn(args))
    except ValueError as exc:
        parser.error(str(exc))
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

