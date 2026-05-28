from __future__ import annotations

import argparse
import sys
from typing import Sequence

from mlblack.catalog.registry import CatalogEntry, get_catalog


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlblack", description="mlblack command line entrypoint.")
    subparsers = parser.add_subparsers(dest="command")

    webui_parser = subparsers.add_parser("webui", help="Refresh DB and launch the catalog WebUI.")
    _add_webui_args(webui_parser)

    catalog_parser = subparsers.add_parser("catalog", help="Discoverability registry (where is X?)")
    catalog_subparsers = catalog_parser.add_subparsers(dest="catalog_command")

    p_search = catalog_subparsers.add_parser("search", help="Search entries by keyword")
    p_search.add_argument("query", help="Search query")
    _add_catalog_filters(p_search)
    p_search.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")
    p_search.set_defaults(func=_cmd_catalog_search)

    p_list = catalog_subparsers.add_parser("list", help="List entries by kind/tag")
    _add_catalog_filters(p_list)
    p_list.set_defaults(func=_cmd_catalog_list)

    p_show = catalog_subparsers.add_parser("show", help="Show one entry with details and relations")
    p_show.add_argument("key", help="Entry key, e.g. problem.temporal_neural_forecasting")
    p_show.add_argument("--profile", default="default", choices=("default", "framework-core"))
    p_show.set_defaults(func=_cmd_catalog_show)

    webui_cat = catalog_subparsers.add_parser("webui", help="Refresh DB and launch the catalog WebUI.")
    _add_webui_args(webui_cat)

    p_project = subparsers.add_parser("project", help="Project scaffold & local catalog")
    project_sub = p_project.add_subparsers(dest="project_command")

    p_init = project_sub.add_parser("init", help="Create a local project scaffold")
    p_init.add_argument("path", help="Target directory for the project")
    p_init.add_argument("--name", default="mlblack_project", help="Project name")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing files")
    p_init.set_defaults(func=_cmd_project_init)

    args = parser.parse_args(argv)
    if args.command == "webui":
        return _run_webui(args)
    if args.command == "catalog":
        if args.catalog_command == "webui":
            return _run_webui(args)
        func = getattr(args, "func", None)
        if func is not None:
            return func(args)
    if args.command == "project":
        func = getattr(args, "func", None)
        if func is not None:
            return func(args)
    parser.print_help()
    return 2


def _cmd_project_init(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .project import create_standard_scaffold

    try:
        result = create_standard_scaffold(
            Path(args.path),
            name=args.name,
            exist_ok=bool(args.force),
        )
    except FileExistsError:
        print(f"Directory already exists and is non-empty: {args.path}", file=sys.stderr)
        print("Use --force to overwrite.", file=sys.stderr)
        return 1
    print(f"Project created at: {result['root']}")
    print("Next:")
    print(f"  cd {args.path}")
    print("  python build_trainer.py")
    return 0


def _add_catalog_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default="default", choices=("default", "framework-core"), help="Catalog profile")
    parser.add_argument("--kind", default=None, help="Filter by component kind (adapter, problem, codec, etc.)")
    parser.add_argument("--tag", default=None, help="Filter by tag")
    parser.add_argument("--show-import", action="store_true", help="Print import_path column")
    parser.add_argument("--show-tags", action="store_true", help="Print tags")
    parser.add_argument("--no-summary", action="store_true", help="Suppress summary column")


def _add_webui_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default="default", help="Materialized catalog profile.")
    parser.add_argument("--db-path", default=None, help="SQLite path or PostgreSQL URL. Defaults to .mlblack/catalog.sqlite.")
    parser.add_argument("--host", default=None, help="Streamlit server address.")
    parser.add_argument("--port", type=int, default=8765, help="Streamlit server port.")
    parser.add_argument("--headless", action="store_true", help="Pass --server.headless true to Streamlit.")
    parser.add_argument("--no-refresh", action="store_true", help="Do not refresh/materialize the DB before launching.")
    parser.add_argument("--dry-run", action="store_true", help="Print the Streamlit command without launching.")


def _run_webui(args: argparse.Namespace) -> int:
    from .catalog.webui import build_catalog_webui_command, launch_catalog_webui

    if bool(args.dry_run):
        print(" ".join(build_catalog_webui_command(
            profile=args.profile,
            db_path=args.db_path,
            host=args.host,
            port=args.port,
            headless=args.headless,
        )))
        return 0
    return launch_catalog_webui(
        profile=args.profile,
        db_path=args.db_path,
        host=args.host,
        port=args.port,
        refresh=not args.no_refresh,
        headless=args.headless,
    )


def _cmd_catalog_search(args: argparse.Namespace) -> int:
    catalog = get_catalog(refresh=True)
    entries = catalog.search(args.query, kind=args.kind, limit=args.limit)
    if args.tag:
        tag_set = set(_parse_tags(args.tag))
        entries = tuple(e for e in entries if tag_set.issubset(set(e.tags)))
    entries = _apply_profile(entries, args.profile)
    print(f"Catalog search: {args.query!r}  (hits={len(entries)})")
    _print_entries(
        entries,
        show_import=args.show_import,
        show_tags=args.show_tags,
        show_summary=not args.no_summary,
    )
    print("Hint: `python -m mlblack catalog show <key>` for details/relations.")
    return 0


def _cmd_catalog_list(args: argparse.Namespace) -> int:
    catalog = get_catalog(refresh=True)
    entries = catalog.list(kind=args.kind)
    if args.tag:
        tag_set = set(_parse_tags(args.tag))
        entries = tuple(e for e in entries if tag_set.issubset(set(e.tags)))
    entries = _apply_profile(entries, args.profile)
    kind_label = f"kind={args.kind!r}" if args.kind else "all"
    print(f"Catalog list: {kind_label}  (count={len(entries)})")
    _print_entries(
        entries,
        show_import=args.show_import,
        show_tags=args.show_tags,
        show_summary=not args.no_summary,
    )
    print("Hint: `python -m mlblack catalog show <key>` for details/relations.")
    return 0


def _cmd_catalog_show(args: argparse.Namespace) -> int:
    catalog = get_catalog(refresh=True)
    try:
        entry = catalog.show(args.key)
    except KeyError:
        print(f"catalog: key not found: {args.key}", file=sys.stderr)
        return 1
    _print_entry_detail(entry, {})
    return 0


def _print_entries(
    entries,
    *,
    show_import: bool = False,
    show_tags: bool = False,
    show_summary: bool = True,
) -> None:
    entries_list = list(entries)
    if not entries_list:
        print("(no entries)")
        return
    key_w = min(max(len(str(e.key)) for e in entries_list), 50)
    kind_w = min(max(len(str(e.kind)) for e in entries_list), 18)
    header = f"{'KEY':<{key_w}}  {'KIND':<{kind_w}}  TITLE"
    print(header)
    print("-" * len(header))
    for e in entries_list:
        import_col = f"  {e.import_path}" if show_import else ""
        tag_col = f"  [{', '.join(e.tags)}]" if show_tags and e.tags else ""
        summary_col = f"\n{'':{key_w+2+kind_w+2}}{e.summary}" if show_summary and e.summary else ""
        print(f"{e.key:<{key_w}}  {e.kind:<{kind_w}}  {e.title}{import_col}{tag_col}{summary_col}")


def _print_entry_detail(entry: CatalogEntry, relations: dict) -> None:
    print(f"Key:         {entry.key}")
    print(f"Title:       {entry.title}")
    print(f"Kind:        {entry.kind}")
    print(f"Import:      {entry.import_path}")
    if entry.tags:
        print(f"Tags:        {', '.join(entry.tags)}")
    if entry.summary:
        print(f"Summary:     {entry.summary}")
    contract = dict(entry.contract)
    if contract:
        requires = contract.get("requires", ())
        provides = contract.get("provides", ())
        mutates = contract.get("mutates", ())
        optional = contract.get("optional", ())
        if requires:
            print(f"Requires:    {', '.join(requires)}")
        if optional:
            print(f"Optional:    {', '.join(optional)}")
        if provides:
            print(f"Provides:    {', '.join(provides)}")
        if mutates:
            print(f"Mutates:     {', '.join(mutates)}")
        for key in ("supports_gradient", "supports_batch", "supports_resume"):
            if key in contract:
                print(f"{key}:  {contract[key]}")
    neighbors = relations.get("neighbors", {})
    if neighbors:
        upstream = neighbors.get("context_upstream", ())
        downstream = neighbors.get("context_downstream", ())
        companions = neighbors.get("role_companions", ())
        if upstream:
            print(f"Upstream:    {', '.join(upstream)}")
        if downstream:
            print(f"Downstream:  {', '.join(downstream)}")
        if companions:
            print(f"Companions:  {', '.join(companions)}")
    flow = relations.get("flow", {})
    if flow:
        stage = flow.get("current_stage", "")
        if stage:
            print(f"Stage:       {stage}")


_NON_CORE_TAGS = {"example", "doc", "demo", "tutorial", "template"}


def _apply_profile(entries: tuple, profile: str) -> tuple:
    if profile != "framework-core":
        return entries
    return tuple(
        e for e in entries
        if not (set(e.tags) & _NON_CORE_TAGS)
    )


def _parse_tags(raw: str | None) -> tuple[str, ...]:

    if not raw:
        return tuple()
    return tuple(sorted({t.strip() for t in str(raw).split(",") if t.strip()}))


if __name__ == "__main__":
    raise SystemExit(main())
