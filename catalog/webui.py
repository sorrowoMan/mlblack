from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .dashboard import build_streamlit_command, launch_catalog_dashboard
from .store import materialize_catalog_db, resolve_catalog_db_path


def launch_catalog_webui(
    *,
    profile: str = "default",
    db_path: str | Path | None = None,
    host: str | None = None,
    port: int = 8765,
    refresh: bool = True,
    headless: bool = False,
    script_path: str | Path | None = None,
) -> int:
    target_db = str(resolve_catalog_db_path(db_path))
    if refresh:
        result = materialize_catalog_db(target_db, profile=profile, refresh=True)
        print(
            "mlblack catalog DB refreshed: "
            f"{result.get('entries', 0)} entries -> {result.get('db_path', target_db)}"
        )
    return launch_catalog_dashboard(
        script_path=script_path or Path(__file__).resolve(),
        profile=profile,
        db_path=target_db,
        host=host,
        port=port,
        headless=headless,
    )


def build_catalog_webui_command(
    *,
    profile: str = "default",
    db_path: str | Path | None = None,
    host: str | None = None,
    port: int = 8765,
    headless: bool = False,
    script_path: str | Path | None = None,
) -> list[str]:
    return build_streamlit_command(
        script_path=script_path or Path(__file__).resolve(),
        profile=profile,
        db_path=str(resolve_catalog_db_path(db_path)),
        host=host,
        port=port,
        headless=headless,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dry_run:
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the mlblack catalog WebUI.")
    parser.add_argument("--profile", default="default", help="Materialized catalog profile.")
    parser.add_argument("--db-path", default=None, help="SQLite path or PostgreSQL URL. Defaults to .mlblack/catalog.sqlite.")
    parser.add_argument("--host", default=None, help="Streamlit server address.")
    parser.add_argument("--port", type=int, default=8765, help="Streamlit server port.")
    parser.add_argument("--headless", action="store_true", help="Pass --server.headless true to Streamlit.")
    parser.add_argument("--no-refresh", action="store_true", help="Do not refresh/materialize the DB before launching.")
    parser.add_argument("--dry-run", action="store_true", help="Print the Streamlit command without launching.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
