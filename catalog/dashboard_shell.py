from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .dashboard_shared import DEFAULT_KIND, DEFAULT_PAGE_SIZE, DEFAULT_PROFILE, DEFAULT_SOURCE


def build_streamlit_command(
    *,
    script_path: str | Path,
    source: str = DEFAULT_SOURCE,
    profile: str = DEFAULT_PROFILE,
    kind: str = DEFAULT_KIND,
    query: str = "",
    tags: str = "",
    db_path: str | None = None,
    host: str | None = None,
    port: int | None = None,
    headless: bool = False,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> list[str]:
    command = [sys.executable, "-m", "streamlit", "run", str(Path(script_path).resolve())]
    if host:
        command.extend(["--server.address", str(host)])
    if port is not None:
        command.extend(["--server.port", str(int(port))])
    if headless:
        command.extend(["--server.headless", "true"])
    command.extend(
        [
            "--",
            "--source",
            str(source),
            "--profile",
            str(profile),
            "--kind",
            str(kind),
            "--page-size",
            str(int(page_size)),
        ]
    )
    if query:
        command.extend(["--query", str(query)])
    if tags:
        command.extend(["--tags", str(tags)])
    if db_path:
        command.extend(["--db-path", str(db_path)])
    return command


def launch_catalog_dashboard(
    *,
    script_path: str | Path,
    source: str = DEFAULT_SOURCE,
    profile: str = DEFAULT_PROFILE,
    kind: str = DEFAULT_KIND,
    query: str = "",
    tags: str = "",
    db_path: str | None = None,
    host: str | None = None,
    port: int | None = None,
    headless: bool = False,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> int:
    return int(
        subprocess.call(
            build_streamlit_command(
                script_path=script_path,
                source=source,
                profile=profile,
                kind=kind,
                query=query,
                tags=tags,
                db_path=db_path,
                host=host,
                port=port,
                headless=headless,
                page_size=page_size,
            )
        )
    )

