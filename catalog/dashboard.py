from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Iterable, Mapping

from .registry import Catalog, CatalogEntry, get_catalog


def catalog_summary(catalog: Catalog | None = None) -> dict[str, Any]:
    cat = catalog or get_catalog()
    entries = cat.list()
    by_kind: dict[str, int] = {}
    for item in entries:
        by_kind[item.kind] = by_kind.get(item.kind, 0) + 1
    return {"total": len(entries), "by_kind": by_kind}


def render_catalog_markdown(catalog: Catalog | None = None) -> str:
    cat = catalog or get_catalog()
    lines = ["# mlblack catalog", ""]
    for item in cat.list():
        tags = ", ".join(str(tag) for tag in item.tags)
        lines.append(f"- `{item.key}` ({item.kind}) - {item.title}; tags: {tags}")
    return "\n".join(lines)


def export_catalog_html(path: str | Path, catalog: Catalog | None = None) -> Path:
    cat = catalog or get_catalog()
    rows = []
    for item in cat.list():
        rows.append(
            "<tr>"
            f"<td><code>{escape(item.key)}</code></td>"
            f"<td>{escape(item.kind)}</td>"
            f"<td>{escape(item.title)}</td>"
            f"<td>{escape(', '.join(str(tag) for tag in item.tags))}</td>"
            f"<td>{escape(item.summary)}</td>"
            "</tr>"
        )
    html = """
<!doctype html>
<html><head><meta charset="utf-8"><title>mlblack catalog</title>
<style>body{font-family:Segoe UI,Arial,sans-serif;margin:32px;background:#f7f3ea;color:#241b13}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #ddd;padding:8px;vertical-align:top}th{background:#342414;color:#fff}code{color:#7b3f00}</style>
</head><body><h1>mlblack catalog</h1><table><thead><tr><th>key</th><th>kind</th><th>title</th><th>tags</th><th>summary</th></tr></thead><tbody>
""" + "\n".join(rows) + "\n</tbody></table></body></html>"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return target


def build_streamlit_command(
    *,
    script_path: str | Path,
    profile: str = "default",
    db_path: str | Path | None = None,
    host: str | None = None,
    port: int = 8765,
    headless: bool = False,
) -> list[str]:
    """Build the Streamlit command used by the catalog WebUI launcher."""

    import sys

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script_path),
        "--server.port",
        str(int(port)),
    ]
    if host:
        cmd.extend(["--server.address", str(host)])
    if headless:
        cmd.extend(["--server.headless", "true"])
    cmd.extend(["--", "--profile", str(profile)])
    if db_path is not None:
        cmd.extend(["--db-path", str(db_path)])
    cmd.append("--no-refresh")
    return cmd


def launch_catalog_dashboard(
    *,
    script_path: str | Path,
    profile: str = "default",
    db_path: str | Path | None = None,
    host: str | None = None,
    port: int = 8765,
    headless: bool = False,
) -> int:
    """Launch the Streamlit catalog dashboard."""

    import subprocess

    cmd = build_streamlit_command(
        script_path=script_path,
        profile=profile,
        db_path=db_path,
        host=host,
        port=port,
        headless=headless,
    )
    return int(subprocess.run(cmd, check=False).returncode)
