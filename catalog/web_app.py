from __future__ import annotations

import json
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

from .dashboard import catalog_summary
from .query import CatalogQuery, query_catalog
from .store import resolve_catalog_db_path


def catalog_web_payload(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return the JSON payload used by the lightweight DB-only catalog web UI."""

    query = CatalogQuery(
        kind=_param(params, "kind") or None,
        query=_param(params, "q"),
        tags=tuple(_split_csv(_param(params, "tags"))),
        fields=_field_filters(params),
        limit=int(_param(params, "limit") or 100),
        profile=_param(params, "profile") or "default",
        source="db",
        db_path=_param(params, "db_path") or None,
    )
    return query_catalog(query).as_dict()


def catalog_summary_payload(params: Mapping[str, Any]) -> dict[str, Any]:
    return catalog_summary(
        db_path=_param(params, "db_path") or None,
        source="db",
        profile=_param(params, "profile") or "default",
    )


def render_catalog_web_page(
    *,
    default_source: str = "db",
    default_profile: str = "default",
    default_db_path: str = "",
) -> str:
    profile = escape(default_profile)
    db_path = escape(default_db_path or str(resolve_catalog_db_path()))
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>mlblack Catalog 控制台</title>
  <style>
    :root {{ --ink:#222c17; --paper:#f3f2e8; --card:#fffdfa; --line:#d5d8c5; --accent:#53622d; --dark:#26311c; }}
    body {{ margin:30px; font-family:Segoe UI,Arial,sans-serif; background:var(--paper); color:var(--ink); }}
    h1 {{ margin:0 0 8px; font-size:30px; }}
    .sub {{ margin:0 0 18px; color:#566047; }}
    .hero {{ background:linear-gradient(135deg,#eef0e4,#d9d7bd 48%,#a9b184); border:1px solid rgba(64,76,36,.16); border-radius:22px; padding:20px 22px; margin-bottom:18px; }}
    .bar {{ display:grid; grid-template-columns:1.4fr .8fr 1fr 1.8fr 130px; gap:10px; margin-bottom:16px; }}
    input, button {{ padding:9px 10px; border:1px solid var(--line); border-radius:9px; background:white; }}
    button {{ background:var(--dark); color:white; cursor:pointer; font-weight:700; }}
    table {{ border-collapse:collapse; width:100%; background:var(--card); }}
    td, th {{ border:1px solid var(--line); padding:8px; vertical-align:top; }}
    th {{ background:var(--dark); color:white; position:sticky; top:0; }}
    code {{ color:var(--accent); }}
    .meta {{ margin:10px 0 16px; color:#566047; }}
    .badge {{ display:inline-block; padding:3px 8px; border-radius:999px; background:rgba(64,76,36,.1); color:#36431f; font-size:12px; font-weight:700; }}
  </style>
</head>
<body>
  <div class="hero">
    <div class="badge">DB-only</div>
    <h1>mlblack Catalog 控制台</h1>
    <p class="sub">轻量查询页只读 materialized DB，不再提供 registry/source-mode 回退。默认数据库固定为仓库根目录 <code>.mlblack/catalog.sqlite</code>。</p>
  </div>
  <div class="bar">
    <input id="q" placeholder="搜索 key/title/summary/tags" />
    <input id="kind" placeholder="分类，如 adapter" />
    <input id="tags" placeholder="标签 CSV" />
    <input id="db_path" placeholder="DB path or postgresql://..." value="{db_path}" />
    <button onclick="runSearch()">查询</button>
  </div>
  <input id="profile" value="{profile}" hidden />
  <div class="meta" id="meta">Ready.</div>
  <table>
    <thead><tr><th>key</th><th>分类</th><th>标题</th><th>标签</th><th>摘要</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
<script>
async function runSearch() {{
  const params = new URLSearchParams();
  for (const id of ['q','kind','tags','db_path','profile']) {{
    const value = document.getElementById(id).value;
    if (value) params.set(id, value);
  }}
  params.set('limit', '200');
  const response = await fetch('/api/catalog?' + params.toString());
  const payload = await response.json();
  if (!response.ok) {{
    document.getElementById('meta').textContent = payload.error || 'request failed';
    return;
  }}
  const rows = payload.entries || [];
  document.getElementById('meta').textContent = `${{rows.length}} entries, source=db, profile=${{payload.query.profile}}`;
  document.getElementById('rows').innerHTML = rows.map(item => `
    <tr>
      <td><code>${{escapeHtml(item.key)}}</code></td>
      <td>${{escapeHtml(item.kind)}}</td>
      <td>${{escapeHtml(item.title)}}</td>
      <td>${{escapeHtml((item.tags || []).join(', '))}}</td>
      <td>${{escapeHtml(item.summary || '')}}</td>
    </tr>`).join('');
}}
function escapeHtml(value) {{
  return String(value).replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}}[ch]));
}}
runSearch();
</script>
</body>
</html>"""


def serve_catalog_web(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    source: str = "db",
    profile: str = "default",
    db_path: str | Path | None = None,
) -> ThreadingHTTPServer:
    defaults = {
        "source": "db",
        "profile": str(profile),
        "db_path": str(db_path or resolve_catalog_db_path()),
    }

    class CatalogHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            params = _parse_params(parsed.query)
            params.setdefault("source", defaults["source"])
            params.setdefault("profile", defaults["profile"])
            params.setdefault("db_path", defaults["db_path"])
            if parsed.path in {"", "/"}:
                self._send_html(
                    render_catalog_web_page(
                        default_source=defaults["source"],
                        default_profile=defaults["profile"],
                        default_db_path=defaults["db_path"],
                    )
                )
                return
            if parsed.path == "/api/catalog":
                self._send_json(lambda: catalog_web_payload(params))
                return
            if parsed.path == "/api/summary":
                self._send_json(lambda: catalog_summary_payload(params))
                return
            self.send_error(404, "not found")

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib name
            return

        def _send_html(self, html: str) -> None:
            payload = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, producer: Any) -> None:
            try:
                payload_obj = producer()
                status = 200
            except Exception as exc:
                payload_obj = {"error": str(exc)}
                status = 500
            payload = json.dumps(payload_obj, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer((host, int(port)), CatalogHandler)
    return server


def run_catalog_web(**kwargs: Any) -> None:
    server = serve_catalog_web(**kwargs)
    host, port = server.server_address[:2]
    print(f"mlblack catalog web: http://{host}:{port}")
    server.serve_forever()


def _parse_params(query: str) -> dict[str, str]:
    parsed = parse_qs(query, keep_blank_values=False)
    return {key: values[-1] for key, values in parsed.items() if values}


def _param(params: Mapping[str, Any], key: str) -> str:
    value = params.get(key, "")
    if isinstance(value, (list, tuple)):
        return str(value[-1]) if value else ""
    return str(value or "")


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value or "").split(",") if part.strip())


def _field_filters(params: Mapping[str, Any]) -> dict[str, str]:
    prefix = "field."
    return {
        str(key)[len(prefix):]: _param(params, str(key))
        for key in params.keys()
        if str(key).startswith(prefix) and _param(params, str(key))
    }


__all__ = [
    "catalog_summary_payload",
    "catalog_web_payload",
    "render_catalog_web_page",
    "run_catalog_web",
    "serve_catalog_web",
]
