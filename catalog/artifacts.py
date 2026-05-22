from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Mapping


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load a JSON artifact payload from disk."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {"value": payload}


def render_artifact_markdown(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    schema = dict(payload.get("schema", payload) or {})
    lines = [
        "# mlblack artifact",
        "",
        f"- artifact_id: `{_field(payload, schema, 'artifact_id')}`",
        f"- artifact_type: `{_field(payload, schema, 'artifact_type')}`",
        f"- schema_key: `{schema.get('schema_key', '')}`",
        f"- stage: `{_stage_name(schema)}`",
        "",
        "## Expressions",
        "",
    ]
    final_expr = schema.get("final_expression", payload.get("final_expression", {}))
    canonical = schema.get("canonical_expression", payload.get("canonical_expression", {}))
    lines.append("final_expression:")
    lines.append("```json")
    lines.append(json.dumps(final_expr, ensure_ascii=False, indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append("canonical_expression:")
    lines.append("```json")
    lines.append(json.dumps(canonical, ensure_ascii=False, indent=2, default=str))
    lines.append("```")
    return "\n".join(lines)


def export_artifact_html(path: str | Path, artifact: Mapping[str, Any] | str | Path) -> Path:
    """Export a readable, static HTML artifact viewer."""

    payload = load_artifact(artifact) if isinstance(artifact, (str, Path)) else dict(artifact)
    schema = dict(payload.get("schema", payload) or {})
    title = _field(payload, schema, "artifact_id") or _field(payload, schema, "name") or "mlblack artifact"
    sections = [
        _section("Summary", _summary_table(payload, schema)),
        _section("Final Expression", _json_block(schema.get("final_expression", payload.get("final_expression", {})))),
        _section("Canonical Expression", _json_block(schema.get("canonical_expression", payload.get("canonical_expression", {})))),
        _section("Truth Recovery", _json_block(schema.get("truth_contract_recovery", {}))),
        _section("Family Recovery", _json_block(schema.get("family_recovery", {}))),
        _section("Phase Equivalence Recovery", _json_block(schema.get("phase_equivalence_recovery", {}))),
        _section("Candidate Lineage", _json_block(schema.get("candidate_lineage", payload.get("candidate_lineage", {})))),
        _section("Evaluation Report", _json_block(schema.get("evaluation_report", {}))),
        _section("Branch Report", _json_block(schema.get("branch_report", {}))),
        _section("Raw Payload", _json_block(payload)),
    ]
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{escape(str(title))}</title>
  <style>
    :root {{ --ink:#1c1a16; --muted:#75695a; --paper:#fbf7ef; --card:#fffdf8; --line:#dfd3c1; --accent:#a24f21; }}
    body {{ margin:32px; font-family:Georgia, 'Times New Roman', serif; background:linear-gradient(135deg,#fbf7ef,#f0e7d7); color:var(--ink); }}
    h1 {{ font-size:34px; margin:0 0 8px; }}
    h2 {{ font-size:20px; margin:28px 0 10px; }}
    .subtitle {{ color:var(--muted); margin-bottom:24px; }}
    section {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px; margin:16px 0; box-shadow:0 8px 24px rgba(50,35,12,.06); }}
    table {{ border-collapse:collapse; width:100%; }}
    td, th {{ border-bottom:1px solid var(--line); padding:8px; text-align:left; vertical-align:top; }}
    th {{ color:var(--muted); width:220px; }}
    code {{ color:var(--accent); }}
    pre {{ white-space:pre-wrap; overflow:auto; background:#1e1b17; color:#f9eedc; border-radius:10px; padding:14px; }}
  </style>
</head>
<body>
  <h1>mlblack Artifact Viewer</h1>
  <div class="subtitle">{escape(str(title))}</div>
  {''.join(sections)}
</body>
</html>"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return target


def _field(payload: Mapping[str, Any], schema: Mapping[str, Any], key: str) -> str:
    return str(schema.get(key, payload.get(key, "")) or "")


def _stage_name(schema: Mapping[str, Any]) -> str:
    stage = schema.get("stage", {})
    return str(dict(stage).get("name", "")) if isinstance(stage, Mapping) else str(stage or "")


def _summary_table(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> str:
    rows = {
        "name": _field(payload, schema, "name"),
        "artifact_id": _field(payload, schema, "artifact_id"),
        "artifact_type": _field(payload, schema, "artifact_type"),
        "schema_key": str(schema.get("schema_key", "")),
        "schema_version": str(schema.get("schema_version", "")),
        "stage": _stage_name(schema),
        "created_at": str(schema.get("created_at", payload.get("created_at", ""))),
    }
    body = "".join(f"<tr><th>{escape(key)}</th><td><code>{escape(value)}</code></td></tr>" for key, value in rows.items())
    return f"<table>{body}</table>"


def _json_block(value: Any) -> str:
    return f"<pre>{escape(json.dumps(value, ensure_ascii=False, indent=2, default=str))}</pre>"


def _section(title: str, body: str) -> str:
    return f"<section><h2>{escape(title)}</h2>{body}</section>"


__all__ = ["export_artifact_html", "load_artifact", "render_artifact_markdown"]
