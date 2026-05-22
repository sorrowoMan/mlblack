from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any, Mapping

from .artifacts import ArtifactBundle, ModelArtifact


def render_artifact_html(artifact: ArtifactBundle | ModelArtifact | Mapping[str, Any], *, title: str = "mlblack artifact") -> str:
    """Render a compact local HTML artifact viewer."""

    payload = _describe_artifact(artifact)
    body = _render_mapping(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --ink: #17211b;
      --muted: #5d6b62;
      --card: #fffaf0;
      --line: #d8c7aa;
      --accent: #2f6f5e;
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top left, #fff4cf 0, transparent 28rem), var(--bg);
      color: var(--ink);
      font-family: Georgia, 'Times New Roman', serif;
    }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 40px 24px; }}
    h1 {{ font-size: 32px; letter-spacing: -0.03em; margin: 0 0 8px; }}
    .subtitle {{ color: var(--muted); margin-bottom: 28px; }}
    section {{
      background: color-mix(in srgb, var(--card) 92%, white);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px 20px;
      margin: 16px 0;
      box-shadow: 0 16px 44px rgba(58, 45, 28, 0.08);
    }}
    h2 {{ font-size: 20px; margin: 0 0 12px; color: var(--accent); }}
    table {{ width: 100%; border-collapse: collapse; }}
    td {{ border-top: 1px solid var(--line); padding: 8px 6px; vertical-align: top; }}
    td:first-child {{ width: 260px; color: var(--muted); font-weight: 700; }}
    code, pre {{ font-family: 'Cascadia Code', 'SFMono-Regular', Consolas, monospace; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #1d2721; color: #f8f0dd; padding: 12px; border-radius: 12px; }}
    .pill {{ display: inline-block; padding: 4px 8px; border: 1px solid var(--line); border-radius: 999px; margin: 2px; color: var(--muted); }}
  </style>
</head>
<body>
<main>
  <h1>{escape(title)}</h1>
  <div class="subtitle">Local artifact view. Payload is rendered from typed artifact metadata, not from a live trainer.</div>
  {body}
</main>
</body>
</html>
"""


def save_artifact_html(
    artifact: ArtifactBundle | ModelArtifact | Mapping[str, Any],
    path: str | Path,
    *,
    title: str = "mlblack artifact",
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_artifact_html(artifact, title=title), encoding="utf-8")
    return output


def _describe_artifact(artifact: ArtifactBundle | ModelArtifact | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(artifact, ArtifactBundle):
        return artifact.describe()
    if isinstance(artifact, ModelArtifact):
        return artifact.describe()
    return dict(artifact)


def _render_mapping(payload: Mapping[str, Any]) -> str:
    sections: list[str] = []
    for key, value in payload.items():
        sections.append(f"<section><h2>{escape(str(key))}</h2>{_render_value(value)}</section>")
    return "\n".join(sections)


def _render_value(value: Any) -> str:
    if isinstance(value, Mapping):
        rows = []
        for key, item in value.items():
            rows.append(f"<tr><td>{escape(str(key))}</td><td>{_render_value(item)}</td></tr>")
        return f"<table>{''.join(rows)}</table>"
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
            return " ".join(f"<span class='pill'>{escape(str(item))}</span>" for item in value)
        return "<pre>" + escape(json.dumps(_json_safe(value), ensure_ascii=False, indent=2)) + "</pre>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return escape(str(value))
    return "<pre>" + escape(json.dumps(_json_safe(value), ensure_ascii=False, indent=2)) + "</pre>"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


__all__ = ["render_artifact_html", "save_artifact_html"]
