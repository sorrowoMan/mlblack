from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Sequence

from mlblack.backends import get_backend, list_backends


DEFAULT_BACKEND_REQUIREMENTS: tuple[str, ...] = (
    "tensor.float_tensor",
    "neural.lowering.mlp",
    "autograd.functional.grad",
    "autograd.backward",
    "optimizer.sgd_step",
    "optimizer.step",
    "loss.mse",
    "loss.cross_entropy",
    "artifact.parameters.summary",
)


def backend_capability_matrix(
    requirements: Sequence[str] | None = None,
    *,
    backend_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    reqs = tuple(str(item) for item in (requirements or DEFAULT_BACKEND_REQUIREMENTS))
    names = tuple(str(name) for name in (backend_names or list_backends()))
    rows: list[dict[str, Any]] = []
    for name in names:
        backend = get_backend(name)
        contract = backend.contract()
        support = {requirement: contract.supports(requirement) for requirement in reqs}
        rows.append(
            {
                "backend": contract.name,
                "metadata": dict(contract.metadata),
                "supports": support,
                "provides_count": int(len(contract.provides)),
                "capabilities": tuple(capability.capability for capability in contract.capabilities),
            }
        )
    return {"requirements": reqs, "rows": tuple(rows)}


def render_backend_matrix_markdown(matrix: dict[str, Any] | None = None) -> str:
    payload = matrix or backend_capability_matrix()
    requirements = tuple(str(item) for item in payload.get("requirements", ()))
    lines = ["# mlblack backend capability matrix", ""]
    lines.append("| backend | " + " | ".join(f"`{item}`" for item in requirements) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in requirements) + " |")
    for row in tuple(payload.get("rows", ())):
        supports = dict(row.get("supports", {}))
        values = ["yes" if bool(supports.get(item, False)) else "no" for item in requirements]
        lines.append(f"| `{row.get('backend')}` | " + " | ".join(values) + " |")
    return "\n".join(lines)


def export_backend_matrix_html(path: str | Path, matrix: dict[str, Any] | None = None) -> Path:
    payload = matrix or backend_capability_matrix()
    requirements = tuple(str(item) for item in payload.get("requirements", ()))
    header = "".join(f"<th><code>{escape(item)}</code></th>" for item in requirements)
    rows = []
    for row in tuple(payload.get("rows", ())):
        supports = dict(row.get("supports", {}))
        cells = []
        for requirement in requirements:
            ok = bool(supports.get(requirement, False))
            cells.append(f"<td class='{'yes' if ok else 'no'}'>{'yes' if ok else 'no'}</td>")
        rows.append(f"<tr><td><code>{escape(str(row.get('backend')))}</code></td>{''.join(cells)}</tr>")
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>mlblack backend matrix</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;background:#f7f3ea;color:#241b13}}
table{{border-collapse:collapse;width:100%;background:white}}
td,th{{border:1px solid #ddd;padding:8px;vertical-align:top}}
th{{background:#342414;color:#fff}}
.yes{{background:#e4f5df;color:#215b1f;font-weight:700}}
.no{{background:#fae4dc;color:#7d2a14}}
code{{color:#7b3f00}}
</style></head>
<body><h1>mlblack backend capability matrix</h1>
<table><thead><tr><th>backend</th>{header}</tr></thead><tbody>
{''.join(rows)}
</tbody></table></body></html>"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


__all__ = [
    "DEFAULT_BACKEND_REQUIREMENTS",
    "backend_capability_matrix",
    "export_backend_matrix_html",
    "render_backend_matrix_markdown",
]
