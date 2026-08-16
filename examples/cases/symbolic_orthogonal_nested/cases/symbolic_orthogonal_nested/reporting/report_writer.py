from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mlblack.catalog.artifacts import export_artifact_html


def write_case_report(
    *,
    output_dir: Path,
    summary: Mapping[str, Any],
    stage1_records: Sequence[Any],
    stage2_records: Sequence[Any],
    basis_artifact: Any,
    task_artifact: Any,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    stage1_path = output_dir / "stage1_records.json"
    stage2_path = output_dir / "stage2_records.json"
    basis_path = output_dir / "basis_artifact.json"
    task_path = output_dir / "task_artifact.json"
    basis_html_path = output_dir / "basis_artifact.html"
    task_html_path = output_dir / "task_artifact.html"
    dashboard_path = output_dir / "artifact_dashboard.html"

    _write_json(summary_path, dict(summary))
    _write_json(stage1_path, [_record_dict(item) for item in stage1_records])
    _write_json(stage2_path, [_record_dict(item) for item in stage2_records])
    basis_payload = basis_artifact.describe() if hasattr(basis_artifact, "describe") else basis_artifact
    task_payload = task_artifact.describe() if hasattr(task_artifact, "describe") else task_artifact
    _write_json(basis_path, basis_payload)
    _write_json(task_path, task_payload)
    export_artifact_html(basis_html_path, basis_payload)
    export_artifact_html(task_html_path, task_payload)
    _write_dashboard(dashboard_path, summary=dict(summary), basis_html=basis_html_path.name, task_html=task_html_path.name)
    return {
        "summary": str(summary_path),
        "stage1_records": str(stage1_path),
        "stage2_records": str(stage2_path),
        "basis_artifact": str(basis_path),
        "task_artifact": str(task_path),
        "basis_artifact_html": str(basis_html_path),
        "task_artifact_html": str(task_html_path),
        "artifact_dashboard": str(dashboard_path),
    }


def _record_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "as_dict"):
        return dict(value.as_dict())
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": str(value)}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_dashboard(path: Path, *, summary: Mapping[str, Any], basis_html: str, task_html: str) -> None:
    stage2 = dict(summary.get("stage2", {}) or {})
    best_record = dict(stage2.get("best_record", {}) or {})
    score = dict(best_record.get("report", {}) or {}).get("candidate_score", {})
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>symbolic orthogonal nested artifacts</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;background:#f5f1e8;color:#1b1a17}}
.card{{background:white;border:1px solid #ded3c2;border-radius:12px;padding:18px;margin:14px 0}}
a{{color:#934719;font-weight:600}} pre{{white-space:pre-wrap;background:#1e1b17;color:#fff3df;padding:12px;border-radius:8px}}
</style></head><body>
<h1>symbolic orthogonal nested artifacts</h1>
<div class="card"><p><a href="{basis_html}">Open Stage 1 basis artifact viewer</a></p><p><a href="{task_html}">Open Stage 2 task artifact viewer</a></p></div>
<div class="card"><h2>Candidate Score</h2><pre>{json.dumps(score, ensure_ascii=False, indent=2, default=str)}</pre></div>
</body></html>"""
    path.write_text(html, encoding="utf-8")
