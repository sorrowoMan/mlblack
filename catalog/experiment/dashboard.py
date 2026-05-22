from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mlblack.capabilities.tracking import ExperimentRecord, InMemoryExperimentStore, SQLiteExperimentStore


def summarize_records(records: Sequence[ExperimentRecord]) -> dict[str, Any]:
    by_event: dict[str, int] = {}
    by_run: dict[str, int] = {}
    for record in records:
        by_event[record.event] = by_event.get(record.event, 0) + 1
        by_run[record.run_name] = by_run.get(record.run_name, 0) + 1
    return {"total": len(records), "by_event": by_event, "by_run": by_run}


def load_records(path: str | Path | None = None, *, run_name: str | None = None) -> tuple[ExperimentRecord, ...]:
    if path is None:
        return tuple()
    return SQLiteExperimentStore(path).list(run_name=run_name)


def render_experiment_markdown(records: Sequence[ExperimentRecord]) -> str:
    summary = summarize_records(records)
    lines = ["# mlblack experiments", "", f"total records: {summary['total']}", ""]
    for record in records:
        lines.append(f"- `{record.run_name}` step={record.step} event={record.event} id={record.record_id}")
    return "\n".join(lines)


def export_experiment_html(path: str | Path, records: Sequence[ExperimentRecord]) -> Path:
    rows = []
    for record in records:
        rows.append(
            "<tr>"
            f"<td><code>{escape(record.record_id)}</code></td>"
            f"<td>{escape(record.run_name)}</td>"
            f"<td>{'' if record.step is None else int(record.step)}</td>"
            f"<td>{escape(record.event)}</td>"
            f"<td><pre>{escape(str(dict(record.payload)))}</pre></td>"
            "</tr>"
        )
    html = """
<!doctype html>
<html><head><meta charset="utf-8"><title>mlblack experiments</title>
<style>body{font-family:Segoe UI,Arial,sans-serif;margin:32px;background:#eef4f2;color:#10231d}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #ddd;padding:8px;vertical-align:top}th{background:#15372d;color:#fff}pre{white-space:pre-wrap;margin:0}</style>
</head><body><h1>mlblack experiments</h1><table><thead><tr><th>record</th><th>run</th><th>step</th><th>event</th><th>payload</th></tr></thead><tbody>
""" + "\n".join(rows) + "\n</tbody></table></body></html>"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return target
