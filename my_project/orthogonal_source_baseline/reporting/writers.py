from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(raw) for key, raw in dict(value).items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return str(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [dict(row) for row in rows]
    if not items:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in items:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)


def _write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "scenario",
        "feature_space",
        "model",
        "feature_count",
        "test_rmse",
        "test_r2",
        "rmse_delta_vs_raw",
        "selected_source_count",
        "mean_source_stability",
        "pair_abs_corr_max",
    ]
    lines = [
        "# Orthogonal Source Baseline Table",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).strip() for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_neural_training_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "scenario",
        "feature_space",
        "model",
        "trainer_key",
        "pipeline_key",
        "target_scaled",
        "n_iter",
        "max_iter",
        "early_stopping",
        "stopped_by",
        "reached_max_iter",
        "final_loss",
        "best_validation_score",
        "convergence_warning_count",
    ]
    lines = [
        "# Neural Training Report",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).strip() for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_suite_outputs(
    *,
    output_dir: Path,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    neural_training_rows: Sequence[Mapping[str, Any]] = (),
    neural_curve_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    table_csv_path = output_dir / "baseline_table.csv"
    table_md_path = output_dir / "baseline_table.md"
    sources_csv_path = output_dir / "orthogonal_source_table.csv"
    sources_json_path = output_dir / "orthogonal_source_table.json"
    neural_training_csv_path = output_dir / "neural_training_report.csv"
    neural_training_md_path = output_dir / "neural_training_report.md"
    neural_training_json_path = output_dir / "neural_training_report.json"
    neural_curve_csv_path = output_dir / "neural_training_curve.csv"
    neural_curve_json_path = output_dir / "neural_training_curve.json"
    summary_path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(table_csv_path, rows)
    _write_markdown(table_md_path, rows)
    _write_csv(sources_csv_path, source_rows)
    sources_json_path.write_text(json.dumps(_jsonable({"rows": list(source_rows)}), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(neural_training_csv_path, neural_training_rows)
    _write_neural_training_markdown(neural_training_md_path, neural_training_rows)
    neural_training_json_path.write_text(
        json.dumps(_jsonable({"rows": list(neural_training_rows)}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(neural_curve_csv_path, neural_curve_rows)
    neural_curve_json_path.write_text(
        json.dumps(_jsonable({"rows": list(neural_curve_rows)}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "summary_json": str(summary_path),
        "baseline_table_csv": str(table_csv_path),
        "baseline_table_md": str(table_md_path),
        "source_table_csv": str(sources_csv_path),
        "source_table_json": str(sources_json_path),
        "neural_training_report_csv": str(neural_training_csv_path),
        "neural_training_report_md": str(neural_training_md_path),
        "neural_training_report_json": str(neural_training_json_path),
        "neural_training_curve_csv": str(neural_curve_csv_path),
        "neural_training_curve_json": str(neural_curve_json_path),
    }


__all__ = ["write_suite_outputs"]
