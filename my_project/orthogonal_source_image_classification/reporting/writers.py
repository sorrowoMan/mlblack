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
        "test_accuracy",
        "test_macro_f1",
        "accuracy_delta_vs_raw_pixels",
        "accuracy_delta_vs_image_representation",
        "selected_source_count",
        "mean_source_stability",
        "pair_abs_corr_max",
    ]
    lines = [
        "# Orthogonal Source Image Classification Table",
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
    representation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    table_csv_path = output_dir / "classification_table.csv"
    table_md_path = output_dir / "classification_table.md"
    representation_csv_path = output_dir / "representation_formula_table.csv"
    representation_json_path = output_dir / "representation_formula_table.json"
    sources_csv_path = output_dir / "orthogonal_source_table.csv"
    sources_json_path = output_dir / "orthogonal_source_table.json"
    summary_path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(table_csv_path, rows)
    _write_markdown(table_md_path, rows)
    _write_csv(representation_csv_path, representation_rows)
    representation_json_path.write_text(
        json.dumps(_jsonable({"rows": list(representation_rows)}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(sources_csv_path, source_rows)
    sources_json_path.write_text(json.dumps(_jsonable({"rows": list(source_rows)}), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "summary_json": str(summary_path),
        "classification_table_csv": str(table_csv_path),
        "classification_table_md": str(table_md_path),
        "representation_formula_table_csv": str(representation_csv_path),
        "representation_formula_table_json": str(representation_json_path),
        "source_table_csv": str(sources_csv_path),
        "source_table_json": str(sources_json_path),
    }


__all__ = ["write_suite_outputs"]
