from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in dict(value).items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    return str(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [dict(row) for row in rows]
    if not items:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in items:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(items)


def write_etf_quant_reports(
    *,
    output_dir: str | Path,
    summary: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
    interval_rows: Sequence[Mapping[str, Any]],
    rolling_rows: Sequence[Mapping[str, Any]],
    backtest_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    metric_path = out / "baseline_metrics.csv"
    interval_path = out / "interval_metrics.csv"
    rolling_path = out / "rolling_metrics.csv"
    backtest_path = out / "rank_backtest_metrics.csv"
    source_path = out / "orthogonal_source_table.csv"
    md_path = out / "report.md"
    summary_path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(metric_path, metric_rows)
    _write_csv(interval_path, interval_rows)
    _write_csv(rolling_path, rolling_rows)
    _write_csv(backtest_path, backtest_rows)
    _write_csv(source_path, source_rows)

    lines = [
        "# ETF Quant Interval Proxy Report",
        "",
        f"dataset: `{summary.get('dataset_label', '')}`",
        f"panel_rows: `{summary.get('panel_rows', '')}`",
        f"feature_count: `{summary.get('feature_count', '')}`",
        f"selected_source_count: `{summary.get('selected_source_count', '')}`",
        "",
        "## Baseline Metrics",
        "",
        "| feature_space | model | test_rmse | test_mae | test_r2 | direction_accuracy |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in metric_rows:
        lines.append(
            "| "
            + " | ".join(
                str(row.get(col, ""))
                for col in ("feature_space", "model", "test_rmse", "test_mae", "test_r2", "direction_accuracy")
            )
            + " |"
        )
    lines.extend([
        "",
        "## Interval Metrics",
        "",
        "| feature_space | model | coverage | avg_width | winkler_score |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for row in interval_rows:
        lines.append(
            "| "
            + " | ".join(
                str(row.get(col, ""))
                for col in ("feature_space", "model", "coverage", "avg_width", "winkler_score")
            )
            + " |"
        )
    lines.extend([
        "",
        "## Rank And Backtest Proxy Metrics",
        "",
        "| feature_space | model | mean_spearman_rank_ic | top1_mean_return | equal_weight_mean_return | top1_max_drawdown_proxy | turnover_proxy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in backtest_rows:
        lines.append(
            "| "
            + " | ".join(
                str(row.get(col, ""))
                for col in (
                    "feature_space",
                    "model",
                    "mean_spearman_rank_ic",
                    "top1_mean_return",
                    "equal_weight_mean_return",
                    "top1_max_drawdown_proxy",
                    "turnover_proxy",
                )
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "summary_json": str(summary_path),
        "baseline_metrics_csv": str(metric_path),
        "interval_metrics_csv": str(interval_path),
        "rolling_metrics_csv": str(rolling_path),
        "rank_backtest_metrics_csv": str(backtest_path),
        "orthogonal_source_table_csv": str(source_path),
        "report_md": str(md_path),
    }


__all__ = ["write_etf_quant_reports"]
