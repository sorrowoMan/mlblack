from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from my_project.orthogonal_source_baseline.config import OrthogonalSourceBaselineConfig
from my_project.orthogonal_source_baseline.pipeline import build_orthogonal_sources, fit_baseline_models
from my_project.orthogonal_source_baseline.pipeline.baselines import summarize_feature_space_winners
from my_project.orthogonal_source_baseline.problem import load_scenario_dataset
from my_project.orthogonal_source_baseline.reporting import write_suite_outputs


def _with_raw_delta(rows: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    raw_by_model = {
        str(row.get("model")): float(row.get("test_rmse"))
        for row in rows
        if str(row.get("feature_space")) == "raw_features"
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        raw_rmse = raw_by_model.get(str(item.get("model")))
        item["rmse_delta_vs_raw"] = "" if raw_rmse is None else float(float(item.get("test_rmse")) - raw_rmse)
        out.append(item)
    return tuple(out)


def run_one_scenario(
    *,
    benchmark_key: str,
    cfg: OrthogonalSourceBaselineConfig,
) -> dict[str, Any]:
    dataset = load_scenario_dataset(
        scenario_key=str(benchmark_key),
        n_total=int(cfg.n_total),
        train_ratio=float(cfg.train_ratio),
        noise_std=float(cfg.noise_std),
        seed=int(cfg.seed),
        max_rows=int(cfg.max_rows),
    )
    source_result = build_orthogonal_sources(dataset, cfg)
    baseline_result = fit_baseline_models(
        raw_train=dataset.X_train,
        raw_test=dataset.X_test,
        basis_train=source_result.train_basis,
        basis_test=source_result.test_basis,
        y_train=dataset.y_train,
        y_test=dataset.y_test,
        seed=int(cfg.seed),
    )
    metric_rows = _with_raw_delta(tuple(dict(row) for row in baseline_result.rows))
    report = dict(source_result.report)
    rows = []
    for row in metric_rows:
        item = dict(row)
        item["scenario"] = str(benchmark_key)
        item["selected_source_count"] = int(report.get("selected_source_count", 0))
        item["mean_source_stability"] = float(report.get("mean_source_stability", 0.0))
        item["pair_abs_corr_max"] = float(report.get("pair_abs_corr_max", 0.0))
        rows.append(item)
    source_rows = []
    for row in source_result.source_rows:
        item = dict(row)
        item["scenario"] = str(benchmark_key)
        source_rows.append(item)
    neural_training_rows = []
    for row in baseline_result.neural_training_rows:
        item = dict(row)
        item["scenario"] = str(benchmark_key)
        neural_training_rows.append(item)
    neural_curve_rows = []
    for row in baseline_result.neural_curve_rows:
        item = dict(row)
        item["scenario"] = str(benchmark_key)
        neural_curve_rows.append(item)
    return {
        "scenario": str(benchmark_key),
        "rows": rows,
        "source_rows": source_rows,
        "neural_training_rows": neural_training_rows,
        "neural_curve_rows": neural_curve_rows,
        "source_report": report,
        "winners": summarize_feature_space_winners(tuple(rows)),
    }


def run_suite(
    cfg: OrthogonalSourceBaselineConfig,
    *,
    suite_id: str | None = None,
    benchmark_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    resolved_suite_id = str(suite_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    keys = tuple(str(key) for key in (tuple(benchmark_keys) if benchmark_keys is not None else tuple(cfg.benchmark_keys)))
    output_dir = Path(cfg.output_dir).expanduser().resolve() / resolved_suite_id
    scenario_reports = []
    metric_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    neural_training_rows: list[dict[str, Any]] = []
    neural_curve_rows: list[dict[str, Any]] = []
    for key in keys:
        report = run_one_scenario(benchmark_key=key, cfg=cfg)
        scenario_reports.append(report)
        metric_rows.extend(tuple(report["rows"]))
        source_rows.extend(tuple(report["source_rows"]))
        neural_training_rows.extend(tuple(report.get("neural_training_rows", ())))
        neural_curve_rows.extend(tuple(report.get("neural_curve_rows", ())))

    summary = {
        "protocol": "orthogonal_source_to_strong_baseline_v1",
        "suite_id": resolved_suite_id,
        "config": {
            "benchmark_keys": keys,
            "n_total": int(cfg.n_total),
            "train_ratio": float(cfg.train_ratio),
            "noise_std": float(cfg.noise_std),
            "seed": int(cfg.seed),
            "max_sources": int(cfg.max_sources),
            "max_pair_abs_corr": float(cfg.max_pair_abs_corr),
            "max_rows": int(cfg.max_rows),
        },
        "scenario_reports": scenario_reports,
    }
    artifacts = write_suite_outputs(
        output_dir=output_dir,
        summary=summary,
        rows=metric_rows,
        source_rows=source_rows,
        neural_training_rows=neural_training_rows,
        neural_curve_rows=neural_curve_rows,
    )
    summary["artifacts"] = artifacts
    return {
        "suite_id": resolved_suite_id,
        "output_dir": str(output_dir),
        "artifacts": artifacts,
        "rows": metric_rows,
        "source_rows": source_rows,
        "neural_training_rows": neural_training_rows,
        "neural_curve_rows": neural_curve_rows,
        "summary": summary,
    }


__all__ = ["run_one_scenario", "run_suite"]
