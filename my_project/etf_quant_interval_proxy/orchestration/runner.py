from __future__ import annotations

from pathlib import Path
from typing import Any

from my_project.etf_quant_interval_proxy.config import EtfQuantIntervalConfig
from my_project.etf_quant_interval_proxy.pipeline import build_orthogonal_etf_sources, fit_interval_baselines
from my_project.etf_quant_interval_proxy.pipeline.baselines import summarize_winners
from my_project.etf_quant_interval_proxy.problem import load_etf_panel_dataset
from my_project.etf_quant_interval_proxy.reporting import write_etf_quant_reports


def run_suite(cfg: EtfQuantIntervalConfig, *, suite_id: str) -> dict[str, Any]:
    output_dir = Path(cfg.output_dir).expanduser().resolve() / str(suite_id)
    dataset = load_etf_panel_dataset(cfg)
    source_result = build_orthogonal_etf_sources(dataset, cfg)
    baseline_result = fit_interval_baselines(
        raw_train=dataset.X_train,
        raw_test=dataset.X_test,
        basis_train=source_result.train_basis,
        basis_test=source_result.test_basis,
        y_train=dataset.y_train,
        y_test=dataset.y_test,
        test_time_idx=dataset.test_time_idx,
        test_symbols=dataset.test_symbols,
        cfg=cfg,
    )
    summary = {
        "suite_id": str(suite_id),
        "protocol": "etf_quant_interval_proxy_v1",
        "dataset_label": str(cfg.dataset_label),
        "dataset_url": str(cfg.dataset_url),
        "config": cfg.__dict__,
        **dict(dataset.metadata),
        "train_rows": int(dataset.X_train.shape[0]),
        "test_rows": int(dataset.X_test.shape[0]),
        "selected_source_count": int(source_result.report.get("selected_source_count", source_result.train_basis.shape[1])),
        "source_report": dict(source_result.report),
        "winners": summarize_winners(tuple(baseline_result.metric_rows)),
    }
    artifacts = write_etf_quant_reports(
        output_dir=output_dir,
        summary=summary,
        metric_rows=baseline_result.metric_rows,
        interval_rows=baseline_result.interval_rows,
        rolling_rows=baseline_result.rolling_rows,
        backtest_rows=baseline_result.backtest_rows,
        source_rows=source_result.source_rows,
    )
    return {
        "summary": summary,
        "artifacts": artifacts,
        "metric_rows": baseline_result.metric_rows,
        "interval_rows": baseline_result.interval_rows,
        "rolling_rows": baseline_result.rolling_rows,
        "backtest_rows": baseline_result.backtest_rows,
        "source_rows": source_result.source_rows,
    }


__all__ = ["run_suite"]
