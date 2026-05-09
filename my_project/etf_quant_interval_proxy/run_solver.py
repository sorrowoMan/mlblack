from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from my_project.etf_quant_interval_proxy.build_solver import (
    EtfQuantIntervalConfig,
    run_etf_quant_interval_proxy,
)
from my_project.etf_quant_interval_proxy.problem import load_etf_panel_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ETF quant interval proxy over real ETF return panel data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--suite-id", type=str, default="")
    parser.add_argument("--dataset-url", type=str, default=EtfQuantIntervalConfig.dataset_url)
    parser.add_argument("--hf-repo-id", type=str, default=EtfQuantIntervalConfig.hf_repo_id)
    parser.add_argument("--hf-filename", type=str, default=EtfQuantIntervalConfig.hf_filename)
    parser.add_argument("--hf-revision", type=str, default=EtfQuantIntervalConfig.hf_revision)
    parser.add_argument("--cache-dir", type=str, default=EtfQuantIntervalConfig.cache_dir)
    parser.add_argument("--dataset-label", type=str, default=EtfQuantIntervalConfig.dataset_label)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.72)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interval-alpha", type=float, default=0.10)
    parser.add_argument("--max-sources", type=int, default=28)
    parser.add_argument("--candidate-keep-top", type=int, default=180)
    parser.add_argument("--max-pair-abs-corr", type=float, default=0.68)
    parser.add_argument("--rolling-window", type=int, default=40)
    parser.add_argument("--output-dir", type=str, default="runs/etf_quant_interval_proxy")
    parser.add_argument("--check", action="store_true")
    return parser


def _cfg_from_args(args: argparse.Namespace) -> EtfQuantIntervalConfig:
    return EtfQuantIntervalConfig(
        dataset_url=str(args.dataset_url),
        hf_repo_id=str(args.hf_repo_id),
        hf_filename=str(args.hf_filename),
        hf_revision=str(args.hf_revision),
        cache_dir=str(args.cache_dir),
        dataset_label=str(args.dataset_label),
        horizon=int(args.horizon),
        lookback=int(args.lookback),
        train_ratio=float(args.train_ratio),
        max_rows=int(args.max_rows),
        seed=int(args.seed),
        interval_alpha=float(args.interval_alpha),
        max_sources=int(args.max_sources),
        candidate_keep_top=int(args.candidate_keep_top),
        max_pair_abs_corr=float(args.max_pair_abs_corr),
        rolling_window=int(args.rolling_window),
        output_dir=str(args.output_dir),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    cfg = _cfg_from_args(args)
    suite_id = str(args.suite_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    if bool(args.check):
        dataset = load_etf_panel_dataset(cfg)
        print(
            "etf_quant_interval_proxy scaffold ok | "
            f"returns_rows={dataset.metadata['dataset_rows']} | "
            f"panel_rows={dataset.metadata['panel_rows']} | "
            f"features={dataset.metadata['feature_count']} | "
            f"symbols={len(dataset.metadata['symbols'])}"
        )
        return
    result = run_etf_quant_interval_proxy(cfg, suite_id=suite_id)
    artifacts = dict(result.get("artifacts", {}) or {})
    summary = dict(result.get("summary", {}) or {})
    print(f"[etf-quant] suite_id={suite_id}")
    print(f"[etf-quant] panel_rows={summary.get('panel_rows')} features={summary.get('feature_count')}")
    print(f"[etf-quant] selected_source_count={summary.get('selected_source_count')}")
    print(f"[etf-quant] report={artifacts.get('report_md')}")
    print(f"[etf-quant] baseline_metrics={artifacts.get('baseline_metrics_csv')}")
    print(f"[etf-quant] interval_metrics={artifacts.get('interval_metrics_csv')}")
    print(f"[etf-quant] rolling_metrics={artifacts.get('rolling_metrics_csv')}")
    print(f"[etf-quant] rank_backtest_metrics={artifacts.get('rank_backtest_metrics_csv')}")
    print(f"[etf-quant] winners={summary.get('winners')}")


if __name__ == "__main__":
    main()
