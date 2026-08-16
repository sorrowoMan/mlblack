from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_solver import build_solver  # type: ignore
else:
    from .build_solver import build_solver

from mlblack.integrations.etf_temporal_forecast import DEFAULT_DATASET_URL, WalkForwardSpec
from mlblack.project.scaffold import print_case_check


def _parse_int_list(text: str) -> tuple[int, ...]:
    values: list[int] = []
    for raw in str(text or "").replace(";", ",").split(","):
        item = raw.strip()
        if item:
            values.append(int(item))
    return tuple(values) if values else (42,)


def _parse_models(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for raw in str(text or "").replace(";", ",").split(","):
        item = raw.strip()
        if item:
            values.append(item)
    return tuple(values) if values else ("ridge", "hist_gradient_boosting")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the mlblack ETF temporal forecast walk-forward case.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-url", type=str, default=str(DEFAULT_DATASET_URL))
    parser.add_argument("--dataset-label", type=str, default="multi_etf_returns_momodel_kaggle")
    parser.add_argument("--suite-id", type=str, default="etf_temporal_forecast")
    parser.add_argument("--output-dir", type=str, default="runs/etf_temporal_forecast")
    parser.add_argument("--models", type=str, default="ridge,hist_gradient_boosting")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--target-horizon", type=int, default=1)
    parser.add_argument("--transaction-cost", type=float, default=0.0005)
    parser.add_argument("--wf-min-train-size", type=int, default=1200)
    parser.add_argument("--wf-test-size", type=int, default=200)
    parser.add_argument("--wf-step-size", type=int, default=200)
    parser.add_argument("--wf-mode", type=str, default="expanding")
    parser.add_argument("--wf-train-window-size", type=int, default=1440)
    parser.add_argument("--wf-max-folds", type=int, default=2)
    parser.add_argument("--wf-max-train-panel-rows", type=int, default=12000)
    parser.add_argument("--wf-max-test-panel-rows", type=int, default=4000)
    parser.add_argument("--quickstart", action="store_true")
    parser.add_argument("--serious", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if bool(args.quickstart):
        args.models = "ridge,hist_gradient_boosting"
        args.seeds = "42"
        args.wf_max_folds = 2
        args.wf_max_train_panel_rows = 12000
        args.wf_max_test_panel_rows = 4000
    if bool(args.serious):
        args.models = "ridge,elasticnet,hist_gradient_boosting,random_forest,mlp_sklearn"
        args.seeds = "42,52,62"
        args.wf_max_folds = 5
        args.wf_max_train_panel_rows = 0
        args.wf_max_test_panel_rows = 0
    wf = WalkForwardSpec(
        min_train_size=int(args.wf_min_train_size),
        test_size=int(args.wf_test_size),
        step_size=int(args.wf_step_size),
        mode=str(args.wf_mode),
        train_window_size=int(args.wf_train_window_size),
        max_folds=int(args.wf_max_folds),
        max_train_panel_rows=int(args.wf_max_train_panel_rows),
        max_test_panel_rows=int(args.wf_max_test_panel_rows),
    )
    runner = build_solver(
        dataset_url=str(args.dataset_url),
        dataset_label=str(args.dataset_label),
        models=_parse_models(str(args.models)),
        seeds=_parse_int_list(str(args.seeds)),
        suite_id=str(args.suite_id),
        output_dir=str(args.output_dir),
        walkforward=wf,
    )
    if bool(args.check):
        print_case_check(runner)
        return
    result = dict(runner.run() or {})
    summary = dict(result.get("summary", {}) or {})
    agg = dict(summary.get("aggregate", {}) or {})
    data = dict(summary.get("dataset", {}) or {})
    print(f"[etf-temporal] output_dir={result.get('output_dir', args.output_dir)}")
    print(
        "[etf-temporal] dataset="
        f"{data.get('rows')}x{data.get('assets')} {data.get('start')}..{data.get('end')}"
    )
    print(f"[etf-temporal] fold_count={summary.get('fold_count')} models={tuple(_parse_models(str(args.models)))}")
    print("[etf-temporal] aggregate=" + json.dumps(agg, sort_keys=True))


if __name__ == "__main__":
    main()
