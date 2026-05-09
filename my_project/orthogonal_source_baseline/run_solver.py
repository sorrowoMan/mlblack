from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from my_project.orthogonal_source_baseline.build_solver import (  # noqa: E402
    OrthogonalSourceBaselineConfig,
    build_orthogonal_source_baseline_components,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Orthogonal Source Layer -> strong baseline benchmark scaffold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--suite-id", type=str, default="")
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--n-total", type=int, default=360)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--noise-std", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sources", type=int, default=10)
    parser.add_argument("--max-pair-abs-corr", type=float, default=0.72)
    parser.add_argument("--max-rows", type=int, default=60000)
    parser.add_argument("--output-dir", type=str, default="runs/orthogonal_source_baseline")
    parser.add_argument("--check", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> OrthogonalSourceBaselineConfig:
    base = OrthogonalSourceBaselineConfig()
    keys = tuple(str(key) for key in args.benchmarks) if args.benchmarks else tuple(base.benchmark_keys)
    return OrthogonalSourceBaselineConfig(
        benchmark_keys=keys,
        n_total=int(args.n_total),
        train_ratio=float(args.train_ratio),
        noise_std=float(args.noise_std),
        seed=int(args.seed),
        max_sources=int(args.max_sources),
        max_pair_abs_corr=float(args.max_pair_abs_corr),
        max_rows=int(args.max_rows),
        output_dir=str(args.output_dir),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    cfg = _config_from_args(args)
    components = build_orthogonal_source_baseline_components(cfg)
    if bool(args.check):
        print(
            "orthogonal_source_baseline scaffold ok | "
            f"scenarios={len(tuple(cfg.benchmark_keys))} | "
            f"component={components['representation_component']}"
        )
        return
    result = components["runner"](cfg, suite_id=str(args.suite_id or None))
    print(f"[orth-source] suite_id={result['suite_id']}")
    print(f"[orth-source] output_dir={result['output_dir']}")
    print(f"[orth-source] table={result['artifacts']['baseline_table_md']}")
    print(f"[orth-source] neural_report={result['artifacts']['neural_training_report_md']}")
    print(f"[orth-source] neural_curve={result['artifacts']['neural_training_curve_csv']}")
    for row in result["rows"]:
        if row["feature_space"] in {"orthogonal_sources", "raw_plus_orthogonal_sources"}:
            print(
                "[orth-source] "
                f"{row['scenario']} {row['feature_space']} {row['model']} "
                f"rmse={float(row['test_rmse']):.6f} "
                f"delta_vs_raw={row['rmse_delta_vs_raw']}"
            )


if __name__ == "__main__":
    main()
