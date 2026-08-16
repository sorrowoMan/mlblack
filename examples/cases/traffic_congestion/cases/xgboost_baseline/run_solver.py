"""Canonical CLI for the traffic XGBoost baseline Trainer."""

from __future__ import annotations

import argparse
import time

from blackbase.project import load_resource_context_from_env

try:
    from .build_solver import build_solver
except ImportError:  # direct script execution
    from build_solver import build_solver

from mlblack.project.scaffold import print_case_check


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the traffic XGBoost baseline.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--mutation-scale", type=float, default=0.2)
    parser.add_argument("--check", action="store_true", help="Build and validate only; do not fit")
    args = parser.parse_args(argv)
    trainer = build_solver(
        config={"population_size": args.population_size, "mutation_scale": args.mutation_scale},
        resource_context=load_resource_context_from_env("mlblack"),
    )
    if args.check:
        print_case_check(trainer)
        return 0

    started = time.time()
    trainer.fit(max_steps=args.steps)
    elapsed = time.time() - started
    data = trainer.traffic_data
    metrics = trainer.build_report().get("best_metrics", {})
    print(f"XGBoost on Traffic CI ({len(data.X_train)} train, {len(data.X_valid)} valid)")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Train RMSE: {metrics.get('train.rmse', 'N/A')}")
    print(f"  Valid RMSE: {metrics.get('valid.rmse', 'N/A')}")
    print(f"  Valid R2:  {metrics.get('valid.r2', 'N/A')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
