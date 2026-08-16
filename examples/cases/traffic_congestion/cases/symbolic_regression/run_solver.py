"""Canonical CLI for the traffic symbolic-compatible regression Trainer."""

from __future__ import annotations

import argparse
import time

import numpy as np

from blackbase.project import load_resource_context_from_env

try:
    from .build_solver import build_solver
    from .pipeline import add_intercept
except ImportError:  # direct script execution
    from build_solver import build_solver
    from pipeline import add_intercept

from mlblack.project.scaffold import print_case_check


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the traffic symbolic-compatible linear baseline.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--check", action="store_true", help="Build and validate only; do not fit")
    args = parser.parse_args(argv)
    trainer = build_solver(
        config={"learning_rate": args.lr},
        resource_context=load_resource_context_from_env("mlblack"),
    )
    if args.check:
        print_case_check(trainer)
        return 0

    data = trainer.traffic_data
    started = time.time()
    result = trainer.fit(max_steps=args.steps)
    elapsed = time.time() - started
    best = result.best_state.as_array()
    pred_train = add_intercept(data.X_train) @ best
    pred_val = add_intercept(data.X_valid) @ best if data.X_valid is not None else None
    train_rmse = np.sqrt(np.mean((pred_train - data.y_train) ** 2))
    val_rmse = np.sqrt(np.mean((pred_val - data.y_valid) ** 2)) if pred_val is not None else float("nan")
    feature_cols = list(data.feature_names)

    print("Symbolic-compatible Linear Regression on Traffic CI")
    print(f"  Features: {data.n_features} (+ intercept), Steps: {args.steps}, LR: {args.lr}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Valid RMSE: {val_rmse:.4f}")
    print(f"  Intercept: {best[0]:+.4f}")
    print("  Top 5 feature coefficients (abs):")
    top_idx = 1 + np.argsort(np.abs(best[1:]))[-5:][::-1]
    for index in top_idx:
        print(f"    {feature_cols[index - 1]:30s}: {best[index]:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
