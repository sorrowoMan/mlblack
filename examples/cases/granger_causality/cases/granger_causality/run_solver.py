# -*- coding: utf-8 -*-
"""Canonical CLI for the granger_causality Case."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_solver import build_granger_trainer, generate_var1_data, ols_var1, standardize  # noqa: E402
else:
    from .build_solver import build_granger_trainer, generate_var1_data, ols_var1, standardize


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Granger causality case: sparse VAR(1) coefficient recovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-timesteps", type=int, default=500, help="Number of timesteps")
    parser.add_argument("--n-vars", type=int, default=5, help="Number of variables")
    parser.add_argument("--noise", type=float, default=0.05, help="Observation noise std")
    parser.add_argument("--l1", type=float, default=0.002, help="L1 sparsity weight")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate")
    parser.add_argument("--max-grad-norm", type=float, default=1e3, help="Gradient clipping")
    parser.add_argument("--steps", type=int, default=1500, help="Gradient descent steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--check", action="store_true", help="Build and validate only")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    print("=" * 64)
    print(" Granger Causality Case -- Sparse VAR(1) Recovery")
    print("=" * 64)
    print(f" Config: {args.n_timesteps} timesteps x {args.n_vars} vars, "
          f"noise={args.noise}, l1={args.l1}, lr={args.lr}")
    print()

    print("[1] Generating synthetic VAR(1) data with known causal structure ...")
    t0 = time.perf_counter()
    X_raw, A_true_raw = generate_var1_data(
        n_timesteps=args.n_timesteps, n_vars=args.n_vars,
        noise=args.noise, seed=args.seed,
    )
    t_gen = time.perf_counter() - t0

    print("    True causal matrix A_true (original scale):")
    print(np.array2string(A_true_raw, precision=3, suppress_small=True))
    print(f"    Data shape: {X_raw.shape}")
    print(f"    Data generation time: {t_gen:.4f}s")
    print()

    print("[2] Standardizing data and computing OLS baseline ...")
    X, X_mean, X_std = standardize(X_raw)
    A_ols = ols_var1(X)
    print("    OLS VAR(1) coefficients (standardized space):")
    print(np.array2string(A_ols, precision=3, suppress_small=True))

    n_nonzero_ols = int(np.sum(np.abs(A_ols) > 0.05))
    print(f"    OLS nonzero entries (|A| > 0.05): {n_nonzero_ols}")
    print()

    print("[3] Building mlblack trainer (stable gradient.sgd) ...")
    trainer = build_granger_trainer(
        X,
        l1_weight=args.l1, lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        run_name="granger_case",
    )

    if args.check:
        problem = getattr(trainer, "problem", None)
        rep = getattr(trainer, "representation_pipeline", None)
        adapter = getattr(trainer, "adapter", None)
        print(f"    [check] assembly ok | "
              f"problem={type(problem).__name__ if problem else 'None'} | "
              f"repr={type(rep).__name__ if rep else 'None'} | "
              f"adapter={type(adapter).__name__ if adapter else 'None'}")
        return

    print(f"\n[4] Training via gradient descent ({args.steps} steps) ...")
    t_train = time.perf_counter()
    result = trainer.fit(max_steps=args.steps)
    t_train = time.perf_counter() - t_train

    A_recovered = trainer.decode_candidate(result.best_state) if result.best_state else None
    final_loss = result.best_feedback.loss if result.best_feedback else float("nan")
    final_mse = result.best_feedback.metrics.get("mse", float("nan")) if result.best_feedback else float("nan")
    final_r2 = result.best_feedback.metrics.get("r2", float("nan")) if result.best_feedback else float("nan")

    print(f"\n{'=' * 64}")
    print(f" Results")
    print(f"{'=' * 64}")
    print(f"  Steps:                     {len(result.history)}")
    print(f"  Training time:             {t_train:.4f}s")
    print(f"  Final loss (MSE + L1*sum|A|): {final_loss:.6f}")
    print(f"  Final MSE:                 {final_mse:.6f}")
    print(f"  Final R2:                  {(final_r2 * 100):.2f}%")

    if A_recovered is not None:
        mae = float(np.mean(np.abs(A_recovered - A_ols)))
        corr = float(np.corrcoef(A_recovered.ravel(), A_ols.ravel())[0, 1])
        n_nonzero_rec = int(np.sum(np.abs(A_recovered) > 0.05))

        print(f"\n{'=' * 64}")
        print(f" Coefficient Comparison (standardized VAR space)")
        print(f"{'=' * 64}")
        print("    OLS (baseline):")
        print(np.array2string(A_ols, precision=4, suppress_small=True))
        print()
        print("    Gradient Descent + L1 (recovered):")
        print(np.array2string(A_recovered, precision=4, suppress_small=True))
        print()
        print(f"  MAE |A_rec - A_ols|:  {mae:.6f}")
        print(f"  Correlation:          {corr:.6f}")
        print(f"  Nonzero entries (|A| > 0.05, OLS):   {n_nonzero_ols}")
        print(f"  Nonzero entries (|A| > 0.05, GD+L1): {n_nonzero_rec}")
        print()

        print(f"\n{'=' * 64}")
        print(f" Granger Causality Interpretation")
        print(f"{'=' * 64}")
        print("  A[i,j] != 0 => variable j Granger-causes variable i")
        print()
        edges_ols = []
        edges_rec = []
        edge_threshold = 0.05
        for i in range(args.n_vars):
            for j in range(args.n_vars):
                if abs(A_ols[i, j]) > edge_threshold:
                    edges_ols.append(f"  {j} -> {i}")
                if abs(A_recovered[i, j]) > edge_threshold:
                    edges_rec.append(f"  {j} -> {i}")
        print("  Causal edges (OLS):  ", ",  ".join(edges_ols) if edges_ols else "  (none)")
        print("  Causal edges (GD+L1):", ",  ".join(edges_rec) if edges_rec else "  (none)")

    print(f"\n{'=' * 64}")
    print(" Done.")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    raise SystemExit(main())
