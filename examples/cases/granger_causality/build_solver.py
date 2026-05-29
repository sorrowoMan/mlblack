# -*- coding: utf-8 -*-
"""Granger Causality Case -- sparse VAR(1) coefficient recovery via gradient descent.

Demonstrates Granger causality testing as a gradient-based mlblack learning problem:
  1. Generate synthetic VAR(1) data with a known sparse coefficient matrix A_true
  2. Standardize data, recover A via framework GradientDescentAdapter through ComposableTrainer
  3. Compare recovered vs OLS baseline coefficients

Architecture:
  MLBlack GradientDescentAdapter   (framework adapter.gradient_descent)
  + GrangerRepresentation          (custom: flat(A) <-> (n_vars, n_vars) A matrix)
  + GrangerCausalityProblem        (custom: VAR(1) MSE + L1 sparsity + analytic gradient)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mlblack.core.trainer import ComposableTrainer
from mlblack.adapters.gradient_descent import GradientDescentAdapter

_HERE = Path(__file__).resolve().parent


def generate_var1_data(n_timesteps=500, n_vars=5, noise=0.05, seed=42):
    rng = np.random.default_rng(seed)
    A_true = np.array([
        [0.0, 0.8, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.7, 0.0, 0.0],
        [0.6, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.9],
        [0.0, 0.0, 0.0, 0.4, 0.0],
    ])

    X = np.zeros((n_timesteps, n_vars))
    X[0] = rng.normal(0.0, 0.1, size=n_vars)
    for t in range(1, n_timesteps):
        X[t] = X[t - 1] @ A_true.T + rng.normal(0.0, noise, size=n_vars)

    return X, A_true


def standardize(X):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    std = np.where(std < 1e-12, 1.0, std)
    X_s = (X - mean) / std
    return X_s, mean, std


def ols_var1(X):
    X_lag = X[:-1]
    X_obs = X[1:]
    A_ols, _, _, _ = np.linalg.lstsq(X_lag, X_obs, rcond=None)
    return A_ols.T


def build_granger_trainer(X, *, l1_weight=0.002, lr=1.0, max_grad_norm=1e3, run_name="granger_demo"):
    from problem.granger_causality_problem import GrangerCausalityProblem
    from pipeline.representation.granger_representation import GrangerRepresentation

    n_vars = X.shape[1]

    problem = GrangerCausalityProblem(X, l1_weight=l1_weight, name="granger_problem")
    representation = GrangerRepresentation(n_vars, init_scale=0.05, name="granger_rep")
    adapter = GradientDescentAdapter(learning_rate=lr, max_grad_norm=max_grad_norm)

    trainer = ComposableTrainer(
        problem=problem,
        representation=representation,
        adapter=adapter,
        run_name=run_name,
        constraint_penalty=1e6,
    )

    return trainer


def build_solver(
    *,
    n_timesteps: int = 500,
    n_vars: int = 5,
    noise: float = 0.05,
    l1_weight: float = 0.002,
    lr: float = 1.0,
    max_grad_norm: float = 1e3,
    seed: int = 42,
):
    """Canonical unified scaffold entry; returns the assembled Trainer."""

    X_raw, _ = generate_var1_data(n_timesteps=n_timesteps, n_vars=n_vars, noise=noise, seed=seed)
    X, _, _ = standardize(X_raw)
    return build_granger_trainer(
        X,
        l1_weight=l1_weight,
        lr=lr,
        max_grad_norm=max_grad_norm,
        run_name="granger_case",
    )


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

    print("[3] Building mlblack trainer (framework GradientDescentAdapter) ...")
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
    main()
