# -*- coding: utf-8 -*-
"""Canonical CLI for the matrix_factorization Case."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_solver import build_mf_trainer, compute_rmse, generate_synthetic_data, svd_baseline  # noqa: E402
else:
    from .build_solver import build_mf_trainer, compute_rmse, generate_synthetic_data, svd_baseline


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Matrix Factorization recommendation case (mlblack framework GD adapter).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-users", type=int, default=100, help="Number of users")
    parser.add_argument("--n-items", type=int, default=200, help="Number of items")
    parser.add_argument("--k", type=int, default=5, help="Embedding dimension / rank")
    parser.add_argument("--sparsity", type=float, default=0.80, help="Fraction of unobserved ratings")
    parser.add_argument("--noise", type=float, default=0.1, help="Observation noise std")
    parser.add_argument("--lr", type=float, default=5.0, help="Learning rate")
    parser.add_argument("--l2", type=float, default=0.0, help="L2 regularization weight (StateL2Bias)")
    parser.add_argument("--reg", type=float, default=0.0, help="L2 reg weight inside problem evaluate()")
    parser.add_argument("--max-grad-norm", type=float, default=1e3, help="Gradient clipping threshold")
    parser.add_argument("--nmf", action="store_true", help="Enforce non-negativity via repair() projection")
    parser.add_argument("--steps", type=int, default=300, help="Number of gradient descent steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-svd", action="store_true", help="Skip SVD baseline comparison")
    parser.add_argument("--check", action="store_true", help="Build and validate only, do not run fit")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    print("=" * 64)
    print(" Matrix Factorization Recommendation Case")
    print("=" * 64)
    print(f" Config: {args.n_users} users x {args.n_items} items, k={args.k}, "
          f"sparsity={args.sparsity}, noise={args.noise}")
    print(f" Optimizer: lr={args.lr}, max_grad_norm={args.max_grad_norm}, "
          f"nmf={args.nmf}, l2={args.l2}, reg={args.reg}")
    print()

    print("[1] Generating synthetic rating matrix ...")
    t0 = time.perf_counter()
    R, R_observed, mask, true_U, true_V, R_true = generate_synthetic_data(
        n_users=args.n_users, n_items=args.n_items, k=args.k,
        sparsity=args.sparsity, noise=args.noise, seed=args.seed,
    )
    t_gen = time.perf_counter() - t0
    print(f"    Observed entries: {mask.sum()} / {mask.size} "
          f"({100.0 * (1 - mask.mean()):.1f}% missing)")
    print(f"    Data generation time: {t_gen:.4f}s")

    print("\n[2] Building mlblack trainer (stable gradient.sgd) ...")
    trainer = build_mf_trainer(
        R, mask,
        k=args.k,
        lr=args.lr, l2_weight=args.l2, reg_weight=args.reg,
        max_grad_norm=args.max_grad_norm, nmf=args.nmf,
        run_name="mf_demo",
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

    print("\n[3] Training via gradient descent ...")
    t_train = time.perf_counter()
    result = trainer.fit(max_steps=args.steps)
    t_train = time.perf_counter() - t_train

    U_learned, V_learned = trainer.decode_candidate(result.best_state) if result.best_state else (None, None)
    final_loss = result.best_feedback.loss if result.best_feedback else float("nan")
    final_rmse = result.best_feedback.metrics.get("rmse", float("nan")) if result.best_feedback else float("nan")

    print(f"\n{'=' * 64}")
    print(f" MLBlack Gradient Descent Results")
    print(f"{'=' * 64}")
    print(f"  Steps:                     {len(result.history)}")
    print(f"  Training time:             {t_train:.4f}s")
    print(f"  Final loss (MSE, obs):     {final_loss:.6f}")
    print(f"  Final RMSE (obs):          {final_rmse:.6f}")

    if U_learned is not None and V_learned is not None:
        rmse_complete = compute_rmse(U_learned, V_learned, R, mask)
        print(f"  Reconstruction RMSE:       {rmse_complete:.6f}")

        if not args.no_svd:
            print(f"\n{'=' * 64}")
            print(f" SVD Baseline Comparison")
            print(f"{'=' * 64}")
            t_svd = time.perf_counter()
            U_svd, V_svd = svd_baseline(R_observed, mask, k=args.k)
            t_svd = time.perf_counter() - t_svd

            svd_rmse = compute_rmse(U_svd, V_svd, R, mask)

            pred_gd = U_learned @ V_learned.T
            pred_svd = U_svd @ V_svd.T
            gd_corr = float(np.corrcoef(
                pred_gd[mask].ravel(), R[mask].ravel()
            )[0, 1])
            svd_corr = float(np.corrcoef(
                pred_svd[mask].ravel(), R[mask].ravel()
            )[0, 1])

            print(f"  SVD time:                  {t_svd:.4f}s")
            print(f"  SVD RMSE (obs):            {svd_rmse:.6f}")
            print(f"  GD time:                   {t_train:.4f}s")
            print(f"  GD RMSE (obs):             {rmse_complete:.6f}")
            print(f"  GD correlation (obs):      {gd_corr:.4f}")
            print(f"  SVD correlation (obs):     {svd_corr:.4f}")

    print(f"\n{'=' * 64}")
    print(" Done.")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    raise SystemExit(main())
