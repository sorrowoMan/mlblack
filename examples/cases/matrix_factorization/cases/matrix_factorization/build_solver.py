# -*- coding: utf-8 -*-
"""Matrix Factorization Recommendation Case -- build and run the MF trainer.

Demonstrates matrix factorization as a gradient-based mlblack learning problem:
  1. Generate synthetic low-rank rating matrix R_ij = U_i @ V_j + noise
  2. Mask 80% of entries as "unobserved"
  3. Recover U and V via stable gradient.sgd through LearningSolver
  4. Compare against sklearn TruncatedSVD baseline

Architecture:
  nsgablack GradientOptimizerAdapter (resolved by stable method ID)
  + MFRepresentation              (custom codec: flat(U,V) <-> U,V matrices)
  + MatrixFactorizationProblem    (custom: sparse MSE + analytic gradients)
  + StateL2Bias                   (framework, optional L2 penalty)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mlblack.integrations.nsgablack_control import build_learning_solver
from mlblack.integrations.nsgablack_optimization import build_optimization_adapter
from mlblack.bias import StateL2Bias

_HERE = Path(__file__).resolve().parent


def generate_synthetic_data(n_users=100, n_items=200, k=5, sparsity=0.80, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    true_U = rng.normal(0.0, 1.0, size=(n_users, k)) * 0.5
    true_V = rng.normal(0.0, 1.0, size=(n_items, k)) * 0.5

    R_true = true_U @ true_V.T
    R = R_true + rng.normal(0.0, noise, size=R_true.shape)
    R = np.clip(R, 1.0, 5.0)

    mask = rng.random(size=R.shape) > sparsity
    if mask.sum() < 10:
        idx = rng.choice(n_users * n_items, size=10, replace=False)
        mask_flat = mask.ravel()
        mask_flat[idx] = True
        mask = mask_flat.reshape(R.shape)

    R_observed = np.where(mask, R, 0.0)
    return R, R_observed, mask, true_U, true_V, R_true


def compute_rmse(U, V, R, mask):
    pred = U @ V.T
    error = mask.astype(float) * (pred - R)
    n = max(int(mask.sum()), 1)
    return float(np.sqrt(np.sum(error ** 2) / n))


def svd_baseline(R_observed, mask, k=5):
    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=k, random_state=42)
    svd.fit(R_observed)
    U_svd = svd.transform(R_observed)
    V_svd = svd.components_.T
    Sigma_sqrt = np.sqrt(svd.singular_values_)
    U_svd_scaled = U_svd * Sigma_sqrt[None, :]
    V_svd_scaled = V_svd * Sigma_sqrt[None, :]
    return U_svd_scaled, V_svd_scaled


def build_mf_trainer(
    R,
    mask,
    *,
    k=5,
    lr=5.0,
    l2_weight=0.0,
    reg_weight=0.0,
    max_grad_norm=1e3,
    nmf=False,
    run_name="mf_rec",
    component_overrides=None,
    resource_context=None,
):
    from problem.matrix_factorization_problem import MatrixFactorizationProblem
    from pipeline.main import build_pipeline

    n_users, n_items = R.shape

    problem = MatrixFactorizationProblem(R, mask, reg_weight=reg_weight, name="mf_problem")
    representation = build_pipeline(
        n_users,
        n_items,
        k=k,
        nmf=nmf,
        component_overrides=component_overrides,
    )
    adapter = build_optimization_adapter(
        "gradient.sgd",
        learning_rate=lr,
        max_gradient_norm=max_grad_norm,
    )

    trainer = build_learning_solver(
        problem=problem,
        representation=representation,
        adapter=adapter,
        run_name=run_name,
        resource_context=resource_context,
    )

    if l2_weight > 0:
        trainer.add_bias(StateL2Bias(weight=l2_weight))

    return trainer


def build_solver(
    *,
    config=None,
    n_users: int = 100,
    n_items: int = 200,
    k: int = 5,
    sparsity: float = 0.80,
    noise: float = 0.1,
    lr: float = 5.0,
    l2_weight: float = 0.0,
    reg_weight: float = 0.0,
    max_grad_norm: float = 1e3,
    nmf: bool = False,
    seed: int = 42,
    resource_context=None,
    component_overrides=None,
):
    """Canonical unified scaffold entry; returns the assembled Trainer."""

    del config
    R, _, mask, _, _, _ = generate_synthetic_data(
        n_users=n_users,
        n_items=n_items,
        k=k,
        sparsity=sparsity,
        noise=noise,
        seed=seed,
    )
    return build_mf_trainer(
        R,
        mask,
        k=k,
        lr=lr,
        l2_weight=l2_weight,
        reg_weight=reg_weight,
        max_grad_norm=max_grad_norm,
        nmf=nmf,
        run_name="mf_demo",
        component_overrides=component_overrides,
        resource_context=resource_context,
    )


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
    main()
