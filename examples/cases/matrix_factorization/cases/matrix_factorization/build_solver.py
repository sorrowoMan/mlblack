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

import numpy as np

from mlblack.integrations.nsgablack_control import build_learning_solver
from mlblack.integrations.nsgablack_optimization import build_optimization_adapter
from mlblack.bias import StateL2Bias

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
