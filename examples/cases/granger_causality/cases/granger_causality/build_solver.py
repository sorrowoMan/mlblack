# -*- coding: utf-8 -*-
"""Granger Causality Case -- sparse VAR(1) coefficient recovery via gradient descent.

Demonstrates Granger causality testing as a gradient-based mlblack learning problem:
  1. Generate synthetic VAR(1) data with a known sparse coefficient matrix A_true
  2. Standardize data, recover A via stable gradient.sgd through LearningSolver
  3. Compare recovered vs OLS baseline coefficients

Architecture:
  nsgablack GradientOptimizerAdapter (stable method gradient.sgd)
  + GrangerRepresentation          (custom: flat(A) <-> (n_vars, n_vars) A matrix)
  + GrangerCausalityProblem        (custom: VAR(1) MSE + L1 sparsity + analytic gradient)
"""

from __future__ import annotations

import numpy as np

from mlblack.integrations.nsgablack_control import build_learning_solver
from mlblack.integrations.nsgablack_optimization import build_optimization_adapter

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


def build_granger_trainer(X, *, l1_weight=0.002, lr=1.0, max_grad_norm=1e3, run_name="granger_demo", component_overrides=None, resource_context=None):
    from problem.granger_causality_problem import GrangerCausalityProblem
    from pipeline.main import build_pipeline

    n_vars = X.shape[1]

    problem = GrangerCausalityProblem(X, l1_weight=l1_weight, name="granger_problem")
    representation = build_pipeline(n_vars, component_overrides=component_overrides)
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

    return trainer


def build_solver(
    *,
    config=None,
    n_timesteps: int = 500,
    n_vars: int = 5,
    noise: float = 0.05,
    l1_weight: float = 0.002,
    lr: float = 1.0,
    max_grad_norm: float = 1e3,
    seed: int = 42,
    resource_context=None,
    component_overrides=None,
):
    """Canonical unified scaffold entry; returns the assembled Trainer."""

    del config
    X_raw, _ = generate_var1_data(n_timesteps=n_timesteps, n_vars=n_vars, noise=noise, seed=seed)
    X, _, _ = standardize(X_raw)
    return build_granger_trainer(
        X,
        l1_weight=l1_weight,
        lr=lr,
        max_grad_norm=max_grad_norm,
        run_name="granger_case",
        component_overrides=component_overrides,
        resource_context=resource_context,
    )
