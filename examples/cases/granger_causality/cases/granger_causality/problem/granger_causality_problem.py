# -*- coding: utf-8 -*-
"""Granger causality as sparse VAR(1) coefficient optimization.

A[i,j] != 0 means variable j Granger-causes variable i.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.core.problem import LearningProblem
from mlblack.core.types import Feedback


class GrangerCausalityProblem(LearningProblem):
    """Minimize VAR(1) prediction MSE + L1 sparsity on coefficient matrix A.

    Model: X[t] = X[t-1] @ A^T
    Loss:  mean((X_pred[t] - X_observed[t])^2) + l1_weight * sum(|A|)
    Gradient: dL/dA = 2 * residuals^T @ X_lag / (n_samples * n_vars)

    Data should be standardized before passing to the problem for
    optimal gradient scaling.
    """

    context_requires = ("candidate.model", "data")
    context_optional = ()
    context_provides = ("feedback.objectives", "feedback.metrics", "feedback.gradients")
    context_mutates = ()
    context_cache = ()
    context_notes = (
        "Reads decoded VAR coefficient matrix A, computes VAR(1) prediction MSE "
        "with L1 sparsity penalty, returns objectives, gradients, and metrics."
    )

    def __init__(self, X, *, l1_weight=0.005, name="granger_causality"):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_timesteps, n_vars), got {X.shape}")
        if X.shape[0] < 2:
            raise ValueError(f"Need at least 2 timesteps, got {X.shape[0]}")

        self.X = X
        self.n_timesteps, self.n_vars = X.shape
        self.X_lag = X[:-1]
        self.X_obs = X[1:]
        self.n_samples = self.n_timesteps - 1
        self.n_elements = self.n_samples * self.n_vars
        self.l1_weight = float(l1_weight)
        self.name = name

    def evaluate(self, model, state, context):
        A = np.asarray(model, dtype=float)
        if A.shape != (self.n_vars, self.n_vars):
            raise ValueError(f"A shape {A.shape} != ({self.n_vars},{self.n_vars})")

        pred = self.X_lag @ A.T
        residuals = pred - self.X_obs

        mse = float(np.mean(residuals ** 2))
        l1_penalty = float(np.sum(np.abs(A)))
        loss = mse + self.l1_weight * l1_penalty

        dMSE_dA = (2.0 / self.n_elements) * (residuals.T @ self.X_lag)
        dL1_dA = self.l1_weight * np.sign(A)
        gradient = dMSE_dA + dL1_dA

        r2 = 1.0 - float(np.sum(residuals ** 2)) / max(
            float(np.sum((self.X_obs - np.mean(self.X_obs)) ** 2)), 1e-12
        )

        return Feedback(
            objectives=np.array([loss]),
            gradients=gradient.ravel(),
            constraints=np.zeros(0, dtype=float),
            loss=loss,
            metrics={
                "mse": float(mse),
                "l1_penalty": l1_penalty,
                "r2": float(r2),
                "n_vars": int(self.n_vars),
                "n_timesteps": int(self.n_timesteps),
            },
        )

    def describe(self):
        return {
            "name": self.name,
            "n_vars": self.n_vars,
            "n_timesteps": self.n_timesteps,
            "l1_weight": self.l1_weight,
            "objective": "minimize VAR(1) MSE + L1 sparsity",
        }
