# -*- coding: utf-8 -*-
"""Matrix factorization as a gradient-based LearningProblem.

Reconstruction MSE on observed rating entries only.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.core.problem import LearningProblem
from mlblack.core.types import Feedback


class MatrixFactorizationProblem(LearningProblem):
    """Minimize squared reconstruction error on observed (user, item) ratings.

    Unknown state encodes [U.flatten(), V.flatten()] for U=(n_users, k), V=(n_items, k).
    The decoded model is the (U, V) tuple.
    """

    context_requires = ("candidate.model", "data")
    context_optional = ()
    context_provides = ("feedback.objectives", "feedback.metrics", "feedback.gradients")
    context_mutates = ()
    context_cache = ()
    context_notes = (
        "Reads candidate.model (U,V tuple), computes MSE over observed mask, "
        "returns objectives, gradients, metrics including RMSE and MAE."
    )

    def __init__(
        self,
        R,
        mask,
        *,
        reg_weight=0.0,
        name="matrix_factorization",
    ):
        R = np.asarray(R, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        if R.shape != mask.shape:
            raise ValueError(f"R shape {R.shape} must match mask shape {mask.shape}")
        self.R = R
        self.mask = mask
        self.n_observed = max(int(mask.sum()), 1)
        self.reg_weight = float(reg_weight)
        self._n_users, self._n_items = R.shape
        self.name = name

    @property
    def n_users(self):
        return self._n_users

    @property
    def n_items(self):
        return self._n_items

    def evaluate(self, model, state, context):
        U, V = model
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)

        pred = U @ V.T
        error = self.mask.astype(float) * (pred - self.R)
        mse = float(np.sum(error ** 2)) / self.n_observed

        n = self.n_observed
        dU = (2.0 / n) * (error @ V)
        dV = (2.0 / n) * (error.T @ U)

        if self.reg_weight > 0:
            mse = mse + self.reg_weight * (float(np.sum(U ** 2)) / U.size + float(np.sum(V ** 2)) / V.size)
            dU = dU + 2.0 * self.reg_weight * U / U.size
            dV = dV + 2.0 * self.reg_weight * V / V.size

        gradient = np.concatenate([dU.ravel(), dV.ravel()])

        rmse = float(np.sqrt(mse * self.n_observed / max(self.n_observed, 1)))
        mae = float(np.sum(np.abs(error))) / self.n_observed

        return Feedback(
            objectives=np.array([mse]),
            gradients=gradient,
            constraints=np.zeros(0, dtype=float),
            loss=mse,
            metrics={"rmse": rmse, "mae": mae, "n_observed": int(self.n_observed)},
        )

    def reconstruct(self, U, V):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        return U @ V.T

    def describe(self):
        return {
            "name": self.name,
            "n_users": self._n_users,
            "n_items": self._n_items,
            "n_observed": int(self.n_observed),
            "sparsity": 1.0 - float(self.n_observed) / (self._n_users * self._n_items),
            "reg_weight": self.reg_weight,
            "objective": "minimize observed-entry MSE",
        }
