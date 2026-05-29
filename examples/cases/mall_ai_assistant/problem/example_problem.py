# -*- coding: utf-8 -*-
"""Example LearningProblem: supervised regression with MSE."""

from __future__ import annotations

import numpy as np

from mlblack.core.problem import LearningProblem
from mlblack.core.types import Feedback


class ExampleRegressionProblem(LearningProblem):
    """Minimize MSE between model prediction and target."""

    context_requires = ()
    context_provides = ("feedback.objectives",)
    context_mutates = ()
    context_cache = ()
    context_notes = "Computes MSE loss from model prediction vs observed target."

    def __init__(self, data, *, name="example_regression"):
        self._X = np.asarray(data.get("X", np.zeros((1, 1))), dtype=float)
        self._y = np.asarray(data.get("y", np.zeros(1)), dtype=float).ravel()
        super().__init__(name=name)

    def evaluate(self, unknown_state):
        pred = np.asarray(unknown_state, dtype=float).ravel()
        if len(pred) != len(self._y):
            pred = np.full_like(self._y, pred[0] if len(pred) else 0.0)
        residuals = pred - self._y
        mse = float(np.mean(residuals ** 2))
        return Feedback(
            objectives=np.array([mse]),
            gradients=residuals,
            constraints=np.zeros(0, dtype=float),
        )

    def describe(self):
        return {
            "name": self.name,
            "n_samples": len(self._y),
            "objective": "minimize MSE",
        }
