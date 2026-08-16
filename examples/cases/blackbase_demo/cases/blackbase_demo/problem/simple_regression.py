"""用于演示统一 substrate 的小型线性回归问题。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.core.problem import LearningProblem
from mlblack.core.types import Feedback, UnknownState


class SimpleRegressionProblem(LearningProblem):
    """以均方误差作为单目标，展示标准 Trainer 评估路径。"""

    name = "simple_regression"

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y, dtype=float).reshape(-1)
        self.n_features = int(self._X.shape[1])

    def evaluate(
        self,
        model: Any,
        state: UnknownState,
        context: Mapping[str, Any] | None = None,
    ) -> Feedback:
        """评估解码后的权重，并保留梯度供观测或新 Adapter 使用。"""

        del state, context
        weights = np.asarray(model, dtype=float).reshape(-1)
        residuals = self._X @ weights - self._y
        mse = float(np.mean(residuals**2))
        gradient = (2.0 / len(self._y)) * (self._X.T @ residuals)
        return Feedback(
            objectives=np.array([mse], dtype=float),
            gradients=gradient,
            constraints=np.zeros(0, dtype=float),
            metrics={"mse": mse},
            residuals=residuals,
        )
