"""线性回归候选的标准 pipeline representation。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.core.representation import ModelRepresentation
from mlblack.core.types import UnknownState


class SimpleRepresentation(ModelRepresentation):
    """把未知状态向量直接解码为回归权重。"""

    name = "simple_regression_weights"

    def __init__(self, n_features: int, *, seed: int = 42):
        self.n_features = int(n_features)
        self._rng = np.random.default_rng(int(seed))

    def init(self, context: Mapping[str, Any]) -> UnknownState:
        del context
        values = self._rng.normal(0.0, 0.1, size=(self.n_features,))
        return UnknownState(values=values)

    def encode(
        self,
        value: Any,
        context: Mapping[str, Any] | None = None,
    ) -> UnknownState:
        del context
        return UnknownState(values=np.asarray(value, dtype=float).reshape(-1))

    def decode(
        self,
        state: UnknownState,
        context: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        del context
        return np.asarray(state.values, dtype=float).copy()

    def mutate(
        self,
        state: UnknownState,
        context: Mapping[str, Any] | None = None,
    ) -> UnknownState:
        del context
        return state.with_values(
            np.asarray(state.values, dtype=float)
            + self._rng.normal(0.0, 0.05, size=(self.n_features,))
        )
