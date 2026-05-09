from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BasePipeline(ABC):
    name = "base"

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "BasePipeline":
        _ = X, y
        return self

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        ...

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> "BasePipeline":
        _ = state
        return self
