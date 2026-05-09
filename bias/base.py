from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class FitContext:
    l2_multiplier: float = 1.0
    sample_weight: np.ndarray | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTrainingBias(ABC):
    name = "base_bias"

    @abstractmethod
    def apply(self, X: np.ndarray, Y: np.ndarray, context: FitContext) -> tuple[np.ndarray, np.ndarray]:
        ...
