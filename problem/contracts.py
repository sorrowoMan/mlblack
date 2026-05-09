from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class DecodedDecision:
    subset_idx: tuple[int, ...]
    meta: dict[str, Any]


DecisionDecodeFn = Callable[[np.ndarray], tuple[list[int], int, dict[str, Any]]]
DecodedEvaluationFn = Callable[[DecodedDecision], np.ndarray]
DecodedBatchEvaluationFn = Callable[[Sequence[DecodedDecision]], np.ndarray]


__all__ = [
    "DecodedDecision",
    "DecisionDecodeFn",
    "DecodedEvaluationFn",
    "DecodedBatchEvaluationFn",
]
