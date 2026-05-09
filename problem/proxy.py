from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np


BatchEvalFn = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


@dataclass
class BatchEvaluationProxyProvider:
    """NSGABLACK L4 provider that proxies population evaluation to MLBLACK side."""

    evaluate_population_fn: BatchEvalFn
    name: str = "mlblack_batch_eval_proxy"
    semantic_mode: str = "equivalent"
    priority: int = 90

    def can_handle_individual(self, solver: Any, x: np.ndarray, context: Mapping[str, Any]) -> bool:
        _ = solver, x, context
        return False

    def evaluate_individual(
        self,
        solver: Any,
        x: np.ndarray,
        context: Mapping[str, Any],
        individual_id: Optional[int] = None,
    ) -> Optional[tuple[np.ndarray, float]]:
        _ = solver, x, context, individual_id
        return None

    def can_handle_population(self, solver: Any, population: np.ndarray, context: Mapping[str, Any]) -> bool:
        _ = solver, context
        pop = np.asarray(population, dtype=float)
        return bool(pop.ndim in (1, 2))

    def evaluate_population(
        self,
        solver: Any,
        population: np.ndarray,
        context: Mapping[str, Any],
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        _ = solver, context
        return self.evaluate_population_fn(np.asarray(population, dtype=float))


__all__ = ["BatchEvaluationProxyProvider"]
