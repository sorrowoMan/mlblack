from __future__ import annotations

from typing import Any, Callable

import numpy as np

from problem import BatchEvaluationProxyProvider


def register_batch_evaluation_proxy(
    solver: Any,
    *,
    evaluate_population_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None,
    enabled: bool,
) -> None:
    if not bool(enabled) or not callable(evaluate_population_fn):
        return
    provider = BatchEvaluationProxyProvider(evaluate_population_fn=evaluate_population_fn)
    solver.evaluation_mediator.register_provider(provider)
