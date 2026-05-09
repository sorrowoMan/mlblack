from __future__ import annotations

import numpy as np

from nsgablack.core.base import BlackBoxProblem

from nowcasting_work_ci.nsgablack_side.pipeline import build_pipeline, seed_pipeline_rngs


class _DummyProblem(BlackBoxProblem):
    def __init__(self) -> None:
        super().__init__(
            name="dummy_pipeline_seed_problem",
            dimension=4,
            bounds=[(-1.0, 1.0)] * 4,
            objectives=["minimize", "minimize", "minimize"],
        )

    def evaluate(self, x):  # type: ignore[no-untyped-def]
        arr = np.asarray(x, dtype=float).reshape(-1)
        return np.asarray([float(np.sum(arr**2)), float(np.sum(np.abs(arr))), float(np.max(np.abs(arr)))])


def test_seed_pipeline_rngs_makes_initializer_and_mutator_reproducible() -> None:
    problem = _DummyProblem()

    pipeline_a = build_pipeline(problem, base_sigma=0.18, sigma_key="mutation_sigma")
    seed_pipeline_rngs(pipeline_a, 123)
    x_a = np.asarray(pipeline_a.init(problem, {}), dtype=float)
    y_a = np.asarray(pipeline_a.mutate(x_a, {"mutation_sigma": 0.3}), dtype=float)

    pipeline_b = build_pipeline(problem, base_sigma=0.18, sigma_key="mutation_sigma")
    seed_pipeline_rngs(pipeline_b, 123)
    x_b = np.asarray(pipeline_b.init(problem, {}), dtype=float)
    y_b = np.asarray(pipeline_b.mutate(x_b, {"mutation_sigma": 0.3}), dtype=float)

    assert np.allclose(x_a, x_b)
    assert np.allclose(y_a, y_b)
