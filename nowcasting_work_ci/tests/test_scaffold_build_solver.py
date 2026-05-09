from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[3]
NSGABLACK_ROOT = DESKTOP / "nsgablack"
MLBLACK_ROOT = DESKTOP / "mlblack"
if str(NSGABLACK_ROOT) not in sys.path:
    sys.path.insert(0, str(NSGABLACK_ROOT))
if str(MLBLACK_ROOT) not in sys.path:
    sys.path.insert(0, str(MLBLACK_ROOT))

from nsgablack.core.base import BlackBoxProblem
from nowcasting_work_ci.build_solver import NowcastingSolverBuildConfig, build_nowcasting_solver


class _DummyProblem(BlackBoxProblem):
    def __init__(self) -> None:
        super().__init__(
            name="dummy",
            dimension=2,
            bounds=[(-1.0, 1.0), (-1.0, 1.0)],
            objectives=["minimize", "minimize", "minimize"],
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1)
        return np.asarray(
            [
                float(np.sum(arr**2)),
                float(np.sum(np.abs(arr))),
                float(np.max(np.abs(arr))),
            ],
            dtype=float,
        )


def _cfg(*, batched: bool) -> NowcastingSolverBuildConfig:
    return NowcastingSolverBuildConfig(
        strategy="nsga2",
        pop_size=4,
        generations=2,
        portfolio_phases_csv="nsga2,moead,vns",
        portfolio_weights_csv="2,1,1",
        moead_neighborhood_size=2,
        moead_delta=0.9,
        moead_nr=1,
        vns_k_max=2,
        vns_batch_size=4,
        batched_eval_enabled=bool(batched),
        seed=42,
    )


def test_build_solver_registers_eval_proxy_and_pipeline() -> None:
    problem = _DummyProblem()

    def _batch_eval(population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pop = np.asarray(population, dtype=float)
        if pop.ndim == 1:
            pop = pop.reshape(1, -1)
        n = int(pop.shape[0])
        return np.ones((n, 3), dtype=float), np.zeros((n,), dtype=float)

    solver, outer_meta, _adapter = build_nowcasting_solver(
        problem=problem,
        cfg=_cfg(batched=True),
        eval_batch_proxy_fn=_batch_eval,
    )
    assert solver.representation_pipeline is not None
    assert int(solver.max_steps) == int(outer_meta.get("max_generations"))
    providers = tuple(solver.evaluation_mediator.list_providers())
    assert len(providers) >= 1

    objs, vios = solver.evaluate_population(np.asarray([[0.0, 0.0], [0.5, -0.5]], dtype=float))
    assert objs.shape == (2, 3)
    assert vios.shape == (2,)


def test_build_solver_without_batch_proxy() -> None:
    problem = _DummyProblem()
    solver, _outer_meta, _adapter = build_nowcasting_solver(
        problem=problem,
        cfg=_cfg(batched=False),
        eval_batch_proxy_fn=None,
    )
    providers = tuple(solver.evaluation_mediator.list_providers())
    assert len(providers) == 0
