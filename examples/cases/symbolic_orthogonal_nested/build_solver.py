from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    from _bootstrap import ensure_case_importable  # noqa: E402
    ensure_case_importable(Path(__file__))
    from config import SymbolicOrthogonalNestedCaseConfig  # noqa: E402
    from pipeline import build_representation_pipeline  # noqa: E402
    from problem import build_stage1_problem, build_stage2_problem, build_symbolic_regression_data  # noqa: E402
else:
    from ._bootstrap import ensure_case_importable  # noqa: E402
    ensure_case_importable(Path(__file__))
    from .config import SymbolicOrthogonalNestedCaseConfig  # noqa: E402
    from .pipeline import build_representation_pipeline  # noqa: E402
    from .problem import build_stage1_problem, build_stage2_problem, build_symbolic_regression_data  # noqa: E402

from mlblack.integrations.nsgablack_symbolic import OrthogonalBasisSetArtifact  # noqa: E402
from mlblack.pipeline.data import NumericDataView  # noqa: E402
from nsgablack.adapters import NSGA2Adapter, NSGA2Config  # noqa: E402
from nsgablack.core.evolution_solver import EvolutionSolver  # noqa: E402


def build_stage1_solver(
    cfg: SymbolicOrthogonalNestedCaseConfig | None = None,
    *,
    suite_id: str,
    data: NumericDataView | None = None,
) -> EvolutionSolver:
    config = cfg or SymbolicOrthogonalNestedCaseConfig()
    output_dir = config.output_root(suite_id) / "stage1"
    data_view = data or build_symbolic_regression_data(
        n_samples=int(config.n_samples),
        valid_fraction=float(config.valid_fraction),
        seed=int(config.seed),
    )
    problem = build_stage1_problem(config, data_view, output_dir=output_dir)
    return _build_solver(
        problem=problem,
        cfg=config,
        pop_size=int(config.stage1_pop_size),
        offspring_size=int(config.stage1_offspring_size),
        generations=int(config.stage1_generations),
        name="symbolic_orthogonal_stage1_nsga2",
        output_dir=output_dir,
        data=data_view,
    )


def build_stage2_solver(
    cfg: SymbolicOrthogonalNestedCaseConfig | None = None,
    *,
    suite_id: str,
    basis_artifact: OrthogonalBasisSetArtifact,
    data: NumericDataView | None = None,
) -> EvolutionSolver:
    config = cfg or SymbolicOrthogonalNestedCaseConfig()
    output_dir = config.output_root(suite_id) / "stage2"
    data_view = data or build_symbolic_regression_data(
        n_samples=int(config.n_samples),
        valid_fraction=float(config.valid_fraction),
        seed=int(config.seed),
    )
    problem = build_stage2_problem(config, data_view, basis_artifact=basis_artifact, output_dir=output_dir)
    return _build_solver(
        problem=problem,
        cfg=config,
        pop_size=int(config.stage2_pop_size),
        offspring_size=int(config.stage2_offspring_size),
        generations=int(config.stage2_generations),
        name="symbolic_orthogonal_stage2_nsga2",
        output_dir=output_dir,
        data=data_view,
    )


def _build_solver(
    *,
    problem,
    cfg: SymbolicOrthogonalNestedCaseConfig,
    pop_size: int,
    offspring_size: int,
    generations: int,
    name: str,
    output_dir: Path,
    data: NumericDataView,
) -> EvolutionSolver:
    pipeline = build_representation_pipeline(problem, mutation_sigma=float(cfg.mutation_sigma))
    adapter = NSGA2Adapter(
        NSGA2Config(
            population_size=max(4, int(pop_size)),
            offspring_size=max(2, int(offspring_size)),
            crossover_rate=float(cfg.crossover_rate),
            objective_aggregation="sum",
        ),
        name=name,
    )
    solver = EvolutionSolver(
        problem=problem,
        adapter=adapter,
        representation_pipeline=pipeline,
        pop_size=max(4, int(pop_size)),
        max_generations=max(1, int(generations)),
        mutation_rate=0.2,
        crossover_rate=float(cfg.crossover_rate),
        random_seed=int(cfg.seed),
        enable_progress_log=False,
        enable_parallel_evaluation=False,
    )
    solver.symbolic_orthogonal_nested_output_dir = output_dir
    solver.symbolic_orthogonal_nested_data = data
    return solver


__all__ = [
    "SymbolicOrthogonalNestedCaseConfig",
    "build_stage1_solver",
    "build_stage2_solver",
]
