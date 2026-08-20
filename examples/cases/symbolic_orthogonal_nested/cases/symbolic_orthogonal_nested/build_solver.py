from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    _THIS_DIR = Path(__file__).resolve().parent
    from _bootstrap import ensure_case_importable  # noqa: E402
    ensure_case_importable(Path(__file__))
    _this_dir_str = str(_THIS_DIR)
    if _this_dir_str in sys.path:
        sys.path.remove(_this_dir_str)
    sys.path.insert(0, _this_dir_str)
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
from mlblack.pipeline.data_views import NumericDataView  # noqa: E402
from nsgablack.adapters import NSGA2Adapter, NSGA2Config  # noqa: E402
from nsgablack.core.evolution_solver import EvolutionSolver  # noqa: E402


def build_stage1_solver(
    cfg: SymbolicOrthogonalNestedCaseConfig | None = None,
    *,
    suite_id: str,
    data: NumericDataView | None = None,
    resource_context=None,
    component_overrides=None,
) -> EvolutionSolver:
    config = cfg or SymbolicOrthogonalNestedCaseConfig()
    output_dir = config.output_root(suite_id) / "stage1"
    data_view = data or build_symbolic_regression_data(
        n_samples=int(config.n_samples),
        valid_fraction=float(config.valid_fraction),
        seed=int(config.seed),
    )
    overrides = dict(component_overrides or {})
    problem = overrides.pop(
        "problem",
        build_stage1_problem(config, data_view, output_dir=output_dir),
    )
    return _build_solver(
        problem=problem,
        cfg=config,
        pop_size=int(config.stage1_pop_size),
        offspring_size=int(config.stage1_offspring_size),
        generations=int(config.stage1_generations),
        name="symbolic_orthogonal_stage1_nsga2",
        output_dir=output_dir,
        data=data_view,
        resource_context=resource_context,
        component_overrides=overrides,
    )


def build_stage2_solver(
    cfg: SymbolicOrthogonalNestedCaseConfig | None = None,
    *,
    suite_id: str,
    basis_artifact: OrthogonalBasisSetArtifact,
    data: NumericDataView | None = None,
    resource_context=None,
    component_overrides=None,
) -> EvolutionSolver:
    config = cfg or SymbolicOrthogonalNestedCaseConfig()
    output_dir = config.output_root(suite_id) / "stage2"
    data_view = data or build_symbolic_regression_data(
        n_samples=int(config.n_samples),
        valid_fraction=float(config.valid_fraction),
        seed=int(config.seed),
    )
    overrides = dict(component_overrides or {})
    problem = overrides.pop(
        "problem",
        build_stage2_problem(
            config,
            data_view,
            basis_artifact=basis_artifact,
            output_dir=output_dir,
        ),
    )
    return _build_solver(
        problem=problem,
        cfg=config,
        pop_size=int(config.stage2_pop_size),
        offspring_size=int(config.stage2_offspring_size),
        generations=int(config.stage2_generations),
        name="symbolic_orthogonal_stage2_nsga2",
        output_dir=output_dir,
        data=data_view,
        resource_context=resource_context,
        component_overrides=overrides,
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
    resource_context=None,
    component_overrides=None,
) -> EvolutionSolver:
    overrides = dict(component_overrides or {})
    pipeline_override = overrides.pop("pipeline", None)
    pipeline = (
        build_representation_pipeline(
            problem,
            mutation_sigma=float(cfg.mutation_sigma),
        )
        if pipeline_override is None
        else pipeline_override(problem, mutation_sigma=float(cfg.mutation_sigma))
        if callable(pipeline_override)
        else pipeline_override
    )
    adapter_override = overrides.pop("adapter", None)
    adapter = adapter_override or NSGA2Adapter(
            NSGA2Config(
                population_size=max(4, int(pop_size)),
                offspring_size=max(2, int(offspring_size)),
                crossover_rate=float(cfg.crossover_rate),
                objective_aggregation="sum",
            ),
            name=name,
        )
    solver_factory = overrides.pop("solver", EvolutionSolver)
    if overrides:
        raise ValueError(
            "unsupported symbolic_orthogonal_nested component overrides: "
            f"{sorted(overrides)}"
        )
    solver = solver_factory(
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
        resource_context=resource_context,
    )
    solver.symbolic_orthogonal_nested_output_dir = output_dir
    solver.symbolic_orthogonal_nested_data = data
    return solver


def build_solver(
    cfg: SymbolicOrthogonalNestedCaseConfig | None = None,
    *,
    suite_id: str = "symbolic_orthogonal_nested",
    stage: int = 1,
    basis_artifact: OrthogonalBasisSetArtifact | None = None,
    data: NumericDataView | None = None,
    resource_context=None,
    component_overrides=None,
) -> EvolutionSolver:
    """Canonical unified scaffold entry for the nested symbolic case."""

    if int(stage) == 1:
        return build_stage1_solver(
            cfg,
            suite_id=suite_id,
            data=data,
            resource_context=resource_context,
            component_overrides=component_overrides,
        )
    if basis_artifact is None:
        raise ValueError("stage=2 requires basis_artifact")
    return build_stage2_solver(
        cfg,
        suite_id=suite_id,
        basis_artifact=basis_artifact,
        data=data,
        resource_context=resource_context,
        component_overrides=component_overrides,
    )


__all__ = [
    "SymbolicOrthogonalNestedCaseConfig",
    "build_solver",
    "build_stage1_solver",
    "build_stage2_solver",
]
