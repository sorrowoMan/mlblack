from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nsgablack.core.base import BlackBoxProblem
from nsgablack.core.composable_solver import ComposableSolver

from .adapter import build_outer_adapter
from .evaluation import register_batch_evaluation_proxy
from .pipeline import build_pipeline, seed_pipeline_rngs
from .plugins import build_flow_plugins, build_ops_plugins


@dataclass(frozen=True)
class NowcastingSolverBuildConfig:
    strategy: str
    pop_size: int
    generations: int
    portfolio_phases_csv: str
    portfolio_weights_csv: str
    moead_neighborhood_size: int
    moead_delta: float
    moead_nr: int
    vns_k_max: int
    vns_batch_size: int
    batched_eval_enabled: bool = True
    seed: int | None = None
    max_steps: int | None = None


def build_nowcasting_solver(
    *,
    problem: BlackBoxProblem,
    cfg: NowcastingSolverBuildConfig,
    eval_batch_proxy_fn: Any | None = None,
) -> tuple[ComposableSolver, dict[str, Any], Any]:
    # 1) problem (already built by caller)
    # 2) pipeline
    adapter, outer_meta = build_outer_adapter(
        strategy=str(cfg.strategy),
        pop_size=int(cfg.pop_size),
        generations=int(cfg.generations),
        seed=None if cfg.seed is None else int(cfg.seed),
        portfolio_phases_csv=str(cfg.portfolio_phases_csv),
        portfolio_weights_csv=str(cfg.portfolio_weights_csv),
        moead_neighborhood_size=int(cfg.moead_neighborhood_size),
        moead_delta=float(cfg.moead_delta),
        moead_nr=int(cfg.moead_nr),
        vns_k_max=int(cfg.vns_k_max),
        vns_batch_size=int(cfg.vns_batch_size),
    )
    pipeline = build_pipeline(problem, base_sigma=0.18, sigma_key="mutation_sigma")
    seed_pipeline_rngs(pipeline, None if cfg.seed is None else int(cfg.seed))
    solver = ComposableSolver(
        problem,
        adapter=adapter,
        representation_pipeline=pipeline,
    )

    # 3) evaluation provider (L4)
    register_batch_evaluation_proxy(
        solver,
        evaluate_population_fn=eval_batch_proxy_fn,
        enabled=bool(cfg.batched_eval_enabled),
    )
    # 4) plugins (L3 flow then L1/L2 ops)
    for plugin in build_flow_plugins():
        solver.add_plugin(plugin)
    for plugin in build_ops_plugins():
        solver.add_plugin(plugin)

    target_steps = int(outer_meta.get("max_generations", cfg.generations))
    if cfg.max_steps is not None:
        target_steps = int(max(1, int(cfg.max_steps)))
    solver.max_steps = int(max(1, target_steps))
    if cfg.seed is not None:
        solver.set_random_seed(int(cfg.seed))
    return solver, outer_meta, adapter


__all__ = ["NowcastingSolverBuildConfig", "build_nowcasting_solver"]
