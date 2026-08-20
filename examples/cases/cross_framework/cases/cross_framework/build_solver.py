from __future__ import annotations

import argparse
from typing import Any, Mapping

from nsgablack.adapters.gaussian_search import (
    GaussianSearchAdapter,
    GaussianSearchConfig,
)
from nsgablack.core.composable_solver import ComposableSolver

from mlblack.project.scaffold import print_case_check

from pipeline.main import build_pipeline
from problem.example_problem import NestedTrainerProblem


class CrossFrameworkOuterSolver(ComposableSolver):
    """Outer optimizer that forwards the injected Case runtime to its Problem."""

    def set_case_runtime(self, runtime):
        self.case_runtime = runtime
        self.problem.set_case_runtime(runtime)
        return self


def build_solver(
    config=None,
    *,
    resource_context: Mapping[str, Any] | None = None,
    component_overrides: Mapping[str, Any] | None = None,
) -> CrossFrameworkOuterSolver:
    del config
    overrides = dict(component_overrides or {})
    problem_builder = overrides.pop("problem", NestedTrainerProblem)
    pipeline_builder = overrides.pop("pipeline", build_pipeline)
    adapter_builder = overrides.pop("adapter", GaussianSearchAdapter)
    if overrides:
        raise ValueError(f"unsupported cross-framework overrides: {sorted(overrides)}")

    problem = problem_builder() if callable(problem_builder) else problem_builder
    pipeline = (
        pipeline_builder(problem, resource_context=resource_context)
        if callable(pipeline_builder)
        else pipeline_builder
    )
    adapter = (
        adapter_builder(
            GaussianSearchConfig(
                population_size=2,
                mutation_scale=0.35,
                random_seed=17,
                initialization="population",
                objective_aggregation="first",
            )
        )
        if adapter_builder is GaussianSearchAdapter
        else adapter_builder() if callable(adapter_builder) else adapter_builder
    )
    solver = CrossFrameworkOuterSolver(
        problem,
        adapter=adapter,
        representation_pipeline=pipeline,
    )
    solver.max_steps = 1
    solver.set_random_seed(17)
    if resource_context is not None:
        solver.set_resource_context(resource_context)
    return solver


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the formal nested cross-framework Case")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    solver = build_solver()
    if args.check:
        print_case_check(solver)
        return 0
    print(solver.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CrossFrameworkOuterSolver", "build_solver", "main"]
