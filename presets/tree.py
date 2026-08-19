from __future__ import annotations

from typing import Any, Mapping

from mlblack.pipeline.data_views import NumericDataView
from mlblack.problems import SupervisedEstimatorFitRegressionProblem
from mlblack.representations import (
    BoostingMechanismSpec,
    TreeMechanismSpec,
    build_tree_boosting_estimator_representation,
    build_tree_estimator_representation,
    make_sklearn_tree_factory,
    make_xgboost_factory,
)


def build_tree_estimator_search_trainer(
    data: NumericDataView,
    *,
    route: str = "random_forest",
    params: Mapping[str, Any] | None = None,
    tunable_params: tuple[str, ...] = ("max_depth",),
    bounds: Mapping[str, tuple[float, float]] | None = None,
    integer_params: tuple[str, ...] = ("max_depth",),
    population_size: int = 8,
    mutation_scale: float = 0.25,
    mechanism: TreeMechanismSpec | None = None,
    run_name: str = "tree_estimator_search",
) -> Any:
    mech = mechanism or TreeMechanismSpec(n_estimators=int((params or {}).get("n_estimators", 50)))
    representation = build_tree_estimator_representation(
        route=route,
        params=params,
        tunable_params=tunable_params,
        bounds=bounds or {"max_depth": (2, 12)},
        integer_params=integer_params,
        factory=make_sklearn_tree_factory(route),
        mechanism=mech,
    )
    problem = SupervisedEstimatorFitRegressionProblem(data)
    adapter = _build_optimization_adapter(
        "search.random_gaussian",
        population_size=population_size,
        mutation_scale=mutation_scale,
        initialization="center",
        include_center_candidate=True,
    )
    return _build_learning_solver(problem=problem, representation=representation, adapter=adapter, run_name=run_name)


def build_tree_boosting_estimator_search_trainer(
    data: NumericDataView,
    *,
    route: str = "xgboost",
    params: Mapping[str, Any] | None = None,
    tunable_params: tuple[str, ...] = ("max_depth", "learning_rate"),
    bounds: Mapping[str, tuple[float, float]] | None = None,
    integer_params: tuple[str, ...] = ("max_depth",),
    population_size: int = 8,
    mutation_scale: float = 0.2,
    mechanism: BoostingMechanismSpec | None = None,
    run_name: str = "tree_boosting_estimator_search",
) -> Any:
    mech = mechanism or BoostingMechanismSpec(n_estimators=int((params or {}).get("n_estimators", 80)))
    representation = build_tree_boosting_estimator_representation(
        route=route,
        params=params,
        tunable_params=tunable_params,
        bounds=bounds or {"max_depth": (2, 8), "learning_rate": (0.02, 0.3)},
        integer_params=integer_params,
        factory=make_xgboost_factory(),
        mechanism=mech,
    )
    problem = SupervisedEstimatorFitRegressionProblem(data)
    adapter = _build_optimization_adapter(
        "search.random_gaussian",
        population_size=population_size,
        mutation_scale=mutation_scale,
        initialization="center",
        include_center_candidate=True,
    )
    return _build_learning_solver(problem=problem, representation=representation, adapter=adapter, run_name=run_name)


def _build_optimization_adapter(method: str, **kwargs):
    from mlblack.integrations.nsgablack_optimization import build_optimization_adapter

    return build_optimization_adapter(method, **kwargs)


def _build_learning_solver(**kwargs):
    from mlblack.integrations.nsgablack_control import build_learning_solver

    return build_learning_solver(**kwargs)



