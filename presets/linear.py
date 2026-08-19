from __future__ import annotations

from typing import Any

from mlblack.pipeline.data_views import NumericDataView
from mlblack.representations.heads import IntervalHead
from mlblack.problems import SupervisedIntervalRegressionProblem, SupervisedRegressionProblem
from mlblack.representations import (
    LinearPointConfig,
    LinearPointRepresentation,
    OrthogonalPointConfig,
    OrthogonalPointLinearRepresentation,
)


def build_linear_point_trainer(
    data: NumericDataView,
    *,
    method: str = "gradient.sgd",
    learning_rate: float = 0.05,
    l2: float = 0.0,
    use_valid_objective: bool = True,
    init_scale: float = 0.01,
    random_seed: int = 42,
    run_name: str = "linear_point",
) -> Any:
    """Build the direct ``[intercept, weights...]`` optimization path.

    The model family and data semantics stay in mlblack.  ``method`` is a
    stable optimization identifier resolved to the nsgablack gradient
    Adapter, while the analytic linear-model evaluation remains owned by the
    ML Problem.
    """

    representation = LinearPointRepresentation.from_data(
        data.X_train,
        feature_names=data.effective_feature_names,
        config=LinearPointConfig(
            n_features=int(data.X_train.shape[1]),
            feature_names=tuple(data.effective_feature_names),
            init_scale=float(init_scale),
            random_seed=int(random_seed),
        ),
    )
    problem = SupervisedRegressionProblem(
        data,
        l2=l2,
        complexity_weight=0.0,
        use_valid_objective=use_valid_objective,
    )
    return _build_gradient_trainer(
        problem=problem,
        representation=representation,
        method=method,
        compute_backend="problem",
        learning_rate=learning_rate,
        max_gradient_norm=1e3,
        run_name=run_name,
    )


def build_orthogonal_linear_point_trainer(
    data: NumericDataView,
    *,
    learning_rate: float = 0.05,
    l2: float = 0.0,
    complexity_weight: float = 0.0,
    max_components: int | None = None,
    energy_threshold: float | None = 0.999,
    run_name: str = "orthogonal_linear_point",
) -> Any:
    """Preset: orthogonal feature map + linear point model + GD adapter."""

    rep_cfg = OrthogonalPointConfig(
        max_components=max_components,
        energy_threshold=energy_threshold,
    )
    representation = OrthogonalPointLinearRepresentation.from_data(
        data.X_train,
        feature_names=data.effective_feature_names,
        config=rep_cfg,
    )
    problem = SupervisedRegressionProblem(
        data,
        l2=l2,
        complexity_weight=complexity_weight,
    )
    return _build_gradient_trainer(
        problem=problem,
        representation=representation,
        method="gradient.sgd",
        compute_backend="problem",
        learning_rate=learning_rate,
        max_gradient_norm=1e3,
        run_name=run_name,
    )


def build_orthogonal_linear_interval_trainer(
    data: NumericDataView,
    *,
    target_coverage: float = 0.9,
    width_weight: float = 1.0,
    miss_weight: float = 10.0,
    max_components: int | None = None,
    energy_threshold: float | None = 0.999,
    population_size: int = 24,
    mutation_scale: float = 0.2,
    run_name: str = "orthogonal_linear_interval",
) -> Any:
    """Preset: orthogonal linear base decoder + interval head + black-box search."""

    rep_cfg = OrthogonalPointConfig(
        max_components=max_components,
        energy_threshold=energy_threshold,
    )
    representation = OrthogonalPointLinearRepresentation.from_data(
        data.X_train,
        feature_names=data.effective_feature_names,
        config=rep_cfg,
        head=IntervalHead(enforce_order=True),
    )
    problem = SupervisedIntervalRegressionProblem(
        data,
        target_coverage=target_coverage,
        width_weight=width_weight,
        miss_weight=miss_weight,
    )
    adapter = _build_optimization_adapter(
        "search.random_gaussian",
        population_size=population_size,
        mutation_scale=mutation_scale,
    )
    return _build_learning_solver(
        problem=problem,
        representation=representation,
        adapter=adapter,
        run_name=run_name,
    )


def _build_gradient_trainer(**kwargs):
    from mlblack.integrations.nsgablack_gradient import build_gradient_trainer

    return build_gradient_trainer(**kwargs)


def _build_optimization_adapter(method: str, **kwargs):
    from mlblack.integrations.nsgablack_optimization import build_optimization_adapter

    return build_optimization_adapter(method, **kwargs)


def _build_learning_solver(**kwargs):
    from mlblack.integrations.nsgablack_control import build_learning_solver

    return build_learning_solver(**kwargs)



