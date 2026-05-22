from __future__ import annotations

from mlblack.adapters import GradientDescentAdapter, GradientDescentConfig, RandomSearchAdapter, RandomSearchConfig
from mlblack.core import Trainer
from mlblack.pipeline.data import NumericDataView
from mlblack.representations.heads import IntervalHead
from mlblack.problems import SupervisedIntervalRegressionProblem, SupervisedRegressionProblem
from mlblack.representations import OrthogonalPointConfig, OrthogonalPointLinearRepresentation


def build_orthogonal_linear_point_trainer(
    data: NumericDataView,
    *,
    learning_rate: float = 0.05,
    l2: float = 0.0,
    complexity_weight: float = 0.0,
    max_components: int | None = None,
    energy_threshold: float | None = 0.999,
    run_name: str = "orthogonal_linear_point",
) -> Trainer:
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
    adapter = GradientDescentAdapter(
        GradientDescentConfig(learning_rate=learning_rate),
    )
    return Trainer(
        problem=problem,
        representation=representation,
        adapter=adapter,
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
) -> Trainer:
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
    adapter = RandomSearchAdapter(
        RandomSearchConfig(population_size=population_size, mutation_scale=mutation_scale),
    )
    return Trainer(
        problem=problem,
        representation=representation,
        adapter=adapter,
        run_name=run_name,
    )



