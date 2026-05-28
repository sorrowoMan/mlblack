from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from mlblack.adapters import RandomSearchAdapter, RandomSearchConfig
from mlblack.core import Trainer
from mlblack.pipeline.data_views import NumericDataView
from mlblack.representations.heads import BinaryLogisticHead, SoftmaxHead
from mlblack.problems import SupervisedClassificationProblem
from mlblack.representations import OrthogonalPointConfig, OrthogonalPointLinearRepresentation


def build_orthogonal_logistic_classification_trainer(
    data: NumericDataView,
    *,
    classes: Sequence[Any] = (0, 1),
    objective_metrics: Sequence[str] = ("log_loss", "error_rate", "auc_roc"),
    max_components: int | None = None,
    energy_threshold: float | None = 0.999,
    population_size: int = 24,
    mutation_scale: float = 0.2,
    temperature: float = 1.0,
    threshold: float = 0.5,
    run_name: str = "orthogonal_logistic_classification",
) -> Trainer:
    rep_cfg = OrthogonalPointConfig(max_components=max_components, energy_threshold=energy_threshold)
    representation = OrthogonalPointLinearRepresentation.from_data(
        data.X_train,
        feature_names=data.effective_feature_names,
        config=rep_cfg,
        head=BinaryLogisticHead(temperature=temperature, threshold=threshold, classes=tuple(classes)),
    )
    problem = SupervisedClassificationProblem(data, objective_metrics=objective_metrics, positive_label=tuple(classes)[-1] if classes else None)
    adapter = RandomSearchAdapter(RandomSearchConfig(population_size=population_size, mutation_scale=mutation_scale))
    return Trainer(problem=problem, representation=representation, adapter=adapter, run_name=run_name)


def build_orthogonal_softmax_classification_trainer(
    data: NumericDataView,
    *,
    classes: Sequence[Any] | None = None,
    objective_metrics: Sequence[str] = ("log_loss", "error_rate", "f1_macro"),
    max_components: int | None = None,
    energy_threshold: float | None = 0.999,
    population_size: int = 32,
    mutation_scale: float = 0.2,
    temperature: float = 1.0,
    run_name: str = "orthogonal_softmax_classification",
) -> Trainer:
    labels = tuple(classes) if classes is not None else tuple(np.unique(data.y_train).tolist())
    rep_cfg = OrthogonalPointConfig(max_components=max_components, energy_threshold=energy_threshold)
    representation = OrthogonalPointLinearRepresentation.from_data(
        data.X_train,
        feature_names=data.effective_feature_names,
        config=rep_cfg,
        head=SoftmaxHead(n_classes=max(2, len(labels)), temperature=temperature, classes=labels),
    )
    problem = SupervisedClassificationProblem(data, objective_metrics=objective_metrics)
    adapter = RandomSearchAdapter(RandomSearchConfig(population_size=population_size, mutation_scale=mutation_scale))
    return Trainer(problem=problem, representation=representation, adapter=adapter, run_name=run_name)


