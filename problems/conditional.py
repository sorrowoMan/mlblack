from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.core.types import Feedback, UnknownState
from mlblack.pipeline.data import NumericDataView
from mlblack.problems.supervised import SupervisedRegressionProblem


class PiecewiseRegressionProblem(SupervisedRegressionProblem):
    """Regression evaluator with branch distribution metrics for piecewise models."""

    name = "piecewise_regression"

    def evaluate(self, model: Any, state: UnknownState, context: Mapping[str, Any]) -> Feedback:
        feedback = super().evaluate(model, state, context)
        metrics = dict(feedback.metrics)
        if hasattr(model, "router") and hasattr(model.router, "route"):
            train_routes = np.asarray(model.router.route(self.data.X_train), dtype=int).reshape(-1)
            metrics.update(_route_metrics(train_routes, prefix="train"))
            if self.data.X_valid is not None:
                valid_routes = np.asarray(model.router.route(self.data.X_valid), dtype=int).reshape(-1)
                metrics.update(_route_metrics(valid_routes, prefix="valid"))
        return Feedback(
            objectives=feedback.objectives,
            constraints=feedback.constraints,
            loss=feedback.loss,
            gradients=feedback.gradients,
            residuals=feedback.residuals,
            metrics=metrics,
            signals={**dict(feedback.signals), "piecewise": hasattr(model, "router")},
        )

    def describe(self) -> Mapping[str, Any]:
        base = dict(super().describe())
        base["name"] = self.name
        base["conditional"] = "piecewise"
        return base


def _route_metrics(routes: np.ndarray, *, prefix: str) -> dict[str, float]:
    if routes.size == 0:
        return {f"{prefix}.branch_count": 0.0}
    unique, counts = np.unique(routes, return_counts=True)
    total = float(np.sum(counts))
    metrics = {f"{prefix}.branch_count": float(len(unique))}
    for branch, count in zip(unique, counts):
        metrics[f"{prefix}.branch_{int(branch)}.ratio"] = float(count) / total
    return metrics


