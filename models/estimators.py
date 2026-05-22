from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np


EstimatorFactory = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class EstimatorSpecModel:
    """Decoded external estimator specification.

    It is intentionally not a trainer. It is a model-side object that knows how
    to construct an estimator for an evaluator or external-fit adapter.
    """

    family: str
    route: str
    params: Mapping[str, Any] = field(default_factory=dict)
    factory: EstimatorFactory | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def mechanisms(self) -> Mapping[str, Any]:
        value = self.metadata.get("mechanisms", {})
        return dict(value) if isinstance(value, Mapping) else {}

    def build_estimator(self) -> Any:
        if self.factory is not None:
            return self.factory(dict(self.params))
        raise RuntimeError(
            f"{self.family}/{self.route} requires an estimator factory. "
            "Install the optional backend or provide a factory."
        )

    def describe(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "route": self.route,
            "params": dict(self.params),
            "mechanisms": dict(self.mechanisms),
            "metadata": dict(self.metadata),
        }


@dataclass
class FittedEstimatorModel:
    """Prediction wrapper around an already fitted external estimator."""

    estimator: Any
    family: str
    route: str
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.estimator, "predict"):
            raise TypeError("wrapped estimator does not expose predict(...)")
        return np.asarray(self.estimator.predict(np.asarray(X, dtype=float))).reshape(-1)

    def describe(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "route": self.route,
            "params": dict(self.params),
            "estimator_type": type(self.estimator).__name__,
            "metadata": dict(self.metadata),
        }

    def fitted_state_summary(self) -> dict[str, Any]:
        summary = {"estimator_type": type(self.estimator).__name__}
        for attr in ("n_features_in_", "n_outputs_", "best_iteration", "best_iteration_", "n_estimators_"):
            if hasattr(self.estimator, attr):
                try:
                    summary[attr] = getattr(self.estimator, attr)
                except Exception:
                    pass
        if hasattr(self.estimator, "get_booster"):
            summary["has_booster"] = True
        if hasattr(self.estimator, "estimators_"):
            try:
                summary["num_estimators"] = len(self.estimator.estimators_)
            except Exception:
                pass
        return summary
