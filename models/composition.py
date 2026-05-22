from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PredictionInputSpec:
    """Input requirement for one component model inside an integrated model."""

    key: str = ""
    kind: str = "numeric_array"
    ndim: int | None = 2
    n_features: int | None = None
    required: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: "PredictionInputSpec | Mapping[str, Any] | None") -> "PredictionInputSpec":
        if value is None:
            return cls()
        if isinstance(value, PredictionInputSpec):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError(f"cannot build PredictionInputSpec from {type(value).__name__}")

    def resolved_key(self, component_name: str) -> str:
        return str(self.key or component_name)

    def validate(self, value: Any, *, component_name: str) -> Any:
        key = str(self.kind or "numeric_array").strip().lower()
        if key in {"any", "object"}:
            return value
        if key not in {"array", "numeric_array", "tensor_like"}:
            raise ValueError(f"unsupported input kind for component {component_name!r}: {self.kind}")
        arr = np.asarray(value, dtype=float)
        if self.ndim is not None and arr.ndim != int(self.ndim):
            raise ValueError(
                f"component {component_name!r} input must be {int(self.ndim)}D, got {arr.ndim}D"
            )
        if self.n_features is not None:
            if arr.ndim < 2:
                raise ValueError(f"component {component_name!r} input has no feature axis")
            if int(arr.shape[1]) != int(self.n_features):
                raise ValueError(
                    f"component {component_name!r} input feature count {arr.shape[1]} "
                    f"does not match required {int(self.n_features)}"
                )
        return arr

    def describe(self) -> dict[str, Any]:
        return {
            "key": str(self.key),
            "kind": str(self.kind),
            "ndim": self.ndim,
            "n_features": self.n_features,
            "required": bool(self.required),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PredictionOutputSpec:
    """Output requirement consumed by the integration strategy."""

    kind: str = "point_vector"
    requires_aligned_rows: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate_prediction(self, value: Any, *, component_name: str) -> np.ndarray:
        key = str(self.kind or "point_vector").strip().lower()
        if key not in {"point_vector", "scalar_vector", "regression_vector"}:
            raise ValueError(f"unsupported integrated output kind: {self.kind}")
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            return arr.reshape(-1)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr[:, 0].reshape(-1)
        raise ValueError(
            f"component {component_name!r} prediction must be a 1D point vector "
            f"or a 2D single-column array, got shape {arr.shape}"
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "requires_aligned_rows": bool(self.requires_aligned_rows),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PredictionIOContract:
    """I/O contract for routing inputs and validating component predictions."""

    component_inputs: Mapping[str, PredictionInputSpec | Mapping[str, Any]] = field(default_factory=dict)
    shared_input_key: str = "shared"
    output: PredictionOutputSpec = field(default_factory=PredictionOutputSpec)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def shared_numeric(
        cls,
        *,
        ndim: int | None = 2,
        n_features: int | None = None,
        output: PredictionOutputSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PredictionIOContract":
        return cls(
            component_inputs={},
            output=output or PredictionOutputSpec(),
            metadata={
                **dict(metadata or {}),
                "default_input": PredictionInputSpec(ndim=ndim, n_features=n_features).describe(),
            },
        )

    @classmethod
    def by_component(
        cls,
        component_inputs: Mapping[str, PredictionInputSpec | Mapping[str, Any]],
        *,
        output: PredictionOutputSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PredictionIOContract":
        return cls(
            component_inputs=dict(component_inputs),
            output=output or PredictionOutputSpec(),
            metadata=dict(metadata or {}),
        )

    def input_spec_for(self, component_name: str) -> PredictionInputSpec:
        raw = dict(self.component_inputs).get(str(component_name))
        if raw is not None:
            return PredictionInputSpec.from_value(raw)
        default = dict(self.metadata).get("default_input")
        if isinstance(default, Mapping):
            return PredictionInputSpec.from_value(default)
        return PredictionInputSpec()

    def component_input(self, inputs: Any, *, component_name: str) -> Any:
        spec = self.input_spec_for(component_name)
        if isinstance(inputs, Mapping):
            input_map = dict(inputs)
            key = spec.resolved_key(component_name)
            if key in input_map:
                return spec.validate(input_map[key], component_name=component_name)
            if self.shared_input_key in input_map:
                return spec.validate(input_map[self.shared_input_key], component_name=component_name)
            if bool(spec.required):
                raise KeyError(
                    f"missing input for component {component_name!r}; expected key {key!r} "
                    f"or shared key {self.shared_input_key!r}"
                )
            return None
        return spec.validate(inputs, component_name=component_name)

    def describe(self) -> dict[str, Any]:
        return {
            "component_inputs": {
                str(name): PredictionInputSpec.from_value(spec).describe()
                for name, spec in dict(self.component_inputs).items()
            },
            "shared_input_key": str(self.shared_input_key),
            "output": self.output.describe(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PredictionIntegrationSpec:
    """How named model predictions are combined into one prediction."""

    kind: str = "additive"
    component_order: Sequence[str] = field(default_factory=tuple)
    weights: Mapping[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def resolved_order(self, components: Mapping[str, Any]) -> tuple[str, ...]:
        order = tuple(str(name) for name in self.component_order)
        if order:
            missing = [name for name in order if name not in components]
            if missing:
                raise KeyError(f"integration component(s) not found: {missing}")
            return order
        return tuple(str(name) for name in components.keys())

    def weight_for(self, name: str) -> float:
        return float(dict(self.weights).get(str(name), 1.0))

    def describe(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "component_order": list(tuple(str(name) for name in self.component_order)),
            "weights": {str(key): float(value) for key, value in dict(self.weights).items()},
            "intercept": float(self.intercept),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class IntegratedPredictionModel:
    """Prediction model assembled from already-trained component models.

    This is a model semantic boundary, not a workflow runner. The owner of the
    training sequence remains the outer orchestrator; this object only defines
    how fitted component models are read at inference/evaluation time.
    """

    components: Mapping[str, Any]
    integration: PredictionIntegrationSpec = field(default_factory=PredictionIntegrationSpec)
    io_contract: PredictionIOContract = field(default_factory=PredictionIOContract)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("IntegratedPredictionModel requires at least one component")
        for name, model in dict(self.components).items():
            if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
                raise TypeError(f"component {name!r} does not expose predict(X)")

    def component_predictions(self, inputs: Any) -> dict[str, np.ndarray]:
        predictions: dict[str, np.ndarray] = {}
        expected_rows: int | None = None
        for name in self.integration.resolved_order(self.components):
            model = self.components[name]
            model_input = self.io_contract.component_input(inputs, component_name=str(name))
            pred = self.io_contract.output.validate_prediction(
                model.predict(model_input),
                component_name=str(name),
            )
            if expected_rows is None or not bool(self.io_contract.output.requires_aligned_rows):
                expected_rows = int(pred.shape[0])
            elif int(pred.shape[0]) != int(expected_rows):
                raise ValueError(f"component {name!r} returned {pred.shape[0]} rows, expected {expected_rows}")
            predictions[str(name)] = pred
        return predictions

    def predict(self, inputs: Any) -> np.ndarray:
        predictions = self.component_predictions(inputs)
        kind = str(self.integration.kind or "additive").strip().lower()
        if kind in {"additive", "sum", "residual_sum"}:
            out = np.full(next(iter(predictions.values())).shape, float(self.integration.intercept), dtype=float)
            for name, pred in predictions.items():
                out = out + (self.integration.weight_for(name) * pred)
            return np.asarray(out, dtype=float).reshape(-1)
        if kind in {"mean", "average"}:
            total_weight = 0.0
            out = np.full(next(iter(predictions.values())).shape, float(self.integration.intercept), dtype=float)
            for name, pred in predictions.items():
                weight = self.integration.weight_for(name)
                total_weight += abs(weight)
                out = out + (weight * pred)
            if total_weight <= 0.0:
                raise ValueError("mean integration requires non-zero total weight")
            return np.asarray(out / total_weight, dtype=float).reshape(-1)
        raise ValueError(f"unsupported prediction integration kind: {self.integration.kind}")

    def describe(self) -> dict[str, Any]:
        return {
            "model_type": "integrated_prediction",
            "components": {
                str(name): _describe_model(model)
                for name, model in dict(self.components).items()
            },
            "integration": self.integration.describe(),
            "io_contract": self.io_contract.describe(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PredictionIntegrationComponent:
    """Builds IntegratedPredictionModel from named fitted component models."""

    integration: PredictionIntegrationSpec = field(default_factory=PredictionIntegrationSpec)
    io_contract: PredictionIOContract = field(default_factory=PredictionIOContract)

    @classmethod
    def additive(
        cls,
        *,
        component_order: Sequence[str] = (),
        weights: Mapping[str, float] | None = None,
        intercept: float = 0.0,
        io_contract: PredictionIOContract | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PredictionIntegrationComponent":
        return cls(
            integration=PredictionIntegrationSpec(
                kind="additive",
                component_order=tuple(component_order),
                weights=dict(weights or {}),
                intercept=float(intercept),
                metadata=dict(metadata or {}),
            ),
            io_contract=io_contract or PredictionIOContract(),
        )

    def compose(
        self,
        components: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> IntegratedPredictionModel:
        return IntegratedPredictionModel(
            components=dict(components),
            integration=self.integration,
            io_contract=self.io_contract,
            metadata=dict(metadata or {}),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": "prediction_integration",
            "integration": self.integration.describe(),
            "io_contract": self.io_contract.describe(),
        }


def _describe_model(model: Any) -> dict[str, Any]:
    describe = getattr(model, "describe", None)
    if callable(describe):
        try:
            payload = describe()
            if isinstance(payload, Mapping):
                return dict(payload)
        except Exception:
            pass
    return {"model_type": type(model).__name__}


__all__ = [
    "IntegratedPredictionModel",
    "PredictionIOContract",
    "PredictionInputSpec",
    "PredictionIntegrationComponent",
    "PredictionIntegrationSpec",
    "PredictionOutputSpec",
]
