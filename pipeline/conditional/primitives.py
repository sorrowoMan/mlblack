from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ConditionalPrimitiveSpec:
    """Declarative conditional primitive used by symbolic grammar lowering."""

    name: str
    family: str
    source_features: tuple[str, ...] = tuple()
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "family": str(self.family),
            "source_features": [str(value) for value in self.source_features],
            "parameters": dict(self.parameters),
        }


class ConditionalPrimitive:
    name = "conditional_primitive"

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


@dataclass(frozen=True)
class BinaryGate(ConditionalPrimitive):
    feature_index: int = 0
    threshold: float = 0.0
    name = "binary_gate"

    def transform(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        return (arr[:, int(self.feature_index)] >= float(self.threshold)).astype(float).reshape(-1, 1)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "feature_index": int(self.feature_index), "threshold": float(self.threshold)}


@dataclass(frozen=True)
class SoftGate(ConditionalPrimitive):
    feature_index: int = 0
    threshold: float = 0.0
    temperature: float = 1.0
    name = "soft_gate"

    def transform(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        z = (arr[:, int(self.feature_index)] - float(self.threshold)) / max(float(self.temperature), 1e-12)
        return (1.0 / (1.0 + np.exp(-z))).reshape(-1, 1)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "feature_index": int(self.feature_index), "threshold": float(self.threshold), "temperature": float(self.temperature)}


@dataclass(frozen=True)
class HingeFeature(ConditionalPrimitive):
    feature_index: int = 0
    knot: float = 0.0
    side: str = "right"
    name = "hinge"

    def transform(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        values = arr[:, int(self.feature_index)] - float(self.knot)
        if self.side == "left":
            values = -values
        return np.maximum(values, 0.0).reshape(-1, 1)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "feature_index": int(self.feature_index), "knot": float(self.knot), "side": self.side}


@dataclass(frozen=True)
class OneHotGate(ConditionalPrimitive):
    feature_index: int = 0
    values: Sequence[float] = field(default_factory=tuple)
    name = "onehot_gate"

    def transform(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        source = arr[:, int(self.feature_index)]
        values = tuple(float(v) for v in self.values)
        return np.column_stack([(source == value).astype(float) for value in values]) if values else np.zeros((arr.shape[0], 0))

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "feature_index": int(self.feature_index), "values": [float(v) for v in self.values]}


def primitive_from_spec(spec: Mapping[str, Any] | str | ConditionalPrimitive) -> ConditionalPrimitive:
    if isinstance(spec, ConditionalPrimitive):
        return spec
    if isinstance(spec, str):
        payload = {"name": spec}
    else:
        payload = dict(spec)
    name = str(payload.get("name", payload.get("kind", ""))).lower()
    params = dict(payload.get("params", payload) or {})
    params.pop("name", None)
    params.pop("kind", None)
    if name in {"binary_gate", "binary", "gate"}:
        return BinaryGate(**params)
    if name in {"soft_gate", "soft"}:
        return SoftGate(**params)
    if name in {"hinge", "hinge_feature"}:
        return HingeFeature(**params)
    if name in {"onehot_gate", "one_hot_gate", "onehot"}:
        return OneHotGate(**params)
    raise ValueError(f"unknown conditional primitive: {name}")


__all__ = [
    "BinaryGate",
    "ConditionalPrimitive",
    "ConditionalPrimitiveSpec",
    "HingeFeature",
    "OneHotGate",
    "SoftGate",
    "primitive_from_spec",
]
