from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .primitives import ConditionalPrimitive, primitive_from_spec


@dataclass(frozen=True)
class PrimitiveFeatureComposer:
    primitives: Sequence[ConditionalPrimitive | Mapping[str, Any] | str]
    include_original: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def transform(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        parts = [arr] if self.include_original else []
        for primitive in self.primitive_objects():
            part = np.asarray(primitive.transform(arr), dtype=float)
            if part.ndim == 1:
                part = part.reshape(-1, 1)
            parts.append(part)
        return np.column_stack(parts) if parts else np.zeros((arr.shape[0], 0))

    def primitive_objects(self) -> tuple[ConditionalPrimitive, ...]:
        return tuple(primitive_from_spec(item) for item in self.primitives)

    def describe(self) -> dict[str, Any]:
        return {
            "name": "primitive_feature_composer",
            "include_original": bool(self.include_original),
            "primitives": [item.describe() for item in self.primitive_objects()],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RouteThenFormulaComposer:
    """Composable condition descriptor for representation builders.

    The class does not train. It produces deterministic conditional features or
    route ids that a representation/problem can consume.
    """

    router: Any
    branch_formulas: Sequence[Any] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def route(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.router, "route"):
            return np.asarray(self.router.route(X), dtype=int).reshape(-1)
        raise TypeError("router must expose route(X)")

    def describe(self) -> dict[str, Any]:
        return {
            "name": "route_then_formula",
            "router": self.router.describe() if hasattr(self.router, "describe") else repr(self.router),
            "num_branch_formulas": len(tuple(self.branch_formulas)),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SharedBackboneResidualComposer:
    backbone: Any
    residual_branches: Sequence[Any]
    router: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.backbone, "predict"):
            raise TypeError("backbone must expose predict(X)")
        base = np.asarray(self.backbone.predict(X), dtype=float).reshape(-1)
        if not self.residual_branches:
            return base
        if self.router is None:
            residual = sum(np.asarray(branch.predict(X), dtype=float).reshape(-1) for branch in self.residual_branches)
            return base + residual
        routes = np.asarray(self.router.route(X), dtype=int).reshape(-1)
        out = base.copy()
        for idx, branch in enumerate(self.residual_branches):
            mask = routes == idx
            if np.any(mask):
                out[mask] += np.asarray(branch.predict(np.asarray(X)[mask]), dtype=float).reshape(-1)
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "name": "shared_backbone_residual",
            "num_residual_branches": len(tuple(self.residual_branches)),
            "has_router": self.router is not None,
            "metadata": dict(self.metadata),
        }
