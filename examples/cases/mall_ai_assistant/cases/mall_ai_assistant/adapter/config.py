# -*- coding: utf-8 -*-
"""Stable optimization-method configuration for this ML Case."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from mlblack.integrations import build_optimization_adapter


@dataclass(frozen=True)
class AdapterSpec:
    key: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterRegistry:
    registry: tuple[AdapterSpec, ...] = ()


def get_adapter_registry() -> AdapterRegistry:
    return AdapterRegistry(
        registry=(
            AdapterSpec(key="gradient.sgd", params={"learning_rate": 0.01}),
        )
    )


def _find_spec(registry: AdapterRegistry, key: str) -> AdapterSpec:
    for spec in tuple(registry.registry or ()):
        if spec.key == key:
            return spec
    raise ValueError(f"Adapter key not registered: {key}")


def build_adapter(registry: AdapterRegistry, key: str) -> object:
    spec = _find_spec(registry, key)
    return build_optimization_adapter(spec.key, **dict(spec.params or {}))
