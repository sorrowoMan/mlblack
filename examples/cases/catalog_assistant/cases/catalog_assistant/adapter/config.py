# -*- coding: utf-8 -*-
"""Adapter-layer configuration: OptimizerAdapter registry + builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict


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
            AdapterSpec(key="example_adapter", params={"learning_rate": 0.01}),
        )
    )


AdapterBuilder = Callable[[Dict[str, Any]], object]
_ADAPTER_BUILDERS: Dict[str, AdapterBuilder] = {}


def register_adapter_builder(key: str, builder: AdapterBuilder) -> None:
    _ADAPTER_BUILDERS[str(key).strip().lower()] = builder


def _find_spec(registry: AdapterRegistry, key: str) -> AdapterSpec:
    for spec in tuple(registry.registry or ()):
        if spec.key == key:
            return spec
    raise ValueError(f"Adapter key not registered: {key}")


def build_adapter(registry: AdapterRegistry, key: str) -> object:
    spec = _find_spec(registry, key)
    builder = _ADAPTER_BUILDERS.get(str(spec.key).strip().lower())
    if builder is None:
        raise ValueError(f"Unknown adapter key: {spec.key}")
    return builder(dict(spec.params or {}))


def _register_builtin_adapters() -> None:
    from adapter.example_adapter import ExampleGradientDescentAdapter
    def _gd_builder(params: Dict[str, Any]) -> object:
        lr = float(params.get("learning_rate", 0.01))
        return ExampleGradientDescentAdapter(learning_rate=lr)
    register_adapter_builder("example_adapter", _gd_builder)

_register_builtin_adapters()
