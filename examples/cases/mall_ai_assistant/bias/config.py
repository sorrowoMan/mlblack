# -*- coding: utf-8 -*-
"""Bias-layer configuration: OptimizationBias registry + builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class BiasSpec:
    key: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BiasRegistry:
    registry: tuple[BiasSpec, ...] = ()


def get_bias_registry() -> BiasRegistry:
    return BiasRegistry(
        registry=(
            BiasSpec(key="none", params={}),
            BiasSpec(key="example_l2", params={"weight": 0.01}),
        )
    )


BiasBuilder = Callable[[Dict[str, Any]], object]
_BIAS_BUILDERS: Dict[str, BiasBuilder] = {}


def register_bias_builder(key: str, builder: BiasBuilder) -> None:
    _BIAS_BUILDERS[str(key).strip().lower()] = builder


def _find_spec(registry: BiasRegistry, key: str) -> BiasSpec:
    for spec in tuple(registry.registry or ()):
        if spec.key == key:
            return spec
    raise ValueError(f"Bias key not registered: {key}")


def build_bias(registry: BiasRegistry, key: str) -> object:
    spec = _find_spec(registry, key)
    if spec.key == "none":
        return None
    builder = _BIAS_BUILDERS.get(spec.key)
    if builder is None:
        raise ValueError(f"Unknown bias key: {spec.key}")
    return builder(dict(spec.params or {}))


def _register_builtin_biases() -> None:
    from bias.example_bias import ExampleL2Bias
    def _l2_builder(params: Dict[str, Any]) -> object:
        w = float(params.get("weight", 0.01))
        return ExampleL2Bias(weight=w)
    register_bias_builder("example_l2", _l2_builder)

_register_builtin_biases()
