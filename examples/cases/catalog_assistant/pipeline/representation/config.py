# -*- coding: utf-8 -*-
"""Representation-layer configuration: ModelRepresentation registry + builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class RepresentationSpec:
    key: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RepresentationRegistry:
    registry: tuple[RepresentationSpec, ...] = ()


def get_representation_registry() -> RepresentationRegistry:
    return RepresentationRegistry(
        registry=(
            RepresentationSpec(key="example_linear", params={"n_features": 1}),
        )
    )


RepresentationBuilder = Callable[[Dict[str, Any]], object]
_REPRESENTATION_BUILDERS: Dict[str, RepresentationBuilder] = {}


def register_representation_builder(key: str, builder: RepresentationBuilder) -> None:
    _REPRESENTATION_BUILDERS[str(key).strip().lower()] = builder


def _find_spec(registry: RepresentationRegistry, key: str) -> RepresentationSpec:
    for spec in tuple(registry.registry or ()):
        if spec.key == key:
            return spec
    raise ValueError(f"Representation key not registered: {key}")


def build_representation(registry: RepresentationRegistry, key: str) -> object:
    spec = _find_spec(registry, key)
    builder = _REPRESENTATION_BUILDERS.get(str(spec.key).strip().lower())
    if builder is None:
        raise ValueError(f"Unknown representation key: {spec.key}")
    return builder(dict(spec.params or {}))


def _register_builtin_representations() -> None:
    from pipeline.representation.example_representation import ExampleLinearRepresentation
    def _linear_builder(params: Dict[str, Any]) -> object:
        n_features = int(params.get("n_features", 1))
        return ExampleLinearRepresentation(n_features=n_features)
    register_representation_builder("example_linear", _linear_builder)

_register_builtin_representations()
