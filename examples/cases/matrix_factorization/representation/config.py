# -*- coding: utf-8 -*-
"""Representation config: MFRepresentation registry."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class ReprSpec:
    key: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReprRegistry:
    registry: tuple[ReprSpec, ...] = ()


def get_repr_registry() -> ReprRegistry:
    return ReprRegistry(
        registry=(ReprSpec(key="mf_representation", params={}),)
    )


ReprBuilder = Callable[[Dict[str, Any]], object]
_REPR_BUILDERS: Dict[str, ReprBuilder] = {}


def register_repr_builder(key: str, builder: ReprBuilder) -> None:
    _REPR_BUILDERS[str(key).strip().lower()] = builder


def build_representation(registry: ReprRegistry, key: str) -> object:
    for spec in registry.registry:
        if spec.key == key:
            builder = _REPR_BUILDERS.get(key)
            if builder:
                return builder(dict(spec.params or {}))
    raise ValueError(f"Repr key not found: {key}")


def _register_builtin() -> None:
    from representation.mf_representation import MFRepresentation
    register_repr_builder("mf_representation", lambda p: MFRepresentation())

_register_builtin()
