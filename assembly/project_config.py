# -*- coding: utf-8 -*-
"""MLBlack project-level configuration aggregator (registries only).

This mirrors nsgablack's ProjectConfig pattern: centralize all component registries
(problem, representation, adapter, bias, capability, pipeline) so they can be
selected by key at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

__all__ = [
    "ProblemRegistry",
    "RepresentationRegistry",
    "AdapterRegistry",
    "BiasRegistry",
    "CapabilityRegistry",
    "PipelineRegistry",
    "PresetRegistry",
    "TrainerProjectConfig",
    "build_problem",
    "build_representation",
    "build_adapter",
    "build_bias",
    "build_capability",
    "build_pipeline",
    "register_problem_builder",
    "register_representation_builder",
    "register_adapter_builder",
    "register_bias_builder",
    "register_capability_builder",
    "register_pipeline_builder",
]


# Spec and Registry definitions (mirrors nsgablack pattern)

@dataclass(frozen=True)
class _ComponentSpec:
    """Base spec for all components (internal use only)."""
    key: str
    params: Dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            object.__setattr__(self, "params", {})


@dataclass(frozen=True)
class ProblemRegistry:
    registry: Tuple[_ComponentSpec, ...] = ()


@dataclass(frozen=True)
class RepresentationRegistry:
    registry: Tuple[_ComponentSpec, ...] = ()


@dataclass(frozen=True)
class AdapterRegistry:
    registry: Tuple[_ComponentSpec, ...] = ()


@dataclass(frozen=True)
class BiasRegistry:
    registry: Tuple[_ComponentSpec, ...] = ()


@dataclass(frozen=True)
class CapabilityRegistry:
    registry: Tuple[_ComponentSpec, ...] = ()


@dataclass(frozen=True)
class PipelineRegistry:
    registry: Tuple[_ComponentSpec, ...] = ()


@dataclass(frozen=True)
class PresetRegistry:
    registry: Tuple[_ComponentSpec, ...] = ()


@dataclass(frozen=True)
class TrainerProjectConfig:
    """Centralized container for all MLBlack component registries.
    
    Use this in your project's config.py to aggregate all component registries,
    mirroring nsgablack's ProjectConfig pattern.
    """

    problems: ProblemRegistry
    representations: RepresentationRegistry
    adapters: AdapterRegistry
    biases: BiasRegistry
    capabilities: CapabilityRegistry
    pipelines: PipelineRegistry
    presets: PresetRegistry


# Registry builders (selection by key)

_problem_builders: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
_representation_builders: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
_adapter_builders: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
_bias_builders: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
_capability_builders: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
_pipeline_builders: Dict[str, Callable[[Dict[str, Any]], Any]] = {}


def register_problem_builder(key: str, builder: Callable) -> None:
    _problem_builders[str(key).strip().lower()] = builder


def register_representation_builder(key: str, builder: Callable) -> None:
    _representation_builders[str(key).strip().lower()] = builder


def register_adapter_builder(key: str, builder: Callable) -> None:
    _adapter_builders[str(key).strip().lower()] = builder


def register_bias_builder(key: str, builder: Callable) -> None:
    _bias_builders[str(key).strip().lower()] = builder


def register_capability_builder(key: str, builder: Callable) -> None:
    _capability_builders[str(key).strip().lower()] = builder


def register_pipeline_builder(key: str, builder: Callable) -> None:
    _pipeline_builders[str(key).strip().lower()] = builder


def build_problem(registry: ProblemRegistry, key: str) -> Any:
    """Build a problem by key from registry."""
    lookup = str(key).strip().lower()
    for spec in (registry.registry or ()):
        if str(spec.key).strip().lower() == lookup:
            builder = _problem_builders.get(lookup)
            if builder is None:
                raise ValueError(f"No builder registered for problem key: {key}")
            return builder(dict(spec.params or {}))
    raise ValueError(f"Problem key not registered: {key}")


def build_representation(registry: RepresentationRegistry, key: str) -> Any:
    """Build a representation by key from registry."""
    lookup = str(key).strip().lower()
    for spec in (registry.registry or ()):
        if str(spec.key).strip().lower() == lookup:
            builder = _representation_builders.get(lookup)
            if builder is None:
                raise ValueError(f"No builder registered for representation key: {key}")
            return builder(dict(spec.params or {}))
    raise ValueError(f"Representation key not registered: {key}")


def build_adapter(registry: AdapterRegistry, key: str) -> Any:
    """Build an adapter by key from registry."""
    lookup = str(key).strip().lower()
    for spec in (registry.registry or ()):
        if str(spec.key).strip().lower() == lookup:
            builder = _adapter_builders.get(lookup)
            if builder is None:
                raise ValueError(f"No builder registered for adapter key: {key}")
            return builder(dict(spec.params or {}))
    raise ValueError(f"Adapter key not registered: {key}")


def build_bias(registry: BiasRegistry, key: str) -> Any:
    """Build a bias by key from registry."""
    lookup = str(key).strip().lower()
    for spec in (registry.registry or ()):
        if str(spec.key).strip().lower() == lookup:
            builder = _bias_builders.get(lookup)
            if builder is None:
                raise ValueError(f"No builder registered for bias key: {key}")
            return builder(dict(spec.params or {}))
    raise ValueError(f"Bias key not registered: {key}")


def build_capability(registry: CapabilityRegistry, key: str) -> Any:
    """Build a capability by key from registry."""
    lookup = str(key).strip().lower()
    for spec in (registry.registry or ()):
        if str(spec.key).strip().lower() == lookup:
            builder = _capability_builders.get(lookup)
            if builder is None:
                raise ValueError(f"No builder registered for capability key: {key}")
            return builder(dict(spec.params or {}))
    raise ValueError(f"Capability key not registered: {key}")


def build_pipeline(registry: PipelineRegistry, key: str) -> Any:
    """Build a pipeline by key from registry."""
    lookup = str(key).strip().lower()
    for spec in (registry.registry or ()):
        if str(spec.key).strip().lower() == lookup:
            builder = _pipeline_builders.get(lookup)
            if builder is None:
                raise ValueError(f"No builder registered for pipeline key: {key}")
            return builder(dict(spec.params or {}))
    raise ValueError(f"Pipeline key not registered: {key}")

