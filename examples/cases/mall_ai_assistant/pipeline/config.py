# -*- coding: utf-8 -*-
"""Pipeline-layer configuration: DataPipeline registry + builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Sequence


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineSpec:
    key: str
    components: Sequence[ComponentSpec] = ()
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineRegistry:
    registry: tuple[PipelineSpec, ...] = ()


def get_pipeline_registry() -> PipelineRegistry:
    return PipelineRegistry(
        registry=(
            PipelineSpec(key="default", components=()),
        )
    )


PipelineBuilder = Callable[[PipelineSpec], object]
_PIPELINE_BUILDERS: Dict[str, PipelineBuilder] = {}


def register_pipeline_builder(key: str, builder: PipelineBuilder) -> None:
    _PIPELINE_BUILDERS[str(key).strip().lower()] = builder


def _find_spec(registry: PipelineRegistry, key: str) -> PipelineSpec:
    for spec in tuple(registry.registry or ()):
        if spec.key == key:
            return spec
    raise ValueError(f"Pipeline key not registered: {key}")


def build_pipeline(registry: PipelineRegistry, key: str) -> object:
    spec = _find_spec(registry, key)
    builder = _PIPELINE_BUILDERS.get(str(spec.key).strip().lower())
    if builder is None:
        raise ValueError(f"Unknown pipeline key: {spec.key}")
    return builder(spec)


def _register_builtin_pipelines() -> None:
    from pipeline.example_pipeline import build_data_view
    def _data_view_builder(spec: PipelineSpec) -> object:
        return build_data_view(None, None)
    register_pipeline_builder("default", _data_view_builder)

_register_builtin_pipelines()
