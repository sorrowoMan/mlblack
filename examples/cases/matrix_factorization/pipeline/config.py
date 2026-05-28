# -*- coding: utf-8 -*-
"""Pipeline config: MF data pipeline registry."""

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
        registry=(PipelineSpec(key="default", components=()),)
    )


PipelineBuilder = Callable[[PipelineSpec], object]
_PIPELINE_BUILDERS: Dict[str, PipelineBuilder] = {}


def register_pipeline_builder(key: str, builder: PipelineBuilder) -> None:
    _PIPELINE_BUILDERS[str(key).strip().lower()] = builder


def build_pipeline(registry: PipelineRegistry, key: str) -> object:
    for spec in registry.registry:
        if spec.key == key:
            builder = _PIPELINE_BUILDERS.get(key)
            if builder:
                return builder(spec)
    raise ValueError(f"Pipeline key not found: {key}")


def _register_builtin() -> None:
    from pipeline.mf_pipeline import build_rating_data_view
    register_pipeline_builder("default", lambda s: build_rating_data_view(None, None))

_register_builtin()
