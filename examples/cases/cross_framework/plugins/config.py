# -*- coding: utf-8 -*-
"""Plugin-layer configuration: Plugin registry + builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class PluginSpec:
    key: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PluginRegistry:
    registry: tuple[PluginSpec, ...] = ()


def get_plugin_registry() -> PluginRegistry:
    return PluginRegistry(
        registry=(
            PluginSpec(key="none", params={}),
            PluginSpec(key="example_checkpoint", params={"interval": 10}),
        )
    )


PluginBuilder = Callable[[Dict[str, Any]], object]
_PLUGIN_BUILDERS: Dict[str, PluginBuilder] = {}


def register_plugin_builder(key: str, builder: PluginBuilder) -> None:
    _PLUGIN_BUILDERS[str(key).strip().lower()] = builder


def _find_spec(registry: PluginRegistry, key: str) -> PluginSpec:
    for spec in tuple(registry.registry or ()):
        if spec.key == key:
            return spec
    raise ValueError(f"Plugin key not registered: {key}")


def build_plugin(registry: PluginRegistry, key: str) -> object:
    spec = _find_spec(registry, key)
    if spec.key == "none":
        return None
    builder = _PLUGIN_BUILDERS.get(spec.key)
    if builder is None:
        raise ValueError(f"Unknown plugin key: {spec.key}")
    return builder(dict(spec.params or {}))


def _register_builtin_plugins() -> None:
    from plugins.example_plugin import ExampleCheckpointPlugin
    def _ckpt_builder(params: Dict[str, Any]) -> object:
        interval = int(params.get("interval", 10))
        return ExampleCheckpointPlugin(interval=interval)
    register_plugin_builder("example_checkpoint", _ckpt_builder)

_register_builtin_plugins()
