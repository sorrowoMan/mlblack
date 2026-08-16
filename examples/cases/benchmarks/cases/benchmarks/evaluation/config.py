# -*- coding: utf-8 -*-
"""L4 evaluation runtime configuration — provider registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class EvaluationSpec:
    key: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRegistry:
    registry: tuple[EvaluationSpec, ...] = ()


def get_evaluation_registry() -> EvaluationRegistry:
    return EvaluationRegistry(registry=())


ProviderBuilder = Callable[[Dict[str, Any]], object]
_EVAL_PROVIDER_BUILDERS: Dict[str, ProviderBuilder] = {}


def register_evaluation_provider_builder(key: str, builder: ProviderBuilder) -> None:
    _EVAL_PROVIDER_BUILDERS[str(key).strip().lower()] = builder


def build_evaluation_providers(registry: EvaluationRegistry, keys) -> list:
    providers = []
    for spec in registry.registry:
        if spec.key in keys:
            builder = _EVAL_PROVIDER_BUILDERS.get(spec.key)
            if builder:
                providers.append(builder(dict(spec.params or {})))
    return providers
