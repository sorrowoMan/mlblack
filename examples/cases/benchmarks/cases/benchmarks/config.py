# -*- coding: utf-8 -*-
"""Project-level configuration aggregator: centralize all component registries.

Mirrors nsgablack's ProjectConfig pattern. This allows runtime selection of
components (problem, representation, adapter, bias, plugin, pipeline)
by key, similar to nsgablack's build_solver(problem_key=..., adapter_key=...).
"""

from __future__ import annotations

from dataclasses import dataclass

from mlblack.assembly import PresetRegistry, TrainerProjectConfig

__all__ = ["TrainerProjectConfig", "get_project_config"]


def get_project_config() -> TrainerProjectConfig:
    """Aggregate all component registries."""

    from problem.config import get_problem_registry
    from pipeline.config import get_pipeline_registry
    from pipeline.representation.config import get_representation_registry
    from adapter.config import get_adapter_registry
    from bias.config import get_bias_registry
    from plugins.config import get_plugin_registry

    return TrainerProjectConfig(
        problems=get_problem_registry(),
        pipelines=get_pipeline_registry(),
        representations=get_representation_registry(),
        adapters=get_adapter_registry(),
        biases=get_bias_registry(),
        plugins=get_plugin_registry(),
        presets=PresetRegistry(),
    )
