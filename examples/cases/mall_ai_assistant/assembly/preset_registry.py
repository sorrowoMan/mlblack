# -*- coding: utf-8 -*-
"""Preset registry: available trainer presets (e.g., orthogonal_linear_point)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class PresetSpec:
    key: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PresetRegistry:
    registry: tuple[PresetSpec, ...] = ()


def get_preset_registry() -> PresetRegistry:
    return PresetRegistry(
        registry=(
            PresetSpec(key="example_preset", params={}),
        )
    )
