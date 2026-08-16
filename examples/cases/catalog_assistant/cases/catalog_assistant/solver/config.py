# -*- coding: utf-8 -*-
"""Trainer/Solver core configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrainerCoreConfig:
    max_steps: int = 20
    learning_rate: float = 0.01
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainerProfileSpec:
    key: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainerProfileRegistry:
    registry: tuple[TrainerProfileSpec, ...] = ()


def get_trainer_profile_registry() -> TrainerProfileRegistry:
    return TrainerProfileRegistry(
        registry=(
            TrainerProfileSpec(key="default", params={"max_steps": 20, "learning_rate": 0.01}),
            TrainerProfileSpec(key="quick", params={"max_steps": 5, "learning_rate": 0.05}),
        )
    )
