# -*- coding: utf-8 -*-
"""L0 runtime configuration — PoolScheduler, resource profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from blackbase.resources import PoolScheduler


@dataclass(frozen=True)
class RuntimeProfile:
    key: str = "local_cpu"
    threads: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeRegistry:
    profiles: tuple[RuntimeProfile, ...] = ()


def get_runtime_registry() -> RuntimeRegistry:
    return RuntimeRegistry(
        profiles=(
            RuntimeProfile(key="local_cpu", threads=1),
            RuntimeProfile(key="threaded_cpu", threads=4),
        )
    )


def build_pool(profile: RuntimeProfile) -> PoolScheduler:
    return PoolScheduler(total_threads=max(1, int(profile.threads)))
