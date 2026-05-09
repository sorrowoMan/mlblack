from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnownRelationSymbolicBuildConfig:
    n_total: int = 2400
    train_ratio: float = 0.8
    noise_std: float = 0.025
    seed: int = 42
    outer_solver_backend: str = "native_placeholder"


__all__ = ["KnownRelationSymbolicBuildConfig"]
