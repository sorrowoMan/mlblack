from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingFallbackPolicy:
    allow_fresh_fallback: bool = False
    strict: bool = True


__all__ = ["TrainingFallbackPolicy"]
