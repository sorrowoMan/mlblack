from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StructureScoreConfig:
    score_corr_bonus: float = 0.04
    score_complexity_penalty: float = 7e-4
    score_grad_guidance_bonus: float = 0.0


class StructureOptimizer:
    """Combine residual gain and gradient guidance into one candidate score."""

    def __init__(self, config: StructureScoreConfig | None = None) -> None:
        self.config = config or StructureScoreConfig()

    def combine(
        self,
        *,
        projected_gain: float,
        abs_corr: float,
        complexity: float,
        grad_alignment: float,
    ) -> dict[str, Any]:
        gain = float(projected_gain)
        corr_bonus = float(self.config.score_corr_bonus) * float(abs_corr)
        complexity_penalty = float(self.config.score_complexity_penalty) * float(complexity)
        grad_bonus = float(self.config.score_grad_guidance_bonus) * float(grad_alignment)
        score = gain + corr_bonus - complexity_penalty + grad_bonus

        return {
            "score": float(score),
            "score_parts": {
                "projected_gain": float(gain),
                "corr_bonus": float(corr_bonus),
                "complexity_penalty": float(complexity_penalty),
                "grad_bonus": float(grad_bonus),
            },
        }


__all__ = [
    "StructureScoreConfig",
    "StructureOptimizer",
]
