from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from core.orchestration.capabilities import FlowCapability
from workflow.reproducibility import ReproducibilityConfig, apply_reproducibility

from nowcasting_work_ci.mlblack_side.runtime.contracts import RuntimeContextKey, ctx_set


@dataclass
class ReproducibilityPlugin(FlowCapability):
    name: str = "reproducibility"
    priority: int = 10
    enabled: bool = True
    is_algorithmic: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    context_requires: Sequence[str] = tuple()
    context_provides: Sequence[str] = (
        RuntimeContextKey.REPRODUCIBILITY.value,
        RuntimeContextKey.RUNTIME_SEED.value,
    )
    context_mutates: Sequence[str] = (
        RuntimeContextKey.REPRODUCIBILITY.value,
        RuntimeContextKey.RUNTIME_SEED.value,
    )
    context_cache: Sequence[str] = tuple()
    context_notes: str | None = "Applies deterministic seeding and records reproducibility metadata."
    seed: int = 0
    deterministic_torch: bool = True
    torch_warn_only: bool = True

    def on_experiment_start(self, context: Mapping[str, Any]) -> None:
        info = apply_reproducibility(
            ReproducibilityConfig(
                seed=int(self.seed),
                deterministic_torch=bool(self.deterministic_torch),
                torch_warn_only=bool(self.torch_warn_only),
            )
        )
        if isinstance(context, dict):
            ctx_set(context, RuntimeContextKey.REPRODUCIBILITY, info)
            ctx_set(context, RuntimeContextKey.RUNTIME_SEED, int(self.seed))

    def on_stage_start(self, stage: str, context: Mapping[str, Any]) -> None:
        return

    def on_stage_end(self, stage: str, payload: Mapping[str, Any], context: Mapping[str, Any]) -> None:
        return

    def on_stage_error(self, stage: str, error: Exception, context: Mapping[str, Any]) -> None:
        return

    def on_experiment_finish(self, result: Any, context: Mapping[str, Any]) -> None:
        return

    def on_experiment_error(self, error: Exception, context: Mapping[str, Any]) -> None:
        return


__all__ = ["ReproducibilityPlugin"]
