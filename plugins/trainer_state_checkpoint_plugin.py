from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from core.orchestration.capabilities import FlowCapability


@dataclass
class TrainerStateCheckpointPlugin(FlowCapability):
    """Persist trainer_state through the flow capability plane instead of trainer-side ad hoc I/O."""

    name: str = "trainer_state_checkpoint"
    priority: int = 250
    enabled: bool = True
    is_algorithmic: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    relpath: str = "trainer_state/latest.pt"
    output_dir: str | None = None
    strict: bool = False
    context_requires: Sequence[str] = ("trainer", "trainer_state")
    context_provides: Sequence[str] = ("trainer_state_checkpoint_path",)
    context_mutates: Sequence[str] = ("report",)
    context_cache: Sequence[str] = tuple()
    context_notes: str | None = "Persists trainer_state using trainer.save_trainer_state(path, state)."

    def on_pre_persist(self, context: MutableMapping[str, Any]) -> None:
        if not self.enabled:
            return
        trainer = context.get("trainer")
        trainer_state = context.get("trainer_state")
        if trainer is None or trainer_state is None:
            return
        save_fn = getattr(trainer, "save_trainer_state", None)
        if not callable(save_fn):
            self._maybe_raise(
                TypeError(
                    f"trainer '{getattr(trainer, 'name', type(trainer).__name__)}' does not expose save_trainer_state(path, state)"
                )
            )
            return

        base_dir = self._resolve_base_dir(context)
        if base_dir is None:
            return
        out_path = Path(base_dir).resolve() / str(self.relpath)
        saved = save_fn(out_path, trainer_state)
        context["trainer_state_checkpoint_path"] = str(saved)

        report = context.get("report")
        if isinstance(report, dict):
            training = dict(report.get("training", {}))
            training["trainer_state_checkpoint"] = str(saved)
            report["training"] = training

    def _resolve_base_dir(self, context: Mapping[str, Any]) -> str | None:
        if self.output_dir:
            return str(self.output_dir)
        flow_spec = context.get("flow_spec")
        output_dir = getattr(flow_spec, "output_dir", None)
        if output_dir:
            return str(output_dir)
        return None

    def _maybe_raise(self, error: Exception) -> None:
        if self.strict:
            raise error


__all__ = ["TrainerStateCheckpointPlugin"]
