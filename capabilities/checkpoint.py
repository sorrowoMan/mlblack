from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mlblack.core.capability import Capability
from mlblack.core.contracts import ComponentContract


@dataclass(frozen=True)
class CheckpointConfig:
    every_steps: int = 1
    context_key: str = "checkpoint.last_ref"
    snapshot_kind: str = "trainer_state"


class CheckpointCapability(Capability):
    """Generic trainer-state checkpoint capability."""

    name = "checkpoint"
    context_requires = ('trainer.get_state', 'trainer.snapshot_store')
    context_optional = ()
    context_provides = ('checkpoint.ref',)
    context_mutates = ('trainer.context',)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: trainer.get_state, trainer.snapshot_store; provides checkpoint.ref; mutates trainer.context.'
    contract = ComponentContract(
        name=name,
        requires=("trainer.get_state", "trainer.snapshot_store"),
        provides=("checkpoint.ref",),
        mutates=("trainer.context",),
        supports_resume=True,
        metadata={"capability": "checkpoint"},
    )

    def __init__(self, config: CheckpointConfig | None = None) -> None:
        self.config = config or CheckpointConfig()
        self.written_refs: list[str] = []

    def on_step_end(self, trainer: Any, context: Mapping[str, Any], row: Mapping[str, Any]) -> None:
        _ = context
        step = int(row.get("step", getattr(trainer, "step_index", 0)))
        every = max(1, int(self.config.every_steps))
        if (step + 1) % every != 0:
            return
        if not hasattr(trainer, "get_state") or not hasattr(trainer, "write_snapshot"):
            return
        ref = trainer.write_snapshot(
            dict(trainer.get_state()),
            key=None,
            context_key=self.config.context_key,
        )
        self.written_refs.append(str(ref))

    def latest_ref(self) -> str | None:
        return None if not self.written_refs else self.written_refs[-1]

