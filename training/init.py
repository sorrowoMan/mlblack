from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .capabilities import TrainingMode

if TYPE_CHECKING:
    from .inner_runtime import InnerRuntimeHook


@dataclass(frozen=True)
class TrainingInit:
    mode: TrainingMode = "fresh"
    parent_artifact: Any | None = None
    parent_state: Any | None = None
    inner_runtime_hooks: Sequence["InnerRuntimeHook"] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", str(self.mode).strip().lower() or "fresh")
        object.__setattr__(self, "inner_runtime_hooks", tuple(self.inner_runtime_hooks))
        object.__setattr__(self, "metadata", dict(self.metadata))


__all__ = ["TrainingInit"]
