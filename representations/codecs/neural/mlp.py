from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class NeuralBackboneSpec:
    hidden_layers: tuple[int, ...] = (64, 32)
    activation: str = "relu"
    dropout: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "hidden_layers": tuple(int(v) for v in self.hidden_layers),
            "activation": str(self.activation),
            "dropout": float(self.dropout),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NeuralOptimizationSpec:
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_steps: int = 120
    early_stopping: bool = True
    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-6
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "optimizer": str(self.optimizer),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "max_steps": int(self.max_steps),
            "early_stopping": bool(self.early_stopping),
            "early_stop_patience": int(self.early_stop_patience),
            "early_stop_min_delta": float(self.early_stop_min_delta),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NeuralBatchingSpec:
    batch_size: int | None = 64
    shuffle: bool = True
    drop_last: bool = False
    num_workers: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "shuffle": bool(self.shuffle),
            "drop_last": bool(self.drop_last),
            "num_workers": int(self.num_workers),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NeuralEngineSpec:
    engine: str = "torch"
    device: str = "cpu"
    checkpoint_enabled: bool = True
    resume_enabled: bool = True
    trainer_state_enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": str(self.engine),
            "device": str(self.device),
            "checkpoint_enabled": bool(self.checkpoint_enabled),
            "resume_enabled": bool(self.resume_enabled),
            "trainer_state_enabled": bool(self.trainer_state_enabled),
            "metadata": dict(self.metadata),
        }

