"""
Forwarding module for state.

This module re-exports from blackbase for seamless migration.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from blackbase.context import (
    SnapshotHandle,
    SnapshotRecord,
    build_minimal_context,
    validate_minimal_context,
)


@dataclass(frozen=True)
class TrainerState:
    payload: Mapping[str, Any] = field(default_factory=dict)
    signature: str = ""
    version: str = "mlblack.trainer_state.v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        payload = dict(self.payload)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.signature:
            object.__setattr__(self, "signature", stable_state_signature(payload))

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "signature": self.signature,
            "payload": dict(self.payload),
            "metadata": dict(self.metadata),
        }


def build_trainer_state(trainer: Any, *, metadata: Mapping[str, Any] | None = None) -> TrainerState:
    if hasattr(trainer, "get_state"):
        raw_state = trainer.get_state()
        if isinstance(raw_state, TrainerState):
            payload = dict(raw_state.payload)
            signature = str(raw_state.signature or "")
            state_metadata = dict(raw_state.metadata)
        elif isinstance(raw_state, Mapping):
            is_envelope = isinstance(raw_state.get("payload"), Mapping) and set(raw_state).issubset(
                {"payload", "signature", "version", "metadata"}
            )
            payload = dict(raw_state.get("payload", {})) if is_envelope else dict(raw_state)
            signature = str(raw_state.get("signature", "")) if is_envelope else ""
            state_metadata = dict(raw_state.get("metadata", {}) or {}) if is_envelope else {}
        else:
            raise TypeError("trainer.get_state() must return a mapping or TrainerState")
        if metadata:
            state_metadata.update(dict(metadata))
        return TrainerState(
            payload=payload,
            signature=signature,
            metadata=state_metadata,
        )
    if metadata:
        return TrainerState(metadata=dict(metadata))
    return TrainerState()


def stable_state_signature(state: TrainerState | Mapping[str, Any]) -> str:
    raw_payload = state.payload if isinstance(state, TrainerState) else state
    payload = _canonical_state_value(raw_payload)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def restore_trainer_state(
    trainer: Any,
    state: TrainerState | Mapping[str, Any],
) -> Any:
    setter = getattr(trainer, "set_state", None)
    if not callable(setter):
        raise TypeError("trainer must expose set_state(...)")
    if isinstance(state, TrainerState):
        payload = state.payload
    elif isinstance(state.get("payload"), Mapping):
        payload = state.get("payload", {})
    else:
        payload = state
    setter(dict(payload))
    return trainer


def _canonical_state_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, np.generic):
        return _canonical_state_value(value.item())
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": _canonical_state_value(value.tolist()),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_state_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_state_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_state_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False),
        )
    for method_name in ("to_protocol_payload", "as_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            return {
                "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
                "payload": _canonical_state_value(method()),
            }
    return {
        "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def replay_trainer(
    trainer: Any,
    state: TrainerState | Mapping[str, Any],
    *,
    max_steps: int = 0,
) -> Any:
    restore_trainer_state(trainer, state)
    if int(max_steps) <= 0:
        return trainer
    fit = getattr(trainer, "fit", None)
    if not callable(fit):
        raise TypeError("trainer must expose fit(max_steps=...) to continue replay")
    return fit(max_steps=int(max_steps))


__all__ = [
    "SnapshotHandle",
    "SnapshotRecord",
    "build_minimal_context",
    "validate_minimal_context",
    "TrainerState",
    "build_trainer_state",
    "replay_trainer",
    "restore_trainer_state",
    "stable_state_signature",
]
