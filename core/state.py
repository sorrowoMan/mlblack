from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainerState:
    """Serializable trainer replay payload.

    This is intentionally separate from model artifacts. A model artifact is a
    product; trainer state is a replay/resume boundary.
    """

    payload: Mapping[str, Any]
    version: str = "mlblack.trainer_state.v1"
    signature: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        payload = dict(self.payload)
        signature = str(self.signature or stable_state_signature(payload))
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "signature", signature)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "signature": self.signature,
            "payload": dict(self.payload),
            "metadata": dict(self.metadata),
        }


def build_trainer_state(trainer: Any, *, metadata: Mapping[str, Any] | None = None) -> TrainerState:
    if not hasattr(trainer, "get_state"):
        raise TypeError("trainer must expose get_state()")
    payload = dict(trainer.get_state())
    return TrainerState(payload=payload, metadata=dict(metadata or {}))


def restore_trainer_state(trainer: Any, state: TrainerState | Mapping[str, Any]) -> Any:
    if not hasattr(trainer, "set_state"):
        raise TypeError("trainer must expose set_state(...)")
    payload = state.payload if isinstance(state, TrainerState) else dict(state).get("payload", state)
    trainer.set_state(dict(payload))
    return trainer


def replay_trainer(trainer: Any, state: TrainerState | Mapping[str, Any], *, max_steps: int = 0) -> Any:
    restore_trainer_state(trainer, state)
    if max_steps <= 0:
        return trainer
    return trainer.fit(max_steps=max_steps)


def stable_state_signature(payload: Mapping[str, Any]) -> str:
    safe = _json_safe(payload)
    text = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)

