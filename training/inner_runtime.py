from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .inner_runtime_events import describe_inner_runtime_event_table, resolve_inner_runtime_event


def _freeze_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {} if value is None else dict(value)


@dataclass(frozen=True)
class InnerRuntimeStartPayload:
    run_id: str
    runtime_key: str
    trainer_name: str
    total_rounds: int
    input_shape: tuple[int, int]
    feature_names: tuple[str, ...] = ()
    seed_terms: int = 0
    context: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", str(self.run_id))
        object.__setattr__(self, "runtime_key", str(self.runtime_key))
        object.__setattr__(self, "trainer_name", str(self.trainer_name))
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "input_shape", tuple(int(v) for v in self.input_shape))
        object.__setattr__(self, "context", _freeze_mapping(self.context))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class InnerRuntimeRoundPayload:
    run_id: str
    runtime_key: str
    trainer_name: str
    round_index: int
    total_rounds: int
    genome_size: int
    score_trace: tuple[float, ...] = ()
    history_entry: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", str(self.run_id))
        object.__setattr__(self, "runtime_key", str(self.runtime_key))
        object.__setattr__(self, "trainer_name", str(self.trainer_name))
        object.__setattr__(self, "score_trace", tuple(float(v) for v in self.score_trace))
        object.__setattr__(self, "history_entry", _freeze_mapping(self.history_entry))
        object.__setattr__(self, "context", _freeze_mapping(self.context))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class InnerRuntimeFinishPayload:
    run_id: str
    runtime_key: str
    trainer_name: str
    total_rounds: int
    completed_rounds: int
    genome_size: int
    final_metrics: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", str(self.run_id))
        object.__setattr__(self, "runtime_key", str(self.runtime_key))
        object.__setattr__(self, "trainer_name", str(self.trainer_name))
        object.__setattr__(self, "final_metrics", _freeze_mapping(self.final_metrics))
        object.__setattr__(self, "context", _freeze_mapping(self.context))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class InnerRuntimeErrorPayload:
    run_id: str
    runtime_key: str
    trainer_name: str
    error: str
    round_index: int | None = None
    context: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", str(self.run_id))
        object.__setattr__(self, "runtime_key", str(self.runtime_key))
        object.__setattr__(self, "trainer_name", str(self.trainer_name))
        object.__setattr__(self, "error", str(self.error))
        object.__setattr__(self, "context", _freeze_mapping(self.context))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


class InnerRuntimeHook(Protocol):
    def on_inner_run_start(self, payload: InnerRuntimeStartPayload) -> None:
        ...

    def on_inner_round_end(self, payload: InnerRuntimeRoundPayload) -> None:
        ...

    def on_inner_run_finish(self, payload: InnerRuntimeFinishPayload) -> None:
        ...

    def on_inner_run_error(self, payload: InnerRuntimeErrorPayload) -> None:
        ...


class InnerRuntimeDispatcher:
    def __init__(
        self,
        hooks: Sequence[InnerRuntimeHook] = (),
        *,
        strict: bool = False,
    ) -> None:
        self.hooks = tuple(hooks)
        self.strict = bool(strict)

    @property
    def enabled(self) -> bool:
        return bool(self.hooks)

    @classmethod
    def from_hooks(
        cls,
        hooks: Sequence[InnerRuntimeHook] = (),
        *,
        strict: bool = False,
    ) -> "InnerRuntimeDispatcher":
        return cls(hooks=hooks, strict=strict)

    def emit_start(self, payload: InnerRuntimeStartPayload) -> None:
        self._emit("on_inner_run_start", payload)

    def emit_round_end(self, payload: InnerRuntimeRoundPayload) -> None:
        self._emit("on_inner_round_end", payload)

    def emit_finish(self, payload: InnerRuntimeFinishPayload) -> None:
        self._emit("on_inner_run_finish", payload)

    def emit_error(self, payload: InnerRuntimeErrorPayload) -> None:
        self._emit("on_inner_run_error", payload)

    def describe_event_table(self) -> tuple[dict[str, Any], ...]:
        return describe_inner_runtime_event_table()

    def resolve_event_spec(self, runtime_key: str) -> dict[str, Any] | None:
        spec = resolve_inner_runtime_event(runtime_key)
        return None if spec is None else spec.as_dict()

    def _emit(self, method_name: str, payload: Any) -> None:
        for hook in self.hooks:
            method = getattr(hook, method_name, None)
            if not callable(method):
                continue
            try:
                method(payload)
            except Exception as exc:
                message = (
                    f"Inner runtime hook '{type(hook).__name__}.{method_name}' failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                if self.strict:
                    raise RuntimeError(message) from exc
                warnings.warn(message, RuntimeWarning, stacklevel=2)


__all__ = [
    "InnerRuntimeDispatcher",
    "InnerRuntimeErrorPayload",
    "describe_inner_runtime_event_table",
    "InnerRuntimeFinishPayload",
    "InnerRuntimeHook",
    "InnerRuntimeRoundPayload",
    "InnerRuntimeStartPayload",
    "resolve_inner_runtime_event",
]
