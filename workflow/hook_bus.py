from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from core.orchestration.capabilities import FlowCapability
from core.orchestration.lifecycle_runtime import LifecycleRuntime
from core.orchestration.lifecycle_dispatcher import LifecycleDispatcher
from training import describe_inner_runtime_event_table


class RuntimeHook(Protocol):
    def on_experiment_start(self, context: Mapping[str, Any]) -> None: ...

    def on_stage_start(self, stage: str, context: Mapping[str, Any]) -> None: ...

    def on_stage_end(self, stage: str, payload: Mapping[str, Any], context: Mapping[str, Any]) -> None: ...

    def on_stage_error(self, stage: str, error: Exception, context: Mapping[str, Any]) -> None: ...

    def on_experiment_finish(self, result: Any, context: Mapping[str, Any]) -> None: ...

    def on_experiment_error(self, error: Exception, context: Mapping[str, Any]) -> None: ...


HookLike = RuntimeHook | FlowCapability


@dataclass
class HookBus:
    strict: bool = False
    runtime: LifecycleRuntime | None = None
    dispatcher: LifecycleDispatcher | None = None

    def __post_init__(self) -> None:
        if self.runtime is None:
            self.runtime = LifecycleRuntime(strict=bool(self.strict), dispatcher=self.dispatcher)
        self.dispatcher = self.runtime.dispatcher

    def register(self, hook: HookLike) -> None:
        assert self.runtime is not None
        self.runtime.register(hook)

    def on_experiment_start(self, context: Mapping[str, Any]) -> None:
        self._emit("on_experiment_start", context)

    def on_stage_start(self, stage: str, context: Mapping[str, Any]) -> None:
        self._emit("on_stage_start", stage, context)

    def on_stage_end(self, stage: str, payload: Mapping[str, Any], context: Mapping[str, Any]) -> None:
        self._emit("on_stage_end", stage, payload, context)

    def on_stage_error(self, stage: str, error: Exception, context: Mapping[str, Any]) -> None:
        self._emit("on_stage_error", stage, error, context)

    def on_experiment_finish(self, result: Any, context: Mapping[str, Any]) -> None:
        self._emit("on_experiment_finish", result, context)

    def on_experiment_error(self, error: Exception, context: Mapping[str, Any]) -> None:
        self._emit("on_experiment_error", error, context)

    def list_hooks(self) -> tuple[HookLike, ...]:
        assert self.runtime is not None
        return tuple(self.runtime.list_participants())

    def describe_event_table(self) -> tuple[dict[str, Any], ...]:
        assert self.runtime is not None
        return self.runtime.describe_event_table()

    def describe_inner_event_table(self) -> tuple[dict[str, Any], ...]:
        return describe_inner_runtime_event_table()

    def describe_control_plane_contract(self) -> dict[str, Any]:
        assert self.runtime is not None
        return self.runtime.describe_control_plane_contract()

    def _emit(self, method: str, *args: Any) -> None:
        assert self.runtime is not None
        self.runtime.emit(method, *args)


__all__ = ["RuntimeHook", "HookBus"]
