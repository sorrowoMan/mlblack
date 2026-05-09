from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .capabilities import CapabilityManager, FlowCapability
from .control_plane_contract import describe_control_plane_contract
from .lifecycle_dispatcher import LifecycleDispatcher


@dataclass
class LifecycleRuntime:
    """
    Mainline control-plane runtime for lifecycle dispatch.

    The runtime owns the dispatcher and exposes two explicit registration lanes:

    - `register_capability(...)` for plugin/capability participants
    - `register_hook(...)` for runtime-local hook participants

    This keeps the control plane object separate from plugin implementations,
    while preserving one unified lifecycle event stream underneath.
    """

    strict: bool = False
    dispatcher: LifecycleDispatcher | None = None
    capability_manager: CapabilityManager | None = None
    _hooks: list[object] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.dispatcher is None:
            self.dispatcher = LifecycleDispatcher(strict=bool(self.strict))
        if self.capability_manager is None:
            self.capability_manager = CapabilityManager(
                strict=bool(self.strict),
                dispatcher=self.dispatcher,
            )

    @classmethod
    def create(
        cls,
        *,
        strict: bool = False,
        dispatcher: LifecycleDispatcher | None = None,
        capabilities: Sequence[FlowCapability] = (),
        hooks: Sequence[object] = (),
    ) -> "LifecycleRuntime":
        runtime = cls(strict=bool(strict), dispatcher=dispatcher)
        runtime.register_capabilities(capabilities)
        runtime.register_hooks(hooks)
        return runtime

    def register(self, participant: object) -> None:
        if isinstance(participant, FlowCapability):
            self.register_capability(participant)
            return
        self.register_hook(participant)

    def register_capability(self, capability: FlowCapability) -> None:
        assert self.capability_manager is not None
        self.capability_manager.register(capability)

    def register_capabilities(self, capabilities: Sequence[FlowCapability]) -> None:
        for capability in tuple(capabilities):
            self.register_capability(capability)

    def register_hook(self, hook: object) -> None:
        assert self.dispatcher is not None
        self._hooks.append(hook)
        self.dispatcher.register(hook)

    def register_hooks(self, hooks: Sequence[object]) -> None:
        for hook in tuple(hooks):
            self.register_hook(hook)

    def list_capabilities(self) -> tuple[FlowCapability, ...]:
        assert self.capability_manager is not None
        return self.capability_manager.list_capabilities()

    def list_hooks(self) -> tuple[object, ...]:
        assert self.dispatcher is not None
        hook_ids = {id(hook) for hook in self._hooks}
        return tuple(
            participant
            for participant in self.dispatcher.list_participants()
            if id(participant) in hook_ids
        )

    def list_participants(self) -> tuple[object, ...]:
        assert self.dispatcher is not None
        return tuple(self.dispatcher.list_participants())

    def emit(self, event: str, *args: Any) -> None:
        assert self.dispatcher is not None
        self.dispatcher.emit(event, *args)

    dispatch = emit

    def build_report(self) -> dict[str, Any]:
        assert self.dispatcher is not None
        return self.dispatcher.build_report()

    def build_capability_report(self) -> dict[str, Any]:
        assert self.capability_manager is not None
        return self.capability_manager.build_report()

    def describe_event_table(self) -> tuple[dict[str, Any], ...]:
        assert self.dispatcher is not None
        return self.dispatcher.describe_event_table()

    def describe_control_plane_contract(self) -> dict[str, Any]:
        return describe_control_plane_contract(lifecycle_events=self.describe_event_table())


__all__ = ["LifecycleRuntime"]
