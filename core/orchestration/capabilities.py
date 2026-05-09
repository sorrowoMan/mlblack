from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Sequence

from .lifecycle_dispatcher import LifecycleDispatcher


@dataclass
class FlowCapability:
    """Base capability for train-flow and runtime lifecycle hooks."""

    name: str
    priority: int = 0
    enabled: bool = True
    is_algorithmic: bool = False
    config: Dict[str, Any] = field(default_factory=dict)

    # Optional context-contract metadata
    context_requires: Sequence[str] = field(default_factory=tuple)
    context_provides: Sequence[str] = field(default_factory=tuple)
    context_mutates: Sequence[str] = field(default_factory=tuple)
    context_cache: Sequence[str] = field(default_factory=tuple)
    context_notes: str | None = None

    def on_flow_start(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_data_ready(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_pre_fit(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_post_fit(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_pre_eval(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_post_eval(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_pre_persist(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_post_persist(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_flow_finish(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_flow_error(self, error: Exception, context: MutableMapping[str, Any]) -> None:
        return None

    # Optional generic runtime hooks used by stage-oriented orchestrators.
    def on_experiment_start(self, context: MutableMapping[str, Any]) -> None:
        return None

    def on_stage_start(self, stage: str, context: MutableMapping[str, Any]) -> None:
        return None

    def on_stage_end(self, stage: str, payload: Mapping[str, Any], context: MutableMapping[str, Any]) -> None:
        return None

    def on_stage_error(self, stage: str, error: Exception, context: MutableMapping[str, Any]) -> None:
        return None

    def on_experiment_finish(self, result: Any, context: MutableMapping[str, Any]) -> None:
        return None

    def on_experiment_error(self, error: Exception, context: MutableMapping[str, Any]) -> None:
        return None

    def get_context_contract(self) -> Dict[str, Any]:
        return {
            "requires": tuple(str(x) for x in self.context_requires),
            "provides": tuple(str(x) for x in self.context_provides),
            "mutates": tuple(str(x) for x in self.context_mutates),
            "cache": tuple(str(x) for x in self.context_cache),
            "notes": self.context_notes,
        }


class CapabilityManager:
    """Lifecycle dispatcher for flow capabilities."""

    def __init__(
        self,
        *,
        strict: bool = False,
        dispatcher: LifecycleDispatcher | None = None,
    ) -> None:
        self.strict = bool(strict)
        self._dispatcher = dispatcher or LifecycleDispatcher(strict=bool(strict))
        self._capabilities: list[FlowCapability] = []

    def register(self, capability: FlowCapability) -> None:
        if any(c.name == capability.name for c in self._capabilities):
            raise ValueError(f"Capability '{capability.name}' already registered")
        self._capabilities.append(capability)
        self._dispatcher.register(capability)

    def list_capabilities(self) -> tuple[FlowCapability, ...]:
        return tuple(sorted(self._capabilities, key=lambda x: (int(x.priority), str(x.name))))

    def dispatch(self, event: str, *args: Any) -> None:
        self._dispatcher.dispatch(event, *args)

    def build_report(self) -> Dict[str, Any]:
        return self._dispatcher.build_report(participants=self.list_capabilities())

    def describe_event_table(self) -> tuple[dict[str, Any], ...]:
        return self._dispatcher.describe_event_table()


__all__ = [
    "FlowCapability",
    "CapabilityManager",
]
