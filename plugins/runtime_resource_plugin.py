from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from core.orchestration.capabilities import FlowCapability
from nowcasting_work_ci.mlblack_side.runtime.contracts import RuntimeContextKey, ctx_get, ctx_set


@dataclass
class RuntimeResourcePlugin(FlowCapability):
    name: str = "runtime_resource_cleanup"
    priority: int = 300
    enabled: bool = True
    is_algorithmic: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    context_requires: Sequence[str] = (RuntimeContextKey.GRAPH_CACHE_RESOURCE.value,)
    context_provides: Sequence[str] = tuple()
    context_mutates: Sequence[str] = (RuntimeContextKey.GRAPH_CACHE_RESOURCE.value,)
    context_cache: Sequence[str] = (RuntimeContextKey.GRAPH_CACHE_RESOURCE.value,)
    context_notes: str | None = "Closes runtime resources such as graph-cache handles on finish/error."
    graph_cache_key: str = RuntimeContextKey.GRAPH_CACHE_RESOURCE.value

    def on_experiment_start(self, context: Mapping[str, Any]) -> None:
        return

    def on_stage_start(self, stage: str, context: Mapping[str, Any]) -> None:
        return

    def on_stage_end(self, stage: str, payload: Mapping[str, Any], context: Mapping[str, Any]) -> None:
        return

    def on_stage_error(self, stage: str, error: Exception, context: Mapping[str, Any]) -> None:
        self._cleanup(context)

    def on_experiment_finish(self, result: Any, context: Mapping[str, Any]) -> None:
        self._cleanup(context)

    def on_experiment_error(self, error: Exception, context: Mapping[str, Any]) -> None:
        self._cleanup(context)

    def _cleanup(self, context: Mapping[str, Any]) -> None:
        resource = ctx_get(context, RuntimeContextKey.GRAPH_CACHE_RESOURCE, default=None)
        if resource is None:
            return
        close_fn = getattr(resource, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                return
        if isinstance(context, dict):
            ctx_set(context, RuntimeContextKey.GRAPH_CACHE_RESOURCE, None)


__all__ = ["RuntimeResourcePlugin"]
