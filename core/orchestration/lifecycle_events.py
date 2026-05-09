from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


LifecycleArgsAdapter = Callable[[tuple[Any, ...]], tuple[Any, ...]]


def _identity_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(args)


def _context_only_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(args) == 1 and isinstance(args[0], Mapping):
        return (args[0],)
    if len(args) >= 2 and isinstance(args[-1], Mapping):
        return (args[-1],)
    raise TypeError("lifecycle event requires a context mapping as the last argument")


def _result_and_context_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(args) == 2 and isinstance(args[1], Mapping):
        return (args[0], args[1])
    if len(args) == 1 and isinstance(args[0], Mapping):
        context = args[0]
        result = context.get("result")
        return (result, context)
    raise TypeError("lifecycle event requires either (result, context) or (context)")


def _error_and_context_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(args) == 2 and isinstance(args[1], Mapping):
        return (args[0], args[1])
    raise TypeError("lifecycle error event requires (error, context)")


@dataclass(frozen=True)
class LifecycleHookBinding:
    hook_name: str
    adapter_name: str = "identity"
    adapter: LifecycleArgsAdapter = field(default=_identity_args, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "hook_name": str(self.hook_name),
            "adapter": str(self.adapter_name),
        }


@dataclass(frozen=True)
class LifecycleEventSpec:
    semantic_key: str
    scope: str
    dispatch_names: tuple[str, ...]
    hook_bindings: tuple[LifecycleHookBinding, ...]
    description: str
    payload_contract: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "semantic_key": str(self.semantic_key),
            "scope": str(self.scope),
            "dispatch_names": tuple(str(x) for x in self.dispatch_names),
            "hook_bindings": tuple(binding.as_dict() for binding in self.hook_bindings),
            "description": str(self.description),
            "payload_contract": {} if self.payload_contract is None else dict(self.payload_contract),
        }


LIFECYCLE_EVENT_TABLE: tuple[LifecycleEventSpec, ...] = (
    LifecycleEventSpec(
        semantic_key="run_start",
        scope="shared",
        dispatch_names=("on_flow_start", "on_experiment_start"),
        hook_bindings=(
            LifecycleHookBinding("on_flow_start", adapter_name="context_only", adapter=_context_only_args),
            LifecycleHookBinding("on_experiment_start", adapter_name="context_only", adapter=_context_only_args),
        ),
        description="Run-level start event shared by train-flow and runtime orchestrator.",
        payload_contract={
            "args": ("context",),
            "typed_payload": None,
            "context_required": True,
        },
    ),
    LifecycleEventSpec(
        semantic_key="data_ready",
        scope="flow",
        dispatch_names=("on_data_ready",),
        hook_bindings=(
            LifecycleHookBinding("on_data_ready", adapter_name="context_only", adapter=_context_only_args),
        ),
        description="Processed data is ready for downstream fit/eval stages.",
        payload_contract={
            "args": ("context",),
            "typed_payload": None,
            "context_required": True,
        },
    ),
    LifecycleEventSpec(
        semantic_key="pre_fit",
        scope="flow",
        dispatch_names=("on_pre_fit",),
        hook_bindings=(
            LifecycleHookBinding("on_pre_fit", adapter_name="context_only", adapter=_context_only_args),
        ),
        description="Immediately before trainer.fit_task(...) runs.",
        payload_contract={
            "args": ("context",),
            "typed_payload": None,
            "context_required": True,
        },
    ),
    LifecycleEventSpec(
        semantic_key="post_fit",
        scope="flow",
        dispatch_names=("on_post_fit",),
        hook_bindings=(
            LifecycleHookBinding("on_post_fit", adapter_name="context_only", adapter=_context_only_args),
        ),
        description="Immediately after trainer.fit_task(...) completes.",
        payload_contract={
            "args": ("context",),
            "typed_payload": None,
            "context_required": True,
        },
    ),
    LifecycleEventSpec(
        semantic_key="pre_eval",
        scope="flow",
        dispatch_names=("on_pre_eval",),
        hook_bindings=(
            LifecycleHookBinding("on_pre_eval", adapter_name="context_only", adapter=_context_only_args),
        ),
        description="Before evaluation metrics are computed.",
        payload_contract={
            "args": ("context",),
            "typed_payload": None,
            "context_required": True,
        },
    ),
    LifecycleEventSpec(
        semantic_key="post_eval",
        scope="flow",
        dispatch_names=("on_post_eval",),
        hook_bindings=(
            LifecycleHookBinding("on_post_eval", adapter_name="context_only", adapter=_context_only_args),
        ),
        description="After evaluation metrics are computed.",
        payload_contract={
            "args": ("context",),
            "typed_payload": None,
            "context_required": True,
        },
    ),
    LifecycleEventSpec(
        semantic_key="pre_persist",
        scope="flow",
        dispatch_names=("on_pre_persist",),
        hook_bindings=(
            LifecycleHookBinding("on_pre_persist", adapter_name="context_only", adapter=_context_only_args),
        ),
        description="Before artifact/report/checkpoint persistence.",
        payload_contract={
            "args": ("context",),
            "typed_payload": None,
            "context_required": True,
        },
    ),
    LifecycleEventSpec(
        semantic_key="post_persist",
        scope="flow",
        dispatch_names=("on_post_persist",),
        hook_bindings=(
            LifecycleHookBinding("on_post_persist", adapter_name="context_only", adapter=_context_only_args),
        ),
        description="After artifact/report/checkpoint persistence.",
        payload_contract={
            "args": ("context",),
            "typed_payload": None,
            "context_required": True,
        },
    ),
    LifecycleEventSpec(
        semantic_key="run_finish",
        scope="shared",
        dispatch_names=("on_flow_finish", "on_experiment_finish"),
        hook_bindings=(
            LifecycleHookBinding("on_flow_finish", adapter_name="context_only", adapter=_context_only_args),
            LifecycleHookBinding("on_experiment_finish", adapter_name="result_and_context", adapter=_result_and_context_args),
        ),
        description="Run-level finish event shared by train-flow and runtime orchestrator.",
        payload_contract={
            "args": ("result", "context"),
            "typed_payload": None,
            "context_required": True,
        },
    ),
    LifecycleEventSpec(
        semantic_key="run_error",
        scope="shared",
        dispatch_names=("on_flow_error", "on_experiment_error"),
        hook_bindings=(
            LifecycleHookBinding("on_flow_error", adapter_name="error_and_context", adapter=_error_and_context_args),
            LifecycleHookBinding("on_experiment_error", adapter_name="error_and_context", adapter=_error_and_context_args),
        ),
        description="Run-level error event shared by train-flow and runtime orchestrator.",
        payload_contract={
            "args": ("error", "context"),
            "typed_payload": None,
            "context_required": True,
        },
    ),
    LifecycleEventSpec(
        semantic_key="stage_start",
        scope="runtime",
        dispatch_names=("on_stage_start",),
        hook_bindings=(
            LifecycleHookBinding("on_stage_start", adapter_name="identity", adapter=_identity_args),
        ),
        description="Runtime stage start event.",
        payload_contract={
            "args": ("stage", "context"),
            "typed_payload": None,
            "context_required": True,
        },
    ),
    LifecycleEventSpec(
        semantic_key="stage_end",
        scope="runtime",
        dispatch_names=("on_stage_end",),
        hook_bindings=(
            LifecycleHookBinding("on_stage_end", adapter_name="identity", adapter=_identity_args),
        ),
        description="Runtime stage end event.",
        payload_contract={
            "args": ("stage", "payload", "context"),
            "typed_payload": "StageLifecyclePayload",
            "typed_result_descriptor": "StageResultDescriptor",
            "payload_fields": (
                "stage",
                "status",
                "started_at",
                "finished_at",
                "duration_sec",
                "result_descriptor",
                "context_keys",
                "new_context_keys",
            ),
            "context_required": True,
        },
    ),
    LifecycleEventSpec(
        semantic_key="stage_error",
        scope="runtime",
        dispatch_names=("on_stage_error",),
        hook_bindings=(
            LifecycleHookBinding("on_stage_error", adapter_name="identity", adapter=_identity_args),
        ),
        description="Runtime stage error event.",
        payload_contract={
            "args": ("stage", "error", "context"),
            "typed_payload": None,
            "error_fields": ("type", "message"),
            "context_required": True,
        },
    ),
)

_EVENT_INDEX: dict[str, LifecycleEventSpec] = {
    str(dispatch_name): spec
    for spec in LIFECYCLE_EVENT_TABLE
    for dispatch_name in spec.dispatch_names
}


def resolve_lifecycle_event(name: str) -> LifecycleEventSpec | None:
    return _EVENT_INDEX.get(str(name).strip())


def describe_lifecycle_event_table() -> tuple[dict[str, Any], ...]:
    return tuple(spec.as_dict() for spec in LIFECYCLE_EVENT_TABLE)


__all__ = [
    "LifecycleEventSpec",
    "LifecycleHookBinding",
    "LIFECYCLE_EVENT_TABLE",
    "describe_lifecycle_event_table",
    "resolve_lifecycle_event",
]
