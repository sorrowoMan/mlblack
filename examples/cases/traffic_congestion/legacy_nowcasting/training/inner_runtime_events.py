from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


INNER_RUNTIME_SYMBOLIC_STRUCTURE_SEARCH = "symbolic_structure_search"
INNER_RUNTIME_SYMBOLIC_INTERVAL_CORE = "symbolic_interval_core"
INNER_RUNTIME_SYMBOLIC_INTERVAL_PIECEWISE = "symbolic_interval_piecewise"
INNER_RUNTIME_BRANCH_EVALUATION_GLOBAL_FOLD = "branch_evaluation.global_fold"
INNER_RUNTIME_BRANCH_EVALUATION_REGIME_FOLD = "branch_evaluation.regime_fold"
INNER_RUNTIME_BRANCH_EVALUATION_FOLD_BATCH = "branch_evaluation.fold_batch"


@dataclass(frozen=True)
class InnerRuntimeEventSpec:
    runtime_key: str
    source_layers: tuple[str, ...]
    source_modules: tuple[str, ...]
    dispatch_names: tuple[str, ...]
    description: str
    payload_contract: Mapping[str, Any]
    context_contract: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "runtime_key": str(self.runtime_key),
            "source_layers": tuple(str(x) for x in self.source_layers),
            "source_modules": tuple(str(x) for x in self.source_modules),
            "dispatch_names": tuple(str(x) for x in self.dispatch_names),
            "description": str(self.description),
            "payload_contract": dict(self.payload_contract),
            "context_contract": dict(self.context_contract),
        }


_DEFAULT_DISPATCH_NAMES = (
    "on_inner_run_start",
    "on_inner_round_end",
    "on_inner_run_finish",
    "on_inner_run_error",
)

_DEFAULT_CONTEXT_CONTRACT = {
    "required": ("task_id", "run_id", "runtime_key", "trainer_name"),
    "forbidden": (),
}


INNER_RUNTIME_EVENT_TABLE: tuple[InnerRuntimeEventSpec, ...] = (
    InnerRuntimeEventSpec(
        runtime_key=INNER_RUNTIME_SYMBOLIC_STRUCTURE_SEARCH,
        source_layers=("trainer",),
        source_modules=(
            "core.trainers.symbolic_stagewise_trainer",
            "core.symbolic.symbolic_structure_search",
        ),
        dispatch_names=_DEFAULT_DISPATCH_NAMES,
        description="Stagewise symbolic structure search inner loop.",
        payload_contract={
            "start_payload": "InnerRuntimeStartPayload",
            "round_payload": "InnerRuntimeRoundPayload",
            "finish_payload": "InnerRuntimeFinishPayload",
            "error_payload": "InnerRuntimeErrorPayload",
            "round_unit": "search_round",
            "typed_payloads": True,
        },
        context_contract={
            **_DEFAULT_CONTEXT_CONTRACT,
            "recommended": ("search_driver", "structure_mode"),
        },
    ),
    InnerRuntimeEventSpec(
        runtime_key=INNER_RUNTIME_SYMBOLIC_INTERVAL_CORE,
        source_layers=("trainer",),
        source_modules=("core.trainers.symbolic_torch_interval_trainer",),
        dispatch_names=_DEFAULT_DISPATCH_NAMES,
        description="Symbolic interval trainer core epoch loop.",
        payload_contract={
            "start_payload": "InnerRuntimeStartPayload",
            "round_payload": "InnerRuntimeRoundPayload",
            "finish_payload": "InnerRuntimeFinishPayload",
            "error_payload": "InnerRuntimeErrorPayload",
            "round_unit": "epoch",
            "typed_payloads": True,
        },
        context_contract={
            **_DEFAULT_CONTEXT_CONTRACT,
            "recommended": ("run_tag", "scope", "device"),
        },
    ),
    InnerRuntimeEventSpec(
        runtime_key=INNER_RUNTIME_SYMBOLIC_INTERVAL_PIECEWISE,
        source_layers=("trainer",),
        source_modules=("core.trainers.symbolic_torch_interval_trainer",),
        dispatch_names=_DEFAULT_DISPATCH_NAMES,
        description="Symbolic interval piecewise regime aggregation loop.",
        payload_contract={
            "start_payload": "InnerRuntimeStartPayload",
            "round_payload": "InnerRuntimeRoundPayload",
            "finish_payload": "InnerRuntimeFinishPayload",
            "error_payload": "InnerRuntimeErrorPayload",
            "round_unit": "regime_model",
            "typed_payloads": True,
        },
        context_contract={
            **_DEFAULT_CONTEXT_CONTRACT,
            "recommended": ("scope", "gate_feature_count"),
        },
    ),
    InnerRuntimeEventSpec(
        runtime_key=INNER_RUNTIME_BRANCH_EVALUATION_GLOBAL_FOLD,
        source_layers=("problem", "evaluation"),
        source_modules=(
            "evaluation.problem_callbacks",
            "core.symbolic.feature_space.branch_evaluator",
        ),
        dispatch_names=_DEFAULT_DISPATCH_NAMES,
        description="Global fold evaluation loop for branch/regime-aware problems.",
        payload_contract={
            "start_payload": "InnerRuntimeStartPayload",
            "round_payload": "InnerRuntimeRoundPayload",
            "finish_payload": "InnerRuntimeFinishPayload",
            "error_payload": "InnerRuntimeErrorPayload",
            "round_unit": "fold",
            "typed_payloads": True,
        },
        context_contract={
            **_DEFAULT_CONTEXT_CONTRACT,
            "recommended": ("fold_id", "fold_kind", "train_size", "val_size"),
        },
    ),
    InnerRuntimeEventSpec(
        runtime_key=INNER_RUNTIME_BRANCH_EVALUATION_REGIME_FOLD,
        source_layers=("problem", "evaluation"),
        source_modules=(
            "evaluation.problem_callbacks",
            "core.symbolic.feature_space.branch_evaluator",
        ),
        dispatch_names=_DEFAULT_DISPATCH_NAMES,
        description="Per-regime branch evaluation loop inside one fold.",
        payload_contract={
            "start_payload": "InnerRuntimeStartPayload",
            "round_payload": "InnerRuntimeRoundPayload",
            "finish_payload": "InnerRuntimeFinishPayload",
            "error_payload": "InnerRuntimeErrorPayload",
            "round_unit": "regime_branch",
            "typed_payloads": True,
        },
        context_contract={
            **_DEFAULT_CONTEXT_CONTRACT,
            "recommended": ("fold_id", "fold_kind", "train_size", "val_size"),
        },
    ),
    InnerRuntimeEventSpec(
        runtime_key=INNER_RUNTIME_BRANCH_EVALUATION_FOLD_BATCH,
        source_layers=("problem", "evaluation"),
        source_modules=(
            "evaluation.problem_callbacks",
            "core.symbolic.feature_space.branch_evaluator",
        ),
        dispatch_names=_DEFAULT_DISPATCH_NAMES,
        description="Batched fold interval evaluation loop over branch-aware regimes.",
        payload_contract={
            "start_payload": "InnerRuntimeStartPayload",
            "round_payload": "InnerRuntimeRoundPayload",
            "finish_payload": "InnerRuntimeFinishPayload",
            "error_payload": "InnerRuntimeErrorPayload",
            "round_unit": "regime_batch",
            "typed_payloads": True,
        },
        context_contract={
            **_DEFAULT_CONTEXT_CONTRACT,
            "recommended": ("fold_id", "fold_kind", "batch_key_prefix", "batch_size"),
        },
    ),
)

_INNER_RUNTIME_EVENT_INDEX: dict[str, InnerRuntimeEventSpec] = {
    str(spec.runtime_key): spec for spec in INNER_RUNTIME_EVENT_TABLE
}


def resolve_inner_runtime_event(runtime_key: str) -> InnerRuntimeEventSpec | None:
    return _INNER_RUNTIME_EVENT_INDEX.get(str(runtime_key).strip())


def describe_inner_runtime_event_table() -> tuple[dict[str, Any], ...]:
    return tuple(spec.as_dict() for spec in INNER_RUNTIME_EVENT_TABLE)


__all__ = [
    "INNER_RUNTIME_BRANCH_EVALUATION_FOLD_BATCH",
    "INNER_RUNTIME_BRANCH_EVALUATION_GLOBAL_FOLD",
    "INNER_RUNTIME_BRANCH_EVALUATION_REGIME_FOLD",
    "INNER_RUNTIME_EVENT_TABLE",
    "INNER_RUNTIME_SYMBOLIC_INTERVAL_CORE",
    "INNER_RUNTIME_SYMBOLIC_INTERVAL_PIECEWISE",
    "INNER_RUNTIME_SYMBOLIC_STRUCTURE_SEARCH",
    "InnerRuntimeEventSpec",
    "describe_inner_runtime_event_table",
    "resolve_inner_runtime_event",
]
