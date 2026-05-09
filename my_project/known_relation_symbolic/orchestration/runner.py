from __future__ import annotations

from typing import Any, Mapping, Sequence

from config import CapabilitySpec, FlowAssemblySpec, NumericizerSpec, TrainerAssemblySpec
from training import TrainingInit
from workflow import SemanticTrainFlowSpec

from my_project.known_relation_symbolic.orchestration.hints import trainer_params_overrides


def normalize_orthogonal_override_key(key: Any) -> str:
    text = str(key or "").strip()
    if text.startswith("orth_"):
        return str(text[len("orth_") :]).strip()
    return text


def resolve_orthogonal_trainer_overrides(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    raw_overrides = dict(trainer_params_overrides(metadata) or {})
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in raw_overrides.items():
        key = normalize_orthogonal_override_key(raw_key)
        if not key:
            continue
        normalized[str(key)] = raw_value
    return normalized


def build_known_relation_semantic_flow_spec(
    *,
    trainer_params: Mapping[str, Any],
    run_name: str,
    output_dir: str,
    db_path: str,
    namespace: str,
    tag: str,
    training_init: TrainingInit | None = None,
    trainer_key: str = "symbolic",
    eval_splits: Sequence[str] = ("train", "test"),
) -> SemanticTrainFlowSpec:
    return SemanticTrainFlowSpec(
        assembly=FlowAssemblySpec(
            trainer=TrainerAssemblySpec(
                trainer_key=str(trainer_key),
                trainer_params=dict(trainer_params),
            ),
            numericizer=NumericizerSpec(key="default", params={}),
            capabilities=(
                CapabilitySpec(
                    key="experiment_tracker",
                    params={
                        "db_path": str(db_path),
                        "namespace": str(namespace),
                        "tag": str(tag),
                        "io_mode": "batched",
                        "commit_interval": 0,
                    },
                ),
            ),
        ),
        eval_splits=tuple(str(split) for split in tuple(eval_splits)),
        output_dir=str(output_dir),
        save_artifact=True,
        save_report=True,
        capability_strict=True,
        run_name=str(run_name),
        training_init=training_init,
    )


__all__ = [
    "build_known_relation_semantic_flow_spec",
    "normalize_orthogonal_override_key",
    "resolve_orthogonal_trainer_overrides",
]
