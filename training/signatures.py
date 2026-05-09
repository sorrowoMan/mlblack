from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from core.common.contracts import ProcessedDataset, SampleDataset


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "shape": tuple(int(v) for v in value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    return str(value)


def _stable_hash(payload: Any) -> str | None:
    if payload is None:
        return None
    encoded = json.dumps(_jsonable(payload), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    if encoded in {"null", "{}", "[]", "\"\""}:
        return None
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _mapping_copy(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {} if value is None else {str(k): v for k, v in dict(value).items()}


def _sequence_or_none(value: Any) -> list[str] | None:
    if value is None:
        return None
    return [str(v) for v in tuple(value)]


def _first_sample_feature_keys(data: SampleDataset) -> list[str] | None:
    if data.feature_cell_keys is not None:
        return [str(v) for v in tuple(data.feature_cell_keys)]
    if not data.samples:
        return None
    first = data.samples[0]
    return [str(v) for v in sorted(first.cells.keys())]


def _data_metadata(data: ProcessedDataset | SampleDataset) -> dict[str, Any]:
    if isinstance(data, ProcessedDataset):
        return _mapping_copy(data.metadata)
    return {}


def _data_protocol(data: ProcessedDataset | SampleDataset) -> str:
    if isinstance(data, ProcessedDataset):
        meta = _data_metadata(data)
        return str(meta.get("flow_input_protocol") or meta.get("input_protocol") or "processed_dataset")
    return "sample_dataset"


def _feature_descriptor(data: ProcessedDataset | SampleDataset) -> dict[str, Any]:
    if isinstance(data, ProcessedDataset):
        x = np.asarray(data.X_train)
        dim = int(x.shape[1]) if x.ndim == 2 else 1
        return {
            "protocol": _data_protocol(data),
            "feature_names": _sequence_or_none(data.feature_names),
            "feature_dim": dim,
        }
    return {
        "protocol": _data_protocol(data),
        "feature_keys": _first_sample_feature_keys(data),
        "feature_count_hint": None
        if _first_sample_feature_keys(data) is None
        else int(len(_first_sample_feature_keys(data) or ())),
    }


def _target_descriptor(data: ProcessedDataset | SampleDataset) -> dict[str, Any]:
    if isinstance(data, ProcessedDataset):
        y = np.asarray(data.y_train)
        dim = int(y.shape[1]) if y.ndim == 2 else 1
        return {
            "protocol": _data_protocol(data),
            "target_names": _sequence_or_none(data.target_names),
            "target_dim": dim,
        }
    return {
        "protocol": _data_protocol(data),
        "target_key": str(data.target_key),
        "target_names": _sequence_or_none(data.target_names),
    }


def _schema_descriptor(task: Any) -> dict[str, Any]:
    return {
        "protocol": _data_protocol(task.data),
        "schema": None if task.schema is None else _jsonable(task.schema),
        "feature": _feature_descriptor(task.data),
        "target": _target_descriptor(task.data),
    }


def _objective_descriptor(task: Any) -> Any:
    if task.objective is not None:
        return _jsonable(task.objective)
    return None


def _trainer_family(trainer: Any | None) -> str | None:
    if trainer is None:
        return None
    name = getattr(trainer, "name", None)
    if name is not None:
        return str(name)
    return str(type(trainer).__name__)


def _pipeline_name_from_trainer(trainer: Any | None) -> str | None:
    if trainer is None:
        return None
    pipeline = getattr(trainer, "pipeline", None)
    if pipeline is None:
        return None
    return str(getattr(pipeline, "name", type(pipeline).__name__))


def _numericizer_descriptor(task: Any, trainer: Any | None) -> dict[str, Any] | None:
    name: str | None = None
    if trainer is not None:
        numericizer = getattr(trainer, "numericizer", None)
        if numericizer is not None:
            name = str(getattr(numericizer, "name", type(numericizer).__name__))

    data_meta = _data_metadata(task.data)
    encoding_plan = data_meta.get("encoding_plan")
    data_numericizer = data_meta.get("numericizer")
    if name is None and data_numericizer is not None:
        name = str(data_numericizer)

    if name is None and encoding_plan is None:
        return None

    return {
        "numericizer_name": name,
        "encoding_plan": _jsonable(encoding_plan),
    }


def _regime_descriptor(task: Any) -> dict[str, Any] | None:
    task_meta = _mapping_copy(getattr(task, "metadata", None))
    data_meta = _data_metadata(task.data)
    keys = (
        "regime_policy",
        "strict4_regime_policy",
        "router_spec",
        "regime_router",
        "branch_policy",
        "gate_piecewise",
    )
    payload: dict[str, Any] = {}
    for key in keys:
        if key in task_meta:
            payload[str(key)] = _jsonable(task_meta.get(key))
        elif key in data_meta:
            payload[str(key)] = _jsonable(data_meta.get(key))
    return payload or None


def _symbolic_family_descriptor(task: Any, trainer: Any | None) -> dict[str, Any] | None:
    if trainer is not None:
        family = getattr(trainer, "symbolic_family_spec", None)
        if family is not None:
            if hasattr(family, "family_signature_payload"):
                try:
                    return _jsonable(family.family_signature_payload())
                except Exception:
                    pass
            if hasattr(family, "description_dict"):
                try:
                    return _jsonable(family.description_dict())
                except Exception:
                    pass
            if hasattr(family, "as_dict"):
                return _jsonable(family.as_dict())
            if isinstance(family, Mapping):
                return _jsonable(dict(family))
        trainer_name = str(getattr(trainer, "name", "")).strip().lower()
        if trainer_name.startswith("symbolic"):
            resolve_engine = getattr(trainer, "_resolve_structure_engine", None)
            structure_engine = None
            if callable(resolve_engine):
                try:
                    structure_engine = resolve_engine()
                except Exception:
                    structure_engine = None
            elif getattr(getattr(trainer, "config", None), "structure_engine", None) is not None:
                structure_engine = getattr(getattr(trainer, "config", None), "structure_engine", None)
            if structure_engine is not None:
                if hasattr(structure_engine, "as_dict"):
                    structure_payload = structure_engine.as_dict()
                elif isinstance(structure_engine, Mapping):
                    structure_payload = dict(structure_engine)
                else:
                    structure_payload = {"structure_mode": str(structure_engine)}
                return _jsonable(
                    {
                        "trainer_key": trainer_name,
                        "structure_engine": structure_payload,
                        "parameter_backend": {
                            "backend": "torch" if "torch" in trainer_name else "ridge",
                        },
                        "task_head": {
                            "task": "interval" if "interval" in trainer_name else "point",
                        },
                    }
                )

    task_meta = _mapping_copy(getattr(task, "metadata", None))
    data_meta = _data_metadata(task.data)
    for key in ("symbolic_family", "symbolic_family_spec"):
        if key in task_meta:
            return _jsonable(task_meta.get(key))
        if key in data_meta:
            return _jsonable(data_meta.get(key))
    return None


def _symbolic_family_signature_value(
    task: Any,
    trainer: Any | None,
    *,
    symbolic_family_desc: dict[str, Any] | None,
) -> str | None:
    if trainer is not None:
        family = getattr(trainer, "symbolic_family_spec", None)
        if family is not None and hasattr(family, "family_signature"):
            try:
                value = family.family_signature()
            except Exception:
                value = None
            if value:
                return str(value)

    task_meta = _mapping_copy(getattr(task, "metadata", None))
    data_meta = _data_metadata(task.data)
    for container in (task_meta, data_meta):
        value = container.get("symbolic_family_signature")
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return _stable_hash(symbolic_family_desc)


@dataclass(frozen=True)
class TrainingSignature:
    trainer_family: str | None = None
    data_protocol: str | None = None
    schema_signature: str | None = None
    feature_signature: str | None = None
    target_signature: str | None = None
    objective_signature: str | None = None
    pipeline_signature: str | None = None
    numericizer_signature: str | None = None
    regime_signature: str | None = None
    symbolic_family_signature: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trainer_family", None if self.trainer_family is None else str(self.trainer_family))
        object.__setattr__(self, "data_protocol", None if self.data_protocol is None else str(self.data_protocol))
        object.__setattr__(
            self,
            "symbolic_family_signature",
            None if self.symbolic_family_signature is None else str(self.symbolic_family_signature),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "trainer_family": self.trainer_family,
            "data_protocol": self.data_protocol,
            "schema_signature": self.schema_signature,
            "feature_signature": self.feature_signature,
            "target_signature": self.target_signature,
            "objective_signature": self.objective_signature,
            "pipeline_signature": self.pipeline_signature,
            "numericizer_signature": self.numericizer_signature,
            "regime_signature": self.regime_signature,
            "symbolic_family_signature": self.symbolic_family_signature,
            "metadata": dict(self.metadata),
        }


def coerce_training_signature(
    value: TrainingSignature | Mapping[str, Any] | None,
) -> TrainingSignature:
    if value is None:
        return TrainingSignature()
    if isinstance(value, TrainingSignature):
        return value
    raw = dict(value)
    return TrainingSignature(
        trainer_family=raw.get("trainer_family"),
        data_protocol=raw.get("data_protocol"),
        schema_signature=raw.get("schema_signature"),
        feature_signature=raw.get("feature_signature"),
        target_signature=raw.get("target_signature"),
        objective_signature=raw.get("objective_signature"),
        pipeline_signature=raw.get("pipeline_signature"),
        numericizer_signature=raw.get("numericizer_signature"),
        regime_signature=raw.get("regime_signature"),
        symbolic_family_signature=raw.get("symbolic_family_signature"),
        metadata=dict(raw.get("metadata", {})),
    )


def build_task_signature(task: Any, *, trainer: Any | None = None) -> TrainingSignature:
    feature_desc = _feature_descriptor(task.data)
    target_desc = _target_descriptor(task.data)
    schema_desc = _schema_descriptor(task)
    objective_desc = _objective_descriptor(task)
    numericizer_desc = _numericizer_descriptor(task, trainer)
    regime_desc = _regime_descriptor(task)
    symbolic_family_desc = _symbolic_family_descriptor(task, trainer)
    pipeline_name = _pipeline_name_from_trainer(trainer)

    return TrainingSignature(
        trainer_family=_trainer_family(trainer),
        data_protocol=_data_protocol(task.data),
        schema_signature=_stable_hash(schema_desc),
        feature_signature=_stable_hash(feature_desc),
        target_signature=_stable_hash(target_desc),
        objective_signature=_stable_hash(objective_desc),
        pipeline_signature=_stable_hash({"pipeline_name": pipeline_name}),
        numericizer_signature=_stable_hash(numericizer_desc),
        regime_signature=_stable_hash(regime_desc),
        symbolic_family_signature=_symbolic_family_signature_value(
            task,
            trainer,
            symbolic_family_desc=symbolic_family_desc,
        ),
        metadata={
            "task_id": str(getattr(task, "task_id", "train_task")),
            "feature_descriptor": feature_desc,
            "target_descriptor": target_desc,
            "pipeline_name": pipeline_name,
            "numericizer": numericizer_desc,
            "regime": regime_desc,
            "symbolic_family": symbolic_family_desc,
        },
    )


def signature_from_artifact(artifact: Any) -> TrainingSignature:
    metadata = _mapping_copy(getattr(artifact, "metadata", None))
    embedded = metadata.get("training_signature")
    if isinstance(embedded, Mapping):
        return coerce_training_signature(embedded)

    pipeline_name = getattr(artifact, "pipeline_name", None)
    if pipeline_name is None:
        pipeline_name = metadata.get("pipeline_name") or metadata.get("pipeline")

    numericizer_name = metadata.get("numericizer")
    numericizer_desc = None
    if numericizer_name is not None or metadata.get("data_metadata", {}).get("encoding_plan") is not None:
        numericizer_desc = {
            "numericizer_name": None if numericizer_name is None else str(numericizer_name),
            "encoding_plan": _jsonable(_mapping_copy(metadata.get("data_metadata")).get("encoding_plan")),
        }

    feature_desc = {
        "protocol": str(_mapping_copy(metadata.get("data_metadata")).get("flow_input_protocol", "processed_dataset")),
        "feature_names": _sequence_or_none(getattr(artifact, "feature_names", None)),
        "feature_dim": None if getattr(artifact, "coef", None) is None else int(np.asarray(artifact.coef).shape[0]),
    }
    target_desc = {
        "protocol": str(_mapping_copy(metadata.get("data_metadata")).get("flow_input_protocol", "processed_dataset")),
        "target_names": _sequence_or_none(getattr(artifact, "target_names", None)),
        "target_dim": None
        if getattr(artifact, "coef", None) is None
        else int(np.asarray(artifact.coef).shape[1] if np.asarray(artifact.coef).ndim == 2 else 1),
    }
    regime_desc = None
    for key in (
        "regime_policy",
        "strict4_regime_policy",
        "router_spec",
        "regime_router",
        "branch_policy",
        "gate_piecewise",
        ):
        if key in metadata:
            regime_desc = {str(key): _jsonable(metadata[key])}
            break
    symbolic_family_desc = None
    for key in ("symbolic_family", "symbolic_family_spec"):
        if key in metadata:
            symbolic_family_desc = _jsonable(metadata[key])
            break

    return TrainingSignature(
        trainer_family=None if metadata.get("trainer_family") is None else str(metadata.get("trainer_family")),
        data_protocol=str(_mapping_copy(metadata.get("data_metadata")).get("flow_input_protocol", "processed_dataset")),
        schema_signature=None if metadata.get("schema_signature") is None else str(metadata.get("schema_signature")),
        feature_signature=_stable_hash(feature_desc),
        target_signature=_stable_hash(target_desc),
        objective_signature=None
        if metadata.get("objective_signature") is None
        else str(metadata.get("objective_signature")),
        pipeline_signature=_stable_hash({"pipeline_name": None if pipeline_name is None else str(pipeline_name)}),
        numericizer_signature=_stable_hash(numericizer_desc),
        regime_signature=_stable_hash(regime_desc),
        symbolic_family_signature=(
            None
            if metadata.get("symbolic_family_signature") is None and symbolic_family_desc is None
            else str(metadata.get("symbolic_family_signature"))
            if metadata.get("symbolic_family_signature") is not None
            else _stable_hash(symbolic_family_desc)
        ),
        metadata={
            "artifact_id": str(getattr(artifact, "artifact_id", type(artifact).__name__)),
            "feature_descriptor": feature_desc,
            "target_descriptor": target_desc,
            "pipeline_name": None if pipeline_name is None else str(pipeline_name),
            "numericizer": numericizer_desc,
            "regime": regime_desc,
            "symbolic_family": symbolic_family_desc,
        },
    )


def signature_from_state(state: Any) -> TrainingSignature:
    metadata = _mapping_copy(getattr(state, "metadata", None))
    embedded = metadata.get("training_signature")
    if isinstance(embedded, Mapping):
        return coerce_training_signature(embedded)

    return TrainingSignature(
        trainer_family=str(getattr(state, "trainer_name", type(state).__name__)),
        data_protocol=None if metadata.get("data_protocol") is None else str(metadata.get("data_protocol")),
        schema_signature=None if getattr(state, "schema_signature", None) is None else str(state.schema_signature),
        feature_signature=None if getattr(state, "feature_signature", None) is None else str(state.feature_signature),
        target_signature=None if getattr(state, "target_signature", None) is None else str(state.target_signature),
        objective_signature=None
        if getattr(state, "objective_signature", None) is None
        else str(state.objective_signature),
        pipeline_signature=None
        if getattr(state, "pipeline_signature", None) is None
        else str(state.pipeline_signature),
        numericizer_signature=None
        if getattr(state, "numericizer_signature", None) is None
        else str(state.numericizer_signature),
        regime_signature=None if getattr(state, "regime_signature", None) is None else str(state.regime_signature),
        symbolic_family_signature=None
        if getattr(state, "symbolic_family_signature", None) is None
        else str(state.symbolic_family_signature),
        metadata=metadata,
    )


def attach_signature_to_artifact(artifact: Any, signature: TrainingSignature) -> None:
    metadata = getattr(artifact, "metadata", None)
    if metadata is None:
        try:
            setattr(artifact, "metadata", {})
            metadata = getattr(artifact, "metadata", None)
        except Exception:
            return
    if not isinstance(metadata, dict):
        try:
            metadata = dict(metadata)
            setattr(artifact, "metadata", metadata)
        except Exception:
            return

    metadata["training_signature"] = signature.as_dict()
    metadata.setdefault("trainer_family", signature.trainer_family)
    metadata.setdefault("schema_signature", signature.schema_signature)
    metadata.setdefault("objective_signature", signature.objective_signature)
    metadata.setdefault("symbolic_family_signature", signature.symbolic_family_signature)
    symbolic_family_desc = dict(signature.metadata).get("symbolic_family")
    if symbolic_family_desc is not None:
        metadata.setdefault("symbolic_family", symbolic_family_desc)


__all__ = [
    "TrainingSignature",
    "attach_signature_to_artifact",
    "build_task_signature",
    "coerce_training_signature",
    "signature_from_artifact",
    "signature_from_state",
]
