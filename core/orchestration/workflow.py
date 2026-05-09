from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Protocol, Sequence, TypeAlias, runtime_checkable

import numpy as np

from config import (
    ExecutionSpec,
    FlowAssemblySpec,
    TrainerAssemblySpec,
    build_flow_components,
    build_trainer,
    coerce_execution_spec,
    validate_flow_assembly,
)
from core.common.contracts import ProcessedDataset, SampleDataset, SurrogateArtifact
from core.execution import (
    ExecutionRuntime,
    ExecutionResourceRequest,
    ExecutionTask,
    assert_phase_resource_budget,
    coerce_execution_resource_request,
    discover_execution_devices,
    normalize_execution_device_token,
    sum_execution_resource_requests,
)
from core.execution.resources import detect_local_execution_offer, resolve_phase_worker_count
from core.state import (
    ARTIFACT_REF,
    BUNDLE_REF,
    EVAL_SPLITS,
    FLOW_SPEC_REF,
    METRICS_REF,
    MODEL_PROCESSED_REF,
    MODEL_SPEC_REF,
    NUMERICIZER_REF,
    PROCESSED_REF,
    REPORT_REF,
    RESULT_REF,
    RUN_FINISHED_AT,
    RUN_NAME,
    RUN_STAGE,
    RUN_STARTED_AT,
    TRAINER_REF,
    TRAINER_STATE_REF,
    create_context_store,
    create_snapshot_store,
)
from .control_plane_contract import describe_control_plane_contract
from .capabilities import FlowCapability
from .lifecycle_runtime import LifecycleRuntime
from numericizer import BaseNumericizer, DefaultNumericizer
from training import (
    TrainTask,
    TrainingInit,
    coerce_trainer_capabilities,
    describe_inner_runtime_event_table,
)


@dataclass(frozen=True)
class TrainDataBundle:
    """Data container returned by the reading layer."""

    train: ProcessedDataset | SampleDataset
    valid: ProcessedDataset | SampleDataset | None = None
    test: ProcessedDataset | SampleDataset | None = None
    metadata: Mapping[str, Any] | None = None


@runtime_checkable
class BaseDataReader(Protocol):
    """Reader protocol: implement read() to provide training data."""

    def read(self) -> TrainDataBundle:
        ...


@dataclass
class MemoryDataReader:
    """In-memory reader for already-built datasets."""

    bundle: TrainDataBundle

    def read(self) -> TrainDataBundle:
        return self.bundle


TrainFlowInput: TypeAlias = TrainDataBundle | BaseDataReader | ProcessedDataset | SampleDataset


@dataclass(frozen=True)
class TrainFlowSpec:
    """Assembly + flow runtime options."""

    assembly: TrainerAssemblySpec
    execution: ExecutionSpec | Mapping[str, Any] | None = None
    training_init: TrainingInit | Mapping[str, Any] | None = None
    eval_splits: Sequence[str] = ("train", "valid", "test")
    output_dir: str | None = None
    save_artifact: bool = True
    save_report: bool = True
    save_checkpoint: bool = False
    checkpoint_dir: str | None = None
    replay_from_checkpoint: str | None = None
    model_spec: "ModelSpec | None" = None
    capabilities: Sequence[FlowCapability] = field(default_factory=tuple)
    capability_strict: bool = False
    run_name: str = "train_flow"
    context_store: Any | None = None
    snapshot_store: Any | None = None


@dataclass(frozen=True)
class SemanticTrainFlowSpec:
    """Semantic-complete assembly spec (numericizer + trainer + runtime)."""

    assembly: FlowAssemblySpec = field(default_factory=FlowAssemblySpec)
    execution: ExecutionSpec | Mapping[str, Any] | None = None
    training_init: TrainingInit | Mapping[str, Any] | None = None
    eval_splits: Sequence[str] = ("train", "valid", "test")
    output_dir: str | None = None
    save_artifact: bool = True
    save_report: bool = True
    save_checkpoint: bool = False
    checkpoint_dir: str | None = None
    replay_from_checkpoint: str | None = None
    model_spec: "ModelSpec | None" = None
    capabilities: Sequence[FlowCapability] = field(default_factory=tuple)
    capability_strict: bool = False
    run_name: str = "semantic_train_flow"
    context_store: Any | None = None
    snapshot_store: Any | None = None
    portfolio_parallel_mode: str = "serial"  # serial | thread | process
    portfolio_max_workers: int | None = None
    portfolio_fail_fast: bool = True
    portfolio_gpu_strategy: str = "none"  # none | fixed | round_robin | auto
    portfolio_gpu_devices: Sequence[str | int] = field(default_factory=tuple)


@dataclass
class TrainFlowResult:
    artifact: SurrogateArtifact
    processed: ProcessedDataset
    metrics: Dict[str, Dict[str, float]]
    report: Dict[str, Any]
    trainer_state: Any | None = None
    output_dir: str | None = None


@dataclass
class TrainPortfolioResult:
    """Collection result for multi-ModelSpec portfolio runs."""

    runs: Dict[str, TrainFlowResult]
    summary: Dict[str, Any]
    output_dir: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    """Explicit model-space selection for features and targets."""

    model_id: str = "default_model"
    feature_names: Sequence[str] | None = None
    target_names: Sequence[str] | None = None
    feature_indices: Sequence[int] | None = None
    target_indices: Sequence[int] | None = None
    strict: bool = True
    metadata: Mapping[str, Any] | None = None


def _execution_spec_as_mapping(execution: ExecutionSpec) -> Dict[str, Any]:
    return dict(execution.to_dict())


def _resolve_train_flow_execution(spec: TrainFlowSpec) -> ExecutionSpec:
    return coerce_execution_spec(spec.execution)


def _resolve_semantic_execution(spec: SemanticTrainFlowSpec) -> ExecutionSpec:
    return coerce_execution_spec(
        spec.execution,
        fallback_backend=str(spec.portfolio_parallel_mode),
        fallback_max_workers=spec.portfolio_max_workers,
        fallback_fail_fast=bool(spec.portfolio_fail_fast),
        fallback_gpu_strategy=str(spec.portfolio_gpu_strategy),
        fallback_gpu_devices=tuple(spec.portfolio_gpu_devices),
    )


def _clone_trainer_assembly_with_params(
    assembly: TrainerAssemblySpec,
    *,
    trainer_params: Mapping[str, Any],
) -> TrainerAssemblySpec:
    return TrainerAssemblySpec(
        trainer_key=str(assembly.trainer_key),
        trainer_params=dict(trainer_params),
        pipeline_key=str(assembly.pipeline_key),
        pipeline_params=dict(assembly.pipeline_params),
        biases=tuple(assembly.biases),
    )


def _apply_declared_execution_default_device(
    *,
    assembly: TrainerAssemblySpec,
    default_device: str | int | None,
) -> tuple[TrainerAssemblySpec, str | None]:
    trainer_key = str(assembly.trainer_key).strip().lower()
    trainer_params = dict(assembly.trainer_params)
    if default_device is None:
        return _clone_trainer_assembly_with_params(assembly, trainer_params=trainer_params), None

    requested = str(default_device).strip().lower()
    applied_device: str | None = None

    if trainer_key in _TORCH_DEVICE_TRAINER_KEYS:
        current = str(trainer_params.get("device", "auto")).strip().lower()
        if current in {"", "auto"}:
            trainer_params["device"] = str(default_device)
            applied_device = str(default_device)
        else:
            applied_device = current
    elif trainer_key in _STAGEWISE_INNER_OPT_TRAINER_KEYS:
        current = str(trainer_params.get("search_inner_opt_device", "auto")).strip().lower()
        if current in {"", "auto"}:
            trainer_params["search_inner_opt_device"] = str(default_device)
            applied_device = str(default_device)
        else:
            applied_device = current
    else:
        applied_device = None if not requested else str(default_device)

    return _clone_trainer_assembly_with_params(assembly, trainer_params=trainer_params), applied_device


def _merge_metadata(*parts: Mapping[str, Any] | None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in parts:
        if item:
            out.update(dict(item))
    return out


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]

    return str(value)


def _is_finite_matrix(arr: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(np.asarray(arr, dtype=float))))


def _evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)

    if yt.shape != yp.shape:
        raise ValueError(f"prediction shape mismatch: y_true={yt.shape}, y_pred={yp.shape}")

    err = yp - yt
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    yt_flat = yt.reshape(-1)
    yp_flat = yp.reshape(-1)
    ss_tot = float(np.sum((yt_flat - np.mean(yt_flat)) ** 2))
    ss_res = float(np.sum((yp_flat - yt_flat) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def _normalize_eval_splits(splits: Sequence[str]) -> tuple[str, ...]:
    allowed = {"train", "valid", "test"}
    out: list[str] = []

    for item in splits:
        key = str(item).strip().lower()
        if key not in allowed:
            raise ValueError(f"Unsupported eval split '{item}'. Allowed: {sorted(allowed)}")
        if key not in out:
            out.append(key)

    return tuple(out)


def _encode_sampleset(
    dataset: SampleDataset,
    *,
    numericizer: BaseNumericizer,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    samples = list(dataset.samples)
    if not samples:
        raise ValueError(f"SampleDataset for split '{split_name}' is empty")

    X = np.asarray(numericizer.transform_features(samples), dtype=float)
    Y = np.asarray(numericizer.transform_targets(samples), dtype=float)

    if X.ndim != 2:
        raise ValueError(f"Encoded X for split '{split_name}' must be 2D")
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y.ndim != 2:
        raise ValueError(f"Encoded y for split '{split_name}' must be 2D")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Row mismatch in split '{split_name}': X={X.shape}, y={Y.shape}")

    return X, Y


def _to_processed_bundle(
    bundle: TrainDataBundle,
    *,
    numericizer: BaseNumericizer | None = None,
) -> tuple[ProcessedDataset, BaseNumericizer | None]:
    train = bundle.train

    if isinstance(train, ProcessedDataset):
        if bundle.valid is not None and not isinstance(bundle.valid, ProcessedDataset):
            raise TypeError("When train is ProcessedDataset, valid must be ProcessedDataset or None")
        if bundle.test is not None and not isinstance(bundle.test, ProcessedDataset):
            raise TypeError("When train is ProcessedDataset, test must be ProcessedDataset or None")

        X_valid = train.X_valid
        y_valid = train.y_valid
        X_test = train.X_test
        y_test = train.y_test

        if isinstance(bundle.valid, ProcessedDataset):
            X_valid = bundle.valid.X_train
            y_valid = bundle.valid.y_train

        if isinstance(bundle.test, ProcessedDataset):
            X_test = bundle.test.X_train
            y_test = bundle.test.y_train

        processed = ProcessedDataset(
            X_train=np.asarray(train.X_train, dtype=float),
            y_train=np.asarray(train.y_train, dtype=float),
            X_valid=None if X_valid is None else np.asarray(X_valid, dtype=float),
            y_valid=None if y_valid is None else np.asarray(y_valid, dtype=float),
            X_test=None if X_test is None else np.asarray(X_test, dtype=float),
            y_test=None if y_test is None else np.asarray(y_test, dtype=float),
            feature_names=train.feature_names,
            target_names=train.target_names,
            metadata=_merge_metadata(
                train.metadata,
                bundle.metadata,
                {
                    "flow_input_protocol": "processed_dataset",
                },
            ),
        )
        return processed, None

    if not isinstance(train, SampleDataset):
        raise TypeError("bundle.train must be ProcessedDataset or SampleDataset")

    if bundle.valid is not None and not isinstance(bundle.valid, SampleDataset):
        raise TypeError("When train is SampleDataset, valid must be SampleDataset or None")
    if bundle.test is not None and not isinstance(bundle.test, SampleDataset):
        raise TypeError("When train is SampleDataset, test must be SampleDataset or None")

    encoder = numericizer or DefaultNumericizer()
    encoder.fit(train)

    X_train, y_train = _encode_sampleset(train, numericizer=encoder, split_name="train")

    X_valid: np.ndarray | None = None
    y_valid: np.ndarray | None = None
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None

    if isinstance(bundle.valid, SampleDataset):
        X_valid, y_valid = _encode_sampleset(bundle.valid, numericizer=encoder, split_name="valid")

    if isinstance(bundle.test, SampleDataset):
        X_test, y_test = _encode_sampleset(bundle.test, numericizer=encoder, split_name="test")

    plan = getattr(encoder, "plan", None)
    feature_names = getattr(plan, "feature_names", None)
    target_names = getattr(plan, "target_names", None)

    processed = ProcessedDataset(
        X_train=np.asarray(X_train, dtype=float),
        y_train=np.asarray(y_train, dtype=float),
        X_valid=None if X_valid is None else np.asarray(X_valid, dtype=float),
        y_valid=None if y_valid is None else np.asarray(y_valid, dtype=float),
        X_test=None if X_test is None else np.asarray(X_test, dtype=float),
        y_test=None if y_test is None else np.asarray(y_test, dtype=float),
        feature_names=feature_names,
        target_names=target_names,
        metadata=_merge_metadata(
            bundle.metadata,
            {
                "flow_input_protocol": "sample_dataset",
                "numericizer": str(getattr(encoder, "name", type(encoder).__name__)),
                "encoding_plan": None if plan is None else _jsonable(getattr(plan, "to_metadata")()),
            },
        ),
    )

    return processed, encoder


def _resolve_column_indices(
    *,
    dim: int,
    names: Sequence[str] | None,
    selected_names: Sequence[str] | None,
    selected_indices: Sequence[int] | None,
    axis_label: str,
    strict: bool,
) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()

    if selected_names is not None:
        if names is None:
            raise ValueError(f"Cannot select {axis_label}_names because names are unavailable in dataset")
        idx_map = {str(name): i for i, name in enumerate(tuple(names))}
        for raw in tuple(selected_names):
            key = str(raw)
            idx = idx_map.get(key)
            if idx is None:
                if strict:
                    raise ValueError(f"Unknown {axis_label} name in model_spec: '{key}'")
                continue
            if idx not in seen:
                order.append(int(idx))
                seen.add(int(idx))

    if selected_indices is not None:
        for raw in tuple(selected_indices):
            idx = int(raw)
            if idx < 0 or idx >= int(dim):
                if strict:
                    raise ValueError(
                        f"{axis_label} index out of range in model_spec: {idx} (dim={int(dim)})"
                    )
                continue
            if idx not in seen:
                order.append(int(idx))
                seen.add(int(idx))

    if not order:
        order = list(range(int(dim)))
    return order


def _slice_matrix_columns(arr: np.ndarray | None, cols: Sequence[int]) -> np.ndarray | None:
    if arr is None:
        return None
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    return np.asarray(out[:, list(cols)], dtype=float)


def _apply_model_spec_to_processed(processed: ProcessedDataset, model_spec: ModelSpec) -> ProcessedDataset:
    X_train = np.asarray(processed.X_train, dtype=float)
    y_train = np.asarray(processed.y_train, dtype=float)

    if X_train.ndim != 2:
        raise ValueError("ProcessedDataset.X_train must be 2D")
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_train.ndim != 2:
        raise ValueError("ProcessedDataset.y_train must be 1D/2D")

    feature_idx = _resolve_column_indices(
        dim=int(X_train.shape[1]),
        names=processed.feature_names,
        selected_names=model_spec.feature_names,
        selected_indices=model_spec.feature_indices,
        axis_label="feature",
        strict=bool(model_spec.strict),
    )
    target_idx = _resolve_column_indices(
        dim=int(y_train.shape[1]),
        names=processed.target_names,
        selected_names=model_spec.target_names,
        selected_indices=model_spec.target_indices,
        axis_label="target",
        strict=bool(model_spec.strict),
    )

    if not feature_idx:
        raise ValueError("model_spec resolved empty feature set")
    if not target_idx:
        raise ValueError("model_spec resolved empty target set")

    selected_feature_names: Sequence[str] | None = None
    if processed.feature_names is not None:
        src = tuple(str(x) for x in processed.feature_names)
        selected_feature_names = tuple(src[i] for i in feature_idx)

    selected_target_names: Sequence[str] | None = None
    if processed.target_names is not None:
        src = tuple(str(x) for x in processed.target_names)
        selected_target_names = tuple(src[i] for i in target_idx)

    model_meta = {
        "model_id": str(model_spec.model_id),
        "feature_indices": tuple(int(i) for i in feature_idx),
        "target_indices": tuple(int(i) for i in target_idx),
        "strict": bool(model_spec.strict),
        "feature_names_selected": None if selected_feature_names is None else tuple(selected_feature_names),
        "target_names_selected": None if selected_target_names is None else tuple(selected_target_names),
        "metadata": None if model_spec.metadata is None else dict(model_spec.metadata),
    }

    return ProcessedDataset(
        X_train=_slice_matrix_columns(processed.X_train, feature_idx),
        y_train=_slice_matrix_columns(processed.y_train, target_idx),
        X_valid=_slice_matrix_columns(processed.X_valid, feature_idx),
        y_valid=_slice_matrix_columns(processed.y_valid, target_idx),
        X_test=_slice_matrix_columns(processed.X_test, feature_idx),
        y_test=_slice_matrix_columns(processed.y_test, target_idx),
        feature_names=selected_feature_names,
        target_names=selected_target_names,
        metadata=_merge_metadata(processed.metadata, {"model_spec": model_meta}),
    )


def _collect_eval_pairs(processed: ProcessedDataset) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    out: Dict[str, tuple[np.ndarray, np.ndarray]] = {
        "train": (np.asarray(processed.X_train, dtype=float), np.asarray(processed.y_train, dtype=float)),
    }

    if processed.X_valid is not None and processed.y_valid is not None:
        out["valid"] = (np.asarray(processed.X_valid, dtype=float), np.asarray(processed.y_valid, dtype=float))

    if processed.X_test is not None and processed.y_test is not None:
        out["test"] = (np.asarray(processed.X_test, dtype=float), np.asarray(processed.y_test, dtype=float))

    return out


def _coerce_training_init(
    value: TrainingInit | Mapping[str, Any] | None,
    *,
    run_name: str,
    flow_kind: str,
) -> TrainingInit:
    if value is None:
        base = TrainingInit()
    elif isinstance(value, TrainingInit):
        base = value
    elif isinstance(value, Mapping):
        raw = dict(value)
        raw_metadata = raw.get("metadata")
        base = TrainingInit(
            mode=raw.get("mode", "fresh"),
            parent_artifact=raw.get("parent_artifact"),
            parent_state=raw.get("parent_state"),
            inner_runtime_hooks=tuple(raw.get("inner_runtime_hooks", ()) or ()),
            metadata={} if raw_metadata is None else dict(raw_metadata),
        )
    else:
        raise TypeError("training_init must be None, TrainingInit, or mapping-like payload")

    metadata = dict(base.metadata)
    metadata.setdefault("run_name", str(run_name))
    metadata.setdefault("flow_kind", str(flow_kind))
    return TrainingInit(
        mode=base.mode,
        parent_artifact=base.parent_artifact,
        parent_state=base.parent_state,
        inner_runtime_hooks=tuple(base.inner_runtime_hooks),
        metadata=metadata,
    )


def _training_init_summary(training_init: TrainingInit | None) -> Dict[str, Any]:
    init_eff = training_init or TrainingInit()
    parent_artifact = getattr(init_eff, "parent_artifact", None)
    parent_state = getattr(init_eff, "parent_state", None)
    inner_runtime_hooks = tuple(getattr(init_eff, "inner_runtime_hooks", ()) or ())
    metadata = dict(getattr(init_eff, "metadata", {}) or {})
    parent_state_source = None
    parent_state_path = None
    parent_state_snapshot_ref = None
    parent_state_context_key = None
    if metadata.get("parent_state_snapshot_ref") is not None:
        parent_state_source = "snapshot_ref"
        parent_state_snapshot_ref = str(metadata.get("parent_state_snapshot_ref"))
    if metadata.get("parent_state_context_key") is not None:
        parent_state_source = "context_key"
        parent_state_context_key = str(metadata.get("parent_state_context_key"))
    if metadata.get("parent_state_path") is not None:
        parent_state_source = "path"
        parent_state_path = str(metadata.get("parent_state_path"))
    if parent_state_source is None and isinstance(parent_state, Mapping):
        raw_parent_state = dict(parent_state)
        snapshot_ref = raw_parent_state.get("snapshot_ref", raw_parent_state.get("snapshot_id"))
        context_key = raw_parent_state.get("context_key")
        if snapshot_ref is not None:
            parent_state_source = "snapshot_ref"
            parent_state_snapshot_ref = str(snapshot_ref)
        elif context_key is not None:
            parent_state_source = "context_key"
            parent_state_context_key = str(context_key)
    elif parent_state_source is None and isinstance(parent_state, (str, os.PathLike)):
        raw_text = str(parent_state)
        if raw_text.startswith("snapshot://"):
            parent_state_source = "snapshot_ref"
            parent_state_snapshot_ref = raw_text[len("snapshot://") :]
        elif raw_text.startswith("context://"):
            parent_state_source = "context_key"
            parent_state_context_key = raw_text[len("context://") :]
        else:
            parent_state_source = "path"
            parent_state_path = raw_text
    elif parent_state_source is None and parent_state is not None:
        parent_state_source = "object"
    return {
        "mode": str(init_eff.mode),
        "has_parent_artifact": bool(parent_artifact is not None),
        "parent_artifact_id": (
            None if parent_artifact is None else str(getattr(parent_artifact, "artifact_id", type(parent_artifact).__name__))
        ),
        "has_parent_state": bool(parent_state is not None),
        "parent_state_source": parent_state_source,
        "parent_state_path": parent_state_path,
        "parent_state_snapshot_ref": parent_state_snapshot_ref,
        "parent_state_context_key": parent_state_context_key,
        "parent_state_trainer": (
            None
            if parent_state is None or isinstance(parent_state, (str, os.PathLike, Mapping))
            else str(getattr(parent_state, "trainer_name", type(parent_state).__name__))
        ),
        "inner_runtime_hook_count": int(len(inner_runtime_hooks)),
        "inner_runtime_hook_types": tuple(type(hook).__name__ for hook in inner_runtime_hooks),
        "metadata": _jsonable(metadata),
    }


def _read_snapshot_payload(snapshot_store: Any, snapshot_ref: str) -> Any:
    read_fn = getattr(snapshot_store, "read", None)
    if not callable(read_fn):
        raise TypeError("snapshot_store must provide read(snapshot_id)")
    return read_fn(str(snapshot_ref))


def _ensure_trainer_state_like(payload: Any, *, source: str) -> Any:
    if payload is None:
        raise TypeError(f"{source} resolved to None instead of TrainerState-like payload")
    if not hasattr(payload, "trainer_name") or not hasattr(payload, "payload"):
        raise TypeError(
            f"{source} did not resolve to TrainerState-like payload; got {type(payload).__name__}"
        )
    return payload


def _resolve_parent_state_locator(
    parent_state: Any,
    *,
    context_store: Any,
    snapshot_store: Any,
) -> tuple[Any, Dict[str, Any]] | None:
    if isinstance(parent_state, Mapping):
        raw_parent_state = dict(parent_state)
        snapshot_ref = raw_parent_state.get("snapshot_ref", raw_parent_state.get("snapshot_id"))
        context_key = raw_parent_state.get("context_key")
        if snapshot_ref is not None:
            loaded_state = _read_snapshot_payload(snapshot_store, str(snapshot_ref))
            return _ensure_trainer_state_like(
                loaded_state,
                source=f"snapshot_ref '{snapshot_ref}'",
            ), {
                "parent_state_snapshot_ref": str(snapshot_ref),
            }
        if context_key is not None:
            get_fn = getattr(context_store, "get", None)
            if not callable(get_fn):
                raise TypeError("context_store must provide get(key, default=None)")
            resolved_snapshot_ref = get_fn(str(context_key))
            if resolved_snapshot_ref is None:
                raise KeyError(f"context key '{context_key}' does not contain a snapshot ref")
            loaded_state = _read_snapshot_payload(snapshot_store, str(resolved_snapshot_ref))
            return _ensure_trainer_state_like(
                loaded_state,
                source=f"context_key '{context_key}' -> snapshot_ref '{resolved_snapshot_ref}'",
            ), {
                "parent_state_context_key": str(context_key),
                "parent_state_snapshot_ref": str(resolved_snapshot_ref),
            }
        return None

    if isinstance(parent_state, (str, os.PathLike)):
        raw_text = str(parent_state)
        if raw_text.startswith("snapshot://"):
            snapshot_ref = raw_text[len("snapshot://") :]
            loaded_state = _read_snapshot_payload(snapshot_store, snapshot_ref)
            return _ensure_trainer_state_like(
                loaded_state,
                source=f"snapshot_ref '{snapshot_ref}'",
            ), {
                "parent_state_snapshot_ref": str(snapshot_ref),
            }
        if raw_text.startswith("context://"):
            context_key = raw_text[len("context://") :]
            get_fn = getattr(context_store, "get", None)
            if not callable(get_fn):
                raise TypeError("context_store must provide get(key, default=None)")
            resolved_snapshot_ref = get_fn(str(context_key))
            if resolved_snapshot_ref is None:
                raise KeyError(f"context key '{context_key}' does not contain a snapshot ref")
            loaded_state = _read_snapshot_payload(snapshot_store, str(resolved_snapshot_ref))
            return _ensure_trainer_state_like(
                loaded_state,
                source=f"context_key '{context_key}' -> snapshot_ref '{resolved_snapshot_ref}'",
            ), {
                "parent_state_context_key": str(context_key),
                "parent_state_snapshot_ref": str(resolved_snapshot_ref),
            }
    return None


def _resolve_training_init_runtime(
    training_init: TrainingInit,
    *,
    trainer: Any,
    context_store: Any,
    snapshot_store: Any,
) -> TrainingInit:
    parent_state = training_init.parent_state
    resolved_locator = _resolve_parent_state_locator(
        parent_state,
        context_store=context_store,
        snapshot_store=snapshot_store,
    )
    if resolved_locator is not None:
        loaded_state, locator_metadata = resolved_locator
        metadata = dict(training_init.metadata)
        metadata.update({k: v for k, v in locator_metadata.items() if v is not None})
        return TrainingInit(
            mode=training_init.mode,
            parent_artifact=training_init.parent_artifact,
            parent_state=loaded_state,
            inner_runtime_hooks=training_init.inner_runtime_hooks,
            metadata=metadata,
        )

    if isinstance(parent_state, (str, os.PathLike)):
        load_fn = getattr(trainer, "load_trainer_state", None)
        if not callable(load_fn):
            raise TypeError(
                f"trainer '{getattr(trainer, 'name', type(trainer).__name__)}' does not support parent_state path loading"
            )
        loaded_state = load_fn(parent_state)
        metadata = dict(training_init.metadata)
        metadata.setdefault("parent_state_path", str(parent_state))
        return TrainingInit(
            mode=training_init.mode,
            parent_artifact=training_init.parent_artifact,
            parent_state=loaded_state,
            inner_runtime_hooks=training_init.inner_runtime_hooks,
            metadata=metadata,
        )
    return training_init


def _build_report(
    *,
    spec: TrainFlowSpec,
    execution: ExecutionSpec,
    effective_assembly: TrainerAssemblySpec | None = None,
    processed: ProcessedDataset,
    metrics: Mapping[str, Mapping[str, float]],
    artifact: SurrogateArtifact,
    trainer_name: str,
    fit_report: Mapping[str, Any] | None = None,
    fit_lineage: Mapping[str, Any] | None = None,
    trainer_capabilities: Mapping[str, Any] | None = None,
    trainer_state_available: bool = False,
    requested_training_init: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    train_X = np.asarray(processed.X_train, dtype=float)
    train_y = np.asarray(processed.y_train, dtype=float)
    assembly_payload = asdict(effective_assembly if effective_assembly is not None else spec.assembly)
    artifact_metadata = dict(getattr(artifact, "metadata", {}) or {})
    artifact_block: Dict[str, Any] = {
        "artifact_id": str(getattr(artifact, "artifact_id", "unknown")),
        "metadata": _jsonable(artifact_metadata),
    }
    symbolic_schema = artifact_metadata.get("symbolic_artifact_schema")
    if isinstance(symbolic_schema, Mapping):
        artifact_block["symbolic_artifact_schema"] = _jsonable(dict(symbolic_schema))
        artifact_block["symbolic_complexity_metrics"] = _jsonable(
            dict(dict(symbolic_schema).get("complexity_metrics", {}))
        )
        artifact_block["symbolic_head_semantics"] = _jsonable(dict(dict(symbolic_schema).get("head_semantics", {})))

    fit_report_payload = dict(fit_report) if isinstance(fit_report, Mapping) else {}
    task_signature_payload = (
        dict(fit_report_payload.get("task_signature", {}))
        if isinstance(fit_report_payload.get("task_signature"), Mapping)
        else {}
    )
    task_signature_meta = (
        dict(task_signature_payload.get("metadata", {}))
        if isinstance(task_signature_payload.get("metadata"), Mapping)
        else {}
    )
    symbolic_family_payload = (
        dict(task_signature_meta.get("symbolic_family", {}))
        if isinstance(task_signature_meta.get("symbolic_family"), Mapping)
        else {}
    )
    compatibility_payload = (
        dict(fit_report_payload.get("compatibility", {}))
        if isinstance(fit_report_payload.get("compatibility"), Mapping)
        else {}
    )

    training_block: Dict[str, Any] = {
        "requested_init": _jsonable(requested_training_init or {}),
        "fit_report": _jsonable(fit_report_payload),
        "lineage": _jsonable(fit_lineage or {}),
        "trainer_capabilities": _jsonable(trainer_capabilities or {}),
        "trainer_state_available": bool(trainer_state_available),
        "inner_runtime_events": _jsonable(describe_inner_runtime_event_table()),
    }
    if task_signature_payload:
        training_block["task_signature"] = _jsonable(task_signature_payload)
    if symbolic_family_payload:
        training_block["symbolic_family"] = {
            "signature": task_signature_payload.get("symbolic_family_signature"),
            "search_mechanism_contracts": _jsonable(symbolic_family_payload.get("search_mechanism_contracts", ())),
            "search_family_signature_contracts": _jsonable(
                symbolic_family_payload.get("search_family_signature_contracts", ())
            ),
            "structure_contracts": _jsonable(symbolic_family_payload.get("structure_contracts", {})),
        }
    if compatibility_payload:
        training_block["compatibility"] = _jsonable(compatibility_payload)
        if compatibility_payload.get("signature_comparison") is not None:
            training_block["signature_comparison"] = _jsonable(compatibility_payload.get("signature_comparison"))
        if compatibility_payload.get("symbolic_family_signature_drift") is not None:
            training_block["symbolic_family_signature_drift"] = _jsonable(
                compatibility_payload.get("symbolic_family_signature_drift")
            )

    report = {
        "run_name": str(spec.run_name),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "assembly": _jsonable(assembly_payload),
        "execution": _jsonable(_execution_spec_as_mapping(execution)),
        "trainer_name": str(trainer_name),
        "metrics": _jsonable(metrics),
        "data": {
            "train": {
                "n_samples": int(train_X.shape[0]),
                "feature_dim": int(train_X.shape[1]) if train_X.ndim == 2 else -1,
                "target_dim": int(train_y.shape[1]) if train_y.ndim == 2 else 1,
                "finite_X": _is_finite_matrix(train_X),
                "finite_y": _is_finite_matrix(train_y),
            },
            "has_valid": bool(processed.X_valid is not None and processed.y_valid is not None),
            "has_test": bool(processed.X_test is not None and processed.y_test is not None),
            "feature_names": None if processed.feature_names is None else list(processed.feature_names),
            "target_names": None if processed.target_names is None else list(processed.target_names),
            "metadata": _jsonable(processed.metadata),
        },
        "artifact": artifact_block,
        "training": training_block,
    }
    return report


def _build_failure_report(
    *,
    spec: TrainFlowSpec,
    execution: ExecutionSpec,
    effective_assembly: TrainerAssemblySpec | None,
    processed: ProcessedDataset | None,
    trainer_name: str | None,
    trainer_capabilities: Mapping[str, Any] | None,
    requested_training_init: Mapping[str, Any] | None,
    failed_stage: Any,
    error: Exception,
) -> Dict[str, Any]:
    assembly_payload = asdict(effective_assembly if effective_assembly is not None else spec.assembly)
    training_block: Dict[str, Any] = {
        "requested_init": _jsonable(requested_training_init or {}),
        "trainer_capabilities": _jsonable(trainer_capabilities or {}),
        "trainer_state_available": False,
        "inner_runtime_events": _jsonable(describe_inner_runtime_event_table()),
    }
    error_block: Dict[str, Any] = {
        "type": str(type(error).__name__),
        "message": str(error),
        "failed_stage": None if failed_stage is None else str(failed_stage),
    }

    compatibility_verdict = getattr(error, "verdict", None)
    if compatibility_verdict is not None:
        training_block["compatibility"] = _jsonable(dict(getattr(compatibility_verdict, "metadata", {})))
        training_block["compatibility_reasons"] = [str(v) for v in tuple(getattr(compatibility_verdict, "reasons", ()))]
        training_block["compatibility_warnings"] = [str(v) for v in tuple(getattr(compatibility_verdict, "warnings", ()))]
        current_signature = dict(dict(getattr(compatibility_verdict, "metadata", {})).get("current_signature", {}))
        if current_signature:
            training_block["task_signature"] = _jsonable(current_signature)
            signature_meta = (
                dict(current_signature.get("metadata", {}))
                if isinstance(current_signature.get("metadata"), Mapping)
                else {}
            )
            symbolic_family_payload = (
                dict(signature_meta.get("symbolic_family", {}))
                if isinstance(signature_meta.get("symbolic_family"), Mapping)
                else {}
            )
            if symbolic_family_payload:
                training_block["symbolic_family"] = {
                    "signature": current_signature.get("symbolic_family_signature"),
                    "search_mechanism_contracts": _jsonable(
                        symbolic_family_payload.get("search_mechanism_contracts", ())
                    ),
                    "search_family_signature_contracts": _jsonable(
                        symbolic_family_payload.get("search_family_signature_contracts", ())
                    ),
                    "structure_contracts": _jsonable(symbolic_family_payload.get("structure_contracts", {})),
                }
        if dict(getattr(compatibility_verdict, "metadata", {})).get("signature_comparison") is not None:
            training_block["signature_comparison"] = _jsonable(
                dict(getattr(compatibility_verdict, "metadata", {})).get("signature_comparison")
            )
        if dict(getattr(compatibility_verdict, "metadata", {})).get("symbolic_family_signature_drift") is not None:
            training_block["symbolic_family_signature_drift"] = _jsonable(
                dict(getattr(compatibility_verdict, "metadata", {})).get("symbolic_family_signature_drift")
            )
        error_block["compatibility"] = {
            "supported": bool(getattr(compatibility_verdict, "supported", False)),
            "reasons": [str(v) for v in tuple(getattr(compatibility_verdict, "reasons", ()))],
            "warnings": [str(v) for v in tuple(getattr(compatibility_verdict, "warnings", ()))],
        }

    if processed is None:
        data_block: Dict[str, Any] = {
            "train": None,
            "has_valid": False,
            "has_test": False,
            "feature_names": None,
            "target_names": None,
            "metadata": None,
        }
    else:
        train_X = np.asarray(processed.X_train, dtype=float)
        train_y = np.asarray(processed.y_train, dtype=float)
        data_block = {
            "train": {
                "n_samples": int(train_X.shape[0]),
                "feature_dim": int(train_X.shape[1]) if train_X.ndim == 2 else -1,
                "target_dim": int(train_y.shape[1]) if train_y.ndim == 2 else 1,
                "finite_X": _is_finite_matrix(train_X),
                "finite_y": _is_finite_matrix(train_y),
            },
            "has_valid": bool(processed.X_valid is not None and processed.y_valid is not None),
            "has_test": bool(processed.X_test is not None and processed.y_test is not None),
            "feature_names": None if processed.feature_names is None else list(processed.feature_names),
            "target_names": None if processed.target_names is None else list(processed.target_names),
            "metadata": _jsonable(processed.metadata),
        }

    return {
        "run_name": str(spec.run_name),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "assembly": _jsonable(assembly_payload),
        "execution": _jsonable(_execution_spec_as_mapping(execution)),
        "trainer_name": None if trainer_name is None else str(trainer_name),
        "data": data_block,
        "artifact": None,
        "training": training_block,
        "error": error_block,
    }


def _snapshot_count(snapshot_store: Any) -> int:
    fn = getattr(snapshot_store, "count", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            return 0

    keys_fn = getattr(snapshot_store, "keys", None)
    if callable(keys_fn):
        try:
            return int(len(tuple(keys_fn())))
        except Exception:
            return 0
    return 0


def _snapshot_describe(snapshot_store: Any) -> list[Dict[str, Any]]:
    fn = getattr(snapshot_store, "describe", None)
    if callable(fn):
        try:
            rows = list(fn())
            return [dict(x) for x in rows if isinstance(x, Mapping)]
        except Exception:
            return []
    return []


def _write_snapshot_ref(
    *,
    snapshot_store: Any,
    context_store: Any,
    context_key: str,
    payload: Any,
    kind: str,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    write_fn = getattr(snapshot_store, "write", None)
    if not callable(write_fn):
        raise TypeError("snapshot_store must provide write(payload, *, kind, metadata)")
    sid = str(write_fn(payload, kind=str(kind), metadata=dict(metadata or {})))
    context_store.set(str(context_key), sid)
    return sid


def _state_block(context_store: Any, snapshot_store: Any) -> Dict[str, Any]:
    return {
        "context_refs": _jsonable(context_store.to_dict()),
        "snapshot_count": int(_snapshot_count(snapshot_store)),
        "snapshots": _jsonable(_snapshot_describe(snapshot_store)),
    }


def _refresh_capability_state(
    capability_context: Dict[str, Any],
    context_store: Any,
    snapshot_store: Any,
) -> None:
    capability_context["context_refs"] = context_store.to_dict()
    capability_context["snapshot_count"] = int(_snapshot_count(snapshot_store))


def _resolve_context_store(spec_value: Any) -> Any:
    if spec_value is None:
        return create_context_store(backend="memory")
    if isinstance(spec_value, Mapping):
        cfg = dict(spec_value)
        backend = str(cfg.pop("backend", "memory"))
        return create_context_store(backend=backend, **cfg)
    required = ("set", "get", "to_dict")
    if all(callable(getattr(spec_value, x, None)) for x in required):
        return spec_value
    raise TypeError("context_store must be None, mapping config, or ContextStore-like object")


def _resolve_snapshot_store(spec_value: Any) -> Any:
    if spec_value is None:
        return create_snapshot_store(backend="memory")
    if isinstance(spec_value, Mapping):
        cfg = dict(spec_value)
        backend = str(cfg.pop("backend", "memory"))
        return create_snapshot_store(backend=backend, **cfg)
    required = ("write", "read", "has")
    if all(callable(getattr(spec_value, x, None)) for x in required):
        return spec_value
    raise TypeError("snapshot_store must be None, mapping config, or SnapshotStore-like object")


def _coerce_train_bundle(data: TrainFlowInput) -> TrainDataBundle:
    if isinstance(data, TrainDataBundle):
        return data
    if isinstance(data, (ProcessedDataset, SampleDataset)):
        return TrainDataBundle(train=data)
    if isinstance(data, BaseDataReader):
        bundle = data.read()
        if not isinstance(bundle, TrainDataBundle):
            raise TypeError(
                f"DataReader.read() must return TrainDataBundle, got {type(bundle).__name__}"
            )
        return bundle
    raise TypeError(
        "data must be one of: TrainDataBundle, BaseDataReader, ProcessedDataset, SampleDataset"
    )


def run_train_flow(
    data: TrainFlowInput,
    *,
    spec: TrainFlowSpec,
    numericizer: BaseNumericizer | None = None,
):
    """Standard orchestration: read -> encode -> train -> evaluate -> save."""

    if spec.replay_from_checkpoint:
        from core.orchestration.checkpoint import load_train_checkpoint

        return load_train_checkpoint(str(spec.replay_from_checkpoint))

    bundle = _coerce_train_bundle(data)
    requested_training_init = _coerce_training_init(
        spec.training_init,
        run_name=str(spec.run_name),
        flow_kind="train_flow",
    )
    resolved_execution = _resolve_train_flow_execution(spec)
    effective_assembly, declared_execution_device = _apply_declared_execution_default_device(
        assembly=spec.assembly,
        default_device=resolved_execution.default_device,
    )

    context_store = _resolve_context_store(spec.context_store)
    snapshot_store = _resolve_snapshot_store(spec.snapshot_store)

    context_store.set(RUN_NAME, str(spec.run_name))
    context_store.set(RUN_STAGE, "init")
    context_store.set(RUN_STARTED_AT, datetime.now(timezone.utc).isoformat())

    _write_snapshot_ref(
        snapshot_store=snapshot_store,
        context_store=context_store,
        context_key=FLOW_SPEC_REF,
        payload={
            "run_name": str(spec.run_name),
            "assembly": _jsonable(asdict(effective_assembly)),
            "execution": _execution_spec_as_mapping(resolved_execution),
            "eval_splits": tuple(str(x) for x in tuple(spec.eval_splits)),
            "output_dir": spec.output_dir,
            "save_artifact": bool(spec.save_artifact),
            "save_report": bool(spec.save_report),
            "save_checkpoint": bool(spec.save_checkpoint),
            "checkpoint_dir": spec.checkpoint_dir,
            "model_spec": (
                None
                if spec.model_spec is None
                else {
                    "model_id": str(spec.model_spec.model_id),
                    "feature_names": None
                    if spec.model_spec.feature_names is None
                    else tuple(str(x) for x in tuple(spec.model_spec.feature_names)),
                    "target_names": None
                    if spec.model_spec.target_names is None
                    else tuple(str(x) for x in tuple(spec.model_spec.target_names)),
                    "feature_indices": None
                    if spec.model_spec.feature_indices is None
                    else tuple(int(x) for x in tuple(spec.model_spec.feature_indices)),
                    "target_indices": None
                    if spec.model_spec.target_indices is None
                    else tuple(int(x) for x in tuple(spec.model_spec.target_indices)),
                    "strict": bool(spec.model_spec.strict),
                    "metadata": None if spec.model_spec.metadata is None else dict(spec.model_spec.metadata),
                }
            ),
            "capability_strict": bool(spec.capability_strict),
            "capabilities": tuple(str(getattr(c, "name", type(c).__name__)) for c in tuple(spec.capabilities)),
            "training_init": _training_init_summary(requested_training_init),
        },
        kind="flow_spec",
        metadata={"run_name": str(spec.run_name)},
    )
    _write_snapshot_ref(
        snapshot_store=snapshot_store,
        context_store=context_store,
        context_key=BUNDLE_REF,
        payload=bundle,
        kind="train_bundle",
        metadata={"stage": "input_bundle"},
    )

    lifecycle_runtime = LifecycleRuntime.create(
        strict=bool(spec.capability_strict),
        capabilities=tuple(spec.capabilities),
    )

    capability_context: Dict[str, Any] = {
        "run_name": str(spec.run_name),
        "flow_spec": spec,
        "requested_training_init": requested_training_init,
        "model_spec": spec.model_spec,
        "execution": resolved_execution.to_dict(),
        "declared_execution_device": declared_execution_device,
        "bundle": bundle,
        "context_store": context_store,
        "snapshot_store": snapshot_store,
    }
    _refresh_capability_state(capability_context, context_store, snapshot_store)
    processed: ProcessedDataset | None = None
    trainer: Any | None = None
    trainer_caps: Any | None = None
    training_init = requested_training_init
    try:
        lifecycle_runtime.dispatch("on_flow_start", capability_context)

        context_store.set(RUN_STAGE, "data_ready")
        processed_raw, fitted_numericizer = _to_processed_bundle(bundle, numericizer=numericizer)
        _write_snapshot_ref(
            snapshot_store=snapshot_store,
            context_store=context_store,
            context_key=PROCESSED_REF,
            payload=processed_raw,
            kind="processed_dataset",
            metadata={"stage": "data_ready_raw"},
        )
        if fitted_numericizer is not None:
            _write_snapshot_ref(
                snapshot_store=snapshot_store,
                context_store=context_store,
                context_key=NUMERICIZER_REF,
                payload=fitted_numericizer,
                kind="numericizer",
                metadata={"name": str(getattr(fitted_numericizer, "name", type(fitted_numericizer).__name__))},
            )

        processed = processed_raw
        if spec.model_spec is not None:
            _write_snapshot_ref(
                snapshot_store=snapshot_store,
                context_store=context_store,
                context_key=MODEL_SPEC_REF,
                payload={
                    "model_id": str(spec.model_spec.model_id),
                    "feature_names": (
                        None
                        if spec.model_spec.feature_names is None
                        else tuple(str(x) for x in tuple(spec.model_spec.feature_names))
                    ),
                    "target_names": (
                        None
                        if spec.model_spec.target_names is None
                        else tuple(str(x) for x in tuple(spec.model_spec.target_names))
                    ),
                    "feature_indices": (
                        None
                        if spec.model_spec.feature_indices is None
                        else tuple(int(x) for x in tuple(spec.model_spec.feature_indices))
                    ),
                    "target_indices": (
                        None
                        if spec.model_spec.target_indices is None
                        else tuple(int(x) for x in tuple(spec.model_spec.target_indices))
                    ),
                    "strict": bool(spec.model_spec.strict),
                    "metadata": None if spec.model_spec.metadata is None else dict(spec.model_spec.metadata),
                },
                kind="model_spec",
                metadata={"model_id": str(spec.model_spec.model_id)},
            )
            processed = _apply_model_spec_to_processed(processed_raw, spec.model_spec)
            _write_snapshot_ref(
                snapshot_store=snapshot_store,
                context_store=context_store,
                context_key=MODEL_PROCESSED_REF,
                payload=processed,
                kind="model_processed_dataset",
                metadata={"model_id": str(spec.model_spec.model_id)},
            )

        capability_context["processed_raw"] = processed_raw
        capability_context["processed"] = processed
        capability_context["numericizer"] = fitted_numericizer
        _refresh_capability_state(capability_context, context_store, snapshot_store)
        lifecycle_runtime.dispatch("on_data_ready", capability_context)

        context_store.set(RUN_STAGE, "fit")
        trainer = build_trainer(effective_assembly)
        _write_snapshot_ref(
            snapshot_store=snapshot_store,
            context_store=context_store,
            context_key=TRAINER_REF,
            payload=trainer,
            kind="trainer",
            metadata={"name": str(getattr(trainer, "name", type(trainer).__name__))},
        )
        training_init = _resolve_training_init_runtime(
            requested_training_init,
            trainer=trainer,
            context_store=context_store,
            snapshot_store=snapshot_store,
        )
        capability_context["trainer"] = trainer
        capability_context["requested_training_init"] = requested_training_init
        _refresh_capability_state(capability_context, context_store, snapshot_store)
        lifecycle_runtime.dispatch("on_pre_fit", capability_context)
        trainer_caps = coerce_trainer_capabilities(getattr(trainer, "capabilities", lambda: None)())
        train_task = TrainTask.from_data(
            processed,
            metadata={
                "run_name": str(spec.run_name),
                "trainer_name": str(getattr(trainer, "name", type(trainer).__name__)),
                "model_id": None if spec.model_spec is None else str(spec.model_spec.model_id),
            },
            task_id=f"{str(spec.run_name)}::fit",
        )
        capability_context["train_task"] = train_task
        capability_context["training_init"] = training_init
        capability_context["trainer_capabilities"] = trainer_caps.as_dict()
        _refresh_capability_state(capability_context, context_store, snapshot_store)
        fit_result = trainer.fit_task(train_task, training_init)
        artifact = fit_result.artifact
        _write_snapshot_ref(
            snapshot_store=snapshot_store,
            context_store=context_store,
            context_key=ARTIFACT_REF,
            payload=artifact,
            kind="artifact",
            metadata={"artifact_id": str(getattr(artifact, "artifact_id", "unknown"))},
        )
        capability_context["artifact"] = artifact
        capability_context["fit_result"] = fit_result
        if fit_result.trainer_state is not None:
            _write_snapshot_ref(
                snapshot_store=snapshot_store,
                context_store=context_store,
                context_key=TRAINER_STATE_REF,
                payload=fit_result.trainer_state,
                kind="trainer_state",
                metadata={"trainer_name": str(getattr(fit_result.trainer_state, "trainer_name", "unknown"))},
            )
            capability_context["trainer_state"] = fit_result.trainer_state
        _refresh_capability_state(capability_context, context_store, snapshot_store)
        lifecycle_runtime.dispatch("on_post_fit", capability_context)

        context_store.set(RUN_STAGE, "eval")
        eval_splits = _normalize_eval_splits(spec.eval_splits)
        context_store.set(EVAL_SPLITS, tuple(eval_splits))
        pairs = _collect_eval_pairs(processed)
        capability_context["eval_splits"] = tuple(eval_splits)
        _refresh_capability_state(capability_context, context_store, snapshot_store)
        lifecycle_runtime.dispatch("on_pre_eval", capability_context)

        metrics: Dict[str, Dict[str, float]] = {}
        for split in eval_splits:
            pair = pairs.get(split)
            if pair is None:
                continue
            X_split, y_split = pair
            pred = np.asarray(artifact.predict(X_split), dtype=float)
            metrics[split] = _evaluate_regression(y_split, pred)
        _write_snapshot_ref(
            snapshot_store=snapshot_store,
            context_store=context_store,
            context_key=METRICS_REF,
            payload=metrics,
            kind="metrics",
            metadata={"splits": tuple(metrics.keys())},
        )
        capability_context["metrics"] = metrics
        _refresh_capability_state(capability_context, context_store, snapshot_store)
        lifecycle_runtime.dispatch("on_post_eval", capability_context)

        context_store.set(RUN_STAGE, "report")
        trainer_name = str(getattr(trainer, "name", type(trainer).__name__))
        report = _build_report(
            spec=spec,
            execution=resolved_execution,
            effective_assembly=effective_assembly,
            processed=processed,
            metrics=metrics,
            artifact=artifact,
            trainer_name=trainer_name,
            fit_report=fit_result.report,
            fit_lineage=fit_result.lineage.as_dict(),
            trainer_capabilities=trainer_caps.as_dict(),
            trainer_state_available=fit_result.trainer_state is not None,
            requested_training_init=_training_init_summary(training_init),
        )
        if spec.model_spec is not None:
            report["model_spec"] = {
                "model_id": str(spec.model_spec.model_id),
                "feature_names": None
                if spec.model_spec.feature_names is None
                else [str(x) for x in tuple(spec.model_spec.feature_names)],
                "target_names": None
                if spec.model_spec.target_names is None
                else [str(x) for x in tuple(spec.model_spec.target_names)],
                "feature_indices": None
                if spec.model_spec.feature_indices is None
                else [int(x) for x in tuple(spec.model_spec.feature_indices)],
                "target_indices": None
                if spec.model_spec.target_indices is None
                else [int(x) for x in tuple(spec.model_spec.target_indices)],
                "strict": bool(spec.model_spec.strict),
                "metadata": None if spec.model_spec.metadata is None else dict(spec.model_spec.metadata),
            }
        control_plane_contract = describe_control_plane_contract(
            lifecycle_events=lifecycle_runtime.describe_event_table(),
        )
        report["capabilities"] = _jsonable(lifecycle_runtime.build_capability_report())
        report["lifecycle_events"] = _jsonable(control_plane_contract.get("lifecycle_events", ()))
        report["inner_runtime_events"] = _jsonable(control_plane_contract.get("inner_runtime_events", ()))
        report["control_plane_contract"] = _jsonable(control_plane_contract)
        report["state"] = _state_block(context_store, snapshot_store)
        capability_context["report"] = report

        if fitted_numericizer is not None:
            report["numericizer"] = {
                "name": str(getattr(fitted_numericizer, "name", type(fitted_numericizer).__name__)),
            }

        _write_snapshot_ref(
            snapshot_store=snapshot_store,
            context_store=context_store,
            context_key=REPORT_REF,
            payload=report,
            kind="report",
            metadata={"stage": "pre_persist"},
        )

        out_dir: str | None = None
        run_dir: Path | None = None
        context_store.set(RUN_STAGE, "persist")
        _refresh_capability_state(capability_context, context_store, snapshot_store)
        lifecycle_runtime.dispatch("on_pre_persist", capability_context)
        if spec.output_dir:
            run_dir = Path(spec.output_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            out_dir = str(run_dir)

            if bool(spec.save_artifact):
                artifact.save(str(run_dir / "artifact"))

            if bool(spec.save_report):
                (run_dir / "flow_report.json").write_text(
                    json.dumps(_jsonable(report), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        capability_context["output_dir"] = out_dir
        _refresh_capability_state(capability_context, context_store, snapshot_store)
        lifecycle_runtime.dispatch("on_post_persist", capability_context)

        result = TrainFlowResult(
            artifact=artifact,
            processed=processed,
            metrics=metrics,
            report=report,
            trainer_state=fit_result.trainer_state,
            output_dir=out_dir,
        )

        _write_snapshot_ref(
            snapshot_store=snapshot_store,
            context_store=context_store,
            context_key=RESULT_REF,
            payload={
                "output_dir": out_dir,
                "metrics": metrics,
                "artifact_id": str(getattr(artifact, "artifact_id", "unknown")),
            },
            kind="result",
            metadata={"stage": "result"},
        )

        if bool(spec.save_checkpoint):
            from core.orchestration.checkpoint import save_train_checkpoint

            cp_dir = spec.checkpoint_dir
            if cp_dir is None:
                if run_dir is not None:
                    cp_dir = str(run_dir / "checkpoint")
                else:
                    cp_dir = str(Path("runs") / f"{str(spec.run_name)}_checkpoint")

            saved_cp = save_train_checkpoint(
                checkpoint_dir=str(cp_dir),
                artifact=result.artifact,
                processed=result.processed,
                metrics=result.metrics,
                report=result.report,
                run_name=str(spec.run_name),
                output_dir=result.output_dir,
            )
            result.report["checkpoint"] = {
                "path": str(saved_cp),
                "replayable": True,
            }
            if run_dir is not None and bool(spec.save_report):
                (run_dir / "flow_report.json").write_text(
                    json.dumps(_jsonable(result.report), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        context_store.set(RUN_STAGE, "finished")
        context_store.set(RUN_FINISHED_AT, datetime.now(timezone.utc).isoformat())
        result.report["state"] = _state_block(context_store, snapshot_store)

        capability_context["result"] = result
        _refresh_capability_state(capability_context, context_store, snapshot_store)
        lifecycle_runtime.dispatch("on_flow_finish", capability_context)

        if run_dir is not None and bool(spec.save_report):
            (run_dir / "flow_report.json").write_text(
                json.dumps(_jsonable(result.report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return result
    except Exception as exc:
        failed_stage = context_store.get(RUN_STAGE)
        context_store.set(RUN_STAGE, "failed")
        context_store.set(RUN_FINISHED_AT, datetime.now(timezone.utc).isoformat())
        trainer_name = None if trainer is None else str(getattr(trainer, "name", type(trainer).__name__))
        failure_report = _build_failure_report(
            spec=spec,
            execution=resolved_execution,
            effective_assembly=effective_assembly,
            processed=processed,
            trainer_name=trainer_name,
            trainer_capabilities=None if trainer_caps is None else trainer_caps.as_dict(),
            requested_training_init=_training_init_summary(training_init),
            failed_stage=failed_stage,
            error=exc,
        )
        capability_context["report"] = failure_report
        if spec.output_dir and bool(spec.save_report):
            run_dir = Path(spec.output_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            capability_context["output_dir"] = str(run_dir)
            (run_dir / "flow_report.json").write_text(
                json.dumps(_jsonable(failure_report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        capability_context["error"] = exc
        capability_context["failed_stage"] = failed_stage
        _refresh_capability_state(capability_context, context_store, snapshot_store)
        try:
            lifecycle_runtime.dispatch("on_flow_error", exc, capability_context)
        except Exception:
            pass
        raise





def run_semantic_train_flow(
    data: TrainFlowInput,
    *,
    spec: SemanticTrainFlowSpec,
    config: Any | None = None,
) -> TrainFlowResult:
    """Run flow with semantic-complete assembly (numericizer + trainer)."""

    resolved_execution = _resolve_semantic_execution(spec)
    trainer_assembly_with_device, _ = _apply_declared_execution_default_device(
        assembly=spec.assembly.trainer,
        default_device=resolved_execution.default_device,
    )
    effective_assembly = FlowAssemblySpec(
        trainer=trainer_assembly_with_device,
        numericizer=spec.assembly.numericizer,
        capabilities=tuple(spec.assembly.capabilities),
    )

    validate_flow_assembly(effective_assembly)

    components = build_flow_components(effective_assembly, config=config)
    numericizer = components.get("numericizer")
    assembly_capabilities = tuple(components.get("capabilities", tuple()))
    runtime_capabilities = tuple(spec.capabilities)
    all_capabilities = tuple(assembly_capabilities) + tuple(runtime_capabilities)

    flow_spec = TrainFlowSpec(
        assembly=effective_assembly.trainer,
        execution=resolved_execution,
        training_init=spec.training_init,
        eval_splits=tuple(spec.eval_splits),
        output_dir=spec.output_dir,
        save_artifact=bool(spec.save_artifact),
        save_report=bool(spec.save_report),
        save_checkpoint=bool(spec.save_checkpoint),
        checkpoint_dir=spec.checkpoint_dir,
        replay_from_checkpoint=spec.replay_from_checkpoint,
        model_spec=spec.model_spec,
        capabilities=all_capabilities,
        capability_strict=bool(spec.capability_strict),
        run_name=str(spec.run_name),
        context_store=spec.context_store,
        snapshot_store=spec.snapshot_store,
    )

    result = run_train_flow(
        data,
        spec=flow_spec,
        numericizer=numericizer,
    )

    result.report["semantic_assembly"] = _jsonable(asdict(effective_assembly))
    result.report["execution"] = _jsonable(_execution_spec_as_mapping(resolved_execution))

    if result.output_dir and bool(spec.save_report):
        run_dir = Path(result.output_dir)
        (run_dir / "flow_report.json").write_text(
            json.dumps(_jsonable(result.report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return result


def _portfolio_selection_rmse(metrics: Mapping[str, Mapping[str, Any]]) -> float:
    for split in ("test", "valid", "train"):
        row = metrics.get(split)
        if not isinstance(row, Mapping):
            continue
        val = row.get("rmse")
        try:
            rmse = float(val)
        except Exception:
            continue
        if np.isfinite(rmse):
            return float(rmse)
    return float("inf")


def _clone_model_spec(item: ModelSpec) -> ModelSpec:
    return ModelSpec(
        model_id=str(item.model_id),
        feature_names=None if item.feature_names is None else tuple(str(x) for x in tuple(item.feature_names)),
        target_names=None if item.target_names is None else tuple(str(x) for x in tuple(item.target_names)),
        feature_indices=None if item.feature_indices is None else tuple(int(x) for x in tuple(item.feature_indices)),
        target_indices=None if item.target_indices is None else tuple(int(x) for x in tuple(item.target_indices)),
        strict=bool(item.strict),
        metadata=None if item.metadata is None else dict(item.metadata),
    )


_PORTFOLIO_PARALLEL_MODES: frozenset[str] = frozenset({"serial", "thread", "process"})
_PORTFOLIO_GPU_STRATEGIES: frozenset[str] = frozenset({"none", "fixed", "round_robin", "auto"})
_TORCH_DEVICE_TRAINER_KEYS: frozenset[str] = frozenset({"mlp_torch", "symbolic_torch", "symbolic_torch_interval"})
_STAGEWISE_INNER_OPT_TRAINER_KEYS: frozenset[str] = frozenset({"symbolic_stagewise"})


def _normalize_portfolio_parallel_mode(value: str) -> str:
    key = str(value or "serial").strip().lower()
    if key not in _PORTFOLIO_PARALLEL_MODES:
        allowed = ", ".join(sorted(_PORTFOLIO_PARALLEL_MODES))
        raise ValueError(f"Unsupported portfolio_parallel_mode '{value}'. Allowed: [{allowed}]")
    return key


def _normalize_portfolio_gpu_strategy(value: str) -> str:
    key = str(value or "none").strip().lower()
    if key not in _PORTFOLIO_GPU_STRATEGIES:
        allowed = ", ".join(sorted(_PORTFOLIO_GPU_STRATEGIES))
        raise ValueError(f"Unsupported portfolio_gpu_strategy '{value}'. Allowed: [{allowed}]")
    return key


def _normalize_cuda_device_token(raw: str | int) -> str:
    return normalize_execution_device_token(raw)


def _normalize_portfolio_gpu_devices(devices: Sequence[str | int]) -> tuple[str, ...]:
    out: list[str] = []
    for raw in tuple(devices):
        token = _normalize_cuda_device_token(raw)
        if token not in out:
            out.append(token)
    return tuple(out)


def _discover_cuda_devices() -> tuple[str, ...]:
    return discover_execution_devices("cuda")


def _select_portfolio_device(
    *,
    strategy: str,
    normalized_devices: Sequence[str],
    model_index: int,
) -> str | None:
    key = str(strategy).strip().lower()
    pool = tuple(str(x) for x in tuple(normalized_devices))

    if key == "none":
        return None

    if key == "fixed":
        fixed_pool = tuple(pool) if pool else _discover_cuda_devices()
        if not fixed_pool:
            return None
        return str(fixed_pool[0])

    if key == "round_robin":
        rr_pool = tuple(pool) if pool else _discover_cuda_devices()
        if not rr_pool:
            return None
        return str(rr_pool[int(model_index) % len(rr_pool)])

    if key == "auto":
        auto_pool = tuple(pool) if pool else _discover_cuda_devices()
        if not auto_pool:
            return None
        return str(auto_pool[int(model_index) % len(auto_pool)])

    raise ValueError(f"Unsupported GPU strategy: {strategy}")


def _clone_assembly_with_trainer_params(
    assembly: FlowAssemblySpec,
    *,
    trainer_params: Mapping[str, Any],
) -> FlowAssemblySpec:
    trainer_spec = _clone_trainer_assembly_with_params(assembly.trainer, trainer_params=trainer_params)
    return FlowAssemblySpec(
        trainer=trainer_spec,
        numericizer=assembly.numericizer,
        capabilities=tuple(assembly.capabilities),
    )


def _apply_portfolio_device_assignment(
    *,
    assembly: FlowAssemblySpec,
    assigned_device: str | None,
    gpu_strategy: str,
) -> tuple[FlowAssemblySpec, str | None]:
    trainer_key = str(assembly.trainer.trainer_key).strip().lower()
    trainer_params = dict(assembly.trainer.trainer_params)

    if assigned_device is None:
        return _clone_assembly_with_trainer_params(assembly, trainer_params=trainer_params), None

    strategy = str(gpu_strategy).strip().lower()
    applied_device: str | None = None

    if trainer_key in _TORCH_DEVICE_TRAINER_KEYS:
        current = str(trainer_params.get("device", "auto")).strip().lower()
        if strategy != "auto" or current in {"", "auto"}:
            trainer_params["device"] = str(assigned_device)
            applied_device = str(assigned_device)
        else:
            applied_device = current
    elif trainer_key in _STAGEWISE_INNER_OPT_TRAINER_KEYS:
        current = str(trainer_params.get("search_inner_opt_device", "auto")).strip().lower()
        if strategy != "auto" or current in {"", "auto"}:
            trainer_params["search_inner_opt_device"] = str(assigned_device)
            applied_device = str(assigned_device)
        else:
            applied_device = current

    return _clone_assembly_with_trainer_params(assembly, trainer_params=trainer_params), applied_device


def _run_semantic_portfolio_task(
    data: TrainFlowInput,
    run_spec: SemanticTrainFlowSpec,
    config: Any | None,
    model_id: str,
) -> tuple[str, TrainFlowResult]:
    result = run_semantic_train_flow(data, spec=run_spec, config=config)
    return str(model_id), result


def _coerce_resource_component_bundle(
    value: Any,
    *,
    default_label: str,
) -> tuple[ExecutionResourceRequest, ...]:
    if value is None:
        return tuple()

    if isinstance(value, Mapping):
        payload = dict(value)
        if "components" in payload:
            items = tuple(payload.get("components", ()))
            return tuple(coerce_execution_resource_request(item) for item in items)
        if "request" in payload:
            return (coerce_execution_resource_request(payload.get("request"), label=default_label),)
        if "execution_resources" in payload:
            return _coerce_resource_component_bundle(payload.get("execution_resources"), default_label=default_label)
        if "execution_resource_requests" in payload:
            return _coerce_resource_component_bundle(
                payload.get("execution_resource_requests"),
                default_label=default_label,
            )
        if "execution_resource_request" in payload:
            return _coerce_resource_component_bundle(
                payload.get("execution_resource_request"),
                default_label=default_label,
            )
        direct_keys = {"threads", "backend", "device_tokens", "devices", "gpu_devices", "metadata", "label"}
        if any(key in payload for key in direct_keys):
            return (coerce_execution_resource_request(payload, label=default_label),)
        return tuple()

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = tuple(value)
        out: list[ExecutionResourceRequest] = []
        for idx, item in enumerate(items):
            item_label = default_label if len(items) == 1 else f"{default_label}:{idx}"
            out.extend(_coerce_resource_component_bundle(item, default_label=item_label))
        return tuple(out)

    return (coerce_execution_resource_request(value, label=default_label),)


def _collect_model_spec_resource_requests(model_spec: ModelSpec | None) -> tuple[ExecutionResourceRequest, ...]:
    if model_spec is None or model_spec.metadata is None:
        return tuple()

    metadata = dict(model_spec.metadata)
    candidates = (
        ("execution_resources", metadata.get("execution_resources")),
        ("execution_resource_requests", metadata.get("execution_resource_requests")),
        ("execution_resource_request", metadata.get("execution_resource_request")),
        ("problem_execution_resources", metadata.get("problem_execution_resources")),
        ("evaluation_execution_resources", metadata.get("evaluation_execution_resources")),
    )
    out: list[ExecutionResourceRequest] = []
    for key, raw in candidates:
        out.extend(_coerce_resource_component_bundle(raw, default_label=f"model_spec:{key}"))
    return tuple(out)


def _collect_portfolio_trainer_resource_requests(
    trainer: Any,
    *,
    run_spec: SemanticTrainFlowSpec,
) -> tuple[ExecutionResourceRequest, ...]:
    portfolio_getter = getattr(trainer, "portfolio_execution_resource_requests", None)
    if callable(portfolio_getter):
        return tuple(
            coerce_execution_resource_request(item)
            for item in tuple(
                portfolio_getter(
                    run_spec=run_spec,
                    model_spec=run_spec.model_spec,
                )
            )
        )

    request_getter_many = getattr(trainer, "execution_resource_requests", None)
    if callable(request_getter_many):
        return tuple(coerce_execution_resource_request(item) for item in tuple(request_getter_many()))

    request_getter_one = getattr(trainer, "execution_resource_request", None)
    if callable(request_getter_one):
        return (coerce_execution_resource_request(request_getter_one()),)

    return tuple()


def _build_portfolio_phase_request(
    *,
    run_spec: SemanticTrainFlowSpec,
    config: Any | None,
    model_id: str,
    order: int,
    phase_backend: str,
) -> ExecutionResourceRequest:
    trainer = build_trainer(run_spec.assembly.trainer, config=config)
    normalized_components: list[ExecutionResourceRequest] = []
    component_sources: list[dict[str, Any]] = []

    trainer_components = _collect_portfolio_trainer_resource_requests(trainer, run_spec=run_spec)
    if trainer_components:
        normalized_components.extend(trainer_components)
        component_sources.append(
            {
                "source": "trainer",
                "count": int(len(trainer_components)),
                "components": [component.as_dict() for component in trainer_components],
            }
        )

    model_spec_components = _collect_model_spec_resource_requests(run_spec.model_spec)
    if model_spec_components:
        normalized_components.extend(model_spec_components)
        component_sources.append(
            {
                "source": "model_spec.metadata",
                "count": int(len(model_spec_components)),
                "components": [component.as_dict() for component in model_spec_components],
            }
        )

    if not normalized_components:
        normalized_components = [
            ExecutionResourceRequest(
                threads=1,
                backend="serial",
                label=str(getattr(trainer, "name", type(trainer).__name__)),
            )
        ]
        component_sources.append(
            {
                "source": "portfolio_fallback",
                "count": 1,
                "components": [normalized_components[0].as_dict()],
            }
        )

    component_tuple = tuple(normalized_components)
    request = sum_execution_resource_requests(component_tuple, label=f"portfolio:{str(model_id)}")
    metadata = dict(request.metadata)
    metadata.update(
        {
            "model_id": str(model_id),
            "order": int(order),
            "trainer_key": str(run_spec.assembly.trainer.trainer_key),
            "components": [component.as_dict() for component in component_tuple],
            "component_sources": list(component_sources),
        }
    )
    return ExecutionResourceRequest(
        threads=max(1, int(request.threads)),
        backend=str(phase_backend),
        label=f"portfolio:{str(model_id)}",
        device_tokens=tuple(request.device_tokens),
        metadata=metadata,
    )


def run_semantic_portfolio_flow(
    data: TrainFlowInput,
    *,
    spec: SemanticTrainFlowSpec,
    model_specs: Sequence[ModelSpec],
    config: Any | None = None,
) -> TrainPortfolioResult:
    """Run one semantic flow per ModelSpec and aggregate into a comparable portfolio report."""

    if spec.model_spec is not None:
        raise ValueError("run_semantic_portfolio_flow expects spec.model_spec=None; pass model_specs only")

    model_specs_seq = tuple(model_specs)
    if not model_specs_seq:
        raise ValueError("model_specs must not be empty")

    resolved_execution = _resolve_semantic_execution(spec)
    resource_offer = detect_local_execution_offer()
    parallel_mode = _normalize_portfolio_parallel_mode(resolved_execution.backend)
    gpu_strategy = _normalize_portfolio_gpu_strategy(resolved_execution.gpu_strategy)
    normalized_gpu_devices = _normalize_portfolio_gpu_devices(tuple(resolved_execution.gpu_devices))
    fail_fast = bool(resolved_execution.fail_fast)
    max_workers = resolve_phase_worker_count(
        resolved_execution.max_workers,
        n_tasks=len(model_specs_seq),
        offer=resource_offer,
    )

    seen_model_ids: set[str] = set()
    run_results: Dict[str, TrainFlowResult] = {}
    run_errors: Dict[str, str] = {}
    model_rows: list[Dict[str, Any]] = []
    prepared_runs: list[Dict[str, Any]] = []

    for i, raw in enumerate(model_specs_seq):
        model_spec = _clone_model_spec(raw)
        model_id = str(model_spec.model_id).strip() or f"model_{i}"
        if model_id in seen_model_ids:
            raise ValueError(f"Duplicate model_id in portfolio: '{model_id}'")
        seen_model_ids.add(model_id)

        model_output_dir: str | None = None
        if spec.output_dir:
            model_output_dir = str(Path(spec.output_dir) / model_id)

        context_store_cfg = spec.context_store
        if isinstance(context_store_cfg, Mapping):
            context_store_cfg = dict(context_store_cfg)

        snapshot_store_cfg = spec.snapshot_store
        if isinstance(snapshot_store_cfg, Mapping):
            snapshot_store_cfg = dict(snapshot_store_cfg)

        requested_device = _select_portfolio_device(
            strategy=gpu_strategy,
            normalized_devices=normalized_gpu_devices,
            model_index=i,
        )
        assembly_with_device, applied_device = _apply_portfolio_device_assignment(
            assembly=spec.assembly,
            assigned_device=requested_device,
            gpu_strategy=gpu_strategy,
        )
        effective_default_device = applied_device if applied_device is not None else resolved_execution.default_device
        run_execution = ExecutionSpec(
            backend=str(resolved_execution.backend),
            max_workers=resolved_execution.max_workers,
            fail_fast=bool(resolved_execution.fail_fast),
            gpu_strategy=str(resolved_execution.gpu_strategy),
            gpu_devices=tuple(resolved_execution.gpu_devices),
            default_device=effective_default_device,
        )

        run_spec = SemanticTrainFlowSpec(
            assembly=assembly_with_device,
            execution=run_execution,
            training_init=spec.training_init,
            eval_splits=tuple(spec.eval_splits),
            output_dir=model_output_dir,
            save_artifact=bool(spec.save_artifact),
            save_report=bool(spec.save_report),
            save_checkpoint=bool(spec.save_checkpoint),
            checkpoint_dir=spec.checkpoint_dir,
            replay_from_checkpoint=spec.replay_from_checkpoint,
            model_spec=model_spec,
            capabilities=tuple(spec.capabilities),
            capability_strict=bool(spec.capability_strict),
            run_name=f"{spec.run_name}__{model_id}",
            context_store=context_store_cfg,
            snapshot_store=snapshot_store_cfg,
            portfolio_parallel_mode=str(resolved_execution.backend),
            portfolio_max_workers=resolved_execution.max_workers,
            portfolio_fail_fast=bool(resolved_execution.fail_fast),
            portfolio_gpu_strategy=str(resolved_execution.gpu_strategy),
            portfolio_gpu_devices=tuple(resolved_execution.gpu_devices),
        )

        prepared_runs.append(
            {
                "order": int(i),
                "model_id": model_id,
                "model_spec": model_spec,
                "run_spec": run_spec,
                "run_execution": run_execution,
                "requested_device": requested_device,
                "assigned_device": applied_device,
            }
        )

    actual_parallel_mode = str(parallel_mode)
    if int(len(prepared_runs)) <= 1 or int(max_workers or 1) <= 1:
        actual_parallel_mode = "serial"

    concurrent_slots = 1 if actual_parallel_mode == "serial" else int(max_workers or 1)
    active_phase_runs = tuple(sorted(prepared_runs, key=lambda item: int(item["order"]))[:concurrent_slots])
    portfolio_budget = assert_phase_resource_budget(
        "portfolio",
        tuple(
            _build_portfolio_phase_request(
                run_spec=item["run_spec"],
                config=config,
                model_id=str(item["model_id"]),
                order=int(item["order"]),
                phase_backend=str(actual_parallel_mode),
            )
            for item in active_phase_runs
        ),
        offer=resource_offer,
    )
    portfolio_budget["parallel_mode"] = str(actual_parallel_mode)
    portfolio_budget["concurrent_slots"] = int(concurrent_slots)
    portfolio_budget["concurrent_model_ids"] = [str(item["model_id"]) for item in active_phase_runs]

    execution_runtime = ExecutionRuntime()
    execution_tasks = tuple(
        ExecutionTask(
            task_id=str(item["model_id"]),
            fn=_run_semantic_portfolio_task,
            args=(data, item["run_spec"], config, item["model_id"]),
            metadata={
                "order": int(item["order"]),
                "requested_device": item["requested_device"],
                "assigned_device": item["assigned_device"],
            },
        )
        for item in prepared_runs
    )
    execution_batch = execution_runtime.map(
        execution_tasks,
        backend=actual_parallel_mode,
        max_workers=max_workers,
        fail_fast=bool(fail_fast),
    )
    for record in execution_batch.records:
        model_id = str(record.task_id)
        if not bool(record.ok):
            run_errors[model_id] = str(record.error or "unknown_error")
            continue
        completed_model_id, result = record.value
        run_results[str(completed_model_id)] = result
    runtime_execution = execution_batch.as_dict()

    for item in sorted(prepared_runs, key=lambda x: int(x["order"])):
        model_spec = item["model_spec"]
        model_id = str(item["model_id"])
        model_spec_payload = {
            "feature_names": model_spec.feature_names,
            "target_names": model_spec.target_names,
            "feature_indices": model_spec.feature_indices,
            "target_indices": model_spec.target_indices,
            "strict": bool(model_spec.strict),
            "metadata": model_spec.metadata,
        }

        if model_id in run_results:
            result = run_results[model_id]
            model_rows.append(
                {
                    "model_id": model_id,
                    "status": "ok",
                    "run_name": str(item["run_spec"].run_name),
                    "output_dir": result.output_dir,
                    "metrics": _jsonable(result.metrics),
                    "selection_rmse": _portfolio_selection_rmse(result.metrics),
                    "requested_device": item["requested_device"],
                    "assigned_device": item["assigned_device"],
                    "model_spec": _jsonable(model_spec_payload),
                }
            )
        else:
            model_rows.append(
                {
                    "model_id": model_id,
                    "status": "failed",
                    "run_name": str(item["run_spec"].run_name),
                    "output_dir": str(item["run_spec"].output_dir) if item["run_spec"].output_dir else None,
                    "metrics": {},
                    "selection_rmse": float("inf"),
                    "requested_device": item["requested_device"],
                    "assigned_device": item["assigned_device"],
                    "error": run_errors.get(model_id, "unknown_error"),
                    "model_spec": _jsonable(model_spec_payload),
                }
            )

    best_model_id: str | None = None
    best_rmse = float("inf")
    for row in model_rows:
        if str(row.get("status", "ok")) != "ok":
            continue
        rmse = float(row.get("selection_rmse", float("inf")))
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_id = str(row["model_id"])

    summary = {
        "run_name": str(spec.run_name),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "portfolio_size": int(len(model_rows)),
        "selection_metric": "rmse(test->valid->train)",
        "best_model_id": best_model_id,
        "runtime": {
            "declared_execution": _execution_spec_as_mapping(resolved_execution),
            "parallel_mode": actual_parallel_mode,
            "max_workers": None if actual_parallel_mode == "serial" else int(max_workers or 1),
            "fail_fast": bool(fail_fast),
            "gpu_strategy": gpu_strategy,
            "gpu_devices": tuple(normalized_gpu_devices),
            "resource_budget": portfolio_budget,
            "execution": runtime_execution,
        },
        "models": model_rows,
    }
    if run_errors:
        summary["errors"] = dict(run_errors)

    out_dir: str | None = None
    if spec.output_dir:
        run_dir = Path(spec.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        out_dir = str(run_dir)
        if bool(spec.save_report):
            (run_dir / "portfolio_report.json").write_text(
                json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    return TrainPortfolioResult(
        runs=run_results,
        summary=summary,
        output_dir=out_dir,
    )

