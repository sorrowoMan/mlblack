from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


RUN_SURFACE_CONTRACT_VERSION = "run-surface.v1"


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def stable_json_dumps(value: Any) -> str:
    return json.dumps(_to_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_signature(value: Any) -> str | None:
    payload = stable_json_dumps(value)
    if not payload:
        return None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _coerce_str(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _coerce_str_tuple(values: Sequence[Any] | None) -> tuple[str, ...]:
    if not values:
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class SurfaceRecord:
    contract_version: str = RUN_SURFACE_CONTRACT_VERSION
    framework: str = ""
    project_root: str | None = None
    scaffold_root: str | None = None
    surface_kind: str = ""
    surface_key: str = ""
    surface_label: str = ""
    entry_path: str | None = None
    entry_module: str | None = None
    entry_symbol: str | None = None
    driver_ref: str | None = None
    family_ref: str | None = None
    tags: tuple[str, ...] = ()
    metadata_json: dict[str, Any] = field(default_factory=dict)
    surface_signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(_to_jsonable(asdict(self)))


@dataclass(frozen=True, slots=True)
class AssemblyRecord:
    contract_version: str = RUN_SURFACE_CONTRACT_VERSION
    framework: str = ""
    surface_key: str = ""
    assembly_key: str | None = None
    driver_ref: str | None = None
    family_ref: str | None = None
    preset_ref: str | None = None
    head_ref: str | None = None
    solver_ref: str | None = None
    trainer_ref: str | None = None
    adapter_ref: str | None = None
    representation_refs: tuple[str, ...] = ()
    bias_refs: tuple[str, ...] = ()
    component_refs: tuple[str, ...] = ()
    provider_refs: tuple[str, ...] = ()
    plugin_refs: tuple[str, ...] = ()
    pipeline_refs: tuple[str, ...] = ()
    mount_order: tuple[str, ...] = ()
    component_slots_json: dict[str, Any] = field(default_factory=dict)
    metadata_json: dict[str, Any] = field(default_factory=dict)
    assembly_signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(_to_jsonable(asdict(self)))


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    contract_version: str = RUN_SURFACE_CONTRACT_VERSION
    framework: str = ""
    run_id: str = ""
    artifact_id: str = ""
    artifact_kind: str = ""
    artifact_role: str = ""
    producer_ref: str | None = None
    surface_key: str | None = None
    assembly_signature: str | None = None
    path: str | None = None
    uri: str | None = None
    format: str | None = None
    created_at_utc: str | None = None
    metrics_json: dict[str, Any] = field(default_factory=dict)
    metadata_json: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    artifact_signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(_to_jsonable(asdict(self)))


@dataclass(frozen=True, slots=True)
class RunRecord:
    contract_version: str = RUN_SURFACE_CONTRACT_VERSION
    framework: str = ""
    run_id: str = ""
    namespace: str | None = None
    tag: str | None = None
    status: str = ""
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    duration_s: float | None = None
    surface_key: str = ""
    surface_kind: str = ""
    surface_signature: str | None = None
    assembly_signature: str | None = None
    subject_kind: str | None = None
    subject_key: str | None = None
    subject_signature: str | None = None
    param_signature: str | None = None
    driver_ref: str | None = None
    family_ref: str | None = None
    output_dir: str | None = None
    primary_metric_name: str | None = None
    primary_metric_value: float | None = None
    metric_summary_json: dict[str, Any] = field(default_factory=dict)
    params_json: dict[str, Any] = field(default_factory=dict)
    result_json: dict[str, Any] = field(default_factory=dict)
    component_refs: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    metadata_json: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(_to_jsonable(asdict(self)))


def make_surface_record(
    *,
    framework: str,
    surface_kind: str,
    surface_key: str,
    surface_label: str,
    project_root: str | None = None,
    scaffold_root: str | None = None,
    entry_path: str | None = None,
    entry_module: str | None = None,
    entry_symbol: str | None = None,
    driver_ref: str | None = None,
    family_ref: str | None = None,
    tags: Sequence[Any] | None = None,
    metadata_json: Mapping[str, Any] | None = None,
) -> SurfaceRecord:
    payload = {
        "framework": _coerce_str(framework) or "",
        "project_root": _coerce_str(project_root),
        "scaffold_root": _coerce_str(scaffold_root),
        "surface_kind": _coerce_str(surface_kind) or "",
        "surface_key": _coerce_str(surface_key) or "",
        "surface_label": _coerce_str(surface_label) or "",
        "entry_path": _coerce_str(entry_path),
        "entry_module": _coerce_str(entry_module),
        "entry_symbol": _coerce_str(entry_symbol),
        "driver_ref": _coerce_str(driver_ref),
        "family_ref": _coerce_str(family_ref),
        "tags": _coerce_str_tuple(tags),
        "metadata_json": dict(_to_jsonable(metadata_json or {})),
    }
    signature = stable_signature(payload)
    return SurfaceRecord(**payload, surface_signature=signature)


def make_assembly_record(
    *,
    framework: str,
    surface_key: str,
    assembly_key: str | None = None,
    driver_ref: str | None = None,
    family_ref: str | None = None,
    preset_ref: str | None = None,
    head_ref: str | None = None,
    solver_ref: str | None = None,
    trainer_ref: str | None = None,
    adapter_ref: str | None = None,
    representation_refs: Sequence[Any] | None = None,
    bias_refs: Sequence[Any] | None = None,
    component_refs: Sequence[Any] | None = None,
    provider_refs: Sequence[Any] | None = None,
    plugin_refs: Sequence[Any] | None = None,
    pipeline_refs: Sequence[Any] | None = None,
    mount_order: Sequence[Any] | None = None,
    component_slots_json: Mapping[str, Any] | None = None,
    metadata_json: Mapping[str, Any] | None = None,
) -> AssemblyRecord:
    payload = {
        "framework": _coerce_str(framework) or "",
        "surface_key": _coerce_str(surface_key) or "",
        "assembly_key": _coerce_str(assembly_key),
        "driver_ref": _coerce_str(driver_ref),
        "family_ref": _coerce_str(family_ref),
        "preset_ref": _coerce_str(preset_ref),
        "head_ref": _coerce_str(head_ref),
        "solver_ref": _coerce_str(solver_ref),
        "trainer_ref": _coerce_str(trainer_ref),
        "adapter_ref": _coerce_str(adapter_ref),
        "representation_refs": _coerce_str_tuple(representation_refs),
        "bias_refs": _coerce_str_tuple(bias_refs),
        "component_refs": _coerce_str_tuple(component_refs),
        "provider_refs": _coerce_str_tuple(provider_refs),
        "plugin_refs": _coerce_str_tuple(plugin_refs),
        "pipeline_refs": _coerce_str_tuple(pipeline_refs),
        "mount_order": _coerce_str_tuple(mount_order),
        "component_slots_json": dict(_to_jsonable(component_slots_json or {})),
        "metadata_json": dict(_to_jsonable(metadata_json or {})),
    }
    signature = stable_signature(payload)
    return AssemblyRecord(**payload, assembly_signature=signature)


def make_artifact_record(
    *,
    framework: str,
    run_id: str,
    artifact_id: str,
    artifact_kind: str,
    artifact_role: str,
    producer_ref: str | None = None,
    surface_key: str | None = None,
    assembly_signature: str | None = None,
    path: str | None = None,
    uri: str | None = None,
    format: str | None = None,
    created_at_utc: str | None = None,
    metrics_json: Mapping[str, Any] | None = None,
    metadata_json: Mapping[str, Any] | None = None,
    tags: Sequence[Any] | None = None,
) -> ArtifactRecord:
    payload = {
        "framework": _coerce_str(framework) or "",
        "run_id": _coerce_str(run_id) or "",
        "artifact_id": _coerce_str(artifact_id) or "",
        "artifact_kind": _coerce_str(artifact_kind) or "",
        "artifact_role": _coerce_str(artifact_role) or "",
        "producer_ref": _coerce_str(producer_ref),
        "surface_key": _coerce_str(surface_key),
        "assembly_signature": _coerce_str(assembly_signature),
        "path": _coerce_str(path),
        "uri": _coerce_str(uri),
        "format": _coerce_str(format),
        "created_at_utc": _coerce_str(created_at_utc),
        "metrics_json": dict(_to_jsonable(metrics_json or {})),
        "metadata_json": dict(_to_jsonable(metadata_json or {})),
        "tags": _coerce_str_tuple(tags),
    }
    signature = stable_signature(payload)
    return ArtifactRecord(**payload, artifact_signature=signature)


def make_run_record(
    *,
    framework: str,
    run_id: str,
    status: str,
    surface_key: str,
    surface_kind: str,
    surface_signature: str | None,
    assembly_signature: str | None,
    namespace: str | None = None,
    tag: str | None = None,
    started_at_utc: str | None = None,
    finished_at_utc: str | None = None,
    duration_s: float | None = None,
    subject_kind: str | None = None,
    subject_key: str | None = None,
    subject_json: Mapping[str, Any] | None = None,
    params_json: Mapping[str, Any] | None = None,
    driver_ref: str | None = None,
    family_ref: str | None = None,
    output_dir: str | None = None,
    primary_metric_name: str | None = None,
    primary_metric_value: float | None = None,
    metric_summary_json: Mapping[str, Any] | None = None,
    result_json: Mapping[str, Any] | None = None,
    component_refs: Sequence[Any] | None = None,
    artifact_ids: Sequence[Any] | None = None,
    metadata_json: Mapping[str, Any] | None = None,
) -> RunRecord:
    subject_payload = {
        "subject_kind": _coerce_str(subject_kind),
        "subject_key": _coerce_str(subject_key),
        "subject_json": dict(_to_jsonable(subject_json or {})),
    }
    param_signature = stable_signature(dict(_to_jsonable(params_json or {})))
    payload = {
        "framework": _coerce_str(framework) or "",
        "run_id": _coerce_str(run_id) or "",
        "namespace": _coerce_str(namespace),
        "tag": _coerce_str(tag),
        "status": _coerce_str(status) or "",
        "started_at_utc": _coerce_str(started_at_utc),
        "finished_at_utc": _coerce_str(finished_at_utc),
        "duration_s": None if duration_s is None else float(duration_s),
        "surface_key": _coerce_str(surface_key) or "",
        "surface_kind": _coerce_str(surface_kind) or "",
        "surface_signature": _coerce_str(surface_signature),
        "assembly_signature": _coerce_str(assembly_signature),
        "subject_kind": _coerce_str(subject_payload["subject_kind"]),
        "subject_key": _coerce_str(subject_payload["subject_key"]),
        "subject_signature": stable_signature(subject_payload),
        "param_signature": param_signature,
        "driver_ref": _coerce_str(driver_ref),
        "family_ref": _coerce_str(family_ref),
        "output_dir": _coerce_str(output_dir),
        "primary_metric_name": _coerce_str(primary_metric_name),
        "primary_metric_value": None if primary_metric_value is None else float(primary_metric_value),
        "metric_summary_json": dict(_to_jsonable(metric_summary_json or {})),
        "params_json": dict(_to_jsonable(params_json or {})),
        "result_json": dict(_to_jsonable(result_json or {})),
        "component_refs": _coerce_str_tuple(component_refs),
        "artifact_ids": _coerce_str_tuple(artifact_ids),
        "metadata_json": dict(_to_jsonable(metadata_json or {})),
    }
    return RunRecord(**payload)


__all__ = [
    "RUN_SURFACE_CONTRACT_VERSION",
    "SurfaceRecord",
    "AssemblyRecord",
    "ArtifactRecord",
    "RunRecord",
    "make_surface_record",
    "make_assembly_record",
    "make_artifact_record",
    "make_run_record",
    "stable_json_dumps",
    "stable_signature",
]
