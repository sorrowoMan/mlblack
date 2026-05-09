from __future__ import annotations

from collections.abc import Iterator, Mapping as MappingABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


class LifecyclePayload(MappingABC[str, Any]):
    """Small mapping-compatible payload base for lifecycle contracts."""

    def to_mapping(self) -> dict[str, Any]:
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        return self.to_mapping()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_mapping())

    def __len__(self) -> int:
        return len(self.to_mapping())


@dataclass(frozen=True)
class StageResultDescriptor(LifecyclePayload):
    result_type: str
    payload_kind: str
    payload_type: str | None = None
    mapping_keys: tuple[str, ...] = tuple()

    @classmethod
    def from_result(cls, result: Any) -> "StageResultDescriptor":
        payload_type: str | None = None
        mapping_keys: tuple[str, ...] = tuple()

        if result is None:
            return cls(result_type="NoneType", payload_kind="none", payload_type=None, mapping_keys=tuple())

        mapping_payload: Mapping[str, Any] | None = None
        to_mapping = getattr(result, "to_mapping", None)
        if callable(to_mapping):
            try:
                raw_mapping = to_mapping()
            except Exception:
                raw_mapping = None
            if isinstance(raw_mapping, Mapping):
                mapping_payload = {str(k): raw_mapping[k] for k in raw_mapping}
                payload_type = str(type(result).__name__)

        if mapping_payload is not None:
            mapping_keys = tuple(str(k) for k in mapping_payload.keys())
            return cls(
                result_type=str(type(result).__name__),
                payload_kind="typed_mapping",
                payload_type=payload_type,
                mapping_keys=mapping_keys,
            )

        if isinstance(result, Mapping):
            mapping_keys = tuple(str(k) for k in result.keys())
            return cls(
                result_type=str(type(result).__name__),
                payload_kind="mapping",
                payload_type=None,
                mapping_keys=mapping_keys,
            )

        if isinstance(result, (list, tuple, set)):
            return cls(
                result_type=str(type(result).__name__),
                payload_kind="sequence",
                payload_type=None,
                mapping_keys=tuple(),
            )

        return cls(
            result_type=str(type(result).__name__),
            payload_kind="scalar",
            payload_type=None,
            mapping_keys=tuple(),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "result_type": str(self.result_type),
            "payload_kind": str(self.payload_kind),
            "payload_type": None if self.payload_type is None else str(self.payload_type),
            "mapping_keys": [str(x) for x in self.mapping_keys],
        }


@dataclass(frozen=True)
class StageLifecyclePayload(LifecyclePayload):
    stage: str
    status: str
    started_at: float | None = None
    finished_at: float | None = None
    duration_sec: float | None = None
    result_descriptor: StageResultDescriptor | None = None
    context_keys: tuple[str, ...] = tuple()
    new_context_keys: tuple[str, ...] = tuple()
    error_type: str | None = None
    error_message: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "stage": str(self.stage),
            "status": str(self.status),
            "started_at": None if self.started_at is None else float(self.started_at),
            "finished_at": None if self.finished_at is None else float(self.finished_at),
            "duration_sec": None if self.duration_sec is None else float(self.duration_sec),
            "result_descriptor": None if self.result_descriptor is None else self.result_descriptor.to_mapping(),
            "context_keys": [str(x) for x in self.context_keys],
            "new_context_keys": [str(x) for x in self.new_context_keys],
            "error_type": None if self.error_type is None else str(self.error_type),
            "error_message": None if self.error_message is None else str(self.error_message),
        }


@dataclass(frozen=True)
class LifecycleStatePayload(LifecyclePayload):
    status: str
    started_at: float | None = None
    finished_at: float | None = None
    failed_at: float | None = None
    duration_sec: float | None = None
    last_stage: str | None = None
    failed_stage: str | None = None
    context_keys: tuple[str, ...] = tuple()
    stage_count: int = 0
    stage_results: Mapping[str, StageResultDescriptor] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "started_at": None if self.started_at is None else float(self.started_at),
            "finished_at": None if self.finished_at is None else float(self.finished_at),
            "failed_at": None if self.failed_at is None else float(self.failed_at),
            "duration_sec": None if self.duration_sec is None else float(self.duration_sec),
            "last_stage": None if self.last_stage is None else str(self.last_stage),
            "failed_stage": None if self.failed_stage is None else str(self.failed_stage),
            "context_keys": [str(x) for x in self.context_keys],
            "stage_count": int(self.stage_count),
            "stage_results": {
                str(k): v.to_mapping()
                for k, v in self.stage_results.items()
            },
        }


@dataclass(frozen=True)
class ExperimentLifecycleReport(LifecyclePayload):
    run_name: str
    timestamp_utc: str
    result_type: str | None
    capabilities: Mapping[str, Any]
    lifecycle_events: tuple[Mapping[str, Any], ...]
    state: LifecycleStatePayload
    stages: tuple[StageLifecyclePayload, ...]
    inner_runtime_events: tuple[Mapping[str, Any], ...] = tuple()
    control_plane_contract: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        run_name: str,
        result: Any,
        capabilities: Mapping[str, Any],
        lifecycle_events: tuple[Mapping[str, Any], ...],
        inner_runtime_events: tuple[Mapping[str, Any], ...] = tuple(),
        control_plane_contract: Mapping[str, Any] | None = None,
        state: LifecycleStatePayload,
        stages: tuple[StageLifecyclePayload, ...],
    ) -> "ExperimentLifecycleReport":
        return cls(
            run_name=str(run_name),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            result_type=None if result is None else str(type(result).__name__),
            capabilities=dict(capabilities),
            lifecycle_events=tuple(dict(x) for x in lifecycle_events),
            inner_runtime_events=tuple(dict(x) for x in inner_runtime_events),
            control_plane_contract={} if control_plane_contract is None else dict(control_plane_contract),
            state=state,
            stages=stages,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "run_name": str(self.run_name),
            "timestamp_utc": str(self.timestamp_utc),
            "result_type": None if self.result_type is None else str(self.result_type),
            "capabilities": dict(self.capabilities),
            "lifecycle_events": [dict(x) for x in self.lifecycle_events],
            "inner_runtime_events": [dict(x) for x in self.inner_runtime_events],
            "control_plane_contract": dict(self.control_plane_contract),
            "state": self.state.to_mapping(),
            "stages": [stage.to_mapping() for stage in self.stages],
        }


__all__ = [
    "ExperimentLifecycleReport",
    "LifecyclePayload",
    "LifecycleStatePayload",
    "StageLifecyclePayload",
    "StageResultDescriptor",
]
