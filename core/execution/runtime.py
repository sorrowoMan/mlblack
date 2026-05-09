from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from core.execution.resources import clamp_worker_count, detect_local_execution_offer
from core.execution.registry import (
    ExecutionBackendRegistry,
    ExecutionBackendSpec,
    ExecutionDeviceRegistry,
)


ExecutionBackend = str


@dataclass(frozen=True)
class ExecutionTask:
    task_id: str
    fn: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", str(self.task_id).strip() or "task")
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ExecutionRecord:
    task_id: str
    ok: bool
    value: Any = None
    error: str | None = None
    backend: ExecutionBackend = "serial"
    index: int = 0
    latency_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", str(self.task_id).strip() or "task")
        object.__setattr__(self, "backend", str(self.backend).strip().lower() or "serial")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": str(self.task_id),
            "ok": bool(self.ok),
            "error": self.error,
            "backend": str(self.backend),
            "index": int(self.index),
            "latency_ms": None if self.latency_ms is None else float(self.latency_ms),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ExecutionBatchResult:
    backend: ExecutionBackend
    max_workers: int | None
    submitted: int
    succeeded: int
    failed: int
    records: tuple[ExecutionRecord, ...]
    duration_ms: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", str(self.backend).strip().lower() or "serial")
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": str(self.backend),
            "max_workers": None if self.max_workers is None else int(self.max_workers),
            "submitted": int(self.submitted),
            "succeeded": int(self.succeeded),
            "failed": int(self.failed),
            "duration_ms": float(self.duration_ms),
            "records": [record.as_dict() for record in self.records],
            "metadata": dict(self.metadata),
        }


class ExecutionRuntimeError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        record: ExecutionRecord | None = None,
        partial_records: Sequence[ExecutionRecord] = (),
    ) -> None:
        super().__init__(message)
        self.record = record
        self.partial_records = tuple(partial_records)


def _execute_task(task: ExecutionTask) -> Any:
    return task.fn(*tuple(task.args), **dict(task.kwargs))


def _normalize_max_workers(max_workers: int | None, *, n_tasks: int) -> int | None:
    offer = detect_local_execution_offer()
    return clamp_worker_count(max_workers, n_tasks=n_tasks, offer=offer)


class ExecutionRuntime:
    """Minimal L0 execution substrate for sync/parallel task dispatch."""

    def __init__(
        self,
        *,
        backend_registry: ExecutionBackendRegistry | None = None,
        device_registry: ExecutionDeviceRegistry | None = None,
    ) -> None:
        self.backend_registry = backend_registry or ExecutionBackendRegistry.global_registry()
        self.device_registry = device_registry or ExecutionDeviceRegistry.global_registry()

    def run(
        self,
        task: ExecutionTask,
        *,
        backend: str = "serial",
    ) -> ExecutionRecord:
        batch = self.map((task,), backend=backend, max_workers=1, fail_fast=False)
        if not batch.records:
            raise RuntimeError("ExecutionRuntime.run produced no record")
        return batch.records[0]

    def describe_surface(self, *, torch_module: Any | None = None) -> dict[str, Any]:
        return {
            "backends": list(self.backend_registry.describe_specs()),
            "device_kinds": list(self.device_registry.describe_kinds(torch_module=torch_module)),
        }

    def map(
        self,
        tasks: Sequence[ExecutionTask],
        *,
        backend: str = "serial",
        max_workers: int | None = None,
        fail_fast: bool = False,
    ) -> ExecutionBatchResult:
        tasks_seq = tuple(tasks)
        backend_spec = self.backend_registry.resolve(backend)
        actual_backend = str(backend_spec.key)
        resource_offer = detect_local_execution_offer(device_registry=self.device_registry)
        worker_count = clamp_worker_count(max_workers, n_tasks=len(tasks_seq), offer=resource_offer)
        if not bool(backend_spec.supports_parallel) or len(tasks_seq) <= 1 or int(worker_count or 1) <= 1:
            actual_backend = "serial"
            worker_count = 1
            backend_spec = self.backend_registry.resolve("serial")

        started_at = time.perf_counter()
        if actual_backend == "serial":
            records = self._run_serial(tasks_seq, fail_fast=fail_fast)
        else:
            records = self._run_parallel(
                tasks_seq,
                backend_spec=backend_spec,
                max_workers=int(worker_count or 1),
                fail_fast=fail_fast,
            )

        duration_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        succeeded = sum(1 for record in records if bool(record.ok))
        failed = sum(1 for record in records if not bool(record.ok))
        return ExecutionBatchResult(
            backend=actual_backend,
            max_workers=(None if actual_backend == "serial" else int(worker_count or 1)),
            submitted=int(len(tasks_seq)),
            succeeded=int(succeeded),
            failed=int(failed),
            records=tuple(records),
            duration_ms=float(duration_ms),
            metadata={
                "requested_backend": str(backend),
                "requested_max_workers": None if max_workers is None else int(max_workers),
                "effective_max_workers": int(worker_count or 1),
                "resource_offer": resource_offer.as_dict(),
            },
        )

    @staticmethod
    def _run_serial(
        tasks: Sequence[ExecutionTask],
        *,
        fail_fast: bool,
    ) -> tuple[ExecutionRecord, ...]:
        records: list[ExecutionRecord] = []
        for index, task in enumerate(tasks):
            started_at = time.perf_counter()
            try:
                value = _execute_task(task)
                record = ExecutionRecord(
                    task_id=task.task_id,
                    ok=True,
                    value=value,
                    backend="serial",
                    index=int(index),
                    latency_ms=max(0.0, (time.perf_counter() - started_at) * 1000.0),
                    metadata=task.metadata,
                )
            except Exception as exc:
                record = ExecutionRecord(
                    task_id=task.task_id,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    backend="serial",
                    index=int(index),
                    latency_ms=max(0.0, (time.perf_counter() - started_at) * 1000.0),
                    metadata=task.metadata,
                )
                records.append(record)
                if fail_fast:
                    raise ExecutionRuntimeError(
                        f"Execution task '{task.task_id}' failed in serial mode: {record.error}",
                        record=record,
                        partial_records=tuple(records),
                    ) from exc
                continue
            records.append(record)
        return tuple(records)

    @staticmethod
    def _run_parallel(
        tasks: Sequence[ExecutionTask],
        *,
        backend_spec: ExecutionBackendSpec,
        max_workers: int,
        fail_fast: bool,
    ) -> tuple[ExecutionRecord, ...]:
        backend = str(backend_spec.key)
        executor_factory = backend_spec.executor_factory
        if executor_factory is None:
            raise ValueError(f"Parallel backend '{backend}' does not provide executor_factory")
        records_by_index: dict[int, ExecutionRecord] = {}
        with executor_factory(max_workers) as executor:
            future_map: dict[concurrent.futures.Future[Any], tuple[int, ExecutionTask, float]] = {}
            for index, task in enumerate(tasks):
                future = executor.submit(_execute_task, task)
                future_map[future] = (int(index), task, time.perf_counter())

            for future in concurrent.futures.as_completed(future_map):
                index, task, started_at = future_map[future]
                try:
                    value = future.result()
                    record = ExecutionRecord(
                        task_id=task.task_id,
                        ok=True,
                        value=value,
                        backend=backend,
                        index=int(index),
                        latency_ms=max(0.0, (time.perf_counter() - started_at) * 1000.0),
                        metadata=task.metadata,
                    )
                except Exception as exc:
                    record = ExecutionRecord(
                        task_id=task.task_id,
                        ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                        backend=backend,
                        index=int(index),
                        latency_ms=max(0.0, (time.perf_counter() - started_at) * 1000.0),
                        metadata=task.metadata,
                    )
                    records_by_index[index] = record
                    if fail_fast:
                        for pending in future_map:
                            pending.cancel()
                        partial = tuple(records_by_index[k] for k in sorted(records_by_index))
                        raise ExecutionRuntimeError(
                            f"Execution task '{task.task_id}' failed in {backend} mode: {record.error}",
                            record=record,
                            partial_records=partial,
                        ) from exc
                records_by_index[index] = record

        return tuple(records_by_index[k] for k in sorted(records_by_index))


__all__ = [
    "ExecutionBatchResult",
    "ExecutionRecord",
    "ExecutionRuntime",
    "ExecutionRuntimeError",
    "ExecutionTask",
]
