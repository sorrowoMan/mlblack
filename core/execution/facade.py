from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol, Sequence

from .runtime import ExecutionBatchResult, ExecutionRecord, ExecutionRuntime, ExecutionTask


class ExecutionHandle(Protocol):
    def done(self) -> bool:
        ...

    def result(self, timeout: float | None = None) -> ExecutionRecord | ExecutionBatchResult:
        ...

    def cancel(self) -> bool:
        ...

    def status(self) -> str:
        ...


@dataclass(frozen=True)
class ScopeBackendBinding:
    scope: str
    backend: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", str(self.scope).strip().lower() or "default")
        object.__setattr__(self, "backend", str(self.backend).strip().lower() or "serial")

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": str(self.scope),
            "backend": str(self.backend),
        }


class _FutureExecutionHandle:
    def __init__(
        self,
        future: concurrent.futures.Future[ExecutionRecord | ExecutionBatchResult],
        *,
        executor: concurrent.futures.Executor,
    ) -> None:
        self._future = future
        self._executor = executor
        self._future.add_done_callback(lambda _done: self._executor.shutdown(wait=False))

    def done(self) -> bool:
        return bool(self._future.done())

    def result(self, timeout: float | None = None) -> ExecutionRecord | ExecutionBatchResult:
        return self._future.result(timeout=timeout)

    def cancel(self) -> bool:
        return bool(self._future.cancel())

    def status(self) -> str:
        if self._future.cancelled():
            return "cancelled"
        if self._future.running():
            return "running"
        if self._future.done():
            return "finished"
        return "pending"


class ExecutionFacade:
    """Scope-aware L0 execution facade aligned with nsgablack-style acceleration usage."""

    def __init__(self, runtime: ExecutionRuntime | None = None) -> None:
        self.runtime = runtime or ExecutionRuntime()
        self._lock = RLock()
        self._default_backends: dict[str, str] = {}

    def set_default_backend(self, *, scope: str, backend: str) -> None:
        scope_key = str(scope).strip().lower() or "default"
        backend_key = str(backend).strip().lower() or "serial"
        self.runtime.backend_registry.resolve(backend_key)
        with self._lock:
            self._default_backends[scope_key] = backend_key

    def clear_default_backend(self, *, scope: str) -> None:
        scope_key = str(scope).strip().lower() or "default"
        with self._lock:
            self._default_backends.pop(scope_key, None)

    def get_default_backend(self, *, scope: str) -> str | None:
        scope_key = str(scope).strip().lower() or "default"
        with self._lock:
            return self._default_backends.get(scope_key)

    def list_scope_defaults(self) -> tuple[ScopeBackendBinding, ...]:
        with self._lock:
            rows = tuple(
                ScopeBackendBinding(scope=scope, backend=backend)
                for scope, backend in sorted(self._default_backends.items())
            )
        return rows

    def describe_surface(self, *, torch_module: Any | None = None) -> dict[str, Any]:
        return {
            "scope_defaults": [row.as_dict() for row in self.list_scope_defaults()],
            "runtime": self.runtime.describe_surface(torch_module=torch_module),
        }

    def run(
        self,
        task: ExecutionTask,
        *,
        scope: str = "default",
        backend: str | None = None,
    ) -> ExecutionRecord:
        chosen = self._resolve_backend(scope=scope, backend=backend)
        return self.runtime.run(task, backend=chosen)

    def map(
        self,
        tasks: Sequence[ExecutionTask],
        *,
        scope: str = "default",
        backend: str | None = None,
        max_workers: int | None = None,
        fail_fast: bool = False,
    ) -> ExecutionBatchResult:
        chosen = self._resolve_backend(scope=scope, backend=backend)
        return self.runtime.map(
            tasks,
            backend=chosen,
            max_workers=max_workers,
            fail_fast=fail_fast,
        )

    def submit(
        self,
        task: ExecutionTask,
        *,
        scope: str = "default",
        backend: str | None = None,
    ) -> ExecutionHandle:
        chosen = self._resolve_backend(scope=scope, backend=backend)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlblack-exec-run")
        future = executor.submit(self.runtime.run, task, backend=chosen)
        return _FutureExecutionHandle(future, executor=executor)

    def map_async(
        self,
        tasks: Sequence[ExecutionTask],
        *,
        scope: str = "default",
        backend: str | None = None,
        max_workers: int | None = None,
        fail_fast: bool = False,
    ) -> ExecutionHandle:
        chosen = self._resolve_backend(scope=scope, backend=backend)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlblack-exec-map")
        future = executor.submit(
            self.runtime.map,
            tasks,
            backend=chosen,
            max_workers=max_workers,
            fail_fast=fail_fast,
        )
        return _FutureExecutionHandle(future, executor=executor)

    def _resolve_backend(self, *, scope: str, backend: str | None) -> str:
        if backend is not None and str(backend).strip():
            chosen = str(backend).strip().lower()
            self.runtime.backend_registry.resolve(chosen)
            return chosen
        default_backend = self.get_default_backend(scope=scope)
        if default_backend is not None:
            self.runtime.backend_registry.resolve(default_backend)
            return str(default_backend)
        return "serial"


__all__ = ["ExecutionFacade", "ExecutionHandle", "ScopeBackendBinding"]
