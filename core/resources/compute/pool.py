"""L0 Compute: PoolScheduler — shared thread pool with semaphore budget.

Unlike ``ResourceAllocator`` (lease = exclusive ownership), this is a shared
pool. Multiple callers submit tasks; each waits until enough threads are free,
runs, then returns threads to the pool.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class PoolTask:
    name: str
    threads: int
    fn: Callable[..., Any]
    args: tuple = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    submit_order: int = 0


@dataclass
class PoolResult:
    task_name: str
    submitted_at: float
    started_at: float
    finished_at: float
    threads_used: int
    pool_available_at_start: int
    result: Any = None
    error: Optional[str] = None

    @property
    def elapsed(self) -> float:
        return self.finished_at - self.started_at if self.started_at else 0.0

    @property
    def wait_time(self) -> float:
        return self.started_at - self.submitted_at if self.started_at else 0.0


class PoolScheduler:
    """L0 shared thread pool — tasks acquire threads, run, release.

    Usage::

        pool = PoolScheduler(total_threads=8)
        fut1 = pool.submit("outer_eval", threads=4, fn=run_outer)
        fut2 = pool.submit("inner_copt", threads=4, fn=run_copt)
        pool.wait_all()
        print(pool.report())
    """

    def __init__(self, total_threads: int) -> None:
        total = max(1, int(total_threads))
        self._sem = threading.BoundedSemaphore(total)
        self._executor = ThreadPoolExecutor(max_workers=total)
        self._futures: list[Future] = []
        self._results: list[PoolResult] = []
        self._counter = 0
        self._lock = threading.Lock()
        self._closed = False

    @property
    def total_threads(self) -> int:
        return self._sem._value if not self._closed else 0

    def available(self) -> int:
        return self._sem._value

    def submit(
        self, name: str, threads: int, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Future:
        if self._closed:
            raise RuntimeError("PoolScheduler is closed")
        needed = max(1, int(threads))
        with self._lock:
            self._counter += 1
            order = self._counter
        task = PoolTask(name=name, threads=needed, fn=fn, args=args, kwargs=kwargs, submit_order=order)
        fut = self._executor.submit(self._run_task, task)
        self._futures.append(fut)
        return fut

    def _run_task(self, task: PoolTask) -> PoolResult:
        submitted_at = time.monotonic()
        available_before = self._sem._value
        acquired = 0
        for _ in range(task.threads):
            self._sem.acquire()
            acquired += 1
        started_at = time.monotonic()
        try:
            result = task.fn(*task.args, **task.kwargs)
            error = None
        except Exception as exc:
            result = None
            error = f"{type(exc).__name__}: {exc}"
        finished_at = time.monotonic()
        for _ in range(acquired):
            self._sem.release()
        pool_result = PoolResult(
            task_name=task.name, submitted_at=submitted_at, started_at=started_at,
            finished_at=finished_at, threads_used=task.threads,
            pool_available_at_start=available_before, result=result, error=error,
        )
        with self._lock:
            self._results.append(pool_result)
        return pool_result

    def as_executor(self, threads: int):
        """Return a ThreadPoolExecutor-like object backed by this pool.

        Usage::

            with pool.as_executor(4) as ex:
                results = ex.map(eval_fn, items)
        """
        return _PoolExecutor(self, threads)

    def wait_all(self, timeout: Optional[float] = None) -> list[PoolResult]:
        for fut in list(self._futures):
            try:
                fut.result(timeout=timeout)
            except Exception:
                pass
        self._futures.clear()
        with self._lock:
            return list(self._results)

    def report(self) -> dict[str, Any]:
        with self._lock:
            results = list(self._results)
        return {
            "total_threads": self._sem._value if not self._closed else 0,
            "tasks_completed": len(results),
            "tasks": [{
                "name": r.task_name, "threads": r.threads_used,
                "wait_ms": round(r.wait_time * 1000, 1),
                "elapsed_ms": round(r.elapsed * 1000, 1),
                "pool_avail_before": r.pool_available_at_start, "error": r.error,
            } for r in results],
        }

    def close(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=True)


class _PoolExecutor:
    """Executor facade that acquires/releases pool threads for map/submit."""

    def __init__(self, pool: PoolScheduler, threads: int) -> None:
        self._pool = pool
        self._threads = max(1, int(threads))
        self._acquired = 0

    def __enter__(self):
        for _ in range(self._threads):
            self._pool._sem.acquire()
            self._acquired += 1
        return self

    def __exit__(self, *_):
        for _ in range(self._acquired):
            self._pool._sem.release()
        return False

    def map(self, fn, *iterables, timeout=None):
        tasks = [(fn,) + tuple(args) for args in zip(*iterables)]
        futures = [self._pool._executor.submit(t[0], *t[1:]) for t in tasks]
        return [f.result(timeout=timeout) for f in futures]

    def submit(self, fn, *args, **kwargs):
        return self._pool._executor.submit(fn, *args, **kwargs)

    def shutdown(self, wait=True):
        pass
