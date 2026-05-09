from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from core.orchestration.capabilities import FlowCapability
from core.orchestration.lifecycle_runtime import LifecycleRuntime
from core.orchestration.lifecycle_payloads import (
    ExperimentLifecycleReport,
    LifecycleStatePayload,
    StageLifecyclePayload,
    StageResultDescriptor,
)
from training import describe_inner_runtime_event_table
from .hook_bus import HookBus


StageRunner = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class ExperimentStage:
    name: str
    runner: StageRunner


@dataclass
class ExperimentOrchestrator:
    runtime: LifecycleRuntime | None = None
    hook_bus: HookBus | None = None
    capabilities: Sequence[FlowCapability] = field(default_factory=tuple)
    hooks: Sequence[object] = field(default_factory=tuple)
    strict: bool = False
    last_report: dict[str, Any] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.runtime is None:
            if self.hook_bus is not None and self.hook_bus.runtime is not None:
                self.runtime = self.hook_bus.runtime
            else:
                self.runtime = LifecycleRuntime(strict=bool(self.strict))
        if self.hook_bus is None:
            self.hook_bus = HookBus(strict=bool(self.strict), runtime=self.runtime)
        else:
            self.hook_bus.runtime = self.runtime
            self.hook_bus.dispatcher = self.runtime.dispatcher
        assert self.runtime is not None
        for capability in tuple(self.capabilities):
            self.runtime.register_capability(capability)
        for hook in tuple(self.hooks):
            self.runtime.register_hook(hook)

    def run(
        self,
        stages: Sequence[ExperimentStage],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        ctx = self._build_context(context)
        assert self.runtime is not None
        self.runtime.emit("on_experiment_start", ctx)
        try:
            result: Any = None
            for stage in stages:
                result = self._run_stage(stage, ctx)
            ctx["finished_at"] = float(time.time())
            ctx["duration_sec"] = float(ctx["finished_at"] - ctx["started_at"])
            report = self._build_lifecycle_report(ctx, result=result, status="finished")
            published_result = self._attach_lifecycle_report(result, report)
            ctx["result"] = published_result
            ctx["lifecycle_report"] = report
            self.last_report = dict(report)
            self.runtime.emit("on_experiment_finish", published_result, ctx)
            return published_result
        except Exception as exc:
            ctx["failed_at"] = float(time.time())
            ctx["duration_sec"] = float(ctx["failed_at"] - ctx["started_at"])
            ctx["error"] = exc
            report = self._build_lifecycle_report(ctx, result=None, status="failed")
            ctx["lifecycle_report"] = report
            self.last_report = dict(report)
            self.runtime.emit("on_experiment_error", exc, ctx)
            raise

    def describe_event_table(self) -> tuple[dict[str, Any], ...]:
        assert self.runtime is not None
        return self.runtime.describe_event_table()

    def describe_inner_event_table(self) -> tuple[dict[str, Any], ...]:
        return describe_inner_runtime_event_table()

    def describe_control_plane_contract(self) -> dict[str, Any]:
        assert self.runtime is not None
        return self.runtime.describe_control_plane_contract()

    def _build_context(self, context: Mapping[str, Any] | None) -> dict[str, Any]:
        ctx = dict(context or {})
        ctx.setdefault("stage_results", {})
        ctx.setdefault("stage_reports", [])
        ctx["started_at"] = float(time.time())
        return ctx

    def _run_stage(self, stage: ExperimentStage, context: MutableMapping[str, Any]) -> Any:
        stage_started = float(time.time())
        assert self.runtime is not None
        self.runtime.emit("on_stage_start", stage.name, context)
        context_before = set(str(k) for k in context.keys())
        try:
            result = stage.runner(context)
        except Exception as exc:
            stage_finished = float(time.time())
            payload = StageLifecyclePayload(
                stage=str(stage.name),
                status="failed",
                started_at=stage_started,
                finished_at=stage_finished,
                duration_sec=float(stage_finished - stage_started),
                result_descriptor=None,
                context_keys=tuple(sorted(str(k) for k in context.keys())),
                new_context_keys=tuple(sorted(str(k) for k in set(str(k) for k in context.keys()) - context_before)),
                error_type=str(type(exc).__name__),
                error_message=str(exc),
            )
            context["failed_stage"] = str(stage.name)
            context["last_stage"] = str(stage.name)
            context["last_stage_payload"] = payload
            self._record_stage_payload(context, payload)
            self.runtime.emit("on_stage_error", stage.name, exc, context)
            raise

        stage_results = context.setdefault("stage_results", {})
        if isinstance(stage_results, dict):
            stage_results[str(stage.name)] = result
        context["last_stage"] = str(stage.name)
        context["last_stage_result"] = result

        stage_finished = float(time.time())
        context_after = tuple(sorted(str(k) for k in context.keys()))
        payload = StageLifecyclePayload(
            stage=str(stage.name),
            status="completed",
            started_at=stage_started,
            finished_at=stage_finished,
            duration_sec=float(stage_finished - stage_started),
            result_descriptor=StageResultDescriptor.from_result(result),
            context_keys=context_after,
            new_context_keys=tuple(sorted(str(k) for k in set(context_after) - context_before)),
        )
        context["last_stage_payload"] = payload
        self._record_stage_payload(context, payload)
        self.runtime.emit("on_stage_end", stage.name, payload, context)
        return result

    def _record_stage_payload(self, context: MutableMapping[str, Any], payload: StageLifecyclePayload) -> None:
        stage_reports = context.setdefault("stage_reports", [])
        if isinstance(stage_reports, list):
            stage_reports.append(payload)

    def _build_lifecycle_report(
        self,
        context: Mapping[str, Any],
        *,
        result: Any,
        status: str,
    ) -> dict[str, Any]:
        assert self.runtime is not None
        stage_payloads_raw = context.get("stage_reports", [])
        stage_payloads: tuple[StageLifecyclePayload, ...] = tuple(
            payload
            for payload in tuple(stage_payloads_raw)
            if isinstance(payload, StageLifecyclePayload)
        )
        stage_results: dict[str, StageResultDescriptor] = {}
        for payload in stage_payloads:
            if payload.result_descriptor is not None:
                stage_results[str(payload.stage)] = payload.result_descriptor

        state = LifecycleStatePayload(
            status=str(status),
            started_at=self._as_float(context.get("started_at")),
            finished_at=self._as_float(context.get("finished_at")),
            failed_at=self._as_float(context.get("failed_at")),
            duration_sec=self._as_float(context.get("duration_sec")),
            last_stage=self._as_optional_str(context.get("last_stage")),
            failed_stage=self._as_optional_str(context.get("failed_stage")),
            context_keys=tuple(sorted(str(k) for k in context.keys() if str(k) != "lifecycle_report")),
            stage_count=int(len(stage_payloads)),
            stage_results=stage_results,
        )
        report = ExperimentLifecycleReport.create(
            run_name=self._resolve_run_name(context),
            result=result,
            capabilities=self.runtime.build_report(),
            lifecycle_events=self.describe_event_table(),
            inner_runtime_events=self.describe_inner_event_table(),
            control_plane_contract=self.describe_control_plane_contract(),
            state=state,
            stages=stage_payloads,
        )
        return report.to_mapping()

    def _attach_lifecycle_report(self, result: Any, report: Mapping[str, Any]) -> Any:
        if result is None:
            return {"lifecycle_report": dict(report)}

        to_mapping = getattr(result, "to_mapping", None)
        if callable(to_mapping):
            try:
                payload = to_mapping()
            except Exception:
                payload = None
            if isinstance(payload, Mapping):
                merged = dict(payload)
                merged["lifecycle_report"] = dict(report)
                return merged

        if isinstance(result, Mapping):
            merged = dict(result)
            merged["lifecycle_report"] = dict(report)
            return merged

        return result

    @staticmethod
    def _resolve_run_name(context: Mapping[str, Any]) -> str:
        raw = context.get("run_name", "experiment_orchestrator")
        text = str(raw).strip()
        return text or "experiment_orchestrator"

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _as_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value)
        return text if text else None


__all__ = ["ExperimentStage", "ExperimentOrchestrator", "StageRunner"]
