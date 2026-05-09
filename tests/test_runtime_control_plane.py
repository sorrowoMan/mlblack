from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from core.orchestration.capabilities import FlowCapability
from core.orchestration.lifecycle_runtime import LifecycleRuntime
from workflow import ExperimentOrchestrator, ExperimentStage, HookBus


@dataclass
class _RecorderHook:
    events: list[tuple[str, str]] = field(default_factory=list)
    finish_result: Any | None = None
    error_report: Mapping[str, Any] | None = None

    def on_experiment_start(self, context: Mapping[str, Any]) -> None:
        self.events.append(("experiment", "start"))

    def on_stage_start(self, stage: str, context: Mapping[str, Any]) -> None:
        self.events.append(("stage_start", str(stage)))

    def on_stage_end(self, stage: str, payload: Mapping[str, Any], context: Mapping[str, Any]) -> None:
        self.events.append(("stage_end", str(stage)))

    def on_stage_error(self, stage: str, error: Exception, context: Mapping[str, Any]) -> None:
        self.events.append(("stage_error", str(stage)))

    def on_experiment_finish(self, result: Any, context: Mapping[str, Any]) -> None:
        self.events.append(("experiment", "finish"))
        self.finish_result = result

    def on_experiment_error(self, error: Exception, context: Mapping[str, Any]) -> None:
        self.events.append(("experiment", "error"))
        raw = context.get("lifecycle_report")
        if isinstance(raw, Mapping):
            self.error_report = raw


@dataclass
class _CapabilityHook(FlowCapability):
    name: str = "capability_hook"
    priority: int = 0
    enabled: bool = True
    events: list[tuple[str, str]] = field(default_factory=list)
    sink: list[str] = field(default_factory=list)

    def on_experiment_start(self, context: Mapping[str, Any]) -> None:
        self.events.append(("experiment", str(self.name)))
        self.sink.append(f"experiment:{self.name}")

    def on_stage_start(self, stage: str, context: Mapping[str, Any]) -> None:
        self.events.append(("stage_start", f"{self.name}:{stage}"))
        self.sink.append(f"stage_start:{self.name}:{stage}")


@dataclass
class _FlowAliasCapability(FlowCapability):
    name: str = "flow_alias"
    events: list[tuple[str, str]] = field(default_factory=list)

    def on_flow_start(self, context: Mapping[str, Any]) -> None:
        self.events.append(("flow", "start"))

    def on_flow_finish(self, context: Mapping[str, Any]) -> None:
        self.events.append(("flow", "finish"))

    def on_flow_error(self, error: Exception, context: Mapping[str, Any]) -> None:
        self.events.append(("flow", f"error:{type(error).__name__}:{context.get('failed_stage')}"))


@dataclass
class _StagePayloadHook:
    payloads: list[Mapping[str, Any]] = field(default_factory=list)

    def on_experiment_start(self, context: Mapping[str, Any]) -> None:
        return

    def on_stage_start(self, stage: str, context: Mapping[str, Any]) -> None:
        return

    def on_stage_end(self, stage: str, payload: Mapping[str, Any], context: Mapping[str, Any]) -> None:
        self.payloads.append(payload)

    def on_stage_error(self, stage: str, error: Exception, context: Mapping[str, Any]) -> None:
        return

    def on_experiment_finish(self, result: Any, context: Mapping[str, Any]) -> None:
        return

    def on_experiment_error(self, error: Exception, context: Mapping[str, Any]) -> None:
        return


def test_experiment_orchestrator_runs_stage_sequence_and_preserves_context() -> None:
    hook = _RecorderHook()
    bus = HookBus()
    bus.register(hook)

    def stage_a(context: dict[str, Any]) -> dict[str, int]:
        context["value"] = 2
        return {"value": 2}

    def stage_b(context: dict[str, Any]) -> int:
        return int(context["value"]) + 3

    orchestrator = ExperimentOrchestrator(hook_bus=bus)
    result = orchestrator.run(
        [
            ExperimentStage("prepare", stage_a),
            ExperimentStage("solve", stage_b),
        ],
        context={"run_name": "test"},
    )

    assert result == 5
    assert hook.events == [
        ("experiment", "start"),
        ("stage_start", "prepare"),
        ("stage_end", "prepare"),
        ("stage_start", "solve"),
        ("stage_end", "solve"),
        ("experiment", "finish"),
    ]
    assert hook.finish_result == 5
    assert orchestrator.last_report is not None
    lifecycle_report = orchestrator.last_report
    assert lifecycle_report["state"]["status"] == "finished"
    assert lifecycle_report["state"]["stage_count"] == 2
    assert lifecycle_report["stages"][0]["stage"] == "prepare"
    assert lifecycle_report["stages"][1]["stage"] == "solve"
    assert "control_plane_contract" in lifecycle_report
    contract = dict(lifecycle_report["control_plane_contract"])
    assert "lifecycle_events" in contract
    assert "inner_runtime_events" in contract


def test_experiment_orchestrator_emits_stage_error() -> None:
    hook = _RecorderHook()
    bus = HookBus()
    bus.register(hook)

    def bad_stage(context: dict[str, Any]) -> None:
        raise RuntimeError("boom")

    orchestrator = ExperimentOrchestrator(hook_bus=bus)
    try:
        orchestrator.run([ExperimentStage("bad", bad_stage)])
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected RuntimeError")

    assert ("stage_error", "bad") in hook.events
    assert ("experiment", "error") in hook.events
    assert hook.error_report is not None
    assert hook.error_report["state"]["status"] == "failed"
    assert hook.error_report["state"]["failed_stage"] == "bad"
    assert hook.error_report["stages"][0]["status"] == "failed"


def test_hook_bus_accepts_flow_capability_and_orders_by_priority() -> None:
    calls: list[str] = []
    sink: list[str] = []
    early = _CapabilityHook(name="early", priority=10, events=[], sink=sink)
    late = _CapabilityHook(name="late", priority=20, events=[], sink=sink)

    def stage(context: dict[str, Any]) -> str:
        calls.append("stage")
        return "ok"

    bus = HookBus()
    bus.register(late)
    bus.register(early)

    ExperimentOrchestrator(hook_bus=bus).run([ExperimentStage("fit", stage)])

    assert early.events[0] == ("experiment", "early")
    assert late.events[0] == ("experiment", "late")
    assert early.events[1] == ("stage_start", "early:fit")
    assert late.events[1] == ("stage_start", "late:fit")
    assert calls == ["stage"]
    assert sink[:4] == [
        "experiment:early",
        "experiment:late",
        "stage_start:early:fit",
        "stage_start:late:fit",
    ]


def test_lifecycle_runtime_separates_capabilities_from_runtime_hooks() -> None:
    sink: list[str] = []
    capability = _CapabilityHook(name="cap", priority=3, events=[], sink=sink)
    hook = _RecorderHook()

    runtime = LifecycleRuntime()
    runtime.register_capability(capability)
    runtime.register_hook(hook)

    assert runtime.list_capabilities() == (capability,)
    assert runtime.list_hooks() == (hook,)

    capability_report = runtime.build_capability_report()
    assert int(capability_report.get("count", -1)) == 1
    assert capability_report["items"][0]["name"] == "cap"

    full_report = runtime.build_report()
    assert int(full_report.get("count", -1)) == 2


def test_experiment_orchestrator_accepts_capabilities_without_manual_hook_bus() -> None:
    sink: list[str] = []
    capability = _CapabilityHook(name="auto_cap", priority=5, events=[], sink=sink)

    def stage(context: dict[str, Any]) -> str:
        context["status"] = "ok"
        return "done"

    orchestrator = ExperimentOrchestrator(capabilities=(capability,))
    result = orchestrator.run([ExperimentStage("prepare", stage)])

    assert result == "done"
    assert sink[:2] == [
        "experiment:auto_cap",
        "stage_start:auto_cap:prepare",
    ]
    assert orchestrator.last_report is not None
    assert orchestrator.last_report["state"]["status"] == "finished"


def test_runtime_lifecycle_dispatches_flow_alias_hooks() -> None:
    capability = _FlowAliasCapability()

    def stage(context: dict[str, Any]) -> str:
        context["status"] = "ok"
        return "done"

    orchestrator = ExperimentOrchestrator(capabilities=(capability,))
    result = orchestrator.run([ExperimentStage("prepare", stage)])

    assert result == "done"
    assert capability.events == [
        ("flow", "start"),
        ("flow", "finish"),
    ]
    event_table = orchestrator.describe_event_table()
    contract = orchestrator.describe_control_plane_contract()
    assert any("on_flow_start" in tuple(row.get("dispatch_names", ())) for row in event_table)
    assert any("on_experiment_start" in tuple(row.get("dispatch_names", ())) for row in event_table)
    assert any(str(row.get("runtime_key")) == "branch_evaluation.regime_fold" for row in contract["inner_runtime_events"])
    stage_end_row = next(row for row in event_table if "on_stage_end" in tuple(row.get("dispatch_names", ())))
    assert stage_end_row["payload_contract"]["typed_payload"] == "StageLifecyclePayload"
    assert stage_end_row["payload_contract"]["typed_result_descriptor"] == "StageResultDescriptor"


def test_runtime_error_dispatches_flow_error_alias() -> None:
    capability = _FlowAliasCapability()

    def bad_stage(context: dict[str, Any]) -> None:
        raise RuntimeError("boom")

    orchestrator = ExperimentOrchestrator(capabilities=(capability,))
    try:
        orchestrator.run([ExperimentStage("bad", bad_stage)])
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected RuntimeError")

    assert capability.events == [
        ("flow", "start"),
        ("flow", "error:RuntimeError:bad"),
    ]


def test_stage_end_payload_exposes_typed_result_descriptor() -> None:
    hook = _StagePayloadHook()
    bus = HookBus()
    bus.register(hook)

    def stage(context: dict[str, Any]) -> dict[str, Any]:
        context["produced"] = True
        return {"status": "ok", "score": 1.0}

    result = ExperimentOrchestrator(hook_bus=bus).run([ExperimentStage("score", stage)])

    assert isinstance(result, dict)
    assert result["lifecycle_report"]["state"]["status"] == "finished"
    payload = hook.payloads[0]
    assert payload["stage"] == "score"
    assert payload["status"] == "completed"
    assert payload["result_descriptor"]["payload_kind"] == "mapping"
    assert payload["result_descriptor"]["result_type"] == "dict"
    assert payload["result_descriptor"]["mapping_keys"] == ["status", "score"]
    assert "produced" in payload["new_context_keys"]
