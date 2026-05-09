from __future__ import annotations

import numpy as np

from core.orchestration.capabilities import FlowCapability
from nowcasting_work_ci.mlblack_side.runtime.contracts import (
    FinalStageResult,
    ResultSummaryPayload,
    RuntimeContextKey,
    SummaryReportPayload,
)
from plugins import ReproducibilityPlugin, RuntimeResourcePlugin
from workflow.reproducibility import ReproducibilityConfig, apply_reproducibility


class _FakeGraphCache:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def test_summary_report_payload_roundtrip() -> None:
    payload = SummaryReportPayload(
        report={"status": "ok"},
        out_root="C:/tmp/example",
        graph_cache_snapshot={"hits": 3},
        sym_rmse=1.2,
        xgb_rmse=1.1,
        sym_interval={"pinaw": 0.3},
        xgb_interval={"pinaw": 0.2},
        interval_alpha=0.1,
    )

    restored = SummaryReportPayload.from_mapping(payload.to_mapping())

    assert restored.out_root == "C:/tmp/example"
    assert restored.graph_cache_snapshot["hits"] == 3
    assert restored.sym_interval["pinaw"] == 0.3


def test_final_stage_result_roundtrip_and_mapping_access() -> None:
    payload = SummaryReportPayload(
        report={"status": "ok"},
        out_root="C:/tmp/example",
        graph_cache_snapshot={"hits": 3},
        sym_rmse=1.2,
        xgb_rmse=1.1,
        sym_interval={"pinaw": 0.3},
        xgb_interval={"pinaw": 0.2},
        interval_alpha=0.1,
    )
    summary = ResultSummaryPayload(
        out_root="C:/tmp/example",
        best_obj_coverage_error=0.02,
        best_obj_pinaw=0.3,
        best_obj_interval_score=20.0,
        symbolic_test_rmse=1.2,
        xgb_test_rmse=1.1,
        symbolic_test_pinaw=0.3,
        symbolic_test_interval_score=20.0,
    )

    result = FinalStageResult(status="completed", report_payload=payload, result_summary=summary)
    restored = FinalStageResult.from_mapping(result.to_mapping())

    assert result["status"] == "completed"
    assert result["report_payload"]["out_root"] == "C:/tmp/example"
    assert restored.result_summary.symbolic_test_rmse == 1.2


def test_runtime_resource_plugin_closes_graph_cache_and_clears_context() -> None:
    cache = _FakeGraphCache()
    context = {RuntimeContextKey.GRAPH_CACHE_RESOURCE.value: cache}

    plugin = RuntimeResourcePlugin()
    plugin.on_experiment_finish(result={}, context=context)

    assert cache.close_calls == 1
    assert context[RuntimeContextKey.GRAPH_CACHE_RESOURCE.value] is None


def test_reproducibility_plugin_and_helper_reset_seeded_numpy_sequence() -> None:
    context: dict[str, object] = {}
    plugin = ReproducibilityPlugin(seed=123, deterministic_torch=False)
    plugin.on_experiment_start(context)
    first = np.random.rand(5)

    info = apply_reproducibility(ReproducibilityConfig(seed=123, deterministic_torch=False))
    second = np.random.rand(5)

    assert np.allclose(first, second)
    assert context[RuntimeContextKey.RUNTIME_SEED.value] == 123
    assert RuntimeContextKey.REPRODUCIBILITY.value in context
    assert info["numpy_seeded"] is True


def test_runtime_plugins_share_flow_capability_contract_surface() -> None:
    repro = ReproducibilityPlugin(seed=7)
    resource = RuntimeResourcePlugin()

    assert isinstance(repro, FlowCapability)
    assert isinstance(resource, FlowCapability)
    assert RuntimeContextKey.RUNTIME_SEED.value in repro.get_context_contract()["provides"]
    assert RuntimeContextKey.GRAPH_CACHE_RESOURCE.value in resource.get_context_contract()["mutates"]
