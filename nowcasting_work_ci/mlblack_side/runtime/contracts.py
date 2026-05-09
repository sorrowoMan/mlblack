from __future__ import annotations

from collections.abc import Iterator, Mapping as MappingABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class RuntimeContextKey(str, Enum):
    ARGV = "argv"
    RUNTIME_SEED = "runtime_seed"
    REPRODUCIBILITY = "reproducibility"
    STARTED_AT = "started_at"
    FINISHED_AT = "finished_at"
    FAILED_AT = "failed_at"
    DURATION_SEC = "duration_sec"
    STAGE_RESULTS = "stage_results"
    ARGS = "args"
    PREPARED = "prepared"
    SEARCH = "search"
    COMPARISON = "comparison"
    FINAL_RESULT = "final_result"
    OUTPUT_ROOT = "out_root"
    GRAPH_CACHE_RESOURCE = "graph_cache_resource"
    SUMMARY_PATH = "summary_path"
    FAILED_STAGE = "failed_stage"
    LAST_STAGE = "last_stage"
    LAST_STAGE_RESULT = "last_stage_result"


class RuntimeStageName(str, Enum):
    PARSE_CLI = "parse_cli"
    BUILD_RUNTIME = "build_runtime"
    OUTER_SEARCH = "outer_search"
    EVALUATE_FINAL = "evaluate_final"
    ASSEMBLE_RESULT = "assemble_result"


@dataclass(frozen=True)
class RuntimeStageContract:
    name: RuntimeStageName
    requires: tuple[RuntimeContextKey, ...]
    provides: tuple[RuntimeContextKey, ...]
    allows_direct_io: bool = False
    notes: str = ""


@dataclass(frozen=True)
class RuntimeLayerRule:
    layer: str
    responsibility: str
    allows_direct_io: bool
    forbidden: tuple[str, ...] = ()


class RuntimePayload(MappingABC[str, Any]):
    """Small mapping-compatible payload base for stable stage interfaces."""

    def to_mapping(self) -> dict[str, Any]:
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        return self.to_mapping()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_mapping())

    def __len__(self) -> int:
        return len(self.to_mapping())


RUNTIME_STAGE_CONTRACTS: tuple[RuntimeStageContract, ...] = (
    RuntimeStageContract(
        name=RuntimeStageName.PARSE_CLI,
        requires=(RuntimeContextKey.ARGV,),
        provides=(RuntimeContextKey.ARGS, RuntimeContextKey.RUNTIME_SEED),
        allows_direct_io=False,
        notes="Parse argv into RuntimeCliConfig only.",
    ),
    RuntimeStageContract(
        name=RuntimeStageName.BUILD_RUNTIME,
        requires=(RuntimeContextKey.ARGS,),
        provides=(
            RuntimeContextKey.PREPARED,
            RuntimeContextKey.OUTPUT_ROOT,
            RuntimeContextKey.GRAPH_CACHE_RESOURCE,
        ),
        allows_direct_io=False,
        notes="Assemble runtime resources and feature space only.",
    ),
    RuntimeStageContract(
        name=RuntimeStageName.OUTER_SEARCH,
        requires=(RuntimeContextKey.ARGS, RuntimeContextKey.PREPARED),
        provides=(RuntimeContextKey.SEARCH,),
        allows_direct_io=False,
        notes="Run outer optimization and keep search state in memory only.",
    ),
    RuntimeStageContract(
        name=RuntimeStageName.EVALUATE_FINAL,
        requires=(RuntimeContextKey.ARGS, RuntimeContextKey.PREPARED, RuntimeContextKey.SEARCH),
        provides=(RuntimeContextKey.COMPARISON,),
        allows_direct_io=False,
        notes="Run final symbolic/xgb comparison only.",
    ),
    RuntimeStageContract(
        name=RuntimeStageName.ASSEMBLE_RESULT,
        requires=(
            RuntimeContextKey.ARGS,
            RuntimeContextKey.PREPARED,
            RuntimeContextKey.SEARCH,
            RuntimeContextKey.COMPARISON,
        ),
        provides=(RuntimeContextKey.FINAL_RESULT,),
        allows_direct_io=False,
        notes="Assemble report payload and final result only; plugins own side effects.",
    ),
)


RUNTIME_LAYER_RULES: tuple[RuntimeLayerRule, ...] = (
    RuntimeLayerRule(
        layer="runtime/config.py",
        responsibility="CLI/config source of truth",
        allows_direct_io=False,
        forbidden=("file write", "report generation", "resource cleanup"),
    ),
    RuntimeLayerRule(
        layer="runtime/assembly.py",
        responsibility="register helpers and runtime context assembly",
        allows_direct_io=False,
        forbidden=("summary write", "plot write", "report print"),
    ),
    RuntimeLayerRule(
        layer="runtime/build_runtime.py",
        responsibility="total runtime assembly entry",
        allows_direct_io=False,
        forbidden=("summary write", "cache close"),
    ),
    RuntimeLayerRule(
        layer="runtime/stages.py",
        responsibility="control-plane stage wiring",
        allows_direct_io=False,
        forbidden=("model fit details", "summary write", "graph cleanup"),
    ),
    RuntimeLayerRule(
        layer="runtime/actions/*",
        responsibility="stage implementation",
        allows_direct_io=False,
        forbidden=("summary write", "plot write", "resource close"),
    ),
    RuntimeLayerRule(
        layer="plugins/*",
        responsibility="runtime side effects and resource lifecycle",
        allows_direct_io=True,
        forbidden=("search policy", "candidate decoding", "objective mutation"),
    ),
)


@dataclass(frozen=True)
class SearchStageResult(RuntimePayload):
    problem: Any
    outer_sec: float
    outer_meta: Mapping[str, Any]
    resource_budget: Mapping[str, Any]
    run: Mapping[str, Any]
    top_cache: tuple[Mapping[str, Any], ...]
    best_row: Mapping[str, Any]
    best_genome: tuple[Mapping[str, Any], ...]
    best_k: int
    best_decode_meta: Mapping[str, Any]
    best_subset_idx: tuple[int, ...]
    dynamic_epoch_logs: tuple[Mapping[str, Any], ...]
    candidates: tuple[Any, ...]

    @property
    def best_obj_coverage_error(self) -> float:
        return float(self.best_row.get("obj_coverage_error", float("inf")))

    @property
    def best_obj_pinaw(self) -> float:
        return float(self.best_row.get("obj_pinaw", float("inf")))

    @property
    def best_obj_interval_score(self) -> float:
        return float(self.best_row.get("obj_interval_score", float("inf")))

    @property
    def n_cached_evals(self) -> int:
        cache = getattr(self.problem, "_cache", None)
        return int(len(cache)) if cache is not None else 0

    def to_mapping(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "outer_sec": float(self.outer_sec),
            "outer_meta": dict(self.outer_meta),
            "resource_budget": dict(self.resource_budget),
            "run": dict(self.run),
            "top_cache": [dict(row) for row in self.top_cache],
            "best_row": dict(self.best_row),
            "best_genome": list(self.best_genome),
            "best_k": int(self.best_k),
            "best_decode_meta": dict(self.best_decode_meta),
            "best_subset_idx": [int(v) for v in self.best_subset_idx],
            "dynamic_epoch_logs": [dict(row) for row in self.dynamic_epoch_logs],
            "candidates": list(self.candidates),
        }


@dataclass(frozen=True)
class ComparisonStageResult(RuntimePayload):
    fit_final: Mapping[str, Any]
    sym_rmse: float
    sym_mae: float
    interval_alpha: float
    interval_method: str
    sym_interval_info: Mapping[str, Any]
    sym_interval: Mapping[str, Any]
    xgb_rmse: float
    xgb_mae: float
    xgb_calib_q: float
    xgb_interval: Mapping[str, Any]

    @property
    def symbolic_test_pinaw(self) -> float:
        return float(self.sym_interval.get("pinaw", float("inf")))

    @property
    def symbolic_test_interval_score(self) -> float:
        return float(self.sym_interval.get("interval_score", float("inf")))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "fit_final": dict(self.fit_final),
            "sym_rmse": float(self.sym_rmse),
            "sym_mae": float(self.sym_mae),
            "interval_alpha": float(self.interval_alpha),
            "interval_method": str(self.interval_method),
            "sym_interval_info": dict(self.sym_interval_info),
            "sym_interval": dict(self.sym_interval),
            "xgb_rmse": float(self.xgb_rmse),
            "xgb_mae": float(self.xgb_mae),
            "xgb_calib_q": float(self.xgb_calib_q),
            "xgb_interval": dict(self.xgb_interval),
        }


@dataclass(frozen=True)
class SummaryReportPayload(RuntimePayload):
    report: Mapping[str, Any]
    out_root: str
    graph_cache_snapshot: Mapping[str, Any]
    sym_rmse: float
    xgb_rmse: float
    sym_interval: Mapping[str, Any]
    xgb_interval: Mapping[str, Any]
    interval_alpha: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "report": dict(self.report),
            "out_root": str(self.out_root),
            "graph_cache_snapshot": dict(self.graph_cache_snapshot),
            "sym_rmse": float(self.sym_rmse),
            "xgb_rmse": float(self.xgb_rmse),
            "sym_interval": dict(self.sym_interval),
            "xgb_interval": dict(self.xgb_interval),
            "interval_alpha": float(self.interval_alpha),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SummaryReportPayload":
        required_keys = (
            "report",
            "out_root",
            "graph_cache_snapshot",
            "sym_rmse",
            "xgb_rmse",
            "sym_interval",
            "xgb_interval",
            "interval_alpha",
        )
        missing = [k for k in required_keys if k not in payload]
        if missing:
            raise KeyError(f"missing summary report payload keys: {missing}")
        return cls(
            report=dict(payload["report"]),
            out_root=str(payload["out_root"]),
            graph_cache_snapshot=dict(payload["graph_cache_snapshot"]),
            sym_rmse=float(payload["sym_rmse"]),
            xgb_rmse=float(payload["xgb_rmse"]),
            sym_interval=dict(payload["sym_interval"]),
            xgb_interval=dict(payload["xgb_interval"]),
            interval_alpha=float(payload["interval_alpha"]),
        )


@dataclass(frozen=True)
class ResultSummaryPayload(RuntimePayload):
    out_root: str
    best_obj_coverage_error: float
    best_obj_pinaw: float
    best_obj_interval_score: float
    symbolic_test_rmse: float
    xgb_test_rmse: float
    symbolic_test_pinaw: float
    symbolic_test_interval_score: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "out_root": str(self.out_root),
            "best_obj_coverage_error": float(self.best_obj_coverage_error),
            "best_obj_pinaw": float(self.best_obj_pinaw),
            "best_obj_interval_score": float(self.best_obj_interval_score),
            "symbolic_test_rmse": float(self.symbolic_test_rmse),
            "xgb_test_rmse": float(self.xgb_test_rmse),
            "symbolic_test_pinaw": float(self.symbolic_test_pinaw),
            "symbolic_test_interval_score": float(self.symbolic_test_interval_score),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ResultSummaryPayload":
        required_keys = (
            "out_root",
            "best_obj_coverage_error",
            "best_obj_pinaw",
            "best_obj_interval_score",
            "symbolic_test_rmse",
            "xgb_test_rmse",
            "symbolic_test_pinaw",
            "symbolic_test_interval_score",
        )
        missing = [k for k in required_keys if k not in payload]
        if missing:
            raise KeyError(f"missing result summary payload keys: {missing}")
        return cls(
            out_root=str(payload["out_root"]),
            best_obj_coverage_error=float(payload["best_obj_coverage_error"]),
            best_obj_pinaw=float(payload["best_obj_pinaw"]),
            best_obj_interval_score=float(payload["best_obj_interval_score"]),
            symbolic_test_rmse=float(payload["symbolic_test_rmse"]),
            xgb_test_rmse=float(payload["xgb_test_rmse"]),
            symbolic_test_pinaw=float(payload["symbolic_test_pinaw"]),
            symbolic_test_interval_score=float(payload["symbolic_test_interval_score"]),
        )


@dataclass(frozen=True)
class FinalStageResult(RuntimePayload):
    status: str
    report_payload: SummaryReportPayload
    result_summary: ResultSummaryPayload

    def to_mapping(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "report_payload": self.report_payload.to_mapping(),
            "result_summary": self.result_summary.to_mapping(),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FinalStageResult":
        required_keys = ("status", "report_payload", "result_summary")
        missing = [k for k in required_keys if k not in payload]
        if missing:
            raise KeyError(f"missing final stage payload keys: {missing}")
        return cls(
            status=str(payload["status"]),
            report_payload=SummaryReportPayload.from_mapping(payload["report_payload"]),
            result_summary=ResultSummaryPayload.from_mapping(payload["result_summary"]),
        )


def ctx_get(context: Mapping[str, Any], key: RuntimeContextKey, default: Any = None) -> Any:
    if not isinstance(context, Mapping):
        return default
    return context.get(str(key.value), default)


def ctx_require(context: Mapping[str, Any], key: RuntimeContextKey) -> Any:
    value = ctx_get(context, key, default=None)
    if value is None and str(key.value) not in context:
        raise KeyError(f"missing required runtime context key: {key.value}")
    return value


def ctx_set(context: dict[str, Any], key: RuntimeContextKey, value: Any) -> None:
    context[str(key.value)] = value


__all__ = [
    "ComparisonStageResult",
    "FinalStageResult",
    "RUNTIME_LAYER_RULES",
    "RUNTIME_STAGE_CONTRACTS",
    "ResultSummaryPayload",
    "RuntimeContextKey",
    "RuntimeLayerRule",
    "RuntimePayload",
    "RuntimeStageContract",
    "RuntimeStageName",
    "SearchStageResult",
    "SummaryReportPayload",
    "ctx_get",
    "ctx_require",
    "ctx_set",
]
