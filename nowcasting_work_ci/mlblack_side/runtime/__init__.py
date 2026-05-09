from __future__ import annotations

from workflow import ExperimentOrchestrator, ExperimentStage, HookBus, RuntimeHook

from plugins import ReportWriterPlugin
from .build_runtime import build_runtime
from .config import RuntimeCliConfig, build_parser, parse_runtime_args, parse_runtime_config
from .contracts import (
    ComparisonStageResult,
    FinalStageResult,
    ResultSummaryPayload,
    RuntimeContextKey,
    RuntimeStageName,
    SearchStageResult,
    SummaryReportPayload,
)
from .workflow import main

__all__ = [
    "ComparisonStageResult",
    "main",
    "build_parser",
    "FinalStageResult",
    "HookBus",
    "RuntimeHook",
    "ResultSummaryPayload",
    "RuntimeContextKey",
    "RuntimeStageName",
    "SearchStageResult",
    "SummaryReportPayload",
    "ExperimentStage",
    "ExperimentOrchestrator",
    "RuntimeCliConfig",
    "ReportWriterPlugin",
    "parse_runtime_args",
    "parse_runtime_config",
    "build_runtime",
]
