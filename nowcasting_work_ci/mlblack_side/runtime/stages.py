from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NSGABLACK_ROOT = ROOT.parent / "nsgablack"
if str(NSGABLACK_ROOT) not in sys.path:
    sys.path.append(str(NSGABLACK_ROOT))

from workflow import ExperimentStage

from .contracts import RuntimeContextKey, RuntimeStageName
from .actions import (
    run_assemble_result_stage,
    run_build_runtime_stage,
    run_evaluate_final_stage,
    run_outer_search_stage,
    run_parse_cli_stage,
)


def build_experiment_stages(argv: Sequence[str] | None = None) -> list[ExperimentStage]:
    return [
        ExperimentStage(RuntimeStageName.PARSE_CLI.value, lambda context, argv=argv: run_parse_cli_stage(context, argv)),
        ExperimentStage(RuntimeStageName.BUILD_RUNTIME.value, run_build_runtime_stage),
        ExperimentStage(RuntimeStageName.OUTER_SEARCH.value, run_outer_search_stage),
        ExperimentStage(RuntimeStageName.EVALUATE_FINAL.value, run_evaluate_final_stage),
        ExperimentStage(RuntimeStageName.ASSEMBLE_RESULT.value, run_assemble_result_stage),
    ]


def run_stage_sequence(argv: Sequence[str] | None = None) -> Mapping[str, Any]:
    context: dict[str, Any] = {RuntimeContextKey.ARGV.value: list(argv) if argv is not None else []}
    result: Mapping[str, Any] | None = None
    for stage in build_experiment_stages(argv):
        result = stage.runner(context)
    if result is None:
        raise RuntimeError("no stages were executed")
    return result


__all__ = ["build_experiment_stages", "run_stage_sequence"]
