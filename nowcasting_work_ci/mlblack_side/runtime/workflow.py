from __future__ import annotations

import sys
from typing import Mapping, Sequence

from plugins import ReportWriterPlugin, ReproducibilityPlugin, RuntimeResourcePlugin
from nowcasting_work_ci.mlblack_side.runtime.stages import build_experiment_stages
from nowcasting_work_ci.mlblack_side.runtime.contracts import RuntimeContextKey
from nowcasting_work_ci.mlblack_side.runtime.config import parse_runtime_args
from workflow import ExperimentOrchestrator, RuntimeHook


def main(
    argv: Sequence[str] | None = None,
    *,
    hooks: Sequence[RuntimeHook] | None = None,
    enable_default_plugins: bool = True,
) -> Mapping[str, object]:
    effective_argv = list(argv) if argv is not None else list(sys.argv[1:])
    parsed_args = parse_runtime_args(effective_argv)
    default_plugins = []
    if enable_default_plugins:
        default_plugins.extend(
            [
                ReproducibilityPlugin(seed=int(parsed_args.seed)),
                ReportWriterPlugin(),
                RuntimeResourcePlugin(),
            ]
        )
    orchestrator = ExperimentOrchestrator(
        capabilities=tuple(default_plugins),
        hooks=tuple(hooks or ()),
        strict=False,
    )
    return orchestrator.run(
        build_experiment_stages(effective_argv),
        context={
            RuntimeContextKey.ARGV.value: list(effective_argv),
            RuntimeContextKey.RUNTIME_SEED.value: int(parsed_args.seed),
        },
    )


__all__ = ["main"]
