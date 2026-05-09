from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

from my_project.runtime.config import build_parser


def run_pipeline(*, components: Mapping[str, Any], run_id: str) -> str:
    cfg = components["config"]
    problem = components["problem_builder"](cfg.problem)
    features = components["feature_builder"](problem, cfg.features)
    model_result = components["model_trainer"](features, cfg.model)

    payload = {
        "problem": asdict(problem),
        "runtime": asdict(cfg.runtime),
        "model": {"rmse": float(model_result.rmse)},
    }
    path = components["report_writer"](payload, cfg.reporting, run_id)
    return str(path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    from my_project.build_runtime import build_runtime_components

    components = build_runtime_components()
    path = run_pipeline(components=components, run_id=str(args.run_id))
    print(f"summary={path}")
