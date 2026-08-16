from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from mlblack.project.scaffold import print_case_check

if __package__ in {None, ""}:
    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    from _bootstrap import ensure_case_importable  # noqa: E402
    ensure_case_importable(Path(__file__))
    from build_solver import SymbolicOrthogonalNestedCaseConfig, build_stage1_solver, build_stage2_solver  # noqa: E402
    from problem import build_symbolic_regression_data  # noqa: E402
    from reporting import write_case_report  # noqa: E402
else:
    from ._bootstrap import ensure_case_importable  # noqa: E402
    ensure_case_importable(Path(__file__))
    from .build_solver import SymbolicOrthogonalNestedCaseConfig, build_stage1_solver, build_stage2_solver  # noqa: E402
    from .problem import build_symbolic_regression_data  # noqa: E402
    from .reporting import write_case_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run nested symbolic orthogonal search: nsgablack outer, mlblack inner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--suite-id", type=str, default="")
    parser.add_argument("--output-dir", type=str, default=SymbolicOrthogonalNestedCaseConfig.output_dir)
    parser.add_argument("--seed", type=int, default=SymbolicOrthogonalNestedCaseConfig.seed)
    parser.add_argument("--n-samples", type=int, default=SymbolicOrthogonalNestedCaseConfig.n_samples)
    parser.add_argument("--stage1-generations", "--generations", dest="stage1_generations", type=int, default=SymbolicOrthogonalNestedCaseConfig.stage1_generations)
    parser.add_argument("--stage1-pop-size", "--pop-size", dest="stage1_pop_size", type=int, default=SymbolicOrthogonalNestedCaseConfig.stage1_pop_size)
    parser.add_argument("--stage1-offspring-size", "--offspring-size", dest="stage1_offspring_size", type=int, default=SymbolicOrthogonalNestedCaseConfig.stage1_offspring_size)
    parser.add_argument("--stage1-inner-steps", type=int, default=SymbolicOrthogonalNestedCaseConfig.stage1_inner_steps)
    parser.add_argument("--stage2-generations", type=int, default=SymbolicOrthogonalNestedCaseConfig.stage2_generations)
    parser.add_argument("--stage2-pop-size", type=int, default=SymbolicOrthogonalNestedCaseConfig.stage2_pop_size)
    parser.add_argument("--stage2-offspring-size", type=int, default=SymbolicOrthogonalNestedCaseConfig.stage2_offspring_size)
    parser.add_argument("--stage2-inner-steps", type=int, default=SymbolicOrthogonalNestedCaseConfig.stage2_inner_steps)
    parser.add_argument("--stage2-task-kind", type=str, default=SymbolicOrthogonalNestedCaseConfig.stage2_task_kind)
    parser.add_argument("--stage2-head-kind", type=str, default=SymbolicOrthogonalNestedCaseConfig.stage2_head_kind)
    parser.add_argument("--enable-path-memory", action="store_true")
    parser.add_argument("--disable-graph-cache", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> SymbolicOrthogonalNestedCaseConfig:
    return SymbolicOrthogonalNestedCaseConfig(
        output_dir=str(args.output_dir),
        seed=int(args.seed),
        n_samples=int(args.n_samples),
        stage1_generations=int(args.stage1_generations),
        stage1_pop_size=int(args.stage1_pop_size),
        stage1_offspring_size=int(args.stage1_offspring_size),
        stage1_inner_steps=int(args.stage1_inner_steps),
        stage2_generations=int(args.stage2_generations),
        stage2_pop_size=int(args.stage2_pop_size),
        stage2_offspring_size=int(args.stage2_offspring_size),
        stage2_inner_steps=int(args.stage2_inner_steps),
        stage2_task_kind=str(args.stage2_task_kind),
        stage2_head_kind=str(args.stage2_head_kind),
        enable_path_memory=bool(args.enable_path_memory),
        enable_graph_cache=not bool(args.disable_graph_cache),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    suite_id = str(args.suite_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    cfg = _config_from_args(args)
    data = build_symbolic_regression_data(n_samples=int(cfg.n_samples), valid_fraction=float(cfg.valid_fraction), seed=int(cfg.seed))
    stage1_solver = build_stage1_solver(cfg, suite_id=suite_id, data=data)

    if bool(args.check):
        print_case_check(stage1_solver, resource_context=cfg.resource_context)
        return

    stage1_result = stage1_solver.run(return_dict=True)
    stage1_problem = stage1_solver.problem
    basis_artifact = stage1_problem.build_artifact()

    stage2_solver = build_stage2_solver(cfg, suite_id=suite_id, basis_artifact=basis_artifact, data=data)
    stage2_result = stage2_solver.run(return_dict=True)
    stage2_problem = stage2_solver.problem
    task_artifact = stage2_problem.build_artifact()

    output_dir = cfg.output_root(suite_id)
    summary = {
        "suite_id": suite_id,
        "protocol": "nsgablack_outer_mlblack_inner_symbolic_orthogonal_nested_v1",
        "config": cfg.__dict__,
        "resource_context": dict(cfg.resource_context),
        "stage1": {
            "solver_result": stage1_result,
            "evaluation_count": int(getattr(stage1_problem, "evaluation_count", 0)),
            "record_count": int(len(stage1_problem.evaluation_records)),
            "best_record": None if stage1_problem.best_record is None else stage1_problem.best_record.as_dict(),
            "basis_artifact": basis_artifact.describe(include_record=False),
        },
        "stage2": {
            "solver_result": stage2_result,
            "evaluation_count": int(getattr(stage2_problem, "evaluation_count", 0)),
            "record_count": int(len(stage2_problem.evaluation_records)),
            "best_record": None if stage2_problem.best_record is None else stage2_problem.best_record.as_dict(),
            "task_artifact": task_artifact.describe(),
        },
    }
    artifacts = write_case_report(
        output_dir=output_dir,
        summary=summary,
        stage1_records=stage1_problem.evaluation_records,
        stage2_records=stage2_problem.evaluation_records,
        basis_artifact=basis_artifact,
        task_artifact=task_artifact,
    )
    print(f"[symbolic-orthogonal-nested] suite_id={suite_id}")
    print(f"[symbolic-orthogonal-nested] output_dir={output_dir}")
    print(f"[symbolic-orthogonal-nested] stage1_records={len(stage1_problem.evaluation_records)}")
    print(f"[symbolic-orthogonal-nested] stage2_records={len(stage2_problem.evaluation_records)}")
    print(f"[symbolic-orthogonal-nested] basis_artifact={basis_artifact.artifact_id}")
    print(f"[symbolic-orthogonal-nested] task_score={summary['stage2']['best_record']['report']['candidate_score']['score'] if summary['stage2']['best_record'] else None}")
    print(f"[symbolic-orthogonal-nested] summary={artifacts['summary']}")


if __name__ == "__main__":
    main()
