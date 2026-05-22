from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_CASE_DIR = Path(__file__).resolve().parents[1] / "cases" / "symbolic_orthogonal_nested"
if str(_CASE_DIR) not in sys.path:
    sys.path.insert(0, str(_CASE_DIR))

from _bootstrap import ensure_case_importable  # noqa: E402
ensure_case_importable(Path(__file__))
from build_solver import SymbolicOrthogonalNestedCaseConfig, build_stage1_solver, build_stage2_solver  # noqa: E402
from problem import build_symbolic_regression_data  # noqa: E402
from reporting import write_case_report  # noqa: E402
from mlblack.pipeline.data import NumericDataView  # noqa: E402


VARIANTS: dict[str, tuple[str, str]] = {
    "point": ("regression", "point"),
    "interval": ("interval", "interval_center_radius"),
    "classification": ("classification", "binary_logistic"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small nsgablack outer + mlblack inner symbolic head benchmark matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--variants", type=str, default="point,interval,classification")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "runs" / "symbolic_orthogonal_nested"))
    parser.add_argument("--suite-id", type=str, default="")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--pop-size", type=int, default=4)
    parser.add_argument("--inner-steps", type=int, default=3)
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    requested = tuple(v.strip() for v in str(args.variants).split(",") if v.strip())
    unknown = [name for name in requested if name not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; expected one of {sorted(VARIANTS)}")
    suite_id = str(args.suite_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_root = Path(args.output_dir).expanduser().resolve() / suite_id
    if args.check:
        print(
            "symbolic_nested_benchmark scaffold ok | "
            f"variants={list(requested)} output_root={output_root}"
        )
        return

    rows = []
    for variant in requested:
        task_kind, head_kind = VARIANTS[variant]
        cfg = SymbolicOrthogonalNestedCaseConfig(
            output_dir=str(output_root),
            seed=int(args.seed),
            n_samples=int(args.n_samples),
            stage1_generations=int(args.generations),
            stage1_pop_size=int(args.pop_size),
            stage1_offspring_size=int(args.pop_size),
            stage1_inner_steps=int(args.inner_steps),
            stage1_inner_population_size=max(4, int(args.pop_size)),
            stage2_generations=int(args.generations),
            stage2_pop_size=int(args.pop_size),
            stage2_offspring_size=int(args.pop_size),
            stage2_inner_steps=int(args.inner_steps),
            stage2_inner_population_size=max(4, int(args.pop_size)),
            stage2_task_kind=task_kind,
            stage2_head_kind=head_kind,
            enable_graph_cache=True,
        )
        data = _data_for_variant(task_kind, seed=int(args.seed), n_samples=int(args.n_samples))
        variant_id = f"{variant}_{suite_id}"
        stage1_solver = build_stage1_solver(cfg, suite_id=variant_id, data=data)
        stage1_result = stage1_solver.run(return_dict=True)
        stage1_problem = stage1_solver.problem
        basis_artifact = stage1_problem.build_artifact()
        stage2_solver = build_stage2_solver(cfg, suite_id=variant_id, basis_artifact=basis_artifact, data=data)
        stage2_result = stage2_solver.run(return_dict=True)
        stage2_problem = stage2_solver.problem
        task_artifact = stage2_problem.build_artifact()
        summary = {
            "suite_id": variant_id,
            "variant": variant,
            "task_kind": task_kind,
            "head_kind": head_kind,
            "protocol": "benchmark_nsgablack_outer_mlblack_inner_symbolic_v1",
            "stage1": {
                "solver_result": stage1_result,
                "record_count": int(len(stage1_problem.evaluation_records)),
                "best_record": None if stage1_problem.best_record is None else stage1_problem.best_record.as_dict(),
                "basis_artifact": basis_artifact.describe(include_record=False),
            },
            "stage2": {
                "solver_result": stage2_result,
                "record_count": int(len(stage2_problem.evaluation_records)),
                "best_record": None if stage2_problem.best_record is None else stage2_problem.best_record.as_dict(),
                "task_artifact": task_artifact.describe(),
            },
        }
        artifacts = write_case_report(
            output_dir=cfg.output_root(variant_id),
            summary=summary,
            stage1_records=stage1_problem.evaluation_records,
            stage2_records=stage2_problem.evaluation_records,
            basis_artifact=basis_artifact,
            task_artifact=task_artifact,
        )
        score = _candidate_score(summary)
        rows.append({"variant": variant, "task_kind": task_kind, "head_kind": head_kind, "score": score, "summary": artifacts["summary"], "dashboard": artifacts["artifact_dashboard"]})

    output_root.mkdir(parents=True, exist_ok=True)
    matrix_path = output_root / "benchmark_matrix.json"
    matrix_path.write_text(json.dumps({"suite_id": suite_id, "rows": rows}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"suite_id": suite_id, "matrix": str(matrix_path), "rows": rows}, ensure_ascii=False, indent=2, default=str))


def _data_for_variant(task_kind: str, *, seed: int, n_samples: int) -> NumericDataView:
    data = build_symbolic_regression_data(n_samples=int(n_samples), valid_fraction=0.25, seed=int(seed))
    if str(task_kind) != "classification":
        return data
    threshold = float(np.median(data.y_train))
    return NumericDataView(
        X_train=data.X_train,
        y_train=(data.y_train > threshold).astype(int),
        X_valid=data.X_valid,
        y_valid=None if data.y_valid is None else (data.y_valid > threshold).astype(int),
        feature_names=data.feature_names,
        target_name="synthetic_symbolic_class",
        metadata={**dict(data.metadata), "task_kind": "classification", "threshold": threshold},
    )


def _candidate_score(summary: Mapping[str, Any]) -> float | None:
    best = dict(dict(summary.get("stage2", {}) or {}).get("best_record", {}) or {})
    score = dict(dict(best.get("report", {}) or {}).get("candidate_score", {}) or {}).get("score")
    return None if score is None else float(score)


if __name__ == "__main__":
    main()
