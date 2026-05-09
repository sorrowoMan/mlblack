from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_project.known_relation_symbolic.problem import known_relation_benchmark_keys


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(raw) for key, raw in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _metric(summary: dict[str, Any], *, side: str, split: str, name: str) -> float | None:
    try:
        value = dict(dict(summary.get(side, {}) or {}).get("metrics", {}) or {}).get(split, {}) or {}
        raw = dict(value).get(name)
        return None if raw is None else float(raw)
    except Exception:
        return None


def _basis_metric(summary: dict[str, Any], *, side: str, name: str) -> float | None:
    try:
        value = dict(dict(summary.get(side, {}) or {}).get("basis_recovery", {}) or {}).get(name)
        return None if value is None else float(value)
    except Exception:
        return None


def _basis_flag(summary: dict[str, Any], *, side: str, name: str) -> bool | None:
    try:
        value = dict(dict(summary.get(side, {}) or {}).get("basis_recovery", {}) or {}).get(name)
        return None if value is None else bool(value)
    except Exception:
        return None


def _parse_summary_path(stdout: str) -> Path:
    for line in str(stdout or "").splitlines():
        text = str(line).strip()
        if text.startswith("summary="):
            return Path(text.split("=", 1)[1].strip()).resolve()
    raise RuntimeError("Unable to parse summary path from runner output.")


def _run_scenario(
    *,
    scenario: str,
    db_path: str,
    namespace: str,
    output_root: Path,
    seed: int,
    n_total: int,
    train_ratio: float,
    noise_std: float,
    baseline_max_added_terms: int,
    baseline_topk_features: int,
    orth_candidate_limit: int,
    orth_group_count: int,
    orth_min_basis_count: int,
    orth_max_basis_count: int,
    orth_selection_mode: str,
    orth_assembler_max_added_terms: int,
    orth_assembler_topk_features: int,
    orth_assembler_max_pair_terms: int,
    orth_assembler_max_candidates_per_iter: int,
    orth_assembler_candidate_keep_top: int,
    orth_assembler_max_expr_depth: int,
    orth_assembler_ridge_l2: float,
) -> tuple[dict[str, Any], str]:
    scenario_root = output_root / str(scenario)
    scenario_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "examples" / "run_symbolic_known_relation_compare.py"),
        "--scenario",
        str(scenario),
        "--db-path",
        str(db_path),
        "--namespace",
        str(namespace),
        "--tag",
        f"{scenario}_suite",
        "--output-root",
        str(scenario_root),
        "--seed",
        str(int(seed)),
        "--n-total",
        str(int(n_total)),
        "--train-ratio",
        str(float(train_ratio)),
        "--noise-std",
        str(float(noise_std)),
        "--baseline-max-added-terms",
        str(int(baseline_max_added_terms)),
        "--baseline-topk-features",
        str(int(baseline_topk_features)),
        "--orth-candidate-limit",
        str(int(orth_candidate_limit)),
        "--orth-group-count",
        str(int(orth_group_count)),
        "--orth-min-basis-count",
        str(int(orth_min_basis_count)),
        "--orth-max-basis-count",
        str(int(orth_max_basis_count)),
        "--orth-selection-mode",
        str(orth_selection_mode),
        "--orth-assembler-max-added-terms",
        str(int(orth_assembler_max_added_terms)),
        "--orth-assembler-topk-features",
        str(int(orth_assembler_topk_features)),
        "--orth-assembler-max-pair-terms",
        str(int(orth_assembler_max_pair_terms)),
        "--orth-assembler-max-candidates-per-iter",
        str(int(orth_assembler_max_candidates_per_iter)),
        "--orth-assembler-candidate-keep-top",
        str(int(orth_assembler_candidate_keep_top)),
        "--orth-assembler-max-expr-depth",
        str(int(orth_assembler_max_expr_depth)),
        "--orth-assembler-ridge-l2",
        str(float(orth_assembler_ridge_l2)),
    ]
    proc = subprocess.run(
        command,
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    merged_output = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Scenario '{scenario}' failed with exit code {proc.returncode}.\n{merged_output}"
        )
    summary_path = _parse_summary_path(proc.stdout)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary, merged_output


def _summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    scenario = str(summary.get("scenario") or "")
    comparison = dict(summary.get("comparison", {}) or {})
    baseline = dict(summary.get("baseline", {}) or {})
    orthogonal = dict(summary.get("orthogonal", {}) or {})
    return {
        "scenario": scenario,
        "baseline_test_rmse": _metric(summary, side="baseline", split="test", name="rmse"),
        "orthogonal_test_rmse": _metric(summary, side="orthogonal", split="test", name="rmse"),
        "delta_test_rmse": comparison.get("delta_test_rmse"),
        "baseline_test_r2": _metric(summary, side="baseline", split="test", name="r2"),
        "orthogonal_test_r2": _metric(summary, side="orthogonal", split="test", name="r2"),
        "delta_test_r2": comparison.get("delta_test_r2"),
        "baseline_exact_term_recovery_score": _basis_metric(summary, side="baseline", name="exact_term_recovery_score"),
        "orthogonal_exact_term_recovery_score": _basis_metric(summary, side="orthogonal", name="exact_term_recovery_score"),
        "baseline_phase_equivalent_term_recovery_score": _basis_metric(summary, side="baseline", name="phase_equivalent_term_recovery_score"),
        "orthogonal_phase_equivalent_term_recovery_score": _basis_metric(summary, side="orthogonal", name="phase_equivalent_term_recovery_score"),
        "baseline_family_level_term_recovery_score": _basis_metric(summary, side="baseline", name="family_level_term_recovery_score"),
        "orthogonal_family_level_term_recovery_score": _basis_metric(summary, side="orthogonal", name="family_level_term_recovery_score"),
        "orthogonal_orthogonality_score": _basis_metric(summary, side="orthogonal", name="orthogonality_score"),
        "orthogonal_contains_ratio_basis": _basis_flag(summary, side="orthogonal", name="contains_ratio_basis"),
        "orthogonal_contains_periodic_basis": _basis_flag(summary, side="orthogonal", name="contains_periodic_basis"),
        "orthogonal_contains_piecewise_basis": _basis_flag(summary, side="orthogonal", name="contains_piecewise_basis"),
        "orthogonal_active_term_count": _basis_metric(summary, side="orthogonal", name="active_term_count"),
        "baseline_run_id": str(baseline.get("run_id") or ""),
        "orthogonal_run_id": str(orthogonal.get("run_id") or ""),
        "summary_path": str(summary.get("summary_path") or ""),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(value) for key, value in row.items()})


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("# Known Relation Benchmark Table\n\nNo rows.\n", encoding="utf-8")
        return
    columns = [
        "scenario",
        "baseline_test_rmse",
        "orthogonal_test_rmse",
        "delta_test_rmse",
        "baseline_test_r2",
        "orthogonal_test_r2",
        "baseline_exact_term_recovery_score",
        "orthogonal_exact_term_recovery_score",
        "baseline_phase_equivalent_term_recovery_score",
        "orthogonal_phase_equivalent_term_recovery_score",
        "baseline_family_level_term_recovery_score",
        "orthogonal_family_level_term_recovery_score",
        "orthogonal_orthogonality_score",
    ]
    lines = [
        "# Known Relation Benchmark Table",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run all registered known-relation symbolic benchmarks and aggregate baseline vs "
            "orthogonal results into a formal benchmark table."
        ),
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(known_relation_benchmark_keys()),
        help="Comma-separated scenario keys. Defaults to all registered known-relation benchmarks.",
    )
    parser.add_argument("--db-path", type=str, default=str(ROOT / "runs" / "known_relation_benchmark_suite.sqlite3"))
    parser.add_argument("--namespace", type=str, default="known_relation_benchmark_suite")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(ROOT / "examples" / "out" / "known_relation_symbolic_benchmark_suite"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-total", type=int, default=2400)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--noise-std", type=float, default=0.025)
    parser.add_argument("--baseline-max-added-terms", type=int, default=10)
    parser.add_argument("--baseline-topk-features", type=int, default=5)
    parser.add_argument("--orth-candidate-limit", type=int, default=160)
    parser.add_argument("--orth-group-count", type=int, default=16)
    parser.add_argument("--orth-min-basis-count", type=int, default=3)
    parser.add_argument("--orth-max-basis-count", type=int, default=8)
    parser.add_argument("--orth-selection-mode", type=str, default="rmse_first")
    parser.add_argument("--orth-assembler-max-added-terms", type=int, default=4)
    parser.add_argument("--orth-assembler-topk-features", type=int, default=4)
    parser.add_argument("--orth-assembler-max-pair-terms", type=int, default=8)
    parser.add_argument("--orth-assembler-max-candidates-per-iter", type=int, default=96)
    parser.add_argument("--orth-assembler-candidate-keep-top", type=int, default=6)
    parser.add_argument("--orth-assembler-max-expr-depth", type=int, default=6)
    parser.add_argument("--orth-assembler-ridge-l2", type=float, default=1e-4)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    scenarios = [
        str(item).strip()
        for item in str(args.scenarios or "").split(",")
        if str(item).strip()
    ]
    invalid = [scenario for scenario in scenarios if scenario not in set(known_relation_benchmark_keys())]
    if invalid:
        choices = ", ".join(known_relation_benchmark_keys())
        raise SystemExit(f"Unknown scenarios: {', '.join(invalid)}. Expected one of: {choices}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_root = Path(args.output_root).resolve() / stamp
    suite_root.mkdir(parents=True, exist_ok=True)

    scenario_results: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    execution_logs: list[dict[str, Any]] = []
    for scenario in scenarios:
        summary, output = _run_scenario(
            scenario=scenario,
            db_path=str(args.db_path),
            namespace=str(args.namespace),
            output_root=suite_root,
            seed=int(args.seed),
            n_total=int(args.n_total),
            train_ratio=float(args.train_ratio),
            noise_std=float(args.noise_std),
            baseline_max_added_terms=int(args.baseline_max_added_terms),
            baseline_topk_features=int(args.baseline_topk_features),
            orth_candidate_limit=int(args.orth_candidate_limit),
            orth_group_count=int(args.orth_group_count),
            orth_min_basis_count=int(args.orth_min_basis_count),
            orth_max_basis_count=int(args.orth_max_basis_count),
            orth_selection_mode=str(args.orth_selection_mode),
            orth_assembler_max_added_terms=int(args.orth_assembler_max_added_terms),
            orth_assembler_topk_features=int(args.orth_assembler_topk_features),
            orth_assembler_max_pair_terms=int(args.orth_assembler_max_pair_terms),
            orth_assembler_max_candidates_per_iter=int(args.orth_assembler_max_candidates_per_iter),
            orth_assembler_candidate_keep_top=int(args.orth_assembler_candidate_keep_top),
            orth_assembler_max_expr_depth=int(args.orth_assembler_max_expr_depth),
            orth_assembler_ridge_l2=float(args.orth_assembler_ridge_l2),
        )
        summary["summary_path"] = str(Path(summary.get("output_root", suite_root)) / "summary.json")
        scenario_results.append(summary)
        benchmark_rows.append(_summary_row(summary))
        execution_logs.append({"scenario": scenario, "output": output})
        print(f"[suite] completed {scenario}")

    suite_summary = {
        "generated_at": datetime.now().isoformat(),
        "db_path": str(args.db_path),
        "namespace": str(args.namespace),
        "suite_root": str(suite_root),
        "scenarios": list(scenarios),
        "config": {
            "seed": int(args.seed),
            "n_total": int(args.n_total),
            "train_ratio": float(args.train_ratio),
            "noise_std": float(args.noise_std),
            "baseline_max_added_terms": int(args.baseline_max_added_terms),
            "baseline_topk_features": int(args.baseline_topk_features),
            "orth_candidate_limit": int(args.orth_candidate_limit),
            "orth_group_count": int(args.orth_group_count),
            "orth_min_basis_count": int(args.orth_min_basis_count),
            "orth_max_basis_count": int(args.orth_max_basis_count),
            "orth_selection_mode": str(args.orth_selection_mode),
            "orth_assembler_max_added_terms": int(args.orth_assembler_max_added_terms),
            "orth_assembler_topk_features": int(args.orth_assembler_topk_features),
            "orth_assembler_max_pair_terms": int(args.orth_assembler_max_pair_terms),
            "orth_assembler_max_candidates_per_iter": int(args.orth_assembler_max_candidates_per_iter),
            "orth_assembler_candidate_keep_top": int(args.orth_assembler_candidate_keep_top),
            "orth_assembler_max_expr_depth": int(args.orth_assembler_max_expr_depth),
            "orth_assembler_ridge_l2": float(args.orth_assembler_ridge_l2),
        },
        "benchmark_rows": benchmark_rows,
        "scenario_summaries": scenario_results,
        "execution_logs": execution_logs,
    }

    summary_json_path = suite_root / "benchmark_suite_summary.json"
    summary_csv_path = suite_root / "benchmark_suite_table.csv"
    summary_md_path = suite_root / "benchmark_suite_table.md"
    summary_json_path.write_text(json.dumps(_jsonable(suite_summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(summary_csv_path, benchmark_rows)
    _write_markdown(summary_md_path, benchmark_rows)

    print("KNOWN RELATION BENCHMARK SUITE")
    print(f"suite_root={suite_root}")
    print(f"summary_json={summary_json_path}")
    print(f"summary_csv={summary_csv_path}")
    print(f"summary_md={summary_md_path}")


if __name__ == "__main__":
    main()
