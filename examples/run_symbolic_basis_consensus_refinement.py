from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.symbolic import annotate_basis_entries, build_core_basis_tables, select_locked_core_seed_genome
from my_project.known_relation_symbolic.orchestration import build_known_relation_semantic_flow_spec
from my_project.known_relation_symbolic.pipeline import build_known_relation_bundle
from my_project.known_relation_symbolic.problem import known_relation_benchmark_keys
from training import TrainerState, TrainingInit
from workflow import TrainDataBundle, run_semantic_train_flow


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(raw) for key, raw in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return str(value)


def _artifact_expression(artifact: Any) -> str | None:
    if not hasattr(artifact, "expression"):
        return None
    try:
        return str(artifact.expression(target_index=0, precision=8, use_feature_names=True))
    except Exception:
        return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _table_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6f}"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _write_csv_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_tuple = tuple(dict(row) for row in tuple(rows))
    fieldnames: list[str] = []
    for row in rows_tuple:
        for key in row.keys():
            if str(key) not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_tuple:
            writer.writerow({key: _table_scalar(row.get(key)) for key in fieldnames})


def _write_markdown_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_tuple = tuple(dict(row) for row in tuple(rows))
    if not rows_tuple:
        path.write_text("_No rows._\n", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows_tuple:
        for key in row.keys():
            if str(key) not in fieldnames:
                fieldnames.append(str(key))
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows_tuple:
        lines.append("| " + " | ".join(_table_scalar(row.get(key)).replace("\n", " ") for key in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _orthogonal_params(
    *,
    stamp: str,
    args: argparse.Namespace,
    gate_feature_names: Sequence[str],
    enable_piecewise_basis: bool,
    search_seed: int,
    lock_seed_basis: bool,
) -> dict[str, Any]:
    return {
        "parameter_backend": "ridge",
        "task": "point",
        "structure_engine": {
            "structure_mode": "orthogonal_basis_search",
            "search_driver": "orthogonal_basis",
            "dynamic_pool_enabled": True,
            "metadata": {"supports_piecewise_basis": bool(enable_piecewise_basis)},
        },
        "artifact_id": f"known_relation_symbolic_consensus_{stamp}",
        "candidate_limit": int(args.orth_candidate_limit),
        "group_count": int(args.orth_group_count),
        "seed_candidate_count": int(args.orth_seed_candidate_count),
        "min_basis_count": int(args.orth_min_basis_count),
        "max_basis_count": int(args.orth_max_basis_count),
        "selection_mode": str(args.orth_selection_mode),
        "random_seed": int(search_seed),
        "greedy_choice_topk": int(args.greedy_choice_topk),
        "random_group_trials": int(args.random_group_trials),
        "lock_seed_basis": bool(lock_seed_basis),
        "enable_piecewise_basis": bool(enable_piecewise_basis),
        "gate_feature_names": tuple(str(value) for value in tuple(gate_feature_names)),
        "gate_quantiles": tuple(float(value) for value in tuple(args.gate_quantiles)),
        "assembler_max_added_terms": int(args.orth_assembler_max_added_terms),
        "assembler_topk_features": int(args.orth_assembler_topk_features),
        "assembler_max_pair_terms": int(args.orth_assembler_max_pair_terms),
        "assembler_max_candidates_per_iter": int(args.orth_assembler_max_candidates_per_iter),
        "assembler_candidate_keep_top": int(args.orth_assembler_candidate_keep_top),
        "assembler_max_expr_depth": int(args.orth_assembler_max_expr_depth),
        "assembler_ridge_l2": float(args.orth_assembler_ridge_l2),
        "search_graph_cache_enabled": False,
    }


def _basis_rows_from_artifact(artifact: Any) -> list[dict[str, Any]]:
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    schema = dict(metadata.get("symbolic_artifact_schema", {}) or {})
    basis_structure = dict(schema.get("basis_structure", {}) or {})
    basis_semantics = dict(basis_structure.get("basis_semantics", {}) or {})
    recorded = dict(basis_semantics.get("recorded", {}) or {})
    basis_terms = recorded.get("basis_terms")
    if isinstance(basis_terms, Sequence) and not isinstance(basis_terms, (str, bytes, bytearray)):
        return [dict(row) for row in basis_terms if isinstance(row, Mapping)]
    selected_basis = metadata.get("selected_basis")
    if isinstance(selected_basis, Sequence) and not isinstance(selected_basis, (str, bytes, bytearray)):
        return [dict(row) for row in selected_basis if isinstance(row, Mapping)]
    symbolic_selected_basis = dict(metadata.get("symbolic", {}) or {}).get("selected_basis")
    if isinstance(symbolic_selected_basis, Sequence) and not isinstance(symbolic_selected_basis, (str, bytes, bytearray)):
        return [dict(row) for row in symbolic_selected_basis if isinstance(row, Mapping)]
    return []


def _outer_basis_genome_from_artifact(artifact: Any) -> tuple[dict[str, Any], ...]:
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    raw = metadata.get("orthogonal_outer_basis_genome")
    if not (isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray))):
        raw = dict(metadata.get("symbolic", {}) or {}).get("orthogonal_outer_basis_genome")
    if not (isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray))):
        return tuple()
    return tuple(dict(term) for term in tuple(raw) if isinstance(term, Mapping))


def _artifact_run_summary(
    *,
    artifact: Any,
    metrics: Mapping[str, Any],
    tracker: Mapping[str, Any],
    output_dir: Path,
    run_index: int,
    search_seed: int,
    phase: str,
) -> dict[str, Any]:
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    schema = dict(metadata.get("symbolic_artifact_schema", {}) or {})
    basis_structure = dict(schema.get("basis_structure", {}) or {})
    orthogonality_status = dict(basis_structure.get("orthogonality_status", {}) or {})
    truth_recovery = dict(schema.get("truth_contract_recovery", {}) or {})
    outer_objective = dict(schema.get("orthogonal_search_objective", {}) or metadata.get("orthogonal_search_objective", {}) or {})
    search_summary = dict(metadata.get("search", {}) or {})
    basis_rows = _basis_rows_from_artifact(artifact)
    outer_basis_genome = _outer_basis_genome_from_artifact(artifact)
    basis_entries = annotate_basis_entries(basis_rows, outer_basis_genome)
    return {
        "phase": str(phase),
        "run_index": int(run_index),
        "search_seed": int(search_seed),
        "run_id": str(tracker.get("run_id") or ""),
        "output_dir": str(output_dir),
        "artifact_id": str(getattr(artifact, "artifact_id", "")),
        "final_expression": _artifact_expression(artifact),
        "metrics": _jsonable(dict(metrics)),
        "test_rmse": dict(metrics.get("test", {}) or {}).get("rmse"),
        "test_r2": dict(metrics.get("test", {}) or {}).get("r2"),
        "orthogonality_score": orthogonality_status.get("orthogonality_score"),
        "pair_abs_corr_mean": orthogonality_status.get("pair_abs_corr_mean"),
        "residual_gain_mean": orthogonality_status.get("residual_gain_mean"),
        "semantic_unique_ratio": orthogonality_status.get("semantic_unique_ratio"),
        "outer_objective_score": outer_objective.get("outer_score"),
        "inner_fit_score": outer_objective.get("inner_fit_score"),
        "exact_basis_hit_score": truth_recovery.get("exact_basis_hit_score"),
        "exact_term_recovery_score": truth_recovery.get("exact_term_recovery_score"),
        "phase_equivalent_term_recovery_score": truth_recovery.get("phase_equivalent_term_recovery_score"),
        "family_level_term_recovery_score": truth_recovery.get("family_level_term_recovery_score"),
        "truth_recovery": _jsonable(truth_recovery),
        "basis_rows": _jsonable(basis_rows),
        "basis_entries": _jsonable(basis_entries),
        "outer_basis_genome": _jsonable(outer_basis_genome),
        "search_summary": _jsonable(search_summary),
        "tracker": _jsonable(dict(tracker)),
    }


def _run_flow_once(
    *,
    bundle: TrainDataBundle,
    trainer_params: Mapping[str, Any],
    training_init: TrainingInit | None,
    run_name: str,
    output_dir: Path,
    db_path: str,
    namespace: str,
    tag: str,
    run_index: int,
    search_seed: int,
    phase: str,
) -> dict[str, Any]:
    spec = build_known_relation_semantic_flow_spec(
        trainer_params=trainer_params,
        run_name=str(run_name),
        output_dir=str(output_dir),
        db_path=str(db_path),
        namespace=str(namespace),
        tag=str(tag),
        training_init=training_init,
    )
    result = run_semantic_train_flow(bundle, spec=spec)
    tracker = dict(result.report.get("experiment_tracker", {}) or {})
    summary = _artifact_run_summary(
        artifact=result.artifact,
        metrics=dict(result.metrics),
        tracker=tracker,
        output_dir=Path(result.output_dir),
        run_index=run_index,
        search_seed=search_seed,
        phase=phase,
    )
    return summary


def _build_consensus_seed_state(
    *,
    seed_genome: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    target_names: Sequence[str],
    equivalence_mode: str,
) -> TrainerState:
    return TrainerState(
        trainer_name="symbolic_orthogonal",
        payload={
            "schema_version": 1,
            "trainer_name": "symbolic_orthogonal",
            "search_completed": True,
            "genome": tuple(dict(term) for term in tuple(seed_genome)),
            "assembled_genome": tuple(dict(term) for term in tuple(seed_genome)),
            "parameter_values": {},
            "readout_weight": np.zeros((max(1, len(tuple(seed_genome))), 1), dtype=float),
            "readout_bias": np.zeros((1,), dtype=float),
            "residual_std": np.ones((1,), dtype=float),
            "feature_names": tuple(str(value) for value in tuple(feature_names)),
            "target_names": tuple(str(value) for value in tuple(target_names)),
            "search_summary": {
                "protocol": "multi_run_consensus_locked_core",
                "equivalence_mode": str(equivalence_mode),
            },
            "seed_protocol": "consensus_locked_core",
        },
        metadata={
            "resume_source": "consensus_locked_core",
            "consensus_equivalence_mode": str(equivalence_mode),
        },
    )


def _metric_float(value: Any, *, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return float(numeric)


def _best_run(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    rows = [dict(row) for row in tuple(runs)]
    if not rows:
        return None
    rows.sort(
        key=lambda row: (
            -_metric_float(row.get("exact_term_recovery_score"), default=-1.0),
            -_metric_float(row.get("phase_equivalent_term_recovery_score"), default=-1.0),
            -_metric_float(row.get("family_level_term_recovery_score"), default=-1.0),
            -_metric_float(row.get("outer_objective_score"), default=-1.0),
            _metric_float(row.get("test_rmse"), default=float("inf")),
            -_metric_float(row.get("test_r2"), default=-1.0),
        )
    )
    return rows[0]


def _mean_metric(runs: Sequence[Mapping[str, Any]], field_name: str) -> float | None:
    values = [
        _metric_float(row.get(field_name), default=float("nan"))
        for row in tuple(runs)
        if math.isfinite(_metric_float(row.get(field_name), default=float("nan")))
    ]
    return None if not values else float(sum(values) / float(len(values)))


def _comparison_row(
    *,
    scenario: str,
    vanilla_runs: Sequence[Mapping[str, Any]],
    locked_runs: Sequence[Mapping[str, Any]],
    core_selection: Mapping[str, Any],
) -> dict[str, Any]:
    vanilla_best = _best_run(vanilla_runs) or {}
    locked_best = _best_run(locked_runs) or {}
    return {
        "scenario": str(scenario),
        "core_equivalence_mode": str(core_selection.get("equivalence_mode") or ""),
        "core_basis_count": int(len(tuple(core_selection.get("selected_core_rows", ()) or ()))),
        "locked_seed_terms": int(len(tuple(core_selection.get("seed_genome", ()) or ()))),
        "vanilla_run_count": int(len(tuple(vanilla_runs))),
        "locked_run_count": int(len(tuple(locked_runs))),
        "vanilla_best_test_rmse": vanilla_best.get("test_rmse"),
        "locked_best_test_rmse": locked_best.get("test_rmse"),
        "delta_best_test_rmse": (
            None
            if vanilla_best.get("test_rmse") is None or locked_best.get("test_rmse") is None
            else float(locked_best["test_rmse"]) - float(vanilla_best["test_rmse"])
        ),
        "vanilla_best_exact_term_recovery_score": vanilla_best.get("exact_term_recovery_score"),
        "locked_best_exact_term_recovery_score": locked_best.get("exact_term_recovery_score"),
        "vanilla_best_phase_term_recovery_score": vanilla_best.get("phase_equivalent_term_recovery_score"),
        "locked_best_phase_term_recovery_score": locked_best.get("phase_equivalent_term_recovery_score"),
        "vanilla_best_family_term_recovery_score": vanilla_best.get("family_level_term_recovery_score"),
        "locked_best_family_term_recovery_score": locked_best.get("family_level_term_recovery_score"),
        "vanilla_best_outer_objective_score": vanilla_best.get("outer_objective_score"),
        "locked_best_outer_objective_score": locked_best.get("outer_objective_score"),
        "vanilla_mean_test_rmse": _mean_metric(vanilla_runs, "test_rmse"),
        "locked_mean_test_rmse": _mean_metric(locked_runs, "test_rmse"),
        "vanilla_mean_exact_term_recovery_score": _mean_metric(vanilla_runs, "exact_term_recovery_score"),
        "locked_mean_exact_term_recovery_score": _mean_metric(locked_runs, "exact_term_recovery_score"),
        "vanilla_mean_outer_objective_score": _mean_metric(vanilla_runs, "outer_objective_score"),
        "locked_mean_outer_objective_score": _mean_metric(locked_runs, "outer_objective_score"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated orthogonal symbolic searches on fixed known-relation datasets, "
            "aggregate a core basis consensus table, and validate locked-core refinement runs."
        ),
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=known_relation_benchmark_keys(),
        default=("arrhenius_gate_like", "redundant_proxy_control"),
    )
    parser.add_argument("--db-path", type=str, default=str(ROOT / "runs" / "known_relation_symbolic_consensus.sqlite3"))
    parser.add_argument("--namespace", type=str, default="known_relation_symbolic_consensus")
    parser.add_argument("--tag", type=str, default="consensus")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(ROOT / "examples" / "out" / "known_relation_symbolic_consensus"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-total", type=int, default=2800)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--noise-std", type=float, default=0.025)
    parser.add_argument("--vanilla-runs", type=int, default=8)
    parser.add_argument("--locked-runs", type=int, default=3)
    parser.add_argument("--search-seed-base", type=int, default=100)
    parser.add_argument("--locked-search-seed-base", type=int, default=900)
    parser.add_argument("--core-equivalence-mode", type=str, choices=("strict", "phase", "family"), default="family")
    parser.add_argument("--core-min-support-rate", type=float, default=0.6)
    parser.add_argument("--core-min-support-count", type=int, default=0)
    parser.add_argument("--core-max-terms", type=int, default=4)
    parser.add_argument("--orth-candidate-limit", type=int, default=220)
    parser.add_argument("--orth-group-count", type=int, default=20)
    parser.add_argument("--orth-seed-candidate-count", type=int, default=24)
    parser.add_argument("--orth-min-basis-count", type=int, default=3)
    parser.add_argument("--orth-max-basis-count", type=int, default=8)
    parser.add_argument("--orth-selection-mode", type=str, default="interval_first")
    parser.add_argument("--greedy-choice-topk", type=int, default=5)
    parser.add_argument("--random-group-trials", type=int, default=12)
    parser.add_argument("--gate-quantiles", nargs="+", type=float, default=(0.25, 0.50, 0.75))
    parser.add_argument("--orth-assembler-max-added-terms", type=int, default=5)
    parser.add_argument("--orth-assembler-topk-features", type=int, default=5)
    parser.add_argument("--orth-assembler-max-pair-terms", type=int, default=10)
    parser.add_argument("--orth-assembler-max-candidates-per-iter", type=int, default=128)
    parser.add_argument("--orth-assembler-candidate-keep-top", type=int, default=8)
    parser.add_argument("--orth-assembler-max-expr-depth", type=int, default=7)
    parser.add_argument("--orth-assembler-ridge-l2", type=float, default=1e-4)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).resolve() / stamp
    output_root.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db_path).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    overall_rows: list[dict[str, Any]] = []
    overall_summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "db_path": str(db_path),
        "namespace": str(args.namespace),
        "output_root": str(output_root),
        "scenarios": [],
    }

    for scenario_index, scenario in enumerate(tuple(args.scenarios)):
        scenario_dir = output_root / str(scenario)
        scenario_dir.mkdir(parents=True, exist_ok=True)
        benchmark, bundle, truth_payload = build_known_relation_bundle(
            benchmark_key=str(scenario),
            n_total=int(args.n_total),
            train_ratio=float(args.train_ratio),
            noise_std=float(args.noise_std),
            seed=int(args.seed),
        )
        search_hints = dict(bundle.metadata.get("search_hints", {}) or {})
        gate_feature_names = tuple(str(value) for value in tuple(search_hints.get("gate_feature_names", ()) or ()))
        enable_piecewise_basis = bool(search_hints.get("enable_piecewise_basis"))

        vanilla_runs: list[dict[str, Any]] = []
        for run_offset in range(int(args.vanilla_runs)):
            search_seed = int(args.search_seed_base) + int(scenario_index) * 1000 + int(run_offset)
            run_summary = _run_flow_once(
                bundle=bundle,
                trainer_params=_orthogonal_params(
                    stamp=stamp,
                    args=args,
                    gate_feature_names=gate_feature_names,
                    enable_piecewise_basis=enable_piecewise_basis,
                    search_seed=search_seed,
                    lock_seed_basis=False,
                ),
                training_init=TrainingInit(mode="fresh"),
                run_name=f"{args.namespace}_{scenario}_orthogonal_{run_offset:02d}",
                output_dir=scenario_dir / "orthogonal_runs" / f"run_{run_offset:02d}",
                db_path=str(db_path),
                namespace=str(args.namespace),
                tag=f"{args.tag}:{scenario}:orthogonal:{run_offset:02d}",
                run_index=run_offset,
                search_seed=search_seed,
                phase="orthogonal",
            )
            vanilla_runs.append(run_summary)

        core_tables = build_core_basis_tables(
            runs=vanilla_runs,
            min_support_count=None if int(args.core_min_support_count) <= 0 else int(args.core_min_support_count),
            min_support_rate=float(args.core_min_support_rate),
        )
        for mode, rows in core_tables.items():
            _write_json(scenario_dir / f"core_basis_table.{mode}.json", {"rows": rows})
            _write_csv_table(scenario_dir / f"core_basis_table.{mode}.csv", rows)
            _write_markdown_table(scenario_dir / f"core_basis_table.{mode}.md", rows)

        core_selection = select_locked_core_seed_genome(
            runs=vanilla_runs,
            equivalence_mode=str(args.core_equivalence_mode),
            min_support_count=None if int(args.core_min_support_count) <= 0 else int(args.core_min_support_count),
            min_support_rate=float(args.core_min_support_rate),
            max_terms=int(args.core_max_terms),
        )
        _write_json(scenario_dir / "locked_core_selection.json", core_selection)

        locked_runs: list[dict[str, Any]] = []
        seed_genome = tuple(core_selection.get("seed_genome", ()) or ())
        if seed_genome:
            parent_state = _build_consensus_seed_state(
                seed_genome=seed_genome,
                feature_names=bundle.train.feature_names,
                target_names=bundle.train.target_names,
                equivalence_mode=str(args.core_equivalence_mode),
            )
            for run_offset in range(int(args.locked_runs)):
                search_seed = int(args.locked_search_seed_base) + int(scenario_index) * 1000 + int(run_offset)
                run_summary = _run_flow_once(
                    bundle=bundle,
                    trainer_params=_orthogonal_params(
                        stamp=stamp,
                        args=args,
                        gate_feature_names=gate_feature_names,
                        enable_piecewise_basis=enable_piecewise_basis,
                        search_seed=search_seed,
                        lock_seed_basis=True,
                    ),
                    training_init=TrainingInit(
                        mode="warm_start",
                        parent_state=parent_state,
                        metadata={
                            "consensus_equivalence_mode": str(args.core_equivalence_mode),
                            "locked_core_terms": int(len(seed_genome)),
                        },
                    ),
                    run_name=f"{args.namespace}_{scenario}_locked_{run_offset:02d}",
                    output_dir=scenario_dir / "locked_core_runs" / f"run_{run_offset:02d}",
                    db_path=str(db_path),
                    namespace=str(args.namespace),
                    tag=f"{args.tag}:{scenario}:locked:{run_offset:02d}",
                    run_index=run_offset,
                    search_seed=search_seed,
                    phase="locked_core",
                )
                locked_runs.append(run_summary)

        comparison = _comparison_row(
            scenario=str(scenario),
            vanilla_runs=vanilla_runs,
            locked_runs=locked_runs,
            core_selection=core_selection,
        )
        overall_rows.append(comparison)
        scenario_summary = {
            "scenario": str(benchmark.key),
            "description": str(benchmark.description),
            "dataset": {
                "n_total": int(args.n_total),
                "train_ratio": float(args.train_ratio),
                "noise_std": float(args.noise_std),
                "seed": int(args.seed),
            },
            "truth": _jsonable(truth_payload),
            "vanilla_runs": _jsonable(vanilla_runs),
            "locked_runs": _jsonable(locked_runs),
            "core_tables": _jsonable(core_tables),
            "core_selection": _jsonable(core_selection),
            "comparison": _jsonable(comparison),
        }
        _write_json(scenario_dir / "scenario_summary.json", scenario_summary)
        overall_summary["scenarios"].append(scenario_summary)

    _write_json(output_root / "benchmark_summary.json", overall_summary)
    _write_csv_table(output_root / "benchmark_table.csv", overall_rows)
    _write_markdown_table(output_root / "benchmark_table.md", overall_rows)

    print("KNOWN RELATION CONSENSUS LOCKED-CORE REFINEMENT")
    print(f"summary={output_root / 'benchmark_summary.json'}")
    print(f"table={output_root / 'benchmark_table.csv'}")
    print(f"db_path={db_path}")
    for row in overall_rows:
        print(
            "{scenario}: core={core_terms} vanilla_rmse={vanilla_rmse} locked_rmse={locked_rmse} "
            "vanilla_exact={vanilla_exact} locked_exact={locked_exact} "
            "vanilla_outer={vanilla_outer} locked_outer={locked_outer}".format(
                scenario=str(row.get("scenario") or ""),
                core_terms=int(row.get("locked_seed_terms", 0) or 0),
                vanilla_rmse=_table_scalar(row.get("vanilla_best_test_rmse")),
                locked_rmse=_table_scalar(row.get("locked_best_test_rmse")),
                vanilla_exact=_table_scalar(row.get("vanilla_best_exact_term_recovery_score")),
                locked_exact=_table_scalar(row.get("locked_best_exact_term_recovery_score")),
                vanilla_outer=_table_scalar(row.get("vanilla_best_outer_objective_score")),
                locked_outer=_table_scalar(row.get("locked_best_outer_objective_score")),
            )
        )


if __name__ == "__main__":
    main()
