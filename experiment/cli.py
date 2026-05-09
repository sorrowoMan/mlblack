from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from core.experiment_db import experiment_db_config_info, resolve_experiment_db_target
from core.flow_experiment_tracker import (
    experiment_tracker_summary,
    list_experiment_artifact_catalog,
    list_experiment_run_catalog,
    show_experiment_artifact_catalog_entry,
    show_experiment_run_catalog_entry,
)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _resolved_db_target(raw: Any) -> str:
    return resolve_experiment_db_target(None if raw is None else str(raw))


def _cmd_summary(args: argparse.Namespace) -> int:
    print(_json_dumps(experiment_tracker_summary(_resolved_db_target(args.db))))
    return 0


def _cmd_list_runs(args: argparse.Namespace) -> int:
    rows = list_experiment_run_catalog(
        _resolved_db_target(args.db),
        status=args.status,
        trainer_name=args.trainer_name,
        surface_key=args.surface_key,
        family_ref=args.family_ref,
        assembly_signature=args.assembly_signature,
        regime_mode=args.regime_mode,
        basis_scope=args.basis_scope,
        assembler_mode=args.assembler_mode,
        piecewise_gate_status=args.piecewise_gate_status,
        orthogonality_status=args.orthogonality_status,
        residual_complementarity_status=args.residual_complementarity_status,
        semantic_dedup_status=args.semantic_dedup_status,
        has_fold_summary=True if bool(args.has_fold_summary) else None,
        max_rmse_std=args.max_rmse_std,
        max_coverage_error_mean=args.max_coverage_error_mean,
        min_exact_basis_hit_score=args.min_exact_basis_hit_score,
        min_exact_term_recovery_score=args.min_exact_term_recovery_score,
        min_outer_objective_score=args.min_outer_objective_score,
        limit=int(args.limit),
    )
    print(_json_dumps(rows))
    return 0


def _cmd_show_run(args: argparse.Namespace) -> int:
    row = show_experiment_run_catalog_entry(_resolved_db_target(args.db), run_id=str(args.run_id))
    print(_json_dumps(row or {}))
    return 0 if row is not None else 1


def _cmd_list_artifacts(args: argparse.Namespace) -> int:
    rows = list_experiment_artifact_catalog(
        _resolved_db_target(args.db),
        trainer_name=args.trainer_name,
        head_task=args.head_task,
        regime_mode=args.regime_mode,
        basis_scope=args.basis_scope,
        assembler_mode=args.assembler_mode,
        piecewise_gate_status=args.piecewise_gate_status,
        orthogonality_status=args.orthogonality_status,
        residual_complementarity_status=args.residual_complementarity_status,
        semantic_dedup_status=args.semantic_dedup_status,
        has_fold_summary=True if bool(args.has_fold_summary) else None,
        max_rmse_std=args.max_rmse_std,
        max_coverage_error_mean=args.max_coverage_error_mean,
        min_exact_basis_hit_score=args.min_exact_basis_hit_score,
        min_exact_term_recovery_score=args.min_exact_term_recovery_score,
        min_outer_objective_score=args.min_outer_objective_score,
        limit=int(args.limit),
    )
    print(_json_dumps(rows))
    return 0


def _cmd_show_artifact(args: argparse.Namespace) -> int:
    row = show_experiment_artifact_catalog_entry(
        _resolved_db_target(args.db),
        run_id=str(args.run_id),
        artifact_id=str(args.artifact_id),
    )
    print(_json_dumps(row or {}))
    return 0 if row is not None else 1


def _cmd_ui(args: argparse.Namespace) -> int:
    from .dashboard import build_streamlit_command

    command = build_streamlit_command(
        db_path=_resolved_db_target(args.db),
        limit=int(args.limit),
        host=args.host,
        port=args.port,
        headless=bool(args.headless),
    )
    proc = subprocess.run(command, check=False)
    return int(proc.returncode)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mlblack experiment surface")
    sub = parser.add_subparsers(dest="cmd", required=True)

    default_info = experiment_db_config_info()
    db_help = (
        "Optional experiment DB target. Accepts a sqlite path or postgresql://... URL. "
        f"Defaults to experiment/db.toml, env, catalog fallback, then local sqlite ({default_info.get('db_target') or 'runs/experiments.sqlite3'})."
    )

    p_summary = sub.add_parser("summary", help="Summarize experiment tracker tables")
    p_summary.add_argument("--db", default=None, help=db_help)
    p_summary.set_defaults(_fn=_cmd_summary)

    p_list_runs = sub.add_parser("list-runs", help="List run catalog rows as JSON")
    p_list_runs.add_argument("--db", default=None, help=db_help)
    p_list_runs.add_argument("--status", default=None, help="Optional run status filter")
    p_list_runs.add_argument("--trainer-name", default=None, help="Optional trainer_name filter")
    p_list_runs.add_argument("--surface-key", default=None, help="Optional surface_key filter")
    p_list_runs.add_argument("--family-ref", default=None, help="Optional family_ref filter")
    p_list_runs.add_argument("--assembly-signature", default=None, help="Optional assembly_signature filter")
    p_list_runs.add_argument("--regime-mode", default=None, help="Optional regime_structure.mode filter")
    p_list_runs.add_argument("--basis-scope", default=None, help="Optional basis_structure.basis_scope filter")
    p_list_runs.add_argument("--assembler-mode", default=None, help="Optional assembler_structure.assembler_mode filter")
    p_list_runs.add_argument("--piecewise-gate-status", default=None, help="Optional piecewise_gate_basis.status filter")
    p_list_runs.add_argument("--orthogonality-status", default=None, help="Optional basis_structure.orthogonality_status.status filter")
    p_list_runs.add_argument("--residual-complementarity-status", default=None, help="Optional basis_structure.residual_complementarity.status filter")
    p_list_runs.add_argument("--semantic-dedup-status", default=None, help="Optional basis_structure.semantic_deduplication.status filter")
    p_list_runs.add_argument("--has-fold-summary", action="store_true", help="Require fold_summary rows")
    p_list_runs.add_argument("--max-rmse-std", type=float, default=None, help="Optional rmse_std upper bound")
    p_list_runs.add_argument(
        "--max-coverage-error-mean",
        type=float,
        default=None,
        help="Optional coverage_error_mean upper bound",
    )
    p_list_runs.add_argument("--min-exact-basis-hit-score", type=float, default=None, help="Optional exact_basis_hit_score lower bound")
    p_list_runs.add_argument("--min-exact-term-recovery-score", type=float, default=None, help="Optional exact_term_recovery_score lower bound")
    p_list_runs.add_argument("--min-outer-objective-score", type=float, default=None, help="Optional outer_objective_score lower bound")
    p_list_runs.add_argument("--limit", type=int, default=50, help="Max rows to return")
    p_list_runs.set_defaults(_fn=_cmd_list_runs)

    p_show_run = sub.add_parser("show-run", help="Show one run catalog row as JSON")
    p_show_run.add_argument("--db", default=None, help=db_help)
    p_show_run.add_argument("--run-id", required=True, help="Run identifier")
    p_show_run.set_defaults(_fn=_cmd_show_run)

    p_list_artifacts = sub.add_parser("list-artifacts", help="List artifact catalog rows as JSON")
    p_list_artifacts.add_argument("--db", default=None, help=db_help)
    p_list_artifacts.add_argument("--trainer-name", default=None, help="Optional trainer_name filter")
    p_list_artifacts.add_argument("--head-task", default=None, help="Optional head_task filter")
    p_list_artifacts.add_argument("--regime-mode", default=None, help="Optional regime_structure.mode filter")
    p_list_artifacts.add_argument("--basis-scope", default=None, help="Optional basis_structure.basis_scope filter")
    p_list_artifacts.add_argument("--assembler-mode", default=None, help="Optional assembler_structure.assembler_mode filter")
    p_list_artifacts.add_argument("--piecewise-gate-status", default=None, help="Optional piecewise_gate_basis.status filter")
    p_list_artifacts.add_argument("--orthogonality-status", default=None, help="Optional basis_structure.orthogonality_status.status filter")
    p_list_artifacts.add_argument("--residual-complementarity-status", default=None, help="Optional basis_structure.residual_complementarity.status filter")
    p_list_artifacts.add_argument("--semantic-dedup-status", default=None, help="Optional basis_structure.semantic_deduplication.status filter")
    p_list_artifacts.add_argument("--has-fold-summary", action="store_true", help="Require fold_summary rows")
    p_list_artifacts.add_argument("--max-rmse-std", type=float, default=None, help="Optional rmse_std upper bound")
    p_list_artifacts.add_argument(
        "--max-coverage-error-mean",
        type=float,
        default=None,
        help="Optional coverage_error_mean upper bound",
    )
    p_list_artifacts.add_argument("--min-exact-basis-hit-score", type=float, default=None, help="Optional exact_basis_hit_score lower bound")
    p_list_artifacts.add_argument("--min-exact-term-recovery-score", type=float, default=None, help="Optional exact_term_recovery_score lower bound")
    p_list_artifacts.add_argument("--min-outer-objective-score", type=float, default=None, help="Optional outer_objective_score lower bound")
    p_list_artifacts.add_argument("--limit", type=int, default=50, help="Max rows to return")
    p_list_artifacts.set_defaults(_fn=_cmd_list_artifacts)

    p_show_artifact = sub.add_parser("show-artifact", help="Show one artifact catalog row as JSON")
    p_show_artifact.add_argument("--db", default=None, help=db_help)
    p_show_artifact.add_argument("--run-id", required=True, help="Run identifier")
    p_show_artifact.add_argument("--artifact-id", required=True, help="Artifact identifier")
    p_show_artifact.set_defaults(_fn=_cmd_show_artifact)

    p_ui = sub.add_parser("ui", help="Launch the experiment dashboard")
    p_ui.add_argument("--db", default=None, help=db_help)
    p_ui.add_argument("--limit", type=int, default=500, help="Max result rows to preload")
    p_ui.add_argument("--host", type=str, default=None, help="Optional Streamlit server address")
    p_ui.add_argument("--port", type=int, default=None, help="Optional Streamlit server port")
    p_ui.add_argument("--headless", action="store_true", help="Launch Streamlit without opening a browser window")
    p_ui.set_defaults(_fn=_cmd_ui)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args._fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
