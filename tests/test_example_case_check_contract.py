from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from blackbase.project import CaseRunResult, execute_project


ROOT = Path(__file__).resolve().parents[1]
RUN_ENTRIES = tuple(sorted((ROOT / "examples" / "cases").glob("*/cases/*/run_solver.py")))


def test_every_standard_case_cli_declares_build_check_contract() -> None:
    assert RUN_ENTRIES
    missing = [path for path in RUN_ENTRIES if "--check" not in path.read_text(encoding="utf-8")]
    assert missing == []


@pytest.mark.parametrize("entry", RUN_ENTRIES, ids=lambda path: path.parent.name)
def test_every_standard_case_build_check_is_side_effect_bounded(entry: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(entry), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "[check]" in completed.stdout


def test_traffic_project_build_check_executes_canonical_builders_with_l0_grants() -> None:
    project_entry = ROOT / "examples" / "cases" / "traffic_congestion" / "run_project.py"
    completed = subprocess.run(
        [sys.executable, str(project_entry), "--group", "default", "--check", "--build-check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "cli_check_unavailable" not in completed.stdout
    assert completed.stdout.count("[project-check]") == 6
    for case_name in (
        "arimax_factor_attribution",
        "gam_linearity_check",
        "granger_causality_check",
        "shap_contribution_check",
        "xgboost_baseline",
    ):
        assert f"diagnostics.{case_name}" in completed.stdout
    assert "symbolic_learning.symbolic_regression" in completed.stdout


def test_temporal_project_build_check_exposes_seven_independent_trainers() -> None:
    project_entry = ROOT / "examples" / "cases" / "temporal_neural_compare" / "run_project.py"
    completed = subprocess.run(
        [sys.executable, str(project_entry), "--check", "--build-check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        # Project build-check validates all canonical builders in one process;
        # normal runs still use the configured isolated CLI entries.
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "cli_check_unavailable" not in completed.stdout
    assert completed.stdout.count("[project-check]") == 7
    for case_name in (
        "temporal_lstm",
        "temporal_tcn",
        "temporal_transformer",
        "temporal_nbeats",
        "temporal_deepar",
        "temporal_patchtst",
        "temporal_tft",
    ):
        assert f"model_comparison.{case_name}" in completed.stdout
        assert f'"pipeline_variant": "{case_name}"' in completed.stdout


def test_cross_framework_example_executes_real_nested_trainer_cases() -> None:
    project_root = ROOT / "examples" / "cases" / "cross_framework"

    result = execute_project(project_root, record=False)

    assert result.ok
    assert len(result.case_results) == 1
    parent = result.case_results[0]
    child_audit = parent.metadata["runtime_audit"]["child_invocations"]
    children = tuple(CaseRunResult.from_dict(item) for item in child_audit["results"])
    assert len(children) == 2
    for child in children:
        assert child.ok
        assert child.identity.parent_case_run_id == parent.identity.case_run_id
        assert child.identity.root_run_id == parent.identity.root_run_id
        assert child.identity.depth == 1
        assert child.request.child_grant is not None
        assert child.request.child_grant.resources["threads"] == 1
        assert child.output["protocol_type"] == "blackbase.trainer_result"
        assert child.output["best_model"] is None
        assert child.output["best_model_ref"]["backend"] == "filesystem"


def test_benchmark_project_runs_independent_trainers_in_parallel_envelopes() -> None:
    project_root = ROOT / "examples" / "cases" / "benchmarks"

    result = execute_project(project_root, record=False)

    assert result.ok
    assert len(result.case_results) == 4
    assert {item.request.case_name for item in result.case_results} == {
        "benchmark_tiny_cnn_classification",
        "benchmark_tiny_gnn_classification",
        "benchmark_tiny_cnn_contrastive",
        "benchmark_tiny_transformer_lm",
    }
    for item in result.case_results:
        assert item.ok
        assert item.request.mode == "build"
        assert item.request.component_overrides == {"max_steps": 2}
        assert item.request.resource_context["grant"]["threads"] == 1
        assert item.output["protocol_type"] == "blackbase.trainer_result"
        runtime = item.output["report"]["optimization_runtime"]
        assert runtime["steps_executed"] == 2
