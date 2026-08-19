from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


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


def test_traffic_project_build_check_executes_cli_contracts_with_l0_grants() -> None:
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
    assert completed.stdout.count("[check]") == 6
    for case_name in (
        "arimax_factor_attribution",
        "gam_linearity_check",
        "granger_causality_check",
        "shap_contribution_check",
        "symbolic_regression",
        "xgboost_baseline",
    ):
        assert f"diagnostics.{case_name}" in completed.stdout


def test_temporal_project_build_check_exposes_seven_independent_trainers() -> None:
    project_entry = ROOT / "examples" / "cases" / "temporal_neural_compare" / "run_project.py"
    completed = subprocess.run(
        [sys.executable, str(project_entry), "--check", "--build-check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        # Seven isolated Windows CLI processes each cold-import the selected
        # neural backend. Keep this a correctness check rather than a host
        # startup-speed benchmark.
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "cli_check_unavailable" not in completed.stdout
    assert completed.stdout.count("[check]") == 7
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
