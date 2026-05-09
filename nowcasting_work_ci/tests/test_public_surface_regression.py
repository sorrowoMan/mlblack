from __future__ import annotations

import sys
from pathlib import Path


DESKTOP = Path(__file__).resolve().parents[3]
MLBLACK_ROOT = DESKTOP / "mlblack"
if str(MLBLACK_ROOT) not in sys.path:
    sys.path.insert(0, str(MLBLACK_ROOT))

from nowcasting_work_ci import aggregate_and_plot_results as top_reporting
from nowcasting_work_ci import assembly, build_solver, run, run_deterministic_smoke_regression, run_solver
from nowcasting_work_ci import run_nowcasting_orthogonal_symbolic_work_ci as top_orthogonal_symbolic
from nowcasting_work_ci import run_nowcasting_symbolic_subset_bridge_work_ci as top_symbolic
from nowcasting_work_ci import (
    run_nowcasting_symbolic_subset_bridge_work_ci_native_interval as top_symbolic_interval,
)
from nowcasting_work_ci.compat import run_nowcasting_symbolic_subset_bridge_work_ci as compat_symbolic
from nowcasting_work_ci.compat import (
    run_nowcasting_symbolic_subset_bridge_work_ci_native_interval as compat_symbolic_interval,
)
from nowcasting_work_ci.nsgablack_side.build_solver import (
    NowcastingSolverBuildConfig as SideNowcastingSolverBuildConfig,
)
from nowcasting_work_ci.nsgablack_side.build_solver import build_nowcasting_solver as side_build_nowcasting_solver
from nowcasting_work_ci.tools import aggregate_and_plot_results as reporting_tool
from nowcasting_work_ci.tools import run_deterministic_smoke_regression as smoke_tool


STABLE_PUBLIC_SURFACE = {
    "assembly.py",
    "build_solver.py",
    "run.py",
}

EXPERIMENT_PUBLIC_SURFACE = {
    "run_nowcasting_orthogonal_symbolic_work_ci.py",
}

DEPRECATED_PUBLIC_SURFACE = {
    "aggregate_and_plot_results.py",
    "run_deterministic_smoke_regression.py",
    "run_nowcasting_symbolic_subset_bridge_work_ci.py",
    "run_nowcasting_symbolic_subset_bridge_work_ci_native_interval.py",
    "run_solver.py",
}

PACKAGE_BOOTSTRAP_FILES = {"__init__.py"}


def test_public_surface_file_sets_stay_stable() -> None:
    package_root = MLBLACK_ROOT / "nowcasting_work_ci"
    top_level_python = {p.name for p in package_root.iterdir() if p.is_file() and p.suffix == ".py"}

    expected = STABLE_PUBLIC_SURFACE | EXPERIMENT_PUBLIC_SURFACE | DEPRECATED_PUBLIC_SURFACE | PACKAGE_BOOTSTRAP_FILES
    assert top_level_python == expected


def test_stable_import_surface_targets_stay_pinned() -> None:
    assert callable(run.main)
    assert callable(assembly.run)
    assert build_solver.NowcastingSolverBuildConfig is SideNowcastingSolverBuildConfig
    assert build_solver.build_nowcasting_solver is side_build_nowcasting_solver


def test_deprecated_shims_forward_to_supported_targets() -> None:
    assert run_solver.run_main is run.main
    assert top_reporting.main is reporting_tool.main
    assert run_deterministic_smoke_regression.main is smoke_tool.main
    assert top_symbolic.main is compat_symbolic.main
    assert top_symbolic_interval.main is compat_symbolic_interval.main


def test_experiment_entry_surface_is_importable() -> None:
    assert callable(top_orthogonal_symbolic.main)
