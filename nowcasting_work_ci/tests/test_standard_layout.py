from __future__ import annotations

import sys
from pathlib import Path


DESKTOP = Path(__file__).resolve().parents[3]
MLBLACK_ROOT = DESKTOP / "mlblack"
if str(MLBLACK_ROOT) not in sys.path:
    sys.path.insert(0, str(MLBLACK_ROOT))

from nowcasting_work_ci.mlblack_side.config import build_output_root, build_runs_root
from nowcasting_work_ci._internal.repo_paths import default_repo_roots


def test_standard_runs_root_is_outside_package_root() -> None:
    project_root = Path(MLBLACK_ROOT)
    package_root = project_root / "nowcasting_work_ci"

    runs_root = build_runs_root(project_root)
    assert runs_root == project_root / "_scenario_runs" / "nowcasting_work_ci"
    assert package_root not in runs_root.parents


def test_output_root_uses_standard_runs_root() -> None:
    project_root = Path(MLBLACK_ROOT)
    package_root = project_root / "nowcasting_work_ci"

    out_root = build_output_root(project_root, seed=42, stamp="20260429_000000")

    assert out_root.parent == project_root / "_scenario_runs" / "nowcasting_work_ci"
    assert out_root.name == "nowcasting_symbolic_subset_bridge_work_ci_seed42_20260429_000000"
    assert package_root not in out_root.parents


def test_top_level_layout_moves_docs_under_docs_dir() -> None:
    package_root = Path(MLBLACK_ROOT) / "nowcasting_work_ci"
    top_names = {p.name for p in package_root.iterdir()}

    assert "docs" in top_names
    assert "STANDARD_LAYOUT.md" in top_names
    assert "README_RUNTIME_CONTRACTS.md" not in top_names
    assert "README_REPORTING.md" not in top_names
    assert "README_ARCH_SPLIT.md" not in top_names

    docs_dir = package_root / "docs"
    assert (docs_dir / "README_RUNTIME_CONTRACTS.md").exists()
    assert (docs_dir / "README_REPORTING.md").exists()
    assert (docs_dir / "README_ARCH_SPLIT.md").exists()


def test_internal_helpers_are_package_private_and_repo_roots_are_correct() -> None:
    package_root = Path(MLBLACK_ROOT) / "nowcasting_work_ci"

    assert (package_root / "_internal").exists()
    assert (package_root / "_internal" / "repo_paths.py").exists()
    assert not (package_root / "_repo_paths.py").exists()

    mlblack_root, nsgablack_root = default_repo_roots()
    assert mlblack_root == Path(MLBLACK_ROOT)
    assert nsgablack_root == DESKTOP / "nsgablack"
