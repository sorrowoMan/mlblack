from __future__ import annotations

from mlblack.project.doctor import _check_project_root, _iter_case_roots


def test_doctor_ignores_cache_only_case_residue(tmp_path) -> None:
    project_root = tmp_path / "example"
    cases_dir = project_root / "cases"
    stale = cases_dir / "removed_case"
    (stale / "pipeline" / "__pycache__").mkdir(parents=True)
    (stale / "pipeline" / "__pycache__" / "main.cpython-313.pyc").write_bytes(b"cache")
    cases_dir.joinpath("__init__.py").write_text("", encoding="utf-8")

    diagnostics = []
    _check_project_root(project_root, diagnostics)

    assert not any(item.code == "case-package-missing-init" for item in diagnostics)

    examples_cases = tmp_path / "examples" / "cases"
    cached_case = examples_cases / "removed" / "cases" / "removed"
    (cached_case / "problem" / "__pycache__").mkdir(parents=True)
    (cached_case / "problem" / "__pycache__" / "problem.cpython-313.pyc").write_bytes(b"cache")

    assert tuple(_iter_case_roots(examples_cases)) == ()
