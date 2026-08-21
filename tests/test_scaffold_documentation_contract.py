from __future__ import annotations

from mlblack.project.doctor import _check_case_root_scaffold


def _case_root(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    (root / "README.md").write_text("# case\n", encoding="utf-8")
    (root / "build_solver.py").write_text(
        "def build_solver(*, resource_context=None, component_overrides=None):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (root / "run_solver.py").write_text("", encoding="utf-8")
    (root / "pipeline.py").write_text("", encoding="utf-8")
    return root


def test_doctor_requires_one_case_readme(tmp_path):
    root = _case_root(tmp_path)
    (root / "README.md").unlink()
    diagnostics = []

    _check_case_root_scaffold(root, diagnostics)

    assert "missing-case-readme" in {item.code for item in diagnostics}


def test_doctor_flags_legacy_case_document_copies(tmp_path):
    root = _case_root(tmp_path)
    (root / "START_HERE.md").write_text("# duplicate\n", encoding="utf-8")
    contract = root / "docs" / "contracts" / "COMPONENT_CONTRACT_TEMPLATE.md"
    contract.parent.mkdir(parents=True)
    contract.write_text("# duplicate\n", encoding="utf-8")
    diagnostics = []

    _check_case_root_scaffold(root, diagnostics)

    rows = [item for item in diagnostics if item.code == "case-legacy-registration-doc"]
    assert len(rows) == 2
