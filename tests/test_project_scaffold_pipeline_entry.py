from __future__ import annotations

from blackbase.project.doctor import run_common_project_doctor
from mlblack.project.scaffold import add_case, create_project


def test_generated_solver_and_trainer_cases_have_canonical_pipeline_entry(tmp_path) -> None:
    project_root = create_project(tmp_path / "ml_project")

    solver_case = add_case("search_case", "solver", project_root=project_root)
    trainer_case = add_case("fit_case", "trainer", project_root=project_root)

    for case_root in (solver_case, trainer_case):
        pipeline_main = case_root / "pipeline" / "main.py"
        assert pipeline_main.is_file()
        assert "def build_pipeline" in pipeline_main.read_text(encoding="utf-8")
        assert "from .pipeline import build_pipeline" in (case_root / "build_solver.py").read_text(encoding="utf-8")
        assert "load_resource_context_from_env" in (case_root / "run_solver.py").read_text(encoding="utf-8")

    report = run_common_project_doctor(project_root, strict=True)
    assert not any(item.code == "case-pipeline-entry-recommended" for item in report.diagnostics)
