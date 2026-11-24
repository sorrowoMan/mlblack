from blackbase.project import build_case, load_case_builder, run_case
from mlblack.project.scaffold import create_project, add_case


def test_generated_ml_case_trains_and_exports(tmp_path):
    root = create_project(tmp_path / "ml_starter")
    add_case("fit", "trainer", project_root=root)
    solver = build_case(load_case_builder(root, "fit", "trainer"), resource_context={"threads": 1, "namespace": "starter.fit"})
    result = run_case(solver, case_kind="trainer")
    assert result is not None
    assert solver.run_progress_steps == 2
    assert solver.adapter.method_ids
