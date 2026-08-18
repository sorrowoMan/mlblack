from __future__ import annotations

from mlblack.project.scaffold.component_templates import render_component_template


def test_mlblack_component_provider_owns_semantic_imports() -> None:
    problem_source = render_component_template("loss", "problem")
    adapter_source = render_component_template("optimizer", "adapter")

    assert "from mlblack.core.problem import LearningProblem" in problem_source
    assert "from mlblack.core.adapter import OptimizerAdapter" in adapter_source
    assert "nsgablack" not in problem_source + adapter_source
