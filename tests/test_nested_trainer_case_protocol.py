from __future__ import annotations

from pathlib import Path

from blackbase.project import CaseRunResult, execute_project
from blackbase.project.scaffold import add_case, create_project


def test_complete_trainer_case_bridge_returns_real_blank_trainer_result(tmp_path) -> None:
    project_root = create_project(tmp_path / "trainer_case_project", framework="blackbase")
    child_root = add_case("inner_trainer", "trainer", project_root=project_root)
    parent_root = add_case("outer_solver", "solver", project_root=project_root)
    (child_root / "build_solver.py").write_text(
        """
from mlblack.core import BlankTrainer, Feedback, UnknownState


class _StepLimit:
    def __init__(self, limit):
        self.limit = int(limit)

    def is_complete(self, *, step, elapsed, ctx):
        del elapsed, ctx
        return int(step) >= self.limit


class InnerTrainer(BlankTrainer):
    def __init__(self, resource_context):
        super().__init__(run_name="nested-real-trainer", resource_context=resource_context)
        self.input_candidate = UnknownState([0.0])

    def set_case_inputs(self, inputs):
        self.input_candidate = UnknownState.from_protocol_payload(inputs["candidate"])
        self.set_completion_policy(_StepLimit(inputs.get("max_steps", 1)))

    def step(self, context=None):
        del context
        value = float(self.input_candidate.as_array()[0])
        feedback = Feedback(objectives=[value * value], constraints=[-1.0])
        self.best_state = self.input_candidate
        self.best_model = object()
        self.best_feedback = feedback
        self.best_score = feedback.scalar_score()
        self.history.append({"step": int(self.step_index), "score": self.best_score})
        return self.history[-1]


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, component_overrides
    return InnerTrainer(resource_context)
""".lstrip(),
        encoding="utf-8",
    )
    (parent_root / "build_solver.py").write_text(
        """
import numpy as np
from mlblack.integrations.nsgablack_trainer_evaluator import NsgablackTrainerCaseEvaluator


class OuterSolver:
    generation = 7

    def run(self):
        evaluator = NsgablackTrainerCaseEvaluator(
            "inner_trainer",
            max_steps=2,
            resource_request={
                "workers": 1,
                "threads": 1,
                "memory_mb": 256,
                "backend": "local",
                "compute_backend": "auto",
                "device": "cpu",
            },
        )
        objectives, violation = evaluator.evaluate(
            solver=self,
            x=np.asarray([3.0]),
            individual_id=4,
            context={"source": "outer"},
        )
        return {"objectives": objectives, "violation": violation}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    del config, resource_context, component_overrides
    return OuterSolver()
""".lstrip(),
        encoding="utf-8",
    )
    (project_root / "project_config.py").write_text(
        """
PROJECT_NAME = "trainer_case_project"
L0 = {
    "namespace": "trainer_case_project",
    "offer": {
        "threads": 1,
        "gpus": 0,
        "backend": "local",
        "metadata": {"memory_mb": 1024},
    },
    "policy": {
        "mode": "strict",
        "max_workers": 1,
        "max_threads": 1,
        "max_memory_mb": 1024,
    },
    "default_request": {
        "workers": 1,
        "threads": 1,
        "memory_mb": 1024,
        "backend": "local",
    },
    "artifacts": {
        "path": ".blackbase/artifacts",
        "allow_unsafe_serializers": True,
    },
}
STAGES = [{"name": "outer", "cases": ["outer_solver"]}]
GROUPS = {"default": {"stages": ["outer"]}}
""".lstrip(),
        encoding="utf-8",
    )

    project_result = execute_project(project_root, record=False)

    assert project_result.ok
    parent_result = project_result.case_results[0]
    assert parent_result.output["objectives"] == [9.0]
    assert parent_result.output["violation"] == 0.0
    child_payload = parent_result.metadata["runtime_audit"]["child_invocations"]["results"][0]
    child_result = CaseRunResult.from_dict(child_payload)
    assert child_result.ok
    assert child_result.identity.parent_case_run_id == parent_result.identity.case_run_id
    assert child_result.request.child_grant.resources["memory_mb"] == 256.0
    assert child_result.output["protocol_type"] == "blackbase.trainer_result"
    assert child_result.output["best_model"] is None
    model_ref = child_result.output["best_model_ref"]
    assert Path(model_ref["uri"]).is_file()
    assert model_ref["backend"] == "filesystem"
    assert model_ref["checksum"].startswith("sha256:")
    assert model_ref["metadata"]["serializer"] == "mlblack_pickle"
    assert child_result.artifact_refs["best_model"].kind == "model"
    assert child_result.output["best_feedback"]["protocol_type"] == "blackbase.feedback"
