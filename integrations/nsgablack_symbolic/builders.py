from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from mlblack.pipeline.data_views import NumericDataView

from .artifacts import OrthogonalBasisSetArtifact
from .orthogonal_problem import OrthogonalBasisEvaluationRecord, OrthogonalBasisOuterProblem, OrthogonalBasisOuterProblemConfig
from .specs import SymbolicOrthogonalNestedPlan, SymbolicStagePlan
from .task_symbolic_problem import (
    BasisConditionedSymbolicTaskConfig,
    BasisConditionedSymbolicTaskProblem,
)


@dataclass
class SymbolicOrthogonalNestedSuite:
    """Serial Stage 1 -> Stage 2 symbolic plan surface.

    This is a problem bundle for nsgablack-facing surfaces. It does not hide
    an outer nsgablack solver, scheduler, runtime or workflow inside mlblack.
    """

    data: NumericDataView
    stage1_problem: OrthogonalBasisOuterProblem
    stage2_config: BasisConditionedSymbolicTaskConfig
    resource_context: Mapping[str, Any] = field(default_factory=dict)
    stage2_problem: BasisConditionedSymbolicTaskProblem | None = None

    def build_stage2_from_record(
        self,
        record: OrthogonalBasisEvaluationRecord | None = None,
    ) -> BasisConditionedSymbolicTaskProblem:
        artifact = self.stage1_problem.build_artifact(record)
        return self.build_stage2_from_artifact(artifact)

    def build_stage2_from_artifact(
        self,
        artifact: OrthogonalBasisSetArtifact,
    ) -> BasisConditionedSymbolicTaskProblem:
        self.stage2_problem = BasisConditionedSymbolicTaskProblem(
            self.data,
            basis_artifact=artifact,
            config=self.stage2_config,
            resource_context=dict(self.resource_context),
        )
        return self.stage2_problem

    def as_spec(self) -> SymbolicOrthogonalNestedPlan:
        return SymbolicOrthogonalNestedPlan(
            basis_stage=SymbolicStagePlan(
                name="orthogonal_basis_search",
                objective_kind="orthogonal_basis",
                function_pool={"source": "pipeline.symbolic.FunctionPoolPipeline"},
                outer_solver={"owner": "nsgablack"},
                inner_training={"owner": "mlblack", "problem": "OrthogonalBasisEvaluationProblem"},
                resource_policy=dict(self.resource_context),
                report_fields=("basis.metrics", "artifact.symbolic_basis_ref", "stage.audit"),
            ),
            task_stage=SymbolicStagePlan(
                name="basis_conditioned_symbolic_task",
                objective_kind="task_regression",
                function_pool={"source": "stage1.basis_artifact"},
                outer_solver={"owner": "nsgablack"},
                inner_training={"owner": "mlblack", "problem": "FixedSymbolicRegressionProblem"},
                resource_policy=dict(self.resource_context),
                report_fields=("task.metrics", "artifact.symbolic_task_ref", "stage.audit"),
            ),
            metadata={"builder": type(self).__name__},
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": "symbolic_orthogonal_nested_plan",
            "data": {
                "n_train": int(self.data.X_train.shape[0]),
                "n_features": int(self.data.n_features),
                "feature_names": list(self.data.effective_feature_names),
                "has_valid": self.data.X_valid is not None,
            },
            "stage1_problem": self.stage1_problem.describe(),
            "stage2_problem": None if self.stage2_problem is None else self.stage2_problem.describe(),
            "spec": self.as_spec().as_dict(),
            "resource_context": dict(self.resource_context),
        }


def build_symbolic_orthogonal_suite(
    data: NumericDataView,
    *,
    stage1_config: OrthogonalBasisOuterProblemConfig | None = None,
    stage2_config: BasisConditionedSymbolicTaskConfig | None = None,
    resource_context: Mapping[str, Any] | None = None,
) -> SymbolicOrthogonalNestedSuite:
    ctx = dict(resource_context or {})
    stage1 = OrthogonalBasisOuterProblem(
        data,
        config=stage1_config or OrthogonalBasisOuterProblemConfig(),
        resource_context=ctx,
    )
    return SymbolicOrthogonalNestedSuite(
        data=data,
        stage1_problem=stage1,
        stage2_config=stage2_config or BasisConditionedSymbolicTaskConfig(),
        resource_context=ctx,
    )


__all__ = [
    "SymbolicOrthogonalNestedSuite",
    "build_symbolic_orthogonal_suite",
]

