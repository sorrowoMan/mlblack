from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Mapping, Sequence

import numpy as np

from mlblack.assembly import build_trainer
from mlblack.assembly.spec import TrainerAssemblySpec
from mlblack.core.contracts import ComponentContract, ContractMixin
from mlblack.pipeline.data import NumericDataView
from mlblack.problems.training import TrainingContract, TrainingResultRecord, TrainingTask

SpecMapper = Callable[[Sequence[float], Mapping[str, Any]], Mapping[str, Any]]


@dataclass
class MLBlackTrainingProxy(ContractMixin):
    """Framework-neutral proxy for outer optimizers.

    It exposes JSON-compatible task/result payloads and never imports nsgablack.
    An outer nsgablack solver may map its candidate vector into a trainer spec,
    inject ResourceContext, then call this proxy.
    """

    data: NumericDataView
    name = "mlblack_training_proxy"
    context_requires = ("training.task", "data.numeric_view")
    context_optional = ("resource.context", "resource_context", "candidate.unknown_state")
    context_provides = ("training.result", "feedback.objectives", "artifact.report")
    context_mutates = ()
    context_cache = ("resource.context",)
    requires_metrics = ("objective",)
    metrics_fallback = "strict"
    context_notes = "Maps an outer optimizer candidate/task into one mlblack trainer run and returns a TrainingResultRecord."
    contract: ClassVar[ComponentContract] = ComponentContract(
        name=name,
        requires=("training.task", "data.numeric_view"),
        optional=("resource.context", "resource_context", "candidate.unknown_state"),
        provides=("training.result", "feedback.objectives", "artifact.report"),
        cache=("resource.context",),
        supports_batch=True,
        supports_resume=False,
        metadata={"layer": "problem_bridge", "cross_framework": True},
    )

    base_trainer_spec: Mapping[str, Any] | None = None
    max_steps: int = 100
    mapper: SpecMapper | None = None
    training_contract: TrainingContract = TrainingContract()

    def make_task(self, candidate: Sequence[float], context: Mapping[str, Any] | None = None) -> TrainingTask:
        ctx = dict(context or {})
        mapped = dict(self.mapper(tuple(float(v) for v in candidate), ctx) if self.mapper else {})
        trainer_spec = dict(mapped.get("trainer_spec", self.base_trainer_spec or {}) or {})
        resource_context = dict(ctx.get("resource_context", ctx.get("resources", {})) or {})
        return TrainingTask(
            trainer_spec=trainer_spec,
            max_steps=int(mapped.get("max_steps", ctx.get("max_steps", self.max_steps))),
            resource_context=resource_context,
            outer_candidate=tuple(float(v) for v in candidate),
            metadata={"proxy": type(self).__name__},
        )

    def run_task(self, task: TrainingTask | Mapping[str, Any]) -> TrainingResultRecord:
        request = TrainingTask.from_value(task)
        if request.trainer_spec:
            trainer_spec = dict(request.trainer_spec)
            if request.resource_context:
                trainer_spec["resource_context"] = {**dict(trainer_spec.get("resource_context", {}) or {}), **dict(request.resource_context)}
            trainer = build_trainer(TrainerAssemblySpec.from_value(trainer_spec), self.data)
            result = trainer.fit(max_steps=request.max_steps)
            artifact_refs = {"trainer": getattr(trainer, "run_name", "trainer")}
        else:
            raise ValueError("TrainingTask requires trainer_spec")
        return TrainingResultRecord.from_trainer_result(
            request.task_id,
            result,
            artifact_refs=artifact_refs,
            resource_context=request.resource_context,
        )

    def evaluate_individual(self, candidate: Sequence[float], context: Mapping[str, Any] | None = None) -> TrainingResultRecord:
        return self.run_task(self.make_task(candidate, context))

    def evaluate_population(self, population: Sequence[Sequence[float]] | np.ndarray, context: Mapping[str, Any] | None = None) -> tuple[TrainingResultRecord, ...]:
        pop = np.asarray(population, dtype=float)
        if pop.ndim == 1:
            pop = pop.reshape(1, -1)
        return tuple(self.evaluate_individual(row, context) for row in pop)

    def describe(self) -> dict[str, Any]:
        return {
            "name": "mlblack_training_proxy",
            "max_steps": int(self.max_steps),
            "component_contract": self.get_contract().describe(),
            "training_contract": self.training_contract.as_dict(),
            "has_mapper": self.mapper is not None,
        }




