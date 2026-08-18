"""One-shot ML diagnostic semantics on the standard Trainer lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from blackbase.contracts import ComponentContract
from .problem import LearningProblem
from .trainer import BlankTrainer
from .types import Feedback, TrainerResult, UnknownState


DiagnosticRunner = Callable[[Mapping[str, Any]], Feedback | Mapping[str, Any] | None]


class DiagnosticProblem(LearningProblem):
    """Problem surface for a single non-search ML diagnostic evaluation."""

    name = "diagnostic_problem"
    context_requires = ()
    context_optional = ("resource_context",)
    context_provides = ("feedback.objectives", "feedback.metrics")
    context_mutates = ()
    context_cache = ()
    context_notes = "Runs one diagnostic and returns compact Feedback; large outputs stay on the problem instance."
    contract = ComponentContract(
        name=name,
        requires=(),
        provides=("feedback.objectives", "feedback.metrics"),
        supports_batch=False,
        supports_resume=False,
        metadata={"family": "diagnostic", "execution": "one_shot"},
    )

    def __init__(self, runner: DiagnosticRunner, *, name: str = name) -> None:
        if not callable(runner):
            raise TypeError("DiagnosticProblem runner must be callable")
        self.runner = runner
        self.name = str(name)
        self.last_result: Feedback | Mapping[str, Any] | None = None

    def evaluate(
        self,
        candidate: Any,
        state: UnknownState | Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Feedback:
        del candidate
        if context is None and isinstance(state, Mapping):
            context = state
        result = self.runner(dict(context or {}))
        self.last_result = result
        if isinstance(result, Feedback):
            return result
        payload = dict(result or {}) if isinstance(result, Mapping) else {}
        objective = float(payload.get("objective", 0.0))
        metrics = _compact_metrics(payload.get("metrics", payload))
        metrics.setdefault("diagnostic.status", str(payload.get("status", "ok")))
        return Feedback(
            objectives=np.asarray([objective], dtype=float),
            constraints=np.zeros(0, dtype=float),
            metrics=metrics,
            info={"result_type": type(result).__name__},
        )

    def describe(self) -> Mapping[str, Any]:
        runner_name = getattr(self.runner, "__name__", type(self.runner).__name__)
        return {
            "name": self.name,
            "family": "diagnostic",
            "execution": "one_shot",
            "runner": str(runner_name),
            "contract": self.get_context_contract(),
        }


class DiagnosticTrainer(BlankTrainer):
    """Trainer control plane for one independently runnable diagnostic task."""

    def __init__(
        self,
        *,
        problem: DiagnosticProblem,
        run_name: str = "diagnostic_run",
        resource_context: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(problem, DiagnosticProblem):
            raise TypeError("DiagnosticTrainer requires DiagnosticProblem")
        super().__init__(
            problem=problem,
            representation=None,
            run_name=run_name,
            resource_context=resource_context,
        )
        self.adapter = None

    def step(self, context: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        ctx = self.build_context(context)
        candidate = UnknownState(
            values=np.zeros(0, dtype=float),
            metadata={"source": "diagnostic", "run_name": self.run_name},
        )
        self.plugin_manager.on_generation_start(self.step_index)
        self.plugin_manager.on_evaluate_start(candidate, ctx)
        feedback = self.problem.evaluate(None, candidate, ctx)
        self.plugin_manager.on_evaluate_end(candidate, feedback, ctx)

        self.population = (candidate,)
        self.feedback = (feedback,)
        self.best_state = candidate
        self.best_feedback = feedback
        self.best_score = feedback.scalar_score(constraint_penalty=self.constraint_penalty)
        snapshot_key = self.write_population_snapshot(
            self.population,
            self.feedback,
            metadata={"trainer": type(self).__name__, "diagnostic": self.problem.name},
        )
        row = self._history_row(
            step=self.step_index,
            population=self.population,
            feedback=self.feedback,
            snapshot_key=snapshot_key,
        )
        self.history.append(row)
        self.plugin_manager.on_generation_end(self.step_index)
        return row

    def fit(self, max_steps: int = 1) -> TrainerResult:
        del max_steps
        return super().fit(max_steps=1)

    def run(self, max_steps: int = 1) -> TrainerResult:
        return self.fit(max_steps=max_steps)

    def build_report(self) -> dict[str, Any]:
        report = super().build_report()
        report["trainer_kind"] = "diagnostic"
        report["diagnostic"] = {
            "name": self.problem.name,
            "result_type": type(self.problem.last_result).__name__,
        }
        return report


def build_diagnostic_trainer(
    runner: DiagnosticRunner,
    *,
    name: str,
    resource_context: Mapping[str, Any] | None = None,
) -> DiagnosticTrainer:
    problem = DiagnosticProblem(runner, name=f"{name}_problem")
    return DiagnosticTrainer(
        problem=problem,
        run_name=str(name),
        resource_context=resource_context,
    )


def _compact_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            metrics[str(key)] = value
    return metrics


__all__ = ["DiagnosticProblem", "DiagnosticRunner", "DiagnosticTrainer", "build_diagnostic_trainer"]
