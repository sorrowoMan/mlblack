from __future__ import annotations

from typing import Any

import numpy as np

from blackbase.project import CaseRunRequest
from nsgablack.core.base import BlackBoxProblem


class NestedTrainerProblem(BlackBoxProblem):
    """Tune one inner Trainer parameter through the standard child-Case protocol."""

    def __init__(self) -> None:
        super().__init__(
            name="nested_inner_learning_rate",
            dimension=1,
            bounds={"log10_learning_rate": (-2.0, -0.7)},
            objectives=("validation_loss",),
        )
        self.variables = ["log10_learning_rate"]
        self._case_runtime: Any | None = None
        self._records: dict[float, dict[str, Any]] = {}

    def set_case_runtime(self, runtime: Any) -> None:
        self._case_runtime = runtime

    def evaluate(self, candidate):
        record = self._evaluate_child(candidate)
        objectives = np.asarray(record["best_objectives"], dtype=float).reshape(-1)
        if objectives.size == 0 or not np.all(np.isfinite(objectives)):
            raise RuntimeError("inner Trainer did not publish finite best_objectives")
        # This outer Problem explicitly optimizes the inner validation loss.
        # Additional ML objectives remain in the child result envelope and are
        # not implicitly scalarized by the shared result boundary.
        return objectives[:1]

    def evaluate_constraints(self, candidate):
        record = self._evaluate_child(candidate)
        feedback = dict(record.get("best_feedback", {}) or {})
        return np.asarray(feedback.get("constraints", ()), dtype=float).reshape(-1)

    def _evaluate_child(self, candidate) -> dict[str, Any]:
        values = np.asarray(candidate, dtype=float).reshape(-1)
        if values.size != 1:
            raise ValueError("cross-framework candidate must contain one log learning rate")
        key = float(np.round(values[0], 12))
        cached = self._records.get(key)
        if cached is not None:
            return cached
        if self._case_runtime is None:
            raise RuntimeError("Case runtime was not injected into the outer Problem")
        learning_rate = float(10.0 ** float(values[0]))
        child = self._case_runtime.invoke(
            CaseRunRequest(
                project_name="cross_framework",
                stage_name="inner_training",
                case_name="inner_training",
                case_kind="trainer",
                resource_request={
                    "workers": 1,
                    "threads": 1,
                    "memory_mb": 512,
                    "device": "cpu",
                    "backend": "local",
                },
                component_overrides={
                    "learning_rate": learning_rate,
                    "max_steps": 2,
                },
                metadata={
                    "candidate_projection": "log10_learning_rate",
                    "learning_rate": learning_rate,
                },
            )
        )
        if not child.ok:
            failure = None if child.failure is None else child.failure.as_dict()
            raise RuntimeError(f"inner Trainer Case failed: {failure}")
        payload = dict(child.output or {})
        if str(payload.get("protocol_type", "")) != "blackbase.trainer_result":
            raise TypeError("inner Case did not return a TrainerResult envelope")
        self._records[key] = payload
        return payload


__all__ = ["NestedTrainerProblem"]
