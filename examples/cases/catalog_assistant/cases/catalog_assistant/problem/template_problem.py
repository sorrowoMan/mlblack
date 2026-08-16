# -*- coding: utf-8 -*-
# Problem template: copy and rename for new LearningProblems.

from __future__ import annotations

import numpy as np

from mlblack.core.problem import LearningProblem
from mlblack.core.types import Feedback


class ProblemTemplate(LearningProblem):
    context_requires = ()
    context_provides = ("feedback.objectives",)
    context_mutates = ()
    context_cache = ()
    context_notes = "Template LearningProblem."

    def __init__(self, *, name="problem_template"):
        super().__init__(name=name)

    def evaluate(self, unknown_state):
        arr = np.asarray(unknown_state, dtype=float).ravel()
        f = float(np.sum(arr ** 2))
        return Feedback(
            objectives=np.array([f]),
            gradients=2.0 * arr,
            constraints=np.zeros(0, dtype=float),
        )

    def describe(self):
        return {"name": self.name}
