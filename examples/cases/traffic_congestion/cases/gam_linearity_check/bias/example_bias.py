# -*- coding: utf-8 -*-
"""Example OptimizationBias: L2 regularization."""

from __future__ import annotations

import numpy as np

from mlblack.core.bias import OptimizationBias


class ExampleL2Bias(OptimizationBias):
    """Soft L2 penalty on unknown state."""

    context_requires = ("candidate.unknown_state",)
    context_provides = ()
    context_mutates = ("feedback.objectives",)
    context_cache = ()
    context_notes = "Adds L2 penalty to the primary objective."
    requires_metrics = ()
    metrics_fallback = "none"

    def __init__(self, weight=0.01, *, name="l2_bias"):
        super().__init__(name=name)
        self.weight = float(weight)

    def compute(self, unknown_state):
        if unknown_state is None:
            return 0.0
        return self.weight * float(np.sum(np.asarray(unknown_state) ** 2))

    def adjust_feedback(self, feedback, unknown_state, context):
        if feedback is None or not hasattr(feedback, "objectives"):
            return feedback
        l2 = self.compute(unknown_state)
        feedback.objectives = np.asarray(feedback.objectives) + l2
        return feedback
