# -*- coding: utf-8 -*-
"""Example OptimizationBias: L2 regularization."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from mlblack.bias.base import OptimizationBias


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
        super().__init__(name=name, weight=float(weight))

    def compute(self, unknown_state):
        if unknown_state is None:
            return 0.0
        return self.weight * float(np.sum(np.asarray(unknown_state) ** 2))

    def adjust_feedback(self, control, states, feedback, context):
        del control, context
        adjusted = []
        for state, item in zip(states, feedback, strict=True):
            objectives = np.asarray(item.objectives, dtype=float).copy()
            objectives += self.compute(state.as_array())
            adjusted.append(replace(item, objectives=objectives))
        return tuple(adjusted)
