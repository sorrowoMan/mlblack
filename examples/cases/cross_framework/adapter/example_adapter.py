# -*- coding: utf-8 -*-
"""Example OptimizerAdapter: simple gradient descent."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from mlblack.core.adapter import OptimizerAdapter


class ExampleGradientDescentAdapter(OptimizerAdapter):
    """Vanilla gradient descent."""

    context_requires = ("feedback.gradients", "candidate.unknown_state")
    context_provides = ("population.candidates",)
    context_mutates = ("adapter.current_state",)
    context_cache = ()
    context_notes = "Reads gradients, proposes next candidates via gradient step."

    def __init__(self, learning_rate=0.01, max_candidates=1, *, name="gd"):
        super().__init__(name=name)
        self.learning_rate = float(learning_rate)
        self.max_candidates = max(1, int(max_candidates))

    def propose(self, trainer, context):
        current = context.get("candidate.unknown_state")
        gradients = context.get("feedback.gradients")
        if current is None:
            dim = getattr(getattr(trainer, "representation", None), "dimension", 1)
            rng = np.random.default_rng()
            return [rng.uniform(-1.0, 1.0, size=(dim,))]
        x = np.asarray(current, dtype=float).ravel()
        g = np.asarray(gradients, dtype=float).ravel()
        if len(g) != len(x):
            g = np.zeros_like(x)
        return [x - self.learning_rate * g]

    def update(self, trainer, feedback, context):
        pass

    def describe(self):
        return {"name": self.name, "learning_rate": self.learning_rate}
