# -*- coding: utf-8 -*-
"""Example ModelRepresentation: linear model with point head."""

from __future__ import annotations

import numpy as np

from mlblack.core.representation import ModelRepresentation


class ExampleLinearRepresentation(ModelRepresentation):
    """Simple linear model: coefficients vector with point head."""

    context_requires = ()
    context_provides = ("model.coefficients",)
    context_mutates = ()
    context_cache = ()
    context_notes = "Encodes/decodes a linear coefficient vector."

    def __init__(self, n_features=1, *, name="linear"):
        self.n_features = max(1, int(n_features))
        super().__init__(name=name)

    def init(self, rng=None):
        rng = rng or np.random.default_rng()
        return rng.normal(0.0, 0.1, size=(self.n_features,))

    def encode(self, coefficients):
        return np.asarray(coefficients, dtype=float).ravel()

    def decode(self, encoded):
        return np.asarray(encoded, dtype=float).ravel()

    def predict(self, encoded, X):
        coef = np.asarray(encoded, dtype=float).ravel()
        return X @ coef

    def describe(self):
        return {"name": self.name, "n_features": self.n_features}
