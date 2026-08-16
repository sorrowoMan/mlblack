# -*- coding: utf-8 -*-
"""Example Codec: float-array encode/decode (mlblack unique layer)."""

from __future__ import annotations

import numpy as np
from mlblack.core.representation import ModelRepresentation


class ExampleFloatCodec(ModelRepresentation):
    """Identity-style codec for float arrays."""

    context_requires = ()
    context_provides = ()
    context_mutates = ()
    context_cache = ()
    context_notes = "Minimal identity codec."

    def __init__(self, dimension=1, *, name="float_codec"):
        self.dimension = max(1, int(dimension))
        super().__init__(name=name)

    def init(self, rng=None):
        rng = rng or np.random.default_rng()
        return rng.uniform(-1.0, 1.0, size=(self.dimension,))

    def encode(self, state):
        return np.asarray(state, dtype=float).ravel()

    def decode(self, encoded):
        return np.asarray(encoded, dtype=float).ravel()
