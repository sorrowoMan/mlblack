# -*- coding: utf-8 -*-
"""t-SNE ModelRepresentation: maps between flattened embedding and 2D matrix."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.core.representation import ModelRepresentation
from mlblack.core.types import UnknownState


class TSNERepresentation(ModelRepresentation):
    """Unknown state is the flattened low-dim embedding (n_samples, output_dim)."""

    context_requires = ()
    context_provides = ("candidate.model",)
    context_mutates = ()
    context_cache = ()
    context_notes = "Encodes/decodes a low-dim embedding for t-SNE."

    def __init__(
        self,
        n_samples: int,
        output_dim: int = 2,
        *,
        init_scale: float = 1e-4,
        name: str = "tsne_embedding",
    ):
        self.n_samples = max(1, int(n_samples))
        self.output_dim = max(2, int(output_dim))
        self.init_scale = float(init_scale)
        self._initialized = False
        self.name = str(name)

    @property
    def dimension(self) -> int:
        return self.n_samples * self.output_dim

    def init(self, context: Mapping[str, Any] | None = None) -> UnknownState:
        rng = np.random.default_rng()
        embedding = rng.normal(0.0, self.init_scale, size=(self.n_samples, self.output_dim))
        return UnknownState(values=embedding.ravel(), metadata={"n_samples": self.n_samples, "output_dim": self.output_dim})

    def decode(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> np.ndarray:
        arr = np.asarray(state.values, dtype=np.float64)
        expected = self.n_samples * self.output_dim
        if arr.size != expected:
            raise ValueError(f"Expected {expected} values, got {arr.size}")
        return arr.reshape(self.n_samples, self.output_dim)

    def encode(self, model: Any, context: Mapping[str, Any] | None = None) -> UnknownState:
        arr = np.asarray(model, dtype=np.float64)
        return UnknownState(values=arr.ravel(), metadata={"n_samples": self.n_samples, "output_dim": self.output_dim})

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "output_dim": self.output_dim,
            "init_scale": self.init_scale,
            "dimension": self.dimension,
        }
