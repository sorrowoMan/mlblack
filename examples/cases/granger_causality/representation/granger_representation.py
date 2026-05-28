# -*- coding: utf-8 -*-
"""Granger causality VAR coefficient representation.

Flat VAR(1) coefficient vector <-> (n_vars, n_vars) matrix A.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.core.representation import ModelRepresentation
from mlblack.core.types import UnknownState


class GrangerRepresentation(ModelRepresentation):
    """Encodes/decodes a VAR(1) coefficient matrix A of shape (n_vars, n_vars).

    The unknown state is the flattened matrix A.ravel().
    decode() returns A; predict(X, A) = X @ A.T.
    """

    context_requires = ("candidate.unknown_state",)
    context_optional = ()
    context_provides = ("candidate.model",)
    context_mutates = ("candidate.repaired_state",)
    context_cache = ()
    context_notes = "Flattens/reshapes (n_vars, n_vars) VAR coefficient matrix."

    def __init__(self, n_vars, *, init_scale=0.1, name="granger_representation"):
        self.n_vars = max(1, int(n_vars))
        self.dimension = self.n_vars * self.n_vars
        self.init_scale = float(init_scale)
        self.name = name

    def init(self, context=None):
        rng = np.random.default_rng()
        A = rng.normal(0.0, self.init_scale, size=(self.n_vars, self.n_vars))
        return UnknownState(
            values=A.ravel(),
            metadata={"n_vars": self.n_vars},
        )

    def decode(self, state, context=None):
        arr = state.as_array()
        A = arr.reshape(self.n_vars, self.n_vars)
        return A

    def encode(self, model, context=None):
        A = np.asarray(model, dtype=float)
        return UnknownState(
            values=A.ravel(),
            metadata={"n_vars": self.n_vars},
        )

    def repair(self, state, context=None):
        arr = state.as_array()
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e4, neginf=-1e4)
        return state.with_values(arr, repaired=True)

    def predict(self, X, A):
        X = np.asarray(X, dtype=float)
        A = np.asarray(A, dtype=float)
        return X @ A.T

    def describe(self):
        return {
            "name": self.name,
            "n_vars": self.n_vars,
            "dimension": self.dimension,
        }
