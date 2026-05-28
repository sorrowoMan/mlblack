# -*- coding: utf-8 -*-
"""Matrix factorization ModelRepresentation: U, V embedding matrices."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.core.representation import ModelRepresentation
from mlblack.core.types import UnknownState


class MFRepresentation(ModelRepresentation):
    """Encodes/decodes a pair of embedding matrices U=(n_users, k), V=(n_items, k).

    The unknown state is the concatenation [U.ravel(), V.ravel()].
    """

    context_requires = ("candidate.unknown_state",)
    context_optional = ()
    context_provides = ("candidate.model",)
    context_mutates = ("candidate.repaired_state",)
    context_cache = ()
    context_notes = "Decodes flattened state into (U, V) embedding tuple."

    def __init__(self, n_users, n_items, k=5, *, init_scale=0.1, nmf=False, name="mf_representation"):
        self.n_users = max(1, int(n_users))
        self.n_items = max(1, int(n_items))
        self.k = max(1, int(k))
        self.init_scale = float(init_scale)
        self.nmf = bool(nmf)
        self._u_size = self.n_users * self.k
        self.name = name

    @property
    def dimension(self):
        return self._u_size + self.n_items * self.k

    def init(self, context=None):
        rng = np.random.default_rng()
        U = rng.normal(0.0, self.init_scale, size=(self.n_users, self.k))
        V = rng.normal(0.0, self.init_scale, size=(self.n_items, self.k))
        return UnknownState(
            values=np.concatenate([U.ravel(), V.ravel()]),
            metadata={"n_users": self.n_users, "n_items": self.n_items, "k": self.k},
        )

    def decode(self, state, context=None):
        arr = state.as_array()
        u_end = self._u_size
        U = arr[:u_end].reshape(self.n_users, self.k)
        V = arr[u_end:].reshape(self.n_items, self.k)
        return (U, V)

    def encode(self, model, context=None):
        U, V = model
        return UnknownState(
            values=np.concatenate([np.asarray(U, dtype=float).ravel(), np.asarray(V, dtype=float).ravel()]),
            metadata={"n_users": self.n_users, "n_items": self.n_items, "k": self.k},
        )

    def repair(self, state, context=None):
        arr = state.as_array()
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e4, neginf=-1e4)
        if self.nmf:
            arr = np.clip(arr, 1e-12, 1e4)
        else:
            arr = np.clip(arr, -1e4, 1e4)
        return state.with_values(arr, repaired=True)

    def predict(self, model):
        U, V = model
        return np.asarray(U, dtype=float) @ np.asarray(V, dtype=float).T

    def describe(self):
        return {
            "name": self.name,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "k": self.k,
            "dimension": self.dimension,
            "nmf": self.nmf,
        }
