from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .common import _as_2d


@dataclass(frozen=True)
class PreferencePairDataView:
    """Token-id pairs for preference/DPO-style training."""

    chosen_train: np.ndarray
    rejected_train: np.ndarray
    chosen_valid: np.ndarray | None = None
    rejected_valid: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "chosen_train", _as_2d(self.chosen_train, name="chosen_train"))
        object.__setattr__(self, "rejected_train", _as_2d(self.rejected_train, name="rejected_train"))
        if self.chosen_train.shape != self.rejected_train.shape:
            raise ValueError("chosen_train and rejected_train must have the same shape")
        if self.chosen_valid is not None:
            object.__setattr__(self, "chosen_valid", _as_2d(self.chosen_valid, name="chosen_valid"))
        if self.rejected_valid is not None:
            object.__setattr__(self, "rejected_valid", _as_2d(self.rejected_valid, name="rejected_valid"))
        if (self.chosen_valid is None) != (self.rejected_valid is None):
            raise ValueError("chosen_valid and rejected_valid must be provided together")
        if self.chosen_valid is not None and self.chosen_valid.shape != self.rejected_valid.shape:
            raise ValueError("chosen_valid and rejected_valid must have the same shape")

    @property
    def n_train(self) -> int:
        return int(self.chosen_train.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.chosen_train.shape[1])

    def describe(self) -> dict[str, Any]:
        return {
            "name": "preference_pair_data_view",
            "n_train": int(self.n_train),
            "has_valid": self.chosen_valid is not None,
            "sequence_length": int(self.sequence_length),
            "metadata": dict(self.metadata),
        }
