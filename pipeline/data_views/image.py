from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .common import _as_1d, _as_4d


@dataclass(frozen=True)
class ImageDataView:
    """Image tensors in NCHW layout."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray | None = None
    y_valid: np.ndarray | None = None
    target_name: str = "target"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "X_train", _as_4d(self.X_train, name="X_train"))
        object.__setattr__(self, "y_train", _as_1d(self.y_train, name="y_train"))
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("X_train and y_train row counts differ")
        if self.X_valid is not None:
            object.__setattr__(self, "X_valid", _as_4d(self.X_valid, name="X_valid"))
        if self.y_valid is not None:
            object.__setattr__(self, "y_valid", _as_1d(self.y_valid, name="y_valid"))
        if (self.X_valid is None) != (self.y_valid is None):
            raise ValueError("X_valid and y_valid must be provided together")
        if self.X_valid is not None and self.X_valid.shape[0] != self.y_valid.shape[0]:
            raise ValueError("X_valid and y_valid row counts differ")

    @property
    def channels(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def height(self) -> int:
        return int(self.X_train.shape[2])

    @property
    def width(self) -> int:
        return int(self.X_train.shape[3])

    def describe(self) -> dict[str, Any]:
        return {
            "name": "image_data_view",
            "n_train": int(self.X_train.shape[0]),
            "has_valid": self.X_valid is not None,
            "layout": "NCHW",
            "channels": int(self.channels),
            "height": int(self.height),
            "width": int(self.width),
            "target_name": str(self.target_name),
            "metadata": dict(self.metadata),
        }
