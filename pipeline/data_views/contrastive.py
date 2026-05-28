from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .common import _as_4d


@dataclass(frozen=True)
class ImageContrastivePairDataView:
    """Anchor/positive/negative image triples for retrieval smoke tasks."""

    anchor_train: np.ndarray
    positive_train: np.ndarray
    negative_train: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor_train", _as_4d(self.anchor_train, name="anchor_train"))
        object.__setattr__(self, "positive_train", _as_4d(self.positive_train, name="positive_train"))
        object.__setattr__(self, "negative_train", _as_4d(self.negative_train, name="negative_train"))
        if self.anchor_train.shape != self.positive_train.shape or self.anchor_train.shape != self.negative_train.shape:
            raise ValueError("anchor/positive/negative image tensors must have the same shape")

    @property
    def channels(self) -> int:
        return int(self.anchor_train.shape[1])

    @property
    def height(self) -> int:
        return int(self.anchor_train.shape[2])

    @property
    def width(self) -> int:
        return int(self.anchor_train.shape[3])

    def describe(self) -> dict[str, Any]:
        return {
            "name": "image_contrastive_pair_data_view",
            "n_train": int(self.anchor_train.shape[0]),
            "layout": "NCHW",
            "channels": int(self.channels),
            "height": int(self.height),
            "width": int(self.width),
            "metadata": dict(self.metadata),
        }
