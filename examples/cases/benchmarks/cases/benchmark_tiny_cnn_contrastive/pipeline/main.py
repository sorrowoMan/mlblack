from __future__ import annotations

import numpy as np

from mlblack.pipeline.data_views import ImageContrastivePairDataView


def build_data() -> ImageContrastivePairDataView:
    anchors = np.zeros((4, 1, 4, 4), dtype=float)
    positives = np.zeros_like(anchors)
    negatives = np.zeros_like(anchors)
    anchors[:, :, :2, :2] = 1.0
    positives[:, :, :2, :2] = 0.9
    negatives[:, :, 2:, 2:] = 1.0
    return ImageContrastivePairDataView(
        anchor_train=anchors,
        positive_train=positives,
        negative_train=negatives,
    )


__all__ = ["build_data"]
