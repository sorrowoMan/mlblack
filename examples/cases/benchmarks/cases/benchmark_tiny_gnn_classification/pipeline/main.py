from __future__ import annotations

import numpy as np

from mlblack.pipeline.data_views import GraphDataView


def build_data() -> GraphDataView:
    node_features = np.zeros((6, 4, 3), dtype=float)
    adjacency = np.zeros((6, 4, 4), dtype=float)
    y = np.asarray([0, 0, 0, 1, 1, 1], dtype=float)
    for idx in range(6):
        adjacency[idx] = np.eye(4)
        if idx < 3:
            adjacency[idx, 0, 1] = adjacency[idx, 1, 0] = 1.0
            node_features[idx, :, 0] = 1.0
        else:
            adjacency[idx, 2, 3] = adjacency[idx, 3, 2] = 1.0
            node_features[idx, :, 1] = 1.0
    return GraphDataView(
        node_features_train=node_features,
        adjacency_train=adjacency,
        y_train=y,
    )


__all__ = ["build_data"]
