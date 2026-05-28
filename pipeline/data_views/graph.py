from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .common import _as_1d, _as_3d, _validate_graph_arrays


@dataclass(frozen=True)
class GraphDataView:
    """Dense graph tensors for small GNN smoke tasks."""

    node_features_train: np.ndarray
    adjacency_train: np.ndarray
    y_train: np.ndarray
    node_features_valid: np.ndarray | None = None
    adjacency_valid: np.ndarray | None = None
    y_valid: np.ndarray | None = None
    target_name: str = "target"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_features_train", _as_3d(self.node_features_train, name="node_features_train"))
        object.__setattr__(self, "adjacency_train", _as_3d(self.adjacency_train, name="adjacency_train"))
        object.__setattr__(self, "y_train", _as_1d(self.y_train, name="y_train"))
        _validate_graph_arrays(self.node_features_train, self.adjacency_train, self.y_train, prefix="train")
        if self.node_features_valid is not None:
            object.__setattr__(self, "node_features_valid", _as_3d(self.node_features_valid, name="node_features_valid"))
        if self.adjacency_valid is not None:
            object.__setattr__(self, "adjacency_valid", _as_3d(self.adjacency_valid, name="adjacency_valid"))
        if self.y_valid is not None:
            object.__setattr__(self, "y_valid", _as_1d(self.y_valid, name="y_valid"))
        has_valid = (self.node_features_valid is not None, self.adjacency_valid is not None, self.y_valid is not None)
        if any(has_valid) and not all(has_valid):
            raise ValueError("node_features_valid, adjacency_valid and y_valid must be provided together")
        if all(has_valid):
            _validate_graph_arrays(self.node_features_valid, self.adjacency_valid, self.y_valid, prefix="valid")

    @property
    def num_nodes(self) -> int:
        return int(self.node_features_train.shape[1])

    @property
    def node_feature_dim(self) -> int:
        return int(self.node_features_train.shape[2])

    def describe(self) -> dict[str, Any]:
        return {
            "name": "graph_data_view",
            "n_train": int(self.node_features_train.shape[0]),
            "has_valid": self.node_features_valid is not None,
            "num_nodes": int(self.num_nodes),
            "node_feature_dim": int(self.node_feature_dim),
            "target_name": str(self.target_name),
            "metadata": dict(self.metadata),
        }
