from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class NumericDataView:
    """Prepared numeric data.

    This is intentionally outside the optimizer. Adapters do not consume data;
    LearningProblem consumes this view during evaluation.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray | None = None
    y_valid: np.ndarray | None = None
    feature_names: Sequence[str] | None = None
    target_name: str = "target"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "X_train", _as_2d(self.X_train, name="X_train"))
        object.__setattr__(self, "y_train", _as_1d(self.y_train, name="y_train"))
        if self.X_valid is not None:
            object.__setattr__(self, "X_valid", _as_2d(self.X_valid, name="X_valid"))
        if self.y_valid is not None:
            object.__setattr__(self, "y_valid", _as_1d(self.y_valid, name="y_valid"))
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("X_train and y_train row counts differ")
        if (self.X_valid is None) != (self.y_valid is None):
            raise ValueError("X_valid and y_valid must be provided together")
        if self.X_valid is not None and self.X_valid.shape[0] != self.y_valid.shape[0]:
            raise ValueError("X_valid and y_valid row counts differ")
        if self.feature_names is not None and len(tuple(self.feature_names)) != self.X_train.shape[1]:
            raise ValueError("feature_names length does not match X_train columns")

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def effective_feature_names(self) -> tuple[str, ...]:
        if self.feature_names is not None:
            return tuple(str(x) for x in self.feature_names)
        return tuple(f"x{i}" for i in range(self.n_features))


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


def as_numeric_data_view(
    X: Sequence[Sequence[float]] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    target_name: str = "target",
) -> NumericDataView:
    return NumericDataView(
        X_train=np.asarray(X, dtype=float),
        y_train=np.asarray(y, dtype=float),
        feature_names=feature_names,
        target_name=target_name,
    )


def train_valid_split(
    X: Sequence[Sequence[float]] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    valid_ratio: float = 0.2,
    seed: int = 42,
    feature_names: Sequence[str] | None = None,
    target_name: str = "target",
) -> NumericDataView:
    X_arr = _as_2d(np.asarray(X, dtype=float), name="X")
    y_arr = _as_1d(np.asarray(y, dtype=float), name="y")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y row counts differ")
    rng = np.random.default_rng(int(seed))
    idx = np.arange(X_arr.shape[0])
    rng.shuffle(idx)
    n_valid = max(1, int(round(float(valid_ratio) * X_arr.shape[0])))
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]
    return NumericDataView(
        X_train=X_arr[train_idx],
        y_train=y_arr[train_idx],
        X_valid=X_arr[valid_idx],
        y_valid=y_arr[valid_idx],
        feature_names=feature_names,
        target_name=target_name,
        metadata={"split": "random", "valid_ratio": float(valid_ratio), "seed": int(seed)},
    )


def _as_2d(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    return arr


def _as_3d(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be 3D")
    return arr


def _as_4d(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 4:
        raise ValueError(f"{name} must be 4D NCHW")
    return arr


def _as_1d(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return arr


def _validate_graph_arrays(node_features: np.ndarray, adjacency: np.ndarray, y: np.ndarray, *, prefix: str) -> None:
    if node_features.shape[0] != adjacency.shape[0] or node_features.shape[0] != y.shape[0]:
        raise ValueError(f"{prefix} graph row counts differ")
    if adjacency.shape[1] != adjacency.shape[2]:
        raise ValueError(f"{prefix} adjacency must be square")
    if node_features.shape[1] != adjacency.shape[1]:
        raise ValueError(f"{prefix} node count does not match adjacency")
