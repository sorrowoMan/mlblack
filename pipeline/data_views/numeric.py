from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .common import _as_1d, _as_2d


@dataclass(frozen=True)
class NumericDataView:
    """Prepared numeric supervised data consumed by LearningProblem implementations."""

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

    def describe(self) -> dict[str, Any]:
        return {
            "name": "numeric_data_view",
            "n_train": int(self.X_train.shape[0]),
            "n_valid": 0 if self.X_valid is None else int(self.X_valid.shape[0]),
            "n_features": int(self.n_features),
            "feature_names": self.effective_feature_names,
            "target_name": str(self.target_name),
            "metadata": dict(self.metadata),
        }


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
