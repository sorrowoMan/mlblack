from __future__ import annotations

from typing import Any

import numpy as np


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
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] > 1:
        return arr
    arr = arr.reshape(-1)
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
