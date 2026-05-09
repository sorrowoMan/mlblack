from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ImageClassificationDataset:
    dataset_key: str
    X_train: np.ndarray
    X_test: np.ndarray
    X_repr_train: np.ndarray
    X_repr_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    representation_feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    metadata: dict[str, Any]


def _limit_rows(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int).reshape(-1)
    if int(max_rows) <= 0 or X_arr.shape[0] <= int(max_rows):
        return X_arr, y_arr
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    classes = tuple(int(v) for v in np.unique(y_arr))
    per_class = max(1, int(max_rows) // max(1, len(classes)))
    for cls in classes:
        idx = np.flatnonzero(y_arr == cls)
        if idx.size <= per_class:
            selected.extend(int(v) for v in idx)
        else:
            selected.extend(int(v) for v in rng.choice(idx, size=per_class, replace=False))
    selected_arr = np.asarray(sorted(selected[: int(max_rows)]), dtype=int)
    return X_arr[selected_arr], y_arr[selected_arr]


def _image_representation_features(X: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Objectify 8x8 pixels into stroke/patch statistics before orthogonalization."""

    images = np.asarray(X, dtype=float).reshape(-1, 8, 8)
    n = images.shape[0]
    rows = np.arange(8, dtype=float).reshape(1, 8, 1)
    cols = np.arange(8, dtype=float).reshape(1, 1, 8)
    total = np.sum(images, axis=(1, 2)).reshape(n, 1)
    denom = np.maximum(total, 1e-8)

    row_center = np.sum(images * rows, axis=(1, 2)).reshape(n, 1) / denom
    col_center = np.sum(images * cols, axis=(1, 2)).reshape(n, 1) / denom
    row_var = np.sum(images * ((rows - row_center.reshape(n, 1, 1)) ** 2), axis=(1, 2)).reshape(n, 1) / denom
    col_var = np.sum(images * ((cols - col_center.reshape(n, 1, 1)) ** 2), axis=(1, 2)).reshape(n, 1) / denom

    row_sum = np.sum(images, axis=2)
    col_sum = np.sum(images, axis=1)

    patch_features: list[np.ndarray] = []
    patch_names: list[str] = []
    for r in range(4):
        for c in range(4):
            patch = images[:, 2 * r : 2 * r + 2, 2 * c : 2 * c + 2]
            patch_features.append(np.sum(patch, axis=(1, 2)).reshape(n, 1))
            patch_names.append(f"patch2x2_r{r}_c{c}_ink")

    grad_h = np.sum(np.abs(np.diff(images, axis=2)), axis=2)
    grad_v = np.sum(np.abs(np.diff(images, axis=1)), axis=1)
    global_grad_h = np.sum(grad_h, axis=1).reshape(n, 1)
    global_grad_v = np.sum(grad_v, axis=1).reshape(n, 1)

    features = [
        total,
        row_center,
        col_center,
        row_var,
        col_var,
        row_sum,
        col_sum,
        *patch_features,
        grad_h,
        grad_v,
        global_grad_h,
        global_grad_v,
    ]
    names = (
        "total_ink",
        "row_center",
        "col_center",
        "row_variance",
        "col_variance",
        *(f"row_{i}_ink" for i in range(8)),
        *(f"col_{i}_ink" for i in range(8)),
        *patch_names,
        *(f"row_{i}_horizontal_edge" for i in range(8)),
        *(f"col_{i}_vertical_edge" for i in range(8)),
        "global_horizontal_edge",
        "global_vertical_edge",
    )
    return np.hstack(features), tuple(names)


def load_image_classification_dataset(
    *,
    dataset_key: str,
    train_ratio: float,
    max_rows: int,
    seed: int,
) -> ImageClassificationDataset:
    key = str(dataset_key).strip().lower()
    if key != "digits":
        raise KeyError(f"Unknown image classification dataset: {dataset_key}")

    raw = load_digits()
    X = np.asarray(raw.data, dtype=float)
    y = np.asarray(raw.target, dtype=int).reshape(-1)
    X, y = _limit_rows(X, y, max_rows=int(max_rows), seed=int(seed))
    X_repr, repr_feature_names = _image_representation_features(X)
    feature_names = tuple(f"px_r{r}_c{c}" for r in range(8) for c in range(8))
    class_names = tuple(str(v) for v in tuple(raw.target_names))
    X_train, X_test, X_repr_train, X_repr_test, y_train, y_test = train_test_split(
        X,
        X_repr,
        y,
        train_size=float(train_ratio),
        random_state=int(seed),
        stratify=y,
    )
    return ImageClassificationDataset(
        dataset_key=key,
        X_train=np.asarray(X_train, dtype=float),
        X_test=np.asarray(X_test, dtype=float),
        X_repr_train=np.asarray(X_repr_train, dtype=float),
        X_repr_test=np.asarray(X_repr_test, dtype=float),
        y_train=np.asarray(y_train, dtype=int),
        y_test=np.asarray(y_test, dtype=int),
        feature_names=feature_names,
        representation_feature_names=repr_feature_names,
        class_names=class_names,
        metadata={
            "scenario": key,
            "source": "sklearn.load_digits",
            "task_type": "classification",
            "image_shape": (8, 8),
            "feature_protocol": "raw_pixels_for_searchable_image_objectification",
            "legacy_reference_representation_protocol": "stroke_patch_edge_statistics",
            "n_classes": int(len(class_names)),
            "n_total_effective": int(X.shape[0]),
        },
    )


__all__ = ["ImageClassificationDataset", "load_image_classification_dataset"]
