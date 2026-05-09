from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from core.common.contracts import ProcessedDataset, SampleDataset
from core.execution import resolve_torch_execution_device


@dataclass(frozen=True)
class PreparedTrainingData:
    normalized: ProcessedDataset
    X: np.ndarray
    Y: np.ndarray
    context: Any
    n: int
    d: int
    m: int
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]


def resolve_feature_target_names(
    normalized: ProcessedDataset,
    *,
    d: int,
    m: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if normalized.feature_names is not None and len(tuple(normalized.feature_names)) == int(d):
        feature_names = tuple(normalized.feature_names)
    else:
        feature_names = tuple(f"x{i}" for i in range(int(d)))

    if normalized.target_names is not None:
        target_names = tuple(normalized.target_names)
    else:
        target_names = tuple(f"y{i}" for i in range(int(m)))

    return feature_names, target_names


def prepare_training_data(
    *,
    data: ProcessedDataset | SampleDataset,
    numericizer: Any,
    pipeline: Any,
    biases: Sequence[Any],
    fit_context_cls: Any,
) -> PreparedTrainingData:
    normalized = numericizer.to_processed(data)

    X_raw = np.asarray(normalized.X_train, dtype=float)
    Y_raw = np.asarray(normalized.y_train, dtype=float)
    if X_raw.ndim != 2:
        raise ValueError("X_train must be 2D")
    if Y_raw.ndim == 1:
        Y_raw = Y_raw.reshape(-1, 1)
    if Y_raw.ndim != 2:
        raise ValueError("y_train must be 1D or 2D")
    if X_raw.shape[0] != Y_raw.shape[0]:
        raise ValueError("X_train and y_train row count mismatch")

    X_pipeline = pipeline.fit_transform(X_raw, Y_raw)

    context = fit_context_cls()
    Xb = np.asarray(X_pipeline, dtype=float)
    Yb = np.asarray(Y_raw, dtype=float)
    for bias in biases:
        Xb, Yb = bias.apply(Xb, Yb, context)

    n, d = Xb.shape
    m = Yb.shape[1]
    feature_names, target_names = resolve_feature_target_names(normalized, d=int(d), m=int(m))

    return PreparedTrainingData(
        normalized=normalized,
        X=np.asarray(Xb, dtype=float),
        Y=np.asarray(Yb, dtype=float),
        context=context,
        n=int(n),
        d=int(d),
        m=int(m),
        feature_names=feature_names,
        target_names=target_names,
    )


def split_train_val_indices(
    n: int,
    *,
    val_ratio: float,
    seed: int,
    min_no_val_below: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    if int(n) <= 1:
        return np.arange(int(n), dtype=int), np.asarray([], dtype=int)

    rng = np.random.default_rng(int(seed))
    idx = np.arange(int(n), dtype=int)
    rng.shuffle(idx)

    raw_val = int(round(float(val_ratio) * float(n)))
    n_val = max(0, min(raw_val, int(n) - 1))
    if int(n) < int(min_no_val_below):
        n_val = 0

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def resolve_torch_device(
    torch_module: Any,
    requested: str,
) -> Any:
    return resolve_torch_execution_device(torch_module, requested)


def set_torch_seed(torch_module: Any, seed: int) -> None:
    torch_module.manual_seed(int(seed))
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(int(seed))
