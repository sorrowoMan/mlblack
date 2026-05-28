# -*- coding: utf-8 -*-
"""Synthetic time series data generator with lag features for temporal forecasting."""

from __future__ import annotations

import numpy as np

from mlblack.pipeline.data_views import NumericDataView


def generate_synthetic_series(
    n_points: int = 300,
    noise_std: float = 0.15,
    freq: float = 0.1,
    random_seed: int = 42,
):
    rng = np.random.RandomState(random_seed)
    t = np.arange(n_points, dtype=float)
    signal = np.sin(freq * t)
    noise = rng.randn(n_points).astype(float) * noise_std
    series = signal + noise
    return series, t


def create_lag_features(series: np.ndarray, seq_len: int):
    n = len(series)
    if n <= seq_len:
        raise ValueError(f"Series length {n} must be > seq_len {seq_len}")
    X = np.zeros((n - seq_len, seq_len), dtype=float)
    y = np.zeros(n - seq_len, dtype=float)
    for i in range(n - seq_len):
        X[i] = series[i : i + seq_len]
        y[i] = series[i + seq_len]
    return X, y


def build_data_view(
    n_train: int = 200,
    n_val: int = 50,
    seq_len: int = 12,
    noise_std: float = 0.15,
    random_seed: int = 42,
):
    n_points = n_train + n_val + seq_len
    series, _t = generate_synthetic_series(
        n_points=n_points,
        noise_std=noise_std,
        random_seed=random_seed,
    )

    X_all, y_all = create_lag_features(series, seq_len)

    train_end = n_train
    val_end = train_end + n_val

    X_train = X_all[:train_end]
    y_train = y_all[:train_end]
    X_valid = X_all[train_end:val_end]
    y_valid = y_all[train_end:val_end]

    return NumericDataView(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        feature_names=[f"lag_{i}" for i in range(seq_len)],
        target_name="next_value",
    )
