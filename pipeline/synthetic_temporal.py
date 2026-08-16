"""Reusable deterministic synthetic data for temporal forecast Case examples."""

from __future__ import annotations

import numpy as np

from .data_views import NumericDataView


def generate_sine_series(
    n_points: int = 300,
    *,
    noise_std: float = 0.15,
    frequency: float = 0.1,
    random_seed: int = 42,
) -> np.ndarray:
    if int(n_points) <= 0:
        raise ValueError("n_points must be positive")
    rng = np.random.default_rng(int(random_seed))
    time_index = np.arange(int(n_points), dtype=float)
    return np.sin(float(frequency) * time_index) + rng.normal(
        0.0,
        float(noise_std),
        size=int(n_points),
    )


def build_sine_forecast_data_view(
    *,
    n_train: int = 200,
    n_valid: int = 50,
    sequence_length: int = 12,
    noise_std: float = 0.15,
    frequency: float = 0.1,
    random_seed: int = 42,
) -> NumericDataView:
    train_count = int(n_train)
    valid_count = int(n_valid)
    window = int(sequence_length)
    if train_count <= 0 or valid_count <= 0 or window <= 0:
        raise ValueError("n_train, n_valid, and sequence_length must be positive")
    series = generate_sine_series(
        train_count + valid_count + window,
        noise_std=noise_std,
        frequency=frequency,
        random_seed=random_seed,
    )
    features = np.stack(
        [series[index : index + window] for index in range(train_count + valid_count)],
        axis=0,
    )
    targets = series[window : window + train_count + valid_count]
    return NumericDataView(
        X_train=features[:train_count],
        y_train=targets[:train_count],
        X_valid=features[train_count:],
        y_valid=targets[train_count:],
        feature_names=tuple(f"lag_{index}" for index in range(window)),
        target_name="next_value",
    )


__all__ = ["build_sine_forecast_data_view", "generate_sine_series"]
