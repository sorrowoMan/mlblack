from __future__ import annotations

import numpy as np

from mlblack.pipeline.synthetic_temporal import build_sine_forecast_data_view


def test_sine_forecast_data_view_is_deterministic_and_shape_stable() -> None:
    first = build_sine_forecast_data_view(n_train=20, n_valid=5, sequence_length=4, random_seed=7)
    second = build_sine_forecast_data_view(n_train=20, n_valid=5, sequence_length=4, random_seed=7)

    assert first.X_train.shape == (20, 4)
    assert first.X_valid.shape == (5, 4)
    assert first.y_train.shape == (20,)
    assert first.y_valid.shape == (5,)
    assert np.array_equal(first.X_train, second.X_train)
    assert np.array_equal(first.y_valid, second.y_valid)
