from __future__ import annotations

import numpy as np

from mlblack.integrations.nsgablack_symbolic import build_symbolic_benchmark_data


def test_symbolic_benchmark_data_is_reproducible_and_carries_truth_contracts() -> None:
    first = build_symbolic_benchmark_data("ohm_like", n_total=40, train_ratio=0.75, seed=7)
    second = build_symbolic_benchmark_data("ohm_like", n_total=40, train_ratio=0.75, seed=7)

    assert first.X_train.shape == (30, 3)
    assert first.X_valid.shape == (10, 3)
    assert first.effective_feature_names == ("resistance", "current", "ambient")
    assert first.metadata["truth_contracts"] == ["resistance*current"]
    np.testing.assert_allclose(first.X_train, second.X_train)
    np.testing.assert_allclose(first.y_valid, second.y_valid)


def test_arrhenius_benchmark_uses_named_mechanism_features() -> None:
    data = build_symbolic_benchmark_data("arrhenius_gate_like", n_total=32, seed=11)

    assert data.effective_feature_names == (
        "temperature",
        "activation_energy",
        "catalyst_bias",
    )
    assert "activation_energy/temperature" in data.metadata["truth_contracts"]
