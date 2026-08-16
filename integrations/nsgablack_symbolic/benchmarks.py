from __future__ import annotations

from typing import Any

import numpy as np

from mlblack.pipeline.data_views import NumericDataView


def build_symbolic_benchmark_data(
    benchmark_key: str,
    *,
    n_total: int = 240,
    train_ratio: float = 0.8,
    noise_std: float = 0.025,
    seed: int = 42,
) -> NumericDataView:
    """Build the reproducible ML data contract consumed by symbolic bridges."""

    key = str(benchmark_key or "synthetic_symbolic").strip().lower()
    rng = np.random.default_rng(int(seed))
    n = max(24, int(n_total))

    if key in {"ohm", "ohm_like"}:
        resistance = rng.uniform(0.5, 12.0, size=n)
        current = rng.uniform(-3.0, 3.0, size=n)
        ambient = rng.normal(0.0, 1.0, size=n)
        X = np.column_stack([resistance, current, ambient])
        clean = resistance * current
        feature_names = ("resistance", "current", "ambient")
        truth_contracts = ("resistance*current",)
    elif key in {"arrhenius", "arrhenius_gate_like"}:
        temperature = rng.uniform(0.7, 1.4, size=n)
        activation_energy = rng.uniform(0.4, 1.6, size=n)
        catalyst_bias = rng.uniform(-0.5, 0.5, size=n)
        X = np.column_stack([temperature, activation_energy, catalyst_bias])
        clean = catalyst_bias - (activation_energy / temperature)
        feature_names = ("temperature", "activation_energy", "catalyst_bias")
        truth_contracts = ("activation_energy/temperature", "catalyst_bias")
    else:
        x0 = rng.uniform(-2.0, 2.0, size=n)
        x1 = rng.uniform(-1.5, 1.5, size=n)
        x2 = rng.uniform(-1.0, 1.0, size=n)
        X = np.column_stack([x0, x1, x2])
        clean = 0.7 * np.sin(1.6 * x0) + 0.35 * (x1 * x1) - 0.25 * x0 * x2
        feature_names = ("x0", "x1", "x2")
        truth_contracts = ("sin(x0)", "x1*x1", "x0*x2")

    scale = max(float(np.std(clean)), 1.0e-8)
    y = clean + float(max(0.0, noise_std)) * scale * rng.normal(size=n)
    order = rng.permutation(n)
    train_n = int(np.clip(round(float(train_ratio) * n), 12, n - 4))
    train_idx = order[:train_n]
    valid_idx = order[train_n:]
    metadata: dict[str, Any] = {
        "benchmark_key": key,
        "truth_contracts": list(truth_contracts),
        "seed": int(seed),
        "noise_std": float(noise_std),
        "source": "mlblack.integrations.nsgablack_symbolic",
    }
    return NumericDataView(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_valid=X[valid_idx],
        y_valid=y[valid_idx],
        feature_names=feature_names,
        target_name=f"{key}_target",
        metadata=metadata,
    )


__all__ = ["build_symbolic_benchmark_data"]
