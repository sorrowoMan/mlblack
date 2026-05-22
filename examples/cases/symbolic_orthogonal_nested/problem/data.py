from __future__ import annotations

import numpy as np

from mlblack.pipeline.data import NumericDataView


def build_symbolic_regression_data(*, n_samples: int = 96, valid_fraction: float = 0.25, seed: int = 11) -> NumericDataView:
    rng = np.random.default_rng(int(seed))
    n = max(24, int(n_samples))
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    x2 = rng.uniform(-1.0, 1.0, size=n)
    X = np.column_stack([x0, x1, x2])
    y = 0.7 * np.sin(1.6 * x0) + 0.35 * (x1 * x1) - 0.25 * x0 * x2 + 0.05 * rng.normal(size=n)

    order = rng.permutation(n)
    valid_n = int(np.clip(round(float(valid_fraction) * n), 4, n // 2))
    valid_idx = order[:valid_n]
    train_idx = order[valid_n:]
    return NumericDataView(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_valid=X[valid_idx],
        y_valid=y[valid_idx],
        feature_names=("x0", "x1", "x2"),
        target_name="synthetic_symbolic_target",
        metadata={"case": "symbolic_orthogonal_nested", "seed": int(seed)},
    )
