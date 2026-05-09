from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import TrainerAssemblySpec, build_trainer
from core.common.contracts import Cell, ProcessedDataset, Sample, SampleDataset
from numericizer import DefaultNumericizer


def _build_samples(n: int = 1600, seed: int = 42) -> list[Sample]:
    rng = np.random.default_rng(seed)

    speed = rng.uniform(20.0, 120.0, size=n)
    route = rng.normal(0.0, 1.0, size=n)
    car = rng.choice(["sedan", "suv", "truck", "van"], size=n, p=[0.45, 0.25, 0.2, 0.1])
    weather = rng.choice(["sunny", "cloudy", "rainy", "storm"], size=n, p=[0.35, 0.35, 0.22, 0.08])

    car_bias = {"sedan": 0.0, "suv": 8.0, "truck": 16.0, "van": 5.0}
    weather_bias = {"sunny": 0.0, "cloudy": 2.5, "rainy": 9.0, "storm": 20.0}

    noise = rng.normal(0.0, 3.5, size=n)
    y = (
        0.32 * speed
        + 14.0 * route
        + 9.0 * np.sin(speed / 11.0)
        + np.array([car_bias[c] for c in car], dtype=float)
        + np.array([weather_bias[w] for w in weather], dtype=float)
        + noise
    )

    samples: list[Sample] = []
    for i in range(n):
        samples.append(
            Sample(
                sample_id=f"s{i}",
                cells={
                    "speed": Cell(name="speed", payload=float(speed[i]), modality="numeric"),
                    "route": Cell(name="route", payload=float(route[i]), modality="numeric"),
                    "car": Cell(name="car", payload=str(car[i]), modality="categorical"),
                    "weather": Cell(name="weather", payload=str(weather[i]), modality="categorical"),
                },
                labels={"target": float(y[i])},
            )
        )

    return samples


def _split_samples(samples: list[Sample], ratio: float = 0.8, seed: int = 42) -> tuple[list[Sample], list[Sample]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    cut = int(float(ratio) * len(samples))

    train_idx = idx[:cut]
    test_idx = idx[cut:]
    return [samples[int(i)] for i in train_idx], [samples[int(i)] for i in test_idx]


def _metric_report(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    yt = y_true.reshape(-1)
    yp = pred.reshape(-1)
    err = yp - yt

    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1.0 - np.sum(err**2) / ss_tot)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def main() -> None:
    samples = _build_samples()
    train_samples, test_samples = _split_samples(samples)

    train_ds = SampleDataset(
        samples=train_samples,
        target_key="target",
        feature_cell_keys=("speed", "route", "car", "weather"),
        description="trainer_compare_benchmark",
    )

    numericizer = DefaultNumericizer(categorical_unknown="zero")
    numericizer.fit(train_ds)

    X_train = numericizer.transform_features(train_samples)
    y_train = numericizer.transform_targets(train_samples)
    X_test = numericizer.transform_features(test_samples)
    y_test = numericizer.transform_targets(test_samples)

    processed_train = ProcessedDataset(
        X_train=X_train,
        y_train=y_train,
        feature_names=tuple(numericizer.plan.feature_names),
        target_names=tuple(numericizer.plan.target_names),
        metadata={"source": "synthetic_benchmark"},
    )

    trainer_params = {
        "ridge": {
            "l2": 1.0,
        },
        "sklearn_mlp": {
            "hidden_layer_sizes": [64, 32],
            "max_iter": 350,
            "early_stopping": True,
            "random_seed": 42,
        },
        "mlp_torch": {
            "hidden_dims": [64, 32],
            "epochs": 120,
            "batch_size": 64,
            "lr": 1e-3,
            "early_stop_patience": 16,
            "device": "cpu",
            "random_seed": 42,
        },
        "xgboost": {
            "n_estimators": 320,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
            "random_seed": 42,
        },
    }

    rows = []
    for trainer_key, params in trainer_params.items():
        spec = TrainerAssemblySpec(
            trainer_key=trainer_key,
            pipeline_key="identity",
            trainer_params=params,
        )
        trainer = build_trainer(spec)

        t0 = time.perf_counter()
        artifact = trainer.fit(processed_train)
        train_s = time.perf_counter() - t0

        pred = artifact.predict(X_test)
        metrics = _metric_report(y_test, pred)

        rows.append(
            {
                "trainer": trainer_key,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "train_s": float(train_s),
            }
        )

    rows = sorted(rows, key=lambda x: x["rmse"])

    print("RESULTS")
    for row in rows:
        print(
            f"{row['trainer']:12s} rmse={row['rmse']:.4f}  "
            f"mae={row['mae']:.4f}  r2={row['r2']:.4f}  train_s={row['train_s']:.3f}"
        )


if __name__ == "__main__":
    main()

