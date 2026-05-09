from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import TrainerAssemblySpec
from core.common.contracts import Cell, Sample, SampleDataset
from core.orchestration.workflow import TrainDataBundle, TrainFlowSpec, run_train_flow


def _make_dataset(n: int = 1200, seed: int = 7) -> tuple[SampleDataset, SampleDataset]:
    rng = np.random.default_rng(seed)

    speed = rng.uniform(20.0, 120.0, size=n)
    slope = rng.normal(0.0, 1.0, size=n)
    car = rng.choice(["sedan", "suv", "truck"], size=n, p=[0.5, 0.3, 0.2])
    weather = rng.choice(["sunny", "cloudy", "rainy"], size=n, p=[0.4, 0.35, 0.25])

    car_bias = {"sedan": 0.0, "suv": 6.0, "truck": 14.0}
    weather_bias = {"sunny": 0.0, "cloudy": 2.0, "rainy": 8.0}

    y = (
        0.36 * speed
        + 10.0 * slope
        + 5.5 * np.sin(speed / 12.0)
        + np.array([car_bias[c] for c in car], dtype=float)
        + np.array([weather_bias[w] for w in weather], dtype=float)
        + rng.normal(0.0, 3.0, size=n)
    )

    samples: list[Sample] = []
    for i in range(n):
        samples.append(
            Sample(
                sample_id=f"sample_{i}",
                cells={
                    "speed": Cell(name="speed", payload=float(speed[i]), modality="numeric"),
                    "slope": Cell(name="slope", payload=float(slope[i]), modality="numeric"),
                    "car": Cell(name="car", payload=str(car[i]), modality="categorical"),
                    "weather": Cell(name="weather", payload=str(weather[i]), modality="categorical"),
                },
                labels={"target": float(y[i])},
            )
        )

    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(0.8 * n)

    train_samples = [samples[int(i)] for i in idx[:cut]]
    test_samples = [samples[int(i)] for i in idx[cut:]]

    train_ds = SampleDataset(
        samples=train_samples,
        target_key="target",
        feature_cell_keys=("speed", "slope", "car", "weather"),
        description="train_flow_train_split",
    )
    test_ds = SampleDataset(
        samples=test_samples,
        target_key="target",
        feature_cell_keys=("speed", "slope", "car", "weather"),
        description="train_flow_test_split",
    )
    return train_ds, test_ds


def main() -> None:
    train_ds, test_ds = _make_dataset()

    flow_spec = TrainFlowSpec(
        assembly=TrainerAssemblySpec(
            trainer_key="xgboost",
            pipeline_key="identity",
            trainer_params={
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "tree_method": "hist",
                "random_seed": 42,
            },
        ),
        eval_splits=("train", "test"),
        output_dir=str(ROOT / "examples" / "out" / "train_flow_demo"),
        save_artifact=True,
        save_report=True,
        run_name="xgboost_demo",
    )

    result = run_train_flow(
        TrainDataBundle(train=train_ds, test=test_ds),
        spec=flow_spec,
    )

    print("TRAIN FLOW RESULT")
    print(f"trainer={result.report['trainer_name']}")
    for split, m in result.metrics.items():
        print(f"{split:5s} rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  r2={m['r2']:.4f}")
    print(f"output_dir={result.output_dir}")


if __name__ == "__main__":
    main()

