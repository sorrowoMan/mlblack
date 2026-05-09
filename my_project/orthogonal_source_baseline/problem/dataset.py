from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_covtype, fetch_openml

from my_project.known_relation_symbolic.pipeline import build_known_relation_bundle


@dataclass(frozen=True)
class ScenarioDataset:
    benchmark_key: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]
    truth_payload: dict[str, Any]


def load_known_relation_dataset(
    *,
    benchmark_key: str,
    n_total: int,
    train_ratio: float,
    noise_std: float,
    seed: int,
) -> ScenarioDataset:
    definition, bundle, truth_payload = build_known_relation_bundle(
        benchmark_key=str(benchmark_key),
        n_total=int(n_total),
        train_ratio=float(train_ratio),
        noise_std=float(noise_std),
        seed=int(seed),
    )
    return ScenarioDataset(
        benchmark_key=str(definition.key),
        X_train=np.asarray(bundle.train.X_train, dtype=float),
        y_train=np.asarray(bundle.train.y_train, dtype=float).reshape(-1),
        X_test=np.asarray(bundle.test.X_train, dtype=float),
        y_test=np.asarray(bundle.test.y_train, dtype=float).reshape(-1),
        feature_names=tuple(str(name) for name in tuple(definition.feature_names)),
        metadata=dict(bundle.metadata or {}),
        truth_payload=dict(truth_payload or {}),
    )


def _split_by_order(
    *,
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if int(max_rows) > 0 and X_arr.shape[0] > int(max_rows):
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(np.arange(X_arr.shape[0]), size=int(max_rows), replace=False))
        X_arr = X_arr[idx]
        y_arr = y_arr[idx]
    cut = int(round(float(train_ratio) * float(X_arr.shape[0])))
    cut = max(1, min(cut, X_arr.shape[0] - 1))
    return X_arr[:cut], y_arr[:cut], X_arr[cut:], y_arr[cut:]


def _one_hot_numeric_frame(
    frame: pd.DataFrame,
    *,
    target_name: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if str(target_name) not in frame.columns:
        raise KeyError(f"target column '{target_name}' not found in frame")
    y = pd.to_numeric(frame[str(target_name)], errors="raise").to_numpy(dtype=float)
    features = frame.drop(columns=[str(target_name)])
    encoded = pd.get_dummies(features, dummy_na=False, drop_first=False)
    encoded = encoded.apply(pd.to_numeric, errors="raise")
    feature_names = tuple(str(name) for name in tuple(encoded.columns))
    X = encoded.to_numpy(dtype=float)
    return X, y, feature_names


def load_open_tabular_dataset(
    *,
    dataset_key: str,
    train_ratio: float,
    max_rows: int,
    seed: int,
) -> ScenarioDataset:
    key = str(dataset_key).strip().lower()
    if key == "california_housing":
        raw = fetch_california_housing(as_frame=True)
        frame = raw.frame
        target_name = str(raw.target_names[0])
        feature_names = tuple(str(name) for name in tuple(raw.feature_names))
        X = frame.loc[:, list(feature_names)].to_numpy(dtype=float)
        y = frame[target_name].to_numpy(dtype=float)
        description = "Sklearn California housing regression dataset."
    elif key == "covtype_numeric":
        raw = fetch_covtype(as_frame=True)
        frame = raw.frame
        target_name = "Cover_Type"
        feature_names = tuple(str(name) for name in frame.columns if str(name) != target_name)
        X = frame.loc[:, list(feature_names)].to_numpy(dtype=float)
        y = frame[target_name].to_numpy(dtype=float)
        description = "Sklearn Covertype large-tabular dataset treated as numeric target for regression pressure testing."
    elif key in {"diamonds", "diamonds_price"}:
        raw = fetch_openml(name="diamonds", version=1, as_frame=True, parser="auto")
        frame = raw.frame
        target_name = "price"
        X, y, feature_names = _one_hot_numeric_frame(frame, target_name=target_name)
        description = (
            "OpenML diamonds real regression dataset. Target is continuous price; "
            "categorical cut/color/clarity fields are one-hot encoded."
        )
    else:
        raise KeyError(f"Unknown open tabular dataset: {dataset_key}")
    X_train, y_train, X_test, y_test = _split_by_order(
        X=X,
        y=y,
        train_ratio=float(train_ratio),
        max_rows=int(max_rows),
        seed=int(seed),
    )
    return ScenarioDataset(
        benchmark_key=key,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        metadata={
            "scenario": key,
            "source": "sklearn_fetch_cache",
            "description": description,
            "open_tabular": True,
            "target_name": target_name,
            "task_type": "regression" if key in {"california_housing", "diamonds", "diamonds_price"} else "numeric_pressure_test",
            "n_total_effective": int(X_train.shape[0] + X_test.shape[0]),
        },
        truth_payload={},
    )


def load_scenario_dataset(
    *,
    scenario_key: str,
    n_total: int,
    train_ratio: float,
    noise_std: float,
    seed: int,
    max_rows: int,
) -> ScenarioDataset:
    key = str(scenario_key).strip()
    if key.startswith("open:"):
        return load_open_tabular_dataset(
            dataset_key=key.split(":", 1)[1],
            train_ratio=float(train_ratio),
            max_rows=int(max_rows),
            seed=int(seed),
        )
    return load_known_relation_dataset(
        benchmark_key=key,
        n_total=int(n_total),
        train_ratio=float(train_ratio),
        noise_std=float(noise_std),
        seed=int(seed),
    )


__all__ = [
    "ScenarioDataset",
    "load_known_relation_dataset",
    "load_open_tabular_dataset",
    "load_scenario_dataset",
]
