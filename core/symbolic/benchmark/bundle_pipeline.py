from __future__ import annotations

from typing import Any

import numpy as np

from core.common.contracts import ProcessedDataset
from workflow import TrainDataBundle

from core.symbolic.benchmark.contracts import SymbolicBenchmarkScenarioDefinition, truth_contract_for_scenario


def split_indices(
    rng: np.random.Generator,
    *,
    n_total: int,
    train_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.arange(int(n_total))
    rng.shuffle(order)
    cut = int(round(float(train_ratio) * float(n_total)))
    cut = max(1, min(cut, int(n_total) - 1))
    return order[:cut], order[cut:]


def build_symbolic_train_bundle_from_arrays(
    *,
    definition: SymbolicBenchmarkScenarioDefinition,
    X: np.ndarray,
    y: np.ndarray,
    y_true: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    noise_std: float,
    metadata_extra: dict[str, Any] | None = None,
) -> tuple[TrainDataBundle, dict[str, Any]]:
    truth_formula = truth_contract_for_scenario(definition).as_formula_payload()
    metadata = {
        "scenario": str(definition.key),
        "n_total": int(X.shape[0]),
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "noise_std": float(noise_std),
        "feature_names": tuple(definition.feature_names),
        "truth_formula": truth_formula,
        "search_hints": {
            "gate_feature_names": tuple(definition.gate_feature_names),
            "periodic_feature_names": tuple(definition.periodic_feature_names),
            "enable_piecewise_basis": bool(definition.enable_piecewise_basis),
        },
    }
    if metadata_extra:
        metadata.update(dict(metadata_extra))
    bundle = TrainDataBundle(
        train=ProcessedDataset(
            X_train=np.asarray(X[train_idx], dtype=float),
            y_train=np.asarray(y[train_idx], dtype=float),
            feature_names=definition.feature_names,
            target_names=definition.target_names,
            metadata={
                "scenario": str(definition.key),
                "split": "train",
            },
        ),
        test=ProcessedDataset(
            X_train=np.asarray(X[test_idx], dtype=float),
            y_train=np.asarray(y[test_idx], dtype=float),
            feature_names=definition.feature_names,
            target_names=definition.target_names,
            metadata={
                "scenario": str(definition.key),
                "split": "test",
            },
        ),
        metadata=metadata,
    )
    truth_payload = {
        "formula": truth_formula,
        "train_target_truth_stats": {
            "mean": float(np.mean(y_true[train_idx])),
            "std": float(np.std(y_true[train_idx])),
            "min": float(np.min(y_true[train_idx])),
            "max": float(np.max(y_true[train_idx])),
        },
        "test_target_truth_stats": {
            "mean": float(np.mean(y_true[test_idx])),
            "std": float(np.std(y_true[test_idx])),
            "min": float(np.min(y_true[test_idx])),
            "max": float(np.max(y_true[test_idx])),
        },
    }
    return bundle, truth_payload


def build_symbolic_benchmark_bundle(
    *,
    definition: SymbolicBenchmarkScenarioDefinition,
    n_total: int,
    train_ratio: float,
    noise_std: float,
    seed: int,
) -> tuple[SymbolicBenchmarkScenarioDefinition, TrainDataBundle, dict[str, Any]]:
    X, y, extras_raw = definition.builder(int(n_total), float(train_ratio), float(noise_std), int(seed))
    extras = dict(extras_raw)
    truth = np.asarray(extras.pop("truth"), dtype=float)
    rng = np.random.default_rng(int(seed))
    train_idx, test_idx = split_indices(rng, n_total=int(n_total), train_ratio=float(train_ratio))
    bundle, truth_payload = build_symbolic_train_bundle_from_arrays(
        definition=definition,
        X=np.asarray(X, dtype=float),
        y=np.asarray(y, dtype=float),
        y_true=truth,
        train_idx=train_idx,
        test_idx=test_idx,
        noise_std=float(noise_std),
        metadata_extra=extras,
    )
    return definition, bundle, truth_payload


__all__ = [
    "build_symbolic_benchmark_bundle",
    "build_symbolic_train_bundle_from_arrays",
    "split_indices",
]
