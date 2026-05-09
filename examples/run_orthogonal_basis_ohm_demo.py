from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nowcasting_work_ci.mlblack_side.orthogonal_basis import (  # noqa: E402
    OrthogonalBasisSearchConfig,
    run_orthogonal_symbolic_experiment,
)
from pipeline.feature_space import FeatureBundle  # noqa: E402


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(raw) for key, raw in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _make_ohm_like_bundle(seed: int = 42) -> FeatureBundle:
    rng = np.random.default_rng(seed)
    n_train = 900
    n_test = 320
    voltage_train = rng.uniform(0.8, 9.5, size=n_train)
    resistance_train = rng.uniform(0.4, 6.0, size=n_train)
    temperature_train = rng.uniform(-1.8, 1.8, size=n_train)
    material_bias_train = rng.normal(0.0, 0.4, size=n_train)
    voltage_test = rng.uniform(0.8, 9.5, size=n_test)
    resistance_test = rng.uniform(0.4, 6.0, size=n_test)
    temperature_test = rng.uniform(-1.8, 1.8, size=n_test)
    material_bias_test = rng.normal(0.0, 0.4, size=n_test)

    def _target(voltage: np.ndarray, resistance: np.ndarray, temperature: np.ndarray, material_bias: np.ndarray) -> np.ndarray:
        base_current = voltage / (resistance + 0.25)
        warm_regime = np.maximum(temperature - 0.2, 0.0)
        cool_regime = np.maximum(-temperature - 0.35, 0.0)
        return (
            1.35 * base_current
            + 0.55 * np.sin(temperature)
            + 0.85 * warm_regime
            - 0.45 * cool_regime
            - 0.18 * material_bias
        )

    y_train = (
        _target(voltage_train, resistance_train, temperature_train, material_bias_train)
        + rng.normal(scale=0.025, size=n_train)
    ).reshape(-1, 1)
    y_test = (
        _target(voltage_test, resistance_test, temperature_test, material_bias_test)
        + rng.normal(scale=0.025, size=n_test)
    ).reshape(-1, 1)
    return FeatureBundle(
        X_train=np.asarray(
            np.stack([voltage_train, resistance_train, temperature_train, material_bias_train], axis=1),
            dtype=float,
        ),
        y_train=np.asarray(y_train, dtype=float),
        X_test=np.asarray(
            np.stack([voltage_test, resistance_test, temperature_test, material_bias_test], axis=1),
            dtype=float,
        ),
        y_test=np.asarray(y_test, dtype=float),
        feature_names=("voltage", "resistance", "temperature", "material_bias"),
        n_features_raw=4,
        feature_names_raw=("voltage", "resistance", "temperature", "material_bias"),
        lag_added_features=tuple(),
        lag_cross_added_features=tuple(),
        dropped_features=tuple(),
    )


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "examples" / "out" / f"orthogonal_basis_ohm_demo_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = run_orthogonal_symbolic_experiment(
        feature_bundle=_make_ohm_like_bundle(),
        cfg=OrthogonalBasisSearchConfig(
            candidate_limit=72,
            seed_candidate_count=14,
            group_count=8,
            min_basis_count=3,
            max_basis_count=6,
            max_pair_abs_corr=0.52,
            rolling_folds=3,
            rolling_val_ratio=0.20,
            min_train_ratio=0.45,
            interval_alpha=0.20,
            selection_mode="rmse_first",
            enable_piecewise_basis=True,
            gate_feature_names=("temperature",),
            gate_quantiles=(0.35, 0.5, 0.65),
        ),
        experiment_name="orthogonal_ohm_like_piecewise_demo",
        output_dir=out_dir,
        extra_metadata={
            "benchmark": "ohm_like_piecewise",
            "description": "Known-relation benchmark with ratio basis and gate/piecewise temperature regime.",
        },
    )
    best_group = dict(summary.get("best_group", {}))
    metrics = dict(best_group.get("test_metrics", {}))
    orthogonality = dict(best_group.get("orthogonality_metrics", {}))
    gate_basis = dict(dict(best_group.get("symbolic_structure_surface", {})).get("piecewise_gate_basis", {}))
    report = {
        "summary_path": summary.get("summary_path"),
        "rmse": metrics.get("rmse"),
        "r2": metrics.get("r2"),
        "orthogonality_score": orthogonality.get("orthogonality_score"),
        "residual_gain_mean": orthogonality.get("residual_gain_mean"),
        "semantic_unique_ratio": orthogonality.get("semantic_unique_ratio"),
        "gate_basis_count": gate_basis.get("gate_basis_count"),
        "final_expression": best_group.get("final_expression"),
    }
    (out_dir / "benchmark_report.json").write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
