from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


DESKTOP = Path(__file__).resolve().parents[3]
MLBLACK_ROOT = DESKTOP / "mlblack"
if str(MLBLACK_ROOT) not in sys.path:
    sys.path.insert(0, str(MLBLACK_ROOT))

from nowcasting_work_ci.mlblack_side.orthogonal_basis import (
    OrthogonalBasisSearchConfig,
    run_orthogonal_symbolic_experiment,
)
from pipeline.feature_space import CandidatePoolConfig, FeatureBundle, build_full_candidate_pool


def _make_bundle(seed: int = 20260504) -> FeatureBundle:
    rng = np.random.default_rng(seed)
    x_train = rng.normal(size=(320, 4))
    x_test = rng.normal(size=(120, 4))
    y_train = (
        2.4 * (x_train[:, 0] * x_train[:, 1])
        + 1.7 * np.sin(x_train[:, 2])
        - 0.9 * x_train[:, 3]
        + rng.normal(scale=0.03, size=x_train.shape[0])
    ).reshape(-1, 1)
    y_test = (
        2.4 * (x_test[:, 0] * x_test[:, 1])
        + 1.7 * np.sin(x_test[:, 2])
        - 0.9 * x_test[:, 3]
        + rng.normal(scale=0.03, size=x_test.shape[0])
    ).reshape(-1, 1)
    return FeatureBundle(
        X_train=np.asarray(x_train, dtype=float),
        y_train=np.asarray(y_train, dtype=float),
        X_test=np.asarray(x_test, dtype=float),
        y_test=np.asarray(y_test, dtype=float),
        feature_names=("x0", "x1", "x2", "x3"),
        n_features_raw=4,
        feature_names_raw=("x0", "x1", "x2", "x3"),
        lag_added_features=tuple(),
        lag_cross_added_features=tuple(),
        dropped_features=tuple(),
    )


def _make_ohm_like_bundle(seed: int = 20260504) -> FeatureBundle:
    rng = np.random.default_rng(seed)
    n_train = 420
    n_test = 160
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


def test_orthogonal_basis_search_recovers_known_terms() -> None:
    bundle = _make_bundle()
    pool = build_full_candidate_pool(bundle, CandidatePoolConfig())
    summary = run_orthogonal_symbolic_experiment(
        feature_bundle=bundle,
        cfg=OrthogonalBasisSearchConfig(
            candidate_limit=48,
            seed_candidate_count=12,
            group_count=6,
            min_basis_count=3,
            max_basis_count=5,
            max_pair_abs_corr=0.45,
            rolling_folds=2,
            rolling_val_ratio=0.20,
            min_train_ratio=0.45,
            interval_alpha=0.20,
            selection_mode="rmse_first",
        ),
        candidates=pool,
        experiment_name="synthetic_orthogonal_basis_test",
    )

    best_group = dict(summary["best_group"])
    names = [str(row.get("term_name", "")) for row in list(dict(best_group["final_expression"]).get("terms", ()))]
    joined = " | ".join(names)
    assert any("x0*x1" in name or ("x0" in name and "x1" in name) for name in names), joined
    assert any("sin(x2)" in name or ("sin" in name and "x2" in name) for name in names), joined
    assert any(name == "x3" or "x3" in name for name in names), joined

    orthogonality = dict(best_group["orthogonality_metrics"])
    assert float(orthogonality["pair_abs_corr_mean"]) <= 0.45
    assert float(orthogonality["orthogonality_score"]) > 0.40
    assert float(orthogonality["residual_gain_mean"]) > 0.05
    assert float(orthogonality["semantic_unique_ratio"]) >= 0.66

    metrics = dict(best_group["test_metrics"])
    assert float(metrics["rmse"]) < 0.35
    assert dict(best_group["residual_complementarity_report"]).get("status") == "reported"
    assert dict(best_group["semantic_dedup_report"]).get("status") == "reported"
    assert str(dict(best_group["symbolic_artifact_schema"]).get("schema_key")) == "symbolic_artifact_v1"


def test_orthogonal_basis_search_handles_ohm_like_piecewise_benchmark() -> None:
    bundle = _make_ohm_like_bundle()
    summary = run_orthogonal_symbolic_experiment(
        feature_bundle=bundle,
        cfg=OrthogonalBasisSearchConfig(
            candidate_limit=72,
            seed_candidate_count=14,
            group_count=8,
            min_basis_count=3,
            max_basis_count=6,
            max_pair_abs_corr=0.52,
            rolling_folds=2,
            rolling_val_ratio=0.22,
            min_train_ratio=0.45,
            interval_alpha=0.20,
            selection_mode="rmse_first",
            gate_feature_names=("temperature",),
            gate_quantiles=(0.35, 0.5, 0.65),
            enable_piecewise_basis=True,
        ),
        experiment_name="synthetic_ohm_like_piecewise_test",
    )

    best_group = dict(summary["best_group"])
    terms = list(dict(best_group.get("final_expression", {})).get("terms", ()))
    term_text = " | ".join(str(row.get("expression", "")) for row in terms)
    assert any(
        {"voltage", "resistance"} <= {str(name) for name in tuple(row.get("feature_names", ()))}
        or ("x0" in str(row.get("expression", "")) and "x1" in str(row.get("expression", "")))
        for row in terms
    ), term_text
    assert any(
        "temperature" in {str(name) for name in tuple(row.get("feature_names", ()))}
        or "x2" in str(row.get("expression", ""))
        for row in terms
    ), term_text

    orthogonality = dict(best_group.get("orthogonality_metrics", {}))
    assert float(orthogonality.get("semantic_unique_ratio", 0.0)) >= 0.66
    assert float(orthogonality.get("residual_gain_mean", 0.0)) > 0.04
    assert int(orthogonality.get("piecewise_gate_term_count", 0)) >= 1

    residual = dict(best_group.get("residual_complementarity_report", {}))
    assert residual.get("status") == "reported"
    assert float(residual.get("mean_marginal_r2_gain", 0.0)) > 0.04

    semantic = dict(best_group.get("semantic_dedup_report", {}))
    assert semantic.get("status") == "reported"

    structure_surface = dict(best_group.get("symbolic_structure_surface", {}))
    gate_basis = dict(structure_surface.get("piecewise_gate_basis", {}))
    assert gate_basis.get("available") is True
    assert int(gate_basis.get("gate_basis_count", 0)) >= 1

    family_payload = dict(best_group.get("symbolic_family", {}))
    structure_contracts = dict(family_payload.get("structure_contracts", {}))
    assert str(dict(structure_contracts.get("regime_discovery", {})).get("regime_mode")) == "piecewise_gate"
    assert str(dict(structure_contracts.get("basis_discovery", {})).get("basis_scope")) == "global+local"

    metrics = dict(best_group.get("test_metrics", {}))
    assert float(metrics.get("rmse", 1.0)) < 0.45
