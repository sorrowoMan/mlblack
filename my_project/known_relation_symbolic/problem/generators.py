from __future__ import annotations

from typing import Any

import numpy as np


def _build_ohm_like(
    n_total: int,
    train_ratio: float,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    voltage = rng.uniform(0.8, 11.5, size=n_total)
    resistance = rng.uniform(0.35, 6.5, size=n_total)
    temperature = rng.uniform(-2.2, 2.2, size=n_total)
    material_bias = rng.normal(0.0, 0.45, size=n_total)
    sensor_noise = rng.normal(0.0, 0.8, size=n_total)

    ratio_basis = voltage / (np.abs(resistance) + 1e-3)
    warm_gate = np.maximum(temperature - 0.25, 0.0)
    cool_gate = np.maximum(-temperature - 0.35, 0.0)
    y_true = (
        1.42 * ratio_basis
        + 0.58 * np.sin(temperature)
        + 0.72 * warm_gate
        - 0.44 * cool_gate
        - 0.18 * material_bias
    )
    y = y_true + rng.normal(0.0, float(noise_std), size=n_total)
    X = np.stack([voltage, resistance, temperature, material_bias, sensor_noise], axis=1)
    return X, y.reshape(-1, 1), {
        "truth": y_true.reshape(-1, 1),
        "truth_components": {
            "ratio_basis_mean": float(np.mean(ratio_basis)),
            "warm_gate_fraction": float(np.mean(warm_gate > 0.0)),
            "cool_gate_fraction": float(np.mean(cool_gate > 0.0)),
        },
        "orchestrator_hints": {
            "trainer_params_overrides": {
                "orth_selection_mode": "rmse_first",
                "orth_max_pair_abs_corr": 0.5,
                "family_diversity_bonus": 0.08,
                "piecewise_gate_bonus": 0.28,
                "residual_gain_weight": 0.95,
                "cross_explanatory_rejection_mode": "proxy_group_hard",
                "trivial_nonlinearity_penalty_mode": "proxy_group_explainability_penalty",
                "environment_invariance_audit_mode": "median_split_report",
                "periodic_equivalence_disambiguation_mode": "center_edge_holdout_penalty",
                "phase_spectrum_audit_mode": "center_edge_holdout_report",
                "periodic_family_prior_mode": "semantic_family_boost",
                "periodic_family_prior_weight": 0.18,
                "periodic_candidate_screen_reserve": 1,
                "residual_regime_identification_mode": "selected_basis_residual_scan",
                "regional_correction_basis_mode": "screened_piecewise_candidates",
                "regional_correction_promotion_mode": "topk_residual_gain",
                "regional_correction_feature_scope": "gate_only",
                "regional_correction_topk": 1,
                "regional_correction_min_r2_gain": 0.005,
                "proxy_group_policy": "metadata_or_correlation_cluster",
                "source_overlap_penalty_mode": "feature_overlap+proxy_overlap",
            },
            "core_selection": {
                "run_weight_field": "outer_objective_score",
                "backfill_mode": "weighted_rank",
                "min_seed_terms": 2,
                "core_min_support_rate_ceiling": 0.65,
            },
        },
    }


def _build_ideal_gas_like(
    n_total: int,
    train_ratio: float,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    amount = rng.uniform(0.8, 3.8, size=n_total)
    temperature = rng.uniform(0.9, 4.6, size=n_total)
    volume = rng.uniform(0.6, 3.2, size=n_total)
    material_bias = rng.normal(0.0, 0.35, size=n_total)
    sensor_noise = rng.normal(0.0, 0.7, size=n_total)

    product_ratio = (amount * temperature) / (np.abs(volume) + 1e-3)
    y_true = 2.15 * product_ratio - 0.22 * material_bias
    y = y_true + rng.normal(0.0, float(noise_std), size=n_total)
    X = np.stack([amount, temperature, volume, material_bias, sensor_noise], axis=1)
    return X, y.reshape(-1, 1), {
        "truth": y_true.reshape(-1, 1),
        "truth_components": {
            "product_ratio_mean": float(np.mean(product_ratio)),
            "volume_safe_min": float(np.min(np.abs(volume) + 1e-3)),
        },
        "orchestrator_hints": {
            "trainer_params_overrides": {
                "cross_explanatory_rejection_mode": "proxy_group_hard",
                "trivial_nonlinearity_penalty_mode": "proxy_group_explainability_penalty",
                "environment_invariance_audit_mode": "median_split_report",
                "residual_regime_identification_mode": "selected_basis_residual_scan",
                "regional_correction_basis_mode": "screened_piecewise_candidates",
                "regional_correction_promotion_mode": "topk_residual_gain",
                "regional_correction_feature_scope": "gate_only",
                "regional_correction_topk": 2,
                "regional_correction_min_r2_gain": 0.005,
                "proxy_group_policy": "metadata_or_correlation_cluster",
                "source_overlap_penalty_mode": "feature_overlap+proxy_overlap",
            },
        },
    }


def _build_arrhenius_gate_like(
    n_total: int,
    train_ratio: float,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    activation_energy = rng.uniform(0.8, 3.8, size=n_total)
    temperature = rng.uniform(0.7, 3.4, size=n_total)
    catalyst_bias = rng.normal(0.0, 0.4, size=n_total)
    pressure_bias = rng.normal(0.0, 0.6, size=n_total)
    sensor_noise = rng.normal(0.0, 0.75, size=n_total)

    arrhenius_basis = np.exp(-activation_energy / (np.abs(temperature) + 1e-3))
    warm_gate = np.maximum(temperature - 1.8, 0.0)
    y_true = 1.75 * arrhenius_basis + 0.63 * warm_gate - 0.27 * catalyst_bias
    y = y_true + rng.normal(0.0, float(noise_std), size=n_total)
    X = np.stack([activation_energy, temperature, catalyst_bias, pressure_bias, sensor_noise], axis=1)
    return X, y.reshape(-1, 1), {
        "truth": y_true.reshape(-1, 1),
        "truth_components": {
            "arrhenius_basis_mean": float(np.mean(arrhenius_basis)),
            "warm_gate_fraction": float(np.mean(warm_gate > 0.0)),
        },
        "orchestrator_hints": {
            "trainer_params_overrides": {
                "orth_selection_mode": "rmse_first",
                "orth_max_pair_abs_corr": 0.55,
                "piecewise_gate_bonus": 0.32,
                "residual_gain_weight": 0.95,
                "orth_gate_candidate_screen_reserve": 3,
                "orth_require_gate_candidate_in_group": True,
                "orth_min_gate_basis_terms": 1,
                "orth_mechanistic_feature_groups": (("activation_energy", "temperature"),),
                "orth_mechanistic_screen_bonus": 0.80,
                "orth_mechanistic_group_bonus": 0.30,
                "orth_assembler_basis_binding_mode": "bound",
                "orth_assembler_escape_policy": "budgeted_escape",
                "orth_assembler_escape_feature_names": ("catalyst_bias",),
                "cross_explanatory_rejection_mode": "proxy_group_hard",
                "trivial_nonlinearity_penalty_mode": "proxy_group_explainability_penalty",
                "environment_invariance_audit_mode": "median_split_report",
                "periodic_equivalence_disambiguation_mode": "center_edge_holdout_penalty",
                "phase_spectrum_audit_mode": "center_edge_holdout_report",
                "periodic_family_prior_mode": "semantic_family_boost",
                "periodic_family_prior_weight": 0.45,
                "periodic_candidate_screen_reserve": 2,
                "residual_regime_identification_mode": "selected_basis_residual_scan",
                "regional_correction_basis_mode": "screened_piecewise_candidates",
                "regional_correction_promotion_mode": "topk_residual_gain",
                "regional_correction_feature_scope": "gate_only",
                "regional_correction_topk": 2,
                "regional_correction_min_r2_gain": 0.005,
                "proxy_group_policy": "metadata_or_correlation_cluster",
                "source_overlap_penalty_mode": "feature_overlap+proxy_overlap",
            },
            "core_selection": {
                "run_weight_field": "outer_objective_score",
                "backfill_mode": "weighted_rank",
                "min_seed_terms": 2,
                "core_min_support_rate_ceiling": 0.65,
            },
            "lane_specs": (
                {
                    "lane_id": "mechanistic_gate_lane",
                    "lane_family": "mechanistic_gate",
                    "description": "Reserve gate candidates and strengthen mechanistic cross-feature bias.",
                    "screening_protocol": "target_corr+residual_gain+semantic_novelty+consensus_prior+gate_reserve",
                    "challenger_objective_protocol": "outer_objective+mechanistic_group_bonus+gate_presence",
                    "pool_expansion_bias_protocol": "mechanistic_cross_feature_bias",
                    "lane_weight": 1.0,
                    "repeat_count": 1,
                    "locked_repeat_count": 1,
                    "trainer_params_overrides": {
                        "orth_gate_candidate_screen_reserve": 4,
                        "orth_require_gate_candidate_in_group": True,
                        "orth_min_gate_basis_terms": 1,
                        "orth_mechanistic_screen_bonus": 0.95,
                        "orth_mechanistic_group_bonus": 0.35,
                        "piecewise_gate_bonus": 0.35,
                    },
                },
                {
                    "lane_id": "family_diversity_lane",
                    "lane_family": "family_diversity",
                    "description": "Bias toward family-diverse challengers so exp-ratio and gate terms co-exist.",
                    "screening_protocol": "target_corr+residual_gain+semantic_novelty+consensus_prior",
                    "challenger_objective_protocol": "outer_objective+family_diversity+semantic_dedup",
                    "pool_expansion_bias_protocol": "family_diversity_bias",
                    "lane_weight": 0.95,
                    "repeat_count": 1,
                    "locked_repeat_count": 1,
                    "trainer_params_overrides": {
                        "family_diversity_bonus": 0.14,
                        "semantic_family_bonus": 0.12,
                        "semantic_dup_penalty": 0.42,
                        "piecewise_gate_bonus": 0.24,
                    },
                },
                {
                    "lane_id": "budgeted_escape_lane",
                    "lane_family": "budgeted_escape",
                    "description": "Allow limited locked-stage escape so catalyst bias can re-enter without collapsing the core.",
                    "screening_protocol": "target_corr+residual_gain+semantic_novelty+consensus_prior+mechanistic_escape",
                    "challenger_objective_protocol": "outer_objective+inner_fit+budgeted_escape",
                    "pool_expansion_bias_protocol": "budgeted_escape_feature_bias",
                    "lane_weight": 0.9,
                    "repeat_count": 1,
                    "locked_repeat_count": 1,
                    "trainer_params_overrides": {
                        "orth_assembler_basis_binding_mode": "bound",
                        "orth_assembler_escape_policy": "budgeted_escape",
                        "orth_assembler_escape_feature_names": ("catalyst_bias",),
                        "residual_gain_weight": 1.0,
                    },
                },
            ),
        },
    }


def _build_periodic_gate_like(
    n_total: int,
    train_ratio: float,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    phase_angle = rng.uniform(-np.pi, np.pi, size=n_total)
    load = rng.uniform(-1.4, 1.4, size=n_total)
    material_bias = rng.normal(0.0, 0.35, size=n_total)
    sensor_noise = rng.normal(0.0, 0.65, size=n_total)
    trend_bias = rng.normal(0.0, 0.45, size=n_total)

    warm_gate = np.maximum(phase_angle - 0.45, 0.0)
    y_true = 0.94 * np.sin(phase_angle) + 0.58 * warm_gate - 0.21 * material_bias
    y = y_true + rng.normal(0.0, float(noise_std), size=n_total)
    X = np.stack([phase_angle, load, material_bias, sensor_noise, trend_bias], axis=1)
    return X, y.reshape(-1, 1), {
        "truth": y_true.reshape(-1, 1),
        "truth_components": {
            "phase_sine_mean": float(np.mean(np.sin(phase_angle))),
            "warm_gate_fraction": float(np.mean(warm_gate > 0.0)),
        },
        "orchestrator_hints": {
            "trainer_params_overrides": {
                "cross_explanatory_rejection_mode": "proxy_group_hard",
                "trivial_nonlinearity_penalty_mode": "proxy_group_explainability_penalty",
                "environment_invariance_audit_mode": "median_split_report",
                "periodic_equivalence_disambiguation_mode": "center_edge_holdout_penalty",
                "phase_spectrum_audit_mode": "center_edge_holdout_report",
                "periodic_family_prior_mode": "semantic_family_boost",
                "periodic_family_prior_weight": 0.55,
                "periodic_candidate_screen_reserve": 3,
                "residual_regime_identification_mode": "selected_basis_residual_scan",
                "regional_correction_basis_mode": "screened_piecewise_candidates",
                "regional_correction_promotion_mode": "topk_residual_gain",
                "regional_correction_feature_scope": "gate_only",
                "regional_correction_topk": 2,
                "regional_correction_min_r2_gain": 0.005,
                "proxy_group_policy": "metadata_or_correlation_cluster",
                "source_overlap_penalty_mode": "feature_overlap+proxy_overlap",
            },
            "lane_specs": (
                {
                    "lane_id": "periodic_truth_lane",
                    "lane_family": "periodic_truth",
                    "description": "Keep periodic-family challengers alive and penalize local tanh-style surrogates on phase features.",
                    "screening_protocol": "target_corr+residual_gain+semantic_novelty+consensus_prior+phase_spectrum",
                    "challenger_objective_protocol": "outer_objective+periodic_disambiguation",
                    "pool_expansion_bias_protocol": "periodic_family_bias",
                    "lane_weight": 1.0,
                    "repeat_count": 1,
                    "locked_repeat_count": 1,
                    "trainer_params_overrides": {
                        "periodic_candidate_screen_reserve": 3,
                        "periodic_family_prior_weight": 0.65,
                    },
                },
                {
                    "lane_id": "regional_correction_lane",
                    "lane_family": "regional_correction",
                    "description": "Promote residual-driven gate correction candidates into the basis-conditioned stage.",
                    "screening_protocol": "target_corr+residual_gain+semantic_novelty+consensus_prior+regional_green_channel",
                    "challenger_objective_protocol": "outer_objective+regional_correction_gain",
                    "pool_expansion_bias_protocol": "regional_gate_bias",
                    "lane_weight": 0.95,
                    "repeat_count": 1,
                    "locked_repeat_count": 1,
                    "trainer_params_overrides": {
                        "regional_correction_topk": 2,
                        "regional_correction_min_r2_gain": 0.003,
                    },
                },
            ),
        },
    }


def _build_redundant_proxy_control(
    n_total: int,
    train_ratio: float,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    primary_signal = rng.uniform(-1.8, 1.8, size=n_total)
    primary_signal_proxy = primary_signal + rng.normal(0.0, 0.08, size=n_total)
    phase_angle = rng.uniform(-np.pi, np.pi, size=n_total)
    drift_bias = rng.normal(0.0, 0.35, size=n_total)
    sensor_noise = rng.normal(0.0, 0.75, size=n_total)

    signal_gate = np.maximum(primary_signal - 0.1, 0.0)
    y_true = (
        1.26 * primary_signal
        + 0.61 * np.sin(phase_angle)
        + 0.44 * signal_gate
        - 0.17 * drift_bias
    )
    y = y_true + rng.normal(0.0, float(noise_std), size=n_total)
    X = np.stack([primary_signal, primary_signal_proxy, phase_angle, drift_bias, sensor_noise], axis=1)
    return X, y.reshape(-1, 1), {
        "truth": y_true.reshape(-1, 1),
        "truth_components": {
            "proxy_correlation": float(np.corrcoef(primary_signal, primary_signal_proxy)[0, 1]),
            "signal_gate_fraction": float(np.mean(signal_gate > 0.0)),
        },
        "redundant_feature_groups": {
            "primary_signal_group": ("primary_signal", "primary_signal_proxy"),
        },
        "orchestrator_hints": {
            "trainer_params_overrides": {
                "orth_selection_mode": "rmse_first",
                "orth_max_feature_reuse": 1,
                "feature_overlap_penalty": 0.35,
                "semantic_dup_penalty": 0.55,
                "family_diversity_bonus": 0.08,
                "piecewise_gate_bonus": 0.22,
                "cross_explanatory_rejection_mode": "proxy_group_hard",
                "trivial_nonlinearity_penalty_mode": "proxy_group_explainability_penalty",
                "environment_invariance_audit_mode": "median_split_report",
                "periodic_equivalence_disambiguation_mode": "center_edge_holdout_penalty",
                "phase_spectrum_audit_mode": "center_edge_holdout_report",
                "periodic_family_prior_mode": "semantic_family_boost",
                "periodic_family_prior_weight": 0.32,
                "periodic_candidate_screen_reserve": 2,
                "residual_regime_identification_mode": "selected_basis_residual_scan",
                "regional_correction_basis_mode": "screened_piecewise_candidates",
                "regional_correction_promotion_mode": "topk_residual_gain",
                "regional_correction_feature_scope": "gate_or_selected",
                "regional_correction_topk": 2,
                "regional_correction_min_r2_gain": 0.005,
                "proxy_group_policy": "metadata_or_correlation_cluster",
                "source_overlap_penalty_mode": "feature_overlap+proxy_overlap",
            },
            "core_selection": {
                "run_weight_field": "outer_objective_score",
                "backfill_mode": "weighted_rank",
                "min_seed_terms": 2,
                "core_min_support_rate_ceiling": 0.65,
            },
            "lane_specs": (
                {
                    "lane_id": "proxy_guard_lane",
                    "lane_family": "proxy_guard",
                    "description": "Suppress redundant proxy reuse and force primary-signal exclusivity pressure.",
                    "screening_protocol": "target_corr+residual_gain+semantic_novelty+consensus_prior+proxy_guard",
                    "challenger_objective_protocol": "outer_objective+feature_overlap_penalty+semantic_dedup",
                    "pool_expansion_bias_protocol": "proxy_suppression_bias",
                    "lane_weight": 1.0,
                    "repeat_count": 1,
                    "locked_repeat_count": 1,
                    "trainer_params_overrides": {
                        "orth_max_feature_reuse": 1,
                        "feature_overlap_penalty": 0.45,
                        "semantic_dup_penalty": 0.65,
                        "family_diversity_bonus": 0.06,
                    },
                },
                {
                    "lane_id": "residual_gate_lane",
                    "lane_family": "residual_gate",
                    "description": "Push harder on residual gain so the gate term can challenge the linear proxy family.",
                    "screening_protocol": "target_corr+residual_gain+semantic_novelty+consensus_prior+gate_push",
                    "challenger_objective_protocol": "outer_objective+residual_complementarity+gate_presence",
                    "pool_expansion_bias_protocol": "residual_gate_bias",
                    "lane_weight": 0.95,
                    "repeat_count": 1,
                    "locked_repeat_count": 1,
                    "trainer_params_overrides": {
                        "screen_residual_gain_weight": 0.95,
                        "residual_gain_weight": 1.05,
                        "piecewise_gate_bonus": 0.30,
                        "feature_overlap_penalty": 0.28,
                    },
                },
                {
                    "lane_id": "periodic_challenger_lane",
                    "lane_family": "periodic_challenger",
                    "description": "Keep periodic challengers alive so sin(phase_angle) survives proxy-heavy screens.",
                    "screening_protocol": "target_corr+residual_gain+semantic_novelty+consensus_prior+periodic_challenger",
                    "challenger_objective_protocol": "outer_objective+family_diversity+periodic_family_challenge",
                    "pool_expansion_bias_protocol": "periodic_family_bias",
                    "lane_weight": 0.9,
                    "repeat_count": 1,
                    "locked_repeat_count": 1,
                    "trainer_params_overrides": {
                        "family_diversity_bonus": 0.12,
                        "semantic_family_bonus": 0.14,
                        "screen_semantic_novelty_weight": 0.28,
                        "piecewise_gate_bonus": 0.20,
                        "periodic_candidate_screen_reserve": 2,
                        "periodic_family_prior_weight": 0.40,
                    },
                },
                {
                    "lane_id": "regional_correction_lane",
                    "lane_family": "regional_correction",
                    "description": "Promote signal-gate corrections after proxy suppression so the gate term can re-enter cleanly.",
                    "screening_protocol": "target_corr+residual_gain+semantic_novelty+consensus_prior+regional_green_channel",
                    "challenger_objective_protocol": "outer_objective+regional_correction_gain",
                    "pool_expansion_bias_protocol": "regional_gate_bias",
                    "lane_weight": 0.92,
                    "repeat_count": 1,
                    "locked_repeat_count": 1,
                    "trainer_params_overrides": {
                        "regional_correction_feature_scope": "gate_or_selected",
                        "regional_correction_topk": 2,
                        "regional_correction_min_r2_gain": 0.003,
                    },
                },
            ),
        },
    }


def _build_coupled_reaction_transport_like(
    n_total: int,
    train_ratio: float,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    flow_rate = rng.uniform(0.7, 3.4, size=n_total)
    concentration = rng.uniform(0.55, 2.7, size=n_total)
    temperature = rng.uniform(0.95, 4.2, size=n_total)
    activation_energy = rng.uniform(0.75, 3.7, size=n_total)
    phase_angle = rng.uniform(-np.pi, np.pi, size=n_total)
    load = rng.uniform(-1.55, 1.55, size=n_total)
    load_proxy = load + rng.normal(0.0, 0.075, size=n_total)
    catalyst_bias = rng.normal(0.0, 0.42, size=n_total)
    sensor_noise = rng.normal(0.0, 0.80, size=n_total)

    transport_ratio = (flow_rate * concentration) / (np.abs(temperature) + 1e-3)
    arrhenius_basis = np.exp(-activation_energy / (np.abs(temperature) + 1e-3))
    phase_basis = np.sin(phase_angle)
    load_gate = np.maximum(load - 0.28, 0.0)
    y_true = (
        0.86 * transport_ratio
        + 1.34 * arrhenius_basis
        + 0.52 * phase_basis
        + 0.43 * load_gate
        - 0.22 * catalyst_bias
    )
    y = y_true + rng.normal(0.0, float(noise_std), size=n_total)
    X = np.stack(
        [
            flow_rate,
            concentration,
            temperature,
            activation_energy,
            phase_angle,
            load,
            load_proxy,
            catalyst_bias,
            sensor_noise,
        ],
        axis=1,
    )
    return X, y.reshape(-1, 1), {
        "truth": y_true.reshape(-1, 1),
        "truth_components": {
            "transport_ratio_mean": float(np.mean(transport_ratio)),
            "arrhenius_basis_mean": float(np.mean(arrhenius_basis)),
            "phase_basis_std": float(np.std(phase_basis)),
            "load_gate_fraction": float(np.mean(load_gate > 0.0)),
            "load_proxy_correlation": float(np.corrcoef(load, load_proxy)[0, 1]),
        },
        "redundant_feature_groups": {
            "load_group": ("load", "load_proxy"),
        },
        "orchestrator_hints": {
            "trainer_params_overrides": {
                "orth_selection_mode": "rmse_first",
                "orth_max_pair_abs_corr": 0.70,
                "orth_max_feature_reuse": 2,
                "orth_family_diversity_bonus": 0.12,
                "orth_semantic_family_bonus": 0.12,
                "orth_residual_gain_weight": 1.0,
                "orth_screen_residual_gain_weight": 0.95,
                "orth_native_trunk_candidate_screen_reserve": 3,
                "orth_support_expansion_candidate_screen_reserve": 2,
                "orth_canonical_trunk_candidate_screen_reserve": 2,
                "orth_periodic_candidate_screen_reserve": 2,
                "orth_gate_candidate_screen_reserve": 2,
                "orth_require_periodic_candidate_in_group": True,
                "orth_min_periodic_basis_terms": 1,
                "orth_require_gate_candidate_in_group": True,
                "orth_min_gate_basis_terms": 1,
                "orth_mechanistic_feature_groups": (
                    ("flow_rate", "concentration", "temperature"),
                    ("activation_energy", "temperature"),
                ),
                "orth_mechanistic_screen_bonus": 0.90,
                "orth_mechanistic_group_bonus": 0.35,
                "orth_assembler_max_added_terms": 6,
                "orth_assembler_topk_features": 6,
                "orth_assembler_max_pair_terms": 12,
                "orth_assembler_max_candidates_per_iter": 160,
                "orth_assembler_candidate_keep_top": 8,
                "orth_assembler_max_expr_depth": 7,
                "cross_explanatory_rejection_mode": "proxy_group_hard",
                "trivial_nonlinearity_penalty_mode": "proxy_group_explainability_penalty",
                "environment_invariance_audit_mode": "median_split_report",
                "periodic_equivalence_disambiguation_mode": "center_edge_holdout_penalty",
                "phase_spectrum_audit_mode": "center_edge_holdout_report",
                "periodic_family_prior_mode": "semantic_family_boost",
                "periodic_family_prior_weight": 0.48,
                "residual_regime_identification_mode": "selected_basis_residual_scan",
                "regional_correction_basis_mode": "screened_piecewise_candidates",
                "regional_correction_promotion_mode": "topk_residual_gain",
                "regional_correction_feature_scope": "gate_only",
                "regional_correction_topk": 2,
                "regional_correction_min_r2_gain": 0.004,
                "proxy_group_policy": "metadata_or_correlation_cluster",
                "source_overlap_penalty_mode": "feature_overlap+proxy_overlap",
            },
            "core_selection": {
                "run_weight_field": "outer_objective_score",
                "backfill_mode": "weighted_rank",
                "min_seed_terms": 3,
                "core_min_support_rate_ceiling": 0.65,
            },
        },
    }





__all__ = [
    "_build_ohm_like",
    "_build_ideal_gas_like",
    "_build_arrhenius_gate_like",
    "_build_periodic_gate_like",
    "_build_redundant_proxy_control",
    "_build_coupled_reaction_transport_like",
]
