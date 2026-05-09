from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.artifacts.piecewise_symbolic_interval_artifact import PiecewiseSymbolicIntervalSurrogateArtifact
from core.artifacts.symbolic_artifact import SymbolicSurrogateArtifact
from core.artifacts.symbolic_interval_artifact import SymbolicIntervalSurrogateArtifact
from core.symbolic import build_symbolic_structure_surface_payload, build_unified_symbolic_family_spec


def _point_metadata() -> dict[str, object]:
    return {
        "data_metadata": {
            "truth_formula": {
                "expression": "y = 1.5 * temperature - 0.25 * sin(humidity)",
                "basis_contract": ("temperature", "sin(humidity)"),
                "strict_contract": ("temperature", "sin(humidity)"),
                "phase_equivalent_contract": ("temperature", "periodic_phase_equivalent(humidity)"),
                "family_level_contract": ("linear_feature_family(temperature)", "periodic_family(humidity)"),
            }
        },
        "orthogonal_search_objective": {
            "protocol": "orthogonal_structure_search_with_budgeted_symbolic_assembler",
            "inner_fit_score": 0.93,
            "orthogonality_score": 0.81,
            "residual_complementarity_score": 0.52,
            "semantic_dedup_score": 1.0,
            "outer_score": 1.58,
        },
        "symbolic_family": {
            "structure_engine": {
                "structure_mode": "seed_library",
                "search_driver": "local_seed_builder",
            },
            "task_head": {
                "task": "point",
                "outputs": ("mean",),
                "objective_family": "regression",
                "calibration_mode": "none",
            },
        },
        "symbolic": {
            "genome_build": {
                "status": "seed_library",
                "seed_library_version": "v2",
            },
            "structure_engine": {
                "structure_mode": "seed_library",
                "search_driver": "local_seed_builder",
            },
        },
        "equivalence_expression_protocol": "EquivalenceExpressionHandlingProtocol",
        "equivalence_expression_mode": "family+phase_equivalent+semantic",
        "equivalence_class_scope": "candidate_screen+consensus+truth_recovery",
        "equivalence_expression_handling": {
            "protocol": "EquivalenceExpressionHandlingProtocol",
            "mode": "family+phase_equivalent+semantic",
            "class_scope": "candidate_screen+consensus+truth_recovery",
            "implemented_submodes": ("semantic_family_equivalence", "phase_equivalent_truth_recovery", "periodic_mode"),
            "child_modes": {
                "periodic_mode": {
                    "canonical_mode_name": "periodic_mode",
                    "leaf_protocol_name": "PeriodicEquivalenceDisambiguationMechanism",
                    "artifact_slot": "periodic_equivalence_disambiguation",
                    "status": "enabled",
                    "mode": "center_edge_holdout_penalty",
                    "periodic_feature_names": ("humidity",),
                }
            },
            "current_narrowness": (
                "Representative selection inside an equivalence class remains heuristic.",
                "Local-equivalence disambiguation is only specialized for periodic features.",
            ),
        },
        "interference_feature_protocol": "InterferenceFeatureHandlingProtocol",
        "interference_feature_mode": "feature_overlap+semantic_dedup+mechanistic_bias",
        "cross_explanatory_rejection_mode": "proxy_group_hard",
        "trivial_nonlinearity_penalty_mode": "proxy_group_explainability_penalty",
        "proxy_group_policy": "hint_if_available",
        "interference_feature_handling": {
            "protocol": "InterferenceFeatureHandlingProtocol",
            "mode": "feature_overlap+semantic_dedup+mechanistic_bias",
            "implemented_submodes": (
                "proxy_suppression_mode",
                "trivial_nonlinearity_rejection_mode",
                "regional_correction_mode",
            ),
            "child_modes": {
                "proxy_suppression_mode": {
                    "canonical_mode_name": "proxy_suppression_mode",
                    "status": "enabled",
                    "cross_explanatory_rejection_mode": "proxy_group_hard",
                },
                "trivial_nonlinearity_rejection_mode": {
                    "canonical_mode_name": "trivial_nonlinearity_rejection_mode",
                    "status": "enabled",
                    "trivial_nonlinearity_penalty_mode": "proxy_group_explainability_penalty",
                },
                "regional_correction_mode": {
                    "canonical_mode_name": "regional_correction_mode",
                    "semantic_slot_name": "regional_residual_correction",
                    "leaf_protocol_name": "RegionalCorrectionBasisProtocol",
                    "artifact_slot": "regional_correction_basis",
                    "status": "enabled",
                },
            },
            "current_narrowness": (
                "Proxy suppression still relies on proxy-group hints and pairwise explainability.",
                "Regional correction is a residual scan rather than a full reopened structure search.",
            ),
        },
        "periodic_equivalence_disambiguation": {
            "protocol": "PeriodicEquivalenceDisambiguationMechanism",
            "parent_protocol": "EquivalenceExpressionHandlingProtocol",
            "parent_mode_slot": "periodic_mode",
            "canonical_mode_name": "periodic_mode",
            "mode": "center_edge_holdout_penalty",
            "phase_spectrum_audit_mode": "center_edge_holdout_report",
            "periodic_family_prior_mode": "semantic_family_boost",
            "periodic_feature_names": ("humidity",),
        },
        "regional_correction_basis": {
            "protocol": "RegionalCorrectionBasisProtocol",
            "parent_protocol": "InterferenceFeatureHandlingProtocol",
            "parent_mode_slot": "regional_correction_mode",
            "canonical_mode_name": "regional_correction_mode",
            "semantic_slot_name": "regional_residual_correction",
            "residual_regime_identification_mode": "selected_basis_residual_scan",
            "regional_correction_basis_mode": "screened_piecewise_candidates",
            "regional_correction_promotion_mode": "topk_residual_gain",
        },
    }


def _fold_report_demo() -> dict[str, object]:
    return {
        "objective_schema": ("coverage_error", "pinaw", "interval_score"),
        "selection_coverage_error_threshold": 0.08,
        "selection_meets_coverage_threshold": True,
        "subset_size": 2,
        "subset_idx": (0, 1),
        "subset_names": ("x0", "x0_plus_x1"),
        "subset_families": ("linear", "interaction"),
        "fold_coverage_error": (0.05, 0.06, 0.07),
        "fold_pinaw": (0.12, 0.11, 0.13),
        "fold_interval_score": (0.22, 0.20, 0.24),
        "fold_picp": (0.90, 0.92, 0.91),
        "fold_mean_width": (1.4, 1.3, 1.5),
        "fold_rmse": (0.33, 0.35, 0.31),
        "fold_branch_detail": ({"fold": 0}, {"fold": 1}, {"fold": 2}),
        "fold_interval_info": ({"coverage": 0.90}, {"coverage": 0.92}, {"coverage": 0.91}),
        "coverage_error_mean": 0.06,
        "pinaw_mean": 0.12,
        "interval_score_mean": 0.22,
        "picp_mean": 0.91,
        "mean_width_mean": 1.4,
        "rmse_mean": 0.33,
        "rmse_std": 0.01632993161855452,
        "rmse_drift": 0.03,
        "complexity_raw": 2.0,
        "family_concentration": 0.5,
        "feature_concentration": 0.5,
        "decode_meta": {"complexity_scale": 1.0},
    }


def _interval_metadata(*, fold_report: dict[str, object] | None = None) -> dict[str, object]:
    meta = {
        "symbolic_family": {
            "structure_engine": {
                "structure_mode": "seed_library",
                "search_driver": "local_seed_builder",
            },
            "task_head": {
                "task": "interval",
                "outputs": ("lower", "upper"),
                "objective_family": "quantile_interval",
                "calibration_mode": "none",
            },
        },
        "symbolic": {
            "genome_build": {
                "status": "seed_library",
                "seed_library_version": "v2",
            },
            "structure_engine": {
                "structure_mode": "seed_library",
                "search_driver": "local_seed_builder",
            },
        },
    }
    if fold_report is not None:
        meta["fold_report"] = dict(fold_report)
    return meta


def _make_point_artifact() -> SymbolicSurrogateArtifact:
    genome = (
        {"name": "x0", "expr": {"type": "feature", "index": 0}},
        {"name": "sin_x1", "expr": {"type": "unary", "op": "sin", "arg": {"type": "feature", "index": 1}}},
    )
    return SymbolicSurrogateArtifact(
        artifact_id="symbolic_point_schema_demo",
        genome=genome,
        parameter_values={},
        readout_weight=np.asarray([[1.5], [-0.25]], dtype=float),
        readout_bias=np.asarray([0.1], dtype=float),
        x_mean=np.asarray([0.0, 0.0], dtype=float),
        x_std=np.asarray([1.0, 1.0], dtype=float),
        residual_std=np.asarray([0.2], dtype=float),
        feature_names=("temperature", "humidity"),
        target_names=("y",),
        metadata=_point_metadata(),
    )


def _make_interval_artifact(
    *,
    artifact_id: str,
    metadata: dict[str, object] | None = None,
) -> SymbolicIntervalSurrogateArtifact:
    genome_low = (
        {"name": "x0", "expr": {"type": "feature", "index": 0}},
    )
    genome_high = (
        {
            "name": "x0_plus_x1",
            "expr": {
                "type": "binary",
                "op": "add",
                "left": {"type": "feature", "index": 0},
                "right": {"type": "feature", "index": 1},
            },
        },
    )
    return SymbolicIntervalSurrogateArtifact(
        artifact_id=str(artifact_id),
        lower_quantile=0.1,
        upper_quantile=0.9,
        genome_low=genome_low,
        parameter_values_low={},
        readout_weight_low=np.asarray([[0.8]], dtype=float),
        readout_bias_low=np.asarray([-0.2], dtype=float),
        genome_high=genome_high,
        parameter_values_high={},
        readout_weight_high=np.asarray([[1.2]], dtype=float),
        readout_bias_high=np.asarray([0.3], dtype=float),
        x_mean=np.asarray([0.0, 0.0], dtype=float),
        x_std=np.asarray([1.0, 1.0], dtype=float),
        residual_std=np.asarray([0.15], dtype=float),
        calibration_margin=np.asarray([0.05], dtype=float),
        feature_names=("temperature", "humidity"),
        target_names=("y",),
        metadata=_interval_metadata() if metadata is None else metadata,
    )


class TestSymbolicArtifactSchema(unittest.TestCase):
    def test_point_artifact_schema_is_persisted(self) -> None:
        artifact = _make_point_artifact()
        schema = dict(artifact.metadata.get("symbolic_artifact_schema", {}))

        self.assertEqual(str(schema.get("schema_key")), "symbolic_artifact_v1")
        self.assertEqual(str(dict(schema.get("head_semantics", {})).get("task")), "point")
        self.assertEqual(int(dict(schema.get("complexity_metrics", {})).get("term_count", 0)), 2)
        self.assertIn("temperature", tuple(dict(schema.get("feature_usage", {})).get("used_features", ())))
        self.assertEqual(str(dict(schema.get("regime_structure", {})).get("mode")), "global_only")
        self.assertEqual(int(dict(schema.get("basis_structure", {})).get("basis_count", 0)), 2)
        self.assertEqual(int(dict(schema.get("assembler_structure", {})).get("output_expression_count", 0)), 1)
        self.assertFalse(bool(dict(schema.get("piecewise_gate_basis", {})).get("available")))
        truth_recovery = dict(schema.get("truth_contract_recovery", {}))
        self.assertEqual(str(truth_recovery.get("status")), "reported")
        self.assertAlmostEqual(float(truth_recovery.get("exact_basis_hit_score", 0.0)), 1.0, places=6)
        self.assertAlmostEqual(float(truth_recovery.get("exact_term_recovery_score", 0.0)), 1.0, places=6)
        self.assertAlmostEqual(float(truth_recovery.get("phase_equivalent_term_recovery_score", 0.0)), 1.0, places=6)
        self.assertAlmostEqual(float(truth_recovery.get("family_level_term_recovery_score", 0.0)), 1.0, places=6)
        objective = dict(schema.get("orthogonal_search_objective", {}))
        self.assertEqual(str(objective.get("status")), "reported")
        self.assertAlmostEqual(float(objective.get("outer_score", 0.0)), 1.58, places=6)
        equivalence = dict(schema.get("equivalence_expression_handling", {}))
        self.assertEqual(str(equivalence.get("protocol")), "EquivalenceExpressionHandlingProtocol")
        self.assertIn("periodic_mode", tuple(equivalence.get("implemented_submodes", ())))
        child_modes = dict(equivalence.get("child_modes", {}) or {})
        self.assertEqual(
            str(dict(child_modes.get("periodic_mode", {}) or {}).get("leaf_protocol_name")),
            "PeriodicEquivalenceDisambiguationMechanism",
        )
        interference = dict(schema.get("interference_feature_handling", {}) or {})
        self.assertEqual(str(interference.get("protocol")), "InterferenceFeatureHandlingProtocol")
        self.assertIn("regional_correction_mode", tuple(interference.get("implemented_submodes", ())))
        interference_child_modes = dict(interference.get("child_modes", {}) or {})
        self.assertEqual(
            str(dict(interference_child_modes.get("regional_correction_mode", {}) or {}).get("semantic_slot_name")),
            "regional_residual_correction",
        )
        periodic_leaf = dict(schema.get("periodic_equivalence_disambiguation", {}) or {})
        self.assertEqual(str(periodic_leaf.get("parent_mode_slot")), "periodic_mode")
        regional_leaf = dict(schema.get("regional_correction_basis", {}) or {})
        self.assertEqual(str(regional_leaf.get("parent_mode_slot")), "regional_correction_mode")

        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "artifact"
            artifact.save(str(artifact_dir))

            schema_path = artifact_dir / "symbolic_schema.json"
            self.assertTrue(schema_path.exists())

            saved_schema = json.loads(schema_path.read_text(encoding="utf-8"))
            self.assertEqual(str(saved_schema.get("schema_key")), "symbolic_artifact_v1")
            self.assertEqual(str(dict(saved_schema.get("head_semantics", {})).get("task")), "point")

            loaded = SymbolicSurrogateArtifact.load(str(artifact_dir))
            loaded_schema = dict(loaded.metadata.get("symbolic_artifact_schema", {}))
            self.assertEqual(int(dict(loaded_schema.get("complexity_metrics", {})).get("term_count", 0)), 2)

    def test_interval_and_piecewise_schema_capture_bounds_and_regimes(self) -> None:
        global_artifact = _make_interval_artifact(artifact_id="symbolic_interval_global")
        local_artifact = _make_interval_artifact(artifact_id="symbolic_interval_local")

        interval_schema = dict(global_artifact.metadata.get("symbolic_artifact_schema", {}))
        self.assertEqual(str(dict(interval_schema.get("head_semantics", {})).get("task")), "interval")
        self.assertIn("low", dict(dict(interval_schema.get("final_expression", {})).get("y", {})))
        self.assertIn("high", dict(dict(interval_schema.get("term_contributions", {})).get("y", {})))
        self.assertEqual(str(dict(interval_schema.get("regime_structure", {})).get("mode")), "global_only")
        self.assertEqual(str(dict(interval_schema.get("basis_structure", {})).get("basis_scope")), "global")
        self.assertEqual(str(dict(interval_schema.get("assembler_structure", {})).get("assembler_mode")), "budgeted_symbolic_regression")

        piecewise = PiecewiseSymbolicIntervalSurrogateArtifact(
            artifact_id="symbolic_interval_piecewise",
            global_artifact=global_artifact,
            local_artifacts={"1|0": local_artifact},
            gate_feature_names=("temperature",),
            blend_kappa=32.0,
            regime_counts={"1|0": 24},
            feature_names=("temperature", "humidity"),
            target_names=("y",),
            metadata={
                "aggregate_manifest": {
                    "selected_regime_keys": ("1|0",),
                    "failed_regimes": {},
                    "local_regimes": {"1|0": {"count": 24}},
                },
                "gate_piecewise": {
                    "gate_feature_names": ("temperature",),
                },
            },
        )
        piecewise_schema = dict(piecewise.metadata.get("symbolic_artifact_schema", {}))

        self.assertTrue(bool(dict(piecewise_schema.get("head_semantics", {})).get("piecewise_enabled")))
        self.assertIn("1|0", dict(dict(piecewise_schema.get("final_expression", {})).get("local_by_regime", {})))
        self.assertEqual(int(dict(piecewise_schema.get("complexity_metrics", {})).get("selected_regime_count", 0)), 1)
        self.assertEqual(str(dict(piecewise_schema.get("regime_structure", {})).get("mode")), "piecewise_gate")
        self.assertIn("1|0", tuple(dict(piecewise_schema.get("regime_structure", {})).get("selected_regime_keys", ())))
        self.assertIn("1|0", dict(dict(piecewise_schema.get("basis_structure", {})).get("local_basis_by_regime", {})))
        self.assertTrue(bool(dict(piecewise_schema.get("piecewise_gate_basis", {})).get("available")))
        self.assertTrue(bool(dict(piecewise_schema.get("piecewise_gate_basis", {})).get("enabled")))
        self.assertIn("1|0", dict(dict(piecewise_schema.get("piecewise_gate_basis", {})).get("local_basis_counts", {})))
        self.assertEqual(
            str(dict(piecewise_schema.get("assembler_structure", {})).get("assembler_mode")),
            "piecewise_budgeted_symbolic_regression",
        )

        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "piecewise_artifact"
            piecewise.save(str(artifact_dir))
            self.assertTrue((artifact_dir / "symbolic_schema.json").exists())

    def test_interval_schema_normalizes_fold_stability_report(self) -> None:
        artifact = _make_interval_artifact(
            artifact_id="symbolic_interval_fold_schema",
            metadata=_interval_metadata(fold_report=_fold_report_demo()),
        )
        schema = dict(artifact.metadata.get("symbolic_artifact_schema", {}))
        stability = dict(schema.get("stability_metrics", {}))

        self.assertTrue(bool(stability.get("fold_stability_available")))
        self.assertEqual(int(stability.get("fold_schema_version", 0)), 1)
        self.assertEqual(int(stability.get("fold_count", 0)), 3)
        self.assertTrue(bool(stability.get("selection_meets_coverage_threshold")))
        self.assertAlmostEqual(float(stability.get("coverage_error_mean", 0.0)), 0.06, places=6)

        fold_summary = dict(stability.get("fold_summary", {}))
        self.assertAlmostEqual(float(fold_summary.get("rmse_mean", 0.0)), 0.33, places=6)

        fold_stability = dict(stability.get("fold_stability", {}))
        self.assertEqual(str(fold_stability.get("source")), "metadata.fold_report")
        self.assertIn("rmse", dict(fold_stability.get("metrics_by_name", {})))
        self.assertEqual(int(dict(fold_stability.get("detail_counts", {})).get("branch_detail_count", 0)), 3)

    def test_structure_surface_records_residual_semantic_and_gate_basis(self) -> None:
        family = build_unified_symbolic_family_spec(
            parameter_backend="ridge",
            task="point",
            supports_piecewise_basis=True,
        )
        basis_rows = [
            {
                "term_name": "v_over_r",
                "expression": "voltage/(resistance+0.25)",
                "feature_names": ["voltage", "resistance"],
                "semantic_signature": "binary:div(feature:0,binary:add(feature:1,const))",
                "semantic_family": "ratio_or_reciprocal",
                "uses_piecewise_gate": False,
            },
            {
                "term_name": "gate_temperature",
                "expression": "soft_step(temperature>0.5)",
                "feature_names": ["temperature"],
                "semantic_signature": "piecewise:temperature",
                "semantic_family": "piecewise_gate",
                "uses_piecewise_gate": True,
            },
        ]
        metadata = {
            "symbolic_family": family.description_dict(),
            "selected_basis": basis_rows,
            "basis_semantics": {
                "source": "test",
                "basis_scope": "global",
                "basis_terms": basis_rows,
            },
            "basis_overlap_report": {
                "source": "test",
                "basis_count": 2,
                "pair_abs_corr_mean": 0.18,
                "pair_abs_corr_max": 0.24,
                "orthogonality_score": 0.76,
            },
            "residual_complementarity_report": {
                "source": "test",
                "status": "reported",
                "mean_marginal_r2_gain": 0.31,
                "min_marginal_r2_gain": 0.14,
                "steps": [{"term_name": "gate_temperature"}],
            },
            "semantic_dedup_report": {
                "source": "test",
                "status": "reported",
                "semantic_unique_ratio": 1.0,
                "piecewise_gate_term_count": 1,
                "semantic_groups": [],
            },
            "assembler_budget": {
                "source": "test",
                "assembler_mode": "budgeted_symbolic_regression",
                "recorded_values": {"basis_count": 2},
                "output_expression_count": 1,
                "selected_basis_count": 2,
                "uses_piecewise_gate": True,
            },
            "structure_head": "expression",
            "prediction_head": "point",
            "search_input_space": "basis_object_space",
            "pool_expansion_unit": "basis_object",
            "gradient_guidance_mode": "basis_object_gradient",
            "basis_binding_mode": "defining",
            "escape_policy": "forbid",
            "stage_head_protocols": {
                "basis_discovery": {
                    "structure_head": "basis_set",
                    "prediction_head": "none",
                    "search_input_space": "raw_feature_space",
                    "pool_expansion_unit": "raw_feature",
                    "gradient_guidance_mode": "raw_feature_gradient",
                    "basis_binding_mode": "off",
                    "escape_policy": "fallback_to_generic",
                },
                "assembler": {
                    "structure_head": "expression",
                    "prediction_head": "point",
                    "search_input_space": "basis_object_space",
                    "pool_expansion_unit": "basis_object",
                    "gradient_guidance_mode": "basis_object_gradient",
                    "basis_binding_mode": "defining",
                    "escape_policy": "forbid",
                },
            },
            "basis_context": {
                "basis_source": "orthogonal_basis_discovery",
                "binding_mode": "defining",
                "equivalence_mode": "family-level",
                "selected_basis": [
                    {"object_key": "v_over_r", "expression": "voltage/(resistance+0.25)"},
                    {"object_key": "gate_temperature", "expression": "soft_step(temperature>0.5)"},
                ],
                "locked_basis_keys": ["v_over_r", "gate_temperature"],
            },
            "basis_object_gradient_pool": {
                "available": True,
                "protocol": "basis_object_gradient_pool_expansion_v1",
                "top_object_signals": [{"object_key": "v_over_r", "gradient_score": 0.42}],
                "expansion_candidates": [{"candidate_key": "sin(v_over_r)", "priority": 0.3}],
            },
            "gate_piecewise": {
                "gate_feature_names": ["temperature"],
                "gate_indices": [2],
                "gate_basis_terms": [basis_rows[1]],
            },
        }
        payload = build_symbolic_structure_surface_payload(
            metadata=metadata,
            final_expression={"expression": "(1.0)*(voltage/(resistance+0.25)) + (0.4)*(soft_step(temperature>0.5))"},
            global_basis=basis_rows,
            gate_basis=[basis_rows[1]],
            piecewise_enabled=False,
            basis_scope="global",
            gate_feature_names=("temperature",),
            gate_indices=(2,),
        )
        basis_structure = dict(payload.get("basis_structure", {}))
        orthogonality = dict(basis_structure.get("orthogonality_status", {}))
        residual = dict(basis_structure.get("residual_complementarity", {}))
        semantic = dict(basis_structure.get("semantic_deduplication", {}))
        basis_stage = dict(basis_structure.get("basis_discovery_stage", {}) or {})
        basis_context = dict(basis_structure.get("basis_context", {}) or {})
        assembler_structure = dict(payload.get("assembler_structure", {}))
        gate_basis = dict(payload.get("piecewise_gate_basis", {}))

        self.assertEqual(str(orthogonality.get("status")), "reported")
        self.assertAlmostEqual(float(orthogonality.get("orthogonality_score", 0.0)), 0.76, places=6)
        self.assertEqual(str(residual.get("status")), "reported")
        self.assertAlmostEqual(
            float(dict(residual.get("recorded", {})).get("mean_marginal_r2_gain", 0.0)),
            0.31,
            places=6,
        )
        self.assertEqual(str(semantic.get("status")), "reported")
        self.assertAlmostEqual(
            float(dict(semantic.get("recorded", {})).get("semantic_unique_ratio", 0.0)),
            1.0,
            places=6,
        )
        self.assertEqual(str(basis_stage.get("structure_head")), "basis_set")
        self.assertEqual(str(dict(assembler_structure.get("stage_protocol", {}) or {}).get("structure_head")), "expression")
        self.assertEqual(str(assembler_structure.get("search_input_space")), "basis_object_space")
        self.assertEqual(str(assembler_structure.get("pool_expansion_unit")), "basis_object")
        self.assertEqual(str(assembler_structure.get("gradient_guidance_mode")), "basis_object_gradient")
        self.assertTrue(bool(dict(assembler_structure.get("object_gradient_pool", {}) or {})))
        self.assertEqual(str(basis_context.get("basis_source")), "orthogonal_basis_discovery")
        self.assertEqual(str(gate_basis.get("status")), "enabled")
        self.assertEqual(int(gate_basis.get("gate_basis_count", 0)), 1)


if __name__ == "__main__":
    unittest.main()
