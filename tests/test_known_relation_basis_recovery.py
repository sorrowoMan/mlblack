from __future__ import annotations

import unittest

from core.symbolic.truth_contracts import build_truth_contract_recovery, term_row_view
from examples.known_relation_benchmark_suite import build_known_relation_bundle, known_relation_benchmark_keys


class TestKnownRelationBasisRecovery(unittest.TestCase):
    def test_truth_recovery_distinguishes_strict_phase_and_family_matches(self) -> None:
        summary = build_truth_contract_recovery(
            truth_formula={
                "expression": "signal = sin(phase_angle) + relu(phase_angle-0.45) - material_bias",
                "basis_contract": (
                    "sin(phase_angle)",
                    "piecewise_hinge(phase_angle)",
                    "material_bias",
                ),
                "phase_equivalent_contract": (
                    "periodic_phase_equivalent(phase_angle)",
                    "piecewise_hinge(phase_angle)",
                    "material_bias",
                ),
                "family_level_contract": (
                    "periodic_family(phase_angle)",
                    "piecewise_gate_family(phase_angle)",
                    "linear_feature_family(material_bias)",
                ),
            },
            basis_rows=[
                term_row_view(
                    {
                        "expression_named": "cos(phase_angle)",
                        "feature_names": ["phase_angle"],
                        "semantic_family": "single_feature_periodic",
                        "semantic_signature": "unary:cos(feature:0)",
                    }
                ),
                term_row_view(
                    {
                        "expression_named": "((0.5)*(((((phase_angle)-(0.45)))+(abs(((phase_angle)-(0.45)))))))",
                        "feature_names": ["phase_angle"],
                    }
                ),
                term_row_view(
                    {
                        "expression_named": "material_bias",
                        "feature_names": ["material_bias"],
                        "semantic_family": "linear_feature",
                        "coefficient": -0.18,
                        "normalized_weight": 0.05,
                    }
                ),
            ],
            active_term_rows=[
                term_row_view(
                    {
                        "expression_named": "cos(phase_angle)",
                        "feature_names": ["phase_angle"],
                        "semantic_family": "single_feature_periodic",
                        "semantic_signature": "unary:cos(feature:0)",
                        "coefficient": 0.58,
                        "normalized_weight": 0.3,
                    }
                ),
                term_row_view(
                    {
                        "expression_named": "((0.5)*(((((phase_angle)-(0.45)))+(abs(((phase_angle)-(0.45)))))))",
                        "feature_names": ["phase_angle"],
                        "coefficient": 0.44,
                        "normalized_weight": 0.26,
                    }
                ),
                term_row_view(
                    {
                        "expression_named": "material_bias",
                        "feature_names": ["material_bias"],
                        "semantic_family": "linear_feature",
                        "coefficient": -0.18,
                        "normalized_weight": 0.05,
                    }
                ),
            ],
        )

        self.assertAlmostEqual(float(summary["exact_term_recovery_score"]), 2.0 / 3.0, places=6)
        self.assertAlmostEqual(float(summary["phase_equivalent_term_recovery_score"]), 1.0, places=6)
        self.assertAlmostEqual(float(summary["family_level_term_recovery_score"]), 1.0, places=6)

    def test_benchmark_suite_registers_all_formal_scenarios(self) -> None:
        expected = {
            "ohm_like",
            "ideal_gas_like",
            "arrhenius_gate_like",
            "periodic_gate_like",
            "redundant_proxy_control",
            "coupled_reaction_transport_like",
        }
        self.assertEqual(set(known_relation_benchmark_keys()), expected)
        for scenario in expected:
            definition, bundle, truth = build_known_relation_bundle(
                benchmark_key=scenario,
                n_total=128,
                train_ratio=0.75,
                noise_std=0.01,
                seed=7,
            )
            truth_formula = dict(bundle.metadata.get("truth_formula", {}) or {})
            self.assertEqual(str(definition.key), scenario)
            self.assertTrue(tuple(truth_formula.get("strict_contract", ()) or ()))
            self.assertTrue(tuple(truth_formula.get("phase_equivalent_contract", ()) or ()))
            self.assertTrue(tuple(truth_formula.get("family_level_contract", ()) or ()))
            self.assertEqual(dict(truth).get("formula"), truth_formula)

    def test_truth_recovery_reports_outer_chart_vs_inner_realization(self) -> None:
        summary = build_truth_contract_recovery(
            truth_formula={
                "expression": "pressure = amount * temperature / volume",
                "basis_contract": ("product_ratio(amount,temperature,volume)",),
            },
            basis_rows=[
                term_row_view(
                    {
                        "expression_named": "(amount)/(volume)",
                        "feature_names": ["amount", "volume"],
                        "semantic_family": "ratio_or_reciprocal",
                        "selection_channel": "canonical_trunk",
                        "chart_signature": "identity",
                    }
                ),
                term_row_view(
                    {
                        "expression_named": "temperature",
                        "feature_names": ["temperature"],
                        "semantic_family": "linear_feature",
                        "selection_channel": "native_trunk",
                    }
                ),
            ],
            active_term_rows=[
                term_row_view(
                    {
                        "expression_named": "((amount)*(temperature/volume))",
                        "feature_names": ["amount", "temperature", "volume"],
                        "semantic_family": "product_ratio",
                        "coefficient": 2.0,
                        "normalized_weight": 0.8,
                    }
                )
            ],
        )

        self.assertAlmostEqual(float(summary["outer_chart_hit_score"]), 0.0, places=6)
        self.assertAlmostEqual(float(summary["inner_realization_hit_score"]), 1.0, places=6)
        self.assertAlmostEqual(float(summary["inner_realization_only_score"]), 1.0, places=6)

    def test_exp_ratio_exact_recovery_requires_chart_direction(self) -> None:
        summary = build_truth_contract_recovery(
            truth_formula={
                "expression": "rate = exp(-activation_energy / temperature)",
                "basis_contract": ("exp_ratio(activation_energy,temperature)",),
                "family_level_contract": ("exp_ratio_family(activation_energy,temperature)",),
            },
            basis_rows=[],
            active_term_rows=[
                term_row_view(
                    {
                        "expression_named": "exp(((-1)*(((temperature)/(activation_energy)))))",
                        "feature_names": ["activation_energy", "temperature"],
                        "semantic_family": "basis_realization",
                        "semantic_signature": "unary:exp_neg(binary:div(feature:temperature,feature:activation_energy))",
                        "coefficient": -0.99,
                        "normalized_weight": 0.5,
                    }
                )
            ],
        )

        self.assertAlmostEqual(float(summary["exact_term_recovery_score"]), 0.0, places=6)
        self.assertAlmostEqual(float(summary["family_level_term_recovery_score"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
