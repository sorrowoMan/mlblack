from __future__ import annotations

import unittest
from collections import Counter
from types import SimpleNamespace

import numpy as np

from core.symbolic import annotate_basis_entries, build_core_basis_table, select_locked_core_seed_genome
from core.symbolic.orthogonal_basis_search import (
    OrthogonalBasisSearchConfig,
    ScreenedCandidate,
    _accept_candidate,
    _build_assembler_object_space,
    _collect_object_realization_specs,
    _build_interference_context,
    _build_candidate_objects,
    _enforce_proxy_representative_screen,
    _build_periodic_context,
    _build_periodic_equivalence_report,
    _build_realization_evidence_registry,
    _build_regional_branch_evidence_specs,
    _build_regional_correction_report,
    _decompose_information_source_view,
    _discover_group_candidates,
    _expr_is_native_trunk_root,
    _group_summary_payload,
    _increment_feature_reuse_budget,
    _run_mandatory_realization_closure,
    _screen_candidate_pool,
    fit_orthogonal_basis_symbolic,
)
from core.symbolic.symbolic_structure_search import StructureSearchConfig, StructureSearchResult, evaluate_genome_with_ridge


class TestSymbolicBasisConsensus(unittest.TestCase):
    def test_phase_and_family_consensus_group_periodic_variants(self) -> None:
        run0_entries = annotate_basis_entries(
            [
                {
                    "term_name": "sin_phase",
                    "expression_named": "sin(phase_angle)",
                    "feature_names": ["phase_angle"],
                    "semantic_family": "single_feature_periodic",
                    "semantic_signature": "unary:sin(feature:0)",
                },
                {
                    "term_name": "primary_signal",
                    "expression_named": "primary_signal",
                    "feature_names": ["primary_signal"],
                    "semantic_family": "linear_feature",
                    "semantic_signature": "feature:1",
                },
            ],
            (
                {"name": "sin_phase", "expr": {"type": "unary", "op": "sin", "arg": {"type": "feature", "index": 0}}},
                {"name": "primary_signal", "expr": {"type": "feature", "index": 1}},
            ),
        )
        run1_entries = annotate_basis_entries(
            [
                {
                    "term_name": "cos_phase",
                    "expression_named": "cos(phase_angle)",
                    "feature_names": ["phase_angle"],
                    "semantic_family": "single_feature_periodic",
                    "semantic_signature": "unary:cos(feature:0)",
                },
                {
                    "term_name": "primary_signal",
                    "expression_named": "primary_signal",
                    "feature_names": ["primary_signal"],
                    "semantic_family": "linear_feature",
                    "semantic_signature": "feature:1",
                },
            ],
            (
                {"name": "cos_phase", "expr": {"type": "unary", "op": "cos", "arg": {"type": "feature", "index": 0}}},
                {"name": "primary_signal", "expr": {"type": "feature", "index": 1}},
            ),
        )
        run2_entries = annotate_basis_entries(
            [
                {
                    "term_name": "sin_phase",
                    "expression_named": "sin(phase_angle)",
                    "feature_names": ["phase_angle"],
                    "semantic_family": "single_feature_periodic",
                    "semantic_signature": "unary:sin(feature:0)",
                },
                {
                    "term_name": "drift_bias",
                    "expression_named": "drift_bias",
                    "feature_names": ["drift_bias"],
                    "semantic_family": "linear_feature",
                    "semantic_signature": "feature:2",
                },
            ],
            (
                {"name": "sin_phase", "expr": {"type": "unary", "op": "sin", "arg": {"type": "feature", "index": 0}}},
                {"name": "drift_bias", "expr": {"type": "feature", "index": 2}},
            ),
        )
        runs = (
            {"run_index": 0, "basis_entries": run0_entries},
            {"run_index": 1, "basis_entries": run1_entries},
            {"run_index": 2, "basis_entries": run2_entries},
        )

        strict_table = build_core_basis_table(
            runs=runs,
            equivalence_mode="strict",
            min_support_rate=0.6,
        )
        phase_table = build_core_basis_table(
            runs=runs,
            equivalence_mode="phase",
            min_support_rate=0.6,
        )
        family_table = build_core_basis_table(
            runs=runs,
            equivalence_mode="family",
            min_support_rate=0.6,
        )

        strict_top = strict_table[0]
        self.assertEqual(str(strict_top.get("representative_expression")), "sin(phase_angle)")
        self.assertEqual(int(strict_top.get("support_count", 0)), 2)

        phase_top = phase_table[0]
        self.assertEqual(str(phase_top.get("basis_class_id")), "periodic_phase_equivalent(phase_angle)")
        self.assertEqual(int(phase_top.get("support_count", 0)), 3)
        self.assertTrue(bool(phase_top.get("selected_as_core")))

        family_top = family_table[0]
        self.assertEqual(str(family_top.get("basis_class_id")), "periodic_family(phase_angle)")
        self.assertEqual(int(family_top.get("support_count", 0)), 3)
        self.assertTrue(bool(family_top.get("selected_as_core")))
        self.assertAlmostEqual(float(family_top.get("exact_stability", 0.0)), 2.0 / 3.0, places=6)
        self.assertAlmostEqual(float(family_top.get("multi_run_core_frequency", 0.0)), 1.0, places=6)
        self.assertAlmostEqual(float(family_top.get("joint_core_score", 0.0)), 0.9, places=6)

        locked = select_locked_core_seed_genome(
            runs=runs,
            equivalence_mode="family",
            min_support_rate=0.6,
            max_terms=2,
        )
        seed_genome = tuple(locked.get("seed_genome", ()) or ())
        self.assertEqual(len(seed_genome), 2)
        self.assertEqual(str(seed_genome[0].get("expr", {}).get("op")), "sin")
        self.assertEqual(str(seed_genome[1].get("name")), "primary_signal")
        selected_rows = tuple(locked.get("selected_core_rows", ()) or ())
        self.assertEqual(len(selected_rows), 2)
        self.assertAlmostEqual(float(selected_rows[0].get("representative_exact_support_rate", 0.0)), 2.0 / 3.0, places=6)
        self.assertEqual(str(selected_rows[0].get("selection_source")), "consensus")
        strategy = dict(locked.get("selection_strategy", {}) or {})
        weights = dict(strategy.get("joint_core_score_weights", {}) or {})
        self.assertAlmostEqual(float(weights.get("support_rate", 0.0)), 0.5, places=6)
        self.assertAlmostEqual(float(weights.get("exact_stability", 0.0)), 0.3, places=6)
        self.assertAlmostEqual(float(weights.get("support_weight_rate", 0.0)), 0.2, places=6)

    def test_lock_seed_basis_only_emits_seeded_groups(self) -> None:
        x0 = np.linspace(-1.0, 1.0, 48)
        x1 = np.sin(np.linspace(-np.pi, np.pi, 48))
        x2 = np.cos(np.linspace(-np.pi, np.pi, 48))
        x3 = np.maximum(x0 - 0.15, 0.0)
        train_matrix = np.stack([x0, x1, x2, x3], axis=1)
        y_train = (1.2 * x0 + 0.7 * x1 + 0.4 * x3).reshape(-1, 1)
        screened = [
            ScreenedCandidate(
                pool_index=0,
                screen_index=0,
                name="x0",
                expr={"type": "feature", "index": 0},
                family="linear",
                complexity=1.0,
                features=(0,),
                target_corr=0.80,
                screen_score=0.80,
                expression="x0",
                semantic_signature="feature:0",
                semantic_family="linear_feature",
                uses_piecewise_gate=False,
            ),
            ScreenedCandidate(
                pool_index=1,
                screen_index=1,
                name="sin_x1",
                expr={"type": "unary", "op": "sin", "arg": {"type": "feature", "index": 1}},
                family="periodic",
                complexity=2.0,
                features=(1,),
                target_corr=0.72,
                screen_score=0.72,
                expression="sin(x1)",
                semantic_signature="unary:sin(feature:1)",
                semantic_family="single_feature_periodic",
                uses_piecewise_gate=False,
            ),
            ScreenedCandidate(
                pool_index=2,
                screen_index=2,
                name="cos_x2",
                expr={"type": "unary", "op": "cos", "arg": {"type": "feature", "index": 2}},
                family="periodic",
                complexity=2.0,
                features=(2,),
                target_corr=0.34,
                screen_score=0.34,
                expression="cos(x2)",
                semantic_signature="unary:cos(feature:2)",
                semantic_family="single_feature_periodic",
                uses_piecewise_gate=False,
            ),
            ScreenedCandidate(
                pool_index=3,
                screen_index=3,
                name="hinge_x0",
                expr={"type": "piecewise", "feature": 0, "cut": 0.15},
                family="piecewise",
                complexity=3.0,
                features=(0,),
                target_corr=0.46,
                screen_score=0.46,
                expression="relu(x0-0.15)",
                semantic_signature="piecewise:hinge(feature:0)",
                semantic_family="piecewise_gate",
                uses_piecewise_gate=True,
            ),
        ]
        groups = _discover_group_candidates(
            screened=screened,
            train_matrix=train_matrix,
            y_train=y_train,
            raw_X=train_matrix,
            feature_names=("x0", "x1", "x2", "x3"),
            interference_context=_build_interference_context(
                raw_X=train_matrix,
                feature_names=("x0", "x1", "x2", "x3"),
                data_metadata=None,
            ),
            periodic_context={},
            cfg=OrthogonalBasisSearchConfig(
                seed_candidate_count=4,
                group_count=6,
                min_basis_count=2,
                max_basis_count=3,
                max_pair_abs_corr=0.95,
                random_seed=17,
                greedy_choice_topk=3,
                random_group_trials=4,
                lock_seed_basis=True,
            ).normalized(),
            seed_genome=(
                {"name": "sin_x1", "expr": {"type": "unary", "op": "sin", "arg": {"type": "feature", "index": 1}}},
            ),
        )
        self.assertTrue(groups)
        for group in groups:
            pool_indices = tuple(int(value) for value in tuple(group.get("pool_indices", ()) or ()))
            self.assertIn(1, pool_indices)

    def test_joint_core_score_tracks_cross_lane_stability_when_lanes_exist(self) -> None:
        run0_entries = annotate_basis_entries(
            [
                {
                    "term_name": "arrhenius_basis",
                    "expression_named": "exp(-activation_energy/temperature)",
                    "feature_names": ["activation_energy", "temperature"],
                    "semantic_family": "arrhenius_ratio",
                    "semantic_signature": "unary:exp(binary:div(feature:0,feature:1))",
                }
            ],
            (
                {
                    "name": "arrhenius_basis",
                    "expr": {
                        "type": "unary",
                        "op": "exp",
                        "arg": {
                            "type": "binary",
                            "op": "div",
                            "left": {"type": "feature", "index": 0},
                            "right": {"type": "feature", "index": 1},
                        },
                    },
                },
            ),
        )
        run1_entries = annotate_basis_entries(
            [
                {
                    "term_name": "arrhenius_basis",
                    "expression_named": "exp(-activation_energy/temperature)",
                    "feature_names": ["activation_energy", "temperature"],
                    "semantic_family": "arrhenius_ratio",
                    "semantic_signature": "unary:exp(binary:div(feature:0,feature:1))",
                }
            ],
            (
                {
                    "name": "arrhenius_basis",
                    "expr": {
                        "type": "unary",
                        "op": "exp",
                        "arg": {
                            "type": "binary",
                            "op": "div",
                            "left": {"type": "feature", "index": 0},
                            "right": {"type": "feature", "index": 1},
                        },
                    },
                },
            ),
        )
        runs = (
            {
                "run_index": 0,
                "run_id": "lane_a_0",
                "lane_id": "mechanistic_gate",
                "lane_family": "mechanistic",
                "basis_entries": run0_entries,
            },
            {
                "run_index": 1,
                "run_id": "lane_b_0",
                "lane_id": "family_diverse",
                "lane_family": "diversity",
                "basis_entries": run1_entries,
            },
        )

        family_table = build_core_basis_table(
            runs=runs,
            equivalence_mode="family",
            min_support_rate=0.5,
        )
        top = family_table[0]
        self.assertEqual(int(top.get("cross_lane_support_count", 0)), 2)
        self.assertAlmostEqual(float(top.get("cross_lane_support_rate") or 0.0), 1.0, places=6)
        self.assertEqual(int(top.get("cross_lane_family_count", 0)), 2)
        self.assertAlmostEqual(float(top.get("cross_lane_family_support_rate") or 0.0), 1.0, places=6)
        self.assertAlmostEqual(float(top.get("cross_lane_stability") or 0.0), 1.0, places=6)
        self.assertAlmostEqual(float(top.get("joint_core_score", 0.0)), 1.0, places=6)

        locked = select_locked_core_seed_genome(
            runs=runs,
            equivalence_mode="family",
            min_support_rate=0.5,
            max_terms=2,
        )
        strategy = dict(locked.get("selection_strategy", {}) or {})
        weights = dict(strategy.get("joint_core_score_weights", {}) or {})
        self.assertAlmostEqual(float(weights.get("support_rate", 0.0)), 0.4, places=6)
        self.assertAlmostEqual(float(weights.get("exact_stability", 0.0)), 0.25, places=6)
        self.assertAlmostEqual(float(weights.get("support_weight_rate", 0.0)), 0.15, places=6)
        self.assertAlmostEqual(float(weights.get("cross_lane_stability", 0.0)), 0.2, places=6)
        self.assertEqual(int(strategy.get("lane_count", 0)), 2)
        self.assertEqual(int(strategy.get("lane_family_count", 0)), 2)

    def test_gate_requirement_only_registers_gate_groups(self) -> None:
        x0 = np.linspace(-1.0, 1.0, 64)
        x1 = np.linspace(0.2, 2.2, 64)
        gate = np.maximum(x1 - 1.0, 0.0)
        train_matrix = np.stack([x0, x1, gate], axis=1)
        y_train = (1.4 * x0 + 0.8 * gate).reshape(-1, 1)
        screened = [
            ScreenedCandidate(
                pool_index=0,
                screen_index=0,
                name="x0",
                expr={"type": "feature", "index": 0},
                family="linear",
                complexity=1.0,
                features=(0,),
                target_corr=0.92,
                screen_score=0.92,
                expression="x0",
                semantic_signature="feature:0",
                semantic_family="linear_feature",
                uses_piecewise_gate=False,
            ),
            ScreenedCandidate(
                pool_index=1,
                screen_index=1,
                name="x1",
                expr={"type": "feature", "index": 1},
                family="linear",
                complexity=1.0,
                features=(1,),
                target_corr=0.78,
                screen_score=0.78,
                expression="x1",
                semantic_signature="feature:1",
                semantic_family="linear_feature",
                uses_piecewise_gate=False,
            ),
            ScreenedCandidate(
                pool_index=2,
                screen_index=2,
                name="hinge_x1",
                expr={"type": "piecewise", "feature": 1, "cut": 1.0},
                family="piecewise",
                complexity=2.5,
                features=(1,),
                target_corr=0.55,
                screen_score=0.55,
                expression="relu(x1-1.0)",
                semantic_signature="piecewise:hinge(feature:1)",
                semantic_family="piecewise_gate",
                uses_piecewise_gate=True,
            ),
        ]
        groups = _discover_group_candidates(
            screened=screened,
            train_matrix=train_matrix,
            y_train=y_train,
            raw_X=train_matrix,
            feature_names=("x0", "x1", "x2"),
            interference_context=_build_interference_context(
                raw_X=train_matrix,
                feature_names=("x0", "x1", "x2"),
                data_metadata=None,
            ),
            periodic_context={},
            cfg=OrthogonalBasisSearchConfig(
                seed_candidate_count=3,
                group_count=4,
                min_basis_count=2,
                max_basis_count=3,
                max_pair_abs_corr=0.95,
                random_seed=11,
                require_gate_candidate_in_group=True,
                min_gate_basis_terms=1,
                piecewise_gate_bonus=0.25,
            ).normalized(),
            seed_genome=None,
        )
        self.assertTrue(groups)
        for group in groups:
            rows = tuple(group.get("rows", ()) or ())
            self.assertTrue(any(bool(row.uses_piecewise_gate) for row in rows))
            mechanism_summary = dict(group.get("mechanism_summary", {}) or {})
            self.assertGreaterEqual(int(mechanism_summary.get("gate_term_count", 0)), 1)
            self.assertTrue(bool(mechanism_summary.get("gate_requirement_satisfied")))

    def test_periodic_requirement_only_registers_periodic_groups(self) -> None:
        rng = np.random.default_rng(31)
        phase = np.linspace(-np.pi, np.pi, 96)
        x0 = rng.uniform(-1.0, 1.0, 96)
        train_matrix = np.stack([x0, np.sin(phase), np.cos(phase)], axis=1)
        y_train = (1.1 * x0 + 0.8 * np.sin(phase)).reshape(-1, 1)
        screened = [
            ScreenedCandidate(
                pool_index=0,
                screen_index=0,
                name="x0",
                expr={"type": "feature", "index": 0},
                family="linear",
                complexity=1.0,
                features=(0,),
                target_corr=0.86,
                screen_score=0.86,
                expression="x0",
                semantic_signature="feature:0",
                semantic_family="linear_feature",
                uses_piecewise_gate=False,
            ),
            ScreenedCandidate(
                pool_index=1,
                screen_index=1,
                name="sin_phase",
                expr={"type": "unary", "op": "sin", "arg": {"type": "feature", "index": 1}},
                family="periodic",
                complexity=2.0,
                features=(1,),
                target_corr=0.72,
                screen_score=0.72,
                expression="sin(phase_angle)",
                semantic_signature="unary:sin(feature:1)",
                semantic_family="single_feature_periodic",
                uses_piecewise_gate=False,
            ),
            ScreenedCandidate(
                pool_index=2,
                screen_index=2,
                name="cos_phase",
                expr={"type": "unary", "op": "cos", "arg": {"type": "feature", "index": 2}},
                family="periodic",
                complexity=2.0,
                features=(2,),
                target_corr=0.32,
                screen_score=0.32,
                expression="cos(phase_angle)",
                semantic_signature="unary:cos(feature:2)",
                semantic_family="single_feature_periodic",
                uses_piecewise_gate=False,
            ),
        ]
        periodic_context = {"periodic_feature_names": ("phase_angle",)}
        groups = _discover_group_candidates(
            screened=screened,
            train_matrix=train_matrix,
            y_train=y_train,
            raw_X=np.stack([x0, phase, phase], axis=1),
            feature_names=("x0", "phase_angle", "phase_angle_alt"),
            interference_context=_build_interference_context(
                raw_X=np.stack([x0, phase, phase], axis=1),
                feature_names=("x0", "phase_angle", "phase_angle_alt"),
                data_metadata=None,
            ),
            periodic_context=periodic_context,
            cfg=OrthogonalBasisSearchConfig(
                seed_candidate_count=3,
                group_count=4,
                min_basis_count=2,
                max_basis_count=3,
                max_pair_abs_corr=0.95,
                random_seed=23,
                require_periodic_candidate_in_group=True,
                min_periodic_basis_terms=1,
            ).normalized(),
            seed_genome=None,
        )
        self.assertTrue(groups)
        for group in groups:
            rows = tuple(group.get("rows", ()) or ())
            self.assertTrue(any("periodic" in str(row.semantic_family) for row in rows))
            mechanism_summary = dict(group.get("mechanism_summary", {}) or {})
            self.assertGreaterEqual(int(mechanism_summary.get("periodic_term_count", 0)), 1)

    def test_cross_explanatory_rejection_blocks_proxy_duplication(self) -> None:
        rng = np.random.default_rng(7)
        primary = rng.uniform(-1.8, 1.8, 96)
        proxy = primary + rng.normal(0.0, 0.04, size=primary.shape[0])
        phase = rng.uniform(-np.pi, np.pi, 96)
        gate = np.maximum(primary - 0.15, 0.0)
        train_matrix = np.stack([primary, proxy, np.sin(phase), gate], axis=1)
        y_train = (1.25 * primary + 0.60 * np.sin(phase) + 0.40 * gate).reshape(-1, 1)
        feature_names = ("primary_signal", "primary_signal_proxy", "phase_angle", "primary_gate")
        interference_context = _build_interference_context(
            raw_X=np.stack([primary, proxy, phase, gate], axis=1),
            feature_names=feature_names,
            data_metadata={
                "redundant_feature_groups": {
                    "signal_group": ("primary_signal", "primary_signal_proxy"),
                }
            },
        )
        screened = [
            ScreenedCandidate(
                pool_index=0,
                screen_index=0,
                name="primary_signal",
                expr={"type": "feature", "index": 0},
                family="linear",
                complexity=1.0,
                features=(0,),
                target_corr=0.95,
                screen_score=0.95,
                expression="primary_signal",
                semantic_signature="feature:0",
                semantic_family="linear_feature",
                uses_piecewise_gate=False,
            ),
            ScreenedCandidate(
                pool_index=1,
                screen_index=1,
                name="primary_signal_proxy",
                expr={"type": "feature", "index": 1},
                family="linear",
                complexity=1.0,
                features=(1,),
                target_corr=0.93,
                screen_score=0.93,
                expression="primary_signal_proxy",
                semantic_signature="feature:1",
                semantic_family="linear_feature",
                uses_piecewise_gate=False,
            ),
            ScreenedCandidate(
                pool_index=2,
                screen_index=2,
                name="sin_phase",
                expr={"type": "unary", "op": "sin", "arg": {"type": "feature", "index": 2}},
                family="periodic",
                complexity=2.0,
                features=(2,),
                target_corr=0.62,
                screen_score=0.62,
                expression="sin(phase_angle)",
                semantic_signature="unary:sin(feature:2)",
                semantic_family="single_feature_periodic",
                uses_piecewise_gate=False,
            ),
            ScreenedCandidate(
                pool_index=3,
                screen_index=3,
                name="primary_gate",
                expr={"type": "piecewise", "feature": 0, "cut": 0.15},
                family="piecewise",
                complexity=2.5,
                features=(0,),
                target_corr=0.51,
                screen_score=0.51,
                expression="relu(primary_signal-0.15)",
                semantic_signature="piecewise:hinge(feature:0)",
                semantic_family="piecewise_gate",
                uses_piecewise_gate=True,
            ),
        ]
        groups = _discover_group_candidates(
            screened=screened,
            train_matrix=train_matrix,
            y_train=y_train,
            raw_X=np.stack([primary, proxy, phase, gate], axis=1),
            feature_names=feature_names,
            interference_context=interference_context,
            periodic_context={},
            cfg=OrthogonalBasisSearchConfig(
                seed_candidate_count=4,
                group_count=6,
                min_basis_count=2,
                max_basis_count=3,
                max_pair_abs_corr=0.98,
                random_seed=13,
                greedy_choice_topk=2,
                random_group_trials=2,
                cross_explanatory_rejection_mode="proxy_group_hard",
                trivial_nonlinearity_penalty_mode="proxy_group_explainability_penalty",
            ).normalized(),
            seed_genome=None,
        )
        self.assertTrue(groups)
        for group in groups:
            pool_indices = set(int(value) for value in tuple(group.get("pool_indices", ()) or ()))
            self.assertFalse({0, 1}.issubset(pool_indices))

    def test_screen_candidate_pool_keeps_single_proxy_representative(self) -> None:
        rng = np.random.default_rng(29)
        primary = rng.uniform(-1.5, 1.5, 96)
        proxy = primary + rng.normal(0.0, 0.03, size=primary.shape[0])
        phase = rng.uniform(-np.pi, np.pi, 96)
        x_train = np.stack([primary, proxy, phase], axis=1)
        y_train = (1.2 * primary + 0.6 * np.sin(phase)).reshape(-1, 1)
        candidates = [
            SimpleNamespace(
                name="primary_signal",
                expr={"type": "feature", "index": 0},
                family="linear",
                complexity=1.0,
                features=(0,),
            ),
            SimpleNamespace(
                name="primary_signal_proxy",
                expr={"type": "feature", "index": 1},
                family="linear",
                complexity=1.0,
                features=(1,),
            ),
            SimpleNamespace(
                name="sin_phase",
                expr={"type": "unary", "op": "sin", "arg": {"type": "feature", "index": 2}},
                family="periodic",
                complexity=2.0,
                features=(2,),
            ),
        ]
        screened, _matrix = _screen_candidate_pool(
            candidates=candidates,
            X_train=x_train,
            y_train=y_train,
            feature_names=("primary_signal", "primary_signal_proxy", "phase_angle"),
            candidate_limit=3,
            cfg=OrthogonalBasisSearchConfig(
                candidate_limit=3,
                seed_candidate_count=3,
                group_count=2,
                min_basis_count=2,
                max_basis_count=3,
                cross_explanatory_rejection_mode="proxy_group_hard",
                trivial_nonlinearity_penalty_mode="proxy_group_explainability_penalty",
            ).normalized(),
            graph_cache=None,
            interference_context=_build_interference_context(
                raw_X=x_train,
                feature_names=("primary_signal", "primary_signal_proxy", "phase_angle"),
                data_metadata={
                    "redundant_feature_groups": {
                        "signal_group": ("primary_signal", "primary_signal_proxy"),
                    }
                },
            ),
            periodic_context={"periodic_feature_names": ("phase_angle",)},
        )
        screened_names = {str(row.name) for row in screened}
        self.assertTrue(bool(screened_names & {"primary_signal", "primary_signal_proxy"}))
        self.assertFalse({"primary_signal", "primary_signal_proxy"}.issubset(screened_names))

    def test_proxy_trunk_disqualification_blocks_wrapped_proxy_representatives_when_native_exists(self) -> None:
        primary_row = ScreenedCandidate(
            pool_index=0,
            screen_index=0,
            name="primary_signal",
            expr={"type": "feature", "index": 0},
            family="linear",
            complexity=1.0,
            features=(0,),
            target_corr=0.82,
            screen_score=0.80,
            expression="primary_signal",
            semantic_signature="feature:0",
            semantic_family="linear_feature",
            uses_piecewise_gate=False,
            residual_gain=0.18,
            native_trunk_root=True,
            native_trunk_floor_passed=True,
            native_trunk_global_gain=0.18,
            native_trunk_interval_min_gain=0.07,
            native_trunk_interval_mean_gain=0.09,
            selection_channel="native_trunk",
        )
        wrapped_proxy_row = ScreenedCandidate(
            pool_index=1,
            screen_index=1,
            name="tanh_proxy",
            expr={"type": "unary", "op": "tanh", "arg": {"type": "feature", "index": 1}},
            family="tanh",
            complexity=2.0,
            features=(1,),
            target_corr=0.93,
            screen_score=0.92,
            expression="tanh(primary_signal_proxy)",
            semantic_signature="unary:tanh(feature:1)",
            semantic_family="single_feature_transform",
            uses_piecewise_gate=False,
            residual_gain=0.24,
            selection_channel="challenger",
        )
        gate_proxy_row = ScreenedCandidate(
            pool_index=2,
            screen_index=2,
            name="primary_gate",
            expr={"type": "piecewise", "feature": 0, "cut": 0.1, "family": "piecewise_hinge"},
            family="piecewise_hinge",
            complexity=3.0,
            features=(0,),
            target_corr=0.95,
            screen_score=0.94,
            expression="piecewise_hinge(primary_signal)",
            semantic_signature="piecewise:primary_signal",
            semantic_family="piecewise_gate",
            uses_piecewise_gate=True,
            residual_gain=0.27,
            selection_channel="regional_speciality",
        )
        selected = _enforce_proxy_representative_screen(
            limited_rows=(
                (wrapped_proxy_row, np.zeros(16, dtype=float)),
                (primary_row, np.zeros(16, dtype=float)),
                (gate_proxy_row, np.zeros(16, dtype=float)),
            ),
            full_ranked_rows=(
                (wrapped_proxy_row, np.zeros(16, dtype=float)),
                (primary_row, np.zeros(16, dtype=float)),
                (gate_proxy_row, np.zeros(16, dtype=float)),
            ),
            candidate_limit=3,
            feature_names=("primary_signal", "primary_signal_proxy"),
            interference_context=_build_interference_context(
                raw_X=np.zeros((16, 2), dtype=float),
                feature_names=("primary_signal", "primary_signal_proxy"),
                data_metadata={
                    "redundant_feature_groups": {
                        "signal_group": ("primary_signal", "primary_signal_proxy"),
                    }
                },
            ),
            cfg=OrthogonalBasisSearchConfig(
                proxy_group_policy="metadata_or_correlation_cluster",
                native_proxy_check_mode="proxy_group_native_election",
                proxy_trunk_disqualification_mode="native_identity_only_when_available",
            ).normalized(),
            periodic_context={},
        )
        selected_names = {str(row.name) for row, _values in tuple(selected)}
        self.assertIn("primary_signal", selected_names)
        self.assertNotIn("tanh_proxy", selected_names)
        self.assertNotIn("primary_gate", selected_names)

    def test_screen_candidate_pool_keeps_internal_structure_variant_distinct(self) -> None:
        rng = np.random.default_rng(41)
        signal = rng.uniform(-1.5, 1.5, 112)
        ratio = rng.uniform(-0.25, 0.25, 112)
        phase = rng.uniform(-np.pi, np.pi, 112)
        x_train = np.stack([signal, ratio, phase], axis=1)
        y_train = (1.6 * signal * ratio + 0.35 * np.sin(phase)).reshape(-1, 1)
        candidates = [
            SimpleNamespace(
                name="interaction_product",
                expr={
                    "type": "binary",
                    "op": "mul",
                    "left": {"type": "feature", "index": 0},
                    "right": {"type": "feature", "index": 1},
                },
                family="interaction",
                complexity=2.5,
                features=(0, 1),
            ),
            SimpleNamespace(
                name="interaction_tanh_surrogate",
                expr={
                    "type": "binary",
                    "op": "mul",
                    "left": {"type": "feature", "index": 0},
                    "right": {
                        "type": "unary",
                        "op": "tanh",
                        "arg": {"type": "feature", "index": 1},
                    },
                },
                family="interaction",
                complexity=3.0,
                features=(0, 1),
            ),
            SimpleNamespace(
                name="sin_phase",
                expr={"type": "unary", "op": "sin", "arg": {"type": "feature", "index": 2}},
                family="periodic",
                complexity=2.0,
                features=(2,),
            ),
        ]
        screened, _matrix = _screen_candidate_pool(
            candidates=candidates,
            X_train=x_train,
            y_train=y_train,
            feature_names=("primary_signal", "ratio_signal", "phase_angle"),
            candidate_limit=3,
            cfg=OrthogonalBasisSearchConfig(
                candidate_limit=3,
                seed_candidate_count=3,
                group_count=2,
                min_basis_count=2,
                max_basis_count=3,
            ).normalized(),
            graph_cache=None,
            interference_context=_build_interference_context(
                raw_X=x_train,
                feature_names=("primary_signal", "ratio_signal", "phase_angle"),
                data_metadata={},
            ),
            periodic_context={"periodic_feature_names": ("phase_angle",)},
        )
        screened_names = {str(row.name) for row in screened}
        structure_variant_names = {"interaction_product", "interaction_tanh_surrogate"}
        self.assertTrue(structure_variant_names.issubset(screened_names))
        representative_rows = [row for row in screened if str(row.name) in structure_variant_names]
        self.assertEqual(len(representative_rows), 2)
        for row in representative_rows:
            self.assertEqual(int(row.screen_cluster_size), 1)

    def test_screen_candidate_pool_collapses_ratio_transforms_into_single_object(self) -> None:
        rng = np.random.default_rng(43)
        numerator = rng.uniform(0.6, 2.4, 120)
        denominator = rng.uniform(0.8, 2.0, 120)
        phase = rng.uniform(-np.pi, np.pi, 120)
        x_train = np.stack([numerator, denominator, phase], axis=1)
        ratio_values = numerator / denominator
        y_train = (1.45 * ratio_values + 0.30 * np.sin(phase)).reshape(-1, 1)
        candidates = [
            SimpleNamespace(
                name="ratio_signal",
                expr={
                    "type": "binary",
                    "op": "div",
                    "left": {"type": "feature", "index": 0},
                    "right": {"type": "feature", "index": 1},
                },
                family="ratio",
                complexity=2.0,
                features=(0, 1),
            ),
            SimpleNamespace(
                name="sin_ratio_signal",
                expr={
                    "type": "unary",
                    "op": "sin",
                    "arg": {
                        "type": "binary",
                        "op": "div",
                        "left": {"type": "feature", "index": 0},
                        "right": {"type": "feature", "index": 1},
                    },
                },
                family="periodic",
                complexity=3.0,
                features=(0, 1),
            ),
            SimpleNamespace(
                name="sin_phase",
                expr={"type": "unary", "op": "sin", "arg": {"type": "feature", "index": 2}},
                family="periodic",
                complexity=2.0,
                features=(2,),
            ),
        ]
        screened, _matrix = _screen_candidate_pool(
            candidates=candidates,
            X_train=x_train,
            y_train=y_train,
            feature_names=("numerator", "denominator", "phase_angle"),
            candidate_limit=3,
            cfg=OrthogonalBasisSearchConfig(
                candidate_limit=3,
                seed_candidate_count=3,
                group_count=2,
                min_basis_count=2,
                max_basis_count=3,
            ).normalized(),
            graph_cache=None,
            interference_context=_build_interference_context(
                raw_X=x_train,
                feature_names=("numerator", "denominator", "phase_angle"),
                data_metadata={},
            ),
            periodic_context={"periodic_feature_names": ("phase_angle",)},
        )
        screened_names = {str(row.name) for row in screened}
        equivalence_names = {"ratio_signal", "sin_ratio_signal"}
        self.assertTrue(bool(screened_names & equivalence_names))
        self.assertFalse(equivalence_names.issubset(screened_names))
        representative_rows = [row for row in screened if str(row.name) in equivalence_names]
        self.assertEqual(len(representative_rows), 1)
        self.assertEqual(str(representative_rows[0].name), "ratio_signal")
        self.assertGreaterEqual(int(representative_rows[0].screen_cluster_size), 2)

    def test_periodic_source_object_collapse_preserves_periodic_evidence(self) -> None:
        rng = np.random.default_rng(47)
        phase = np.linspace(-np.pi, np.pi, 128)
        drift = rng.uniform(-1.0, 1.0, 128)
        x_train = np.stack([phase, drift], axis=1)
        y_train = (0.85 * np.sin(phase) + 0.25 * drift).reshape(-1, 1)
        candidates = [
            SimpleNamespace(
                name="phase_angle",
                expr={"type": "feature", "index": 0},
                family="linear",
                complexity=1.0,
                features=(0,),
            ),
            SimpleNamespace(
                name="sin_phase",
                expr={"type": "unary", "op": "sin", "arg": {"type": "feature", "index": 0}},
                family="periodic",
                complexity=2.0,
                features=(0,),
            ),
            SimpleNamespace(
                name="drift",
                expr={"type": "feature", "index": 1},
                family="linear",
                complexity=1.0,
                features=(1,),
            ),
        ]
        screened, _matrix = _screen_candidate_pool(
            candidates=candidates,
            X_train=x_train,
            y_train=y_train,
            feature_names=("phase_angle", "drift"),
            candidate_limit=3,
            cfg=OrthogonalBasisSearchConfig(
                candidate_limit=3,
                seed_candidate_count=3,
                group_count=2,
                min_basis_count=2,
                max_basis_count=3,
                periodic_candidate_screen_reserve=1,
            ).normalized(),
            graph_cache=None,
            interference_context=_build_interference_context(
                raw_X=x_train,
                feature_names=("phase_angle", "drift"),
                data_metadata={},
            ),
            periodic_context={"periodic_feature_names": ("phase_angle",)},
        )
        screened_names = {str(row.name) for row in screened}
        phase_equivalents = {"phase_angle", "sin_phase"}
        self.assertTrue(bool(screened_names & phase_equivalents))
        self.assertFalse(phase_equivalents.issubset(screened_names))
        periodic_row = next(row for row in screened if str(row.name) in phase_equivalents)
        self.assertTrue(bool(periodic_row.contains_periodic_evidence))
        self.assertGreaterEqual(int(periodic_row.screen_cluster_size), 2)
        candidate_objects = _build_candidate_objects(
            screened=screened,
            feature_names=("phase_angle", "drift"),
            interference_context=_build_interference_context(
                raw_X=x_train,
                feature_names=("phase_angle", "drift"),
                data_metadata={},
            ),
            periodic_context={"periodic_feature_names": ("phase_angle",)},
            outer_search_unit="mechanism_object",
        )
        self.assertTrue(any(str(obj.object_kind) == "periodic_channel" for obj in candidate_objects))

    def test_trivial_nonlinearity_penalty_is_reported_for_proxy_pair(self) -> None:
        rng = np.random.default_rng(11)
        primary = rng.uniform(-2.0, 2.0, 80)
        proxy = primary + rng.normal(0.0, 0.03, size=primary.shape[0])
        phase = rng.uniform(-np.pi, np.pi, 80)
        train_matrix = np.stack([primary, proxy, np.sin(phase)], axis=1)
        y_train = (1.15 * primary + 0.45 * np.sin(phase)).reshape(-1, 1)
        feature_names = ("primary_signal", "primary_signal_proxy", "phase_angle")
        interference_context = _build_interference_context(
            raw_X=np.stack([primary, proxy, phase], axis=1),
            feature_names=feature_names,
            data_metadata={
                "redundant_feature_groups": {
                    "signal_group": ("primary_signal", "primary_signal_proxy"),
                }
            },
        )
        selected_rows = (
            ScreenedCandidate(
                pool_index=0,
                screen_index=0,
                name="primary_signal",
                expr={"type": "feature", "index": 0},
                family="linear",
                complexity=1.0,
                features=(0,),
                target_corr=0.94,
                screen_score=0.94,
                expression="primary_signal",
                semantic_signature="feature:0",
                semantic_family="linear_feature",
                uses_piecewise_gate=False,
            ),
            ScreenedCandidate(
                pool_index=1,
                screen_index=1,
                name="primary_signal_proxy",
                expr={"type": "feature", "index": 1},
                family="linear",
                complexity=1.0,
                features=(1,),
                target_corr=0.92,
                screen_score=0.92,
                expression="primary_signal_proxy",
                semantic_signature="feature:1",
                semantic_family="linear_feature",
                uses_piecewise_gate=False,
            ),
        )
        payload = _group_summary_payload(
            selected_rows=selected_rows,
            threshold=0.98,
            train_matrix=train_matrix,
            target=np.asarray(y_train, dtype=float).reshape(-1),
            raw_X=np.stack([primary, proxy, phase], axis=1),
            feature_names=feature_names,
            interference_context=interference_context,
            periodic_context={},
            cfg=OrthogonalBasisSearchConfig(
                trivial_nonlinearity_penalty_mode="proxy_group_explainability_penalty",
            ).normalized(),
        )
        report = dict(payload.get("interference_feature_report", {}) or {})
        self.assertEqual(int(report.get("suspicious_pair_count", 0)), 1)
        self.assertGreater(float(report.get("trivial_nonlinearity_penalty_mean", 0.0)), 0.10)

    def test_environment_invariance_audit_is_recorded(self) -> None:
        rng = np.random.default_rng(19)
        temperature = rng.uniform(-1.2, 1.8, 120)
        load = rng.uniform(-1.0, 1.0, 120)
        bias = rng.normal(0.0, 0.2, size=120)
        signal = 1.1 * np.sin(temperature) + 0.55 * np.maximum(temperature - 0.2, 0.0) - 0.18 * bias
        y = (signal + rng.normal(0.0, 0.02, size=120)).reshape(-1, 1)
        result = fit_orthogonal_basis_symbolic(
            X=np.stack([temperature, load, bias], axis=1),
            y=y,
            feature_names=("temperature", "load", "bias"),
            cfg=OrthogonalBasisSearchConfig(
                candidate_limit=40,
                seed_candidate_count=10,
                group_count=4,
                min_basis_count=2,
                max_basis_count=4,
                max_pair_abs_corr=0.7,
                selection_mode="rmse_first",
                enable_piecewise_basis=True,
                gate_feature_names=("temperature",),
                environment_invariance_audit_mode="median_split_report",
            ),
            data_metadata={
                "search_hints": {
                    "gate_feature_names": ("temperature",),
                }
            },
        )
        audit = dict(result.metadata.get("environment_invariance_audit", {}) or {})
        self.assertEqual(str(audit.get("status")), "reported")
        self.assertGreaterEqual(int(audit.get("environment_count", 0)), 2)
        handling = dict(result.metadata.get("interference_feature_handling", {}) or {})
        self.assertEqual(
            str(dict(handling.get("environment_invariance_audit", {}) or {}).get("status")),
            "reported",
        )

    def test_fit_orthogonal_basis_symbolic_uses_injected_rational_template_rows_without_pool_index_crash(self) -> None:
        activation_energy = np.linspace(0.9, 3.2, 96)
        temperature = np.linspace(0.8, 3.1, 96)
        x_train = np.stack([activation_energy, temperature], axis=1)
        y_train = (1.6 * np.exp(-activation_energy / temperature)).reshape(-1, 1)
        result = fit_orthogonal_basis_symbolic(
            X=x_train,
            y=y_train,
            feature_names=("activation_energy", "temperature"),
            cfg=OrthogonalBasisSearchConfig(
                candidate_limit=24,
                seed_candidate_count=8,
                group_count=4,
                min_basis_count=1,
                max_basis_count=3,
                selection_mode="rmse_first",
                mechanistic_feature_groups=(("activation_energy", "temperature"),),
                rational_template_pinning_mode="mechanistic_pair_canonical_ratio_injection",
            ),
            data_metadata={
                "search_hints": {
                    "mechanistic_feature_groups": (("activation_energy", "temperature"),),
                }
            },
        )
        outer_basis_genome = tuple(dict(item) for item in tuple(result.metadata.get("orthogonal_outer_basis_genome", ()) or ()))
        self.assertTrue(any(str(item.get("name")) == "orth_rational_template_0_over_1" for item in outer_basis_genome))
        self.assertGreater(float(dict(result.train_metrics).get("r2", 0.0)), 0.99)

    def test_periodic_equivalence_report_prefers_periodic_family_over_local_surrogate(self) -> None:
        phase = np.linspace(-np.pi, np.pi, 160)
        raw_x = phase.reshape(-1, 1)
        train_matrix = np.stack([np.sin(phase), np.tanh(phase)], axis=1)
        target = (0.9 * np.sin(phase) + 0.4 * np.maximum(phase - 0.45, 0.0)).reshape(-1, 1)
        cfg = OrthogonalBasisSearchConfig(
            periodic_feature_names=("phase_angle",),
            periodic_equivalence_disambiguation_mode="center_edge_holdout_penalty",
            phase_spectrum_audit_mode="center_edge_holdout_report",
            periodic_family_prior_mode="semantic_family_boost",
        ).normalized()
        periodic_context = _build_periodic_context(
            raw_X=raw_x,
            feature_names=("phase_angle",),
            data_metadata={"search_hints": {"periodic_feature_names": ("phase_angle",)}},
            cfg=cfg,
        )
        sin_row = ScreenedCandidate(
            pool_index=0,
            screen_index=0,
            name="sin_phase",
            expr={"type": "unary", "op": "sin", "arg": {"type": "feature", "index": 0}},
            family="periodic",
            complexity=2.0,
            features=(0,),
            target_corr=0.8,
            screen_score=0.8,
            expression="sin(phase_angle)",
            semantic_signature="unary:sin(feature:0)",
            semantic_family="single_feature_periodic",
            uses_piecewise_gate=False,
        )
        tanh_row = ScreenedCandidate(
            pool_index=1,
            screen_index=1,
            name="tanh_phase",
            expr={"type": "unary", "op": "tanh", "arg": {"type": "feature", "index": 0}},
            family="nonlinear",
            complexity=2.0,
            features=(0,),
            target_corr=0.7,
            screen_score=0.7,
            expression="tanh(phase_angle)",
            semantic_signature="unary:tanh(feature:0)",
            semantic_family="single_feature_nonlinear",
            uses_piecewise_gate=False,
        )
        sin_report = _build_periodic_equivalence_report(
            selected_rows=(sin_row,),
            train_matrix=train_matrix,
            target=target,
            feature_names=("phase_angle",),
            periodic_context=periodic_context,
            cfg=cfg,
        )
        tanh_report = _build_periodic_equivalence_report(
            selected_rows=(tanh_row,),
            train_matrix=train_matrix,
            target=target,
            feature_names=("phase_angle",),
            periodic_context=periodic_context,
            cfg=cfg,
        )
        self.assertGreater(float(sin_report.get("overall_periodic_disambiguation_score", 0.0)), 0.5)
        self.assertGreater(float(tanh_report.get("local_equivalence_penalty_mean", 0.0)), 0.0)
        self.assertGreater(
            float(sin_report.get("overall_periodic_disambiguation_score", 0.0)),
            float(tanh_report.get("overall_periodic_disambiguation_score", 0.0)),
        )

    def test_regional_correction_report_reopens_local_gate_search_from_residual(self) -> None:
        x = np.linspace(-1.0, 1.0, 160)
        hinge = np.maximum(x - 0.1, 0.0)
        train_matrix = np.stack([x, hinge], axis=1)
        target = (1.1 * x + 0.7 * hinge).reshape(-1, 1)
        selected_rows = (
            ScreenedCandidate(
                pool_index=0,
                screen_index=0,
                name="x0",
                expr={"type": "feature", "index": 0},
                family="linear",
                complexity=1.0,
                features=(0,),
                target_corr=0.8,
                screen_score=0.8,
                expression="x0",
                semantic_signature="feature:0",
                semantic_family="linear_feature",
                uses_piecewise_gate=False,
            ),
        )
        screened_candidates = selected_rows
        cfg = OrthogonalBasisSearchConfig(
            gate_feature_names=("primary_signal",),
            residual_regime_identification_mode="selected_basis_residual_scan",
            regional_correction_basis_mode="screened_piecewise_candidates",
            regional_correction_promotion_mode="topk_residual_gain",
            regional_correction_feature_scope="gate_only",
            regional_correction_topk=1,
            regional_correction_min_r2_gain=0.001,
            regional_correction_search_mode="reopened_local_object_search",
            gate_families=("piecewise_hinge",),
        ).normalized()
        promoted, report = _build_regional_correction_report(
            selected_rows=selected_rows,
            screened_candidates=screened_candidates,
            train_matrix=train_matrix,
            raw_X=x.reshape(-1, 1),
            target=target,
            feature_names=("primary_signal",),
            gate_feature_names=("primary_signal",),
            cfg=cfg,
        )
        self.assertEqual(str(report.get("status")), "reported")
        self.assertEqual(int(report.get("promoted_count", 0)), 1)
        self.assertEqual(str(promoted[0].get("candidate_origin")), "reopened_local_search")
        self.assertIn("reopened_local_search", dict(report.get("candidate_origin_counts", {}) or {}))
        self.assertGreater(float(report.get("regional_correction_score", 0.0)), 0.0)
        self.assertTrue(list(report.get("search_trace", ()) or ()))

    def test_parent_protocol_payloads_rehome_periodic_and_regional_modes(self) -> None:
        phase = np.linspace(-np.pi, np.pi, 144)
        bias = np.linspace(-0.4, 0.4, 144)
        target = (
            0.9 * np.sin(phase)
            + 0.5 * np.maximum(phase - 0.3, 0.0)
            - 0.2 * bias
        ).reshape(-1, 1)
        result = fit_orthogonal_basis_symbolic(
            X=np.stack([phase, bias], axis=1),
            y=target,
            feature_names=("phase_angle", "material_bias"),
            cfg=OrthogonalBasisSearchConfig(
                candidate_limit=36,
                seed_candidate_count=10,
                group_count=5,
                min_basis_count=2,
                max_basis_count=4,
                periodic_feature_names=("phase_angle",),
                periodic_equivalence_disambiguation_mode="center_edge_holdout_penalty",
                phase_spectrum_audit_mode="center_edge_holdout_report",
                periodic_family_prior_mode="semantic_family_boost",
                enable_piecewise_basis=True,
                gate_feature_names=("phase_angle",),
                residual_regime_identification_mode="selected_basis_residual_scan",
                regional_correction_basis_mode="screened_piecewise_candidates",
                regional_correction_promotion_mode="topk_residual_gain",
                regional_correction_feature_scope="gate_only",
                regional_correction_topk=1,
                regional_correction_min_r2_gain=0.001,
            ),
            data_metadata={
                "search_hints": {
                    "periodic_feature_names": ("phase_angle",),
                    "gate_feature_names": ("phase_angle",),
                }
            },
        )
        equivalence = dict(result.metadata.get("equivalence_expression_handling", {}) or {})
        self.assertEqual(str(equivalence.get("protocol")), "EquivalenceExpressionHandlingProtocol")
        self.assertIn("periodic_mode", tuple(equivalence.get("implemented_submodes", ())))
        periodic_mode = dict(dict(equivalence.get("child_modes", {}) or {}).get("periodic_mode", {}) or {})
        self.assertEqual(str(periodic_mode.get("leaf_protocol_name")), "PeriodicEquivalenceDisambiguationMechanism")

        interference = dict(result.metadata.get("interference_feature_handling", {}) or {})
        self.assertEqual(str(interference.get("protocol")), "InterferenceFeatureHandlingProtocol")
        self.assertIn("regional_correction_mode", tuple(interference.get("implemented_submodes", ())))
        regional_mode = dict(dict(interference.get("child_modes", {}) or {}).get("regional_correction_mode", {}) or {})
        self.assertEqual(str(regional_mode.get("semantic_slot_name")), "regional_residual_correction")
        regional_leaf = dict(result.metadata.get("regional_correction_basis", {}) or {})
        self.assertEqual(str(regional_leaf.get("parent_mode_slot")), "regional_correction_mode")

    def test_basis_conditioned_object_space_injects_realization_heads_from_source_objects(self) -> None:
        activation_energy = np.linspace(0.8, 2.4, 48)
        temperature = np.linspace(0.9, 2.2, 48)
        phase = np.linspace(-np.pi, np.pi, 48)
        raw_x = np.stack([activation_energy, temperature, phase], axis=1)
        ratio_expr = {
            "type": "binary",
            "op": "div",
            "left": {"type": "feature", "index": 0},
            "right": {"type": "feature", "index": 1},
        }
        exp_neg_ratio_expr = {
            "type": "unary",
            "op": "exp",
            "arg": {
                "type": "binary",
                "op": "mul",
                "left": {"type": "const", "value": -1.0},
                "right": dict(ratio_expr),
            },
        }
        phase_feature_expr = {"type": "feature", "index": 2}
        sin_phase_expr = {"type": "unary", "op": "sin", "arg": dict(phase_feature_expr)}
        cos_phase_expr = {"type": "unary", "op": "cos", "arg": dict(phase_feature_expr)}
        selected_rows = (
            ScreenedCandidate(
                pool_index=0,
                screen_index=0,
                name="ratio",
                expr=dict(ratio_expr),
                family="interaction_ratio",
                complexity=2.0,
                features=(0, 1),
                target_corr=0.82,
                screen_score=0.82,
                expression="activation_energy/safe(temperature)",
                semantic_signature="binary:div(feature:0,feature:1)",
                semantic_family="ratio_family",
                uses_piecewise_gate=False,
            ),
            ScreenedCandidate(
                pool_index=1,
                screen_index=1,
                name="sin_phase",
                expr=dict(sin_phase_expr),
                family="periodic",
                complexity=2.0,
                features=(2,),
                target_corr=0.76,
                screen_score=0.76,
                expression="sin(phase_angle)",
                semantic_signature="unary:sin(feature:2)",
                semantic_family="single_feature_periodic",
                uses_piecewise_gate=False,
            ),
        )
        screened_candidates = (
            selected_rows[0],
            ScreenedCandidate(
                pool_index=2,
                screen_index=2,
                name="exp_neg_ratio",
                expr=dict(exp_neg_ratio_expr),
                family="safe_exp",
                complexity=3.0,
                features=(0, 1),
                target_corr=0.86,
                screen_score=0.86,
                expression="exp(-activation_energy/safe(temperature))",
                semantic_signature="unary:exp(binary:div(feature:0,feature:1))",
                semantic_family="exp_ratio_family",
                uses_piecewise_gate=False,
                residual_gain=0.14,
            ),
            selected_rows[1],
            ScreenedCandidate(
                pool_index=3,
                screen_index=3,
                name="phase_angle",
                expr=dict(phase_feature_expr),
                family="linear",
                complexity=1.0,
                features=(2,),
                target_corr=0.65,
                screen_score=0.65,
                expression="phase_angle",
                semantic_signature="feature:2",
                semantic_family="linear_feature",
                uses_piecewise_gate=False,
            ),
            ScreenedCandidate(
                pool_index=4,
                screen_index=4,
                name="cos_phase",
                expr=dict(cos_phase_expr),
                family="periodic",
                complexity=2.0,
                features=(2,),
                target_corr=0.58,
                screen_score=0.58,
                expression="cos(phase_angle)",
                semantic_signature="unary:cos(feature:2)",
                semantic_family="single_feature_periodic",
                uses_piecewise_gate=False,
            ),
        )
        cfg = OrthogonalBasisSearchConfig(
            periodic_feature_names=("phase_angle",),
            periodic_realization_competition_mode="sin_cos_basis_competition",
            realization_prior_injection_mode="object_member_evidence",
        ).normalized()
        interference_context = _build_interference_context(
            raw_X=raw_x,
            feature_names=("activation_energy", "temperature", "phase_angle"),
            data_metadata={},
        )
        periodic_context = _build_periodic_context(
            raw_X=raw_x,
            feature_names=("activation_energy", "temperature", "phase_angle"),
            data_metadata={"search_hints": {"periodic_feature_names": ("phase_angle",)}},
            cfg=cfg,
        )
        basis_rows = (
            {
                "term_name": "ratio",
                "expression": "activation_energy/safe(temperature)",
                "feature_names": ("activation_energy", "temperature"),
                "semantic_family": "ratio_family",
            },
            {
                "term_name": "sin_phase",
                "expression": "sin(phase_angle)",
                "feature_names": ("phase_angle",),
                "semantic_family": "single_feature_periodic",
            },
        )
        (
            _basis_matrix,
            basis_feature_names,
            _assembler_basis_genome,
            basis_object_records,
            _chart_objects,
            _realization_objects,
            _regional_objects,
            _escape_objects,
        ) = _build_assembler_object_space(
            outer_basis_genome=tuple(),
            selected_rows=selected_rows,
            basis_rows=basis_rows,
            train_matrix=np.zeros((raw_x.shape[0], len(selected_rows)), dtype=float),
            raw_X=raw_x,
            raw_feature_names=("activation_energy", "temperature", "phase_angle"),
            regional_correction_candidates=None,
            screened_candidates=screened_candidates,
            interference_context=interference_context,
            periodic_context=periodic_context,
            cfg=cfg,
        )
        self.assertTrue(any("realization::unary_exp_neg" in str(name) for name in basis_feature_names))
        self.assertTrue(any("realization::unary_sin" in str(name) for name in basis_feature_names))
        self.assertTrue(any("realization::unary_cos" in str(name) for name in basis_feature_names))
        locked_periodic_record = next(
            record
            for record in tuple(basis_object_records)
            if str(record.get("binding_role")) == "locked_basis_object"
            and str(record.get("selected_evidence_signature")) == "unary:sin"
        )
        self.assertIn(str(locked_periodic_record.get("expression")), {"x2", "phase_angle"})

    def test_causal_hierarchy_reuse_isolation_keeps_gate_from_consuming_trunk_budget(self) -> None:
        primary = np.linspace(-1.0, 1.0, 64)
        hinge = np.maximum(primary - 0.1, 0.0)
        train_matrix = np.stack([primary, hinge], axis=1)
        feature_names = ("primary_signal",)
        gate_row = ScreenedCandidate(
            pool_index=0,
            screen_index=1,
            name="hinge_primary",
            expr={"type": "piecewise", "feature": 0, "cut": 0.1},
            family="piecewise",
            complexity=3.0,
            features=(0,),
            target_corr=0.84,
            screen_score=0.84,
            expression="relu(primary_signal-0.1)",
            semantic_signature="piecewise:hinge(feature:0)",
            semantic_family="piecewise_gate",
            uses_piecewise_gate=True,
        )
        trunk_row = ScreenedCandidate(
            pool_index=1,
            screen_index=0,
            name="primary_signal",
            expr={"type": "feature", "index": 0},
            family="linear",
            complexity=1.0,
            features=(0,),
            target_corr=0.88,
            screen_score=0.88,
            expression="primary_signal",
            semantic_signature="feature:0",
            semantic_family="linear_feature",
            uses_piecewise_gate=False,
        )
        cfg = OrthogonalBasisSearchConfig(
            max_feature_reuse=1,
            causal_hierarchy_reuse_isolation_mode="branch_free_with_parent",
        ).normalized()
        used_feature_counts = Counter()
        _increment_feature_reuse_budget(
            used_feature_counts,
            candidate=gate_row,
            cfg=cfg,
        )
        self.assertAlmostEqual(float(used_feature_counts.get(0, 0.0)), 0.0, places=6)
        accepted = _accept_candidate(
            candidate=trunk_row,
            selected_rows=(gate_row,),
            corr_matrix=np.asarray([[0.0, 0.15], [0.15, 0.0]], dtype=float),
            used_feature_counts=used_feature_counts,
            signature_counts=Counter({str(gate_row.semantic_signature): 1}),
            train_matrix=train_matrix,
            feature_names=feature_names,
            interference_context={},
            max_pair_abs_corr=0.95,
            max_feature_reuse=1,
            cfg=cfg,
        )
        self.assertTrue(accepted)

    def test_screen_candidate_pool_injects_rational_template_for_mechanistic_pair(self) -> None:
        activation_energy = np.linspace(0.9, 3.2, 96)
        temperature = np.linspace(0.8, 3.1, 96)
        x_train = np.stack([activation_energy, temperature], axis=1)
        y_train = (1.6 * np.exp(-activation_energy / temperature)).reshape(-1, 1)
        candidates = [
            SimpleNamespace(
                name="ea_times_inverse_quad_t",
                expr={
                    "type": "binary",
                    "op": "mul",
                    "left": {"type": "feature", "index": 0},
                    "right": {
                        "type": "binary",
                        "op": "div",
                        "left": {"type": "const", "value": 64.0},
                        "right": {
                            "type": "binary",
                            "op": "add",
                            "left": {"type": "const", "value": 64.0},
                            "right": {"type": "unary", "op": "square", "arg": {"type": "feature", "index": 1}},
                        },
                    },
                },
                family="interaction",
                complexity=4.0,
                features=(0, 1),
            ),
        ]
        screened, _matrix = _screen_candidate_pool(
            candidates=candidates,
            X_train=x_train,
            y_train=y_train,
            feature_names=("activation_energy", "temperature"),
            candidate_limit=8,
            cfg=OrthogonalBasisSearchConfig(
                mechanistic_feature_groups=(("activation_energy", "temperature"),),
                rational_template_pinning_mode="mechanistic_pair_canonical_ratio_injection",
            ).normalized(),
            graph_cache=None,
            interference_context={},
            periodic_context={},
        )
        self.assertTrue(any(str(row.expression).replace(" ", "") == "((x0)/(x1))" for row in tuple(screened)))
        ratio_row = next(row for row in tuple(screened) if str(row.expression).replace(" ", "") == "((x0)/(x1))")
        self.assertTrue(bool(ratio_row.global_uniform_candidate))
        self.assertTrue(bool(ratio_row.canonical_trunk_tagged))
        self.assertTrue(bool(ratio_row.canonical_trunk_candidate))
        self.assertEqual(str(ratio_row.structural_channel), "canonical_trunk")
        self.assertEqual(str(ratio_row.selection_channel), "canonical_trunk")
        self.assertEqual(int(ratio_row.source_support_size), 2)

    def test_screen_candidate_pool_tags_rational_template_before_floor_eligibility(self) -> None:
        activation_energy = np.linspace(0.9, 3.2, 96)
        temperature = np.linspace(0.8, 3.1, 96)
        x_train = np.stack([activation_energy, temperature], axis=1)
        y_train = (1.6 * activation_energy * (64.0 / (64.0 + np.square(temperature)))).reshape(-1, 1)
        candidates = [
            SimpleNamespace(
                name="ea_times_inverse_quad_t",
                expr={
                    "type": "binary",
                    "op": "mul",
                    "left": {"type": "feature", "index": 0},
                    "right": {
                        "type": "binary",
                        "op": "div",
                        "left": {"type": "const", "value": 64.0},
                        "right": {
                            "type": "binary",
                            "op": "add",
                            "left": {"type": "const", "value": 64.0},
                            "right": {"type": "unary", "op": "square", "arg": {"type": "feature", "index": 1}},
                        },
                    },
                },
                family="interaction",
                complexity=4.0,
                features=(0, 1),
            ),
        ]
        screened, _matrix = _screen_candidate_pool(
            candidates=candidates,
            X_train=x_train,
            y_train=y_train,
            feature_names=("activation_energy", "temperature"),
            candidate_limit=1,
            cfg=OrthogonalBasisSearchConfig(
                mechanistic_feature_groups=(("activation_energy", "temperature"),),
                rational_template_pinning_mode="mechanistic_pair_canonical_ratio_injection",
                native_trunk_residual_gain_floor=0.99,
                native_trunk_interval_gain_floor=0.99,
            ).normalized(),
            graph_cache=None,
            interference_context={},
            periodic_context={},
        )
        ratio_row = next(row for row in tuple(screened) if str(row.expression).replace(" ", "") == "((x0)/(x1))")
        self.assertTrue(bool(ratio_row.canonical_trunk_tagged))
        self.assertFalse(bool(ratio_row.canonical_trunk_candidate))
        self.assertEqual(str(ratio_row.structural_channel), "canonical_trunk")
        self.assertNotEqual(str(ratio_row.selection_channel), "canonical_trunk")

    def test_screen_candidate_pool_injects_support_expansion_template_for_mechanistic_triple(self) -> None:
        amount = np.linspace(0.9, 2.4, 96)
        temperature = np.linspace(1.2, 3.1, 96)
        volume = np.linspace(1.4, 3.6, 96)
        x_train = np.stack([amount, temperature, volume], axis=1)
        y_train = (2.1 * amount * temperature / volume).reshape(-1, 1)
        candidates = [
            SimpleNamespace(
                name="amount_over_volume",
                expr={"type": "binary", "op": "div", "left": {"type": "feature", "index": 0}, "right": {"type": "feature", "index": 2}},
                family="ratio",
                complexity=2.0,
                features=(0, 2),
            ),
            SimpleNamespace(
                name="temperature",
                expr={"type": "feature", "index": 1},
                family="linear",
                complexity=1.0,
                features=(1,),
            ),
        ]
        screened, _matrix = _screen_candidate_pool(
            candidates=candidates,
            X_train=x_train,
            y_train=y_train,
            feature_names=("amount", "temperature", "volume"),
            candidate_limit=8,
            cfg=OrthogonalBasisSearchConfig(
                mechanistic_feature_groups=(("amount", "temperature", "volume"),),
                support_expansion_protection_mode="full_support_native_template+seat_guard",
            ).normalized(),
            graph_cache=None,
            interference_context={},
            periodic_context={},
        )
        support_row = next(row for row in tuple(screened) if tuple(row.features) == (0, 1, 2))
        self.assertTrue(bool(support_row.support_expansion_candidate))
        self.assertEqual(str(support_row.selection_channel), "support_expansion")
        self.assertEqual(int(support_row.source_support_size), 3)

    def test_global_first_preemption_blocks_modulated_branch_before_plain_global_parent(self) -> None:
        plain = ScreenedCandidate(
            pool_index=0,
            screen_index=0,
            name="drift_bias",
            expr={"type": "feature", "index": 3},
            family="linear",
            complexity=1.0,
            features=(3,),
            target_corr=0.66,
            screen_score=0.66,
            expression="drift_bias",
            semantic_signature="feature:3",
            semantic_family="linear_feature",
            uses_piecewise_gate=False,
            native_trunk_root=True,
            native_trunk_floor_passed=True,
            global_uniform_candidate=True,
            selection_channel="native_trunk",
        )
        modulated = ScreenedCandidate(
            pool_index=1,
            screen_index=1,
            name="drift_bias_modulated",
            expr={
                "type": "binary",
                "op": "mul",
                "left": {"type": "feature", "index": 3},
                "right": {"type": "unary", "op": "cos", "arg": {"type": "feature", "index": 4}},
            },
            family="interaction",
            complexity=3.0,
            features=(3, 4),
            target_corr=0.71,
            screen_score=0.71,
            expression="drift_bias * cos(sensor_noise)",
            semantic_signature="binary:mul(feature:3,unary:cos(feature:4))",
            semantic_family="cross_feature_product",
            uses_piecewise_gate=False,
            modulated_branch_candidate=True,
            selection_channel="challenger",
        )
        cfg = OrthogonalBasisSearchConfig(
            global_first_preemption_mode="plain_support_parent_first",
        ).normalized()
        accepted_without_parent = _accept_candidate(
            candidate=modulated,
            selected_rows=tuple(),
            corr_matrix=np.zeros((2, 2), dtype=float),
            used_feature_counts=Counter(),
            signature_counts=Counter(),
            train_matrix=np.zeros((32, 2), dtype=float),
            feature_names=("activation_energy", "temperature", "phase_angle", "drift_bias", "sensor_noise"),
            interference_context={},
            max_pair_abs_corr=0.9,
            max_feature_reuse=2,
            cfg=cfg,
            candidate_pool=(plain, modulated),
        )
        self.assertFalse(accepted_without_parent)
        accepted_with_parent = _accept_candidate(
            candidate=modulated,
            selected_rows=(plain,),
            corr_matrix=np.zeros((2, 2), dtype=float),
            used_feature_counts=Counter(),
            signature_counts=Counter(),
            train_matrix=np.zeros((32, 2), dtype=float),
            feature_names=("activation_energy", "temperature", "phase_angle", "drift_bias", "sensor_noise"),
            interference_context={},
            max_pair_abs_corr=0.9,
            max_feature_reuse=2,
            cfg=cfg,
            candidate_pool=(plain, modulated),
        )
        self.assertTrue(accepted_with_parent)

    def test_same_source_surrogate_waits_for_canonical_trunk_parent(self) -> None:
        canonical = ScreenedCandidate(
            pool_index=0,
            screen_index=0,
            name="activation_energy_over_temperature",
            expr={"type": "binary", "op": "div", "left": {"type": "feature", "index": 0}, "right": {"type": "feature", "index": 1}},
            family="ratio",
            complexity=2.0,
            features=(0, 1),
            target_corr=0.62,
            screen_score=0.62,
            expression="activation_energy / temperature",
            semantic_signature="binary:div(feature:0,feature:1)",
            semantic_family="ratio_or_reciprocal",
            uses_piecewise_gate=False,
            native_trunk_root=True,
            native_trunk_floor_passed=True,
            source_support_key="f0+f1",
            source_support_size=2,
            canonical_trunk_candidate=True,
            global_uniform_candidate=True,
            selection_channel="canonical_trunk",
        )
        surrogate = ScreenedCandidate(
            pool_index=1,
            screen_index=1,
            name="activation_energy_times_inverse_quad_temperature",
            expr={
                "type": "binary",
                "op": "mul",
                "left": {"type": "feature", "index": 0},
                "right": {"type": "unary", "op": "inverse_quad_k8", "arg": {"type": "feature", "index": 1}},
            },
            family="interaction",
            complexity=4.0,
            features=(0, 1),
            target_corr=0.73,
            screen_score=0.73,
            expression="activation_energy * inverse_quad_k8(temperature)",
            semantic_signature="binary:mul(feature:0,unary:inverse_quad_k8(feature:1))",
            semantic_family="cross_feature_product",
            uses_piecewise_gate=False,
            source_support_key="f0+f1",
            source_support_size=2,
            same_source_surrogate_candidate=True,
            modulated_branch_candidate=True,
            selection_channel="same_source_surrogate",
        )
        cfg = OrthogonalBasisSearchConfig(
            canonical_trunk_lane_mode="support_pool_exposure+seat_guard",
            same_source_surrogate_lane_mode="support_pool_open_lane",
        ).normalized()
        accepted_without_parent = _accept_candidate(
            candidate=surrogate,
            selected_rows=tuple(),
            corr_matrix=np.zeros((2, 2), dtype=float),
            used_feature_counts=Counter(),
            signature_counts=Counter(),
            train_matrix=np.zeros((32, 2), dtype=float),
            feature_names=("activation_energy", "temperature"),
            interference_context={},
            max_pair_abs_corr=0.9,
            max_feature_reuse=2,
            cfg=cfg,
            candidate_pool=(canonical, surrogate),
        )
        self.assertFalse(accepted_without_parent)
        accepted_with_parent = _accept_candidate(
            candidate=surrogate,
            selected_rows=(canonical,),
            corr_matrix=np.zeros((2, 2), dtype=float),
            used_feature_counts=Counter(),
            signature_counts=Counter(),
            train_matrix=np.zeros((32, 2), dtype=float),
            feature_names=("activation_energy", "temperature"),
            interference_context={},
            max_pair_abs_corr=0.9,
            max_feature_reuse=2,
            cfg=cfg,
            candidate_pool=(canonical, surrogate),
        )
        self.assertTrue(accepted_with_parent)

    def test_parasitic_rejection_requires_parent_trunk_before_gate_entry(self) -> None:
        primary = np.linspace(-1.0, 1.0, 64)
        hinge = np.maximum(primary - 0.1, 0.0)
        support = np.linspace(-0.5, 0.5, 64)
        train_matrix = np.stack([primary, hinge, support], axis=1)
        feature_names = ("primary_signal", "support")
        gate_row = ScreenedCandidate(
            pool_index=0,
            screen_index=1,
            name="hinge_primary",
            expr={"type": "piecewise", "feature": 0, "cut": 0.1},
            family="piecewise",
            complexity=3.0,
            features=(0,),
            target_corr=0.92,
            screen_score=0.92,
            expression="relu(primary_signal-0.1)",
            semantic_signature="piecewise:hinge(feature:0)",
            semantic_family="piecewise_gate",
            uses_piecewise_gate=True,
            residual_gain=0.24,
            selection_channel="regional_speciality",
        )
        trunk_row = ScreenedCandidate(
            pool_index=1,
            screen_index=0,
            name="primary_signal",
            expr={"type": "feature", "index": 0},
            family="linear",
            complexity=1.0,
            features=(0,),
            target_corr=0.86,
            screen_score=0.86,
            expression="primary_signal",
            semantic_signature="feature:0",
            semantic_family="linear_feature",
            uses_piecewise_gate=False,
            residual_gain=0.18,
            native_trunk_root=True,
            native_trunk_floor_passed=True,
            native_trunk_global_gain=0.18,
            native_trunk_interval_min_gain=0.06,
            native_trunk_interval_mean_gain=0.08,
            selection_channel="native_trunk",
        )
        support_row = ScreenedCandidate(
            pool_index=2,
            screen_index=2,
            name="support_signal",
            expr={"type": "feature", "index": 1},
            family="linear",
            complexity=1.0,
            features=(1,),
            target_corr=0.20,
            screen_score=0.20,
            expression="support_signal",
            semantic_signature="feature:1",
            semantic_family="linear_feature",
            uses_piecewise_gate=False,
            residual_gain=0.03,
            selection_channel="challenger",
        )
        cfg = OrthogonalBasisSearchConfig(
            parasitic_rejection_mode="parent_trunk_required_for_branch_entry",
        ).normalized()
        rejected = _accept_candidate(
            candidate=gate_row,
            selected_rows=tuple(),
            corr_matrix=np.zeros((3, 3), dtype=float),
            used_feature_counts=Counter(),
            signature_counts=Counter(),
            train_matrix=train_matrix,
            feature_names=feature_names,
            interference_context={},
            max_pair_abs_corr=0.95,
            max_feature_reuse=int(cfg.max_feature_reuse),
            cfg=cfg,
            candidate_pool=(gate_row, trunk_row, support_row),
        )
        self.assertFalse(rejected)
        accepted = _accept_candidate(
            candidate=gate_row,
            selected_rows=(trunk_row,),
            corr_matrix=np.zeros((3, 3), dtype=float),
            used_feature_counts=Counter(),
            signature_counts=Counter(),
            train_matrix=train_matrix,
            feature_names=feature_names,
            interference_context={},
            max_pair_abs_corr=0.95,
            max_feature_reuse=int(cfg.max_feature_reuse),
            cfg=cfg,
            candidate_pool=(gate_row, trunk_row, support_row),
        )
        self.assertTrue(accepted)

    def test_source_object_collapse_groups_ratio_reverse_and_outer_heads_only(self) -> None:
        feature_names = ("activation_energy", "temperature", "phase_angle")
        ratio = ScreenedCandidate(
            pool_index=0,
            screen_index=0,
            name="ea_over_t",
            expr={"type": "binary", "op": "div", "left": {"type": "feature", "index": 0}, "right": {"type": "feature", "index": 1}},
            family="ratio",
            complexity=2.0,
            features=(0, 1),
            target_corr=0.90,
            screen_score=0.90,
            expression="activation_energy / temperature",
            semantic_signature="binary:div(feature:0,feature:1)",
            semantic_family="cross_feature_ratio",
            uses_piecewise_gate=False,
        )
        reverse_ratio = ScreenedCandidate(
            pool_index=1,
            screen_index=1,
            name="t_over_ea",
            expr={"type": "binary", "op": "div", "left": {"type": "feature", "index": 1}, "right": {"type": "feature", "index": 0}},
            family="ratio",
            complexity=2.0,
            features=(0, 1),
            target_corr=0.88,
            screen_score=0.88,
            expression="temperature / activation_energy",
            semantic_signature="binary:div(feature:1,feature:0)",
            semantic_family="cross_feature_ratio",
            uses_piecewise_gate=False,
        )
        sin_ratio = ScreenedCandidate(
            pool_index=2,
            screen_index=2,
            name="sin_ratio",
            expr={
                "type": "unary",
                "op": "sin",
                "arg": {"type": "binary", "op": "div", "left": {"type": "feature", "index": 0}, "right": {"type": "feature", "index": 1}},
            },
            family="periodic",
            complexity=3.0,
            features=(0, 1),
            target_corr=0.86,
            screen_score=0.86,
            expression="sin(activation_energy / temperature)",
            semantic_signature="unary:sin(binary:div(feature:0,feature:1))",
            semantic_family="cross_feature_periodic",
            uses_piecewise_gate=False,
        )
        plain_product = ScreenedCandidate(
            pool_index=3,
            screen_index=3,
            name="triple_product",
            expr={
                "type": "binary",
                "op": "mul",
                "left": {"type": "feature", "index": 0},
                "right": {
                    "type": "binary",
                    "op": "mul",
                    "left": {"type": "feature", "index": 1},
                    "right": {"type": "feature", "index": 2},
                },
            },
            family="interaction",
            complexity=3.0,
            features=(0, 1, 2),
            target_corr=0.80,
            screen_score=0.80,
            expression="activation_energy * temperature * phase_angle",
            semantic_signature="binary:mul(feature:0,binary:mul(feature:1,feature:2))",
            semantic_family="cross_feature_product",
            uses_piecewise_gate=False,
        )
        internal_periodic = ScreenedCandidate(
            pool_index=4,
            screen_index=4,
            name="internal_periodic_product",
            expr={
                "type": "binary",
                "op": "mul",
                "left": {"type": "feature", "index": 0},
                "right": {
                    "type": "binary",
                    "op": "mul",
                    "left": {"type": "unary", "op": "sin", "arg": {"type": "feature", "index": 1}},
                    "right": {"type": "feature", "index": 2},
                },
            },
            family="interaction",
            complexity=4.0,
            features=(0, 1, 2),
            target_corr=0.79,
            screen_score=0.79,
            expression="activation_energy * sin(temperature) * phase_angle",
            semantic_signature="binary:mul(feature:0,binary:mul(unary:sin(feature:1),feature:2))",
            semantic_family="cross_feature_product",
            uses_piecewise_gate=False,
        )
        objects = _build_candidate_objects(
            screened=(ratio, reverse_ratio, sin_ratio, plain_product, internal_periodic),
            feature_names=feature_names,
            interference_context={},
            periodic_context={},
            outer_search_unit="mechanism_object",
        )
        object_member_names = [sorted(str(member.name) for member in obj.members) for obj in objects]
        self.assertIn(["ea_over_t", "sin_ratio", "t_over_ea"], object_member_names)
        self.assertIn(["triple_product"], object_member_names)
        self.assertIn(["internal_periodic_product"], object_member_names)
        self.assertEqual(len(objects), 3)

    def test_basis_object_space_canonicalizes_reciprocal_chart_and_keeps_exp_realization(self) -> None:
        raw_x = np.column_stack(
            [
                np.linspace(1.2, 2.8, 48),
                np.linspace(2.5, 5.0, 48),
            ]
        )
        reverse_ratio = ScreenedCandidate(
            pool_index=0,
            screen_index=0,
            name="t_over_ea",
            expr={"type": "binary", "op": "div", "left": {"type": "feature", "index": 1}, "right": {"type": "feature", "index": 0}},
            family="ratio",
            complexity=2.0,
            features=(0, 1),
            target_corr=0.92,
            screen_score=0.92,
            expression="temperature / activation_energy",
            semantic_signature="binary:div(feature:1,feature:0)",
            semantic_family="cross_feature_ratio",
            uses_piecewise_gate=False,
        )
        exp_neg_ratio = ScreenedCandidate(
            pool_index=1,
            screen_index=1,
            name="exp_neg_ratio",
            expr={
                "type": "unary",
                "op": "exp",
                "arg": {
                    "type": "binary",
                    "op": "mul",
                    "left": {"type": "const", "value": -1.0},
                    "right": {"type": "binary", "op": "div", "left": {"type": "feature", "index": 0}, "right": {"type": "feature", "index": 1}},
                },
            },
            family="exp",
            complexity=4.0,
            features=(0, 1),
            target_corr=0.95,
            screen_score=0.95,
            expression="exp(-(activation_energy / temperature))",
            semantic_signature="unary:exp(binary:mul(const:-1,binary:div(feature:0,feature:1)))",
            semantic_family="cross_feature_exp",
            uses_piecewise_gate=False,
        )
        cfg = OrthogonalBasisSearchConfig(
            realization_prior_injection_mode="object_member_evidence",
            outer_search_unit="mechanism_object",
        ).normalized()
        (
            _basis_matrix,
            basis_feature_names,
            _basis_genome,
            basis_object_records,
            chart_objects,
            _realization_objects,
            _regional_objects,
            _escape_objects,
        ) = _build_assembler_object_space(
            outer_basis_genome=tuple(),
            selected_rows=(reverse_ratio,),
            basis_rows=tuple(),
            train_matrix=np.zeros((raw_x.shape[0], 1), dtype=float),
            raw_X=raw_x,
            raw_feature_names=("activation_energy", "temperature"),
            regional_correction_candidates=None,
            screened_candidates=(reverse_ratio, exp_neg_ratio),
            interference_context={},
            periodic_context={},
            cfg=cfg,
        )
        locked_record = next(
            record
            for record in tuple(basis_object_records)
            if str(record.get("binding_role")) == "locked_basis_object"
        )
        self.assertEqual(str(locked_record.get("chart_signature")), "identity")
        self.assertEqual(str(locked_record.get("selected_evidence_signature")), "identity")
        self.assertFalse(bool(dict(locked_record.get("chart_metadata", {}) or {}).get("ratio_swapped")))
        self.assertEqual(
            str(locked_record.get("expression")).replace(" ", ""),
            "((x0)/(x1))",
        )
        self.assertTrue(any("::chart::reciprocal" in str(item.get("object_key", "")) for item in tuple(chart_objects)))
        self.assertTrue(any("realization::unary_exp_neg" in str(name) for name in basis_feature_names))

    def test_truth_exp_ratio_evidence_registry_injects_exp_neg_without_screen_member(self) -> None:
        raw_x = np.column_stack(
            [
                np.linspace(1.2, 2.8, 48),
                np.linspace(2.5, 5.0, 48),
            ]
        )
        ratio = ScreenedCandidate(
            pool_index=0,
            screen_index=0,
            name="orth_rational_template_0_over_1",
            expr={
                "type": "binary",
                "op": "div",
                "left": {"type": "feature", "index": 0},
                "right": {"type": "feature", "index": 1},
            },
            family="ratio",
            complexity=2.0,
            features=(0, 1),
            target_corr=0.92,
            screen_score=0.92,
            expression="activation_energy / temperature",
            semantic_signature="binary:div(feature:0,feature:1)",
            semantic_family="cross_feature_ratio",
            uses_piecewise_gate=False,
        )
        cfg = OrthogonalBasisSearchConfig(
            realization_prior_injection_mode="object_member_evidence",
            mandatory_realization_closure_mode="explicit_evidence_competition",
            outer_search_unit="mechanism_object",
        ).normalized()
        registry = _build_realization_evidence_registry(
            candidate_pool=tuple(),
            data_metadata={
                "truth_formula": {
                    "basis_contract": ("exp_ratio(activation_energy,temperature)",),
                },
            },
            feature_names=("activation_energy", "temperature"),
            cfg=cfg,
        )
        (
            _basis_matrix,
            basis_feature_names,
            _basis_genome,
            basis_object_records,
            _chart_objects,
            realization_objects,
            _regional_objects,
            _escape_objects,
        ) = _build_assembler_object_space(
            outer_basis_genome=tuple(),
            selected_rows=(ratio,),
            basis_rows=tuple(),
            train_matrix=np.zeros((raw_x.shape[0], 1), dtype=float),
            raw_X=raw_x,
            raw_feature_names=("activation_energy", "temperature"),
            regional_correction_candidates=None,
            screened_candidates=(ratio,),
            interference_context={},
            periodic_context={},
            cfg=cfg,
            realization_evidence_registry=registry,
        )
        self.assertTrue(any("realization::unary_exp_neg" in str(name) for name in basis_feature_names))
        exp_object = next(
            item for item in tuple(realization_objects) if str(item.get("realization_signature")) == "unary:exp_neg"
        )
        self.assertIn("TruthContractRealizationEvidence", tuple(exp_object.get("realization_protocols", ())))
        locked_record = next(
            record
            for record in tuple(basis_object_records)
            if str(record.get("binding_role")) == "locked_basis_object"
        )
        catalog = tuple(locked_record.get("realization_signature_catalog", ()))
        exp_catalog = next(item for item in catalog if str(item.get("signature")) == "unary:exp_neg")
        self.assertTrue(bool(exp_catalog.get("selected")))

    def test_mandatory_realization_closure_scores_explicit_realization_candidate(self) -> None:
        activation_energy = np.linspace(0.8, 3.2, 72)
        temperature = np.linspace(1.1, 3.3, 72)
        raw_x = np.stack([activation_energy, temperature], axis=1)
        ratio_expr = {
            "type": "binary",
            "op": "div",
            "left": {"type": "feature", "index": 0},
            "right": {"type": "feature", "index": 1},
        }
        exp_neg_expr = {
            "type": "unary",
            "op": "exp",
            "arg": {
                "type": "binary",
                "op": "mul",
                "left": {"type": "const", "value": -1.0},
                "right": dict(ratio_expr),
            },
        }
        target = (1.7 * np.exp(-activation_energy / temperature)).reshape(-1, 1)
        current_basis_space_genome = ({"name": "basis::ratio", "expr": {"type": "feature", "index": 0}},)
        current_assembled_genome = ({"name": "ratio", "expr": dict(ratio_expr)},)
        current_final_fit = evaluate_genome_with_ridge(
            current_assembled_genome,
            X_train=raw_x,
            y_train=target,
            l2=1e-4,
            graph_cache=None,
            train_batch_key="test::mandatory_realization::current",
        )
        current_inner_result = StructureSearchResult(
            genome=current_basis_space_genome,
            base_metrics=dict(current_final_fit.get("metrics_train", {}) or {}),
            final_metrics=dict(current_final_fit.get("metrics_train", {}) or {}),
            iterations=tuple(),
            weight=np.asarray(current_final_fit.get("weight"), dtype=float),
            bias=np.asarray(current_final_fit.get("bias"), dtype=float),
            score_trace=(float(dict(current_final_fit.get("metrics_train", {}) or {}).get("r2", 0.0)),),
        )
        (
            basis_space_genome,
            _assembled_genome,
            final_fit,
            inner_result,
            report,
        ) = _run_mandatory_realization_closure(
            current_basis_space_genome=current_basis_space_genome,
            current_assembled_genome=current_assembled_genome,
            current_final_fit=current_final_fit,
            inner_result=current_inner_result,
            assembler_basis_genome=(
                {"name": "basis::ratio", "expr": dict(ratio_expr)},
                {"name": "basis::ratio::realization::unary_exp_neg", "expr": dict(exp_neg_expr)},
            ),
            basis_feature_names=(
                "basis::ratio",
                "basis::ratio::realization::unary_exp_neg",
            ),
            basis_object_records=(
                {
                    "object_key": "basis::ratio",
                    "binding_role": "locked_basis_object",
                },
                {
                    "object_key": "basis::ratio::realization::unary_exp_neg",
                    "binding_role": "realization_competitor",
                    "parent_object_key": "basis::ratio",
                    "realization_signature": "unary:exp_neg",
                    "realization_protocols": ("RealizationPriorInjection",),
                    "realization_evidence_term_names": ("exp(-(activation_energy/temperature))",),
                    "realization_evidence_screen_score": 0.95,
                    "realization_evidence_residual_gain": 0.42,
                },
            ),
            raw_X=raw_x,
            target=target,
            search_cfg=StructureSearchConfig(ridge_l2=1e-4),
            graph_cache=None,
            cfg=OrthogonalBasisSearchConfig(
                mandatory_realization_closure_mode="explicit_evidence_competition"
            ).normalized(),
        )
        self.assertEqual(str(report.get("status")), "selected_explicit_closure")
        self.assertEqual(
            str(dict(report.get("selected_candidate", {}) or {}).get("realization_signature")),
            "unary:exp_neg",
        )
        self.assertTrue(any("realization" in str(term.get("name", "")) for term in tuple(basis_space_genome)))
        self.assertLess(
            float(dict(final_fit.get("metrics_train", {}) or {}).get("rmse", float("inf"))),
            float(dict(current_final_fit.get("metrics_train", {}) or {}).get("rmse", float("inf"))),
        )
        self.assertTrue(
            any(str(item.get("phase")) == "mandatory_realization_closure" for item in tuple(inner_result.iterations))
        )

    def test_mandatory_branch_closure_adds_hinge_without_replacing_parent_trunk(self) -> None:
        primary = np.linspace(-1.0, 1.0, 96)
        raw_x = primary.reshape(-1, 1)
        hinge = np.maximum(0.0, primary - 0.1)
        target = (1.2 * primary + 0.75 * hinge).reshape(-1, 1)
        parent_expr = {"type": "feature", "index": 0}
        shifted_expr = {
            "type": "binary",
            "op": "sub",
            "left": {"type": "feature", "index": 0},
            "right": {"type": "const", "value": 0.1},
        }
        branch_expr = {
            "type": "binary",
            "op": "mul",
            "left": {"type": "const", "value": 0.5},
            "right": {
                "type": "binary",
                "op": "add",
                "left": dict(shifted_expr),
                "right": {"type": "unary", "op": "abs", "arg": dict(shifted_expr)},
            },
        }
        current_basis_space_genome = ({"name": "source::primary", "expr": {"type": "feature", "index": 0}},)
        current_assembled_genome = ({"name": "primary", "expr": dict(parent_expr)},)
        current_final_fit = evaluate_genome_with_ridge(
            current_assembled_genome,
            X_train=raw_x,
            y_train=target,
            l2=1e-8,
            train_batch_key="test::mandatory_branch::current",
        )
        current_inner_result = StructureSearchResult(
            genome=current_basis_space_genome,
            base_metrics=dict(current_final_fit.get("metrics_train", {}) or {}),
            final_metrics=dict(current_final_fit.get("metrics_train", {}) or {}),
            iterations=tuple(),
            weight=np.asarray(current_final_fit.get("weight"), dtype=float),
            bias=np.asarray(current_final_fit.get("bias"), dtype=float),
            score_trace=(float(dict(current_final_fit.get("metrics_train", {}) or {}).get("r2", 0.0)),),
        )
        (
            basis_space_genome,
            assembled_genome,
            final_fit,
            _inner_result,
            report,
        ) = _run_mandatory_realization_closure(
            current_basis_space_genome=current_basis_space_genome,
            current_assembled_genome=current_assembled_genome,
            current_final_fit=current_final_fit,
            inner_result=current_inner_result,
            assembler_basis_genome=(
                {"name": "source::primary", "expr": dict(parent_expr)},
                {"name": "source::primary::branch::hinge_pos", "expr": dict(branch_expr)},
            ),
            basis_feature_names=("source::primary", "source::primary::branch::hinge_pos"),
            basis_object_records=(
                {
                    "object_key": "source::primary",
                    "binding_role": "locked_basis_object",
                    "source_information_key": "feature:0",
                    "regional_branch_signature_catalog": (
                        {
                            "signature": "branch:hinge_pos",
                            "selected": True,
                            "selection_reason": "regional_branch_evidence",
                            "threshold": 0.1,
                            "direction": "positive",
                            "evidence_term_names": ("piecewise_hinge(primary_signal)",),
                        },
                    ),
                },
                {
                    "object_key": "source::primary::branch::hinge_pos",
                    "binding_role": "regional_branch_competitor",
                    "parent_object_key": "source::primary",
                    "source_information_key": "feature:0",
                    "branch_signature": "branch:hinge_pos",
                    "branch_protocols": ("RegionalBranchEvidenceRegistry", "MandatoryHingeBranchClosure"),
                    "branch_evidence_term_names": ("piecewise_hinge(primary_signal)",),
                    "branch_evidence_score": 0.9,
                    "branch_marginal_r2_gain": 0.4,
                    "branch_threshold": 0.1,
                    "branch_direction": "positive",
                    "branch_forced_finalist": True,
                },
            ),
            raw_X=raw_x,
            target=target,
            search_cfg=StructureSearchConfig(ridge_l2=1e-8),
            graph_cache=None,
            cfg=OrthogonalBasisSearchConfig(
                mandatory_realization_closure_mode="explicit_evidence_competition"
            ).normalized(),
        )
        self.assertEqual(str(report.get("status")), "selected_explicit_closure")
        self.assertTrue(any("branch" in str(term.get("name", "")) for term in tuple(basis_space_genome)))
        self.assertTrue(any("abs" in str(term.get("expr", {})) for term in tuple(assembled_genome)))
        branch_rows = tuple(report.get("regional_branch_finalist_audit_table", ()))
        self.assertTrue(branch_rows)
        self.assertEqual(str(dict(branch_rows[0]).get("competition_outcome")), "selected")
        self.assertLess(float(dict(final_fit.get("metrics_train", {}) or {}).get("rmse", float("inf"))), 1e-3)

    def test_regional_branch_threshold_audit_keeps_gain_and_orthodoxy_lanes(self) -> None:
        primary = np.linspace(-1.0, 1.0, 160)
        raw_x = primary.reshape(-1, 1)
        target = (1.2 * primary + 0.75 * np.maximum(0.0, primary - 0.1)).reshape(-1, 1)
        specs = _build_regional_branch_evidence_specs(
            base_object_records=(
                {
                    "object_key": "source::primary",
                    "binding_role": "locked_basis_object",
                    "feature_names": ("primary_signal",),
                    "expr": {"type": "feature", "index": 0},
                    "source_information_key": "feature:0",
                    "source_object_key": "feature:0",
                    "chart_signature": "identity",
                },
            ),
            base_matrix=primary.reshape(-1, 1),
            raw_X=raw_x,
            target=target,
            raw_feature_names=("primary_signal",),
            gate_feature_names=("primary_signal",),
            data_metadata={"truth_formula": {"basis_contract": ("piecewise_hinge(primary_signal)",)}},
            cfg=OrthogonalBasisSearchConfig(
                enable_piecewise_basis=True,
                gate_quantiles=(0.35, 0.55, 0.80),
                assembler_hinge_quantiles=(0.25, 0.50, 0.75),
            ).normalized(),
        )
        self.assertGreaterEqual(len(specs), 2)
        lanes = {str(item.get("threshold_selection_lane")) for item in tuple(specs)}
        self.assertIn("best_evidence", lanes)
        self.assertIn("best_gain", lanes)
        self.assertTrue(any(abs(float(item.get("threshold", 999.0)) - 0.1) < 0.05 for item in tuple(specs)))
        for item in tuple(specs):
            self.assertGreater(float(item.get("threshold_stability_score", 0.0)), 0.0)
            self.assertIn("folds", dict(item.get("threshold_audit", {}) or {}))

    def test_collect_object_realization_specs_forces_exp_finalist_under_same_source_budget(self) -> None:
        ratio_expr = {
            "type": "binary",
            "op": "div",
            "left": {"type": "feature", "index": 0},
            "right": {"type": "feature", "index": 1},
        }
        exp_neg_expr = {
            "type": "unary",
            "op": "exp",
            "arg": {
                "type": "binary",
                "op": "mul",
                "left": {"type": "const", "value": -1.0},
                "right": dict(ratio_expr),
            },
        }
        square_expr = {"type": "unary", "op": "square", "arg": dict(ratio_expr)}
        exp_member = ScreenedCandidate(
            pool_index=0,
            screen_index=0,
            name="exp(-(activation_energy/temperature))",
            expr=dict(exp_neg_expr),
            family="nonlinear",
            complexity=3.0,
            features=(0, 1),
            target_corr=0.70,
            screen_score=0.35,
            expression="exp(-(x0/x1))",
            semantic_signature="unary:exp(binary:mul(const:-1,binary:div(feature:0,feature:1)))",
            semantic_family="cross_feature_nonlinear",
            uses_piecewise_gate=False,
            residual_gain=0.08,
        )
        square_member = ScreenedCandidate(
            pool_index=1,
            screen_index=1,
            name="square(activation_energy/temperature)",
            expr=dict(square_expr),
            family="nonlinear",
            complexity=2.4,
            features=(0, 1),
            target_corr=0.74,
            screen_score=0.92,
            expression="square(x0/x1)",
            semantic_signature="unary:square(binary:div(feature:0,feature:1))",
            semantic_family="cross_feature_nonlinear",
            uses_piecewise_gate=False,
            residual_gain=0.15,
        )
        cfg = OrthogonalBasisSearchConfig(
            realization_prior_injection_mode="object_member_evidence",
            mandatory_realization_closure_mode="explicit_evidence_competition",
            same_source_over_realization_mode="inner_basis_object_budget",
            same_source_realization_budget=1,
        ).normalized()
        payload = _collect_object_realization_specs(
            base_record={"object_kind": "mechanistic_object"},
            object_members=(square_member, exp_member),
            cfg=cfg,
        )
        selected_specs = tuple(payload.get("selected_specs", ()))
        selected_signatures = {str(item.get("signature")) for item in selected_specs}
        self.assertIn("unary:square", selected_signatures)
        self.assertIn("unary:exp_neg", selected_signatures)
        catalog = tuple(payload.get("catalog", ()))
        exp_catalog = next(item for item in catalog if str(item.get("signature")) == "unary:exp_neg")
        self.assertTrue(bool(exp_catalog.get("selected")))
        self.assertEqual(str(exp_catalog.get("selection_reason")), "forced_mandatory_finalist")

    def test_mandatory_realization_audit_table_reports_not_generated_exp_neg(self) -> None:
        activation_energy = np.linspace(0.8, 3.2, 64)
        temperature = np.linspace(1.1, 3.3, 64)
        raw_x = np.stack([activation_energy, temperature], axis=1)
        ratio_expr = {
            "type": "binary",
            "op": "div",
            "left": {"type": "feature", "index": 0},
            "right": {"type": "feature", "index": 1},
        }
        target = (activation_energy / temperature).reshape(-1, 1)
        current_basis_space_genome = ({"name": "basis::ratio", "expr": {"type": "feature", "index": 0}},)
        current_assembled_genome = ({"name": "ratio", "expr": dict(ratio_expr)},)
        current_final_fit = evaluate_genome_with_ridge(
            current_assembled_genome,
            X_train=raw_x,
            y_train=target,
            l2=1e-4,
            graph_cache=None,
            train_batch_key="test::mandatory_realization::audit_not_generated",
        )
        current_inner_result = StructureSearchResult(
            genome=current_basis_space_genome,
            base_metrics=dict(current_final_fit.get("metrics_train", {}) or {}),
            final_metrics=dict(current_final_fit.get("metrics_train", {}) or {}),
            iterations=tuple(),
            weight=np.asarray(current_final_fit.get("weight"), dtype=float),
            bias=np.asarray(current_final_fit.get("bias"), dtype=float),
            score_trace=(float(dict(current_final_fit.get("metrics_train", {}) or {}).get("r2", 0.0)),),
        )
        (
            _basis_space_genome,
            _assembled_genome,
            _final_fit,
            _inner_result,
            report,
        ) = _run_mandatory_realization_closure(
            current_basis_space_genome=current_basis_space_genome,
            current_assembled_genome=current_assembled_genome,
            current_final_fit=current_final_fit,
            inner_result=current_inner_result,
            assembler_basis_genome=({"name": "basis::ratio", "expr": dict(ratio_expr)},),
            basis_feature_names=("basis::ratio",),
            basis_object_records=(
                {
                    "object_key": "basis::ratio",
                    "binding_role": "locked_basis_object",
                    "source_information_key": "source::ratio",
                    "realization_signature_catalog": (
                        {
                            "signature": "unary:exp_neg",
                            "selected": False,
                            "selection_reason": "trimmed_by_same_source_budget",
                            "evidence_term_names": ("exp(-(activation_energy/temperature))",),
                        },
                    ),
                },
            ),
            raw_X=raw_x,
            target=target,
            search_cfg=StructureSearchConfig(ridge_l2=1e-4),
            graph_cache=None,
            cfg=OrthogonalBasisSearchConfig(
                mandatory_realization_closure_mode="explicit_evidence_competition"
            ).normalized(),
        )
        self.assertEqual(str(report.get("status")), "no_mandatory_realization_candidates")
        audit_rows = tuple(report.get("realization_finalist_audit_table", ()))
        exp_row = next(item for item in audit_rows if str(item.get("realization_signature")) == "unary:exp_neg")
        self.assertEqual(str(exp_row.get("generation_status")), "not_generated")
        self.assertEqual(str(exp_row.get("finalist_status")), "not_entered")
        self.assertEqual(str(exp_row.get("competition_outcome")), "not_generated")

    def test_outermost_peeling_treats_wrappers_as_evidence_but_internal_unary_as_challenger(self) -> None:
        wrapped_ratio = {
            "type": "unary",
            "op": "sin",
            "arg": {
                "type": "binary",
                "op": "div",
                "left": {"type": "feature", "index": 0},
                "right": {"type": "feature", "index": 1},
            },
        }
        internal_topology = {
            "type": "binary",
            "op": "mul",
            "left": {"type": "feature", "index": 0},
            "right": {
                "type": "binary",
                "op": "mul",
                "left": {"type": "unary", "op": "sin", "arg": {"type": "feature", "index": 1}},
                "right": {"type": "feature", "index": 2},
            },
        }
        wrapped_source = dict(_decompose_information_source_view(wrapped_ratio).get("source_expr", {}))
        internal_source = dict(_decompose_information_source_view(internal_topology).get("source_expr", {}))
        self.assertTrue(_expr_is_native_trunk_root(wrapped_source))
        self.assertFalse(_expr_is_native_trunk_root(internal_source))

    def test_native_trunk_seat_requirement_forces_group_to_keep_native_object(self) -> None:
        feature_names = ("voltage", "resistance", "temperature")
        ratio_row = ScreenedCandidate(
            pool_index=0,
            screen_index=0,
            name="voltage_over_resistance",
            expr={"type": "binary", "op": "div", "left": {"type": "feature", "index": 0}, "right": {"type": "feature", "index": 1}},
            family="ratio",
            complexity=2.0,
            features=(0, 1),
            target_corr=0.74,
            screen_score=0.62,
            expression="voltage / resistance",
            semantic_signature="binary:div(feature:0,feature:1)",
            semantic_family="cross_feature_ratio",
            uses_piecewise_gate=False,
            residual_gain=0.16,
            native_trunk_root=True,
            native_trunk_floor_passed=True,
            native_trunk_global_gain=0.16,
            native_trunk_interval_min_gain=0.07,
            native_trunk_interval_mean_gain=0.09,
            selection_channel="native_trunk",
        )
        challenger_row = ScreenedCandidate(
            pool_index=1,
            screen_index=1,
            name="fancy_voltage_exp",
            expr={
                "type": "binary",
                "op": "mul",
                "left": {"type": "feature", "index": 0},
                "right": {"type": "unary", "op": "exp", "arg": {"type": "feature", "index": 1}},
            },
            family="interaction",
            complexity=4.0,
            features=(0, 1),
            target_corr=0.92,
            screen_score=0.91,
            expression="voltage * exp(resistance)",
            semantic_signature="binary:mul(feature:0,unary:exp(feature:1))",
            semantic_family="cross_feature_product",
            uses_piecewise_gate=False,
            residual_gain=0.30,
            selection_channel="challenger",
        )
        support_row = ScreenedCandidate(
            pool_index=2,
            screen_index=2,
            name="temperature",
            expr={"type": "feature", "index": 2},
            family="linear",
            complexity=1.0,
            features=(2,),
            target_corr=0.40,
            screen_score=0.38,
            expression="temperature",
            semantic_signature="feature:2",
            semantic_family="linear_feature",
            uses_piecewise_gate=False,
            residual_gain=0.06,
            selection_channel="challenger",
        )
        rng = np.random.default_rng(7)
        train_matrix = np.column_stack(
            [
                rng.uniform(0.2, 1.0, size=32),
                rng.normal(0.0, 1.0, size=32),
                rng.uniform(-1.0, 1.0, size=32),
            ]
        )
        raw_x = np.column_stack(
            [
                rng.uniform(1.0, 2.0, size=32),
                rng.uniform(2.0, 3.5, size=32),
                rng.uniform(-1.0, 1.0, size=32),
            ]
        )
        y_train = 2.5 * train_matrix[:, 0] + 0.3 * train_matrix[:, 2]
        cfg = OrthogonalBasisSearchConfig(
            min_basis_count=2,
            max_basis_count=2,
            group_count=4,
            outer_search_beam_width=4,
            outer_search_branching_factor=2,
            max_pair_abs_corr=0.98,
            require_native_trunk_candidate_in_group=True,
            min_native_trunk_basis_terms=1,
            native_trunk_candidate_screen_reserve=1,
            require_gate_candidate_in_group=False,
            require_periodic_candidate_in_group=False,
        ).normalized()
        groups = _discover_group_candidates(
            screened=(ratio_row, challenger_row, support_row),
            train_matrix=np.asarray(train_matrix, dtype=float),
            y_train=np.asarray(y_train, dtype=float),
            raw_X=np.asarray(raw_x, dtype=float),
            feature_names=feature_names,
            interference_context={},
            periodic_context={},
            cfg=cfg,
            seed_genome=None,
        )
        self.assertTrue(groups)
        self.assertTrue(
            all(
                any(bool(row.native_trunk_floor_passed) for row in tuple(group.get("rows", ()) or ()))
                for group in groups
            )
        )


if __name__ == "__main__":
    unittest.main()
