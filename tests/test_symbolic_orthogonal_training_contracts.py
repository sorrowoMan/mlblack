from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import TrainerAssemblySpec, build_trainer
from core.common.contracts import ProcessedDataset
from training import TrainTask, TrainingInit


def _make_processed_dataset(seed: int = 20260504) -> ProcessedDataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(96, 4))
    y = (
        1.9 * (x[:, 0] * x[:, 1])
        + 0.8 * np.sin(x[:, 2])
        - 0.35 * x[:, 3]
        + rng.normal(scale=0.02, size=x.shape[0])
    ).reshape(-1, 1)
    return ProcessedDataset(
        X_train=np.asarray(x, dtype=float),
        y_train=np.asarray(y, dtype=float),
        feature_names=("x0", "x1", "x2", "x3"),
        target_names=("y",),
        metadata={"source": "symbolic_orthogonal_training_contract_test"},
    )


class TestSymbolicOrthogonalTrainingContracts(unittest.TestCase):
    def _build_trainer(self) -> object:
        return build_trainer(
            TrainerAssemblySpec(
                trainer_key="symbolic",
                trainer_params={
                    "parameter_backend": "ridge",
                    "task": "point",
                    "structure_engine": {
                        "structure_mode": "orthogonal_basis_search",
                        "search_driver": "orthogonal_basis",
                        "dynamic_pool_enabled": True,
                        "metadata": {"supports_piecewise_basis": True},
                    },
                    "candidate_limit": 36,
                    "group_count": 5,
                    "seed_candidate_count": 8,
                    "min_basis_count": 2,
                    "max_basis_count": 4,
                    "rolling_folds": 2,
                    "selection_mode": "rmse_first",
                    "gate_feature_names": ("x2",),
                    "search_graph_cache_enabled": False,
                },
            )
        )

    def test_fit_task_emits_orthogonal_reports_and_symbolic_schema(self) -> None:
        trainer = self._build_trainer()
        result = trainer.fit_task(
            TrainTask.from_data(_make_processed_dataset(), task_id="symbolic_orthogonal::fresh"),
            TrainingInit(mode="fresh"),
        )

        self.assertIsNotNone(result.trainer_state)
        metadata = dict(result.artifact.metadata)
        self.assertIn("basis_overlap_report", metadata)
        self.assertIn("residual_complementarity_report", metadata)
        self.assertIn("semantic_dedup_report", metadata)
        self.assertIn("assembler_budget", metadata)
        self.assertIn("inner_symbolic_search", metadata)
        self.assertIn("orthogonal_search_objective", metadata)
        self.assertIn("orthogonal_outer_basis_genome", metadata)
        self.assertIn("symbolic_structure_surface", metadata)
        self.assertEqual(str(metadata.get("structure_head")), "expression")
        self.assertEqual(str(metadata.get("search_input_space")), "basis_object_space")
        self.assertEqual(str(metadata.get("pool_expansion_unit")), "basis_object")
        self.assertEqual(str(metadata.get("gradient_guidance_mode")), "basis_object_gradient")
        self.assertTrue(bool(dict(metadata.get("stage_head_protocols", {}) or {})))
        self.assertTrue(bool(dict(metadata.get("basis_context", {}) or {})))
        self.assertTrue(bool(dict(metadata.get("basis_object_gradient_pool", {}) or {})))
        schema = dict(metadata.get("symbolic_artifact_schema", {}))
        head_semantics = dict(schema.get("head_semantics", {}))
        basis_structure = dict(schema.get("basis_structure", {}))
        assembler_structure = dict(schema.get("assembler_structure", {}))
        orthogonality = dict(basis_structure.get("orthogonality_status", {}))
        gate_basis = dict(schema.get("piecewise_gate_basis", {}))
        self.assertEqual(str(head_semantics.get("structure_head")), "expression")
        self.assertEqual(str(head_semantics.get("search_input_space")), "basis_object_space")
        self.assertEqual(str(head_semantics.get("pool_expansion_unit")), "basis_object")
        self.assertEqual(str(head_semantics.get("gradient_guidance_mode")), "basis_object_gradient")
        self.assertEqual(str(assembler_structure.get("structure_head")), "expression")
        self.assertEqual(str(assembler_structure.get("search_input_space")), "basis_object_space")
        self.assertEqual(str(assembler_structure.get("pool_expansion_unit")), "basis_object")
        self.assertEqual(str(assembler_structure.get("gradient_guidance_mode")), "basis_object_gradient")
        self.assertTrue(bool(dict(assembler_structure.get("object_gradient_pool", {}) or {})))
        self.assertEqual(str(orthogonality.get("status")), "reported")
        self.assertGreater(float(orthogonality.get("orthogonality_score", 0.0)), 0.0)
        self.assertIn(str(gate_basis.get("status")), {"configured", "enabled"})
        self.assertEqual(str(metadata.get("training_init", {}).get("mode")), "fresh")
        self.assertTrue(bool(tuple(metadata.get("orthogonal_outer_basis_genome", ()))))
        inner_search = dict(metadata.get("inner_symbolic_search", {}))
        self.assertEqual(str(inner_search.get("protocol")), "budgeted_symbolic_assembler")
        symbolic = dict(metadata.get("symbolic", {}) or {})
        structure_engine = dict(symbolic.get("structure_engine", {}) or {})
        self.assertEqual(str(structure_engine.get("search_driver")), "orthogonal_basis_set_search")
        self.assertEqual(
            str(structure_engine.get("screening_protocol")),
            "target_corr+residual_gain+semantic_novelty+consensus_prior",
        )
        self.assertEqual(
            str(structure_engine.get("outer_search_protocol")),
            "beam_basis_set_structure_search",
        )
        engine_metadata = dict(structure_engine.get("metadata", {}) or {})
        self.assertEqual(
            str(engine_metadata.get("screening_protocol")),
            "target_corr+residual_gain+semantic_novelty+consensus_prior",
        )
        self.assertEqual(
            str(engine_metadata.get("outer_search_protocol")),
            "beam_basis_set_structure_search",
        )
        search_summary = dict(metadata.get("search", {}) or {})
        self.assertEqual(str(metadata.get("realization_prior_injection_protocol")), "RealizationPriorInjection")
        self.assertEqual(
            str(metadata.get("mandatory_realization_closure_protocol")),
            "MandatoryRealizationClosure",
        )
        self.assertEqual(
            str(metadata.get("periodic_realization_competition_protocol")),
            "PeriodicRealizationCompetition",
        )
        self.assertEqual(
            str(metadata.get("causal_hierarchy_reuse_isolation_protocol")),
            "CausalHierarchyReuseIsolation",
        )
        self.assertEqual(int(search_summary.get("outer_search_beam_width", 0)), 12)
        self.assertEqual(int(search_summary.get("outer_search_branching_factor", 0)), 3)
        self.assertEqual(int(search_summary.get("outer_search_max_expansions", 0)), 96)
        consensus_prior_summary = dict(search_summary.get("consensus_prior_summary", {}) or {})
        self.assertEqual(int(consensus_prior_summary.get("row_count", -1)), 0)
        equivalence = dict(metadata.get("equivalence_expression_handling", {}) or {})
        self.assertIn("realization_prior_injection_mode", tuple(equivalence.get("implemented_submodes", ()) or ()))
        self.assertIn("mandatory_realization_closure_mode", tuple(equivalence.get("implemented_submodes", ()) or ()))
        interference = dict(metadata.get("interference_feature_handling", {}) or {})
        self.assertIn(
            "causal_hierarchy_reuse_isolation_mode",
            tuple(interference.get("implemented_submodes", ()) or ()),
        )
        self.assertIn(
            "proxy_trunk_disqualification_mode",
            tuple(interference.get("implemented_submodes", ()) or ()),
        )
        self.assertIn(
            "parasitic_rejection_mode",
            tuple(interference.get("implemented_submodes", ()) or ()),
        )

    def test_state_roundtrip_and_warm_start_reuses_parent_genome(self) -> None:
        data = _make_processed_dataset(seed=20260505)
        trainer = self._build_trainer()
        parent_result = trainer.fit_task(
            TrainTask.from_data(data, task_id="symbolic_orthogonal::parent"),
            TrainingInit(mode="fresh"),
        )
        self.assertIsNotNone(parent_result.trainer_state)

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "symbolic_orthogonal_state.pt"
            trainer.save_trainer_state(state_path, parent_result.trainer_state)
            loaded_state = trainer.load_trainer_state(state_path)
            resumed = self._build_trainer().fit_task(
                TrainTask.from_data(data, task_id="symbolic_orthogonal::resume"),
                TrainingInit(mode="resume", parent_state=loaded_state),
            )
            self.assertIsNotNone(resumed.trainer_state)
            self.assertTrue(bool(resumed.artifact.metadata.get("resume", {}).get("enabled")))
            self.assertEqual(str(getattr(loaded_state, "payload", {}).get("seed_protocol")), "outer_basis_genome")

        warm = self._build_trainer().fit_task(
            TrainTask.from_data(data, task_id="symbolic_orthogonal::warm"),
            TrainingInit(mode="warm_start", parent_artifact=parent_result.artifact),
        )
        self.assertIsNotNone(warm.trainer_state)
        self.assertEqual(str(warm.artifact.metadata.get("training_init", {}).get("parent_kind")), "artifact")
        self.assertEqual(str(getattr(warm.trainer_state, "payload", {}).get("seed_protocol")), "outer_basis_genome")


if __name__ == "__main__":
    unittest.main()
