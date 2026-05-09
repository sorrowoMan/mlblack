from __future__ import annotations

import unittest

import numpy as np

from config import TrainerAssemblySpec, build_trainer, list_registered
from core.symbolic import SymbolicStructureEngineSpec
from core.trainers.symbolic_torch_trainer import (
    SymbolicTorchSurrogateTrainer,
    SymbolicTorchTrainerConfig,
)
from core.trainers.symbolic_torch_interval_trainer import (
    SymbolicTorchIntervalTrainer,
    SymbolicTorchIntervalTrainerConfig,
)


class TestSymbolicUnifiedAssembly(unittest.TestCase):
    def test_registry_exposes_unified_symbolic_entry(self) -> None:
        registered = list_registered()
        self.assertIn("trainers", registered)
        self.assertIn("symbolic", tuple(registered["trainers"]))

    def test_build_symbolic_interval_from_unified_family_spec(self) -> None:
        trainer = build_trainer(
            TrainerAssemblySpec(
                trainer_key="symbolic",
                trainer_params={
                    "parameter_backend": "torch",
                    "task": "interval",
                    "structure_engine": {
                        "structure_mode": "stagewise_search",
                        "search_driver": "nsgablack",
                        "dynamic_pool_enabled": True,
                    },
                    "device": "cpu",
                    "epochs": 2,
                    "batch_size": 8,
                },
            )
        )
        self.assertEqual(str(getattr(trainer, "name", "")), "symbolic_torch_interval")
        self.assertEqual(str(getattr(trainer, "requested_trainer_key", "")), "symbolic")
        self.assertEqual(str(getattr(trainer, "symbolic_router_target", "")), "symbolic_torch_interval")
        family = getattr(trainer, "symbolic_family_spec", None)
        self.assertIsNotNone(family)
        self.assertEqual(str(family.structure_engine.structure_mode), "stagewise_search")
        self.assertEqual(str(trainer.config.structure_engine.structure_mode), "stagewise_search")

    def test_build_symbolic_point_from_unified_family_spec(self) -> None:
        trainer = build_trainer(
            TrainerAssemblySpec(
                trainer_key="symbolic",
                trainer_params={
                    "parameter_backend": "torch",
                    "task": "point",
                    "structure_engine": {
                        "structure_mode": "stagewise_search",
                        "search_driver": "nsgablack",
                        "dynamic_pool_enabled": True,
                    },
                    "device": "cpu",
                    "epochs": 2,
                    "batch_size": 8,
                },
            )
        )
        self.assertEqual(str(getattr(trainer, "name", "")), "symbolic_torch")
        self.assertEqual(str(getattr(trainer, "symbolic_router_target", "")), "symbolic_torch")
        family = getattr(trainer, "symbolic_family_spec", None)
        self.assertIsNotNone(family)
        self.assertEqual(str(family.structure_engine.structure_mode), "stagewise_search")
        self.assertEqual(str(trainer.config.structure_engine.structure_mode), "stagewise_search")

    def test_build_symbolic_stagewise_from_unified_family_spec_exposes_structure_engine(self) -> None:
        trainer = build_trainer(
            TrainerAssemblySpec(
                trainer_key="symbolic",
                trainer_params={
                    "parameter_backend": "ridge",
                    "task": "point",
                    "structure_engine": {
                        "structure_mode": "stagewise_search",
                        "search_driver": "nsgablack",
                        "dynamic_pool_enabled": True,
                    },
                    "search_max_added_terms": 1,
                    "search_topk_features": 3,
                    "search_max_pair_terms": 4,
                    "search_max_candidates_per_iter": 32,
                    "search_candidate_keep_top": 6,
                    "search_online_beam_enabled": False,
                    "search_path_memory_enabled": False,
                    "search_graph_cache_enabled": False,
                    "search_joint_bundle_enabled": False,
                },
            )
        )
        self.assertEqual(str(getattr(trainer, "name", "")), "symbolic_stagewise")
        self.assertEqual(str(getattr(trainer, "symbolic_router_target", "")), "symbolic_stagewise")
        family = getattr(trainer, "symbolic_family_spec", None)
        self.assertIsNotNone(family)
        self.assertEqual(str(family.structure_engine.structure_mode), "stagewise_search")
        self.assertEqual(str(trainer.config.structure_engine.structure_mode), "stagewise_search")
        self.assertEqual(str(trainer.config.structure_engine.search_driver), "nsgablack")

    def test_build_symbolic_orthogonal_from_unified_family_spec(self) -> None:
        trainer = build_trainer(
            TrainerAssemblySpec(
                trainer_key="symbolic",
                trainer_params={
                    "parameter_backend": "ridge",
                    "task": "point",
                    "structure_engine": {
                        "structure_mode": "orthogonal_basis_search",
                        "search_driver": "orthogonal_basis",
                        "dynamic_pool_enabled": True,
                    },
                    "candidate_limit": 24,
                    "group_count": 4,
                    "seed_candidate_count": 6,
                    "min_basis_count": 2,
                    "max_basis_count": 4,
                    "search_graph_cache_enabled": False,
                },
            )
        )
        self.assertEqual(str(getattr(trainer, "name", "")), "symbolic_orthogonal")
        self.assertEqual(str(getattr(trainer, "requested_trainer_key", "")), "symbolic")
        self.assertEqual(str(getattr(trainer, "symbolic_router_target", "")), "symbolic_orthogonal")
        family = getattr(trainer, "symbolic_family_spec", None)
        self.assertIsNotNone(family)
        self.assertEqual(str(family.structure_engine.structure_mode), "orthogonal_basis_search")
        self.assertEqual(str(family.structure_engine.search_driver), "orthogonal_basis")

    def test_legacy_interval_builder_injects_structure_engine(self) -> None:
        trainer = build_trainer(
            TrainerAssemblySpec(
                trainer_key="symbolic_torch_interval",
                trainer_params={
                    "stagewise_warmup_enabled": True,
                    "device": "cpu",
                },
            )
        )
        self.assertEqual(str(getattr(trainer, "name", "")), "symbolic_torch_interval")
        self.assertEqual(
            str(trainer.config.structure_engine.structure_mode),
            "stagewise_warmup_then_seed_library",
        )

    def test_interval_build_genome_consumes_structure_engine(self) -> None:
        trainer = SymbolicTorchIntervalTrainer(
            config=SymbolicTorchIntervalTrainerConfig(
                structure_engine=SymbolicStructureEngineSpec(
                    structure_mode="stagewise_search",
                    search_driver="nsgablack",
                    dynamic_pool_enabled=True,
                ),
                epochs=1,
                batch_size=4,
                device="cpu",
            )
        )

        def _fake_stagewise(**kwargs):
            engine = kwargs["engine"]
            return (
                (
                    {
                        "name": "x0",
                        "kind": "identity",
                        "arity": 1,
                    },
                ),
                {
                    "status": "ok",
                    "engine_spec": engine.as_dict(),
                },
            )

        trainer._stagewise_search_genome = _fake_stagewise  # type: ignore[attr-defined]
        genome, info = trainer._build_genome(
            np.ones((8, 2), dtype=float),
            np.ones((8, 1), dtype=float),
            feature_names=("x0", "x1"),
            target_names=("y",),
            metadata={},
            seed=42,
        )
        self.assertEqual(len(tuple(genome)), 1)
        self.assertEqual(str(info["engine_spec"]["structure_mode"]), "stagewise_search")
        self.assertEqual(str(info["status"]), "ok")

    def test_point_build_genome_consumes_structure_engine(self) -> None:
        trainer = SymbolicTorchSurrogateTrainer(
            config=SymbolicTorchTrainerConfig(
                structure_engine=SymbolicStructureEngineSpec(
                    structure_mode="stagewise_search",
                    search_driver="nsgablack",
                    dynamic_pool_enabled=True,
                ),
                epochs=1,
                batch_size=4,
                device="cpu",
            )
        )

        def _fake_stagewise(**kwargs):
            engine = kwargs["engine"]
            return (
                (
                    {
                        "name": "x0",
                        "kind": "identity",
                        "arity": 1,
                    },
                ),
                {
                    "status": "ok",
                    "engine_spec": engine.as_dict(),
                },
            )

        trainer._stagewise_search_genome = _fake_stagewise  # type: ignore[attr-defined]
        genome, info = trainer._build_genome(
            np.ones((8, 2), dtype=float),
            np.ones((8, 1), dtype=float),
            feature_names=("x0", "x1"),
            target_names=("y",),
            metadata={},
            seed=42,
        )
        self.assertEqual(len(tuple(genome)), 1)
        self.assertEqual(str(info["engine_spec"]["structure_mode"]), "stagewise_search")
        self.assertEqual(str(info["status"]), "ok")


if __name__ == "__main__":
    unittest.main()
