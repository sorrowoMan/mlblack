from __future__ import annotations

import unittest

from config import TrainerAssemblySpec, build_trainer, list_registered


class TestFamilyRouterProtocol(unittest.TestCase):
    def test_registry_exposes_formal_family_entries(self) -> None:
        registered = list_registered()
        trainer_keys = tuple(registered["trainers"])
        self.assertIn("linear", trainer_keys)
        self.assertIn("neural", trainer_keys)
        self.assertIn("tree_ensemble", trainer_keys)
        self.assertIn("tree_boosting", trainer_keys)
        self.assertIn("symbolic", trainer_keys)

    def test_linear_formal_family_routes_to_ridge(self) -> None:
        trainer = build_trainer(
            TrainerAssemblySpec(
                trainer_key="linear",
                trainer_params={
                    "family_spec": {
                        "regularization": {"l2": 0.25},
                    }
                },
            )
        )
        self.assertEqual(str(getattr(trainer, "name", "")), "ridge")
        self.assertEqual(str(getattr(trainer, "requested_trainer_key", "")), "linear")
        self.assertEqual(str(getattr(trainer, "family_router_family", "")), "linear")
        self.assertEqual(str(getattr(trainer, "family_router_target", "")), "ridge")
        self.assertEqual(str(getattr(trainer, "linear_router_target", "")), "ridge")
        family = getattr(trainer, "linear_family_spec", None)
        self.assertIsNotNone(family)
        assert family is not None
        self.assertEqual(str(family.trainer_key), "linear")

    def test_neural_formal_family_routes_to_sklearn_variant(self) -> None:
        trainer = build_trainer(
            TrainerAssemblySpec(
                trainer_key="neural",
                trainer_params={
                    "family_spec": {
                        "backend": {
                            "parameter_backend": "sklearn",
                            "runtime_backend": "scikit-learn",
                            "trainer_kind": "mlp",
                        },
                        "optimization": {
                            "solver": "adam",
                            "max_steps": 8,
                            "early_stopping": False,
                        },
                    }
                },
            )
        )
        self.assertEqual(str(getattr(trainer, "name", "")), "sklearn_mlp")
        self.assertEqual(str(getattr(trainer, "requested_trainer_key", "")), "neural")
        self.assertEqual(str(getattr(trainer, "family_router_family", "")), "neural")
        self.assertEqual(str(getattr(trainer, "family_router_target", "")), "sklearn_mlp")
        self.assertEqual(str(getattr(trainer, "neural_router_target", "")), "sklearn_mlp")
        family = getattr(trainer, "neural_family_spec", None)
        self.assertIsNotNone(family)
        assert family is not None
        self.assertEqual(str(family.trainer_key), "neural")
        self.assertEqual(str(family.backend.parameter_backend), "sklearn")

    def test_tree_boosting_formal_family_routes_to_xgboost(self) -> None:
        trainer = build_trainer(
            TrainerAssemblySpec(
                trainer_key="tree_boosting",
                trainer_params={
                    "family_spec": {
                        "boosting": {"n_estimators": 8},
                        "regularization": {"max_depth": 3},
                    }
                },
            )
        )
        self.assertEqual(str(getattr(trainer, "name", "")), "xgboost")
        self.assertEqual(str(getattr(trainer, "requested_trainer_key", "")), "tree_boosting")
        self.assertEqual(str(getattr(trainer, "family_router_family", "")), "tree_boosting")
        self.assertEqual(str(getattr(trainer, "family_router_target", "")), "xgboost")
        self.assertEqual(str(getattr(trainer, "tree_boosting_router_target", "")), "xgboost")
        family = getattr(trainer, "tree_boosting_family_spec", None)
        self.assertIsNotNone(family)
        assert family is not None
        self.assertEqual(str(family.trainer_key), "tree_boosting")

    def test_tree_ensemble_formal_family_routes_to_bagging(self) -> None:
        trainer = build_trainer(
            TrainerAssemblySpec(
                trainer_key="tree_ensemble",
                trainer_params={
                    "family_spec": {
                        "ensemble": {
                            "ensemble_kind": "bagging",
                            "n_estimators": 6,
                        },
                        "sampling": {
                            "bootstrap": True,
                            "max_samples": 0.8,
                        },
                    }
                },
            )
        )
        self.assertEqual(str(getattr(trainer, "name", "")), "bagging")
        self.assertEqual(str(getattr(trainer, "requested_trainer_key", "")), "tree_ensemble")
        self.assertEqual(str(getattr(trainer, "family_router_family", "")), "tree_ensemble")
        self.assertEqual(str(getattr(trainer, "family_router_target", "")), "bagging")
        self.assertEqual(str(getattr(trainer, "tree_ensemble_router_target", "")), "bagging")
        family = getattr(trainer, "tree_family_spec", None)
        self.assertIsNotNone(family)
        assert family is not None
        self.assertEqual(str(family.trainer_key), "tree_ensemble")
        self.assertEqual(str(family.ensemble.ensemble_kind), "bagging")


if __name__ == "__main__":
    unittest.main()
