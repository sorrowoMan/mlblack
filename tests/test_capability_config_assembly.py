from __future__ import annotations

import unittest

from config import (
    CapabilitySpec,
    describe_execution_spec_schema,
    FlowAssemblySpec,
    NumericizerSpec,
    TrainerAssemblySpec,
    build_flow_components,
    describe_registered,
    describe_trainers,
    list_registered,
)
from project.scaffold import _build_semantic_flow_spec, build_scaffold_spec
from schema import TRAINER_CONTRACTS, TRAINER_RESOURCE_PROFILES, get_trainer_resource_profiles


class TestCapabilityConfigAssembly(unittest.TestCase):
    def test_execution_spec_schema_exposes_l0_enums_and_catalog(self) -> None:
        schema = describe_execution_spec_schema()
        fields = dict(schema.get("fields", {}))
        backend_field = dict(fields.get("backend", {}))
        gpu_field = dict(fields.get("gpu_strategy", {}))
        device_catalog = list(schema.get("device_catalog", []))

        self.assertEqual(str(schema.get("plane")), "L0")
        self.assertIn("serial", tuple(backend_field.get("enum", ())))
        self.assertIn("thread", tuple(backend_field.get("enum", ())))
        self.assertIn("process", tuple(backend_field.get("enum", ())))
        self.assertEqual(tuple(gpu_field.get("enum", ())), ("none", "fixed", "round_robin", "auto"))
        self.assertTrue(any(str(row.get("kind")) == "cpu" for row in device_catalog))
        self.assertTrue(any(str(row.get("kind")) == "cuda" for row in device_catalog))

    def test_registry_builds_declared_capabilities(self) -> None:
        spec = FlowAssemblySpec(
            trainer=TrainerAssemblySpec(trainer_key="ridge", trainer_params={"l2": 0.0}),
            numericizer=NumericizerSpec(key="default", params={}),
            capabilities=(
                CapabilitySpec(
                    key="noop",
                    params={
                        "name": "config_declared_capability",
                        "priority": 2,
                    },
                ),
            ),
        )

        components = build_flow_components(spec)
        caps = tuple(components.get("capabilities", tuple()))
        self.assertEqual(len(caps), 1)
        self.assertEqual(str(getattr(caps[0], "name", "")), "config_declared_capability")
        self.assertEqual(int(getattr(caps[0], "priority", 0)), 2)

        registered = list_registered()
        self.assertIn("capabilities", registered)
        self.assertIn("noop", tuple(registered["capabilities"]))

    def test_scaffold_payload_parses_capability_and_model_fields(self) -> None:
        payload = {
            "data": {
                "csv_path": "data/processed.csv",
                "target_col": "target",
            },
            "train": {
                "trainer_key": "ridge",
                "model_spec": {
                    "model_id": "scaffold_model",
                    "feature_names": ["x1", "x2"],
                    "target_names": ["y"],
                },
                "training_init": {
                    "mode": "resume",
                    "parent_state": "runs/prev/latest.pt",
                    "metadata": {"case": "config_resume"},
                },
                "state_backend": {
                    "context": {"backend": "memory"},
                    "snapshot": {"backend": "memory"},
                },
                "execution": {
                    "backend": "thread",
                    "max_workers": 3,
                    "fail_fast": False,
                    "gpu_strategy": "none",
                    "gpu_devices": [],
                    "default_device": "cpu",
                },
                "capabilities": [
                    {
                        "key": "noop",
                        "params": {"name": "scaffold_declared_cap"},
                    }
                ],
                "capability_strict": True,
            },
        }

        spec = build_scaffold_spec(payload)
        self.assertEqual(len(spec.train.capabilities), 1)
        self.assertEqual(str(spec.train.capabilities[0]["key"]), "noop")
        self.assertEqual(
            str(spec.train.capabilities[0]["params"]["name"]),
            "scaffold_declared_cap",
        )
        self.assertTrue(bool(spec.train.capability_strict))
        self.assertEqual(str(dict(spec.train.model_spec or {}).get("model_id")), "scaffold_model")
        self.assertEqual(str(dict(spec.train.training_init or {}).get("mode")), "resume")
        self.assertEqual(str(dict(spec.train.training_init or {}).get("parent_state")), "runs/prev/latest.pt")
        self.assertEqual(str(dict(spec.train.state_backend or {}).get("context", {}).get("backend")), "memory")
        self.assertEqual(str(dict(spec.train.execution or {}).get("backend")), "thread")
        self.assertEqual(int(dict(spec.train.execution or {}).get("max_workers", 0)), 3)

        flow_spec = _build_semantic_flow_spec(spec)
        self.assertEqual(str(dict(flow_spec.training_init or {}).get("mode")), "resume")
        self.assertEqual(str(dict(flow_spec.training_init or {}).get("parent_state")), "runs/prev/latest.pt")
        self.assertEqual(str(flow_spec.execution.backend), "thread")
        self.assertEqual(int(flow_spec.execution.max_workers or 0), 3)
        self.assertFalse(bool(flow_spec.execution.fail_fast))
        self.assertEqual(str(flow_spec.execution.default_device), "cpu")

    def test_describe_trainers_exposes_normalized_contract(self) -> None:
        trainers = describe_trainers()
        ridge = dict(trainers["ridge"])
        sklearn_mlp = dict(trainers["sklearn_mlp"])
        xgboost = dict(trainers["xgboost"])

        self.assertTrue(bool(ridge.get("contract", {}).get("training_modes", {}).get("resume")))
        self.assertTrue(bool(ridge.get("contract", {}).get("trainer_state", {}).get("save_load")))
        self.assertEqual(int(ridge.get("contract", {}).get("execution_resources", {}).get("request", {}).get("threads", 0)), 1)
        self.assertTrue(bool(sklearn_mlp.get("contract", {}).get("training_modes", {}).get("warm_start")))
        self.assertFalse(bool(sklearn_mlp.get("contract", {}).get("training_modes", {}).get("resume")))
        self.assertGreaterEqual(
            int(xgboost.get("contract", {}).get("execution_resources", {}).get("request", {}).get("threads", 0)),
            1,
        )

    def test_describe_trainers_exposes_mechanism_binding_levels(self) -> None:
        trainers = describe_trainers()

        def binding_map(trainer_key: str) -> dict[str, dict[str, object]]:
            rows = list(dict(trainers[trainer_key]).get("contract", {}).get("mechanism_bindings", []))
            return {str(row["mechanism_key"]): dict(row) for row in rows}

        ridge_bindings = binding_map("ridge")
        random_forest_bindings = binding_map("random_forest")
        extra_trees_bindings = binding_map("extra_trees")
        bagging_bindings = binding_map("bagging")
        adaboost_bindings = binding_map("adaboost")
        xgboost_bindings = binding_map("xgboost")
        sklearn_mlp_bindings = binding_map("sklearn_mlp")
        symbolic_bindings = binding_map("symbolic")

        self.assertEqual(str(ridge_bindings["sampling"]["binding_level"]), "optional")
        self.assertEqual(str(sklearn_mlp_bindings["aggregation"]["binding_level"]), "optional")
        self.assertEqual(str(random_forest_bindings["sampling"]["binding_level"]), "bound")
        self.assertEqual(str(random_forest_bindings["aggregation"]["binding_level"]), "defining")
        self.assertEqual(str(extra_trees_bindings["sampling"]["binding_level"]), "bound")
        self.assertEqual(str(bagging_bindings["aggregation"]["binding_level"]), "defining")
        self.assertEqual(str(adaboost_bindings["sample_weighting"]["binding_level"]), "defining")
        self.assertEqual(str(xgboost_bindings["state_signal_view"]["binding_level"]), "defining")
        self.assertEqual(str(xgboost_bindings["aggregation"]["binding_level"]), "defining")
        self.assertEqual(str(symbolic_bindings["state_signal_view"]["binding_level"]), "bound")

    def test_describe_registered_trainer_metadata_carries_contract_projection(self) -> None:
        registered = describe_registered()
        trainers = {str(item["key"]): dict(item) for item in registered["trainers"]}

        self.assertIn("ridge", trainers)
        ridge_meta = dict(trainers["ridge"].get("metadata", {}))
        ridge_contract = dict(ridge_meta.get("trainer_contract", {}))
        self.assertTrue(bool(ridge_contract.get("training_modes", {}).get("resume")))
        self.assertTrue(bool(ridge_contract.get("trainer_state", {}).get("enabled")))
        self.assertEqual(int(ridge_contract.get("execution_resources", {}).get("request", {}).get("threads", 0)), 1)
        self.assertEqual(
            str(list(ridge_contract.get("mechanism_bindings", []))[0].get("binding_level")),
            "optional",
        )

    def test_symbolic_trainer_contract_exposes_search_mechanism_contracts(self) -> None:
        trainers = describe_trainers()
        symbolic = dict(trainers["symbolic"])
        contract = dict(symbolic.get("contract", {}))
        rows = {str(row["mechanism_key"]): dict(row) for row in contract.get("search_mechanism_contracts", [])}

        self.assertIn("beam_selection", rows)
        self.assertIn("expression_graph_cache", rows)
        self.assertTrue(bool(rows["beam_selection"].get("replayable")))
        self.assertFalse(bool(rows["expression_graph_cache"].get("affects_family_signature")))

    def test_schema_exports_trainer_resource_profiles_for_ui(self) -> None:
        self.assertIn("ridge", TRAINER_CONTRACTS)
        self.assertIn("xgboost", TRAINER_RESOURCE_PROFILES)

        ridge_contract = dict(TRAINER_CONTRACTS["ridge"])
        ridge_profile = dict(TRAINER_RESOURCE_PROFILES["ridge"])
        runtime_profiles = get_trainer_resource_profiles()

        self.assertEqual(
            int(ridge_contract.get("contract", {}).get("execution_resources", {}).get("request", {}).get("threads", 0)),
            1,
        )
        self.assertEqual(int(ridge_profile.get("request", {}).get("threads", 0)), 1)
        self.assertEqual(int(runtime_profiles["ridge"].get("request", {}).get("threads", 0)), 1)
        self.assertIn("components", ridge_profile)


if __name__ == "__main__":
    unittest.main()
