from __future__ import annotations

import unittest
from unittest.mock import patch

from core.symbolic import (
    SYMBOLIC_FORMAL_PRESET_KEY,
    SYMBOLIC_LEGACY_PRESET_KEYS,
    SymbolicRouteSpec,
    SymbolicRegimeDiscoveryContract,
    SymbolicSearchMechanismContract,
    build_symbolic_search_mechanism_contracts,
    build_unified_symbolic_family_spec,
    canonical_symbolic_preset_key,
    is_legacy_symbolic_preset,
    legacy_symbolic_family_spec,
    match_symbolic_routes,
    resolve_symbolic_route_spec,
    resolve_symbolic_router_target,
    serialize_symbolic_route_registry,
    symbolic_route_registry,
    symbolic_surface_contract,
)


class TestSymbolicFamilyContract(unittest.TestCase):
    @staticmethod
    def _contract_kwargs(contract: SymbolicSearchMechanismContract) -> dict[str, object]:
        payload = dict(contract.as_dict())
        payload.pop("contract_version", None)
        return payload

    @staticmethod
    def _structure_contract_kwargs(contract: SymbolicRegimeDiscoveryContract) -> dict[str, object]:
        payload = dict(contract.as_dict())
        payload.pop("contract_version", None)
        payload.pop("contract_type", None)
        return payload

    def test_legacy_stagewise_maps_to_structure_first_family(self) -> None:
        spec = legacy_symbolic_family_spec("symbolic_stagewise")
        self.assertEqual(spec.structure_engine.structure_mode, "stagewise_search")
        self.assertEqual(spec.structure_engine.search_driver, "nsgablack")
        self.assertEqual(spec.parameter_backend.backend, "ridge")
        self.assertEqual(spec.task_head.task, "point")

    def test_legacy_interval_warmup_maps_to_interval_head(self) -> None:
        spec = legacy_symbolic_family_spec(
            "symbolic_torch_interval",
            trainer_params={
                "stagewise_warmup_enabled": True,
                "conformal_calibration": True,
            },
        )
        self.assertEqual(spec.structure_engine.structure_mode, "stagewise_warmup_then_seed_library")
        self.assertEqual(spec.structure_engine.search_driver, "nsgablack_warmup")
        self.assertEqual(spec.parameter_backend.backend, "torch")
        self.assertEqual(spec.task_head.task, "interval")
        self.assertEqual(spec.task_head.outputs, ("lower", "upper"))
        self.assertEqual(spec.task_head.calibration_mode, "conformal")

    def test_unified_family_defaults_to_stagewise_plus_backend_plus_task_head(self) -> None:
        spec = build_unified_symbolic_family_spec(
            parameter_backend="torch",
            task="interval",
            calibration_mode="conformal",
            trainer_state_enabled=True,
            supports_resume=True,
        )
        self.assertEqual(spec.structure_engine.structure_mode, "stagewise_search")
        self.assertEqual(spec.structure_engine.search_driver, "nsgablack")
        self.assertEqual(spec.parameter_backend.backend, "torch")
        self.assertTrue(spec.parameter_backend.trainer_state_enabled)
        self.assertTrue(spec.parameter_backend.supports_resume)
        self.assertEqual(spec.task_head.task, "interval")
        self.assertEqual(spec.task_head.outputs, ("lower", "upper"))
        self.assertEqual(spec.task_head.calibration_mode, "conformal")
        description = dict(spec.description_dict())
        artifact_schema = dict(description.get("artifact_schema", {}))
        self.assertEqual(str(artifact_schema.get("schema_key")), "symbolic_artifact_v1")
        self.assertIn("complexity_metrics", tuple(artifact_schema.get("artifact_schema_fields", ())))
        self.assertIn("regime_structure", tuple(artifact_schema.get("artifact_schema_fields", ())))
        self.assertIn("basis_structure", tuple(artifact_schema.get("artifact_schema_fields", ())))
        self.assertIn("assembler_structure", tuple(artifact_schema.get("artifact_schema_fields", ())))
        self.assertIn("piecewise_gate_basis", tuple(artifact_schema.get("artifact_schema_fields", ())))
        self.assertIn("mode", tuple(artifact_schema.get("regime_fields", ())))
        self.assertIn("basis_scope", tuple(artifact_schema.get("basis_fields", ())))
        self.assertIn("assembler_mode", tuple(artifact_schema.get("assembler_fields", ())))
        self.assertIn("gate_feature_names", tuple(artifact_schema.get("piecewise_gate_fields", ())))
        self.assertTrue(bool(artifact_schema.get("supports_piecewise")))
        search_contracts = {
            str(row["mechanism_key"]): dict(row)
            for row in description.get("search_mechanism_contracts", ())
        }
        self.assertIn("beam_selection", search_contracts)
        self.assertIn("path_memory", search_contracts)
        self.assertTrue(bool(search_contracts["beam_selection"].get("checkpointable")))
        self.assertTrue(bool(search_contracts["beam_selection"].get("affects_family_signature")))
        self.assertFalse(bool(search_contracts["path_memory"].get("affects_family_signature")))
        structure_contracts = {
            str(key): dict(value)
            for key, value in dict(description.get("structure_contracts", {})).items()
        }
        self.assertIn("regime_discovery", structure_contracts)
        self.assertIn("basis_discovery", structure_contracts)
        self.assertIn("budgeted_symbolic_assembler", structure_contracts)
        self.assertEqual(str(structure_contracts["regime_discovery"].get("regime_mode")), "piecewise_gate")
        self.assertEqual(str(structure_contracts["basis_discovery"].get("basis_scope")), "global+local")
        basis_metadata = dict(structure_contracts["basis_discovery"].get("metadata", {}))
        self.assertIn("residual_complementarity", tuple(basis_metadata.get("orthogonality_objectives", ())))
        self.assertIn("semantic_deduplication", tuple(basis_metadata.get("orthogonality_objectives", ())))
        self.assertIn("piecewise_gate_basis", tuple(structure_contracts["basis_discovery"].get("produce", ())))
        self.assertEqual(
            str(structure_contracts["budgeted_symbolic_assembler"].get("assembler_mode")),
            "piecewise_budgeted_symbolic_regression",
        )
        self.assertIn("regime_discovery_contract", description)
        self.assertIn("basis_discovery_contract", description)
        self.assertIn("budgeted_symbolic_assembler_contract", description)

        signature_payload = dict(spec.family_signature_payload())
        self.assertIn("search_mechanism_contracts", signature_payload)
        self.assertIn("search_family_signature_contracts", signature_payload)
        self.assertIn("structure_contracts", signature_payload)
        signature_rows = {
            str(row["mechanism_key"]): dict(row)
            for row in signature_payload.get("search_family_signature_contracts", ())
        }
        self.assertIn("beam_selection", signature_rows)
        self.assertNotIn("path_memory", signature_rows)
        self.assertIn("regime_discovery", dict(signature_payload.get("structure_contracts", {})))

    def test_family_signature_hash_only_tracks_signature_affecting_search_contracts(self) -> None:
        spec = build_unified_symbolic_family_spec(parameter_backend="torch", task="point")
        baseline = spec.family_signature()
        self.assertIsNotNone(baseline)

        rows = tuple(build_symbolic_search_mechanism_contracts())
        changed_beam = tuple(
            SymbolicSearchMechanismContract(
                **(
                    {
                        **self._contract_kwargs(contract),
                        "consume": tuple(contract.consume) + ("new_focus_signal",),
                    }
                    if contract.mechanism_key == "beam_selection"
                    else self._contract_kwargs(contract)
                )
            )
            for contract in rows
        )
        with patch("core.symbolic.trainer_family.build_symbolic_search_mechanism_contracts", return_value=changed_beam):
            self.assertNotEqual(spec.family_signature(), baseline)

        changed_path_memory = tuple(
            SymbolicSearchMechanismContract(
                **(
                    {
                        **self._contract_kwargs(contract),
                        "consume": tuple(contract.consume) + ("new_memory_hint",),
                    }
                    if contract.mechanism_key == "path_memory"
                    else self._contract_kwargs(contract)
                )
            )
            for contract in rows
        )
        with patch("core.symbolic.trainer_family.build_symbolic_search_mechanism_contracts", return_value=changed_path_memory):
            self.assertEqual(spec.family_signature(), baseline)

    def test_family_signature_hash_tracks_structure_contract_drift(self) -> None:
        spec = build_unified_symbolic_family_spec(parameter_backend="torch", task="interval")
        baseline = spec.family_signature()
        self.assertIsNotNone(baseline)

        contract = spec.regime_discovery_contract()
        changed_contract = SymbolicRegimeDiscoveryContract(
            **{
                **self._structure_contract_kwargs(contract),
                "consume": tuple(contract.consume) + ("regime_prior_signal",),
            }
        )
        with patch("core.symbolic.trainer_family.build_symbolic_regime_discovery_contract", return_value=changed_contract):
            self.assertNotEqual(spec.family_signature(), baseline)

    def test_point_family_can_opt_into_piecewise_basis_contracts(self) -> None:
        spec = build_unified_symbolic_family_spec(
            parameter_backend="ridge",
            task="point",
            supports_piecewise_basis=True,
        )
        payload = dict(spec.description_dict())
        structure_contracts = {
            str(key): dict(value)
            for key, value in dict(payload.get("structure_contracts", {})).items()
        }
        self.assertEqual(str(structure_contracts["regime_discovery"].get("regime_mode")), "piecewise_gate")
        self.assertEqual(str(structure_contracts["basis_discovery"].get("basis_scope")), "global+local")
        self.assertEqual(
            str(structure_contracts["budgeted_symbolic_assembler"].get("assembler_mode")),
            "piecewise_budgeted_symbolic_regression",
        )

    def test_symbolic_surface_contract_marks_formal_and_legacy_entries(self) -> None:
        surface = symbolic_surface_contract()
        self.assertEqual(str(surface.get("formal_preset")), SYMBOLIC_FORMAL_PRESET_KEY)
        self.assertEqual(tuple(surface.get("legacy_facades", ())), SYMBOLIC_LEGACY_PRESET_KEYS)
        self.assertIn("symbolic_torch", tuple(surface.get("route_keys", ())))
        self.assertIn("symbolic_orthogonal", tuple(surface.get("route_keys", ())))
        self.assertEqual(len(tuple(surface.get("route_registry", ()))), 4)
        self.assertTrue(is_legacy_symbolic_preset("symbolic_torch"))
        self.assertFalse(is_legacy_symbolic_preset("symbolic"))
        self.assertEqual(canonical_symbolic_preset_key("symbolic_torch_interval"), "symbolic")
        self.assertEqual(canonical_symbolic_preset_key("symbolic"), "symbolic")

    def test_symbolic_route_registry_and_matcher_cover_supported_routes(self) -> None:
        routes = tuple(symbolic_route_registry())
        self.assertEqual(
            tuple(route.route_key for route in routes),
            ("symbolic_stagewise", "symbolic_orthogonal", "symbolic_torch", "symbolic_torch_interval"),
        )
        serialized = tuple(serialize_symbolic_route_registry())
        self.assertEqual(
            tuple(str(row.get("route_key")) for row in serialized),
            ("symbolic_stagewise", "symbolic_orthogonal", "symbolic_torch", "symbolic_torch_interval"),
        )

        stagewise = build_unified_symbolic_family_spec(parameter_backend="ridge", task="point")
        self.assertEqual(resolve_symbolic_router_target(stagewise), "symbolic_stagewise")
        self.assertEqual(resolve_symbolic_route_spec(stagewise).route_key, "symbolic_stagewise")
        self.assertEqual(tuple(route.route_key for route in match_symbolic_routes(stagewise)), ("symbolic_stagewise",))

        orthogonal = type(stagewise)(
            trainer_key=stagewise.trainer_key,
            structure_engine=type(stagewise.structure_engine)(
                structure_mode="orthogonal_basis_search",
                candidate_space=stagewise.structure_engine.candidate_space,
                grammar_source=stagewise.structure_engine.grammar_source,
                search_driver="orthogonal_basis",
                dynamic_pool_enabled=True,
                metadata={"supports_piecewise_basis": True},
            ),
            parameter_backend=stagewise.parameter_backend,
            task_head=stagewise.task_head,
            metadata=dict(stagewise.metadata),
        )
        self.assertEqual(resolve_symbolic_router_target(orthogonal), "symbolic_orthogonal")
        self.assertEqual(resolve_symbolic_route_spec(orthogonal).route_key, "symbolic_orthogonal")

        interval = build_unified_symbolic_family_spec(parameter_backend="torch", task="interval")
        self.assertEqual(resolve_symbolic_router_target(interval), "symbolic_torch_interval")
        self.assertEqual(resolve_symbolic_route_spec(interval).task, "interval")

    def test_symbolic_route_unsupported_error_is_explicit(self) -> None:
        spec = build_unified_symbolic_family_spec(parameter_backend="ridge", task="point")
        spec = type(spec)(
            trainer_key=spec.trainer_key,
            structure_engine=type(spec.structure_engine)(
                structure_mode="explicit_genome",
                candidate_space=spec.structure_engine.candidate_space,
                grammar_source=spec.structure_engine.grammar_source,
                search_driver=spec.structure_engine.search_driver,
                dynamic_pool_enabled=spec.structure_engine.dynamic_pool_enabled,
                metadata=spec.structure_engine.metadata,
            ),
            parameter_backend=spec.parameter_backend,
            task_head=spec.task_head,
            metadata=spec.metadata,
        )
        with self.assertRaisesRegex(ValueError, "failed structure-mode constraints"):
            resolve_symbolic_router_target(spec)

    def test_symbolic_route_conflict_error_is_explicit(self) -> None:
        spec = build_unified_symbolic_family_spec(parameter_backend="torch", task="point")
        conflict_routes = (
            SymbolicRouteSpec(
                route_key="symbolic_torch",
                parameter_backend="torch",
                task="point",
                status="stable",
            ),
            SymbolicRouteSpec(
                route_key="symbolic_torch_v2",
                parameter_backend="torch",
                task="point",
                status="stable",
            ),
        )
        with patch("core.symbolic.trainer_family.symbolic_route_registry", return_value=conflict_routes):
            with self.assertRaisesRegex(ValueError, "route conflict"):
                resolve_symbolic_router_target(spec)


if __name__ == "__main__":
    unittest.main()
