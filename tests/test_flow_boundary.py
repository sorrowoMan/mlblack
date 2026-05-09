from __future__ import annotations

import unittest

from config import CapabilitySpec, FlowAssemblySpec, NumericizerSpec, TrainerAssemblySpec, build_flow_components, validate_flow_assembly
from config.defaults import create_default_config


class TestFlowBoundary(unittest.TestCase):
    def test_reject_numericizer_keys_in_trainer_params(self) -> None:
        spec = FlowAssemblySpec(
            trainer=TrainerAssemblySpec(
                trainer_key="ridge",
                trainer_params={
                    "target_codec": "numeric",
                    "l2": 1.0,
                },
            ),
            numericizer=NumericizerSpec(key="default", params={}),
        )
        with self.assertRaises(ValueError):
            validate_flow_assembly(spec)

    def test_accept_clean_trainer_params(self) -> None:
        spec = FlowAssemblySpec(
            trainer=TrainerAssemblySpec(
                trainer_key="ridge",
                trainer_params={
                    "l2": 1.0,
                },
            ),
            numericizer=NumericizerSpec(key="default", params={}),
        )
        validate_flow_assembly(spec)

    def test_reject_formal_family_without_route_registry_metadata(self) -> None:
        cfg = create_default_config()
        factory = cfg.trainers.get("tree_ensemble")
        assert factory is not None
        cfg.trainers.register(
            "tree_ensemble",
            factory,
            replace=True,
            metadata={
                "name": "tree_ensemble",
                "family": "tree_ensemble",
                "backend": "family_router",
                "surface_status": "formal",
                "route_family": "tree_ensemble",
                "route_registry": (),
            },
        )
        spec = FlowAssemblySpec(
            trainer=TrainerAssemblySpec(
                trainer_key="tree_ensemble",
                trainer_params={},
            ),
            numericizer=NumericizerSpec(key="default", params={}),
        )
        with self.assertRaises(ValueError):
            validate_flow_assembly(spec, config=cfg)

    def test_reject_capability_without_context_contract(self) -> None:
        class BrokenCapability:
            name = "broken_capability"

        cfg = create_default_config()
        cfg.capabilities.register("broken_capability", lambda **_: BrokenCapability())
        spec = FlowAssemblySpec(
            trainer=TrainerAssemblySpec(
                trainer_key="ridge",
                trainer_params={"l2": 1.0},
            ),
            numericizer=NumericizerSpec(key="default", params={}),
            capabilities=(
                CapabilitySpec(key="broken_capability", params={}),
            ),
        )

        with self.assertRaises(ValueError):
            build_flow_components(spec, config=cfg)


if __name__ == "__main__":
    unittest.main()
