from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from config import TrainerAssemblySpec, build_trainer
from core.common.contracts import ProcessedDataset
from core.symbolic import (
    SymbolicRegimeDiscoveryContract,
    SymbolicSearchMechanismContract,
    build_symbolic_search_mechanism_contracts,
)
from training import TrainTask, TrainingCompatibilityError, TrainingInit
from workflow import TrainFlowSpec, run_train_flow


def _make_processed_dataset(seed: int = 123) -> ProcessedDataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(48, 4))
    y = (1.3 * x[:, 0] - 0.6 * x[:, 1] + 0.2 * np.sin(x[:, 2])).reshape(-1, 1)
    return ProcessedDataset(
        X_train=x,
        y_train=y,
        feature_names=("x0", "x1", "x2", "x3"),
        target_names=("y",),
        metadata={"source": "symbolic_training_contract_test"},
    )


class TestSymbolicTorchTrainingContracts(unittest.TestCase):
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

    def _build_symbolic_point_trainer(self, *, structure_mode: str, epochs: int = 2):
        return build_trainer(
            TrainerAssemblySpec(
                trainer_key="symbolic",
                trainer_params={
                    "parameter_backend": "torch",
                    "task": "point",
                    "structure_engine": {
                        "structure_mode": structure_mode,
                        "search_driver": "nsgablack" if structure_mode == "stagewise_search" else "local_seed_builder",
                        "dynamic_pool_enabled": structure_mode == "stagewise_search",
                    },
                    "device": "cpu",
                    "epochs": epochs,
                    "batch_size": 8,
                    "val_ratio": 0.2,
                    "early_stop_patience": 8,
                    "random_seed": 42,
                },
            )
        )

    def test_fit_task_emits_trainer_state_and_signature(self) -> None:
        trainer = self._build_symbolic_point_trainer(structure_mode="seed_library", epochs=2)
        result = trainer.fit_task(
            TrainTask.from_data(_make_processed_dataset(), task_id="symbolic::fresh"),
            TrainingInit(mode="fresh"),
        )

        self.assertIsNotNone(result.trainer_state)
        self.assertEqual(str(result.report["training_mode"]), "fresh")
        self.assertIn("training_signature", result.artifact.metadata)
        self.assertIn("symbolic_family_signature", result.artifact.metadata)
        self.assertIsNotNone(result.trainer_state.symbolic_family_signature)
        training_signature = dict(result.trainer_state.metadata.get("training_signature", {}))
        symbolic_family = dict(training_signature.get("metadata", {}).get("symbolic_family", {}))
        self.assertIn("search_mechanism_contracts", symbolic_family)
        self.assertIn("search_family_signature_contracts", symbolic_family)

    def test_resume_and_warm_start_reuse_symbolic_family_chain(self) -> None:
        data = _make_processed_dataset(seed=321)
        parent_trainer = self._build_symbolic_point_trainer(structure_mode="seed_library", epochs=2)
        parent_result = parent_trainer.fit_task(
            TrainTask.from_data(data, task_id="symbolic::parent"),
            TrainingInit(mode="fresh"),
        )

        resume_trainer = self._build_symbolic_point_trainer(structure_mode="seed_library", epochs=4)
        resume_result = resume_trainer.fit_task(
            TrainTask.from_data(data, task_id="symbolic::resume"),
            TrainingInit(mode="resume", parent_state=parent_result.trainer_state),
        )

        self.assertIsNotNone(resume_result.trainer_state)
        self.assertEqual(str(resume_result.report["training_mode"]), "resume")
        self.assertTrue(bool(resume_result.artifact.metadata.get("resume", {}).get("enabled")))

        warm_trainer = self._build_symbolic_point_trainer(structure_mode="seed_library", epochs=2)
        warm_result = warm_trainer.fit_task(
            TrainTask.from_data(data, task_id="symbolic::warm_start"),
            TrainingInit(mode="warm_start", parent_artifact=parent_result.artifact),
        )

        self.assertIsNotNone(warm_result.trainer_state)
        self.assertEqual(str(warm_result.report["training_mode"]), "warm_start")
        self.assertEqual(
            str(warm_result.artifact.metadata.get("training_init", {}).get("mode")),
            "warm_start",
        )

    def test_warm_start_rejects_symbolic_family_mismatch(self) -> None:
        data = _make_processed_dataset(seed=456)
        parent_trainer = self._build_symbolic_point_trainer(structure_mode="seed_library", epochs=2)
        parent_result = parent_trainer.fit_task(
            TrainTask.from_data(data, task_id="symbolic::parent_seed"),
            TrainingInit(mode="fresh"),
        )

        child_trainer = self._build_symbolic_point_trainer(structure_mode="stagewise_search", epochs=2)
        with self.assertRaises(TrainingCompatibilityError) as ctx:
            child_trainer.fit_task(
                TrainTask.from_data(data, task_id="symbolic::child_stagewise"),
                TrainingInit(mode="warm_start", parent_artifact=parent_result.artifact),
            )

        self.assertIn("symbolic_family_signature", str(ctx.exception))

    def test_warm_start_rejects_signature_affecting_search_mechanism_drift(self) -> None:
        data = _make_processed_dataset(seed=777)
        parent_trainer = self._build_symbolic_point_trainer(structure_mode="seed_library", epochs=2)
        parent_result = parent_trainer.fit_task(
            TrainTask.from_data(data, task_id="symbolic::parent_signature_contract"),
            TrainingInit(mode="fresh"),
        )

        child_trainer = self._build_symbolic_point_trainer(structure_mode="seed_library", epochs=2)
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
            with self.assertRaises(TrainingCompatibilityError) as ctx:
                child_trainer.fit_task(
                    TrainTask.from_data(data, task_id="symbolic::child_signature_contract"),
                    TrainingInit(mode="warm_start", parent_artifact=parent_result.artifact),
                )

        self.assertIn("symbolic_family_signature", str(ctx.exception))
        self.assertIn("beam_selection", str(ctx.exception))
        self.assertIn("consume", str(ctx.exception))
        drift = dict(ctx.exception.metadata).get("symbolic_family_signature_drift", {})
        self.assertIn("parent_artifact", drift)
        changed = tuple(dict(drift["parent_artifact"]).get("changed_mechanisms", ()))
        self.assertTrue(any(str(dict(row).get("mechanism_key")) == "beam_selection" for row in changed))

    def test_warm_start_rejects_structure_contract_drift(self) -> None:
        data = _make_processed_dataset(seed=779)
        parent_trainer = self._build_symbolic_point_trainer(structure_mode="seed_library", epochs=2)
        parent_result = parent_trainer.fit_task(
            TrainTask.from_data(data, task_id="symbolic::parent_structure_contract"),
            TrainingInit(mode="fresh"),
        )

        child_trainer = self._build_symbolic_point_trainer(structure_mode="seed_library", epochs=2)
        changed_contract = SymbolicRegimeDiscoveryContract(
            **{
                **self._structure_contract_kwargs(child_trainer.symbolic_family_spec.regime_discovery_contract()),
                "consume": tuple(child_trainer.symbolic_family_spec.regime_discovery_contract().consume) + ("regime_prior_signal",),
            }
        )
        with patch("core.symbolic.trainer_family.build_symbolic_regime_discovery_contract", return_value=changed_contract):
            with self.assertRaises(TrainingCompatibilityError) as ctx:
                child_trainer.fit_task(
                    TrainTask.from_data(data, task_id="symbolic::child_structure_contract"),
                    TrainingInit(mode="warm_start", parent_artifact=parent_result.artifact),
                )

        self.assertIn("symbolic_family_signature", str(ctx.exception))
        self.assertIn("regime_discovery", str(ctx.exception))
        drift = dict(ctx.exception.metadata).get("symbolic_family_signature_drift", {})
        self.assertIn("parent_artifact", drift)
        changed = tuple(dict(drift["parent_artifact"]).get("changed_structure_contracts", ()))
        self.assertTrue(any(str(dict(row).get("contract_key")) == "regime_discovery" for row in changed))

    def test_failed_flow_report_persists_symbolic_contract_drift(self) -> None:
        data = _make_processed_dataset(seed=778)
        parent_trainer = self._build_symbolic_point_trainer(structure_mode="seed_library", epochs=2)
        parent_result = parent_trainer.fit_task(
            TrainTask.from_data(data, task_id="symbolic::parent_failure_report"),
            TrainingInit(mode="fresh"),
        )
        rows = tuple(build_symbolic_search_mechanism_contracts())
        changed_beam = tuple(
            SymbolicSearchMechanismContract(
                **(
                    {
                        **self._contract_kwargs(contract),
                        "consume": tuple(contract.consume) + ("fresh_signal",),
                    }
                    if contract.mechanism_key == "beam_selection"
                    else self._contract_kwargs(contract)
                )
            )
            for contract in rows
        )

        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "failed_symbolic_contract_drift"
            spec = TrainFlowSpec(
                assembly=TrainerAssemblySpec(
                    trainer_key="symbolic",
                    trainer_params={
                        "parameter_backend": "torch",
                        "task": "point",
                        "structure_engine": {
                            "structure_mode": "seed_library",
                            "search_driver": "local_seed_builder",
                            "dynamic_pool_enabled": False,
                        },
                        "device": "cpu",
                        "epochs": 2,
                        "batch_size": 8,
                        "val_ratio": 0.2,
                        "early_stop_patience": 8,
                        "random_seed": 42,
                    },
                ),
                training_init=TrainingInit(mode="warm_start", parent_artifact=parent_result.artifact),
                save_artifact=False,
                save_report=True,
                output_dir=str(report_dir),
                run_name="symbolic_failed_contract_drift",
            )
            with patch("core.symbolic.trainer_family.build_symbolic_search_mechanism_contracts", return_value=changed_beam):
                with self.assertRaises(TrainingCompatibilityError):
                    run_train_flow(data, spec=spec)

            payload = json.loads((report_dir / "flow_report.json").read_text(encoding="utf-8"))
            self.assertEqual(str(payload.get("status")), "failed")
            training = dict(payload.get("training", {}))
            drift = dict(training.get("symbolic_family_signature_drift", {}))
            self.assertIn("parent_artifact", drift)
            changed = tuple(dict(drift["parent_artifact"]).get("changed_mechanisms", ()))
            self.assertTrue(any(str(dict(row).get("mechanism_key")) == "beam_selection" for row in changed))
            self.assertIn("beam_selection", json.dumps(drift, ensure_ascii=False))

    def test_train_flow_report_surfaces_symbolic_artifact_schema(self) -> None:
        spec = TrainFlowSpec(
            assembly=TrainerAssemblySpec(
                trainer_key="symbolic",
                trainer_params={
                    "parameter_backend": "torch",
                    "task": "point",
                    "structure_engine": {
                        "structure_mode": "seed_library",
                        "search_driver": "local_seed_builder",
                        "dynamic_pool_enabled": False,
                    },
                    "device": "cpu",
                    "epochs": 2,
                    "batch_size": 8,
                    "val_ratio": 0.2,
                    "early_stop_patience": 8,
                    "random_seed": 42,
                },
            ),
            save_artifact=False,
            save_report=False,
            run_name="symbolic_schema_report",
        )

        result = run_train_flow(_make_processed_dataset(seed=654), spec=spec)
        artifact_block = dict(result.report.get("artifact", {}))
        schema = dict(artifact_block.get("symbolic_artifact_schema", {}))
        training_block = dict(result.report.get("training", {}))
        symbolic_family = dict(training_block.get("symbolic_family", {}))

        self.assertEqual(str(dict(schema.get("head_semantics", {})).get("task")), "point")
        self.assertIn("complexity_metrics", schema)
        self.assertGreater(int(dict(artifact_block.get("symbolic_complexity_metrics", {})).get("term_count", 0)), 0)
        self.assertIn("structure_contracts", symbolic_family)
        self.assertIn("regime_discovery", dict(symbolic_family.get("structure_contracts", {})))


if __name__ == "__main__":
    unittest.main()
