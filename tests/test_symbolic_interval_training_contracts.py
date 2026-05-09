from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from config import TrainerAssemblySpec, build_trainer
from core.common.contracts import ProcessedDataset
from plugins.trainer_state_checkpoint_plugin import TrainerStateCheckpointPlugin
from training import (
    InnerRuntimeErrorPayload,
    InnerRuntimeFinishPayload,
    InnerRuntimeRoundPayload,
    InnerRuntimeStartPayload,
    TrainTask,
    TrainingCompatibilityError,
    TrainingInit,
)
from workflow import TrainFlowSpec, run_train_flow


def _make_processed_dataset(seed: int = 777) -> ProcessedDataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(56, 4))
    y = (1.1 * x[:, 0] - 0.4 * x[:, 1] + 0.3 * np.cos(x[:, 2])).reshape(-1, 1)
    return ProcessedDataset(
        X_train=x,
        y_train=y,
        feature_names=("x0", "x1", "x2", "x3"),
        target_names=("y",),
        metadata={"source": "symbolic_interval_training_contract_test"},
    )


def _make_piecewise_processed_dataset(seed: int = 1701) -> ProcessedDataset:
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for g0, g1, offset in ((0.0, 0.0, -0.4), (0.0, 1.0, 0.2), (1.0, 0.0, 0.8), (1.0, 1.0, 1.4)):
        x_rest = rng.normal(size=(24, 3))
        gate = np.column_stack(
            [
                np.full((24,), g0, dtype=float),
                np.full((24,), g1, dtype=float),
            ]
        )
        x = np.concatenate([gate, x_rest], axis=1)
        y = (0.7 * x[:, 2] - 0.25 * x[:, 3] + 0.15 * np.sin(x[:, 4]) + offset).reshape(-1, 1)
        blocks.append(x)
        targets.append(y)
    X = np.concatenate(blocks, axis=0)
    Y = np.concatenate(targets, axis=0)
    return ProcessedDataset(
        X_train=X,
        y_train=Y,
        feature_names=("gate_feature_0", "gate_feature_1", "x0", "x1", "x2"),
        target_names=("y",),
        metadata={"source": "symbolic_interval_piecewise_contract_test"},
    )


class TestSymbolicIntervalTrainingContracts(unittest.TestCase):
    def _build_symbolic_interval_trainer(
        self,
        *,
        structure_mode: str,
        epochs: int = 2,
        checkpoint_dir: str | None = None,
        extra_params: dict[str, object] | None = None,
    ):
        params = {
            "parameter_backend": "torch",
            "task": "interval",
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
            "conformal_calibration": False,
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_every_epochs": 1 if checkpoint_dir is not None else 0,
        }
        if extra_params:
            params.update(dict(extra_params))
        return build_trainer(
            TrainerAssemblySpec(
                trainer_key="symbolic",
                trainer_params=params,
            )
        )

    def test_fit_task_emits_interval_trainer_state_and_signature(self) -> None:
        trainer = self._build_symbolic_interval_trainer(structure_mode="seed_library", epochs=2)
        result = trainer.fit_task(
            TrainTask.from_data(_make_processed_dataset(), task_id="symbolic_interval::fresh"),
            TrainingInit(mode="fresh"),
        )

        self.assertIsNotNone(result.trainer_state)
        self.assertEqual(str(result.report["training_mode"]), "fresh")
        self.assertIn("training_signature", result.artifact.metadata)
        self.assertIn("symbolic_family_signature", result.artifact.metadata)
        self.assertIsNotNone(result.trainer_state.symbolic_family_signature)

    def test_resume_and_warm_start_reuse_interval_family_chain(self) -> None:
        data = _make_processed_dataset(seed=888)
        parent_trainer = self._build_symbolic_interval_trainer(structure_mode="seed_library", epochs=2)
        parent_result = parent_trainer.fit_task(
            TrainTask.from_data(data, task_id="symbolic_interval::parent"),
            TrainingInit(mode="fresh"),
        )

        resume_trainer = self._build_symbolic_interval_trainer(structure_mode="seed_library", epochs=4)
        resume_result = resume_trainer.fit_task(
            TrainTask.from_data(data, task_id="symbolic_interval::resume"),
            TrainingInit(mode="resume", parent_state=parent_result.trainer_state),
        )

        self.assertIsNotNone(resume_result.trainer_state)
        self.assertEqual(str(resume_result.report["training_mode"]), "resume")
        self.assertTrue(bool(resume_result.artifact.metadata.get("resume", {}).get("enabled")))

        warm_trainer = self._build_symbolic_interval_trainer(structure_mode="seed_library", epochs=2)
        warm_result = warm_trainer.fit_task(
            TrainTask.from_data(data, task_id="symbolic_interval::warm_start"),
            TrainingInit(mode="warm_start", parent_artifact=parent_result.artifact),
        )

        self.assertIsNotNone(warm_result.trainer_state)
        self.assertEqual(str(warm_result.report["training_mode"]), "warm_start")

    def test_warm_start_rejects_interval_family_mismatch(self) -> None:
        data = _make_processed_dataset(seed=999)
        parent_trainer = self._build_symbolic_interval_trainer(structure_mode="seed_library", epochs=2)
        parent_result = parent_trainer.fit_task(
            TrainTask.from_data(data, task_id="symbolic_interval::parent_seed"),
            TrainingInit(mode="fresh"),
        )

        child_trainer = self._build_symbolic_interval_trainer(structure_mode="stagewise_search", epochs=2)
        with self.assertRaises(TrainingCompatibilityError) as ctx:
            child_trainer.fit_task(
                TrainTask.from_data(data, task_id="symbolic_interval::child_stagewise"),
                TrainingInit(mode="warm_start", parent_artifact=parent_result.artifact),
            )

        self.assertIn("symbolic_family_signature", str(ctx.exception))

    def test_interval_checkpoint_files_and_state_loading_roundtrip(self) -> None:
        data = _make_processed_dataset(seed=1010)
        with tempfile.TemporaryDirectory() as tmp:
            ck_dir = Path(tmp) / "symbolic_interval_ckpt"
            trainer = self._build_symbolic_interval_trainer(
                structure_mode="seed_library",
                epochs=2,
                checkpoint_dir=str(ck_dir),
            )
            parent_result = trainer.fit_task(
                TrainTask.from_data(data, task_id="symbolic_interval::checkpoint_parent"),
                TrainingInit(mode="fresh"),
            )

            global_latest = ck_dir / "global" / "latest.pt"
            trainer_state_latest = ck_dir / "trainer_state" / "latest.pt"
            self.assertTrue(global_latest.exists())
            self.assertTrue(trainer_state_latest.exists())

            loaded_state = trainer.load_trainer_state(trainer_state_latest)
            child_trainer = self._build_symbolic_interval_trainer(structure_mode="seed_library", epochs=4)
            resumed = child_trainer.fit_task(
                TrainTask.from_data(data, task_id="symbolic_interval::checkpoint_resume"),
                TrainingInit(mode="resume", parent_state=loaded_state),
            )

            self.assertIsNotNone(resumed.trainer_state)
            self.assertEqual(str(resumed.report["training_mode"]), "resume")
            self.assertTrue(bool(parent_result.trainer_state is not None))

    def test_piecewise_interval_state_contains_explicit_aggregate_manifest(self) -> None:
        data = _make_piecewise_processed_dataset()
        with tempfile.TemporaryDirectory() as tmp:
            ck_dir = Path(tmp) / "symbolic_interval_piecewise_ckpt"
            trainer = self._build_symbolic_interval_trainer(
                structure_mode="seed_library",
                epochs=2,
                checkpoint_dir=str(ck_dir),
                extra_params={
                    "gate_piecewise_enabled": True,
                    "gate_feature_names": ("gate_feature_0", "gate_feature_1"),
                    "gate_min_leaf": 12,
                    "gate_max_local_models": 4,
                    "gate_blend_kappa": 32.0,
                },
            )
            result = trainer.fit_task(
                TrainTask.from_data(data, task_id="symbolic_interval::piecewise_manifest"),
                TrainingInit(mode="fresh"),
            )

            manifest = dict(result.trainer_state.payload.get("aggregate_manifest", {}))
            manifest_path = ck_dir / "aggregate_manifest.json"

            self.assertEqual(str(result.trainer_state.payload.get("mode")), "piecewise")
            self.assertTrue(bool(manifest))
            self.assertTrue(bool(manifest.get("selected_regime_keys")))
            self.assertIn("local_regimes", manifest)
            self.assertTrue(manifest_path.exists())

            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(str(loaded_manifest.get("mode")), "piecewise")
            self.assertTrue(bool(loaded_manifest.get("selected_regime_keys")))
            self.assertIn("global_checkpoint", loaded_manifest)
            self.assertIn("local_regimes", loaded_manifest)

    def test_interval_inner_runtime_hooks_cover_core_and_piecewise_loops(self) -> None:
        class CollectingHook:
            def __init__(self) -> None:
                self.events: list[tuple[str, object]] = []

            def on_inner_run_start(self, payload: InnerRuntimeStartPayload) -> None:
                self.events.append(("start", payload))

            def on_inner_round_end(self, payload: InnerRuntimeRoundPayload) -> None:
                self.events.append(("round", payload))

            def on_inner_run_finish(self, payload: InnerRuntimeFinishPayload) -> None:
                self.events.append(("finish", payload))

            def on_inner_run_error(self, payload: InnerRuntimeErrorPayload) -> None:
                self.events.append(("error", payload))

        hook = CollectingHook()
        trainer = self._build_symbolic_interval_trainer(
            structure_mode="seed_library",
            epochs=2,
            extra_params={
                "gate_piecewise_enabled": True,
                "gate_feature_names": ("gate_feature_0", "gate_feature_1"),
                "gate_min_leaf": 12,
                "gate_max_local_models": 4,
                "gate_blend_kappa": 32.0,
            },
        )
        result = trainer.fit_task(
            TrainTask.from_data(_make_piecewise_processed_dataset(seed=1702), task_id="symbolic_interval::hooked"),
            TrainingInit(mode="fresh", inner_runtime_hooks=(hook,)),
        )

        self.assertIsNotNone(result.trainer_state)
        event_names = [name for name, _ in hook.events]
        runtime_keys = {
            str(getattr(payload, "runtime_key", ""))
            for _, payload in hook.events
        }
        self.assertIn("start", event_names)
        self.assertIn("round", event_names)
        self.assertIn("finish", event_names)
        self.assertNotIn("error", event_names)
        self.assertIn("symbolic_interval_core", runtime_keys)
        self.assertIn("symbolic_interval_piecewise", runtime_keys)

        piecewise_finish = next(
            payload
            for name, payload in hook.events
            if name == "finish" and str(getattr(payload, "runtime_key", "")) == "symbolic_interval_piecewise"
        )
        self.assertGreaterEqual(int(piecewise_finish.completed_rounds), 1)
        self.assertEqual(str(piecewise_finish.context.get("task_id")), "symbolic_interval::hooked")

    def test_workflow_plugin_persists_trainer_state_checkpoint(self) -> None:
        data = _make_processed_dataset(seed=1111)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "flow_output"
            spec = TrainFlowSpec(
                assembly=TrainerAssemblySpec(
                    trainer_key="symbolic",
                    trainer_params={
                        "parameter_backend": "torch",
                        "task": "interval",
                        "structure_engine": {"structure_mode": "seed_library"},
                        "device": "cpu",
                        "epochs": 2,
                        "batch_size": 8,
                        "val_ratio": 0.2,
                        "early_stop_patience": 8,
                        "random_seed": 42,
                        "conformal_calibration": False,
                    },
                ),
                output_dir=str(out_dir),
                capabilities=(TrainerStateCheckpointPlugin(relpath="custom_state/latest.pt"),),
                save_artifact=False,
                save_report=False,
                eval_splits=("train",),
                run_name="symbolic_interval_plugin_checkpoint",
            )
            result = run_train_flow(data, spec=spec)

            expected_path = out_dir / "custom_state" / "latest.pt"
            self.assertTrue(expected_path.exists())
            self.assertEqual(
                str(result.report.get("training", {}).get("trainer_state_checkpoint")),
                str(expected_path.resolve()),
            )


if __name__ == "__main__":
    unittest.main()
