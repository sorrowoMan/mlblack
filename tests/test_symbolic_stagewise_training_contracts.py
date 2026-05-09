from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import TrainerAssemblySpec, build_trainer
from core.common.contracts import ProcessedDataset
from training import InnerRuntimeErrorPayload, InnerRuntimeFinishPayload, InnerRuntimeRoundPayload, InnerRuntimeStartPayload
from training import TrainTask, TrainingInit


def _make_processed_dataset(seed: int = 2026) -> ProcessedDataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(48, 4))
    y = (1.2 * x[:, 0] - 0.5 * x[:, 1] + 0.25 * np.sin(x[:, 2]) + 0.1 * x[:, 3] ** 2).reshape(-1, 1)
    return ProcessedDataset(
        X_train=x,
        y_train=y,
        feature_names=("x0", "x1", "x2", "x3"),
        target_names=("y",),
        metadata={"source": "symbolic_stagewise_training_contract_test"},
    )


class TestSymbolicStagewiseTrainingContracts(unittest.TestCase):
    def _build_stagewise_trainer(self, *, max_added_terms: int = 1):
        return build_trainer(
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
                    "force_linear_base": "on",
                    "keep_search_trace": True,
                    "search_max_added_terms": max_added_terms,
                    "search_topk_features": 3,
                    "search_max_pair_terms": 4,
                    "search_max_candidates_per_iter": 64,
                    "search_candidate_keep_top": 6,
                    "search_online_beam_enabled": False,
                    "search_path_memory_enabled": False,
                    "search_graph_cache_enabled": False,
                    "search_joint_bundle_enabled": False,
                    "search_include_hinge": False,
                    "search_unary_ops": ("square", "sin"),
                },
            )
        )

    def test_fit_task_emits_stagewise_trainer_state_and_signature(self) -> None:
        trainer = self._build_stagewise_trainer(max_added_terms=1)
        result = trainer.fit_task(
            TrainTask.from_data(_make_processed_dataset(), task_id="symbolic_stagewise::fresh"),
            TrainingInit(mode="fresh"),
        )

        self.assertIsNotNone(result.trainer_state)
        self.assertEqual(str(result.report["training_mode"]), "fresh")
        self.assertEqual(str(result.artifact.metadata.get("training_init", {}).get("mode")), "fresh")
        self.assertIn("training_signature", result.artifact.metadata)
        self.assertIn("symbolic_family_signature", result.artifact.metadata)
        self.assertIsNotNone(result.trainer_state.symbolic_family_signature)

    def test_stagewise_state_roundtrip_resume_and_seeded_modes(self) -> None:
        data = _make_processed_dataset(seed=2027)
        parent_trainer = self._build_stagewise_trainer(max_added_terms=1)
        parent_result = parent_trainer.fit_task(
            TrainTask.from_data(data, task_id="symbolic_stagewise::parent"),
            TrainingInit(mode="fresh"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "stagewise_state.pt"
            parent_trainer.save_trainer_state(state_path, parent_result.trainer_state)
            loaded_state = parent_trainer.load_trainer_state(state_path)

            resume_trainer = self._build_stagewise_trainer(max_added_terms=2)
            resume_result = resume_trainer.fit_task(
                TrainTask.from_data(data, task_id="symbolic_stagewise::resume"),
                TrainingInit(mode="resume", parent_state=loaded_state),
            )
            self.assertIsNotNone(resume_result.trainer_state)
            self.assertEqual(str(resume_result.report["training_mode"]), "resume")
            self.assertTrue(bool(resume_result.artifact.metadata.get("resume", {}).get("enabled")))
            self.assertEqual(str(resume_result.artifact.metadata.get("training_init", {}).get("parent_kind")), "trainer_state")

        warm_trainer = self._build_stagewise_trainer(max_added_terms=1)
        warm_result = warm_trainer.fit_task(
            TrainTask.from_data(data, task_id="symbolic_stagewise::warm_start"),
            TrainingInit(mode="warm_start", parent_artifact=parent_result.artifact),
        )
        self.assertIsNotNone(warm_result.trainer_state)
        self.assertEqual(str(warm_result.report["training_mode"]), "warm_start")
        self.assertEqual(str(warm_result.artifact.metadata.get("training_init", {}).get("parent_kind")), "artifact")

        inc_trainer = self._build_stagewise_trainer(max_added_terms=2)
        inc_result = inc_trainer.fit_task(
            TrainTask.from_data(data, task_id="symbolic_stagewise::incremental"),
            TrainingInit(mode="incremental", parent_state=parent_result.trainer_state),
        )
        self.assertIsNotNone(inc_result.trainer_state)
        self.assertEqual(str(inc_result.report["training_mode"]), "incremental")
        self.assertTrue(bool(inc_result.artifact.metadata.get("resume", {}).get("enabled")))

    def test_stagewise_inner_runtime_hooks_receive_search_events(self) -> None:
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
        trainer = self._build_stagewise_trainer(max_added_terms=1)
        result = trainer.fit_task(
            TrainTask.from_data(_make_processed_dataset(seed=2030), task_id="symbolic_stagewise::hooked"),
            TrainingInit(mode="fresh", inner_runtime_hooks=(hook,)),
        )

        self.assertIsNotNone(result.trainer_state)
        event_names = [name for name, _ in hook.events]
        self.assertIn("start", event_names)
        self.assertIn("round", event_names)
        self.assertIn("finish", event_names)
        self.assertNotIn("error", event_names)

        start_payload = next(payload for name, payload in hook.events if name == "start")
        finish_payload = next(payload for name, payload in hook.events if name == "finish")
        self.assertEqual(str(start_payload.runtime_key), "symbolic_structure_search")
        self.assertEqual(str(start_payload.trainer_name), "symbolic_stagewise")
        self.assertGreaterEqual(int(finish_payload.completed_rounds), 1)
        self.assertEqual(str(finish_payload.context.get("task_id")), "symbolic_stagewise::hooked")


if __name__ == "__main__":
    unittest.main()
