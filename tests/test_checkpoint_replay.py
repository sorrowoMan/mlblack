from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import TrainerAssemblySpec
from core.common.contracts import ProcessedDataset
from workflow import TrainFlowSpec, run_train_flow


class TestCheckpointReplay(unittest.TestCase):
    def test_save_and_replay_checkpoint(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.normal(size=(64, 5))
        y = (1.2 * X[:, 0] - 0.7 * X[:, 1] + 0.1).reshape(-1, 1)
        data = ProcessedDataset(
            X_train=X,
            y_train=y,
            feature_names=tuple(f"x{i}" for i in range(5)),
            target_names=("y",),
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cp_dir = tmp_path / "checkpoint_case"

            fit_spec = TrainFlowSpec(
                assembly=TrainerAssemblySpec(
                    trainer_key="ridge",
                    trainer_params={"l2": 0.0},
                ),
                eval_splits=("train",),
                save_artifact=False,
                save_report=False,
                save_checkpoint=True,
                checkpoint_dir=str(cp_dir),
                run_name="checkpoint_case",
            )
            fit_res = run_train_flow(data, spec=fit_spec)
            self.assertTrue((cp_dir / "manifest.json").exists())

            replay_spec = TrainFlowSpec(
                assembly=TrainerAssemblySpec(trainer_key="ridge"),
                replay_from_checkpoint=str(cp_dir),
                run_name="checkpoint_case_replay",
            )
            replay_res = run_train_flow(data, spec=replay_spec)

            y_fit = np.asarray(fit_res.artifact.predict(X), dtype=float)
            y_replay = np.asarray(replay_res.artifact.predict(X), dtype=float)
            np.testing.assert_allclose(y_fit, y_replay, rtol=1e-10, atol=1e-10)
            self.assertEqual(set(replay_res.metrics.keys()), {"train"})


if __name__ == "__main__":
    unittest.main()
