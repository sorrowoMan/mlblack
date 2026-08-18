from __future__ import annotations

from mlblack.core.trainer import BlankTrainer


def test_trainer_generic_snapshot_roundtrip_returns_original_payload() -> None:
    trainer = BlankTrainer(run_name="snapshot-test")
    payload = {"weights": [1.0, 2.0], "metadata": {"kind": "model"}}

    key = trainer.write_snapshot(payload, key="model")

    assert trainer.read_snapshot(key) == payload
