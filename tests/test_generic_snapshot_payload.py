from __future__ import annotations

from mlblack.core.trainer import BlankTrainer
from mlblack.core.trainer_stage import ArtifactRef


def test_trainer_generic_snapshot_roundtrip_returns_original_payload() -> None:
    trainer = BlankTrainer(run_name="snapshot-test")
    payload = {"weights": [1.0, 2.0], "metadata": {"kind": "model"}}

    key = trainer.write_snapshot(payload, key="model")

    assert trainer.read_snapshot(key) == payload
    assert ArtifactRef(key="model", uri=key).resolve(trainer.snapshot_store) == payload


def test_trainer_reads_legacy_key_wrapped_snapshot() -> None:
    trainer = BlankTrainer(run_name="snapshot-test")
    payload = {"legacy": True}
    trainer.snapshot_store.write({"legacy-key": payload}, key="legacy-key")

    assert trainer.read_snapshot("legacy-key") == payload
    assert ArtifactRef(key="legacy-key", uri="legacy-key").resolve(trainer.snapshot_store) == payload
