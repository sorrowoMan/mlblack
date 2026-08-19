from __future__ import annotations

from blackbase.context import InMemorySnapshotStore, unwrap_snapshot_payload, wrap_snapshot_payload


def test_generic_snapshot_roundtrip_belongs_to_shared_store() -> None:
    store = InMemorySnapshotStore()
    payload = {"weights": [1.0, 2.0], "metadata": {"kind": "model"}}
    handle = store.write(wrap_snapshot_payload(payload), key="model")
    assert unwrap_snapshot_payload(store.read(handle.key).data) == payload
