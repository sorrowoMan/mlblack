from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def clone_pickled_trainer_payload(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.asarray(value).copy()
    if isinstance(value, Mapping):
        return {str(k): clone_pickled_trainer_payload(v) for k, v in dict(value).items()}
    if isinstance(value, tuple):
        return tuple(clone_pickled_trainer_payload(v) for v in value)
    if isinstance(value, list):
        return [clone_pickled_trainer_payload(v) for v in value]
    return value


def save_pickled_trainer_state_file(
    path: str | Path,
    *,
    trainer_name: str,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> str:
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "trainer_name": str(trainer_name),
        "payload": clone_pickled_trainer_payload(dict(payload)),
        "metadata": {} if metadata is None else dict(metadata),
    }
    with out_path.open("wb") as fh:
        pickle.dump(checkpoint, fh)
    return str(out_path)


def load_pickled_trainer_state_file(path: str | Path) -> dict[str, Any]:
    in_path = Path(path).resolve()
    with in_path.open("rb") as fh:
        checkpoint = pickle.load(fh)
    if isinstance(checkpoint, dict) and "payload" in checkpoint and "trainer_name" in checkpoint:
        payload = checkpoint.get("payload")
        if not isinstance(payload, Mapping):
            raise TypeError("pickled trainer state file contains non-mapping payload")
        raw = dict(payload)
    elif isinstance(checkpoint, Mapping):
        raw = dict(checkpoint)
    else:
        raise TypeError(f"trainer state file must contain a mapping payload, got {type(checkpoint).__name__}")

    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("unsupported pickled trainer_state schema_version")
    return raw


__all__ = [
    "clone_pickled_trainer_payload",
    "load_pickled_trainer_state_file",
    "save_pickled_trainer_state_file",
]
