from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover - optional import guard
    raise ImportError(
        "PyTorch is required for symbolic trainer-state persistence. Install torch before using symbolic checkpoints."
    ) from exc


def clone_symbolic_payload_cpu(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return np.asarray(value).copy()
    if isinstance(value, Mapping):
        return {str(k): clone_symbolic_payload_cpu(v) for k, v in dict(value).items()}
    if isinstance(value, tuple):
        return tuple(clone_symbolic_payload_cpu(v) for v in value)
    if isinstance(value, list):
        return [clone_symbolic_payload_cpu(v) for v in value]
    return value


def save_symbolic_trainer_state_file(
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
        "payload": clone_symbolic_payload_cpu(dict(payload)),
        "metadata": {} if metadata is None else dict(metadata),
    }
    torch.save(checkpoint, out_path)
    return str(out_path)


def load_symbolic_trainer_state_file(path: str | Path) -> dict[str, Any]:
    in_path = Path(path).resolve()
    checkpoint = torch.load(in_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "payload" in checkpoint and "trainer_name" in checkpoint:
        payload = checkpoint.get("payload")
        if not isinstance(payload, Mapping):
            raise TypeError("symbolic trainer state file contains non-mapping payload")
        raw = dict(payload)
    elif isinstance(checkpoint, Mapping):
        raw = dict(checkpoint)
    else:
        raise TypeError(f"trainer state file must contain a mapping payload, got {type(checkpoint).__name__}")

    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("unsupported symbolic trainer_state schema_version")
    return raw


__all__ = [
    "clone_symbolic_payload_cpu",
    "load_symbolic_trainer_state_file",
    "save_symbolic_trainer_state_file",
]
