"""ML-owned model artifact providers for the shared Case runtime."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from blackbase.resources import ArtifactSerializer, DataRef


class ArtifactProvider(Protocol):
    """Publish a fitted model and return its durable reference."""

    def publish_best_model(self, trainer: Any) -> DataRef | None: ...


SerializerSelector = Callable[[Any], str | ArtifactSerializer]


@dataclass
class CaseRuntimeArtifactProvider:
    """Publish through ``case_runtime`` without owning storage or L0 leases.

    Safe transport-native values use blackbase's built-in serializers. Torch
    modules and native ``save_model`` implementations receive explicit ML
    codecs. Other Python estimators fall back to an unsafe pickle serializer,
    which is rejected unless the Project artifact authority explicitly opts in.
    """

    serializer_selector: SerializerSelector | None = None
    artifact_name: str = "best_model"

    def publish_best_model(self, trainer: Any) -> DataRef | None:
        model = getattr(trainer, "best_model", None)
        if model is None:
            return None
        runtime = getattr(trainer, "case_runtime", None)
        publish = getattr(runtime, "publish_artifact", None)
        if not callable(publish):
            return None
        existing = getattr(runtime, "artifact_refs", {})
        if isinstance(existing, Mapping) and self.artifact_name in existing:
            ref = existing[self.artifact_name]
            return ref if isinstance(ref, DataRef) else DataRef.from_dict(ref)
        selector = self.serializer_selector or select_model_serializer
        serializer = selector(model)
        return publish(
            self.artifact_name,
            model,
            serializer=serializer,
            kind="model",
            metadata={
                "framework": "mlblack",
                "run_name": str(getattr(trainer, "run_name", "")),
                "model_type": type(model).__name__,
                "model_module": type(model).__module__,
            },
        )


def select_model_serializer(model: Any) -> str | ArtifactSerializer:
    """Choose a serializer without ever manufacturing a synthetic reference."""

    if _is_torch_module(model):
        return ArtifactSerializer(
            name="mlblack_torch_state_npz",
            extension=".npz",
            media_type="application/x-npz",
            dump=_dump_torch_state,
        )
    save_model = getattr(model, "save_model", None)
    if callable(save_model):
        return ArtifactSerializer(
            name="mlblack_native_model",
            extension=".json",
            media_type="application/json",
            dump=lambda value, path: value.save_model(str(path)),
        )
    if _is_transport_native(model):
        return "auto"
    return ArtifactSerializer(
        name="mlblack_pickle",
        extension=".pkl",
        media_type="application/x-python-pickle",
        dump=_dump_pickle,
        unsafe=True,
    )


def _is_torch_module(model: Any) -> bool:
    if not callable(getattr(model, "state_dict", None)):
        return False
    return any(
        base.__name__ == "Module" and str(base.__module__).startswith("torch.")
        for base in type(model).__mro__
    )


def _dump_torch_state(model: Any, path: Path) -> None:
    state = model.state_dict()
    arrays: dict[str, np.ndarray] = {}
    for name, value in state.items():
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        arrays[str(name)] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def _dump_pickle(model: Any, path: Path) -> None:
    with path.open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _is_transport_native(value: Any) -> bool:
    if value is None or isinstance(
        value,
        (str, bytes, bytearray, memoryview, bool, int, float, Path, np.ndarray),
    ):
        return True
    if isinstance(value, (dict, list, tuple)):
        return True
    return callable(getattr(value, "as_dict", None))


__all__ = [
    "ArtifactProvider",
    "CaseRuntimeArtifactProvider",
    "SerializerSelector",
    "select_model_serializer",
]
