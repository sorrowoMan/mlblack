"""ML-owned model artifact providers for the shared Case runtime."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from blackbase import CallCandidate, invoke_bound_once
from blackbase.resources import ArtifactSerializer, DataRef


class ArtifactFinalizationSession(Protocol):
    """Stage one complete ML result for the Case authority to seal."""

    def publish_best_model(self) -> DataRef | None: ...

    def publish_payload(
        self,
        name: str,
        value: Any,
        *,
        kind: str,
    ) -> DataRef | None: ...

    def prepare(self) -> Mapping[str, DataRef]: ...

    def abort(self, reason: str = "aborted") -> None: ...


class ArtifactProvider(Protocol):
    """Provide both compatibility publication and transactional finalization."""

    def begin_finalization(
        self,
        trainer: Any,
        *,
        label: str = "mlblack_result_finalization",
    ) -> ArtifactFinalizationSession: ...

    def publish_best_model(self, trainer: Any) -> DataRef | None: ...

    def publish_payload(
        self,
        trainer: Any,
        name: str,
        value: Any,
        *,
        kind: str,
    ) -> DataRef | None: ...


SerializerSelector = Callable[[Any], str | ArtifactSerializer]


@dataclass
class _CaseRuntimeArtifactFinalizationSession:
    provider: "CaseRuntimeArtifactProvider"
    trainer: Any
    transaction: Any | None
    _active: bool = True

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("ML Artifact finalization session is closed")

    def publish_best_model(self) -> DataRef | None:
        self._require_active()
        model = getattr(self.trainer, "best_model", None)
        if model is None or self.transaction is None:
            return None
        artifact = materialize_model_artifact(self.trainer, model)
        publish_value, semantic_metadata = artifact_storage_payload(artifact)
        selector = self.provider.serializer_selector or select_model_serializer
        serializer = selector(publish_value)
        return self.transaction.publish(
            self.provider.artifact_name,
            publish_value,
            serializer=serializer,
            kind="model",
            metadata={
                "framework": "mlblack",
                "run_name": str(getattr(self.trainer, "run_name", "")),
                "model_type": type(model).__name__,
                "model_module": type(model).__module__,
                "artifact_type": type(artifact).__name__,
                "artifact_module": type(artifact).__module__,
                **semantic_metadata,
            },
        )

    def publish_payload(
        self,
        name: str,
        value: Any,
        *,
        kind: str,
    ) -> DataRef | None:
        self._require_active()
        if self.transaction is None:
            return None
        artifact_name = str(name or "").strip()
        if not artifact_name:
            raise ValueError("result artifact name must be non-empty")
        return self.transaction.publish(
            artifact_name,
            value,
            serializer="auto",
            kind=str(kind or "trainer_result"),
            metadata={
                "framework": "mlblack",
                "run_name": str(getattr(self.trainer, "run_name", "")),
                "result_field": artifact_name,
            },
        )

    def prepare(self) -> Mapping[str, DataRef]:
        self._require_active()
        # A Case finalization session may only prepare stable result refs.  The
        # BlackBase CaseExecutor seals the shared transaction after output
        # serialization, heartbeat shutdown and runtime cleanup all succeed.
        result = (
            {}
            if self.transaction is None
            else dict(self.transaction.prepare())
        )
        self._active = False
        return result

    def abort(self, reason: str = "aborted") -> None:
        if not self._active:
            return
        try:
            if self.transaction is not None:
                self.transaction.abort(reason)
        finally:
            self._active = False


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

    def begin_finalization(
        self,
        trainer: Any,
        *,
        label: str = "mlblack_result_finalization",
    ) -> ArtifactFinalizationSession:
        runtime = getattr(trainer, "case_runtime", None)
        if runtime is None:
            return _CaseRuntimeArtifactFinalizationSession(self, trainer, None)
        begin = getattr(runtime, "begin_finalization_transaction", None)
        if not callable(begin):
            begin = getattr(runtime, "begin_artifact_transaction", None)
        if not callable(begin):
            raise RuntimeError(
                "Case runtime must implement begin_finalization_transaction(...) "
                "for atomic ML result publication"
            )
        return _CaseRuntimeArtifactFinalizationSession(
            self,
            trainer,
            begin(label),
        )

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
        artifact = materialize_model_artifact(trainer, model)
        publish_value, semantic_metadata = artifact_storage_payload(artifact)
        selector = self.serializer_selector or select_model_serializer
        serializer = selector(publish_value)
        return publish(
            self.artifact_name,
            publish_value,
            serializer=serializer,
            kind="model",
            metadata={
                "framework": "mlblack",
                "run_name": str(getattr(trainer, "run_name", "")),
                "model_type": type(model).__name__,
                "model_module": type(model).__module__,
                "artifact_type": type(artifact).__name__,
                "artifact_module": type(artifact).__module__,
                **semantic_metadata,
            },
        )

    def publish_payload(
        self,
        trainer: Any,
        name: str,
        value: Any,
        *,
        kind: str,
    ) -> DataRef | None:
        """Publish a transport-native result field through Case authority."""

        runtime = getattr(trainer, "case_runtime", None)
        publish = getattr(runtime, "publish_artifact", None)
        if not callable(publish):
            return None
        artifact_name = str(name or "").strip()
        if not artifact_name:
            raise ValueError("result artifact name must be non-empty")
        existing = getattr(runtime, "artifact_refs", {})
        if isinstance(existing, Mapping) and artifact_name in existing:
            ref = existing[artifact_name]
            return ref if isinstance(ref, DataRef) else DataRef.from_dict(ref)
        return publish(
            artifact_name,
            value,
            serializer="auto",
            kind=str(kind or "trainer_result"),
            metadata={
                "framework": "mlblack",
                "run_name": str(getattr(trainer, "run_name", "")),
                "result_field": artifact_name,
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


def materialize_model_artifact(trainer: Any, model: Any) -> Any:
    """Project a fitted model through the Problem-owned artifact boundary.

    The optimization candidate is an in-process representation.  A semantic
    Problem may expose a different, backend-neutral payload for persistence.
    Invocation is bound before execution so an exception raised inside the
    builder is never mistaken for an alternate call signature.
    """

    problem = getattr(trainer, "problem", None)
    materialize = getattr(problem, "build_model_artifact", None)
    if not callable(materialize):
        return model
    build_context = getattr(trainer, "build_context", None)
    context = build_context() if callable(build_context) else {}
    return invoke_bound_once(
        materialize,
        (
            CallCandidate(args=(model, context), label="model-and-context"),
            CallCandidate(args=(model,), label="model-only"),
        ),
    )


def artifact_storage_payload(artifact: Any) -> tuple[Any, dict[str, Any]]:
    """Separate a semantic artifact envelope from its serializer payload.

    Typed ML artifacts carry audit metadata around an executable model.  The
    executable payload is serialized by its backend codec (for example torch
    state to NPZ), while the bounded semantic description travels in DataRef
    metadata.  This avoids pickling the wrapper merely because it contains a
    backend model.
    """

    from .artifacts import ModelArtifact
    from mlblack.models import FittedEstimatorModel

    if not isinstance(artifact, ModelArtifact):
        return artifact, {}
    publish_value = artifact.model
    semantic_metadata: dict[str, Any] = {
        "semantic_artifact": artifact.describe()
    }
    if isinstance(publish_value, FittedEstimatorModel):
        semantic_metadata["fitted_estimator_model"] = publish_value.describe()
        publish_value = publish_value.estimator
    return publish_value, semantic_metadata


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
    "ArtifactFinalizationSession",
    "ArtifactProvider",
    "CaseRuntimeArtifactProvider",
    "SerializerSelector",
    "artifact_storage_payload",
    "materialize_model_artifact",
    "select_model_serializer",
]
