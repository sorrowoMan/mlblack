from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from blackbase.resources import DataRef

from mlblack.core import ArtifactBuilder, ModelArtifact
from mlblack.core.artifact_provider import CaseRuntimeArtifactProvider


class _Runtime:
    artifact_refs: dict[str, DataRef] = {}

    def __init__(self) -> None:
        self.published: dict[str, Any] = {}

    def publish_artifact(self, name: str, value: Any, **kwargs: Any) -> DataRef:
        self.published[name] = {"value": value, **kwargs}
        return DataRef(
            uri=f"memory://{name}",
            kind=str(kwargs.get("kind", "artifact")),
            backend="memory",
        )


class _Problem:
    def __init__(self) -> None:
        self.calls = 0

    def build_model_artifact(self, model: Any, context: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        return {
            "schema": "test.model_artifact.v1",
            "model_name": model["name"],
            "resource_namespace": context["resource_namespace"],
        }


def _trainer() -> SimpleNamespace:
    problem = _Problem()
    runtime = _Runtime()
    return SimpleNamespace(
        best_model={"name": "candidate"},
        problem=problem,
        case_runtime=runtime,
        run_name="artifact-test",
        build_context=lambda: {"resource_namespace": "case:test"},
    )


def test_case_runtime_provider_publishes_problem_materialized_artifact() -> None:
    trainer = _trainer()

    ref = CaseRuntimeArtifactProvider().publish_best_model(trainer)

    assert ref is not None
    published = trainer.case_runtime.published["best_model"]
    assert published["value"] == {
        "schema": "test.model_artifact.v1",
        "model_name": "candidate",
        "resource_namespace": "case:test",
    }
    assert published["serializer"] == "auto"
    assert published["metadata"]["model_type"] == "dict"
    assert published["metadata"]["artifact_type"] == "dict"
    assert trainer.problem.calls == 1


def test_artifact_builder_uses_the_same_problem_materialization_boundary() -> None:
    trainer = _trainer()
    result = SimpleNamespace(best_model=trainer.best_model, report={})

    bundle = ArtifactBuilder().build(trainer, result)

    assert bundle.model_artifact is not None
    assert bundle.model_artifact.model["schema"] == "test.model_artifact.v1"
    assert trainer.problem.calls == 1


def test_case_runtime_provider_serializes_typed_artifact_model_and_keeps_semantic_metadata() -> None:
    torch = __import__("torch")
    model = torch.nn.Linear(2, 1)
    artifact = ModelArtifact(
        name="linear",
        model=model,
        family="neural",
        metadata={"contract": "test"},
    )
    runtime = _Runtime()
    trainer = SimpleNamespace(
        best_model=model,
        problem=SimpleNamespace(build_model_artifact=lambda value, context: artifact),
        case_runtime=runtime,
        run_name="typed-artifact-test",
        build_context=lambda: {},
    )

    ref = CaseRuntimeArtifactProvider().publish_best_model(trainer)

    assert ref is not None
    published = runtime.published["best_model"]
    assert published["value"] is model
    assert published["serializer"].name == "mlblack_torch_state_npz"
    semantic = published["metadata"]["semantic_artifact"]
    assert semantic["name"] == "linear"
    assert semantic["family"] == "neural"
    assert semantic["metadata"] == {"contract": "test"}


def test_artifact_materialization_enriches_only_local_context():
    trainer = _trainer()
    shared = {"resource_namespace": "case:test"}
    session = object()
    trainer.build_context = lambda: shared
    trainer.enrich_ml_context = lambda context: {**context, "backend.session": session}
    def build(model, context):
        assert context["backend.session"] is session
        return model
    trainer.problem.build_model_artifact = build
    CaseRuntimeArtifactProvider().publish_best_model(trainer)
    assert "backend.session" not in shared


def test_prepared_artifact_rejects_changed_model_instead_of_reopening_services():
    import pytest
    from mlblack.core.artifact_provider import materialize_model_artifact
    trainer = _trainer()
    trainer._prepared_model_artifact = (trainer.best_model, {"artifact": "prepared"})
    assert materialize_model_artifact(trainer, trainer.best_model) == {"artifact": "prepared"}
    with pytest.raises(RuntimeError, match="Best model changed"):
        materialize_model_artifact(trainer, {"another": "model"})
