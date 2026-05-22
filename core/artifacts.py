from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import pickle
from typing import Any, Mapping

from .state import TrainerState, build_trainer_state


@dataclass(frozen=True)
class ModelArtifact:
    """Serializable boundary around the trained model payload."""

    name: str
    model: Any
    family: str = ""
    head: str = "point"
    representation: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def describe(self, *, include_model: bool = False) -> dict[str, Any]:
        data = {
            "name": self.name,
            "family": self.family,
            "head": self.head,
            "representation": dict(self.representation),
            "metadata": dict(self.metadata),
        }
        if include_model:
            data["model"] = self.model
        else:
            data["model_type"] = type(self.model).__name__
        return data


@dataclass(frozen=True)
class TypedModelArtifact(ModelArtifact):
    artifact_type: str = "model"

    def describe(self, *, include_model: bool = False) -> dict[str, Any]:
        data = super().describe(include_model=include_model)
        data["artifact_type"] = self.artifact_type
        return data


@dataclass(frozen=True)
class TreeEnsembleArtifact(TypedModelArtifact):
    artifact_type: str = "tree_ensemble"


@dataclass(frozen=True)
class XGBoostArtifact(TypedModelArtifact):
    artifact_type: str = "xgboost"


@dataclass(frozen=True)
class EstimatorStateArtifact(TypedModelArtifact):
    artifact_type: str = "estimator_state"
    estimator_state: Mapping[str, Any] = field(default_factory=dict)

    def describe(self, *, include_model: bool = False) -> dict[str, Any]:
        data = super().describe(include_model=include_model)
        data["estimator_state"] = dict(self.estimator_state)
        return data


@dataclass(frozen=True)
class SklearnMLPArtifact(TypedModelArtifact):
    artifact_type: str = "sklearn_mlp"


@dataclass(frozen=True)
class TorchModelArtifact(TypedModelArtifact):
    artifact_type: str = "torch_model"


@dataclass(frozen=True)
class IntegratedModelArtifact(TypedModelArtifact):
    artifact_type: str = "integrated_model"


@dataclass(frozen=True)
class NeuralGraphArtifact(TypedModelArtifact):
    artifact_type: str = "neural_graph"
    graph_spec: Mapping[str, Any] = field(default_factory=dict)
    parameter_layout: Mapping[str, Any] = field(default_factory=dict)
    head_artifact: Mapping[str, Any] = field(default_factory=dict)
    audit_artifact: Mapping[str, Any] = field(default_factory=dict)
    graph_spec_digest: str = ""
    parameter_layout_digest: str = ""

    def describe(self, *, include_model: bool = False) -> dict[str, Any]:
        data = super().describe(include_model=include_model)
        data["graph_spec"] = dict(self.graph_spec)
        data["parameter_layout"] = dict(self.parameter_layout)
        data["head_artifact"] = dict(self.head_artifact)
        data["audit_artifact"] = dict(self.audit_artifact)
        data["graph_spec_digest"] = self.graph_spec_digest
        data["parameter_layout_digest"] = self.parameter_layout_digest
        return data


@dataclass(frozen=True)
class SymbolicModelArtifact(TypedModelArtifact):
    artifact_type: str = "symbolic_model"


@dataclass(frozen=True)
class SymbolicIntervalArtifact(TypedModelArtifact):
    artifact_type: str = "symbolic_interval"


@dataclass(frozen=True)
class TrainerStateArtifact:
    """Checkpoint-style trainer state payload."""

    payload: Mapping[str, Any]
    signature: str = ""
    version: str = "mlblack.trainer_state.v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "signature": self.signature,
            "keys": sorted(str(key) for key in self.payload.keys()),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RunReport:
    """Auditable run summary distinct from artifact and trainer state."""

    run_name: str
    status: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    components: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "status": self.status,
            "metrics": dict(self.metrics),
            "components": dict(self.components),
            "resources": dict(self.resources),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ArtifactBundle:
    """Final output bundle with explicit product boundaries."""

    model_artifact: ModelArtifact | None = None
    trainer_state: TrainerStateArtifact | None = None
    run_report: RunReport | None = None
    snapshot_refs: tuple[str, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "model_artifact": None if self.model_artifact is None else self.model_artifact.describe(),
            "trainer_state": None if self.trainer_state is None else self.trainer_state.describe(),
            "run_report": None if self.run_report is None else self.run_report.describe(),
            "snapshot_refs": list(self.snapshot_refs),
            "metadata": dict(self.metadata),
        }

    def save(self, path: str | Path) -> Path:
        return save_artifact_bundle(self, path)


class ArtifactBuilder:
    """Default artifact builder for a finished trainer result."""

    def build(self, trainer: Any, result: Any) -> ArtifactBundle:
        best_model = getattr(result, "best_model", None)
        problem = getattr(trainer, "problem", None)
        materialize = getattr(problem, "build_model_artifact", None)
        if best_model is not None and callable(materialize):
            context = trainer.build_context() if hasattr(trainer, "build_context") else {}
            best_model = materialize(best_model, context)
        report = dict(getattr(result, "report", {}) or {})
        representation = report.get("representation") or {}
        problem = report.get("problem") or {}
        adapter = report.get("adapter") or {}
        model_artifact = None
        if best_model is not None:
            if isinstance(best_model, ModelArtifact):
                model_artifact = best_model
            else:
                model_artifact = _build_typed_model_artifact(
                    name=str(report.get("run_name", getattr(trainer, "run_name", "model"))),
                    model=best_model,
                    family=str(problem.get("family", problem.get("name", ""))),
                    head=str(problem.get("head", "point")),
                    representation=representation,
                    metadata={"problem": problem, "adapter": adapter},
                )

        state = build_trainer_state(trainer) if hasattr(trainer, "get_state") else TrainerState(payload={})
        trainer_state = TrainerStateArtifact(
            payload=dict(state.payload),
            signature=str(report.get("state_signature", state.signature)),
            metadata={"run_name": report.get("run_name", getattr(trainer, "run_name", ""))},
        )
        run_report = RunReport(
            run_name=str(report.get("run_name", getattr(trainer, "run_name", ""))),
            status=str(report.get("status", "finished")),
            metrics=dict(report.get("best_metrics", {})),
            components={
                "problem": problem,
                "representation": representation,
                "adapter": adapter,
            },
            resources=dict(report.get("resources", {})),
            metadata={key: value for key, value in report.items() if key not in {"best_metrics"}},
        )
        snapshot_ref = getattr(trainer, "context", {}).get("last_population_snapshot")
        snapshot_refs = tuple() if snapshot_ref is None else (str(snapshot_ref),)
        return ArtifactBundle(
            model_artifact=model_artifact,
            trainer_state=trainer_state,
            run_report=run_report,
            snapshot_refs=snapshot_refs,
        )


def save_artifact_bundle(bundle: ArtifactBundle, path: str | Path) -> Path:
    """Persist bundle metadata plus pickle payload.

    The JSON file is for audit. The pickle file is the payload boundary for
    arbitrary Python estimator/model objects.
    """

    base = Path(path)
    if base.suffix:
        directory = base.parent
        stem = base.stem
    else:
        directory = base
        stem = "artifact_bundle"
    directory.mkdir(parents=True, exist_ok=True)
    pickle_path = directory / f"{stem}.pkl"
    json_path = directory / f"{stem}.json"
    with pickle_path.open("wb") as f:
        pickle.dump(bundle, f)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "bundle": bundle.describe(),
                "payload": {"pickle": pickle_path.name},
            },
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    return pickle_path


def load_artifact_bundle(path: str | Path) -> ArtifactBundle:
    payload_path = Path(path)
    if payload_path.suffix.lower() == ".json":
        with payload_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        pickle_name = str(meta.get("payload", {}).get("pickle", payload_path.with_suffix(".pkl").name))
        payload_path = payload_path.parent / pickle_name
    with payload_path.open("rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, ArtifactBundle):
        raise TypeError("artifact payload is not an ArtifactBundle")
    return bundle


def _build_typed_model_artifact(**kwargs: Any) -> ModelArtifact:
    model = kwargs.get("model")
    family = str(kwargs.get("family", "") or "").lower()
    route = str(getattr(model, "route", "") or kwargs.get("metadata", {}).get("problem", {}).get("route", "")).lower()
    model_type = type(model).__name__.lower()
    state_summary = None
    if hasattr(model, "fitted_state_summary"):
        try:
            state_summary = model.fitted_state_summary()
        except Exception:
            state_summary = None
    if "xgb" in route or "xgboost" in route or "xgb" in model_type:
        artifact = XGBoostArtifact(**kwargs)
        if state_summary:
            return EstimatorStateArtifact(**{**kwargs, "artifact_type": "xgboost", "estimator_state": state_summary})
        return artifact
    if "neural_graph" in route or "tinytransformer" in model_type or "tiny_transformer" in route:
        return NeuralGraphArtifact(**kwargs)
    if "integratedpredictionmodel" in model_type or "integrated_prediction" in model_type:
        return IntegratedModelArtifact(**kwargs)
    if "mlp" in route or "mlp" in model_type:
        if "torch" in route or "torch" in model_type:
            return TorchModelArtifact(**kwargs)
        return SklearnMLPArtifact(**kwargs)
    if "tree" in family or "tree" in route or "forest" in route or "boost" in route or "tree" in model_type:
        if state_summary:
            return EstimatorStateArtifact(**{**kwargs, "artifact_type": "tree_ensemble", "estimator_state": state_summary})
        return TreeEnsembleArtifact(**kwargs)
    if "numpymlp" in model_type:
        return TorchModelArtifact(**kwargs)
    if "symbolic" in family or "symbolic" in route or "symbolic" in model_type:
        head = str(kwargs.get("head", "") or "").lower()
        if "interval" in head:
            return SymbolicIntervalArtifact(**kwargs)
        return SymbolicModelArtifact(**kwargs)
    return ModelArtifact(**kwargs)

