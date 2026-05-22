from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from mlblack.core.context_contracts import ContextContract
from mlblack.core.contracts import ComponentContract


@dataclass(frozen=True)
class CatalogEntry:
    key: str
    title: str
    kind: str
    import_path: str
    tags: Sequence[str] = tuple()
    summary: str = ""
    contract: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "kind": self.kind,
            "import_path": self.import_path,
            "tags": list(self.tags),
            "summary": self.summary,
            "contract": dict(self.contract),
            "metadata": dict(self.metadata),
        }


class Catalog:
    def __init__(self, entries: Iterable[CatalogEntry]) -> None:
        self._entries = tuple(entries)

    def list(self, *, kind: str | None = None, tag: str | None = None) -> tuple[CatalogEntry, ...]:
        entries = self._entries
        if kind is not None:
            entries = tuple(item for item in entries if item.kind == str(kind))
        if tag is not None:
            entries = tuple(item for item in entries if str(tag) in {str(x) for x in item.tags})
        return entries

    def search(self, query: str, *, kind: str | None = None, limit: int = 20) -> tuple[CatalogEntry, ...]:
        q = str(query).strip().lower()
        entries = self.list(kind=kind)
        if not q:
            return tuple(entries[: max(0, int(limit))])
        matched = [
            item for item in entries
            if q in item.key.lower()
            or q in item.title.lower()
            or q in item.summary.lower()
            or any(q in str(tag).lower() for tag in item.tags)
        ]
        return tuple(matched[: max(0, int(limit))])

    def show(self, key: str) -> CatalogEntry:
        target = str(key)
        for item in self._entries:
            if item.key == target:
                return item
        raise KeyError(f"catalog entry not found: {key}")


def get_catalog() -> Catalog:
    return Catalog(_enrich_entries((*_default_entries(), *_backend_catalog_entries())))


def enrich_catalog_entry(entry: CatalogEntry) -> CatalogEntry:
    """Attach a resolved context contract to a static catalog entry."""

    try:
        obj = _resolve_import_path(entry.import_path)
    except Exception as exc:
        return CatalogEntry(
            key=entry.key,
            title=entry.title,
            kind=entry.kind,
            import_path=entry.import_path,
            tags=tuple(entry.tags),
            summary=entry.summary,
            contract=dict(entry.contract),
            metadata={**dict(entry.metadata), "contract_error": repr(exc)},
        )
    payload = _contract_payload_for_object(obj)
    if not payload:
        return entry
    return CatalogEntry(
        key=entry.key,
        title=entry.title,
        kind=entry.kind,
        import_path=entry.import_path,
        tags=tuple(entry.tags),
        summary=entry.summary,
        contract={**dict(entry.contract), **payload},
        metadata=dict(entry.metadata),
    )


def _enrich_entries(entries: Iterable[CatalogEntry]) -> tuple[CatalogEntry, ...]:
    return tuple(enrich_catalog_entry(entry) for entry in entries)


def _resolve_import_path(import_path: str) -> Any:
    module_name, sep, attr_name = str(import_path).partition(":")
    if not sep:
        return importlib.import_module(module_name)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in attr_name.split("."):
        obj = getattr(obj, part)
    return obj


def _contract_payload_for_object(obj: Any) -> dict[str, Any]:
    raw_contract = getattr(obj, "contract", None)
    has_context_attrs = any(
        hasattr(obj, attr)
        for attr in (
            "context_requires",
            "context_optional",
            "context_provides",
            "context_mutates",
            "context_cache",
            "requires_metrics",
            "metrics_fallback",
            "context_notes",
        )
    )
    if not has_context_attrs and not isinstance(raw_contract, (ComponentContract, ContextContract, Mapping)):
        return {}
    context_contract = ContextContract.from_component(obj, fallback_contract=raw_contract)
    if isinstance(raw_contract, ComponentContract):
        component_contract = ComponentContract.from_context_contract(
            context_contract,
            supports_gradient=raw_contract.supports_gradient,
            supports_batch=raw_contract.supports_batch,
            supports_resume=raw_contract.supports_resume,
            metadata=raw_contract.metadata,
        )
    else:
        component_contract = ComponentContract.from_context_contract(context_contract)
    payload = component_contract.describe()
    payload["unknown_context_keys"] = list(context_contract.unknown_keys())
    payload["unknown_metric_keys"] = list(context_contract.unknown_metric_keys())
    return payload


def _backend_catalog_entries() -> tuple[CatalogEntry, ...]:
    try:
        from mlblack.backends.catalog import list_backend_catalog_entries
    except Exception:
        return tuple()
    entries: list[CatalogEntry] = []
    for item in list_backend_catalog_entries():
        kind = str(item.get("kind", "backend"))
        name = str(item.get("name", "backend"))
        entries.append(
            CatalogEntry(
                key=f"{kind}.{name}",
                title=name,
                kind=kind,
                import_path="mlblack.backends.registry:get_backend",
                tags=("backend", str(item.get("backend", name)).split(".")[0]),
                summary=f"Backend catalog entry for {name}.",
                contract={
                    "provides": tuple(item.get("provides", ())),
                    "methods": dict(item.get("methods", {})),
                },
                metadata=dict(item.get("metadata", item)),
            )
        )
    return tuple(entries)


def _default_entries() -> tuple[CatalogEntry, ...]:
    return (
        CatalogEntry(
            key="trainer.composable",
            title="ComposableTrainer",
            kind="trainer",
            import_path="mlblack.core.trainer:ComposableTrainer",
            tags=("control-plane", "solver-like"),
            summary="Trainer control plane with mounted OptimizerAdapter.",
        ),
        CatalogEntry(
            key="adapter.gradient_descent",
            title="GradientDescentAdapter",
            kind="adapter",
            import_path="mlblack.adapters.gradient_descent:GradientDescentAdapter",
            tags=("gradient", "linear", "resume"),
            summary="Consumes feedback.gradients and updates UnknownState.",
        ),
        CatalogEntry(
            key="adapter.functional_backprop",
            title="FunctionalBackpropAdapter",
            kind="adapter",
            import_path="mlblack.adapters.functional_backprop:FunctionalBackpropAdapter",
            tags=("gradient", "neural", "functional-backend", "resume"),
            summary="Uses problem-owned functional gradients plus backend optimizer.sgd_step.",
        ),
        CatalogEntry(
            key="adapter.random_search",
            title="RandomSearchAdapter",
            kind="adapter",
            import_path="mlblack.adapters.random_search:RandomSearchAdapter",
            tags=("black-box", "interval", "resume"),
            summary="Black-box candidate search for non-gradient heads.",
        ),
        CatalogEntry(
            key="adapter.estimator_spec_search",
            title="EstimatorSpecSearchAdapter",
            kind="adapter",
            import_path="mlblack.adapters.estimator_search:EstimatorSpecSearchAdapter",
            tags=("tree", "xgboost", "sklearn", "resume"),
            summary="Searches decoded external estimator specs.",
        ),
        CatalogEntry(
            key="adapter.torch_backprop",
            title="TorchBackpropAdapter",
            kind="adapter",
            import_path="mlblack.adapters.torch_backprop:TorchBackpropAdapter",
            tags=("neural", "torch", "resource-context", "resume"),
            summary="Torch gradient engine for parameter-vector MLP representations.",
        ),
        CatalogEntry(
            key="representation.orthogonal_linear",
            title="OrthogonalPointLinearRepresentation",
            kind="representation",
            import_path="mlblack.representations.orthogonal_point:OrthogonalPointLinearRepresentation",
            tags=("linear", "orthogonal", "head"),
            summary="Unknown vector decoded through orthogonal feature map and output head.",
        ),
        CatalogEntry(
            key="representation.estimator_spec",
            title="EstimatorSpecRepresentation",
            kind="representation",
            import_path="mlblack.representations.estimator_specs:EstimatorSpecRepresentation",
            tags=("tree", "xgboost", "sklearn", "codec"),
            summary="Unknown vector decoded into a typed external estimator spec.",
        ),
        CatalogEntry(
            key="representation.numpy_mlp",
            title="NumpyMLPPointRepresentation",
            kind="representation",
            import_path="mlblack.representations.neural_mlp:NumpyMLPPointRepresentation",
            tags=("neural", "torch", "head", "codec"),
            summary="Flat parameter vector decoded into a numpy MLP model.",
        ),
        CatalogEntry(
            key="problem.supervised_regression",
            title="SupervisedRegressionProblem",
            kind="problem",
            import_path="mlblack.problems.supervised:SupervisedRegressionProblem",
            tags=("regression", "gradient", "residuals"),
            summary="Data-dependent evaluator for point regression.",
        ),
        CatalogEntry(
            key="model.integrated_prediction",
            title="IntegratedPredictionModel",
            kind="model",
            import_path="mlblack.models.composition:IntegratedPredictionModel",
            tags=("composition", "integration", "residual", "stacking"),
            summary="Combines named fitted model predictions without owning training orchestration.",
        ),
        CatalogEntry(
            key="pipeline.model_conditioned_target",
            title="ModelConditionedTargetComponent",
            kind="pipeline",
            import_path="mlblack.pipeline.model_conditioning:ModelConditionedTargetComponent",
            tags=("pipeline", "residual", "model-conditioned", "stage-surface"),
            summary="Builds next-stage targets by calling an existing model, e.g. y - model.predict(X).",
        ),
        CatalogEntry(
            key="problem.supervised_estimator_fit",
            title="SupervisedEstimatorFitRegressionProblem",
            kind="problem",
            import_path="mlblack.problems.supervised:SupervisedEstimatorFitRegressionProblem",
            tags=("tree", "xgboost", "artifact"),
            summary="Fits decoded estimator specs and scores the fitted estimator.",
        ),
        CatalogEntry(
            key="capability.checkpoint",
            title="CheckpointCapability",
            kind="capability",
            import_path="mlblack.capabilities.checkpoint:CheckpointCapability",
            tags=("state", "resume", "snapshot"),
            summary="Writes trainer state snapshots during fit.",
        ),
        CatalogEntry(
            key="capability.experiment_tracker",
            title="ExperimentTrackerCapability",
            kind="capability",
            import_path="mlblack.capabilities.tracking:ExperimentTrackerCapability",
            tags=("experiment", "sqlite", "run-record"),
            summary="Records fit/step/evaluation events to an experiment store.",
        ),
        CatalogEntry(
            key="capability.resource_audit",
            title="ResourceAuditCapability",
            kind="capability",
            import_path="mlblack.capabilities.resource_audit:ResourceAuditCapability",
            tags=("l0", "resource", "audit"),
            summary="Audits effective ResourceContext during fit.",
        ),
        CatalogEntry(
            key="bias.state_l2",
            title="StateL2Bias",
            kind="bias",
            import_path="mlblack.bias.policies:StateL2Bias",
            tags=("soft-preference", "regularization"),
            summary="Adds a soft L2 penalty to feedback objectives.",
        ),
        CatalogEntry(
            key="bias.objective_weight",
            title="ObjectiveWeightBias",
            kind="bias",
            import_path="mlblack.bias.policies:ObjectiveWeightBias",
            tags=("soft-preference", "multi-objective"),
            summary="Reweights objective dimensions before adapter update.",
        ),
        CatalogEntry(
            key="representation.piecewise",
            title="PiecewiseRepresentation",
            kind="representation",
            import_path="mlblack.representations.conditional:PiecewiseRepresentation",
            tags=("conditional", "piecewise", "router"),
            summary="Concatenates branch states and decodes a routed piecewise model.",
        ),
        CatalogEntry(
            key="problem.supervised_classification",
            title="SupervisedClassificationProblem",
            kind="problem",
            import_path="mlblack.problems.classification:SupervisedClassificationProblem",
            tags=("classification", "probability", "log-loss"),
            summary="Evaluator for classification accuracy and log-loss objectives.",
        ),
        CatalogEntry(
            key="assembly.build_trainer",
            title="build_trainer",
            kind="assembly",
            import_path="mlblack.assembly.builders:build_trainer",
            tags=("scaffold", "trainer", "inner-training"),
            summary="Builds one inner ML trainer; orchestration is delegated to nsgablack.",
        ),
        CatalogEntry(
            key="schema.scaffold_config",
            title="ScaffoldConfig",
            kind="schema",
            import_path="mlblack.assembly.schema.spec:ScaffoldConfig",
            tags=("schema", "config", "scaffold"),
            summary="Top-level JSON-compatible scaffold contract.",
        ),
        CatalogEntry(
            key="problem.training_proxy",
            title="MLBlackTrainingProxy",
            kind="problem_bridge",
            import_path="mlblack.problems.proxy:MLBlackTrainingProxy",
            tags=("cross-framework", "proxy", "training-contract"),
            summary="Framework-neutral proxy for outer optimizers invoking mlblack inner training.",
        ),
        CatalogEntry(
            key="numericizer.default",
            title="DefaultNumericizer",
            kind="numericizer",
            import_path="mlblack.pipeline.numericizer.default:DefaultNumericizer",
            tags=("data", "schema", "feature-space"),
            summary="Converts schema-backed raw rows into NumericDataView.",
        ),
        CatalogEntry(
            key="pipeline.feature_space",
            title="FeatureSpaceComponent",
            kind="pipeline",
            import_path="mlblack.pipeline.feature_space:FeatureSpaceComponent",
            tags=("pipeline", "feature-space", "metadata"),
            summary="Records feature-space metadata in the data pipeline.",
        ),
        CatalogEntry(
            key="conditional.primitives",
            title="Conditional Primitives",
            kind="conditional",
            import_path="mlblack.pipeline.conditional.primitives:ConditionalPrimitive",
            tags=("binary-gate", "soft-gate", "hinge", "onehot"),
            summary="Reusable conditional feature and routing primitives.",
        ),
        CatalogEntry(
            key="conditional.composer",
            title="PrimitiveFeatureComposer",
            kind="conditional",
            import_path="mlblack.pipeline.conditional.composer:PrimitiveFeatureComposer",
            tags=("composer", "feature-engineering", "piecewise"),
            summary="Composes conditional primitives into deterministic feature transforms.",
        ),
        CatalogEntry(
            key="bias.dynamic_pool",
            title="DynamicPoolBias",
            kind="bias",
            import_path="mlblack.bias.policies:DynamicPoolBias",
            tags=("soft-preference", "pool", "event"),
            summary="Projects a context-dependent candidate/model pool hint.",
        ),
        CatalogEntry(
            key="bias.branch_policy",
            title="BranchPolicyBias",
            kind="bias",
            import_path="mlblack.bias.policies:BranchPolicyBias",
            tags=("soft-preference", "conditional", "branch"),
            summary="Exposes branch preferences for conditional/piecewise representations.",
        ),
        CatalogEntry(
            key="bias.objective_policy",
            title="ObjectivePolicyBias",
            kind="bias",
            import_path="mlblack.bias.policies:ObjectivePolicyBias",
            tags=("soft-preference", "multi-objective", "policy"),
            summary="Context-aware objective reweighting policy.",
        ),
        CatalogEntry(
            key="dashboard.catalog_html",
            title="catalog dashboard export",
            kind="dashboard",
            import_path="mlblack.catalog.dashboard:export_catalog_html",
            tags=("catalog", "html", "report"),
            summary="Exports a lightweight HTML catalog report.",
        ),
        CatalogEntry(
            key="dashboard.experiment_html",
            title="experiment dashboard export",
            kind="dashboard",
            import_path="mlblack.catalog.experiment.dashboard:export_experiment_html",
            tags=("experiment", "html", "report"),
            summary="Exports a lightweight HTML experiment record report.",
        ),
        CatalogEntry(
            key="dashboard.artifact_html",
            title="artifact dashboard export",
            kind="dashboard",
            import_path="mlblack.catalog.artifacts:export_artifact_html",
            tags=("artifact", "symbolic", "html", "report"),
            summary="Exports a static HTML viewer for typed mlblack artifacts.",
        ),
        CatalogEntry(
            key="dashboard.backend_matrix_html",
            title="backend capability matrix export",
            kind="dashboard",
            import_path="mlblack.catalog.backend_dashboard:export_backend_matrix_html",
            tags=("backend", "capability", "matrix", "html", "report"),
            summary="Exports a static HTML matrix of backend capability support.",
        ),
        CatalogEntry(
            key="head.binary_logistic",
            title="BinaryLogisticHead",
            kind="head",
            import_path="mlblack.representations.heads.probability:BinaryLogisticHead",
            tags=("classification", "probability", "logistic"),
            summary="Wraps a scalar logit decoder as binary predict_proba output.",
        ),
        CatalogEntry(
            key="head.softmax",
            title="SoftmaxHead",
            kind="head",
            import_path="mlblack.representations.heads.probability:SoftmaxHead",
            tags=("classification", "probability", "multiclass"),
            summary="Allocates one base decoder block per class and returns softmax probabilities.",
        ),
        CatalogEntry(
            key="head.piecewise",
            title="PiecewiseHead",
            kind="head",
            import_path="mlblack.representations.heads.conditional:PiecewiseHead",
            tags=("conditional", "piecewise", "branch"),
            summary="Allocates one base decoder block per branch and returns a PiecewiseModel.",
        ),
        CatalogEntry(
            key="preset.orthogonal_logistic_classification",
            title="orthogonal logistic classification preset",
            kind="preset",
            import_path="mlblack.presets.classification:build_orthogonal_logistic_classification_trainer",
            tags=("classification", "logistic", "orthogonal"),
            summary="Orthogonal linear logits with binary probability head and classification metrics.",
        ),
        CatalogEntry(
            key="catalog.query",
            title="catalog query",
            kind="catalog",
            import_path="mlblack.catalog.query:query_catalog",
            tags=("catalog", "facet", "deep-link"),
            summary="Search catalog entries with facets and deep-link payload.",
        ),
        CatalogEntry(
            key="experiment.query",
            title="experiment query",
            kind="experiment",
            import_path="mlblack.catalog.experiment.query:query_experiments",
            tags=("experiment", "sqlite", "facet"),
            summary="Query SQLite experiment records with simple filters and facets.",
        ),
    )


