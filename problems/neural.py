from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from mlblack.core.artifacts import NeuralGraphArtifact
from mlblack.core.backend_session import get_compute_backend_from_context
from mlblack.core.contracts import ComponentContract
from mlblack.core.problem import LearningProblem
from mlblack.core.types import Feedback, UnknownState
from mlblack.pipeline.data import GraphDataView, ImageContrastivePairDataView, ImageDataView, NumericDataView, PreferencePairDataView


@dataclass(frozen=True)
class BackendLossEvaluation:
    """Backend-native differentiable loss plus scalar evaluation feedback.

    `loss` is intentionally backend-native (for example a torch Tensor). The
    problem computes it, but the adapter decides whether to call backward().
    """

    objectives: np.ndarray
    loss: Any
    loss_value: float
    metrics: Mapping[str, Any] = field(default_factory=dict)
    signals: Mapping[str, Any] = field(default_factory=dict)

    def as_feedback(self) -> Feedback:
        return Feedback(
            objectives=np.asarray(self.objectives, dtype=float),
            loss=float(self.loss_value),
            gradients=None,
            metrics=dict(self.metrics),
            signals=dict(self.signals),
        )


_NEURAL_EVALUATION_PROVIDES = ("feedback.objectives", "feedback.loss", "feedback.metrics", "feedback.signals")
_BACKEND_MODE_REQUIREMENTS = ("tensor.device", "autograd.mode.train", "autograd.mode.eval", "autograd.no_grad")
_DIFFERENTIABLE_LOSS_NOTE = (
    "Computes neural loss/metrics; gradients are owned by NeuralGraphBackpropAdapter via compute_backend_loss()."
)


class TinyTransformerClassificationProblem(LearningProblem):
    """Evaluate tiny Transformer classification heads with torch autograd."""

    name = "tiny_transformer_classification"
    backend_requires = (*_BACKEND_MODE_REQUIREMENTS, "tensor.token_ids", "tensor.class_labels", "loss.cross_entropy", "metrics.classification")
    context_requires = ("candidate.model", "data.X_train", "data.y_train")
    context_optional = ("data.X_valid", "data.y_valid", "resource.device", "neural.attention_maps", "neural.ffn_activations")
    context_provides = _NEURAL_EVALUATION_PROVIDES
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = _DIFFERENTIABLE_LOSS_NOTE
    contract = ComponentContract(
        name=name,
        requires=("candidate.model", "data.X_train", "data.y_train"),
        optional=("data.X_valid", "data.y_valid", "resource.device", "neural.attention_maps", "neural.ffn_activations"),
        provides=context_provides,
        supports_gradient=True,
        supports_batch=False,
        supports_resume=False,
        metadata={"family": "neural", "route": "tiny_transformer", "head": "classification"},
    )

    def __init__(self, data: NumericDataView, *, head_name: str = "classification", use_valid_objective: bool = True) -> None:
        self.data = data
        self.head_name = str(head_name)
        self.use_valid_objective = bool(use_valid_objective)

    def compute_backend_loss(
        self,
        model: Any,
        state: UnknownState,
        context: Mapping[str, Any],
        *,
        differentiable: bool = True,
    ) -> BackendLossEvaluation:
        _ = state
        backend = _backend(context, ("tensor.token_ids", "tensor.class_labels", "loss.cross_entropy", "metrics.classification"))
        device = backend.tensor.device(context)
        if differentiable:
            backend.autograd.train(model, device=device)
        else:
            backend.autograd.eval(model, device=device)
        train_context = nullcontext() if differentiable else backend.autograd.no_grad()
        with train_context:
            X_train = backend.tensor.token_ids(self.data.X_train, device=device)
            y_train = backend.tensor.class_labels(self.data.y_train, device=device)
            train_loss, train_logits = backend.losses.classification_loss(model, X_train, y_train, self.head_name)
            train_metrics = backend.losses.classification_metrics(train_logits, y_train, prefix="train")

        metrics = dict(train_metrics)
        objective_loss = backend.losses.scalar(train_loss)
        objective_prefix = "train"
        if self.data.X_valid is not None and self.data.y_valid is not None:
            backend.autograd.eval(model)
            with backend.autograd.no_grad():
                X_valid = backend.tensor.token_ids(self.data.X_valid, device=device)
                y_valid = backend.tensor.class_labels(self.data.y_valid, device=device)
                valid_loss, valid_logits = backend.losses.classification_loss(model, X_valid, y_valid, self.head_name)
                valid_metrics = backend.losses.classification_metrics(valid_logits, y_valid, prefix="valid")
            metrics.update(valid_metrics)
            if self.use_valid_objective:
                objective_loss = backend.losses.scalar(valid_loss)
                objective_prefix = "valid"
        return BackendLossEvaluation(
            objectives=np.asarray([objective_loss], dtype=float),
            loss=train_loss,
            loss_value=float(objective_loss),
            metrics=metrics,
            signals={"task": "classification", "head": self.head_name, "primary_prefix": objective_prefix},
        )

    def evaluate(self, model: Any, state: UnknownState, context: Mapping[str, Any]) -> Feedback:
        return self.compute_backend_loss(model, state, context, differentiable=False).as_feedback()

    def build_model_artifact(self, model: Any, context: Mapping[str, Any] | None = None) -> NeuralGraphArtifact:
        return _build_neural_graph_artifact(
            name=self.name,
            model=model,
            head="classification",
            head_artifact={"head": self.head_name, "task": "classification"},
            data=self.data,
            context=dict(context or {}),
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "family": "neural",
            "route": "tiny_transformer",
            "head": "classification",
            "n_train": int(self.data.X_train.shape[0]),
            "sequence_length": int(self.data.X_train.shape[1]),
            "has_valid": self.data.X_valid is not None,
        }


class TinyTransformerLanguageModelProblem(LearningProblem):
    """Evaluate tiny Transformer LM heads on next-token prediction."""

    name = "tiny_transformer_language_model"
    backend_requires = (*_BACKEND_MODE_REQUIREMENTS, "tensor.token_ids", "loss.lm_next_token")
    context_requires = ("candidate.model", "data.X_train")
    context_optional = ("data.X_valid", "resource.device")
    context_provides = _NEURAL_EVALUATION_PROVIDES
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = _DIFFERENTIABLE_LOSS_NOTE
    contract = ComponentContract(
        name=name,
        requires=("candidate.model", "data.X_train"),
        optional=("data.X_valid", "resource.device"),
        provides=context_provides,
        supports_gradient=True,
        supports_batch=False,
        supports_resume=False,
        metadata={"family": "neural", "route": "tiny_transformer", "head": "language_modeling"},
    )

    def __init__(self, data: NumericDataView, *, head_name: str = "lm", use_valid_objective: bool = True) -> None:
        self.data = data
        self.head_name = str(head_name)
        self.use_valid_objective = bool(use_valid_objective)

    def compute_backend_loss(
        self,
        model: Any,
        state: UnknownState,
        context: Mapping[str, Any],
        *,
        differentiable: bool = True,
    ) -> BackendLossEvaluation:
        _ = state
        backend = _backend(context, ("tensor.token_ids", "loss.lm_next_token"))
        device = backend.tensor.device(context)
        if differentiable:
            backend.autograd.train(model, device=device)
        else:
            backend.autograd.eval(model, device=device)
        train_context = nullcontext() if differentiable else backend.autograd.no_grad()
        with train_context:
            tokens = backend.tensor.token_ids(self.data.X_train, device=device)
            train_loss = backend.losses.lm_loss(model, tokens, self.head_name)
            train_value = backend.losses.scalar(train_loss)
        metrics = {"train.cross_entropy": train_value, "train.perplexity": backend.losses.perplexity(train_value)}
        objective_loss = train_value
        objective_prefix = "train"
        if self.data.X_valid is not None:
            backend.autograd.eval(model)
            with backend.autograd.no_grad():
                valid_tokens = backend.tensor.token_ids(self.data.X_valid, device=device)
                valid_loss = backend.losses.lm_loss(model, valid_tokens, self.head_name)
            valid_value = backend.losses.scalar(valid_loss)
            metrics.update({"valid.cross_entropy": valid_value, "valid.perplexity": backend.losses.perplexity(valid_value)})
            if self.use_valid_objective:
                objective_loss = valid_value
                objective_prefix = "valid"
        return BackendLossEvaluation(
            objectives=np.asarray([objective_loss], dtype=float),
            loss=train_loss,
            loss_value=float(objective_loss),
            metrics=metrics,
            signals={"task": "language_modeling", "head": self.head_name, "primary_prefix": objective_prefix},
        )

    def evaluate(self, model: Any, state: UnknownState, context: Mapping[str, Any]) -> Feedback:
        return self.compute_backend_loss(model, state, context, differentiable=False).as_feedback()

    def build_model_artifact(self, model: Any, context: Mapping[str, Any] | None = None) -> NeuralGraphArtifact:
        return _build_neural_graph_artifact(
            name=self.name,
            model=model,
            head="language_modeling",
            head_artifact={"head": self.head_name, "task": "language_modeling"},
            data=self.data,
            context=dict(context or {}),
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "family": "neural",
            "route": "tiny_transformer",
            "head": "language_modeling",
            "n_train": int(self.data.X_train.shape[0]),
            "sequence_length": int(self.data.X_train.shape[1]),
            "has_valid": self.data.X_valid is not None,
        }


class TinyTransformerDPOPreferenceProblem(LearningProblem):
    """DPO-style preference loss over chosen/rejected token sequences."""

    name = "tiny_transformer_dpo_preference"
    backend_requires = (*_BACKEND_MODE_REQUIREMENTS, "tensor.token_ids", "loss.dpo")
    context_requires = ("candidate.model", "data.preference_pairs")
    context_optional = ("resource.device", "preference.reference_model")
    context_provides = _NEURAL_EVALUATION_PROVIDES
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = _DIFFERENTIABLE_LOSS_NOTE
    contract = ComponentContract(
        name=name,
        requires=("candidate.model", "data.preference_pairs"),
        optional=("resource.device", "preference.reference_model"),
        provides=context_provides,
        supports_gradient=True,
        supports_batch=False,
        supports_resume=False,
        metadata={"family": "neural", "route": "tiny_transformer", "head": "preference_dpo"},
    )

    def __init__(
        self,
        data: PreferencePairDataView,
        *,
        head_name: str = "lm",
        beta: float = 0.1,
        reference_model: Any | None = None,
        use_valid_objective: bool = True,
    ) -> None:
        self.data = data
        self.head_name = str(head_name)
        self.beta = float(beta)
        self.reference_model = reference_model
        self.use_valid_objective = bool(use_valid_objective)

    def compute_backend_loss(
        self,
        model: Any,
        state: UnknownState,
        context: Mapping[str, Any],
        *,
        differentiable: bool = True,
    ) -> BackendLossEvaluation:
        _ = state
        backend = _backend(context, ("tensor.token_ids", "loss.dpo"))
        device = backend.tensor.device(context)
        if differentiable:
            backend.autograd.train(model, device=device)
        else:
            backend.autograd.eval(model, device=device)
        reference = context.get("preference.reference_model", self.reference_model)
        if reference is not None:
            backend.autograd.eval(reference, device=device)
        train_context = nullcontext() if differentiable else backend.autograd.no_grad()
        with train_context:
            chosen = backend.tensor.token_ids(self.data.chosen_train, device=device)
            rejected = backend.tensor.token_ids(self.data.rejected_train, device=device)
            train_loss, train_metrics = backend.losses.dpo_loss_and_metrics(
                model,
                chosen,
                rejected,
                self.head_name,
                beta=self.beta,
                reference_model=reference,
                prefix="train",
            )
        metrics = dict(train_metrics)
        objective_loss = backend.losses.scalar(train_loss)
        objective_prefix = "train"
        if self.data.chosen_valid is not None and self.data.rejected_valid is not None:
            backend.autograd.eval(model)
            with backend.autograd.no_grad():
                valid_chosen = backend.tensor.token_ids(self.data.chosen_valid, device=device)
                valid_rejected = backend.tensor.token_ids(self.data.rejected_valid, device=device)
                valid_loss, valid_metrics = backend.losses.dpo_loss_and_metrics(
                    model,
                    valid_chosen,
                    valid_rejected,
                    self.head_name,
                    beta=self.beta,
                    reference_model=reference,
                    prefix="valid",
                )
            metrics.update(valid_metrics)
            if self.use_valid_objective:
                objective_loss = backend.losses.scalar(valid_loss)
                objective_prefix = "valid"
        return BackendLossEvaluation(
            objectives=np.asarray([objective_loss], dtype=float),
            loss=train_loss,
            loss_value=float(objective_loss),
            metrics=metrics,
            signals={"task": "preference_dpo", "head": self.head_name, "primary_prefix": objective_prefix},
        )

    def evaluate(self, model: Any, state: UnknownState, context: Mapping[str, Any]) -> Feedback:
        return self.compute_backend_loss(model, state, context, differentiable=False).as_feedback()

    def build_model_artifact(self, model: Any, context: Mapping[str, Any] | None = None) -> NeuralGraphArtifact:
        pseudo_data = NumericDataView(
            X_train=self.data.chosen_train,
            y_train=np.zeros(self.data.n_train, dtype=float),
        )
        return _build_neural_graph_artifact(
            name=self.name,
            model=model,
            head="preference_dpo",
            head_artifact={"head": self.head_name, "task": "preference_dpo", "beta": float(self.beta)},
            data=pseudo_data,
            context=dict(context or {}),
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "family": "neural",
            "route": "tiny_transformer",
            "head": "preference_dpo",
            "n_train": int(self.data.n_train),
            "sequence_length": int(self.data.sequence_length),
            "has_valid": self.data.chosen_valid is not None,
            "beta": float(self.beta),
        }


class TinyCNNImageClassificationProblem(LearningProblem):
    """Evaluate tiny CNN image classification heads."""

    name = "tiny_cnn_image_classification"
    backend_requires = (*_BACKEND_MODE_REQUIREMENTS, "tensor.float_tensor", "tensor.class_labels", "loss.cross_entropy", "metrics.classification")
    context_requires = ("candidate.model", "data.images", "data.y_train")
    context_optional = ("data.X_valid", "data.y_valid", "resource.device")
    context_provides = _NEURAL_EVALUATION_PROVIDES
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = _DIFFERENTIABLE_LOSS_NOTE
    contract = ComponentContract(
        name=name,
        requires=("candidate.model", "data.images", "data.y_train"),
        optional=("data.X_valid", "data.y_valid", "resource.device"),
        provides=context_provides,
        supports_gradient=True,
        supports_batch=False,
        supports_resume=False,
        metadata={"family": "neural", "route": "tiny_cnn", "head": "classification"},
    )

    def __init__(self, data: ImageDataView, *, head_name: str = "classification", use_valid_objective: bool = True) -> None:
        self.data = data
        self.head_name = str(head_name)
        self.use_valid_objective = bool(use_valid_objective)

    def compute_backend_loss(
        self,
        model: Any,
        state: UnknownState,
        context: Mapping[str, Any],
        *,
        differentiable: bool = True,
    ) -> BackendLossEvaluation:
        _ = state
        backend = _backend(context, ("tensor.float_tensor", "tensor.class_labels", "loss.cross_entropy", "metrics.classification"))
        device = backend.tensor.device(context)
        if differentiable:
            backend.autograd.train(model, device=device)
        else:
            backend.autograd.eval(model, device=device)
        train_context = nullcontext() if differentiable else backend.autograd.no_grad()
        with train_context:
            X_train = backend.tensor.float_tensor(self.data.X_train, device=device)
            y_train = backend.tensor.class_labels(self.data.y_train, device=device)
            train_loss, train_logits = backend.losses.generic_classification_loss(model(X_train), y_train, self.head_name)
            metrics = backend.losses.classification_metrics(train_logits, y_train, prefix="train")
        objective_loss = backend.losses.scalar(train_loss)
        objective_prefix = "train"
        if self.data.X_valid is not None and self.data.y_valid is not None:
            backend.autograd.eval(model)
            with backend.autograd.no_grad():
                X_valid = backend.tensor.float_tensor(self.data.X_valid, device=device)
                y_valid = backend.tensor.class_labels(self.data.y_valid, device=device)
                valid_loss, valid_logits = backend.losses.generic_classification_loss(model(X_valid), y_valid, self.head_name)
                metrics.update(backend.losses.classification_metrics(valid_logits, y_valid, prefix="valid"))
            if self.use_valid_objective:
                objective_loss = backend.losses.scalar(valid_loss)
                objective_prefix = "valid"
        return BackendLossEvaluation(
            objectives=np.asarray([objective_loss], dtype=float),
            loss=train_loss,
            loss_value=float(objective_loss),
            metrics=metrics,
            signals={"task": "image_classification", "head": self.head_name, "primary_prefix": objective_prefix},
        )

    def evaluate(self, model: Any, state: UnknownState, context: Mapping[str, Any]) -> Feedback:
        return self.compute_backend_loss(model, state, context, differentiable=False).as_feedback()

    def build_model_artifact(self, model: Any, context: Mapping[str, Any] | None = None) -> NeuralGraphArtifact:
        return _build_generic_neural_artifact(
            name=self.name,
            model=model,
            head="classification",
            task="image_classification",
            context=dict(context or {}),
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "family": "neural",
            "route": "tiny_cnn",
            "head": "classification",
            "n_train": int(self.data.X_train.shape[0]),
            "image_shape": (int(self.data.channels), int(self.data.height), int(self.data.width)),
            "has_valid": self.data.X_valid is not None,
        }


class TinyGNNGraphClassificationProblem(LearningProblem):
    """Evaluate tiny GNN graph classification heads."""

    name = "tiny_gnn_graph_classification"
    backend_requires = (*_BACKEND_MODE_REQUIREMENTS, "tensor.float_tensor", "tensor.class_labels", "loss.cross_entropy", "metrics.classification")
    context_requires = ("candidate.model", "data.graphs", "data.y_train")
    context_optional = ("resource.device",)
    context_provides = _NEURAL_EVALUATION_PROVIDES
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = _DIFFERENTIABLE_LOSS_NOTE
    contract = ComponentContract(
        name=name,
        requires=("candidate.model", "data.graphs", "data.y_train"),
        optional=("resource.device",),
        provides=context_provides,
        supports_gradient=True,
        supports_batch=False,
        supports_resume=False,
        metadata={"family": "neural", "route": "tiny_gnn", "head": "classification"},
    )

    def __init__(self, data: GraphDataView, *, head_name: str = "classification", use_valid_objective: bool = True) -> None:
        self.data = data
        self.head_name = str(head_name)
        self.use_valid_objective = bool(use_valid_objective)

    def compute_backend_loss(
        self,
        model: Any,
        state: UnknownState,
        context: Mapping[str, Any],
        *,
        differentiable: bool = True,
    ) -> BackendLossEvaluation:
        _ = state
        backend = _backend(context, ("tensor.float_tensor", "tensor.class_labels", "loss.cross_entropy", "metrics.classification"))
        device = backend.tensor.device(context)
        if differentiable:
            backend.autograd.train(model, device=device)
        else:
            backend.autograd.eval(model, device=device)
        train_context = nullcontext() if differentiable else backend.autograd.no_grad()
        with train_context:
            X_train = backend.tensor.float_tensor(self.data.node_features_train, device=device)
            A_train = backend.tensor.float_tensor(self.data.adjacency_train, device=device)
            y_train = backend.tensor.class_labels(self.data.y_train, device=device)
            train_loss, train_logits = backend.losses.generic_classification_loss(model(X_train, A_train), y_train, self.head_name)
            metrics = backend.losses.classification_metrics(train_logits, y_train, prefix="train")
        objective_loss = backend.losses.scalar(train_loss)
        objective_prefix = "train"
        if self.data.node_features_valid is not None and self.data.adjacency_valid is not None and self.data.y_valid is not None:
            backend.autograd.eval(model)
            with backend.autograd.no_grad():
                X_valid = backend.tensor.float_tensor(self.data.node_features_valid, device=device)
                A_valid = backend.tensor.float_tensor(self.data.adjacency_valid, device=device)
                y_valid = backend.tensor.class_labels(self.data.y_valid, device=device)
                valid_loss, valid_logits = backend.losses.generic_classification_loss(model(X_valid, A_valid), y_valid, self.head_name)
                metrics.update(backend.losses.classification_metrics(valid_logits, y_valid, prefix="valid"))
            if self.use_valid_objective:
                objective_loss = backend.losses.scalar(valid_loss)
                objective_prefix = "valid"
        return BackendLossEvaluation(
            objectives=np.asarray([objective_loss], dtype=float),
            loss=train_loss,
            loss_value=float(objective_loss),
            metrics=metrics,
            signals={"task": "graph_classification", "head": self.head_name, "primary_prefix": objective_prefix},
        )

    def evaluate(self, model: Any, state: UnknownState, context: Mapping[str, Any]) -> Feedback:
        return self.compute_backend_loss(model, state, context, differentiable=False).as_feedback()

    def build_model_artifact(self, model: Any, context: Mapping[str, Any] | None = None) -> NeuralGraphArtifact:
        return _build_generic_neural_artifact(
            name=self.name,
            model=model,
            head="classification",
            task="graph_classification",
            context=dict(context or {}),
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "family": "neural",
            "route": "tiny_gnn",
            "head": "classification",
            "n_train": int(self.data.node_features_train.shape[0]),
            "num_nodes": int(self.data.num_nodes),
            "node_feature_dim": int(self.data.node_feature_dim),
            "has_valid": self.data.node_features_valid is not None,
        }


class TinyCNNImageContrastiveProblem(LearningProblem):
    """Triplet contrastive/retrieval loss over tiny CNN embedding heads."""

    name = "tiny_cnn_image_contrastive"
    backend_requires = (*_BACKEND_MODE_REQUIREMENTS, "tensor.float_tensor", "loss.triplet")
    context_requires = ("candidate.model", "data.image_pairs")
    context_optional = ("resource.device",)
    context_provides = _NEURAL_EVALUATION_PROVIDES
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = _DIFFERENTIABLE_LOSS_NOTE
    contract = ComponentContract(
        name=name,
        requires=("candidate.model", "data.image_pairs"),
        optional=("resource.device",),
        provides=context_provides,
        supports_gradient=True,
        supports_batch=False,
        supports_resume=False,
        metadata={"family": "neural", "route": "tiny_cnn", "head": "retrieval"},
    )

    def __init__(self, data: ImageContrastivePairDataView, *, head_name: str = "retrieval", margin: float = 0.5) -> None:
        self.data = data
        self.head_name = str(head_name)
        self.margin = float(margin)

    def compute_backend_loss(
        self,
        model: Any,
        state: UnknownState,
        context: Mapping[str, Any],
        *,
        differentiable: bool = True,
    ) -> BackendLossEvaluation:
        _ = state
        backend = _backend(context, ("tensor.float_tensor", "loss.triplet"))
        device = backend.tensor.device(context)
        if differentiable:
            backend.autograd.train(model, device=device)
        else:
            backend.autograd.eval(model, device=device)
        train_context = nullcontext() if differentiable else backend.autograd.no_grad()
        with train_context:
            anchor = backend.losses.embedding_from_output(model(backend.tensor.float_tensor(self.data.anchor_train, device=device)), self.head_name)
            positive = backend.losses.embedding_from_output(model(backend.tensor.float_tensor(self.data.positive_train, device=device)), self.head_name)
            negative = backend.losses.embedding_from_output(model(backend.tensor.float_tensor(self.data.negative_train, device=device)), self.head_name)
            loss, metrics = backend.losses.triplet_loss(anchor, positive, negative, margin=float(self.margin))
        value = backend.losses.scalar(loss)
        return BackendLossEvaluation(
            objectives=np.asarray([value], dtype=float),
            loss=loss,
            loss_value=float(value),
            metrics=metrics,
            signals={"task": "image_contrastive", "head": self.head_name},
        )

    def evaluate(self, model: Any, state: UnknownState, context: Mapping[str, Any]) -> Feedback:
        return self.compute_backend_loss(model, state, context, differentiable=False).as_feedback()

    def build_model_artifact(self, model: Any, context: Mapping[str, Any] | None = None) -> NeuralGraphArtifact:
        return _build_generic_neural_artifact(
            name=self.name,
            model=model,
            head="retrieval",
            task="image_contrastive",
            context=dict(context or {}),
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "family": "neural",
            "route": "tiny_cnn",
            "head": "retrieval",
            "n_train": int(self.data.anchor_train.shape[0]),
            "image_shape": (int(self.data.channels), int(self.data.height), int(self.data.width)),
            "margin": float(self.margin),
        }


def _backend(context: Mapping[str, Any], requirements: tuple[str, ...] = ()) -> Any:
    return get_compute_backend_from_context(
        context,
        (*_BACKEND_MODE_REQUIREMENTS, *tuple(str(item) for item in requirements)),
        consumer="neural problem",
    )


def _build_neural_graph_artifact(
    *,
    name: str,
    model: Any,
    head: str,
    head_artifact: Mapping[str, Any],
    data: NumericDataView,
    context: Mapping[str, Any],
) -> NeuralGraphArtifact:
    backend = _backend(context, ("parameters.summary", "artifact.neural_graph.audit"))
    describe = model.describe() if hasattr(model, "describe") else {"model_type": type(model).__name__}
    graph_spec = dict(getattr(model, "graph_spec", {}) or {})
    parameter_layout = backend.autograd.parameter_layout_summary(model)
    audit_artifact = backend.artifacts.audit_summary(model, data)
    return NeuralGraphArtifact(
        name=name,
        model=model,
        family="neural",
        head=head,
        representation={"model": describe},
        graph_spec=graph_spec,
        parameter_layout=parameter_layout,
        head_artifact=dict(head_artifact),
        audit_artifact=audit_artifact,
        graph_spec_digest=_stable_digest(graph_spec),
        parameter_layout_digest=_stable_digest(parameter_layout),
        metadata={"route": "tiny_transformer", "audit": {"available": bool(audit_artifact)}},
    )


def _build_generic_neural_artifact(
    *,
    name: str,
    model: Any,
    head: str,
    task: str,
    context: Mapping[str, Any],
) -> NeuralGraphArtifact:
    backend = _backend(context, ("parameters.summary",))
    describe = model.describe() if hasattr(model, "describe") else {"model_type": type(model).__name__}
    graph_spec = dict(getattr(model, "graph_spec", {}) or {})
    parameter_layout = backend.autograd.parameter_layout_summary(model)
    return NeuralGraphArtifact(
        name=name,
        model=model,
        family="neural",
        head=head,
        representation={"model": describe},
        graph_spec=graph_spec,
        parameter_layout=parameter_layout,
        head_artifact={"head": head, "task": task},
        graph_spec_digest=_stable_digest(graph_spec),
        parameter_layout_digest=_stable_digest(parameter_layout),
        metadata={"route": getattr(model, "route", "neural_graph")},
    )



def _stable_digest(payload: Mapping[str, Any]) -> str:
    text = json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


__all__ = [
    "BackendLossEvaluation",
    "TinyTransformerClassificationProblem",
    "TinyTransformerDPOPreferenceProblem",
    "TinyTransformerLanguageModelProblem",
    "TinyCNNImageClassificationProblem",
    "TinyCNNImageContrastiveProblem",
    "TinyGNNGraphClassificationProblem",
]
