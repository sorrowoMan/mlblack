from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.core.artifacts import NeuralGraphArtifact
from mlblack.core.backend_session import get_compute_backend_from_context
from blackbase.contracts import ComponentContract
from mlblack.core.problem import LearningProblem
from mlblack.core.types import Feedback, UnknownState
from mlblack.pipeline.data_views import GraphDataView, ImageContrastivePairDataView, ImageDataView, NumericDataView, PreferencePairDataView, TimeSeriesDataView


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
    "Computes neural loss/metrics; gradients are produced by the Evaluation Provider from compute_backend_loss()."
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


class TabularNeuralClassificationProblem(LearningProblem):
    """Differentiable classification semantics for tabular NeuralGraph models."""

    name = "tabular_neural_classification"
    backend_requires = (
        *_BACKEND_MODE_REQUIREMENTS,
        "tensor.float_tensor",
        "tensor.class_labels",
        "loss.cross_entropy",
        "metrics.classification",
    )
    context_requires = ("candidate.model", "data.X_train", "data.y_train")
    context_optional = ("data.X_valid", "data.y_valid", "resource.device")
    context_provides = _NEURAL_EVALUATION_PROVIDES
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = _DIFFERENTIABLE_LOSS_NOTE
    contract = ComponentContract(
        name=name,
        requires=context_requires,
        optional=context_optional,
        provides=context_provides,
        supports_gradient=True,
        supports_batch=False,
        supports_resume=False,
        metadata={"family": "neural", "route": "tabular", "head": "classification"},
    )

    def __init__(
        self,
        data: NumericDataView,
        *,
        head_name: str = "classification",
        use_valid_objective: bool = True,
    ) -> None:
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
        backend = _backend(
            context,
            (
                "tensor.float_tensor",
                "tensor.class_labels",
                "loss.cross_entropy",
                "metrics.classification",
            ),
        )
        device = backend.tensor.device(context)
        if differentiable:
            backend.autograd.train(model, device=device)
        else:
            backend.autograd.eval(model, device=device)
        train_context = nullcontext() if differentiable else backend.autograd.no_grad()
        with train_context:
            X_train = backend.tensor.float_tensor(self.data.X_train, device=device)
            y_train = backend.tensor.class_labels(self.data.y_train, device=device)
            train_loss, train_logits = backend.losses.generic_classification_loss(
                model(X_train),
                y_train,
                self.head_name,
            )
            metrics = backend.losses.classification_metrics(
                train_logits,
                y_train,
                prefix="train",
            )
        objective_loss = backend.losses.scalar(train_loss)
        objective_prefix = "train"
        if self.data.X_valid is not None and self.data.y_valid is not None:
            backend.autograd.eval(model)
            with backend.autograd.no_grad():
                X_valid = backend.tensor.float_tensor(self.data.X_valid, device=device)
                y_valid = backend.tensor.class_labels(self.data.y_valid, device=device)
                valid_loss, valid_logits = backend.losses.generic_classification_loss(
                    model(X_valid),
                    y_valid,
                    self.head_name,
                )
                metrics.update(
                    backend.losses.classification_metrics(
                        valid_logits,
                        y_valid,
                        prefix="valid",
                    )
                )
            if self.use_valid_objective:
                objective_loss = backend.losses.scalar(valid_loss)
                objective_prefix = "valid"
        return BackendLossEvaluation(
            # The exported flat gradient is the derivative of cross-entropy.
            # Error rate remains a metric: publishing it as a second objective
            # would falsely claim that the same gradient differentiates both
            # objectives at the Adapter boundary.
            objectives=np.asarray([objective_loss], dtype=float),
            loss=train_loss,
            loss_value=float(objective_loss),
            metrics=metrics,
            signals={
                "task": "tabular_neural_classification",
                "head": self.head_name,
                "primary_prefix": objective_prefix,
            },
        )

    def evaluate(
        self,
        model: Any,
        state: UnknownState,
        context: Mapping[str, Any],
    ) -> Feedback:
        return self.compute_backend_loss(
            model,
            state,
            context,
            differentiable=False,
        ).as_feedback()

    def build_model_artifact(
        self,
        model: Any,
        context: Mapping[str, Any] | None = None,
    ) -> NeuralGraphArtifact:
        return _build_generic_neural_artifact(
            name=self.name,
            model=model,
            head="classification",
            task="tabular_neural_classification",
            context=dict(context or {}),
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "family": "neural",
            "route": "tabular",
            "head": "classification",
            "n_train": int(self.data.X_train.shape[0]),
            "n_features": int(self.data.X_train.shape[1]),
            "has_valid": self.data.X_valid is not None,
        }


class TabularNeuralRegressionProblem(LearningProblem):
    """Differentiable point-regression semantics for tabular NeuralGraph models."""

    name = "tabular_neural_regression"
    backend_requires = (*_BACKEND_MODE_REQUIREMENTS, "tensor.float_tensor")
    context_requires = ("candidate.model", "data.X_train", "data.y_train")
    context_optional = ("data.X_valid", "data.y_valid", "resource.device")
    context_provides = _NEURAL_EVALUATION_PROVIDES
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = _DIFFERENTIABLE_LOSS_NOTE
    contract = ComponentContract(
        name=name,
        requires=context_requires,
        optional=context_optional,
        provides=context_provides,
        supports_gradient=True,
        supports_batch=False,
        supports_resume=False,
        metadata={"family": "neural", "route": "tabular", "head": "point"},
    )

    def __init__(
        self,
        data: NumericDataView,
        *,
        head_name: str = "point",
        use_valid_objective: bool = True,
    ) -> None:
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
        backend = _backend(context, ("tensor.float_tensor",))
        torch = backend.tensor.torch()
        device = backend.tensor.device(context)
        if differentiable:
            backend.autograd.train(model, device=device)
        else:
            backend.autograd.eval(model, device=device)
        train_context = nullcontext() if differentiable else backend.autograd.no_grad()
        with train_context:
            batch = context.get("data.batch")
            X_source = self.data.X_train if batch is None else batch.X
            y_source = self.data.y_train if batch is None else batch.y
            X_train = backend.tensor.float_tensor(X_source, device=device)
            y_train = backend.tensor.float_tensor(y_source, device=device)
            if y_train.ndim == 1:
                y_train = y_train.unsqueeze(-1)
            train_output = _extract_head_output(model(X_train), self.head_name)
            train_prediction = train_output.reshape(y_train.shape)
            train_loss = torch.nn.functional.mse_loss(train_prediction, y_train)
            metrics = _forecast_regression_metrics(
                y_train.detach().cpu().numpy(),
                train_prediction.detach().cpu().numpy(),
                prefix="train",
            )
        objective_loss = float(train_loss.detach().cpu().item())
        objective_prefix = "train"
        if self.data.X_valid is not None and self.data.y_valid is not None:
            backend.autograd.eval(model)
            with backend.autograd.no_grad():
                X_valid = backend.tensor.float_tensor(self.data.X_valid, device=device)
                y_valid = backend.tensor.float_tensor(self.data.y_valid, device=device)
                if y_valid.ndim == 1:
                    y_valid = y_valid.unsqueeze(-1)
                valid_output = _extract_head_output(model(X_valid), self.head_name)
                valid_prediction = valid_output.reshape(y_valid.shape)
                valid_loss = torch.nn.functional.mse_loss(valid_prediction, y_valid)
                metrics.update(
                    _forecast_regression_metrics(
                        y_valid.detach().cpu().numpy(),
                        valid_prediction.detach().cpu().numpy(),
                        prefix="valid",
                    )
                )
            if self.use_valid_objective:
                objective_loss = float(valid_loss.detach().cpu().item())
                objective_prefix = "valid"
        return BackendLossEvaluation(
            objectives=np.asarray([objective_loss], dtype=float),
            loss=train_loss,
            loss_value=float(objective_loss),
            metrics=metrics,
            signals={
                "task": "tabular_neural_regression",
                "head": self.head_name,
                "primary_prefix": objective_prefix,
            },
        )

    def evaluate(
        self,
        model: Any,
        state: UnknownState,
        context: Mapping[str, Any],
    ) -> Feedback:
        return self.compute_backend_loss(
            model,
            state,
            context,
            differentiable=False,
        ).as_feedback()

    def build_model_artifact(
        self,
        model: Any,
        context: Mapping[str, Any] | None = None,
    ) -> NeuralGraphArtifact:
        return _build_generic_neural_artifact(
            name=self.name,
            model=model,
            head="point",
            task="tabular_neural_regression",
            context=dict(context or {}),
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "family": "neural",
            "route": "tabular",
            "head": "point",
            "n_train": int(self.data.X_train.shape[0]),
            "n_features": int(self.data.X_train.shape[1]),
            "has_valid": self.data.X_valid is not None,
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


class TemporalNeuralForecastingProblem(LearningProblem):
    """Evaluate temporal neural graph models (LSTM/TCN/Transformer) for sequence forecasting."""

    name = "temporal_neural_forecasting"
    backend_requires = (*_BACKEND_MODE_REQUIREMENTS, "tensor.float_tensor")
    context_requires = ("candidate.model", "data.X_train", "data.y_train")
    context_optional = ("data.X_valid", "data.y_valid", "resource.device")
    context_provides = _NEURAL_EVALUATION_PROVIDES
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = _DIFFERENTIABLE_LOSS_NOTE
    contract = ComponentContract(
        name=name,
        requires=("candidate.model", "data.X_train", "data.y_train"),
        optional=("data.X_valid", "data.y_valid", "resource.device"),
        provides=context_provides,
        supports_gradient=True,
        supports_batch=False,
        supports_resume=False,
        metadata={"family": "neural", "route": "temporal", "head": "forecast"},
    )

    def __init__(self, data: NumericDataView, *, head_name: str = "forecast", use_valid_objective: bool = True) -> None:
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
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("TemporalNeuralForecastingProblem requires optional dependency 'torch'") from exc

        backend = _backend(context, ("tensor.float_tensor",))
        device = backend.tensor.device(context)
        if differentiable:
            backend.autograd.train(model, device=device)
        else:
            backend.autograd.eval(model, device=device)
        train_context = nullcontext() if differentiable else backend.autograd.no_grad()
        with train_context:
            X_train = backend.tensor.float_tensor(self.data.X_train, device=device)
            y_train = backend.tensor.float_tensor(self.data.y_train, device=device)
            if y_train.ndim == 1:
                y_train = y_train.unsqueeze(-1)
            output = model(X_train)
            forecast = _extract_head_output(output, self.head_name)
            train_loss = torch.nn.functional.mse_loss(forecast, y_train)
            train_metrics = _forecast_regression_metrics(
                y_train.detach().cpu().numpy(),
                forecast.detach().cpu().numpy(),
                prefix="train",
            )

        metrics = dict(train_metrics)
        objective_loss = train_loss
        objective_prefix = "train"

        if self.data.X_valid is not None and self.data.y_valid is not None:
            backend.autograd.eval(model)
            with backend.autograd.no_grad():
                X_valid = backend.tensor.float_tensor(self.data.X_valid, device=device)
                y_valid = backend.tensor.float_tensor(self.data.y_valid, device=device)
                if y_valid.ndim == 1:
                    y_valid = y_valid.unsqueeze(-1)
                output = model(X_valid)
                valid_forecast = _extract_head_output(output, self.head_name)
                valid_loss = torch.nn.functional.mse_loss(valid_forecast, y_valid)
                valid_metrics = _forecast_regression_metrics(
                    y_valid.detach().cpu().numpy(),
                    valid_forecast.detach().cpu().numpy(),
                    prefix="valid",
                )
            metrics.update(valid_metrics)
            if self.use_valid_objective:
                objective_loss = valid_loss
                objective_prefix = "valid"

        return BackendLossEvaluation(
            objectives=np.asarray([float(objective_loss.detach().cpu().item())], dtype=float),
            loss=train_loss,
            loss_value=float(objective_loss.detach().cpu().item()),
            metrics=metrics,
            signals={"task": "temporal_neural_forecasting", "head": self.head_name, "primary_prefix": objective_prefix},
        )

    def evaluate(self, model: Any, state: UnknownState, context: Mapping[str, Any]) -> Feedback:
        return self.compute_backend_loss(model, state, context, differentiable=False).as_feedback()

    def build_model_artifact(self, model: Any, context: Mapping[str, Any] | None = None) -> NeuralGraphArtifact:
        return _build_generic_neural_artifact(
            name=self.name,
            model=model,
            head="forecast",
            task="temporal_neural_forecasting",
            context=dict(context or {}),
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "family": "neural",
            "route": "temporal",
            "head": "forecast",
            "n_train": int(self.data.X_train.shape[0]),
            "input_shape": tuple(int(v) for v in self.data.X_train.shape),
            "has_valid": self.data.X_valid is not None,
        }


class TemporalNeuralRollingOriginProblem(LearningProblem):
    """Rolling-origin evaluation for temporal neural graph forecasters.

    Unlike TemporalNeuralForecastingProblem which takes pre-windowed
    NumericDataView, this Problem consumes a TimeSeriesDataView and performs
    origin-by-origin sequence extraction internally.  Each origin builds a
    fixed-length sequence from history, forwards it through the model, and
    compares the forecast against the held-out target.
    """

    name = "temporal_neural_rolling_origin"
    backend_requires = (*_BACKEND_MODE_REQUIREMENTS, "tensor.float_tensor")
    context_requires = ("candidate.model", "data.time_series_view")
    context_optional = ("resource.device", "time_series.min_train_size", "time_series.horizon")
    context_provides = _NEURAL_EVALUATION_PROVIDES
    context_mutates = ()
    context_cache = ()
    context_notes = "Rolling-origin evaluation for temporal neural models.  Not differentiable."
    contract = ComponentContract(
        name=name,
        requires=("candidate.model", "data.time_series_view"),
        optional=("resource.device", "time_series.min_train_size", "time_series.horizon"),
        provides=context_provides,
        supports_gradient=False,
        supports_batch=False,
        supports_resume=False,
        metadata={"family": "neural", "route": "temporal", "head": "forecast", "task": "rolling_origin"},
    )

    def __init__(
        self,
        data: TimeSeriesDataView,
        *,
        sequence_length: int | None = None,
        head_name: str = "forecast",
        min_train_size: int | float = 0.6,
        horizon: int = 1,
        max_origins: int | None = None,
        objective_metrics: Sequence[str] = ("rolling.rmse", "rolling.mae"),
        seasonal_period: int = 1,
    ) -> None:
        self.data = data
        self.sequence_length = None if sequence_length is None else int(sequence_length)
        self.head_name = str(head_name)
        self.min_train_size = min_train_size
        self.horizon = int(horizon)
        self.max_origins = None if max_origins is None else int(max_origins)
        self.objective_metrics = tuple(str(m) for m in objective_metrics)
        self.seasonal_period = int(seasonal_period)

    def get_num_objectives(self) -> int:
        count = len(self.objective_metrics)
        if count <= 0:
            raise ValueError(
                "TemporalNeuralRollingOriginProblem requires at least one objective metric"
            )
        return count

    def evaluate(self, model: Any, state: UnknownState, context: Mapping[str, Any]) -> Feedback:
        _ = state
        seq_len = _resolve_sequence_length(model, self.sequence_length)
        horizon = int(context.get("time_series.horizon", self.horizon))
        y = np.asarray(self.data.y, dtype=float).reshape(-1)
        min_train = _resolve_rolling_min_train(
            context.get("time_series.min_train_size", self.min_train_size),
            int(y.shape[0]),
            seq_len,
            horizon,
        )
        origins = _build_rolling_origins(y.shape[0], min_train, horizon, self.max_origins)

        try:
            __import__("torch")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("TemporalNeuralRollingOriginProblem requires optional dependency 'torch'") from exc

        backend = _backend(context, ("tensor.float_tensor",))
        device = backend.tensor.device(context)
        backend.autograd.eval(model, device=device)

        preds: list[float] = []
        targets: list[float] = []
        with backend.autograd.no_grad():
            for origin in origins:
                start = max(0, origin - seq_len + 1)
                window = y[start : origin + 1]
                if window.shape[0] < seq_len:
                    continue
                seq = np.asarray(window[-seq_len:], dtype=float).reshape(1, seq_len, 1)
                seq_tensor = backend.tensor.float_tensor(seq, device=device)
                output = model(seq_tensor)
                forecast = _extract_head_output(output, self.head_name)
                preds.append(float(forecast.detach().cpu().numpy().reshape(-1)[-1]))
                targets.append(float(y[origin + horizon]))

        pred = np.asarray(preds, dtype=float)
        target = np.asarray(targets, dtype=float)
        train_history = np.asarray(y[:min_train], dtype=float)
        metrics = _forecast_rolling_metrics(
            target, pred,
            train_history=train_history,
            prefix="rolling",
            seasonal_period=self.seasonal_period,
        )
        residual = pred - target
        metrics.update(
            {
                "rolling.origins": int(len(origins)),
                "rolling.horizon": int(horizon),
                "rolling.min_train_size": int(min_train),
            }
        )
        objectives = [float(metrics[m]) for m in self.objective_metrics]
        return Feedback(
            objectives=np.asarray(objectives, dtype=float),
            constraints=np.zeros(0, dtype=float),
            loss=float(objectives[0]) if objectives else float(metrics.get("rolling.rmse", 0.0)),
            gradients=None,
            residuals=residual,
            metrics=metrics,
            signals={
                "task": "temporal_neural_rolling_origin",
                "head": self.head_name,
                "has_gradient": False,
                "origins": int(len(origins)),
                "horizon": int(horizon),
            },
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "family": "neural",
            "route": "temporal",
            "head": "forecast",
            "task": "rolling_origin",
            "n_obs": int(self.data.n_obs),
            "min_train_size": self.min_train_size,
            "horizon": int(self.horizon),
            "max_origins": self.max_origins,
        }


class TemporalNeuralProbabilisticForecastingProblem(LearningProblem):
    """Evaluate temporal neural graph models with negative log-likelihood loss for probabilistic forecasting.

    Expects model output to be a dict with ``mu`` and ``log_sigma`` keys (Gaussian
    distribution parameters). Computes NLL, CRPS, prediction interval coverage,
    and RMSE as auxiliary metrics.
    """

    name = "temporal_neural_probabilistic_forecast"
    backend_requires = (*_BACKEND_MODE_REQUIREMENTS, "tensor.float_tensor", "loss.gaussian_nll")
    context_requires = ("candidate.model", "data.X_train", "data.y_train")
    context_optional = ("data.X_valid", "data.y_valid", "resource.device", "autograd.optim.config")
    context_provides = _NEURAL_EVALUATION_PROVIDES
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = _DIFFERENTIABLE_LOSS_NOTE
    contract = ComponentContract(
        name=name,
        requires=("candidate.model", "data.X_train", "data.y_train"),
        optional=("data.X_valid", "data.y_valid", "resource.device"),
        provides=_NEURAL_EVALUATION_PROVIDES,
        supports_gradient=True,
        supports_batch=True,
        supports_resume=True,
        metadata={
            "family": "neural",
            "route": "temporal",
            "head": "deepar",
            "task": "probabilistic_forecasting",
            "loss": "gaussian_nll",
        },
    )

    def __init__(
        self,
        data: NumericDataView,
        *,
        head_name: str = "deepar",
        alpha: float = 0.1,
        use_valid_objective: bool = True,
    ) -> None:
        super().__init__()
        self.data = data
        self.head_name = str(head_name)
        self.alpha = float(alpha)
        self.use_valid_objective = bool(use_valid_objective)

    def compute_backend_loss(
        self,
        model: Any,
        state: UnknownState | None,
        context: Mapping[str, Any],
        *,
        differentiable: bool = True,
    ) -> BackendLossEvaluation:
        _ = state
        backend = _backend(context, ("tensor.float_tensor", "loss.gaussian_nll"))
        device = backend.tensor.device(context)
        if differentiable:
            backend.autograd.train(model, device=device)
        else:
            backend.autograd.eval(model, device=device)
        evaluation_context = nullcontext() if differentiable else backend.autograd.no_grad()
        X_train = backend.tensor.float_tensor(self.data.X_train, device=device)
        y_train = backend.tensor.float_tensor(self.data.y_train, device=device)
        with evaluation_context:
            output = model(X_train)
            loss, mu, sigma = backend.losses.gaussian_nll(output, y_train, self.head_name)
        if not differentiable:
            metrics = _probabilistic_forecast_metrics(
                y_train.detach().cpu().numpy().reshape(-1),
                mu.detach().cpu().numpy().reshape(-1),
                sigma.detach().cpu().numpy().reshape(-1),
                alpha=self.alpha,
                prefix="train",
            )
        else:
            metrics = {}
        valid_loss = None
        valid_metrics = {}
        X_valid = getattr(self.data, "X_valid", None)
        y_valid = getattr(self.data, "y_valid", None)
        if X_valid is not None and y_valid is not None and len(X_valid) > 0 and len(y_valid) > 0:
            X_val = backend.tensor.float_tensor(X_valid, device=device)
            y_val = backend.tensor.float_tensor(y_valid, device=device)
            with backend.autograd.no_grad():
                val_output = model(X_val)
                valid_nll_tensor, val_mu, val_sigma = backend.losses.gaussian_nll(val_output, y_val, self.head_name)
            if val_mu is not None and val_sigma is not None:
                valid_nll = valid_nll_tensor
                valid_loss = float(valid_nll.detach().cpu())
                valid_metrics = _probabilistic_forecast_metrics(
                    y_val.detach().cpu().numpy().reshape(-1),
                    val_mu.detach().cpu().numpy().reshape(-1),
                    val_sigma.detach().cpu().numpy().reshape(-1),
                    alpha=self.alpha,
                    prefix="valid",
                )
        loss_value = float(loss.detach().cpu())
        all_metrics = {**metrics, **valid_metrics}
        if valid_loss is not None:
            all_metrics["valid.nll"] = valid_loss
        return BackendLossEvaluation(
            objectives=np.array([valid_loss if (self.use_valid_objective and valid_loss is not None) else loss_value], dtype=float),
            loss=loss,
            loss_value=loss_value,
            metrics=all_metrics,
            signals={
                "has_gradient": True,
                "device": str(device),
                "head": "deepar",
                "loss_type": "gaussian_nll",
                "task": "probabilistic_forecasting",
            },
        )

    def evaluate(
        self,
        model: Any,
        state: UnknownState | None,
        context: Mapping[str, Any] | None = None,
    ) -> Feedback:
        return self.compute_backend_loss(model, state, dict(context or {}), differentiable=False).as_feedback()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": "neural",
            "route": "temporal",
            "head": self.head_name,
            "task": "probabilistic_forecasting",
            "alpha": float(self.alpha),
            "use_valid_objective": bool(self.use_valid_objective),
        }


def _probabilistic_forecast_metrics(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    alpha: float,
    prefix: str,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(mu, dtype=float).reshape(-1)
    s = np.maximum(np.asarray(sigma, dtype=float).reshape(-1), 1e-8)
    err = p - y
    # NLL
    nll = float(np.mean(0.5 * np.log(2 * np.pi) + np.log(s) + 0.5 * (err / s) ** 2))
    # CRPS (Gaussian closed form)
    crps = float(np.mean(_crps_normal(mu=p, sigma=s, y=y)))
    # Prediction interval coverage
    from scipy.stats import norm as scipy_norm
    z_alpha = float(scipy_norm.ppf(1.0 - alpha / 2.0))
    lower = p - z_alpha * s
    upper = p + z_alpha * s
    covered = np.sum((y >= lower) & (y <= upper))
    picp = float(covered) / float(max(y.shape[0], 1))
    # RMSE
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {
        f"{prefix}.nll": nll,
        f"{prefix}.crps": crps,
        f"{prefix}.picp": picp,
        f"{prefix}.rmse": rmse,
        f"{prefix}.mae": float(np.mean(np.abs(err))),
    }


def _crps_normal(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.stats import norm as scipy_norm
    m = np.asarray(mu, dtype=float).reshape(-1)
    s = np.maximum(np.asarray(sigma, dtype=float).reshape(-1), 1e-8)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    z = (y_arr - m) / s
    return s * (z * (2.0 * scipy_norm.cdf(z) - 1.0) + 2.0 * scipy_norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def _resolve_sequence_length(model: Any, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    sl = getattr(model, "sequence_length", None)
    if sl is not None:
        return int(sl)
    raise ValueError("sequence_length must be specified or available from model.sequence_length")


def _resolve_rolling_min_train(value: int | float, n_obs: int, seq_len: int, horizon: int) -> int:
    minimum = seq_len + horizon
    if isinstance(value, float) and 0.0 < float(value) < 1.0:
        size = max(minimum, int(round(float(value) * float(n_obs))))
    else:
        size = max(minimum, int(value))
    if size >= n_obs:
        raise ValueError("min_train_size must be smaller than series length")
    return size


def _build_rolling_origins(n_obs: int, min_train: int, horizon: int, max_origins: int | None) -> list[int]:
    last = n_obs - horizon - 1
    if last < min_train - 1:
        raise ValueError("series is too short for rolling-origin evaluation")
    origins = list(range(min_train - 1, last + 1))
    if max_origins is not None and max_origins > 0 and len(origins) > max_origins:
        idx = np.linspace(0, len(origins) - 1, num=max_origins, dtype=int)
        origins = [origins[int(i)] for i in idx]
    return origins


def _forecast_rolling_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    train_history: np.ndarray,
    prefix: str,
    seasonal_period: int,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    err = p - y
    abs_err = np.abs(err)
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(abs_err))
    denom = np.maximum(np.abs(y), 1e-12)
    mape = float(np.mean(abs_err / denom))
    smape = float(np.mean((2.0 * abs_err) / np.maximum(np.abs(y) + np.abs(p), 1e-12)))
    scale = _mase_scale(train_history, seasonal_period=seasonal_period)
    mase = float(mae / scale) if scale > 0.0 else (0.0 if mae <= 1e-12 else float("inf"))
    bias = float(np.mean(err))
    return {
        f"{prefix}.mse": mse,
        f"{prefix}.rmse": float(np.sqrt(mse)),
        f"{prefix}.mae": mae,
        f"{prefix}.mape": mape,
        f"{prefix}.smape": smape,
        f"{prefix}.mase": mase,
        f"{prefix}.bias": bias,
    }


def _mase_scale(train_history: np.ndarray, *, seasonal_period: int) -> float:
    y = np.asarray(train_history, dtype=float).reshape(-1)
    lag = max(1, int(seasonal_period))
    if y.shape[0] <= lag:
        return float(np.mean(np.abs(np.diff(y)))) if y.shape[0] > 1 else 0.0
    return float(np.mean(np.abs(y[lag:] - y[:-lag])))


def _extract_head_output(output: Any, head_name: str) -> Any:
    if isinstance(output, dict):
        head_outputs = output.get("head_outputs", {})
        if head_name in head_outputs:
            return head_outputs[head_name]
        if head_name in output:
            return output[head_name]
        for key in ("forecast", "logits"):
            val = output.get(key)
            if val is not None:
                return val
    return output


def _forecast_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, prefix: str) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    if y.shape[0] != p.shape[0]:
        raise ValueError("forecast and target lengths differ")
    err = p - y
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    return {
        f"{prefix}.mse": mse,
        f"{prefix}.rmse": float(np.sqrt(mse)),
        f"{prefix}.mae": mae,
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
    "TemporalNeuralForecastingProblem",
    "TemporalNeuralProbabilisticForecastingProblem",
    "TemporalNeuralRollingOriginProblem",
    "TinyTransformerClassificationProblem",
    "TinyTransformerDPOPreferenceProblem",
    "TinyTransformerLanguageModelProblem",
    "TinyCNNImageClassificationProblem",
    "TinyCNNImageContrastiveProblem",
    "TinyGNNGraphClassificationProblem",
]
