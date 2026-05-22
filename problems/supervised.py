from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.core.backend_session import get_compute_backend_from_context
from mlblack.core.contracts import ComponentContract
from mlblack.core.problem import LearningProblem
from mlblack.core.types import Feedback, UnknownState
from mlblack.pipeline.data import NumericDataView
from mlblack.models import EstimatorSpecModel, FittedEstimatorModel


class SupervisedRegressionProblem(LearningProblem):
    """Data-dependent evaluator for point regression."""

    name = "supervised_regression"
    context_requires = ('candidate.model', 'data.X_train', 'data.y_train')
    context_optional = ('data.X_valid', 'data.y_valid', 'model.parameter_gradient')
    context_provides = ('feedback.objectives', 'feedback.loss', 'feedback.metrics', 'feedback.residuals', 'feedback.gradients', 'feedback.signals')
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: candidate.model, data.X_train, data.y_train; provides feedback.objectives, feedback.loss, feedback.metrics, feedback.residuals, feedback.gradients, feedback.signals.'
    contract = ComponentContract(
        name=name,
        requires=("candidate.model", "data.X_train", "data.y_train"),
        optional=("data.X_valid", "data.y_valid", "model.parameter_gradient"),
        provides=(
            "feedback.objectives",
            "feedback.loss",
            "feedback.metrics",
            "feedback.residuals",
            "feedback.gradients",
            "feedback.signals",
        ),
        supports_gradient=True,
        supports_batch=False,
        supports_resume=False,
    )

    def __init__(
        self,
        data: NumericDataView,
        *,
        l2: float = 0.0,
        complexity_weight: float = 0.0,
        use_valid_objective: bool = True,
    ) -> None:
        self.data = data
        self.l2 = float(l2)
        self.complexity_weight = float(complexity_weight)
        self.use_valid_objective = bool(use_valid_objective)

    def evaluate(self, model: Any, state: UnknownState, context: Mapping[str, Any]) -> Feedback:
        _ = context
        train_pred = np.asarray(model.predict(self.data.X_train), dtype=float).reshape(-1)
        train_metrics = _regression_metrics(self.data.y_train, train_pred, prefix="train")
        train_residual = train_pred - self.data.y_train

        metrics: dict[str, Any] = dict(train_metrics)
        objective_mse = float(train_metrics["train.mse"])
        valid_residual = None
        if self.data.X_valid is not None and self.data.y_valid is not None:
            valid_pred = np.asarray(model.predict(self.data.X_valid), dtype=float).reshape(-1)
            valid_metrics = _regression_metrics(self.data.y_valid, valid_pred, prefix="valid")
            metrics.update(valid_metrics)
            valid_residual = valid_pred - self.data.y_valid
            if self.use_valid_objective:
                objective_mse = float(valid_metrics["valid.mse"])

        values = state.as_array()
        weights = values[1:] if values.shape[0] > 1 else values
        l2_norm = float(np.sum(weights ** 2))
        complexity = float(np.count_nonzero(np.abs(weights) > 1e-10))
        regularized_loss = float(train_metrics["train.mse"] + (self.l2 * l2_norm))

        gradient = None
        grad_fn = getattr(model, "parameter_gradient", None)
        if callable(grad_fn):
            gradient = np.asarray(grad_fn(self.data.X_train, self.data.y_train, l2=self.l2), dtype=float).reshape(-1)

        metrics.update(
            {
                "loss.regularized_train": regularized_loss,
                "complexity.nonzero": complexity,
                "complexity.l2_norm": l2_norm,
            }
        )
        objectives = np.asarray(
            [
                float(objective_mse),
                float(self.complexity_weight) * complexity,
            ],
            dtype=float,
        )
        signals = {
            "primary_objective": "valid.mse" if self.data.X_valid is not None and self.use_valid_objective else "train.mse",
            "has_gradient": gradient is not None,
        }
        return Feedback(
            objectives=objectives,
            constraints=np.zeros(0, dtype=float),
            loss=regularized_loss,
            gradients=gradient,
            residuals=train_residual if valid_residual is None else valid_residual,
            metrics=metrics,
            signals=signals,
        )

    def compute_functional_gradient(
        self,
        model: Any,
        state: UnknownState,
        context: Mapping[str, Any],
        *,
        backend: Any | None = None,
    ) -> np.ndarray:
        _ = state
        active_backend = backend or get_compute_backend_from_context(
            context,
            ("autograd.functional.grad", "autograd.gradients.flat_export"),
            consumer=f"{self.name}.compute_functional_gradient",
        )
        return active_backend.autograd.mse_parameter_gradient(
            model,
            self.data.X_train,
            self.data.y_train,
            l2=self.l2,
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "l2": float(self.l2),
            "complexity_weight": float(self.complexity_weight),
            "n_train": int(self.data.X_train.shape[0]),
            "n_features": int(self.data.X_train.shape[1]),
            "has_valid": self.data.X_valid is not None,
        }


class SupervisedEstimatorFitRegressionProblem(SupervisedRegressionProblem):
    """Evaluator that fits decoded external estimator specs before scoring."""

    name = "supervised_estimator_fit_regression"
    context_requires = ('candidate.model_spec', 'estimator.factory', 'data.X_train', 'data.y_train')
    context_optional = ('data.X_valid', 'data.y_valid')
    context_provides = ('feedback.objectives', 'feedback.loss', 'feedback.metrics', 'feedback.residuals', 'feedback.signals', 'fitted_estimator')
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: candidate.model_spec, estimator.factory, data.X_train, data.y_train; provides feedback.objectives, feedback.loss, feedback.metrics, feedback.residuals, feedback.signals, fitted_estimator.'
    contract = ComponentContract(
        name=name,
        requires=("candidate.model_spec", "estimator.factory", "data.X_train", "data.y_train"),
        optional=("data.X_valid", "data.y_valid"),
        provides=(
            "feedback.objectives",
            "feedback.loss",
            "feedback.metrics",
            "feedback.residuals",
            "feedback.signals",
            "fitted_estimator",
        ),
        supports_gradient=False,
        supports_batch=False,
        supports_resume=False,
    )

    def evaluate(self, model: Any, state: UnknownState, context: Mapping[str, Any]) -> Feedback:
        if not isinstance(model, EstimatorSpecModel):
            return super().evaluate(model, state, context)

        model = _clamp_estimator_spec_to_resource(model, context)
        estimator = model.build_estimator()
        if not hasattr(estimator, "fit"):
            raise TypeError("decoded estimator does not expose fit(...)")
        fit_info = _fit_estimator_with_lifecycle(estimator, model, self.data, context)
        train_pred = np.asarray(estimator.predict(self.data.X_train), dtype=float).reshape(-1)
        train_metrics = _regression_metrics(self.data.y_train, train_pred, prefix="train")
        train_residual = train_pred - self.data.y_train

        metrics: dict[str, Any] = dict(train_metrics)
        objective_mse = float(train_metrics["train.mse"])
        residual = train_residual
        if self.data.X_valid is not None and self.data.y_valid is not None:
            valid_pred = np.asarray(estimator.predict(self.data.X_valid), dtype=float).reshape(-1)
            valid_metrics = _regression_metrics(self.data.y_valid, valid_pred, prefix="valid")
            metrics.update(valid_metrics)
            residual = valid_pred - self.data.y_valid
            if self.use_valid_objective:
                objective_mse = float(valid_metrics["valid.mse"])

        complexity = _safe_estimator_complexity(estimator)
        metrics.update(
            {
                "estimator.family": model.family,
                "estimator.route": model.route,
                "estimator.type": type(estimator).__name__,
                "complexity.estimator": complexity,
                **{f"estimator.lifecycle.{key}": value for key, value in fit_info.items() if isinstance(value, (str, int, float, bool)) or value is None},
            }
        )
        context["last_fitted_estimator_spec"] = model.describe()
        context["last_fitted_estimator_lifecycle"] = fit_info
        objectives = np.asarray(
            [
                float(objective_mse),
                float(self.complexity_weight) * float(complexity),
            ],
            dtype=float,
        )
        return Feedback(
            objectives=objectives,
            constraints=np.zeros(0, dtype=float),
            loss=float(train_metrics["train.mse"]),
            gradients=None,
            residuals=residual,
            metrics=metrics,
            signals={
                "primary_objective": "valid.mse" if self.data.X_valid is not None and self.use_valid_objective else "train.mse",
                "has_gradient": False,
                "fitted_estimator": True,
                "fitted_estimator_type": type(estimator).__name__,
            },
        )

    def describe(self) -> Mapping[str, Any]:
        base = dict(super().describe())
        base["name"] = self.name
        base["fits_decoded_estimator"] = True
        return base

    def build_model_artifact(self, model: Any, context: Mapping[str, Any] | None = None) -> Any:
        if not isinstance(model, EstimatorSpecModel):
            return model
        estimator = model.build_estimator()
        _fit_estimator_with_lifecycle(estimator, model, self.data, dict(context or {}))
        return FittedEstimatorModel(
            estimator=estimator,
            family=model.family,
            route=model.route,
            params=dict(model.params),
            metadata={"source": "SupervisedEstimatorFitRegressionProblem.build_model_artifact"},
        )


class SupervisedIntervalRegressionProblem(LearningProblem):
    """Evaluator for decoded interval-output models."""

    name = "supervised_interval_regression"
    context_requires = ('candidate.interval_model', 'model.predict_interval', 'data.X_train', 'data.y_train')
    context_optional = ('data.X_valid', 'data.y_valid')
    context_provides = ('feedback.objectives', 'feedback.metrics', 'feedback.residuals', 'feedback.constraints', 'feedback.signals')
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: candidate.interval_model, model.predict_interval, data.X_train, data.y_train; provides feedback.objectives, feedback.metrics, feedback.residuals, feedback.constraints, feedback.signals.'
    contract = ComponentContract(
        name=name,
        requires=("candidate.interval_model", "model.predict_interval", "data.X_train", "data.y_train"),
        optional=("data.X_valid", "data.y_valid"),
        provides=(
            "feedback.objectives",
            "feedback.metrics",
            "feedback.residuals",
            "feedback.constraints",
            "feedback.signals",
        ),
        supports_gradient=False,
        supports_batch=False,
        supports_resume=False,
        metadata={"head": "interval"},
    )

    def __init__(
        self,
        data: NumericDataView,
        *,
        target_coverage: float = 0.9,
        width_weight: float = 1.0,
        miss_weight: float = 10.0,
        use_valid_objective: bool = True,
    ) -> None:
        self.data = data
        self.target_coverage = float(target_coverage)
        self.width_weight = float(width_weight)
        self.miss_weight = float(miss_weight)
        self.use_valid_objective = bool(use_valid_objective)

    def evaluate(self, model: Any, state: UnknownState, context: Mapping[str, Any]) -> Feedback:
        _ = state
        _ = context
        train_metrics, train_residual = _interval_metrics(model, self.data.X_train, self.data.y_train, prefix="train")
        metrics: dict[str, Any] = dict(train_metrics)
        objective_metrics = train_metrics
        residual = train_residual

        if self.data.X_valid is not None and self.data.y_valid is not None:
            valid_metrics, valid_residual = _interval_metrics(model, self.data.X_valid, self.data.y_valid, prefix="valid")
            metrics.update(valid_metrics)
            residual = valid_residual
            if self.use_valid_objective:
                objective_metrics = valid_metrics

        prefix = "valid" if objective_metrics is not train_metrics else "train"
        coverage = float(objective_metrics[f"{prefix}.coverage"])
        width = float(objective_metrics[f"{prefix}.mean_width"])
        miss = float(objective_metrics[f"{prefix}.mean_miss_distance"])
        objective = (self.width_weight * width) + (self.miss_weight * miss)
        constraints = np.asarray([self.target_coverage - coverage], dtype=float)
        return Feedback(
            objectives=np.asarray([float(objective), float(width), float(miss)], dtype=float),
            constraints=constraints,
            loss=float(objective),
            gradients=None,
            residuals=residual,
            metrics=metrics,
            signals={
                "head": "interval",
                "primary_objective": f"{prefix}.interval_objective",
                "target_coverage": self.target_coverage,
                "has_gradient": False,
            },
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "target_coverage": float(self.target_coverage),
            "width_weight": float(self.width_weight),
            "miss_weight": float(self.miss_weight),
            "n_train": int(self.data.X_train.shape[0]),
            "n_features": int(self.data.X_train.shape[1]),
            "has_valid": self.data.X_valid is not None,
            "head": "interval",
        }


def _safe_estimator_complexity(estimator: Any) -> float:
    if hasattr(estimator, "estimators_"):
        try:
            return float(len(estimator.estimators_))
        except Exception:
            return 1.0
    if hasattr(estimator, "tree_"):
        try:
            return float(estimator.tree_.node_count)
        except Exception:
            return 1.0
    return 1.0


def _fit_estimator_with_lifecycle(
    estimator: Any,
    model: EstimatorSpecModel,
    data: NumericDataView,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    mechanisms = dict(model.mechanisms)
    params = dict(model.params)
    fit_kwargs: dict[str, Any] = {}
    early_stopping = _extract_early_stopping(mechanisms, params)
    if early_stopping.get("enabled") and data.X_valid is not None and data.y_valid is not None:
        fit_kwargs["eval_set"] = [(data.X_valid, data.y_valid)]
        if "verbose" not in fit_kwargs:
            fit_kwargs["verbose"] = False
    if early_stopping.get("eval_metric") and _safe_accepts_fit_kwarg(estimator, "eval_metric"):
        fit_kwargs["eval_metric"] = early_stopping["eval_metric"]
    resume_payload = context.get("estimator.resume_payload") if isinstance(context, Mapping) else None
    continuation = _extract_continuation(mechanisms)
    if resume_payload is not None and continuation.get("mode") == "xgb_model" and _safe_accepts_fit_kwarg(estimator, "xgb_model"):
        fit_kwargs["xgb_model"] = resume_payload
    try:
        estimator.fit(data.X_train, data.y_train, **fit_kwargs)
        fit_status = "ok"
    except TypeError:
        if early_stopping.get("strict"):
            raise
        fit_kwargs = {}
        estimator.fit(data.X_train, data.y_train)
        fit_status = "fallback_no_fit_kwargs"
    return {
        "status": fit_status,
        "fit_kwargs": tuple(sorted(fit_kwargs.keys())),
        "early_stopping_enabled": bool(early_stopping.get("enabled", False)),
        "early_stopping_rounds": early_stopping.get("rounds"),
        "continuation_mode": continuation.get("mode"),
        "warm_start": bool(continuation.get("warm_start", False)),
        "supports_resume": bool(continuation.get("trainer_state_enabled", False)),
    }


def _extract_early_stopping(mechanisms: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    boosting = mechanisms.get("boosting", {}) if isinstance(mechanisms.get("boosting", {}), Mapping) else {}
    early = boosting.get("early_stopping", {}) if isinstance(boosting.get("early_stopping", {}), Mapping) else {}
    rounds = early.get("rounds", params.get("early_stopping_rounds", boosting.get("early_stopping_rounds")))
    return {
        "enabled": rounds is not None,
        "rounds": None if rounds is None else int(rounds),
        "eval_metric": early.get("eval_metric", params.get("eval_metric")),
        "strict": bool(early.get("strict", False)),
    }


def _extract_continuation(mechanisms: Mapping[str, Any]) -> dict[str, Any]:
    boosting = mechanisms.get("boosting", {}) if isinstance(mechanisms.get("boosting", {}), Mapping) else {}
    continuation = boosting.get("continuation", {}) if isinstance(boosting.get("continuation", {}), Mapping) else {}
    tree = mechanisms.get("tree", {}) if isinstance(mechanisms.get("tree", {}), Mapping) else {}
    tree_cont = tree.get("continuation", {}) if isinstance(tree.get("continuation", {}), Mapping) else {}
    merged = {**tree_cont, **continuation}
    return {
        "mode": merged.get("mode", boosting.get("continuation_mode", "none")),
        "warm_start": bool(merged.get("warm_start", boosting.get("warm_start", False))),
        "trainer_state_enabled": bool(merged.get("trainer_state_enabled", boosting.get("trainer_state_enabled", False))),
    }


def _safe_accepts_fit_kwarg(estimator: Any, key: str) -> bool:
    try:
        import inspect

        sig = inspect.signature(estimator.fit)
        return key in sig.parameters or any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values())
    except Exception:
        return True


def _clamp_estimator_spec_to_resource(model: EstimatorSpecModel, context: Mapping[str, Any]) -> EstimatorSpecModel:
    params = dict(model.params)
    threads = int(context.get("resource.threads", context.get("resource_context", {}).get("threads", 0)) or 0)
    if threads > 0 and "n_jobs" in params:
        try:
            n_jobs = int(params.get("n_jobs", threads))
        except Exception:
            n_jobs = threads
        if n_jobs < 0 or n_jobs > threads:
            params["n_jobs"] = int(threads)
    if params == dict(model.params):
        return model
    return EstimatorSpecModel(
        family=model.family,
        route=model.route,
        params=params,
        factory=model.factory,
        metadata={**dict(model.metadata), "resource_clamped": True, "resource_threads": threads},
    )


def _interval_metrics(model: Any, X: np.ndarray, y: np.ndarray, *, prefix: str) -> tuple[dict[str, float], np.ndarray]:
    target = np.asarray(y, dtype=float).reshape(-1)
    lower, upper = _predict_interval(model, X)
    if lower.shape[0] != target.shape[0] or upper.shape[0] != target.shape[0]:
        raise ValueError("interval prediction length differs from target length")
    below = np.maximum(lower - target, 0.0)
    above = np.maximum(target - upper, 0.0)
    miss_distance = below + above
    covered = (target >= lower) & (target <= upper)
    center = (lower + upper) / 2.0
    residual = center - target
    width = np.maximum(upper - lower, 0.0)
    mse_center = float(np.mean(residual ** 2))
    return (
        {
            f"{prefix}.coverage": float(np.mean(covered)),
            f"{prefix}.mean_width": float(np.mean(width)),
            f"{prefix}.mean_miss_distance": float(np.mean(miss_distance)),
            f"{prefix}.center_mse": mse_center,
            f"{prefix}.center_rmse": float(np.sqrt(mse_center)),
        },
        residual,
    )


def _predict_interval(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "predict_interval"):
        lower, upper = model.predict_interval(X)
        return np.asarray(lower, dtype=float).reshape(-1), np.asarray(upper, dtype=float).reshape(-1)
    pred = np.asarray(model.predict(X), dtype=float)
    if pred.ndim != 2 or pred.shape[1] != 2:
        raise TypeError("interval model must expose predict_interval(...) or predict(...) with two columns")
    return pred[:, 0].reshape(-1), pred[:, 1].reshape(-1)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, prefix: str) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y.shape[0] != pred.shape[0]:
        raise ValueError("prediction length differs from target length")
    err = pred - y
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 if denom <= 0.0 and float(np.sum(err ** 2)) <= 0.0 else 1.0 - float(np.sum(err ** 2)) / denom
    return {
        f"{prefix}.mse": mse,
        f"{prefix}.rmse": rmse,
        f"{prefix}.mae": mae,
        f"{prefix}.r2": float(r2),
    }


