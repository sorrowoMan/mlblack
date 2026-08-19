from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from blackbase.contracts import ComponentContract
from mlblack.core.problem import LearningProblem
from mlblack.core.types import Feedback, UnknownState
from mlblack.models.symbolic import SymbolicBasisSetModel, SymbolicExpressionModel
from mlblack.pipeline.data_views import NumericDataView


class FixedSymbolicRegressionProblem(LearningProblem):
    """Regression evaluator for fixed symbolic expression decoders."""

    name = "fixed_symbolic_regression"
    objective_count = 2
    context_requires = ("candidate.model", "data.X_train", "data.y_train", "symbolic.expression_spec")
    context_optional = ("data.X_valid", "data.y_valid", "model.parameter_gradient")
    context_provides = (
        "feedback.objectives",
        "feedback.loss",
        "feedback.metrics",
        "feedback.residuals",
        "feedback.gradients",
        "feedback.signals",
        "symbolic.artifact",
    )
    context_mutates = ()
    context_cache = ()
    requires_metrics = ("mse", "rmse", "mae")
    metrics_fallback = "strict"
    context_notes = "Fits/evaluates parameters for a fixed symbolic expression model."
    contract = ComponentContract(
        name=name,
        requires=("candidate.model", "data.X_train", "data.y_train", "symbolic.expression_spec"),
        optional=("data.X_valid", "data.y_valid", "model.parameter_gradient"),
        provides=(
            "feedback.objectives",
            "feedback.loss",
            "feedback.metrics",
            "feedback.residuals",
            "feedback.gradients",
            "feedback.signals",
            "symbolic.artifact",
        ),
        supports_gradient=True,
        supports_batch=False,
        supports_resume=False,
        metadata={"family": "symbolic", "structure_search": False},
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
        if not isinstance(model, SymbolicExpressionModel):
            raise TypeError("FixedSymbolicRegressionProblem expects SymbolicExpressionModel")

        train_pred = np.asarray(model.predict(self.data.X_train), dtype=float).reshape(-1)
        train_metrics = _regression_metrics(self.data.y_train, train_pred, prefix="train")
        train_residual = train_pred - self.data.y_train

        metrics: dict[str, Any] = dict(train_metrics)
        residual = train_residual
        objective_mse = float(train_metrics["train.mse"])
        if self.data.X_valid is not None and self.data.y_valid is not None:
            valid_pred = np.asarray(model.predict(self.data.X_valid), dtype=float).reshape(-1)
            valid_metrics = _regression_metrics(self.data.y_valid, valid_pred, prefix="valid")
            metrics.update(valid_metrics)
            residual = valid_pred - self.data.y_valid
            if self.use_valid_objective:
                objective_mse = float(valid_metrics["valid.mse"])

        values = state.as_array()
        l2_norm = float(np.sum(values * values))
        complexity = float(model.describe()["complexity"])
        loss = float(train_metrics["train.mse"] + self.l2 * l2_norm)
        gradient = np.asarray(model.parameter_gradient(self.data.X_train, self.data.y_train, l2=self.l2), dtype=float)

        metrics.update(
            {
                "loss.regularized_train": loss,
                "complexity.expression": complexity,
                "complexity.l2_norm": l2_norm,
                "symbolic.n_parameters": int(values.shape[0]),
            }
        )
        return Feedback(
            objectives=np.asarray([objective_mse, float(self.complexity_weight) * complexity], dtype=float),
            constraints=np.zeros(0, dtype=float),
            loss=loss,
            gradients=gradient,
            residuals=np.asarray(residual, dtype=float).reshape(-1),
            metrics=metrics,
            signals={
                "primary_objective": "valid.mse" if self.data.X_valid is not None and self.use_valid_objective else "train.mse",
                "has_gradient": True,
                "symbolic_expression": model.describe()["expression_string"],
            },
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


class OrthogonalBasisEvaluationProblem(LearningProblem):
    """Evaluate a fitted symbolic basis set for orthogonality objectives."""

    name = "orthogonal_basis_evaluation"
    objective_count = 3
    context_requires = ("candidate.symbolic_basis_model", "data.X_train", "symbolic.genome")
    context_optional = ("data.X_valid", "basis.artifact_ref")
    context_provides = ("feedback.objectives", "feedback.metrics", "feedback.signals", "basis.metrics", "symbolic.artifact")
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Scores a fitted SymbolicBasisSetModel by correlation, rank, condition and complexity."
    contract = ComponentContract(
        name=name,
        requires=("candidate.symbolic_basis_model", "data.X_train", "symbolic.genome"),
        optional=("data.X_valid", "basis.artifact_ref"),
        provides=("feedback.objectives", "feedback.metrics", "feedback.signals", "basis.metrics", "symbolic.artifact"),
        supports_gradient=False,
        supports_batch=False,
        supports_resume=False,
        metadata={"family": "symbolic", "stage": "orthogonal_basis"},
    )

    def __init__(
        self,
        data: NumericDataView,
        *,
        complexity_weight: float = 0.01,
        condition_weight: float = 0.001,
    ) -> None:
        self.data = data
        self.complexity_weight = float(complexity_weight)
        self.condition_weight = float(condition_weight)

    def evaluate(self, model: Any, state: UnknownState, context: Mapping[str, Any]) -> Feedback:
        _ = state
        _ = context
        if not isinstance(model, SymbolicBasisSetModel):
            raise TypeError("OrthogonalBasisEvaluationProblem expects SymbolicBasisSetModel")
        Z = np.asarray(model.transform(self.data.X_train), dtype=float)
        metrics = _basis_metrics(Z)
        complexity = float(model.complexity())
        metrics["complexity.expression"] = complexity
        metrics["basis.n_atoms"] = int(Z.shape[1])
        metrics["basis.expressions"] = list(model.expression_strings())

        objectives = np.asarray(
            [
                float(metrics["basis.max_abs_corr"]),
                float(self.condition_weight) * float(metrics["basis.condition_number"]),
                float(self.complexity_weight) * complexity,
            ],
            dtype=float,
        )
        return Feedback(
            objectives=objectives,
            constraints=np.zeros(0, dtype=float),
            loss=float(np.sum(objectives)),
            gradients=None,
            residuals=None,
            metrics=metrics,
            signals={
                "basis.rank_full": bool(metrics["basis.rank"] >= Z.shape[1]),
                "basis.max_abs_corr": float(metrics["basis.max_abs_corr"]),
                "symbolic_basis": list(model.expression_strings()),
            },
        )

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "complexity_weight": float(self.complexity_weight),
            "condition_weight": float(self.condition_weight),
            "n_train": int(self.data.X_train.shape[0]),
            "n_features": int(self.data.X_train.shape[1]),
        }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, prefix: str) -> dict[str, float]:
    target = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    err = pred - target
    mse = float(np.mean(err * err))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum((target - float(np.mean(target))) ** 2))
    r2 = 0.0 if denom <= 1e-12 else float(1.0 - (np.sum(err * err) / denom))
    return {
        f"{prefix}.mse": mse,
        f"{prefix}.rmse": rmse,
        f"{prefix}.mae": mae,
        f"{prefix}.r2": r2,
    }


def _basis_metrics(Z: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(Z, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("basis output must be 2D")
    if matrix.shape[1] == 0:
        raise ValueError("basis output has no columns")
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    std = np.std(centered, axis=0)
    safe = np.where(std <= 1e-12, 1.0, std)
    normalized = centered / safe
    corr = np.corrcoef(normalized, rowvar=False) if matrix.shape[1] > 1 else np.eye(1)
    corr = np.asarray(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
    if corr.ndim == 0:
        corr = np.eye(1)
    off_diag = corr - np.eye(corr.shape[0])
    max_abs_corr = float(np.max(np.abs(off_diag))) if off_diag.size else 0.0
    mean_abs_corr = float(np.sum(np.abs(off_diag)) / max(1, off_diag.size - corr.shape[0]))
    rank = int(np.linalg.matrix_rank(centered))
    try:
        condition = float(np.linalg.cond(centered))
    except Exception:
        condition = float("inf")
    if not np.isfinite(condition):
        condition = 1e12
    return {
        "basis.max_abs_corr": max_abs_corr,
        "basis.mean_abs_corr": mean_abs_corr,
        "basis.rank": float(rank),
        "basis.condition_number": condition,
    }
