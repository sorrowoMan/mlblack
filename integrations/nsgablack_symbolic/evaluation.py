from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.integrations.nsgablack_control import build_learning_solver
from mlblack.core.types import UnknownState
from mlblack.integrations.nsgablack_optimization import build_optimization_adapter
from mlblack.pipeline.data_views import NumericDataView
from mlblack.problems import FixedSymbolicRegressionProblem, SupervisedClassificationProblem, SupervisedIntervalRegressionProblem
from mlblack.representations import SymbolicExpressionConfig, SymbolicExpressionRepresentation
from mlblack.representations.heads import BinaryLogisticHead, CenterRadiusIntervalHead, PointHead, ProbabilityCalibrationHead, SoftmaxHead, TwoModelIntervalHead

from .artifacts import OrthogonalBasisSetArtifact, SymbolicTaskArtifact


@dataclass(frozen=True)
class SymbolicFoldEvaluationConfig:
    n_splits: int = 3
    shuffle: bool = True
    seed: int = 42
    classification_threshold: float = 0.5
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SymbolicFoldEvaluationReport:
    artifact_id: str
    artifact_type: str
    task_kind: str
    head_kind: str
    fold_count: int
    fold_metrics: tuple[Mapping[str, Any], ...]
    aggregate_metrics: Mapping[str, Any]
    config: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": str(self.artifact_id),
            "artifact_type": str(self.artifact_type),
            "task_kind": str(self.task_kind),
            "head_kind": str(self.head_kind),
            "fold_count": int(self.fold_count),
            "fold_metrics": [dict(row) for row in self.fold_metrics],
            "aggregate_metrics": dict(self.aggregate_metrics),
            "config": dict(self.config),
        }


@dataclass(frozen=True)
class SymbolicBranchSpec:
    name: str
    kind: str = "all"
    feature: int | str | None = None
    op: str = ">="
    threshold: float | None = None
    lower: float | None = None
    upper: float | None = None
    q_low: float | None = None
    q_high: float | None = None
    indices: tuple[int, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "kind": str(self.kind),
            "feature": self.feature,
            "op": str(self.op),
            "threshold": None if self.threshold is None else float(self.threshold),
            "lower": None if self.lower is None else float(self.lower),
            "upper": None if self.upper is None else float(self.upper),
            "q_low": None if self.q_low is None else float(self.q_low),
            "q_high": None if self.q_high is None else float(self.q_high),
            "indices": [int(v) for v in self.indices],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SymbolicBranchEvaluationConfig:
    branches: tuple[SymbolicBranchSpec, ...] = tuple()
    include_all_branch: bool = True
    auto_quantile_feature_indices: tuple[int, ...] = tuple()
    auto_quantiles: tuple[float, ...] = (0.0, 0.5, 1.0)
    min_branch_size: int = 3
    enable_branch_refit: bool = False
    branch_refit_steps: int = 8
    branch_refit_population_size: int = 8
    branch_refit_mutation_scale: float = 0.15
    branch_refit_learning_rate: float = 0.03
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SymbolicBranchEvaluationReport:
    artifact_id: str
    artifact_type: str
    task_kind: str
    head_kind: str
    branch_count: int
    branch_metrics: tuple[Mapping[str, Any], ...]
    aggregate_metrics: Mapping[str, Any]
    config: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": str(self.artifact_id),
            "artifact_type": str(self.artifact_type),
            "task_kind": str(self.task_kind),
            "head_kind": str(self.head_kind),
            "branch_count": int(self.branch_count),
            "branch_metrics": [dict(row) for row in self.branch_metrics],
            "aggregate_metrics": dict(self.aggregate_metrics),
            "config": dict(self.config),
        }


class SymbolicFoldEvaluator:
    """Lightweight fold evaluator for symbolic artifacts.

    This is report/evaluation surface only. It evaluates a fitted artifact over
    folds and does not refit parameters or alter search semantics.
    """

    name = "symbolic_fold_evaluation"
    context_requires = ("symbolic.artifact", "data.X_train", "data.y_train")
    context_optional = ("data.X_valid", "data.y_valid", "resource.context")
    context_provides = ("symbolic.fold_report", "symbolic.evaluation_events", "artifact.report")
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Evaluates fitted symbolic artifacts over deterministic folds for audit/stability reporting."

    def __init__(self, config: SymbolicFoldEvaluationConfig | None = None) -> None:
        self.config = config or SymbolicFoldEvaluationConfig()

    def evaluate_task_artifact(self, artifact: SymbolicTaskArtifact, data: NumericDataView) -> SymbolicFoldEvaluationReport:
        X, y = _combined_xy(data)
        model = _task_model(artifact, input_dim=int(data.n_features), feature_names=data.effective_feature_names)
        folds = _fold_indices(X.shape[0], n_splits=int(self.config.n_splits), shuffle=bool(self.config.shuffle), seed=int(self.config.seed))
        rows: list[Mapping[str, Any]] = []
        for fold_idx, eval_idx in enumerate(folds):
            x_eval = X[eval_idx]
            y_eval = y[eval_idx]
            metrics = _task_metrics(model, x_eval, y_eval, task_kind=artifact.task_kind, head_kind=artifact.head_kind)
            rows.append({"fold": int(fold_idx), "n_eval": int(len(eval_idx)), **metrics})
        return SymbolicFoldEvaluationReport(
            artifact_id=str(artifact.artifact_id or artifact.name),
            artifact_type="symbolic_task",
            task_kind=str(artifact.task_kind),
            head_kind=str(artifact.head_kind),
            fold_count=int(len(rows)),
            fold_metrics=tuple(rows),
            aggregate_metrics=_aggregate(rows),
            config=self._config_dict(),
        )

    def evaluate_basis_artifact(self, artifact: OrthogonalBasisSetArtifact, data: NumericDataView) -> SymbolicFoldEvaluationReport:
        X, _y = _combined_xy(data)
        folds = _fold_indices(X.shape[0], n_splits=int(self.config.n_splits), shuffle=bool(self.config.shuffle), seed=int(self.config.seed))
        rows: list[Mapping[str, Any]] = []
        for fold_idx, eval_idx in enumerate(folds):
            Z = artifact.transform(X[eval_idx])
            metrics = _basis_metrics(Z)
            rows.append({"fold": int(fold_idx), "n_eval": int(len(eval_idx)), **metrics})
        return SymbolicFoldEvaluationReport(
            artifact_id=str(artifact.artifact_id or artifact.name),
            artifact_type="symbolic_basis",
            task_kind="orthogonal_basis",
            head_kind="symbolic_basis_set",
            fold_count=int(len(rows)),
            fold_metrics=tuple(rows),
            aggregate_metrics=_aggregate(rows),
            config=self._config_dict(),
        )

    def _config_dict(self) -> dict[str, Any]:
        return {
            "n_splits": int(self.config.n_splits),
            "shuffle": bool(self.config.shuffle),
            "seed": int(self.config.seed),
            "classification_threshold": float(self.config.classification_threshold),
            "metadata": dict(self.config.metadata),
        }

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "config": self._config_dict()}


class SymbolicBranchEvaluator:
    """Branch/regime audit evaluator for fitted symbolic task artifacts.

    This evaluator only evaluates an already fitted artifact on branch subsets.
    It does not refit branch-local models; branch-local refit is a later
    strategy extension on the same evaluation surface.
    """

    name = "symbolic_branch_evaluation"
    context_requires = ("symbolic.artifact", "data.X_train", "data.y_train")
    context_optional = ("data.X_valid", "data.y_valid", "branch.spec", "resource.context")
    context_provides = ("symbolic.branch_report", "symbolic.evaluation_events", "stage.audit", "artifact.report")
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Evaluates fitted symbolic task artifacts over configured branch/regime subsets for audit reporting."

    def __init__(self, config: SymbolicBranchEvaluationConfig | None = None) -> None:
        self.config = config or SymbolicBranchEvaluationConfig()

    def evaluate_task_artifact(
        self,
        artifact: SymbolicTaskArtifact,
        data: NumericDataView,
        *,
        branch_data: NumericDataView | None = None,
        resource_context: Mapping[str, Any] | None = None,
    ) -> SymbolicBranchEvaluationReport:
        X_model, y = _combined_xy(data)
        model = _task_model(artifact, input_dim=int(data.n_features), feature_names=data.effective_feature_names)
        source = branch_data or data
        X_branch, _ = _combined_xy(source)
        if X_branch.shape[0] != X_model.shape[0]:
            raise ValueError("branch_data row count must match artifact evaluation data row count")
        specs = _resolve_branch_specs(self.config, X_branch, source.effective_feature_names)
        rows: list[Mapping[str, Any]] = []
        for branch_index, spec in enumerate(specs):
            mask = _branch_mask(spec, X_branch, source.effective_feature_names)
            n_eval = int(np.sum(mask))
            base_row: dict[str, Any] = {
                "branch_index": int(branch_index),
                "branch_name": str(spec.name),
                "branch_kind": str(spec.kind),
                "n_eval": n_eval,
                "n_total": int(X_model.shape[0]),
                "coverage_ratio": float(n_eval / float(max(1, X_model.shape[0]))),
                "branch_spec": spec.as_dict(),
            }
            if n_eval < int(max(1, self.config.min_branch_size)):
                rows.append({**base_row, "skipped": True, "skip_reason": "min_branch_size"})
                continue
            metrics = _task_metrics(
                model,
                X_model[mask],
                y[mask],
                task_kind=artifact.task_kind,
                head_kind=artifact.head_kind,
            )
            row = {**base_row, "skipped": False, **metrics}
            if bool(self.config.enable_branch_refit):
                row.update(
                    _branch_refit_metrics(
                        artifact,
                        X_model[mask],
                        y[mask],
                        feature_names=data.effective_feature_names,
                        global_metrics=metrics,
                        config=self.config,
                        resource_context=resource_context,
                    )
                )
            rows.append(row)
        return SymbolicBranchEvaluationReport(
            artifact_id=str(artifact.artifact_id or artifact.name),
            artifact_type="symbolic_task",
            task_kind=str(artifact.task_kind),
            head_kind=str(artifact.head_kind),
            branch_count=int(len(rows)),
            branch_metrics=tuple(rows),
            aggregate_metrics=_aggregate(rows),
            config=self._config_dict(),
        )

    def _config_dict(self) -> dict[str, Any]:
        return {
            "branches": [spec.as_dict() for spec in self.config.branches],
            "include_all_branch": bool(self.config.include_all_branch),
            "auto_quantile_feature_indices": [int(v) for v in self.config.auto_quantile_feature_indices],
            "auto_quantiles": [float(v) for v in self.config.auto_quantiles],
            "min_branch_size": int(self.config.min_branch_size),
            "enable_branch_refit": bool(self.config.enable_branch_refit),
            "branch_refit_steps": int(self.config.branch_refit_steps),
            "branch_refit_population_size": int(self.config.branch_refit_population_size),
            "branch_refit_mutation_scale": float(self.config.branch_refit_mutation_scale),
            "branch_refit_learning_rate": float(self.config.branch_refit_learning_rate),
            "metadata": dict(self.config.metadata),
        }

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "config": self._config_dict()}


def _combined_xy(data: NumericDataView) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(data.X_train, dtype=float)
    y = np.asarray(data.y_train).reshape(-1)
    if data.X_valid is not None and data.y_valid is not None:
        X = np.vstack([X, np.asarray(data.X_valid, dtype=float)])
        y = np.concatenate([y, np.asarray(data.y_valid).reshape(-1)])
    return X, y


def _fold_indices(n_rows: int, *, n_splits: int, shuffle: bool, seed: int) -> tuple[np.ndarray, ...]:
    n = max(1, int(n_rows))
    k = int(np.clip(int(n_splits), 1, n))
    idx = np.arange(n)
    if bool(shuffle):
        rng = np.random.default_rng(int(seed))
        rng.shuffle(idx)
    return tuple(np.asarray(part, dtype=int) for part in np.array_split(idx, k) if len(part) > 0)


def _task_model(artifact: SymbolicTaskArtifact, *, input_dim: int, feature_names: Sequence[str]) -> Any:
    head = _head_from_artifact(artifact)
    representation = SymbolicExpressionRepresentation(
        SymbolicExpressionConfig(
            input_dim=int(input_dim),
            expression=dict(artifact.expression),
            name=str(artifact.name),
            feature_names=tuple(feature_names),
        ),
        head=head,
    )
    return representation.decode(UnknownState(values=np.asarray(artifact.fitted_state, dtype=float)))


def _branch_refit_metrics(
    artifact: SymbolicTaskArtifact,
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str],
    global_metrics: Mapping[str, Any],
    config: SymbolicBranchEvaluationConfig,
    resource_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        data = NumericDataView(
            X_train=np.asarray(X, dtype=float),
            y_train=np.asarray(y),
            feature_names=tuple(feature_names),
        )
        head = _head_from_artifact(artifact)
        representation = SymbolicExpressionRepresentation(
            SymbolicExpressionConfig(
                input_dim=int(data.n_features),
                expression=dict(artifact.expression),
                name=f"{artifact.name}_branch_refit",
                feature_names=tuple(feature_names),
            ),
            head=head,
        )
        task = str(artifact.task_kind or "regression").strip().lower()
        if task == "interval" or str(artifact.head_kind).startswith("interval"):
            problem = SupervisedIntervalRegressionProblem(data, use_valid_objective=False)
            adapter = build_optimization_adapter(
                "search.random_gaussian",
                population_size=int(config.branch_refit_population_size),
                mutation_scale=float(config.branch_refit_mutation_scale),
                random_seed=17,
            )
            adapter.set_population((UnknownState(values=artifact.fitted_state),))
        elif task == "classification" or str(artifact.head_kind) in {"binary_logistic", "softmax", "probability_calibration"}:
            problem = SupervisedClassificationProblem(data, use_valid_objective=False)
            adapter = build_optimization_adapter(
                "search.random_gaussian",
                population_size=int(config.branch_refit_population_size),
                mutation_scale=float(config.branch_refit_mutation_scale),
                random_seed=23,
            )
            adapter.set_population((UnknownState(values=artifact.fitted_state),))
        else:
            problem = FixedSymbolicRegressionProblem(data, use_valid_objective=False)
            adapter = build_optimization_adapter(
                "gradient.sgd",
                learning_rate=float(config.branch_refit_learning_rate),
                max_gradient_norm=1e3,
            )
            adapter.set_population((UnknownState(values=artifact.fitted_state),))
        trainer = build_learning_solver(
            problem=problem,
            representation=representation,
            adapter=adapter,
            run_name="symbolic_branch_local_refit",
            resource_context=resource_context,
        )
        result = trainer.fit(max_steps=int(max(1, config.branch_refit_steps)))
        if result.best_model is None:
            return {"branch_refit.status": "no_best_model"}
        local = _task_metrics(result.best_model, data.X_train, data.y_train, task_kind=artifact.task_kind, head_kind=artifact.head_kind)
        out: dict[str, Any] = {
            "branch_refit.status": "ok",
            "branch_refit.steps": int(len(result.history)),
            "branch_refit.best_score": None if result.report.get("best_score") is None else float(result.report["best_score"]),
        }
        for key, value in local.items():
            out[f"branch_refit.{key}"] = float(value)
            if key in global_metrics and _is_number(global_metrics[key]):
                out[f"branch_refit.delta_{key}"] = float(value) - float(global_metrics[key])
        return out
    except Exception as exc:
        return {"branch_refit.status": "error", "branch_refit.error": repr(exc)}


def _head_from_artifact(artifact: SymbolicTaskArtifact) -> Any:
    key = str(artifact.head_kind or "point").strip().lower()
    if key == "point":
        return PointHead()
    if key == "interval_center_radius":
        return CenterRadiusIntervalHead()
    if key == "interval_two_model":
        return TwoModelIntervalHead()
    classes = _artifact_classes(artifact)
    if key == "binary_logistic":
        return BinaryLogisticHead(classes=classes if len(classes) >= 2 else (0, 1))
    if key == "softmax":
        return SoftmaxHead(n_classes=max(2, len(classes)), classes=classes if len(classes) >= 2 else (0, 1))
    if key == "probability_calibration":
        return ProbabilityCalibrationHead()
    return PointHead()


def _artifact_classes(artifact: SymbolicTaskArtifact) -> tuple[Any, ...]:
    meta = dict(artifact.metadata)
    classes = meta.get("classes") or dict(meta.get("task_config", {}) or {}).get("classes")
    if classes:
        return tuple(classes)
    return (0, 1)


def _task_metrics(model: Any, X: np.ndarray, y: np.ndarray, *, task_kind: str, head_kind: str) -> dict[str, float]:
    task = str(task_kind or "regression").strip().lower()
    if task == "interval" or str(head_kind).startswith("interval"):
        lower, upper = model.predict_interval(X)
        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)
        target = np.asarray(y, dtype=float).reshape(-1)
        center = (lower + upper) / 2.0
        width = np.maximum(upper - lower, 0.0)
        miss = np.maximum(lower - target, 0.0) + np.maximum(target - upper, 0.0)
        covered = (target >= lower) & (target <= upper)
        return {
            "coverage": float(np.mean(covered)),
            "mean_width": float(np.mean(width)),
            "mean_miss_distance": float(np.mean(miss)),
            "center_rmse": _rmse(target, center),
        }
    if task == "classification" or str(head_kind) in {"binary_logistic", "softmax", "probability_calibration"}:
        target = np.asarray(y).reshape(-1)
        proba = np.asarray(model.predict_proba(X), dtype=float)
        pred = np.asarray(model.predict(X)).reshape(-1)
        classes = tuple(getattr(model, "classes_", np.unique(target).tolist()))
        class_to_idx = {value: idx for idx, value in enumerate(classes)}
        y_idx = np.asarray([class_to_idx.get(value, 0) for value in target], dtype=int)
        proba = np.clip(proba, 1e-12, 1.0)
        proba = proba / np.sum(proba, axis=1, keepdims=True)
        log_loss = -float(np.mean(np.log(proba[np.arange(target.shape[0]), y_idx])))
        accuracy = float(np.mean(pred == target))
        return {
            "log_loss": log_loss,
            "accuracy": accuracy,
            "error_rate": 1.0 - accuracy,
        }
    target = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(model.predict(X), dtype=float).reshape(-1)
    err = pred - target
    return {
        "rmse": _rmse(target, pred),
        "mae": float(np.mean(np.abs(err))),
        "r2": _r2(target, pred),
    }


def _basis_metrics(Z: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(Z, dtype=float)
    if matrix.ndim != 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    if matrix.shape[1] <= 1:
        max_abs_corr = 0.0
        mean_abs_corr = 0.0
    else:
        corr = np.corrcoef(centered, rowvar=False)
        corr = np.asarray(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
        off_diag = corr - np.eye(corr.shape[0])
        max_abs_corr = float(np.max(np.abs(off_diag))) if off_diag.size else 0.0
        mean_abs_corr = float(np.mean(np.abs(off_diag))) if off_diag.size else 0.0
    try:
        cond = float(np.linalg.cond(centered))
    except Exception:
        cond = 1e12
    if not np.isfinite(cond):
        cond = 1e12
    return {
        "basis_max_abs_corr": max_abs_corr,
        "basis_mean_abs_corr": mean_abs_corr,
        "basis_rank": float(np.linalg.matrix_rank(centered)),
        "basis_condition_number": cond,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    ignored = {"fold", "branch_index", "n_eval", "n_total"}
    keys = sorted({str(key) for row in rows for key, value in row.items() if key not in ignored and _is_number(value)})
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows if key in row and _is_number(row[key])], dtype=float)
        if values.size:
            out[f"{key}.mean"] = float(np.mean(values))
            out[f"{key}.std"] = float(np.std(values))
    return out


def _resolve_branch_specs(
    config: SymbolicBranchEvaluationConfig,
    X_branch: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[SymbolicBranchSpec, ...]:
    specs: list[SymbolicBranchSpec] = []
    if bool(config.include_all_branch):
        specs.append(SymbolicBranchSpec(name="all", kind="all"))
    specs.extend(tuple(config.branches))
    quantiles = tuple(sorted({float(np.clip(q, 0.0, 1.0)) for q in tuple(config.auto_quantiles)}))
    if len(quantiles) >= 2:
        for feature_idx in tuple(config.auto_quantile_feature_indices):
            idx = int(feature_idx)
            if idx < 0 or idx >= int(X_branch.shape[1]):
                continue
            values = np.asarray(X_branch[:, idx], dtype=float)
            feature_name = str(feature_names[idx]) if idx < len(tuple(feature_names)) else f"x{idx}"
            cuts = [float(np.quantile(values, q)) for q in quantiles]
            for pos in range(len(cuts) - 1):
                lower = cuts[pos]
                upper = cuts[pos + 1]
                specs.append(
                    SymbolicBranchSpec(
                        name=f"{feature_name}.q{quantiles[pos]:.2f}_{quantiles[pos + 1]:.2f}",
                        kind="feature_range",
                        feature=idx,
                        lower=lower,
                        upper=upper,
                        q_low=quantiles[pos],
                        q_high=quantiles[pos + 1],
                        metadata={"source": "auto_quantile", "feature_name": feature_name},
                    )
                )
    return tuple(specs)


def _branch_mask(spec: SymbolicBranchSpec, X_branch: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    X = np.asarray(X_branch, dtype=float)
    n = int(X.shape[0])
    kind = str(spec.kind or "all").strip().lower()
    if kind == "all":
        return np.ones(n, dtype=bool)
    if kind in {"mask", "mask_indices", "indices"}:
        mask = np.zeros(n, dtype=bool)
        idx = np.asarray([int(v) for v in tuple(spec.indices)], dtype=int)
        idx = idx[(idx >= 0) & (idx < n)]
        mask[idx] = True
        return mask
    feature_idx = _feature_index(spec.feature, feature_names, X.shape[1])
    values = np.asarray(X[:, feature_idx], dtype=float)
    if kind in {"feature_threshold", "threshold"}:
        threshold = float(0.0 if spec.threshold is None else spec.threshold)
        return _threshold_mask(values, op=str(spec.op), threshold=threshold)
    if kind in {"feature_range", "range"}:
        mask = np.ones(n, dtype=bool)
        if spec.lower is not None:
            mask &= values >= float(spec.lower)
        if spec.upper is not None:
            mask &= values <= float(spec.upper)
        return mask
    if kind in {"feature_quantile", "quantile"}:
        q_low = 0.0 if spec.q_low is None else float(np.clip(spec.q_low, 0.0, 1.0))
        q_high = 1.0 if spec.q_high is None else float(np.clip(spec.q_high, 0.0, 1.0))
        lower = float(np.quantile(values, min(q_low, q_high)))
        upper = float(np.quantile(values, max(q_low, q_high)))
        return (values >= lower) & (values <= upper)
    raise ValueError(f"unsupported branch spec kind: {spec.kind}")


def _feature_index(feature: int | str | None, feature_names: Sequence[str], input_dim: int) -> int:
    if feature is None:
        return 0
    if isinstance(feature, str) and not feature.strip().lstrip("-").isdigit():
        names = tuple(str(v) for v in feature_names)
        if feature not in names:
            raise ValueError(f"unknown branch feature name: {feature}")
        return int(names.index(feature))
    idx = int(feature)
    if idx < 0 or idx >= int(input_dim):
        raise ValueError(f"branch feature index out of range: {idx}")
    return idx


def _threshold_mask(values: np.ndarray, *, op: str, threshold: float) -> np.ndarray:
    key = str(op or ">=").strip()
    if key == ">=":
        return values >= float(threshold)
    if key == ">":
        return values > float(threshold)
    if key == "<=":
        return values <= float(threshold)
    if key == "<":
        return values < float(threshold)
    if key in {"==", "="}:
        return np.isclose(values, float(threshold))
    if key == "!=":
        return ~np.isclose(values, float(threshold))
    raise ValueError(f"unsupported branch threshold op: {op}")


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    err = np.asarray(pred, dtype=float).reshape(-1) - np.asarray(y, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean(err * err)))


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    target = np.asarray(y, dtype=float).reshape(-1)
    err = np.asarray(pred, dtype=float).reshape(-1) - target
    denom = float(np.sum((target - float(np.mean(target))) ** 2))
    return 0.0 if denom <= 1e-12 else float(1.0 - np.sum(err * err) / denom)


__all__ = [
    "SymbolicBranchEvaluationConfig",
    "SymbolicBranchEvaluationReport",
    "SymbolicBranchEvaluator",
    "SymbolicBranchSpec",
    "SymbolicFoldEvaluationConfig",
    "SymbolicFoldEvaluationReport",
    "SymbolicFoldEvaluator",
]

