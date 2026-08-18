from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.adapters import GradientDescentAdapter, RandomSearchAdapter
from mlblack.core import ComposableTrainer
from blackbase.contracts import ComponentContract, ContractMixin
from mlblack.models.symbolic import binary_expr, expression_complexity, param_expr, parameterize_expression
from mlblack.pipeline.data_views import NumericDataView
from mlblack.pipeline.symbolic import FunctionPool, FunctionPoolPipeline, FunctionPoolPipelineConfig
from mlblack.problems import FixedSymbolicRegressionProblem, SupervisedClassificationProblem, SupervisedIntervalRegressionProblem
from mlblack.representations import SymbolicExpressionConfig, SymbolicExpressionRepresentation
from mlblack.representations.heads import BinaryLogisticHead, CenterRadiusIntervalHead, PointHead, ProbabilityCalibrationHead, SoftmaxHead, TwoModelIntervalHead

from .artifacts import OrthogonalBasisSetArtifact, SymbolicTaskArtifact
from .basis_consensus import SymbolicBasisConsensusAnalyzer
from .evaluation import SymbolicBranchEvaluationConfig, SymbolicBranchEvaluator, SymbolicFoldEvaluator
from .expression_audit import SymbolicExpressionAuditProducer
from .graph_cache import ExpressionGraphCache
from .path_memory import SymbolicPathMemory
from .replay import SymbolicReplayRecordBuilder
from .search_policy import CandidateScoreConfig, SymbolicCandidateScorer

try:  # optional integration dependency
    from nsgablack.core.base import BlackBoxProblem as _BlackBoxProblem
except Exception:  # pragma: no cover

    class _BlackBoxProblem:  # type: ignore[no-redef]
        def __init__(self, *, name: str, dimension: int, bounds: Mapping[str, Sequence[float]], objectives: Sequence[str]) -> None:
            self.name = name
            self.dimension = int(dimension)
            self.bounds = dict(bounds)
            self.objectives = tuple(objectives)


@dataclass(frozen=True)
class BasisConditionedSymbolicTaskConfig:
    task_kind: str = "regression"
    head_kind: str = "point"
    task_terms: int = 3
    pool_max_terms: int = 96
    inner_steps: int = 80
    learning_rate: float = 0.03
    inner_adapter: str = "auto"
    inner_population_size: int = 24
    inner_mutation_scale: float = 0.35
    random_seed: int = 42
    parameterize_terms: bool = True
    unary_input_affine: bool = True
    duplicate_penalty: float = 1.0
    complexity_weight: float = 0.001
    objective_names: tuple[str, ...] = tuple()
    target_coverage: float = 0.9
    interval_width_weight: float = 1.0
    interval_miss_weight: float = 10.0
    classification_objective_metrics: tuple[str, ...] = ("log_loss", "error_rate")
    classes: tuple[Any, ...] = tuple()
    positive_label: Any | None = None
    probability_temperature: float = 1.0
    probability_threshold: float = 0.5
    enable_path_memory: bool = False
    path_memory_db_path: str = ""
    path_memory_namespace: str = "symbolic_stage2"
    enable_graph_cache: bool = True
    graph_cache_backend: str = "memory"
    graph_cache_db_path: str = ""
    enable_branch_report: bool = True
    branch_report_feature_indices: tuple[int, ...] = (0,)
    branch_report_quantiles: tuple[float, ...] = (0.0, 0.5, 1.0)
    branch_report_min_size: int = 3
    enable_branch_refit_report: bool = False
    branch_refit_steps: int = 8
    candidate_score_config: CandidateScoreConfig = field(default_factory=CandidateScoreConfig)
    function_pool_config: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BasisConditionedTaskEvaluationRecord:
    outer_candidate: tuple[float, ...]
    selected_indices: tuple[int, ...]
    selected_terms: tuple[dict[str, Any], ...]
    expression: dict[str, Any]
    fitted_state: tuple[float, ...]
    objectives: tuple[float, ...]
    constraints: tuple[float, ...]
    metrics: Mapping[str, Any]
    signals: Mapping[str, Any]
    report: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "outer_candidate": list(self.outer_candidate),
            "selected_indices": list(self.selected_indices),
            "selected_terms": [dict(term) for term in self.selected_terms],
            "expression": dict(self.expression),
            "fitted_state": list(self.fitted_state),
            "objectives": list(self.objectives),
            "constraints": list(self.constraints),
            "metrics": dict(self.metrics),
            "signals": dict(self.signals),
            "report": dict(self.report),
        }


class BasisConditionedSymbolicTaskProblem(_BlackBoxProblem, ContractMixin):
    """nsgablack-compatible Stage 2 problem.

    Outer candidate:
      vector of function-pool indices over Stage 1 basis atoms.

    Inner mlblack run:
      fixed basis-conditioned expression -> parameter fitting -> task metrics.
    """

    name = "basis_conditioned_symbolic_task_problem"
    context_requires = ("data.X_train", "data.y_train", "basis.artifact_ref", "symbolic.function_pool", "resource.context")
    context_optional = ("data.X_valid", "data.y_valid", "resource.lease", "signal.pool")
    context_provides = ("feedback.objectives", "task.metrics", "artifact.symbolic_task_ref", "symbolic.artifact", "symbolic.branch_report")
    context_mutates = ("stage.audit",)
    context_cache = ("task.fitted_model_ref",)
    requires_metrics = ("rmse", "mae", "r2")
    metrics_fallback = "strict"
    context_notes = "Stage 2: outer symbolic task search over Stage 1 basis atoms with inner mlblack parameter fitting."
    contract = ComponentContract(
        name=name,
        requires=("data.X_train", "data.y_train", "basis.artifact_ref", "symbolic.function_pool", "resource.context"),
        optional=("data.X_valid", "data.y_valid", "resource.lease", "signal.pool"),
        provides=("feedback.objectives", "task.metrics", "artifact.symbolic_task_ref", "symbolic.artifact", "symbolic.branch_report"),
        mutates=("stage.audit",),
        cache=("task.fitted_model_ref",),
        supports_batch=False,
        supports_resume=True,
        metadata={"integration": "nsgablack_symbolic", "stage": "basis_conditioned_task"},
    )

    def __init__(
        self,
        data: NumericDataView,
        *,
        basis_artifact: OrthogonalBasisSetArtifact,
        function_pool: FunctionPool | None = None,
        config: BasisConditionedSymbolicTaskConfig | None = None,
        resource_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.original_data = data
        self.basis_artifact = basis_artifact
        self.data = basis_artifact.to_basis_data_view(data)
        self.config = config or BasisConditionedSymbolicTaskConfig()
        self.resource_context = dict(resource_context or {})
        self.function_pool = function_pool or self._build_function_pool()
        if not self.function_pool.terms:
            raise ValueError("function_pool must contain at least one term")
        self.last_record: BasisConditionedTaskEvaluationRecord | None = None
        self.best_record: BasisConditionedTaskEvaluationRecord | None = None
        self.evaluation_records: list[BasisConditionedTaskEvaluationRecord] = []
        self.evaluation_cache: dict[tuple[int, ...], BasisConditionedTaskEvaluationRecord] = {}
        self.path_memory = (
            SymbolicPathMemory(
                db_path=self.config.path_memory_db_path or None,
                namespace=self.config.path_memory_namespace,
            )
            if bool(self.config.enable_path_memory)
            else None
        )
        self.graph_cache = (
            ExpressionGraphCache(
                enabled=True,
                backend=str(self.config.graph_cache_backend),
                db_path=str(self.config.graph_cache_db_path),
                namespace=f"{self.config.path_memory_namespace}:graph",
            )
            if bool(self.config.enable_graph_cache)
            else None
        )
        self.candidate_scorer = SymbolicCandidateScorer(
            self.config.candidate_score_config,
            path_memory=self.path_memory,
        )
        bounds = {
            f"x{i}": [0.0, float(max(0, len(self.function_pool.terms) - 1))]
            for i in range(int(self.config.task_terms))
        }
        super().__init__(
            name=self.name,
            dimension=int(self.config.task_terms),
            bounds=bounds,
            objectives=_objective_names_for_config(self.config),
        )

    def _build_function_pool(self) -> FunctionPool:
        cfg_payload = dict(self.config.function_pool_config)
        cfg = FunctionPoolPipelineConfig(
            mode=str(cfg_payload.get("mode", "dynamic")),
            max_terms=int(cfg_payload.get("max_terms", self.config.pool_max_terms)),
            top_k_features=cfg_payload.get("top_k_features"),
            pair_top_k=int(cfg_payload.get("pair_top_k", 24)),
            recursive_depth=int(cfg_payload.get("recursive_depth", 2)),
            recursive_seed_top_k=int(cfg_payload.get("recursive_seed_top_k", 3)),
            recursive_pair_seed_top_k=int(cfg_payload.get("recursive_pair_seed_top_k", 2)),
            recursive_max_complexity=cfg_payload.get("recursive_max_complexity", 9.5),
            family_budget=dict(cfg_payload.get("family_budget", {})),
        )
        return FunctionPoolPipeline(cfg).build(
            self.data.X_train,
            y=self.data.y_train,
            feature_names=self.data.effective_feature_names,
        ).pool

    def decode_indices(self, x: Sequence[float] | np.ndarray) -> tuple[int, ...]:
        arr = np.asarray(x, dtype=float).reshape(self.dimension)
        hi = max(0, len(self.function_pool.terms) - 1)
        return tuple(int(np.clip(np.round(value), 0, hi)) for value in arr)

    def decode_expression(self, x: Sequence[float] | np.ndarray) -> dict[str, Any]:
        indices = self.decode_indices(x)
        selected = [self.function_pool.terms[index] for index in indices]
        expression = param_expr("task_bias", init=0.0)
        for pos, term in enumerate(selected):
            term_expr = dict(term.expr)
            if bool(self.config.parameterize_terms):
                term_expr = parameterize_expression(
                    term_expr,
                    prefix=f"task{pos}",
                    output_scale=True,
                    output_bias=False,
                    unary_input_affine=bool(self.config.unary_input_affine),
                )
            expression = binary_expr("add", expression, term_expr)
        return expression

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.evaluate_detailed(x).objectives, dtype=float)

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.evaluate_detailed(x).constraints, dtype=float)

    def evaluate_detailed(self, x: Sequence[float] | np.ndarray) -> BasisConditionedTaskEvaluationRecord:
        candidate = tuple(float(v) for v in np.asarray(x, dtype=float).reshape(-1))
        cache_key = tuple(self.decode_indices(candidate))
        cached = self.evaluation_cache.get(cache_key)
        if cached is not None:
            self.last_record = cached
            return cached

        expression = self.decode_expression(candidate)
        if self.graph_cache is not None:
            try:
                self.graph_cache.evaluate_expression(
                    expression,
                    self.data.X_train,
                    expr_key=None,
                    batch_key=f"stage2_{_task_kind(self.config)}_train",
                )
            except Exception:
                pass
        head = _build_head(self.config, self.original_data.y_train)
        representation = SymbolicExpressionRepresentation(
            SymbolicExpressionConfig(
                input_dim=int(self.data.n_features),
                expression=expression,
                name="basis_conditioned_task_expression",
                feature_names=tuple(self.data.effective_feature_names),
            ),
            head=head,
        )
        problem = _build_problem(self.data, self.config)
        adapter = _build_adapter(self.config)
        trainer = ComposableTrainer(
            problem=problem,
            representation=representation,
            adapter=adapter,
            run_name="inner_basis_conditioned_symbolic_fit",
            resource_context=self.resource_context,
        )
        result = trainer.fit(max_steps=int(self.config.inner_steps))
        if result.best_feedback is None:
            raise RuntimeError("inner basis-conditioned symbolic fit produced no feedback")
        metrics = dict(result.best_feedback.metrics)
        signals = dict(result.best_feedback.signals)
        complexity = float(expression_complexity(expression))
        duplicate_count = float(len(cache_key) - len(set(cache_key)))
        objectives = _outer_objectives_for_feedback(result.best_feedback, metrics, complexity, self.config)
        constraints = np.asarray([float(self.config.duplicate_penalty) * duplicate_count], dtype=float)
        selected_function_terms = tuple(self.function_pool.terms[index] for index in cache_key)
        selected_terms = tuple(term.describe(include_values=False) for term in selected_function_terms)
        score_report = self.candidate_scorer.score(
            objectives=objectives,
            constraints=constraints,
            selected_terms=selected_function_terms,
            metrics=metrics,
            expression=expression,
            record_memory=bool(self.path_memory is not None),
        )
        record = BasisConditionedTaskEvaluationRecord(
            outer_candidate=candidate,
            selected_indices=cache_key,
            selected_terms=selected_terms,
            expression=dict(expression),
            fitted_state=tuple([] if result.best_state is None else result.best_state.as_array().tolist()),
            objectives=tuple(float(v) for v in objectives),
            constraints=tuple(float(v) for v in constraints),
            metrics=metrics,
            signals=signals,
            report={
                "inner_report": dict(result.report),
                "representation": representation.describe(),
                "basis_artifact": self.basis_artifact.describe(include_record=False),
                "resource_context": dict(self.resource_context),
                "task_kind": _task_kind(self.config),
                "head_kind": _head_kind(self.config),
                "candidate_score": score_report.as_dict(),
                "graph_cache": None if self.graph_cache is None else self.graph_cache.snapshot(),
                "path_memory": None if self.path_memory is None else self.path_memory.describe(),
            },
        )
        self.last_record = record
        self.evaluation_cache[cache_key] = record
        self.evaluation_records.append(record)
        record_score = float(score_report.score)
        best_score = (
            float(dict(self.best_record.report.get("candidate_score", {})).get("score", np.sum(self.best_record.objectives) + np.sum(self.best_record.constraints)))
            if self.best_record is not None
            else float("inf")
        )
        if self.best_record is None or record_score < best_score:
            self.best_record = record
        return record

    def build_artifact(self, record: BasisConditionedTaskEvaluationRecord | None = None) -> SymbolicTaskArtifact:
        selected = record or self.best_record or self.last_record
        if selected is None:
            raise ValueError("no Stage 2 evaluation record is available")
        replay_record = SymbolicReplayRecordBuilder().build(
            selected,
            stage="basis_conditioned_symbolic_task",
            problem_name=self.name,
            resource_context=self.resource_context,
            extra_inputs={
                "task_kind": _task_kind(self.config),
                "head_kind": _head_kind(self.config),
                "task_terms": int(self.config.task_terms),
                "basis_artifact_id": self.basis_artifact.artifact_id,
            },
            metadata={"integration": "nsgablack_symbolic", "stage": "stage2"},
        ).as_dict()
        metadata: dict[str, Any] = {
            "basis_metrics": dict(self.basis_artifact.metrics),
            "basis_artifact": self.basis_artifact.describe(include_record=False),
            "resource_context": dict(self.resource_context),
            "candidate_score": dict(selected.report.get("candidate_score", {})),
            "graph_cache": dict(selected.report.get("graph_cache", {}) or {}),
            "path_memory": None if self.path_memory is None else self.path_memory.describe(),
            "task_config": {
                "task_kind": _task_kind(self.config),
                "head_kind": _head_kind(self.config),
                "classes": list(self.config.classes),
            },
            "stage_metadata": dict(self.config.metadata),
            "replay_record": replay_record,
        }
        try:
            audit = SymbolicExpressionAuditProducer().analyze(
                {"target": selected.expression},
                selected_terms=selected.selected_terms,
                feature_names=tuple(self.data.effective_feature_names),
                metadata={**dict(self.original_data.metadata), **dict(self.config.metadata)},
                X=self.data.X_train,
            ).as_dict()
            metadata.update(
                {
                    "simplification_trace": audit.get("simplification_trace", []),
                    "truth_contract_recovery": audit.get("truth_contract_recovery", {}),
                    "equivalence_expression_handling": audit.get("equivalence_expression_handling", {}),
                    "interference_feature_handling": audit.get("interference_feature_handling", {}),
                    "periodic_equivalence_disambiguation": audit.get("periodic_equivalence_disambiguation", {}),
                    "simplified_expressions": audit.get("simplified_expressions", {}),
                }
            )
        except Exception:
            pass
        try:
            consensus = SymbolicBasisConsensusAnalyzer().analyze((self.basis_artifact,), X=self.original_data.X_train).as_dict()
            metadata["basis_consensus"] = consensus
            metadata["basis_overlap_report"] = {
                "artifact_overlap": consensus.get("artifact_overlap", []),
                "value_overlap": consensus.get("value_overlap", {}),
            }
        except Exception:
            pass
        def _make_artifact(payload: Mapping[str, Any]) -> SymbolicTaskArtifact:
            return SymbolicTaskArtifact(
                name="basis_conditioned_symbolic_task",
                expression=dict(selected.expression),
                fitted_state=tuple(float(v) for v in selected.fitted_state),
                metrics=dict(selected.metrics),
                objectives=tuple(float(v) for v in selected.objectives),
                constraints=tuple(float(v) for v in selected.constraints),
                selected_indices=tuple(int(v) for v in selected.selected_indices),
                selected_terms=tuple(dict(term) for term in selected.selected_terms),
                task_kind=_task_kind(self.config),
                head_kind=_head_kind(self.config),
                objective_names=tuple(str(v) for v in self.objectives),
                metadata=payload,
            )

        task_artifact = _make_artifact(metadata)
        try:
            fold_report = SymbolicFoldEvaluator().evaluate_task_artifact(task_artifact, self.data).as_dict()
            metadata = {**metadata, "fold_report": fold_report}
        except Exception:
            metadata = {**metadata}
        task_artifact = _make_artifact(metadata)
        if bool(self.config.enable_branch_report):
            try:
                branch_report = SymbolicBranchEvaluator(
                    SymbolicBranchEvaluationConfig(
                        auto_quantile_feature_indices=tuple(int(v) for v in self.config.branch_report_feature_indices),
                        auto_quantiles=tuple(float(v) for v in self.config.branch_report_quantiles),
                        min_branch_size=int(self.config.branch_report_min_size),
                        enable_branch_refit=bool(self.config.enable_branch_refit_report),
                        branch_refit_steps=int(self.config.branch_refit_steps),
                    )
                ).evaluate_task_artifact(
                    task_artifact,
                    self.data,
                    branch_data=self.original_data,
                    resource_context=self.resource_context,
                ).as_dict()
                metadata = {**metadata, "branch_report": branch_report}
            except Exception:
                metadata = {**metadata}
        return _make_artifact(metadata)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dimension": int(self.dimension),
            "objectives": list(self.objectives),
            "bounds": dict(self.bounds),
            "config": {
                "task_kind": _task_kind(self.config),
                "head_kind": _head_kind(self.config),
                "task_terms": int(self.config.task_terms),
                "pool_max_terms": int(self.config.pool_max_terms),
                "inner_steps": int(self.config.inner_steps),
                "learning_rate": float(self.config.learning_rate),
                "inner_adapter": str(self.config.inner_adapter),
                "parameterize_terms": bool(self.config.parameterize_terms),
                "complexity_weight": float(self.config.complexity_weight),
                "enable_path_memory": bool(self.config.enable_path_memory),
                "enable_graph_cache": bool(self.config.enable_graph_cache),
            },
            "basis_artifact": self.basis_artifact.describe(include_record=False),
            "function_pool": self.function_pool.describe(include_values=False),
            "path_memory": None if self.path_memory is None else self.path_memory.describe(),
            "graph_cache": None if self.graph_cache is None else self.graph_cache.describe(),
            "evaluation_cache": {
                "record_count": int(len(self.evaluation_records)),
                "unique_candidate_count": int(len(self.evaluation_cache)),
            },
            "contract": self.get_contract().describe(),
        }


def _task_kind(config: BasisConditionedSymbolicTaskConfig) -> str:
    key = str(config.task_kind or "regression").strip().lower()
    if key in {"point", "regression", "regressor"}:
        return "regression"
    if key in {"interval", "interval_regression"}:
        return "interval"
    if key in {"classification", "classifier", "probability", "probabilistic"}:
        return "classification"
    raise ValueError(f"unsupported symbolic task_kind: {config.task_kind}")


def _head_kind(config: BasisConditionedSymbolicTaskConfig) -> str:
    key = str(config.head_kind or "point").strip().lower()
    aliases = {
        "interval": "interval_center_radius",
        "center_radius": "interval_center_radius",
        "two_model": "interval_two_model",
        "binary": "binary_logistic",
        "logistic": "binary_logistic",
        "probability": "binary_logistic",
        "calibration": "probability_calibration",
    }
    return aliases.get(key, key)


def _objective_names_for_config(config: BasisConditionedSymbolicTaskConfig) -> tuple[str, ...]:
    if config.objective_names:
        return tuple(str(v) for v in config.objective_names)
    task = _task_kind(config)
    if task == "interval":
        return ("task_interval_objective", "task_interval_width", "task_interval_miss", "task_complexity")
    if task == "classification":
        names = tuple(f"task_{name}" for name in config.classification_objective_metrics)
        return (*names, "task_complexity")
    return ("task_rmse", "task_mae", "task_complexity", "task_generalization_gap")


def _build_head(config: BasisConditionedSymbolicTaskConfig, y: np.ndarray | None) -> Any:
    kind = _head_kind(config)
    if kind == "point":
        return PointHead()
    if kind == "interval_center_radius":
        return CenterRadiusIntervalHead()
    if kind == "interval_two_model":
        return TwoModelIntervalHead()
    classes = tuple(config.classes) or _classes_from_y(y)
    if kind == "binary_logistic":
        return BinaryLogisticHead(
            temperature=float(config.probability_temperature),
            threshold=float(config.probability_threshold),
            classes=classes if len(classes) >= 2 else (0, 1),
        )
    if kind == "softmax":
        classes = classes if len(classes) >= 2 else (0, 1)
        return SoftmaxHead(
            n_classes=len(classes),
            temperature=float(config.probability_temperature),
            classes=classes,
        )
    if kind == "probability_calibration":
        return ProbabilityCalibrationHead()
    raise ValueError(f"unsupported symbolic head_kind: {config.head_kind}")


def _build_problem(data: NumericDataView, config: BasisConditionedSymbolicTaskConfig) -> Any:
    task = _task_kind(config)
    if task == "interval":
        return SupervisedIntervalRegressionProblem(
            data,
            target_coverage=float(config.target_coverage),
            width_weight=float(config.interval_width_weight),
            miss_weight=float(config.interval_miss_weight),
            use_valid_objective=True,
        )
    if task == "classification":
        return SupervisedClassificationProblem(
            data,
            use_valid_objective=True,
            complexity_weight=0.0,
            objective_metrics=tuple(config.classification_objective_metrics),
            positive_label=config.positive_label,
        )
    return FixedSymbolicRegressionProblem(
        data,
        complexity_weight=float(config.complexity_weight),
        use_valid_objective=True,
    )


def _build_adapter(config: BasisConditionedSymbolicTaskConfig) -> Any:
    adapter = str(config.inner_adapter or "auto").strip().lower()
    task = _task_kind(config)
    if adapter == "gd" or adapter == "gradient_descent" or (adapter == "auto" and task == "regression" and _head_kind(config) == "point"):
        return GradientDescentAdapter(learning_rate=float(config.learning_rate), require_gradient=True)
    if adapter in {"random", "random_search", "auto"}:
        return RandomSearchAdapter(
            population_size=int(config.inner_population_size),
            mutation_scale=float(config.inner_mutation_scale),
            random_seed=int(config.random_seed),
        )
    raise ValueError(f"unsupported inner_adapter: {config.inner_adapter}")


def _outer_objectives_for_feedback(feedback: Any, metrics: Mapping[str, Any], complexity: float, config: BasisConditionedSymbolicTaskConfig) -> np.ndarray:
    task = _task_kind(config)
    if task == "interval":
        base = np.asarray(feedback.objectives, dtype=float).reshape(-1)
        values = list(base[:3])
        while len(values) < 3:
            values.append(1e12)
        values.append(float(config.complexity_weight) * float(complexity))
        return np.asarray(values, dtype=float)
    if task == "classification":
        base = list(np.asarray(feedback.objectives, dtype=float).reshape(-1))
        base.append(float(config.complexity_weight) * float(complexity))
        return np.asarray(base, dtype=float)
    rmse = float(metrics.get("valid.rmse", metrics.get("train.rmse", 1e12)))
    mae = float(metrics.get("valid.mae", metrics.get("train.mae", 1e12)))
    train_rmse = float(metrics.get("train.rmse", rmse))
    gap = max(0.0, rmse - train_rmse)
    return np.asarray([rmse, mae, float(config.complexity_weight) * complexity, gap], dtype=float)


def _classes_from_y(y: np.ndarray | None) -> tuple[Any, ...]:
    if y is None:
        return (0, 1)
    values = tuple(np.unique(np.asarray(y).reshape(-1)).tolist())
    return values if len(values) >= 2 else (0, 1)


__all__ = [
    "BasisConditionedSymbolicTaskConfig",
    "BasisConditionedSymbolicTaskProblem",
    "BasisConditionedTaskEvaluationRecord",
]

