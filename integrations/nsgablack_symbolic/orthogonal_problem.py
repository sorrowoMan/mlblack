from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.adapters import RandomSearchAdapter
from mlblack.core import ComposableTrainer
from blackbase.contracts import ComponentContract, ContractMixin
from mlblack.models.symbolic import parameterize_expression
from mlblack.pipeline.data_views import NumericDataView
from mlblack.pipeline.symbolic import FunctionPool, FunctionPoolPipeline, FunctionPoolPipelineConfig
from mlblack.problems import OrthogonalBasisEvaluationProblem
from mlblack.representations import SymbolicBasisSetConfig, SymbolicBasisSetRepresentation
from .artifacts import OrthogonalBasisSetArtifact
from .basis_consensus import SymbolicBasisConsensusAnalyzer
from .expression_audit import SymbolicExpressionAuditProducer
from .graph_cache import ExpressionGraphCache
from .path_memory import SymbolicPathMemory
from .replay import SymbolicReplayRecordBuilder
from .search_policy import CandidateScoreConfig, SymbolicCandidateScorer

try:  # optional integration dependency
    from nsgablack.core.base import BlackBoxProblem as _BlackBoxProblem
except Exception:  # pragma: no cover - used when nsgablack is not installed

    class _BlackBoxProblem:  # type: ignore[no-redef]
        def __init__(self, *, name: str, dimension: int, bounds: Mapping[str, Sequence[float]], objectives: Sequence[str]) -> None:
            self.name = name
            self.dimension = int(dimension)
            self.bounds = dict(bounds)
            self.objectives = tuple(objectives)


@dataclass(frozen=True)
class OrthogonalBasisOuterProblemConfig:
    basis_size: int = 3
    pool_max_terms: int = 64
    inner_steps: int = 40
    inner_population_size: int = 24
    inner_mutation_scale: float = 0.35
    random_seed: int = 42
    parameterize_terms: bool = True
    duplicate_penalty: float = 1.0
    rank_penalty: float = 1.0
    enable_path_memory: bool = False
    path_memory_db_path: str = ""
    path_memory_namespace: str = "symbolic_stage1"
    enable_graph_cache: bool = True
    graph_cache_backend: str = "memory"
    graph_cache_db_path: str = ""
    candidate_score_config: CandidateScoreConfig = field(default_factory=CandidateScoreConfig)
    objective_names: tuple[str, ...] = (
        "basis_max_abs_corr",
        "basis_condition",
        "basis_complexity",
        "basis_rank_deficit",
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrthogonalBasisEvaluationRecord:
    outer_candidate: tuple[float, ...]
    selected_indices: tuple[int, ...]
    selected_terms: tuple[dict[str, Any], ...]
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
            "fitted_state": list(self.fitted_state),
            "objectives": list(self.objectives),
            "constraints": list(self.constraints),
            "metrics": dict(self.metrics),
            "signals": dict(self.signals),
            "report": dict(self.report),
        }


class OrthogonalBasisOuterProblem(_BlackBoxProblem, ContractMixin):
    """nsgablack-compatible Stage 1 problem for symbolic basis search.

    Outer candidate:
      vector of function-pool indices.

    Inner mlblack run:
      fixed decoded basis structure -> parameter fitting -> orthogonality metrics.
    """

    name = "orthogonal_symbolic_basis_outer_problem"
    context_requires = ("data.X_train", "symbolic.function_pool", "resource.context")
    context_optional = ("data.y_train", "resource.lease", "signal.pool")
    context_provides = ("feedback.objectives", "basis.metrics", "artifact.symbolic_basis_ref", "symbolic.artifact")
    context_mutates = ("stage.audit",)
    context_cache = ("basis.fitted_ref",)
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Stage 1: outer structure search with inner mlblack basis-parameter fitting."
    contract = ComponentContract(
        name=name,
        requires=("data.X_train", "symbolic.function_pool", "resource.context"),
        optional=("data.y_train", "resource.lease", "signal.pool"),
        provides=("feedback.objectives", "basis.metrics", "artifact.symbolic_basis_ref", "symbolic.artifact"),
        mutates=("stage.audit",),
        cache=("basis.fitted_ref",),
        supports_batch=False,
        supports_resume=True,
        metadata={"integration": "nsgablack_symbolic", "stage": "orthogonal_basis"},
    )

    def __init__(
        self,
        data: NumericDataView,
        *,
        function_pool: FunctionPool | None = None,
        config: OrthogonalBasisOuterProblemConfig | None = None,
        resource_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.data = data
        self.config = config or OrthogonalBasisOuterProblemConfig()
        self.resource_context = dict(resource_context or {})
        self.function_pool = function_pool or FunctionPoolPipeline(
            FunctionPoolPipelineConfig(max_terms=int(self.config.pool_max_terms))
        ).build(
            data.X_train,
            y=data.y_train,
            feature_names=data.effective_feature_names,
        ).pool
        if not self.function_pool.terms:
            raise ValueError("function_pool must contain at least one term")
        self.last_record: OrthogonalBasisEvaluationRecord | None = None
        self.best_record: OrthogonalBasisEvaluationRecord | None = None
        self.evaluation_records: list[OrthogonalBasisEvaluationRecord] = []
        self.evaluation_cache: dict[tuple[int, ...], OrthogonalBasisEvaluationRecord] = {}
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
            for i in range(int(self.config.basis_size))
        }
        super().__init__(
            name=self.name,
            dimension=int(self.config.basis_size),
            bounds=bounds,
            objectives=tuple(self.config.objective_names),
        )

    def decode_indices(self, x: Sequence[float] | np.ndarray) -> tuple[int, ...]:
        arr = np.asarray(x, dtype=float).reshape(self.dimension)
        hi = max(0, len(self.function_pool.terms) - 1)
        return tuple(int(np.clip(np.round(value), 0, hi)) for value in arr)

    def decode_genome(self, x: Sequence[float] | np.ndarray) -> tuple[dict[str, Any], ...]:
        indices = self.decode_indices(x)
        genome: list[dict[str, Any]] = []
        for pos, index in enumerate(indices):
            term = self.function_pool.terms[int(index)]
            expr = dict(term.expr)
            if bool(self.config.parameterize_terms):
                expr = parameterize_expression(expr, prefix=f"basis{pos}")
            genome.append({"name": f"basis_{pos}_{term.name}", "expr": expr})
        return tuple(genome)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.evaluate_detailed(x).objectives, dtype=float)

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.evaluate_detailed(x).constraints, dtype=float)

    def evaluate_detailed(self, x: Sequence[float] | np.ndarray) -> OrthogonalBasisEvaluationRecord:
        candidate = tuple(float(v) for v in np.asarray(x, dtype=float).reshape(-1))
        cache_key = tuple(self.decode_indices(candidate))
        cached = self.evaluation_cache.get(cache_key)
        if cached is not None:
            self.last_record = cached
            return cached

        genome = self.decode_genome(candidate)
        if self.graph_cache is not None:
            for term in genome:
                try:
                    self.graph_cache.evaluate_expression(
                        term["expr"],
                        self.data.X_train,
                        expr_key=str(term.get("name", "")) or None,
                        batch_key="stage1_train",
                    )
                except Exception:
                    # Cache warming must not change search semantics.
                    pass
        representation = SymbolicBasisSetRepresentation(
            SymbolicBasisSetConfig(
                input_dim=int(self.data.n_features),
                genome=genome,
                feature_names=tuple(self.data.effective_feature_names),
            )
        )
        problem = OrthogonalBasisEvaluationProblem(self.data)
        adapter = RandomSearchAdapter(
            population_size=int(self.config.inner_population_size),
            mutation_scale=float(self.config.inner_mutation_scale),
            random_seed=int(self.config.random_seed),
        )
        trainer = ComposableTrainer(
            problem=problem,
            representation=representation,
            adapter=adapter,
            run_name="inner_orthogonal_basis_fit",
            resource_context=self.resource_context,
        )
        result = trainer.fit(max_steps=int(self.config.inner_steps))
        if result.best_feedback is None:
            raise RuntimeError("inner orthogonal basis fit produced no feedback")
        metrics = dict(result.best_feedback.metrics)
        signals = dict(result.best_feedback.signals)
        objectives = np.asarray(result.best_feedback.objectives, dtype=float).reshape(-1)
        rank = float(metrics.get("basis.rank", 0.0) or 0.0)
        rank_deficit = max(0.0, float(self.config.basis_size) - rank)
        duplicate_count = float(len(cache_key) - len(set(cache_key)))
        constraints = np.asarray(
            [
                duplicate_count,
                rank_deficit,
            ],
            dtype=float,
        )
        objective_with_rank = np.asarray(
            [
                float(objectives[0]) if objectives.size > 0 else 1.0,
                float(objectives[1]) if objectives.size > 1 else 0.0,
                float(objectives[2]) if objectives.size > 2 else 0.0,
                float(self.config.rank_penalty) * rank_deficit + float(self.config.duplicate_penalty) * duplicate_count,
            ],
            dtype=float,
        )
        selected_function_terms = tuple(self.function_pool.terms[index] for index in cache_key)
        selected_terms = tuple(term.describe(include_values=False) for term in selected_function_terms)
        score_report = self.candidate_scorer.score(
            objectives=objective_with_rank,
            constraints=constraints,
            selected_terms=selected_function_terms,
            metrics=metrics,
            record_memory=bool(self.path_memory is not None),
        )
        record = OrthogonalBasisEvaluationRecord(
            outer_candidate=candidate,
            selected_indices=cache_key,
            selected_terms=selected_terms,
            fitted_state=tuple([] if result.best_state is None else result.best_state.as_array().tolist()),
            objectives=tuple(float(v) for v in objective_with_rank),
            constraints=tuple(float(v) for v in constraints),
            metrics=metrics,
            signals=signals,
            report={
                "inner_report": dict(result.report),
                "representation": representation.describe(),
                "resource_context": dict(self.resource_context),
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
            float(dict(self.best_record.report.get("candidate_score", {})).get("score", np.sum(self.best_record.objectives)))
            if self.best_record is not None
            else float("inf")
        )
        if self.best_record is None or record_score < best_score:
            self.best_record = record
        return record

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dimension": int(self.dimension),
            "objectives": list(self.objectives),
            "bounds": dict(self.bounds),
            "config": {
                "basis_size": int(self.config.basis_size),
                "pool_max_terms": int(self.config.pool_max_terms),
                "inner_steps": int(self.config.inner_steps),
                "inner_population_size": int(self.config.inner_population_size),
                "inner_mutation_scale": float(self.config.inner_mutation_scale),
                "parameterize_terms": bool(self.config.parameterize_terms),
                "enable_path_memory": bool(self.config.enable_path_memory),
                "enable_graph_cache": bool(self.config.enable_graph_cache),
            },
            "function_pool": self.function_pool.describe(include_values=False),
            "path_memory": None if self.path_memory is None else self.path_memory.describe(),
            "graph_cache": None if self.graph_cache is None else self.graph_cache.describe(),
            "evaluation_cache": {
                "record_count": int(len(self.evaluation_records)),
                "unique_candidate_count": int(len(self.evaluation_cache)),
            },
            "contract": self.get_contract().describe(),
        }

    def build_artifact(self, record: OrthogonalBasisEvaluationRecord | None = None) -> OrthogonalBasisSetArtifact:
        selected = record or self.best_record or self.last_record
        if selected is None:
            raise ValueError("no Stage 1 evaluation record is available")
        replay_record = SymbolicReplayRecordBuilder().build(
            selected,
            stage="orthogonal_basis_search",
            problem_name=self.name,
            resource_context=self.resource_context,
            extra_inputs={
                "basis_size": int(self.config.basis_size),
                "pool_max_terms": int(self.config.pool_max_terms),
            },
            metadata={"integration": "nsgablack_symbolic", "stage": "stage1"},
        ).as_dict()
        artifact = OrthogonalBasisSetArtifact.from_stage1_record(
            selected,
            input_dim=int(self.data.n_features),
            feature_names=tuple(self.data.effective_feature_names),
            parameterize_terms=bool(self.config.parameterize_terms),
            metadata={
                "problem": self.name,
                "config": {
                    "basis_size": int(self.config.basis_size),
                    "pool_max_terms": int(self.config.pool_max_terms),
                    "inner_steps": int(self.config.inner_steps),
                },
                "resource_context": dict(self.resource_context),
                "stage_metadata": dict(self.config.metadata),
                "replay_record": replay_record,
                "candidate_score": dict(selected.report.get("candidate_score", {})),
                "graph_cache": dict(selected.report.get("graph_cache", {}) or {}),
                "path_memory": None if self.path_memory is None else self.path_memory.describe(),
            },
        )
        try:
            audit = SymbolicExpressionAuditProducer().analyze(
                artifact.basis_genome,
                selected_terms=artifact.selected_terms,
                feature_names=artifact.feature_names,
                metadata={**dict(self.data.metadata), **dict(self.config.metadata)},
                X=self.data.X_train,
            ).as_dict()
            artifact = replace(
                artifact,
                metadata={
                    **dict(artifact.metadata),
                    "simplification_trace": audit.get("simplification_trace", []),
                    "truth_contract_recovery": audit.get("truth_contract_recovery", {}),
                    "equivalence_expression_handling": audit.get("equivalence_expression_handling", {}),
                    "interference_feature_handling": audit.get("interference_feature_handling", {}),
                    "periodic_equivalence_disambiguation": audit.get("periodic_equivalence_disambiguation", {}),
                    "simplified_expressions": audit.get("simplified_expressions", {}),
                },
            )
        except Exception:
            pass
        try:
            consensus = SymbolicBasisConsensusAnalyzer().analyze((artifact,), X=self.data.X_train).as_dict()
            meta = {
                **dict(artifact.metadata),
                "basis_consensus": consensus,
                "basis_overlap_report": {
                    "artifact_overlap": consensus.get("artifact_overlap", []),
                    "value_overlap": consensus.get("value_overlap", {}),
                },
            }
            artifact = replace(artifact, metadata=meta)
        except Exception:
            pass
        return artifact
