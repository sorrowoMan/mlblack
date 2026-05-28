from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from mlblack.integrations.nsgablack_symbolic import (
    BasisConditionedSymbolicTaskConfig,
    BasisConditionedSymbolicTaskProblem,
    BasisConsensusConfig,
    CandidateScoreConfig,
    ExpressionGraphCache,
    OrthogonalBasisOuterProblem,
    OrthogonalBasisOuterProblemConfig,
    SymbolicBasisConsensusAnalyzer,
    SymbolicBranchEvaluationConfig,
    SymbolicBranchEvaluator,
    SymbolicExpressionAuditProducer,
    StructureGuardConfig,
    SymbolicFoldEvaluator,
    SymbolicCandidateScorer,
    SymbolicPathMemory,
    SymbolicStructureGuard,
    symbolic_artifact_schema_descriptor,
)
from mlblack.integrations.nsgablack_symbolic.artifacts import SYMBOLIC_ARTIFACT_SCHEMA_KEY
from mlblack.models import (
    binary_expr,
    const_expr,
    expression_canonical_string,
    expression_equivalence_key,
    expression_family_signature,
    feature_expr,
    unary_expr,
)
from mlblack.pipeline.data_views import NumericDataView


def _data(*, classification: bool = False) -> NumericDataView:
    rng = np.random.default_rng(31)
    X = rng.normal(size=(40, 3))
    y_float = 0.5 * np.sin(X[:, 0]) + 0.2 * X[:, 1] * X[:, 1] - 0.1 * X[:, 2]
    y = (y_float > float(np.median(y_float))).astype(int) if classification else y_float
    return NumericDataView(
        X_train=X[:30],
        y_train=y[:30],
        X_valid=X[30:],
        y_valid=y[30:],
        feature_names=("x0", "x1", "x2"),
    )


def _basis_artifact():
    data = _data()
    problem = OrthogonalBasisOuterProblem(
        data,
        config=OrthogonalBasisOuterProblemConfig(
            basis_size=2,
            pool_max_terms=14,
            inner_steps=2,
            inner_population_size=4,
            enable_graph_cache=False,
        ),
    )
    record = problem.evaluate_detailed(np.asarray([0.0, 1.0]))
    return problem.build_artifact(record)


def test_symbolic_artifact_schema_and_stage2_heads() -> None:
    basis = _basis_artifact()
    basis_schema = basis.schema().as_dict()
    assert basis_schema["schema_key"] == SYMBOLIC_ARTIFACT_SCHEMA_KEY
    assert "feature_usage" in basis_schema
    assert "assembler_structure" in basis_schema
    assert basis_schema["candidate_lineage"]["replay_record"]["stage"] == "orthogonal_basis_search"
    assert "basis_consensus" in basis_schema
    assert "basis_overlap_report" in basis_schema
    assert "canonical_expression" in basis_schema
    assert "family_recovery" in basis_schema
    assert "phase_equivalence_recovery" in basis_schema
    assert "branch_report" in symbolic_artifact_schema_descriptor()["fields"]
    assert "canonical_expression" in symbolic_artifact_schema_descriptor()["fields"]

    cases = (
        ("regression", "point", _data()),
        ("interval", "interval_center_radius", _data()),
        ("classification", "binary_logistic", _data(classification=True)),
    )
    for task_kind, head_kind, data in cases:
        problem = BasisConditionedSymbolicTaskProblem(
            data,
            basis_artifact=basis,
            config=BasisConditionedSymbolicTaskConfig(
                task_kind=task_kind,
                head_kind=head_kind,
                task_terms=2,
                pool_max_terms=14,
                inner_steps=2,
                inner_population_size=4,
                enable_graph_cache=False,
            ),
        )
        record = problem.evaluate_detailed(np.asarray([0.0, 1.0]))
        artifact = problem.build_artifact(record)
        schema = artifact.schema().as_dict()
        assert schema["schema_key"] == SYMBOLIC_ARTIFACT_SCHEMA_KEY
        assert schema["task_semantics"]["task_kind"] in {"regression", "interval", "classification"}
        assert schema["head_semantics"]["output_kind"] in {"point", "interval", "probability"}
        assert "feature_usage" in schema
        assert "term_contributions" in schema
        assert "assembler_structure" in schema
        assert schema["candidate_lineage"]["replay_record"]["stage"] == "basis_conditioned_symbolic_task"
        assert "evaluation_report" in schema
        assert "branch_report" in schema
        assert "basis_consensus" in schema
        assert "canonical_expression" in schema
        assert "family_recovery" in schema
        assert "candidate_score" in record.report


def test_path_memory_graph_cache_and_candidate_scoring(tmp_path: Path) -> None:
    memory = SymbolicPathMemory(db_path=str(tmp_path / "path_memory.sqlite3"), namespace="test")
    memory.touch_expr("x0")
    memory.record_expr_outcome("x0", selected_score=0.2, delta_score=0.1, success=True)
    assert memory.get_expr_prior("x0").accept_rate > 0.5

    expr = {"type": "unary", "op": "sin", "arg": {"type": "feature", "index": 0}}
    X = np.linspace(-1, 1, 8).reshape(-1, 1)
    cache = ExpressionGraphCache(enabled=True)
    values1 = cache.evaluate_expression(expr, X, batch_key="batch")
    values2 = cache.evaluate_expression(expr, X, batch_key="batch")
    derivative = cache.differentiate_wrt_feature(expr, feature_index=0)
    assert np.allclose(values1, values2)
    assert cache.stats().value_hits >= 1
    assert derivative["expr"]["type"] in {"binary", "unary", "const"}

    scorer = SymbolicCandidateScorer(CandidateScoreConfig(), path_memory=memory)
    report = scorer.score(
        objectives=np.asarray([0.1, 0.2]),
        selected_terms=({"name": "x0", "expr": {"type": "feature", "index": 0}, "complexity": 1.0, "prior_corr": 0.5},),
        metrics={"train.rmse": 0.2, "valid.rmse": 0.25},
    )
    assert np.isfinite(report.score)
    assert report.prior_summary["count"] == 1
    assert "structure_guard" in report.as_dict()
    memory.close()


def test_structure_guard_reports_symbolic_search_risks() -> None:
    values = np.linspace(-1.0, 1.0, 16)
    risky = {
        "name": "unsafe_ratio",
        "expr": {"type": "binary", "op": "div", "left": {"type": "feature", "index": 0}, "right": {"type": "feature", "index": 0}},
        "family": "interaction_rational",
        "activation_family": "interaction_rational",
        "features": (0,),
        "complexity": 30.0,
        "prior_corr": 0.1,
        "values": values,
    }
    duplicate = dict(risky)
    duplicate["values"] = values.copy()
    guard = SymbolicStructureGuard(
        StructureGuardConfig(
            max_complexity=8.0,
            max_feature_reuse=1,
            max_duplicate_terms=0,
            min_value_stability_score=0.9,
            min_pole_safety_score=0.9,
        )
    )
    report = guard.evaluate((risky, duplicate)).as_dict()
    assert report["triggered"] is True
    assert report["penalty"] > 0.0
    assert "seat_duplicate" in report["reasons"]
    assert "feature_reuse" in report["reasons"]


def test_basis_consensus_and_fold_evaluation_reports() -> None:
    data = _data()
    basis = _basis_artifact()
    consensus = SymbolicBasisConsensusAnalyzer(BasisConsensusConfig(min_support_ratio=1.0)).analyze(
        (basis,),
        X=data.X_train,
    )
    payload = consensus.as_dict()
    assert payload["artifact_count"] == 1
    assert payload["consensus_terms"]
    assert payload["semantic_overlap"]["family_frequency"]
    assert payload["value_overlap"]["available"] is True
    assert "atom_overlap_matrix" in payload["value_overlap"]

    problem = BasisConditionedSymbolicTaskProblem(
        data,
        basis_artifact=basis,
        config=BasisConditionedSymbolicTaskConfig(
            task_kind="regression",
            head_kind="point",
            task_terms=2,
            pool_max_terms=14,
            inner_steps=2,
            inner_population_size=4,
            enable_graph_cache=False,
        ),
    )
    record = problem.evaluate_detailed(np.asarray([0.0, 1.0]))
    artifact = problem.build_artifact(record)
    fold_report = SymbolicFoldEvaluator().evaluate_task_artifact(artifact, basis.to_basis_data_view(data)).as_dict()
    assert fold_report["fold_count"] >= 1
    assert any(key.startswith("rmse.") for key in fold_report["aggregate_metrics"])
    branch_report = SymbolicBranchEvaluator(
        SymbolicBranchEvaluationConfig(auto_quantile_feature_indices=(0,), min_branch_size=2, enable_branch_refit=True, branch_refit_steps=2)
    ).evaluate_task_artifact(artifact, basis.to_basis_data_view(data), branch_data=data).as_dict()
    assert branch_report["branch_count"] >= 2
    assert any(row["branch_name"] == "all" for row in branch_report["branch_metrics"])
    assert any(key.startswith("rmse.") for key in branch_report["aggregate_metrics"])
    assert any(row.get("branch_refit.status") == "ok" for row in branch_report["branch_metrics"])


def test_expression_audit_produces_simplification_and_truth_recovery() -> None:
    expr = {
        "type": "binary",
        "op": "add",
        "left": {"type": "const", "value": 0.0},
        "right": {"type": "unary", "op": "identity", "arg": {"type": "feature", "index": 0}},
    }
    report = SymbolicExpressionAuditProducer().analyze(
        {"target": expr},
        selected_terms=(
            {
                "name": "sin(x0)",
                "expr": {"type": "unary", "op": "sin", "arg": {"type": "feature", "index": 0}},
                "family": "unary_sin",
                "activation_family": "trig",
                "features": (0,),
            },
        ),
        feature_names=("x0",),
        metadata={"truth_contracts": ["sin(x0)", "x0"]},
        X=np.linspace(-1, 1, 12).reshape(-1, 1),
    ).as_dict()
    assert report["simplification_trace"]
    assert report["simplified_expressions"]["target"]["expression_string"] == "x0"
    assert report["truth_contract_recovery"]["matched_contract_count"] >= 1
    assert report["periodic_equivalence_disambiguation"]["periodic_term_count"] >= 1


def test_symbolic_engine_canonicalizes_equivalent_expressions() -> None:
    x0 = feature_expr(0)
    x1 = feature_expr(1)
    x2 = feature_expr(2)

    assert expression_equivalence_key(binary_expr("add", x0, x1)) == expression_equivalence_key(binary_expr("add", x1, x0))
    assert expression_equivalence_key(
        binary_expr("add", binary_expr("add", x0, x1), x2)
    ) == expression_equivalence_key(
        binary_expr("add", x0, binary_expr("add", x1, x2))
    )

    base_key = expression_equivalence_key(x0)
    assert expression_equivalence_key(binary_expr("mul", x0, const_expr(1.0))) == base_key
    assert expression_equivalence_key(binary_expr("add", const_expr(0.0), x0)) == base_key
    assert expression_equivalence_key(unary_expr("identity", x0)) == base_key
    assert expression_canonical_string(binary_expr("add", const_expr(0.0), unary_expr("identity", x0))) == "x0"

    pythag = binary_expr("add", unary_expr("square", unary_expr("sin", x0)), unary_expr("square", unary_expr("cos", x0)))
    assert expression_equivalence_key(pythag) == expression_equivalence_key(const_expr(1.0))
    assert expression_equivalence_key(binary_expr("mul", x0, x0)) == expression_equivalence_key(unary_expr("square", x0))
    assert expression_equivalence_key(binary_expr("div", x0, const_expr(2.0))) == expression_equivalence_key(binary_expr("mul", x0, const_expr(0.5)))


def test_symbolic_audit_scores_phase_and_family_recovery() -> None:
    x0 = feature_expr(0)
    report = SymbolicExpressionAuditProducer().analyze(
        {"phase_shifted": unary_expr("cos", x0)},
        feature_names=("x0",),
        metadata={"truth_contracts": ["sin(x0)"]},
        X=np.linspace(-1, 1, 12).reshape(-1, 1),
    ).as_dict()
    truth = report["truth_contract_recovery"]
    assert truth["matched_contract_count"] == 1
    assert truth["exact_term_recovery_score"] == 0.0
    assert truth["family_recovery_score"] == 1.0
    assert truth["phase_equivalence_recovery_score"] == 1.0
    assert truth["family_recovery"]["family_matched_contract_count"] == 1
    assert report["periodic_equivalence_disambiguation"]["phase_equivalence_policy"] == "scored"

    signature = expression_family_signature(unary_expr("cos", x0), feature_names=("x0",))
    assert signature["family"] == "trig"
    assert signature["phase_equivalence_key"].startswith("phase:trig:")

    exp_log = expression_family_signature(unary_expr("exp", unary_expr("log", x0)), feature_names=("x0",))
    assert exp_log["family"] == "exp_log_chain"
    ratio = expression_family_signature(binary_expr("div", x0, feature_expr(1)), feature_names=("x0", "x1"))
    assert ratio["family"] == "ratio"
    assert ratio["ratio_signature"]["numerator_features"] == [0]
    assert ratio["ratio_signature"]["denominator_features"] == [1]


def test_graph_cache_and_structure_guard_use_canonical_expression_keys() -> None:
    x0 = feature_expr(0)
    x1 = feature_expr(1)
    expr_a = binary_expr("add", x0, x1)
    expr_b = binary_expr("add", x1, x0)
    X = np.linspace(-1.0, 1.0, 16).reshape(8, 2)

    cache = ExpressionGraphCache(enabled=True)
    values_a = cache.evaluate_expression(expr_a, X, batch_key="same_batch")
    values_b = cache.evaluate_expression(expr_b, X, batch_key="same_batch")
    assert np.allclose(values_a, values_b)
    assert cache.stats().value_hits >= 1

    guard = SymbolicStructureGuard(StructureGuardConfig(max_duplicate_terms=0))
    report = guard.evaluate(
        (
            {"name": "x0", "expr": x0, "features": (0,), "complexity": 1.0},
            {"name": "identity_x0", "expr": unary_expr("identity", x0), "features": (0,), "complexity": 2.0},
        )
    ).as_dict()
    assert report["triggered"] is True
    assert "seat_duplicate" in report["reasons"]


def test_formal_case_check_runs() -> None:
    case = Path("examples") / "cases" / "symbolic_orthogonal_nested" / "run_solver.py"
    result = subprocess.run(
        [sys.executable, str(case), "--check"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=90,
        check=True,
    )
    assert "symbolic_orthogonal_nested scaffold ok" in result.stdout


