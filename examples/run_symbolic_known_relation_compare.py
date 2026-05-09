from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_project.known_relation_symbolic.orchestration import (
    build_known_relation_semantic_flow_spec,
    resolve_orthogonal_trainer_overrides,
)
from my_project.known_relation_symbolic.pipeline import build_known_relation_bundle
from my_project.known_relation_symbolic.problem import known_relation_benchmark_keys
from workflow import TrainDataBundle, run_semantic_train_flow


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(raw) for key, raw in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _artifact_expression(artifact: Any) -> str | None:
    if not hasattr(artifact, "expression"):
        return None
    try:
        return str(artifact.expression(target_index=0, precision=8, use_feature_names=True))
    except Exception:
        return None


def _normalized_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalized_feature_tuple(values: Any) -> tuple[str, ...]:
    normalized = [_normalized_text(value) for value in tuple(values or ())]
    return tuple(sorted(value for value in normalized if value))


def _expr_looks_like_safe_ratio(expr: str, *, numerator: str, denominator: str) -> bool:
    normalized = _normalized_text(expr)
    if not normalized:
        return False
    if "/safe(" in normalized:
        return numerator in normalized and denominator in normalized
    return (
        "/" in normalized
        and numerator in normalized
        and denominator in normalized
        and ("abs(" in normalized or "safe" in normalized)
    )


def _expr_looks_like_piecewise_hinge(expr: str, *, feature_name: str) -> bool:
    normalized = _normalized_text(expr)
    if not normalized or feature_name not in normalized:
        return False
    if "relu(" in normalized or "hinge" in normalized or "piecewise" in normalized:
        return True
    # Standard hinge reparameterization: 0.5 * (z + abs(z)).
    return ("abs(" in normalized) and ("0.5" in normalized)


def _first_list_mapping_value(mapping: Any) -> list[dict[str, Any]]:
    for value in dict(mapping or {}).values():
        if isinstance(value, list):
            return [dict(row or {}) for row in value]
    return []


def _term_row_view(row: dict[str, Any]) -> dict[str, Any]:
    expr = str(
        row.get("expression_named")
        or row.get("expression_raw")
        or row.get("expression")
        or row.get("expr")
        or ""
    )
    return {
        "term_name": str(row.get("term_name") or row.get("name") or ""),
        "name": _normalized_text(row.get("term_name") or row.get("name") or ""),
        "expr": _normalized_text(expr),
        "expression": expr,
        "features": _normalized_feature_tuple(row.get("feature_names", ())),
        "semantic_family": _normalized_text(row.get("semantic_family")),
        "semantic_signature": _normalized_text(row.get("semantic_signature")),
        "uses_piecewise_gate": bool(row.get("uses_piecewise_gate")),
        "coefficient": row.get("coefficient"),
        "abs_coefficient": row.get("abs_coefficient"),
        "normalized_weight": row.get("normalized_weight"),
        "node_count": row.get("node_count"),
    }


def _truth_basis_contract_specs(contracts: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for contract in contracts:
        normalized = _normalized_text(contract)
        if normalized == "safe_ratio(voltage,resistance)":
            specs.append(
                {
                    "contract": contract,
                    "family": "safe_ratio",
                    "features": ("resistance", "voltage"),
                    "expected_sign": "positive",
                }
            )
        elif normalized == "sin(temperature)":
            specs.append(
                {
                    "contract": contract,
                    "family": "sin",
                    "features": ("temperature",),
                    "expected_sign": "positive",
                }
            )
        elif normalized == "piecewise_hinge(temperature)":
            specs.append(
                {
                    "contract": contract,
                    "family": "piecewise_hinge",
                    "features": ("temperature",),
                    "expected_sign": None,
                }
            )
        elif normalized == "material_bias":
            specs.append(
                {
                    "contract": contract,
                    "family": "linear_feature",
                    "features": ("material_bias",),
                    "expected_sign": "negative",
                }
            )
        else:
            specs.append(
                {
                    "contract": contract,
                    "family": normalized or "unknown",
                    "features": (),
                    "expected_sign": None,
                }
            )
    return specs


def _matches_exact_truth_contract(contract_spec: dict[str, Any], row: dict[str, Any]) -> bool:
    family = str(contract_spec.get("family") or "")
    features = tuple(contract_spec.get("features", ()))
    if tuple(row.get("features", ())) != features:
        return False
    expr = str(row.get("expr") or "")
    name = str(row.get("name") or "")
    semantic_family = str(row.get("semantic_family") or "")
    semantic_signature = str(row.get("semantic_signature") or "")
    if family == "safe_ratio":
        numerator = str(features[1] if len(features) > 1 else "voltage")
        denominator = str(features[0] if len(features) > 0 else "resistance")
        return (
            ("/safe(" in expr)
            or ("/safe(" in name)
            or _expr_looks_like_safe_ratio(expr, numerator=numerator, denominator=denominator)
            or _expr_looks_like_safe_ratio(name, numerator=numerator, denominator=denominator)
            or ("ratio" in semantic_family)
            or ("binary:div" in semantic_signature)
        )
    if family == "sin":
        return ("sin(" in expr) or ("sin(" in name) or ("unary:sin" in semantic_signature)
    if family == "piecewise_hinge":
        feature_name = str(features[0] if len(features) > 0 else "")
        return bool(
            row.get("uses_piecewise_gate")
            or ("piecewise" in semantic_family)
            or ("hinge" in name)
            or ("relu(" in expr)
            or _expr_looks_like_piecewise_hinge(expr, feature_name=feature_name)
            or _expr_looks_like_piecewise_hinge(name, feature_name=feature_name)
        )
    if family == "linear_feature":
        node_count = row.get("node_count")
        return bool(
            semantic_family == "linear_feature"
            or semantic_signature.startswith("feature:")
            or expr == "material_bias"
            or node_count == 1
        )
    return False


def _row_sign_matches(expected_sign: str | None, row: dict[str, Any]) -> bool:
    coefficient = row.get("coefficient")
    if expected_sign is None or coefficient is None:
        return True
    try:
        numeric = float(coefficient)
    except (TypeError, ValueError):
        return True
    if expected_sign == "positive":
        return numeric > 0.0
    if expected_sign == "negative":
        return numeric < 0.0
    return True


def _row_is_materially_active(row: dict[str, Any], *, min_normalized_weight: float) -> bool:
    weight = row.get("normalized_weight")
    if weight is not None:
        try:
            return float(weight) >= float(min_normalized_weight)
        except (TypeError, ValueError):
            pass
    coefficient = row.get("abs_coefficient")
    if coefficient is None:
        coefficient = row.get("coefficient")
    try:
        return abs(float(coefficient)) > 1e-10
    except (TypeError, ValueError):
        return False


def _exact_recovery_summary(
    *,
    truth_contracts: list[str],
    basis_rows: list[dict[str, Any]],
    active_term_rows: list[dict[str, Any]],
    min_normalized_weight: float,
) -> dict[str, Any]:
    contract_specs = _truth_basis_contract_specs(truth_contracts)
    contract_matches: list[dict[str, Any]] = []
    matched_basis_count = 0
    matched_term_count = 0
    for spec in contract_specs:
        basis_matches = [row for row in basis_rows if _matches_exact_truth_contract(spec, row)]
        active_matches = [
            row
            for row in active_term_rows
            if _matches_exact_truth_contract(spec, row)
            and _row_is_materially_active(row, min_normalized_weight=min_normalized_weight)
            and _row_sign_matches(str(spec.get("expected_sign") or ""), row)
        ]
        basis_hit = bool(basis_matches)
        term_hit = bool(active_matches)
        matched_basis_count += int(basis_hit)
        matched_term_count += int(term_hit)
        contract_matches.append(
            {
                "truth_term": str(spec.get("contract") or ""),
                "truth_family": str(spec.get("family") or ""),
                "truth_features": list(spec.get("features", ())),
                "expected_sign": spec.get("expected_sign"),
                "basis_hit": basis_hit,
                "term_recovered": term_hit,
                "matched_basis_terms": [str(row.get("term_name") or row.get("expression") or "") for row in basis_matches],
                "matched_basis_expressions": [str(row.get("expression") or "") for row in basis_matches],
                "matched_expression_terms": [str(row.get("expression") or "") for row in active_matches],
                "matched_expression_coefficients": [row.get("coefficient") for row in active_matches],
            }
        )
    truth_count = len(contract_specs)
    return {
        "truth_basis_count": truth_count,
        "matched_truth_basis_count": matched_basis_count,
        "matched_truth_term_count": matched_term_count,
        "exact_basis_hit_score": (
            None if truth_count <= 0 else float(matched_basis_count) / float(truth_count)
        ),
        "exact_term_recovery_score": (
            None if truth_count <= 0 else float(matched_term_count) / float(truth_count)
        ),
        "exact_term_min_normalized_weight": float(min_normalized_weight),
        "truth_basis_matches": contract_matches,
    }


def _basis_recovery_summary(artifact: Any) -> dict[str, Any]:
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    schema = dict(metadata.get("symbolic_artifact_schema", {}) or {})
    basis_structure = dict(schema.get("basis_structure", {}) or {})
    piecewise_gate = dict(schema.get("piecewise_gate_basis", {}) or {})
    orthogonality = dict(basis_structure.get("orthogonality_status", {}) or {})
    basis_terms = list(dict(basis_structure.get("basis_semantics", {}) or {}).get("recorded", {}).get("basis_terms", []) or [])
    active_term_rows = _first_list_mapping_value(schema.get("term_contributions", {}))
    basis_names = [str(row.get("term_name") or row.get("expression") or "") for row in basis_terms]
    basis_expr = [str(row.get("expression") or "") for row in basis_terms]
    basis_features = [
        tuple(str(value) for value in tuple(row.get("feature_names", ()) or ()))
        for row in basis_terms
    ]
    basis_semantic_families = [str(row.get("semantic_family") or "") for row in basis_terms]
    row_views = [
        {
            "term_name": str(name),
            "expression": str(expr),
            "name": str(name).lower(),
            "expr": str(expr).lower(),
            "features": _normalized_feature_tuple(features),
            "semantic_family": str(family).lower(),
        }
        for name, expr, features, family in zip(
            basis_names,
            basis_expr,
            basis_features,
            basis_semantic_families,
            strict=False,
        )
    ]
    truth_recovery = dict(schema.get("truth_contract_recovery", {}) or {})
    return {
        "basis_count": int(basis_structure.get("basis_count", 0) or 0),
        "basis_terms": basis_names,
        "basis_expressions": basis_expr,
        "basis_semantic_families": basis_semantic_families,
        "basis_features": basis_features,
        "orthogonality_status": str(orthogonality.get("status") or basis_structure.get("orthogonality_status") or ""),
        "orthogonality_score": orthogonality.get("orthogonality_score"),
        "pair_abs_corr_mean": orthogonality.get("pair_abs_corr_mean"),
        "residual_gain_mean": orthogonality.get("residual_gain_mean"),
        "semantic_unique_ratio": orthogonality.get("semantic_unique_ratio"),
        "piecewise_gate_status": str(piecewise_gate.get("status") or ""),
        "gate_basis_count": int(piecewise_gate.get("gate_basis_count", 0) or 0),
        "gate_feature_names": list(piecewise_gate.get("gate_feature_names", ()) or ()),
        "contains_ratio_basis": any(
            ("/safe(" in row["expr"])
            or ("reciprocal_safe" in row["expr"])
            or ("ratio" in row["semantic_family"])
            for row in row_views
        ),
        "contains_periodic_basis": any(
            ("periodic" in row["semantic_family"])
            or ("sin(" in row["expr"])
            or ("cos(" in row["expr"])
            for row in row_views
        ),
        "contains_piecewise_basis": (
            any(
                (("piecewise" in row["semantic_family"]) or ("hinge" in row["name"]))
                for row in row_views
            )
            or bool(tuple(piecewise_gate.get("gate_feature_names", ()) or ()))
        ),
        "contains_linear_basis": any(row["semantic_family"] == "linear_feature" for row in row_views),
        **truth_recovery,
        "active_term_count": int(len(active_term_rows)),
    }


def _run_one(
    *,
    bundle: TrainDataBundle,
    trainer_params: dict[str, Any],
    run_name: str,
    output_dir: Path,
    db_path: str,
    namespace: str,
    tag: str,
) -> dict[str, Any]:
    spec = build_known_relation_semantic_flow_spec(
        trainer_params=trainer_params,
        run_name=str(run_name),
        output_dir=str(output_dir),
        db_path=str(db_path),
        namespace=str(namespace),
        tag=str(tag),
    )
    result = run_semantic_train_flow(bundle, spec=spec)
    tracker = dict(result.report.get("experiment_tracker", {}) or {})
    artifact = result.artifact
    return {
        "run_name": str(run_name),
        "trainer_name": str(result.report.get("trainer_name") or getattr(artifact, "artifact_id", "")),
        "output_dir": str(result.output_dir),
        "metrics": _jsonable(result.metrics),
        "tracker": _jsonable(tracker),
        "artifact_id": str(getattr(artifact, "artifact_id", "")),
        "family_signature": str(dict(getattr(artifact, "metadata", {}) or {}).get("symbolic_family_signature", "")),
        "final_expression": _artifact_expression(artifact),
        "basis_recovery": _basis_recovery_summary(artifact),
        "run_id": str(tracker.get("run_id", "")),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a formal symbolic family comparison on a registered known-relation benchmark "
            "and materialize both baseline and orthogonal runs into the experiment DB."
        ),
    )
    parser.add_argument("--scenario", type=str, choices=known_relation_benchmark_keys(), default="ohm_like")
    parser.add_argument("--db-path", type=str, default=str(ROOT / "runs" / "known_relation_symbolic_compare.sqlite3"))
    parser.add_argument("--namespace", type=str, default="known_relation_symbolic_compare")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(ROOT / "examples" / "out" / "known_relation_symbolic_family_compare"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-total", type=int, default=2400)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--noise-std", type=float, default=0.025)
    parser.add_argument("--baseline-max-added-terms", type=int, default=10)
    parser.add_argument("--baseline-topk-features", type=int, default=5)
    parser.add_argument("--orth-candidate-limit", type=int, default=160)
    parser.add_argument("--orth-group-count", type=int, default=16)
    parser.add_argument("--orth-min-basis-count", type=int, default=3)
    parser.add_argument("--orth-max-basis-count", type=int, default=8)
    parser.add_argument("--orth-selection-mode", type=str, default="rmse_first")
    parser.add_argument("--orth-assembler-max-added-terms", type=int, default=4)
    parser.add_argument("--orth-assembler-topk-features", type=int, default=4)
    parser.add_argument("--orth-assembler-max-pair-terms", type=int, default=8)
    parser.add_argument("--orth-assembler-max-candidates-per-iter", type=int, default=96)
    parser.add_argument("--orth-assembler-candidate-keep-top", type=int, default=6)
    parser.add_argument("--orth-assembler-max-expr-depth", type=int, default=6)
    parser.add_argument("--orth-assembler-ridge-l2", type=float, default=1e-4)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).resolve() / stamp
    output_root.mkdir(parents=True, exist_ok=True)

    benchmark, bundle, truth_payload = build_known_relation_bundle(
        benchmark_key=str(args.scenario),
        n_total=int(args.n_total),
        train_ratio=float(args.train_ratio),
        noise_std=float(args.noise_std),
        seed=int(args.seed),
    )
    scenario_tag = str(args.tag or benchmark.key)
    search_hints = dict(bundle.metadata.get("search_hints", {}) or {})
    trainer_overrides = _resolve_orthogonal_trainer_overrides(bundle.metadata)
    gate_feature_names = tuple(str(value) for value in tuple(search_hints.get("gate_feature_names", ()) or ()))
    periodic_feature_names = tuple(
        str(value) for value in tuple(search_hints.get("periodic_feature_names", ()) or ())
    )
    enable_piecewise_basis = bool(search_hints.get("enable_piecewise_basis"))

    baseline_params = {
        "parameter_backend": "ridge",
        "task": "point",
        "structure_engine": {
            "structure_mode": "stagewise_search",
            "search_driver": "nsgablack",
            "dynamic_pool_enabled": True,
        },
        "artifact_id": f"known_relation_symbolic_stagewise_{stamp}",
        "force_linear_base": "auto",
        "keep_search_trace": True,
        "search_max_added_terms": int(args.baseline_max_added_terms),
        "search_topk_features": int(args.baseline_topk_features),
        "search_max_pair_terms": 12,
        "search_max_candidates_per_iter": 256,
        "search_candidate_keep_top": 8,
        "search_include_hinge": bool(enable_piecewise_basis),
        "search_hinge_quantiles": (0.25, 0.50, 0.75) if enable_piecewise_basis else tuple(),
        "search_unary_ops": ("square", "sin", "cos", "tanh"),
        "search_nested_unary_patterns": ("sin(square)", "cos(square)"),
        "search_online_beam_enabled": False,
        "search_path_memory_enabled": False,
        "search_graph_cache_enabled": False,
        "search_joint_bundle_enabled": False,
    }
    orthogonal_params = {
        "parameter_backend": "ridge",
        "task": "point",
        "structure_engine": {
            "structure_mode": "orthogonal_basis_search",
            "search_driver": "orthogonal_basis",
            "dynamic_pool_enabled": True,
            "metadata": {"supports_piecewise_basis": True},
        },
        "artifact_id": f"known_relation_symbolic_orthogonal_{stamp}",
        "candidate_limit": int(args.orth_candidate_limit),
        "group_count": int(args.orth_group_count),
        "seed_candidate_count": 18,
        "min_basis_count": int(args.orth_min_basis_count),
        "max_basis_count": int(args.orth_max_basis_count),
        "selection_mode": str(args.orth_selection_mode),
        "random_seed": int(args.seed),
        "enable_piecewise_basis": bool(enable_piecewise_basis),
        "gate_feature_names": gate_feature_names,
        "periodic_feature_names": periodic_feature_names,
        "gate_quantiles": (0.25, 0.5, 0.75) if enable_piecewise_basis else tuple(),
        "assembler_max_added_terms": int(args.orth_assembler_max_added_terms),
        "assembler_topk_features": int(args.orth_assembler_topk_features),
        "assembler_max_pair_terms": int(args.orth_assembler_max_pair_terms),
        "assembler_max_candidates_per_iter": int(args.orth_assembler_max_candidates_per_iter),
        "assembler_candidate_keep_top": int(args.orth_assembler_candidate_keep_top),
        "assembler_max_expr_depth": int(args.orth_assembler_max_expr_depth),
        "assembler_ridge_l2": float(args.orth_assembler_ridge_l2),
        "search_graph_cache_enabled": False,
    }
    orthogonal_params.update(dict(trainer_overrides))

    baseline = _run_one(
        bundle=bundle,
        trainer_params=baseline_params,
        run_name=f"{args.namespace}_baseline_{stamp}",
        output_dir=output_root / "baseline",
        db_path=str(args.db_path),
        namespace=str(args.namespace),
        tag=f"{scenario_tag}:baseline",
    )
    orthogonal = _run_one(
        bundle=bundle,
        trainer_params=orthogonal_params,
        run_name=f"{args.namespace}_orthogonal_{stamp}",
        output_dir=output_root / "orthogonal",
        db_path=str(args.db_path),
        namespace=str(args.namespace),
        tag=f"{scenario_tag}:orthogonal",
    )

    baseline_test = dict(baseline.get("metrics", {})).get("test", {}) or {}
    orthogonal_test = dict(orthogonal.get("metrics", {})).get("test", {}) or {}
    comparison = {
        "delta_test_rmse": (
            None
            if baseline_test.get("rmse") is None or orthogonal_test.get("rmse") is None
            else float(orthogonal_test["rmse"]) - float(baseline_test["rmse"])
        ),
        "delta_test_r2": (
            None
            if baseline_test.get("r2") is None or orthogonal_test.get("r2") is None
            else float(orthogonal_test["r2"]) - float(baseline_test["r2"])
        ),
        "delta_exact_basis_hit_score": (
            None
            if dict(baseline.get("basis_recovery", {})).get("exact_basis_hit_score") is None
            or dict(orthogonal.get("basis_recovery", {})).get("exact_basis_hit_score") is None
            else float(dict(orthogonal.get("basis_recovery", {})).get("exact_basis_hit_score"))
            - float(dict(baseline.get("basis_recovery", {})).get("exact_basis_hit_score"))
        ),
        "delta_exact_term_recovery_score": (
            None
            if dict(baseline.get("basis_recovery", {})).get("exact_term_recovery_score") is None
            or dict(orthogonal.get("basis_recovery", {})).get("exact_term_recovery_score") is None
            else float(dict(orthogonal.get("basis_recovery", {})).get("exact_term_recovery_score"))
            - float(dict(baseline.get("basis_recovery", {})).get("exact_term_recovery_score"))
        ),
        "delta_phase_equivalent_term_recovery_score": (
            None
            if dict(baseline.get("basis_recovery", {})).get("phase_equivalent_term_recovery_score") is None
            or dict(orthogonal.get("basis_recovery", {})).get("phase_equivalent_term_recovery_score") is None
            else float(dict(orthogonal.get("basis_recovery", {})).get("phase_equivalent_term_recovery_score"))
            - float(dict(baseline.get("basis_recovery", {})).get("phase_equivalent_term_recovery_score"))
        ),
        "delta_family_level_term_recovery_score": (
            None
            if dict(baseline.get("basis_recovery", {})).get("family_level_term_recovery_score") is None
            or dict(orthogonal.get("basis_recovery", {})).get("family_level_term_recovery_score") is None
            else float(dict(orthogonal.get("basis_recovery", {})).get("family_level_term_recovery_score"))
            - float(dict(baseline.get("basis_recovery", {})).get("family_level_term_recovery_score"))
        ),
        "baseline_exact_basis_hit_score": dict(baseline.get("basis_recovery", {})).get("exact_basis_hit_score"),
        "baseline_exact_term_recovery_score": dict(baseline.get("basis_recovery", {})).get("exact_term_recovery_score"),
        "baseline_phase_equivalent_term_recovery_score": dict(baseline.get("basis_recovery", {})).get("phase_equivalent_term_recovery_score"),
        "baseline_family_level_term_recovery_score": dict(baseline.get("basis_recovery", {})).get("family_level_term_recovery_score"),
        "orthogonal_exact_basis_hit_score": dict(orthogonal.get("basis_recovery", {})).get("exact_basis_hit_score"),
        "orthogonal_exact_term_recovery_score": dict(orthogonal.get("basis_recovery", {})).get("exact_term_recovery_score"),
        "orthogonal_phase_equivalent_term_recovery_score": dict(orthogonal.get("basis_recovery", {})).get("phase_equivalent_term_recovery_score"),
        "orthogonal_family_level_term_recovery_score": dict(orthogonal.get("basis_recovery", {})).get("family_level_term_recovery_score"),
        "orthogonal_basis_recovery": dict(orthogonal.get("basis_recovery", {}) or {}),
    }

    summary = {
        "generated_at": datetime.now().isoformat(),
        "db_path": str(args.db_path),
        "namespace": str(args.namespace),
        "scenario": str(benchmark.key),
        "tag": str(scenario_tag),
        "output_root": str(output_root),
        "dataset": {
            "n_total": int(args.n_total),
            "train_ratio": float(args.train_ratio),
            "noise_std": float(args.noise_std),
            "seed": int(args.seed),
            "metadata": _jsonable(bundle.metadata),
        },
        "truth": _jsonable(truth_payload),
        "baseline": _jsonable(baseline),
        "orthogonal": _jsonable(orthogonal),
        "comparison": _jsonable(comparison),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"KNOWN RELATION SYMBOLIC FAMILY COMPARE [{benchmark.key}]")
    print(f"summary={summary_path}")
    print(f"db_path={args.db_path}")
    print(
        "baseline   trainer={trainer} run_id={run_id} test_rmse={rmse:.6f} test_r2={r2:.6f}".format(
            trainer=str(baseline.get("trainer_name", "")),
            run_id=str(baseline.get("run_id", "")),
            rmse=float(dict(baseline.get("metrics", {})).get("test", {}).get("rmse", float("nan"))),
            r2=float(dict(baseline.get("metrics", {})).get("test", {}).get("r2", float("nan"))),
        )
    )
    print(
        "orthogonal trainer={trainer} run_id={run_id} test_rmse={rmse:.6f} test_r2={r2:.6f} "
        "orthogonality={orth} ratio_basis={ratio_hit} piecewise={piecewise_hit} "
        "exact_basis_hit={basis_hit:.3f} exact_term_recovery={term_hit:.3f} "
        "phase_term_recovery={phase_hit:.3f} family_term_recovery={family_hit:.3f}".format(
            trainer=str(orthogonal.get("trainer_name", "")),
            run_id=str(orthogonal.get("run_id", "")),
            rmse=float(dict(orthogonal.get("metrics", {})).get("test", {}).get("rmse", float("nan"))),
            r2=float(dict(orthogonal.get("metrics", {})).get("test", {}).get("r2", float("nan"))),
            orth=float(dict(orthogonal.get("basis_recovery", {})).get("orthogonality_score", float("nan"))),
            ratio_hit=bool(dict(orthogonal.get("basis_recovery", {})).get("contains_ratio_basis")),
            piecewise_hit=bool(dict(orthogonal.get("basis_recovery", {})).get("contains_piecewise_basis")),
            basis_hit=float(dict(orthogonal.get("basis_recovery", {})).get("exact_basis_hit_score", float("nan"))),
            term_hit=float(dict(orthogonal.get("basis_recovery", {})).get("exact_term_recovery_score", float("nan"))),
            phase_hit=float(dict(orthogonal.get("basis_recovery", {})).get("phase_equivalent_term_recovery_score", float("nan"))),
            family_hit=float(dict(orthogonal.get("basis_recovery", {})).get("family_level_term_recovery_score", float("nan"))),
        )
    )


if __name__ == "__main__":
    main()
