from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence

import numpy as np

from core.symbolic.symbolic_dsl import expression_to_string
from core.symbolic.truth_contracts import (
    TRUTH_CONTRACT_RECOVERY_MIN_NORMALIZED_WEIGHT,
    build_truth_contract_recovery as build_generic_truth_contract_recovery,
    term_row_view as build_truth_term_row_view,
    truth_basis_rows_from_basis_structure as build_truth_basis_rows_from_basis_structure,
)

SYMBOLIC_ARTIFACT_SCHEMA_NAME = "symbolic_artifact"
SYMBOLIC_ARTIFACT_SCHEMA_KEY = "symbolic_artifact_v1"
SYMBOLIC_ARTIFACT_SCHEMA_VERSION = 1
SYMBOLIC_FOLD_STABILITY_SCHEMA_VERSION = 1

SYMBOLIC_ARTIFACT_FIELDS: tuple[str, ...] = (
    "final_expression",
    "normalized_expression",
    "feature_usage",
    "term_contributions",
    "complexity_metrics",
    "stability_metrics",
    "candidate_lineage",
    "simplification_trace",
    "truth_contract_recovery",
    "orthogonal_search_objective",
    "heterogeneous_lane_consensus",
    "equivalence_expression_handling",
    "interference_feature_handling",
    "periodic_equivalence_disambiguation",
    "regional_correction_basis",
    "head_semantics",
    "regime_structure",
    "basis_structure",
    "assembler_structure",
    "piecewise_gate_basis",
)

SYMBOLIC_COMPLEXITY_FIELDS: tuple[str, ...] = (
    "term_count",
    "expression_size",
    "max_depth",
    "mean_depth",
    "operator_cost",
    "interaction_order",
    "parameter_count",
    "feature_count",
    "unary_op_count",
    "binary_op_count",
)

SYMBOLIC_EXPLAINABILITY_FIELDS: tuple[str, ...] = (
    "final_expression",
    "normalized_expression",
    "feature_usage",
    "term_contributions",
    "candidate_lineage",
    "simplification_trace",
    "truth_contract_recovery",
    "orthogonal_search_objective",
    "heterogeneous_lane_consensus",
    "equivalence_expression_handling",
    "interference_feature_handling",
    "periodic_equivalence_disambiguation",
    "regional_correction_basis",
    "head_semantics",
    "regime_structure",
    "basis_structure",
    "piecewise_gate_basis",
)

SYMBOLIC_REGIME_FIELDS: tuple[str, ...] = (
    "source",
    "mode",
    "piecewise_enabled",
    "structure_mode",
    "search_driver",
    "gate_feature_names",
    "gate_indices",
    "gate_threshold",
    "gate_min_leaf",
    "gate_max_local_models",
    "selected_regime_keys",
    "failed_regimes",
    "local_regimes",
    "counts_all",
    "counts_selected",
    "counts_skipped",
    "local_regime_count",
    "blend_kappa",
)

SYMBOLIC_BASIS_FIELDS: tuple[str, ...] = (
    "source",
    "basis_scope",
    "basis_count",
    "basis_feature_union",
    "global_basis",
    "local_basis_by_regime",
    "gate_basis",
    "orthogonality_status",
    "basis_semantics",
    "residual_complementarity",
    "semantic_deduplication",
    "basis_discovery_stage",
    "basis_context",
)

SYMBOLIC_ASSEMBLER_FIELDS: tuple[str, ...] = (
    "source",
    "assembler_mode",
    "assembly_scope",
    "uses_piecewise_gate",
    "budget_recorded",
    "budget",
    "output_expression_count",
    "composition_targets",
    "structure_head",
    "prediction_head",
    "search_input_space",
    "pool_expansion_unit",
    "gradient_guidance_mode",
    "basis_binding_mode",
    "escape_policy",
    "basis_conditioned",
    "stage_protocol",
    "basis_context",
    "object_gradient_pool",
)

SYMBOLIC_PIECEWISE_GATE_FIELDS: tuple[str, ...] = (
    "available",
    "enabled",
    "status",
    "source",
    "gate_feature_names",
    "gate_indices",
    "gate_threshold",
    "gate_min_leaf",
    "gate_max_local_models",
    "blend_kappa",
    "selected_regime_keys",
    "failed_regimes",
    "counts_all",
    "counts_selected",
    "counts_skipped",
    "local_basis_counts",
    "local_basis_keys",
    "gate_basis_count",
    "gate_term_names",
    "gate_basis_terms",
    "gate_basis",
)

SYMBOLIC_STABILITY_FIELDS: tuple[str, ...] = (
    "residual_std_mean",
    "residual_std_max",
    "fold_stability_available",
    "fold_stability",
    "fold_summary",
    "fold_schema_version",
    "fold_metric_schema",
    "fold_objective_schema",
    "fold_count",
    "selection_meets_coverage_threshold",
    "selection_coverage_error_threshold",
    "coverage_error_mean",
    "pinaw_mean",
    "interval_score_mean",
    "picp_mean",
    "mean_width_mean",
    "rmse_mean",
    "rmse_std",
    "rmse_drift",
    "family_concentration",
    "feature_concentration",
    "gradient_signal_signatures",
    "gradient_stability_min",
    "gradient_stability_max",
    "gradient_stability_last",
)

SYMBOLIC_TRUTH_CONTRACT_RECOVERY_FIELDS: tuple[str, ...] = (
    "status",
    "source",
    "truth_formula_expression",
    "truth_basis_count",
    "matched_truth_basis_count",
    "matched_truth_term_count",
    "exact_basis_hit_score",
    "outer_chart_hit_score",
    "exact_term_recovery_score",
    "inner_realization_hit_score",
    "inner_realization_only_score",
    "exact_term_min_normalized_weight",
    "truth_basis_matches",
    "phase_equivalent_contract_count",
    "phase_equivalent_basis_hit_score",
    "phase_equivalent_term_recovery_score",
    "phase_equivalent_matches",
    "family_level_contract_count",
    "family_level_basis_hit_score",
    "family_level_term_recovery_score",
    "family_level_matches",
)

SYMBOLIC_ORTHOGONAL_OBJECTIVE_FIELDS: tuple[str, ...] = (
    "status",
    "source",
    "protocol",
    "inner_fit_score",
    "orthogonality_score",
    "residual_complementarity_score",
    "semantic_dedup_score",
    "outer_score",
    "weights",
    "inner_metrics",
)

SYMBOLIC_HETEROGENEOUS_LANE_FIELDS: tuple[str, ...] = (
    "status",
    "source",
    "protocol",
    "lane_id",
    "lane_family",
    "lane_label",
    "lane_description",
    "lane_weight",
    "screening_protocol",
    "challenger_objective_protocol",
    "pool_expansion_bias_protocol",
    "joint_core_score",
    "joint_core_score_mean",
    "cross_lane_stability",
    "cross_lane_support_rate",
    "cross_lane_family_support_rate",
    "consensus_prior_row_count",
    "lane_spec",
)

_TRUTH_CONTRACT_RECOVERY_MIN_NORMALIZED_WEIGHT = TRUTH_CONTRACT_RECOVERY_MIN_NORMALIZED_WEIGHT

_UNARY_OPERATOR_COSTS: dict[str, float] = {
    "identity": 1.0,
    "square": 2.0,
    "sin": 2.5,
    "cos": 2.5,
    "tanh": 2.5,
    "exp": 3.0,
    "log": 3.0,
    "abs": 1.5,
    "sqrt": 3.0,
}

_BINARY_OPERATOR_COSTS: dict[str, float] = {
    "add": 1.0,
    "sub": 1.0,
    "mul": 1.5,
    "div": 2.5,
}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(v) for v in value]
    return str(value)


@dataclass(frozen=True)
class SymbolicArtifactSchema:
    payload: Mapping[str, Any] = field(default_factory=dict)
    schema_name: str = SYMBOLIC_ARTIFACT_SCHEMA_NAME
    schema_key: str = SYMBOLIC_ARTIFACT_SCHEMA_KEY
    schema_version: int = SYMBOLIC_ARTIFACT_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        out = {
            "schema_name": str(self.schema_name),
            "schema_key": str(self.schema_key),
            "schema_version": int(self.schema_version),
        }
        for key, value in dict(self.payload).items():
            if key in {"schema_name", "schema_key", "schema_version"}:
                continue
            out[str(key)] = _jsonable(value)
        return out


def _as_float_list(values: Sequence[float] | np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return [float(v) for v in arr.tolist()]


def _feature_label(index: int, feature_names: Sequence[str]) -> str:
    idx = int(index)
    if 0 <= idx < len(tuple(feature_names)):
        return str(tuple(feature_names)[idx])
    return f"x{idx}"


def _replace_feature_tokens(expr: str, feature_names: Sequence[str]) -> str:
    names = tuple(str(v) for v in tuple(feature_names))

    def repl(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if 0 <= idx < len(names):
            return names[idx]
        return match.group(0)

    return re.sub(r"\bx(\d+)\b", repl, str(expr))


def _normalized_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalized_feature_tuple(values: Any) -> tuple[str, ...]:
    normalized = [_normalized_text(value) for value in tuple(values or ())]
    return tuple(sorted(value for value in normalized if value))


def _first_list_mapping_value(mapping: Any) -> list[dict[str, Any]]:
    if isinstance(mapping, Sequence) and not isinstance(mapping, (str, bytes, bytearray)):
        if all(isinstance(row, Mapping) for row in mapping):
            return [dict(row) for row in mapping]
    for value in dict(mapping or {}).values():
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if all(isinstance(row, Mapping) for row in value):
                return [dict(row) for row in value]
        if isinstance(value, Mapping):
            nested = _first_list_mapping_value(value)
            if nested:
                return nested
    return []


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
    return ("abs(" in normalized) and ("0.5" in normalized)


def _truth_contract_specs(contracts: Sequence[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for contract in tuple(contracts):
        text = str(contract or "").strip()
        normalized = _normalized_text(text)
        if not normalized:
            continue
        if normalized.startswith("safe_ratio(") and normalized.endswith(")"):
            args = [part.strip() for part in text[text.find("(") + 1 : -1].split(",") if part.strip()]
            numerator = str(args[0]) if len(args) >= 1 else ""
            denominator = str(args[1]) if len(args) >= 2 else ""
            specs.append(
                {
                    "contract": text,
                    "family": "safe_ratio",
                    "features": _normalized_feature_tuple((numerator, denominator)),
                    "arity": 2,
                }
            )
            continue
        if normalized.startswith("piecewise_hinge(") and normalized.endswith(")"):
            args = [part.strip() for part in text[text.find("(") + 1 : -1].split(",") if part.strip()]
            specs.append(
                {
                    "contract": text,
                    "family": "piecewise_hinge",
                    "features": _normalized_feature_tuple(args[:1]),
                    "arity": 1,
                }
            )
            continue
        unary_match = re.fullmatch(r"([a-zA-Z_][a-zA-Z0-9_]*)\(([^()]*)\)", text)
        if unary_match:
            family = _normalized_text(unary_match.group(1))
            args = [part.strip() for part in unary_match.group(2).split(",") if part.strip()]
            specs.append(
                {
                    "contract": text,
                    "family": family,
                    "features": _normalized_feature_tuple(args),
                    "arity": len(args),
                }
            )
            continue
        specs.append(
            {
                "contract": text,
                "family": "linear_feature",
                "features": _normalized_feature_tuple((text,)),
                "arity": 1,
            }
        )
    return specs


def _term_row_view(row: Mapping[str, Any]) -> dict[str, Any]:
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


def _matches_truth_contract(spec: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    family = str(spec.get("family") or "")
    features = tuple(spec.get("features", ()))
    if tuple(row.get("features", ())) != features:
        return False
    expr = str(row.get("expr") or "")
    name = str(row.get("name") or "")
    semantic_family = str(row.get("semantic_family") or "")
    semantic_signature = str(row.get("semantic_signature") or "")
    if family == "safe_ratio":
        numerator = str(features[1] if len(features) > 1 else "")
        denominator = str(features[0] if len(features) > 0 else "")
        return (
            ("/safe(" in expr)
            or ("/safe(" in name)
            or _expr_looks_like_safe_ratio(expr, numerator=numerator, denominator=denominator)
            or _expr_looks_like_safe_ratio(name, numerator=numerator, denominator=denominator)
            or ("ratio" in semantic_family)
            or ("binary:div" in semantic_signature)
        )
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
        feature_name = str(features[0] if len(features) > 0 else "")
        node_count = row.get("node_count")
        return bool(
            expr == feature_name
            or name == feature_name
            or semantic_family == "linear_feature"
            or semantic_signature.startswith("feature:")
            or node_count == 1
        )
    if len(features) == 1:
        return (
            (f"{family}(" in expr)
            or (f"{family}(" in name)
            or (f"unary:{family}" in semantic_signature)
        )
    return family in semantic_family or family in semantic_signature


def _row_is_materially_active(row: Mapping[str, Any], *, min_normalized_weight: float) -> bool:
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


def _truth_basis_rows_from_basis_structure(basis_structure: Mapping[str, Any]) -> list[dict[str, Any]]:
    recorded = _mapping_or_empty(_mapping_or_empty(_mapping_or_empty(basis_structure).get("basis_semantics")).get("recorded"))
    basis_terms = recorded.get("basis_terms")
    if isinstance(basis_terms, Sequence) and not isinstance(basis_terms, (str, bytes, bytearray)):
        return [dict(row) for row in basis_terms if isinstance(row, Mapping)]
    global_basis = _mapping_or_empty(basis_structure).get("global_basis")
    return _first_list_mapping_value(global_basis)


def _build_truth_contract_recovery(
    *,
    metadata: Mapping[str, Any],
    basis_structure: Mapping[str, Any],
    term_contributions: Mapping[str, Any],
) -> dict[str, Any]:
    data_metadata = _mapping_or_empty(metadata.get("data_metadata"))
    truth_formula = _mapping_or_empty(data_metadata.get("truth_formula"))
    basis_rows = [
        build_truth_term_row_view(row)
        for row in build_truth_basis_rows_from_basis_structure(basis_structure)
    ]
    active_rows = [build_truth_term_row_view(row) for row in _first_list_mapping_value(term_contributions)]
    return build_generic_truth_contract_recovery(
        truth_formula=truth_formula,
        basis_rows=basis_rows,
        active_term_rows=active_rows,
        min_normalized_weight=_TRUTH_CONTRACT_RECOVERY_MIN_NORMALIZED_WEIGHT,
        source="data_metadata.truth_formula",
    )


def _build_orthogonal_search_objective(metadata: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping_or_empty(metadata.get("orthogonal_search_objective"))
    if not payload:
        payload = _mapping_or_empty(_mapping_or_empty(metadata.get("symbolic")).get("orthogonal_search_objective"))
    if not payload:
        return {
            "status": "not_recorded",
            "source": "not_recorded",
        }
    return {
        "status": "reported",
        "source": "metadata.orthogonal_search_objective",
        **_jsonable(payload),
    }


def _build_heterogeneous_lane_consensus(metadata: Mapping[str, Any]) -> dict[str, Any]:
    meta = dict(metadata or {})
    symbolic = _mapping_or_empty(meta.get("symbolic"))
    fit_context = _mapping_or_empty(meta.get("fit_context"))
    lane_context = _mapping_or_empty(meta.get("heterogeneous_multi_lane_context"))
    if not lane_context:
        lane_context = _mapping_or_empty(symbolic.get("heterogeneous_multi_lane_context"))
    if not lane_context:
        lane_context = _mapping_or_empty(fit_context.get("heterogeneous_multi_lane_context"))
    if not lane_context:
        lane_context = {
            str(key): value
            for key, value in {
                "protocol": fit_context.get("heterogeneous_multi_lane_protocol"),
                "lane_id": _first_present(meta.get("lane_id"), fit_context.get("lane_id")),
                "lane_family": _first_present(meta.get("lane_family"), fit_context.get("lane_family")),
                "lane_label": fit_context.get("lane_label"),
                "lane_description": fit_context.get("lane_description"),
                "lane_weight": fit_context.get("lane_weight"),
                "screening_protocol": fit_context.get("screening_protocol"),
                "challenger_objective_protocol": _first_present(
                    meta.get("challenger_objective_protocol"),
                    fit_context.get("challenger_objective_protocol"),
                ),
                "pool_expansion_bias_protocol": _first_present(
                    meta.get("pool_expansion_bias_protocol"),
                    fit_context.get("pool_expansion_bias_protocol"),
                ),
                "lane_spec": fit_context.get("lane_spec"),
            }.items()
            if value is not None and (not isinstance(value, str) or str(value).strip())
        }

    consensus_prior_rows = [
        dict(row)
        for row in tuple(
            meta.get("consensus_prior_rows")
            or symbolic.get("consensus_prior_rows")
            or ()
        )
        if isinstance(row, Mapping)
    ]
    joint_scores = [
        _coerce_float_or_none(row.get("joint_core_score"))
        for row in consensus_prior_rows
        if _coerce_float_or_none(row.get("joint_core_score")) is not None
    ]
    cross_lane_scores = [
        _coerce_float_or_none(row.get("cross_lane_stability"))
        for row in consensus_prior_rows
        if _coerce_float_or_none(row.get("cross_lane_stability")) is not None
    ]
    cross_lane_support_rates = [
        _coerce_float_or_none(row.get("cross_lane_support_rate"))
        for row in consensus_prior_rows
        if _coerce_float_or_none(row.get("cross_lane_support_rate")) is not None
    ]
    cross_lane_family_support_rates = [
        _coerce_float_or_none(row.get("cross_lane_family_support_rate"))
        for row in consensus_prior_rows
        if _coerce_float_or_none(row.get("cross_lane_family_support_rate")) is not None
    ]
    if not lane_context and not consensus_prior_rows:
        return {
            "status": "not_recorded",
            "source": "not_recorded",
        }
    return {
        "status": "reported",
        "source": (
            "metadata.heterogeneous_multi_lane_context+consensus_prior_rows"
            if lane_context and consensus_prior_rows
            else "metadata.heterogeneous_multi_lane_context"
            if lane_context
            else "metadata.consensus_prior_rows"
        ),
        "protocol": _first_present(
            lane_context.get("protocol"),
            fit_context.get("heterogeneous_multi_lane_protocol"),
        ),
        "lane_id": _first_present(meta.get("lane_id"), lane_context.get("lane_id")),
        "lane_family": _first_present(meta.get("lane_family"), lane_context.get("lane_family")),
        "lane_label": lane_context.get("lane_label"),
        "lane_description": lane_context.get("lane_description"),
        "lane_weight": _coerce_float_or_none(lane_context.get("lane_weight")),
        "screening_protocol": lane_context.get("screening_protocol"),
        "challenger_objective_protocol": _first_present(
            meta.get("challenger_objective_protocol"),
            lane_context.get("challenger_objective_protocol"),
        ),
        "pool_expansion_bias_protocol": _first_present(
            meta.get("pool_expansion_bias_protocol"),
            lane_context.get("pool_expansion_bias_protocol"),
        ),
        "joint_core_score": None if not joint_scores else float(max(joint_scores)),
        "joint_core_score_mean": None
        if not joint_scores
        else float(sum(joint_scores) / float(len(joint_scores))),
        "cross_lane_stability": None
        if not cross_lane_scores
        else float(max(cross_lane_scores)),
        "cross_lane_support_rate": None
        if not cross_lane_support_rates
        else float(max(cross_lane_support_rates)),
        "cross_lane_family_support_rate": None
        if not cross_lane_family_support_rates
        else float(max(cross_lane_family_support_rates)),
        "consensus_prior_row_count": int(len(consensus_prior_rows)),
        "lane_spec": _jsonable(lane_context.get("lane_spec")),
    }


def _build_equivalence_expression_handling(metadata: Mapping[str, Any]) -> dict[str, Any]:
    meta = dict(metadata or {})
    symbolic = _mapping_or_empty(meta.get("symbolic"))
    fit_context = _mapping_or_empty(meta.get("fit_context"))
    payload = _mapping_or_empty(
        meta.get("equivalence_expression_handling")
        or symbolic.get("equivalence_expression_handling")
        or fit_context.get("equivalence_expression_handling")
    )
    protocol = _first_present(
        meta.get("equivalence_expression_protocol"),
        symbolic.get("equivalence_expression_protocol"),
        payload.get("protocol"),
    )
    mode = _first_present(
        meta.get("equivalence_expression_mode"),
        symbolic.get("equivalence_expression_mode"),
        payload.get("mode"),
    )
    scope = _first_present(
        meta.get("equivalence_class_scope"),
        symbolic.get("equivalence_class_scope"),
        payload.get("class_scope"),
    )
    if not payload and protocol is None and mode is None and scope is None:
        return {
            "status": "not_recorded",
            "source": "not_recorded",
        }
    enabled_steps = _coerce_str_list(payload.get("enabled_steps"))
    return {
        "status": "reported",
        "source": (
            "metadata.equivalence_expression_handling"
            if payload
            else "metadata.equivalence_expression_protocol"
        ),
        "protocol": protocol,
        "mode": mode,
        "class_scope": scope,
        "equivalence_mode": _first_present(
            payload.get("equivalence_mode"),
            meta.get("core_equivalence_mode"),
        ),
        "implemented_submodes": _coerce_str_list(payload.get("implemented_submodes")),
        "child_modes": _jsonable(payload.get("child_modes")) if payload.get("child_modes") is not None else None,
        "enabled_steps": [str(value) for value in enabled_steps],
        "current_narrowness": _coerce_str_list(payload.get("current_narrowness")),
        "notes": payload.get("notes"),
        "payload": _jsonable(payload) if payload else None,
    }


def _build_interference_feature_handling(metadata: Mapping[str, Any]) -> dict[str, Any]:
    meta = dict(metadata or {})
    symbolic = _mapping_or_empty(meta.get("symbolic"))
    fit_context = _mapping_or_empty(meta.get("fit_context"))
    payload = _mapping_or_empty(
        meta.get("interference_feature_handling")
        or symbolic.get("interference_feature_handling")
        or fit_context.get("interference_feature_handling")
    )
    protocol = _first_present(
        meta.get("interference_feature_protocol"),
        symbolic.get("interference_feature_protocol"),
        payload.get("protocol"),
    )
    mode = _first_present(
        meta.get("interference_feature_mode"),
        symbolic.get("interference_feature_mode"),
        payload.get("mode"),
    )
    if not payload and protocol is None and mode is None:
        return {
            "status": "not_recorded",
            "source": "not_recorded",
        }
    enabled_steps = _coerce_str_list(payload.get("enabled_steps"))
    return {
        "status": "reported",
        "source": (
            "metadata.interference_feature_handling"
            if payload
            else "metadata.interference_feature_protocol"
        ),
        "protocol": protocol,
        "mode": mode,
        "cross_explanatory_rejection_mode": _first_present(
            meta.get("cross_explanatory_rejection_mode"),
            symbolic.get("cross_explanatory_rejection_mode"),
            payload.get("cross_explanatory_rejection_mode"),
        ),
        "trivial_nonlinearity_penalty_mode": _first_present(
            meta.get("trivial_nonlinearity_penalty_mode"),
            symbolic.get("trivial_nonlinearity_penalty_mode"),
            payload.get("trivial_nonlinearity_penalty_mode"),
        ),
        "environment_invariance_audit_mode": _first_present(
            meta.get("environment_invariance_audit_mode"),
            symbolic.get("environment_invariance_audit_mode"),
            payload.get("environment_invariance_audit_mode"),
        ),
        "proxy_group_policy": _first_present(
            meta.get("proxy_group_policy"),
            symbolic.get("proxy_group_policy"),
            payload.get("proxy_group_policy"),
        ),
        "source_overlap_penalty_mode": _first_present(
            meta.get("source_overlap_penalty_mode"),
            symbolic.get("source_overlap_penalty_mode"),
            payload.get("source_overlap_penalty_mode"),
        ),
        "implemented_submodes": _coerce_str_list(payload.get("implemented_submodes")),
        "child_modes": _jsonable(payload.get("child_modes")) if payload.get("child_modes") is not None else None,
        "enabled_steps": [str(value) for value in enabled_steps],
        "current_narrowness": _coerce_str_list(payload.get("current_narrowness")),
        "notes": payload.get("notes"),
        "payload": _jsonable(payload) if payload else None,
    }


def _build_periodic_equivalence_disambiguation(metadata: Mapping[str, Any]) -> dict[str, Any]:
    meta = dict(metadata or {})
    symbolic = _mapping_or_empty(meta.get("symbolic"))
    fit_context = _mapping_or_empty(meta.get("fit_context"))
    payload = _mapping_or_empty(
        meta.get("periodic_equivalence_disambiguation")
        or symbolic.get("periodic_equivalence_disambiguation")
        or fit_context.get("periodic_equivalence_disambiguation")
    )
    protocol = _first_present(
        meta.get("periodic_equivalence_protocol"),
        symbolic.get("periodic_equivalence_protocol"),
        payload.get("protocol"),
    )
    mode = _first_present(
        meta.get("periodic_equivalence_disambiguation_mode"),
        symbolic.get("periodic_equivalence_disambiguation_mode"),
        payload.get("mode"),
    )
    if not payload and protocol is None and mode is None:
        return {
            "status": "not_recorded",
            "source": "not_recorded",
        }
    return {
        "status": "reported",
        "source": (
            "metadata.periodic_equivalence_disambiguation"
            if payload
            else "metadata.periodic_equivalence_protocol"
        ),
        "protocol": protocol,
        "mode": mode,
        "parent_protocol": _first_present(payload.get("parent_protocol")),
        "parent_mode_slot": _first_present(payload.get("parent_mode_slot")),
        "canonical_mode_name": _first_present(payload.get("canonical_mode_name")),
        "phase_spectrum_audit_mode": _first_present(
            meta.get("phase_spectrum_audit_mode"),
            symbolic.get("phase_spectrum_audit_mode"),
            payload.get("phase_spectrum_audit_mode"),
        ),
        "periodic_family_prior_mode": _first_present(
            meta.get("periodic_family_prior_mode"),
            symbolic.get("periodic_family_prior_mode"),
            payload.get("periodic_family_prior_mode"),
        ),
        "periodic_candidate_screen_reserve": _first_present(
            meta.get("periodic_candidate_screen_reserve"),
            symbolic.get("periodic_candidate_screen_reserve"),
            payload.get("periodic_candidate_screen_reserve"),
        ),
        "periodic_feature_names": _coerce_str_list(payload.get("periodic_feature_names")),
        "enabled_steps": _coerce_str_list(payload.get("enabled_steps")),
        "payload": _jsonable(payload) if payload else None,
    }


def _build_regional_correction_basis(metadata: Mapping[str, Any]) -> dict[str, Any]:
    meta = dict(metadata or {})
    symbolic = _mapping_or_empty(meta.get("symbolic"))
    fit_context = _mapping_or_empty(meta.get("fit_context"))
    payload = _mapping_or_empty(
        meta.get("regional_correction_basis")
        or symbolic.get("regional_correction_basis")
        or fit_context.get("regional_correction_basis")
    )
    protocol = _first_present(
        meta.get("regional_correction_protocol"),
        symbolic.get("regional_correction_protocol"),
        payload.get("protocol"),
    )
    if not payload and protocol is None:
        return {
            "status": "not_recorded",
            "source": "not_recorded",
        }
    return {
        "status": "reported",
        "source": (
            "metadata.regional_correction_basis"
            if payload
            else "metadata.regional_correction_protocol"
        ),
        "protocol": protocol,
        "parent_protocol": _first_present(payload.get("parent_protocol")),
        "parent_mode_slot": _first_present(payload.get("parent_mode_slot")),
        "canonical_mode_name": _first_present(payload.get("canonical_mode_name")),
        "semantic_slot_name": _first_present(payload.get("semantic_slot_name")),
        "residual_regime_identification_mode": _first_present(
            meta.get("residual_regime_identification_mode"),
            symbolic.get("residual_regime_identification_mode"),
            payload.get("residual_regime_identification_mode"),
        ),
        "regional_correction_basis_mode": _first_present(
            meta.get("regional_correction_basis_mode"),
            symbolic.get("regional_correction_basis_mode"),
            payload.get("regional_correction_basis_mode"),
        ),
        "regional_correction_promotion_mode": _first_present(
            meta.get("regional_correction_promotion_mode"),
            symbolic.get("regional_correction_promotion_mode"),
            payload.get("regional_correction_promotion_mode"),
        ),
        "regional_correction_feature_scope": _first_present(
            meta.get("regional_correction_feature_scope"),
            symbolic.get("regional_correction_feature_scope"),
            payload.get("regional_correction_feature_scope"),
        ),
        "regional_correction_topk": _first_present(
            meta.get("regional_correction_topk"),
            symbolic.get("regional_correction_topk"),
            payload.get("regional_correction_topk"),
        ),
        "regional_correction_min_r2_gain": _first_present(
            meta.get("regional_correction_min_r2_gain"),
            symbolic.get("regional_correction_min_r2_gain"),
            payload.get("regional_correction_min_r2_gain"),
        ),
        "enabled_steps": _coerce_str_list(payload.get("enabled_steps")),
        "payload": _jsonable(payload) if payload else None,
    }


def _target_labels(target_names: Sequence[str], target_dim: int) -> tuple[str, ...]:
    names = tuple(str(v) for v in tuple(target_names) if str(v))
    if not names:
        return tuple(f"y{i}" for i in range(int(target_dim)))
    return tuple(names[i] if i < len(names) else f"y{i}" for i in range(int(target_dim)))


def _expr_depth(expr: Mapping[str, Any]) -> int:
    kind = str(expr.get("type", "")).strip().lower()
    if kind in {"feature", "const", "param"}:
        return 1
    if kind == "unary":
        return 1 + _expr_depth(dict(expr.get("arg", {})))
    if kind == "binary":
        return 1 + max(
            _expr_depth(dict(expr.get("left", {}))),
            _expr_depth(dict(expr.get("right", {}))),
        )
    return 1


def _expr_node_count(expr: Mapping[str, Any]) -> int:
    kind = str(expr.get("type", "")).strip().lower()
    if kind in {"feature", "const", "param"}:
        return 1
    if kind == "unary":
        return 1 + _expr_node_count(dict(expr.get("arg", {})))
    if kind == "binary":
        return 1 + _expr_node_count(dict(expr.get("left", {}))) + _expr_node_count(dict(expr.get("right", {})))
    return 1


def _expr_collect_feature_indices(expr: Mapping[str, Any]) -> set[int]:
    kind = str(expr.get("type", "")).strip().lower()
    if kind == "feature":
        return {int(expr.get("index", -1))}
    if kind == "unary":
        return _expr_collect_feature_indices(dict(expr.get("arg", {})))
    if kind == "binary":
        return _expr_collect_feature_indices(dict(expr.get("left", {}))) | _expr_collect_feature_indices(
            dict(expr.get("right", {}))
        )
    return set()


def _expr_collect_parameter_names(expr: Mapping[str, Any]) -> set[str]:
    kind = str(expr.get("type", "")).strip().lower()
    if kind == "param":
        name = str(expr.get("name", "")).strip()
        return {name} if name else set()
    if kind == "unary":
        return _expr_collect_parameter_names(dict(expr.get("arg", {})))
    if kind == "binary":
        return _expr_collect_parameter_names(dict(expr.get("left", {}))) | _expr_collect_parameter_names(
            dict(expr.get("right", {}))
        )
    return set()


def _expr_operator_counter(expr: Mapping[str, Any]) -> dict[str, int]:
    kind = str(expr.get("type", "")).strip().lower()
    if kind == "unary":
        out = _expr_operator_counter(dict(expr.get("arg", {})))
        key = f"unary:{str(expr.get('op', '')).strip().lower()}"
        out[key] = int(out.get(key, 0) + 1)
        return out
    if kind == "binary":
        out = _expr_operator_counter(dict(expr.get("left", {})))
        for key, value in _expr_operator_counter(dict(expr.get("right", {}))).items():
            out[key] = int(out.get(key, 0) + int(value))
        op_key = f"binary:{str(expr.get('op', '')).strip().lower()}"
        out[op_key] = int(out.get(op_key, 0) + 1)
        return out
    return {}


def _expr_operator_cost(expr: Mapping[str, Any]) -> float:
    kind = str(expr.get("type", "")).strip().lower()
    if kind == "unary":
        op = str(expr.get("op", "")).strip().lower()
        return float(_UNARY_OPERATOR_COSTS.get(op, 1.0)) + _expr_operator_cost(dict(expr.get("arg", {})))
    if kind == "binary":
        op = str(expr.get("op", "")).strip().lower()
        return (
            float(_BINARY_OPERATOR_COSTS.get(op, 1.0))
            + _expr_operator_cost(dict(expr.get("left", {})))
            + _expr_operator_cost(dict(expr.get("right", {})))
        )
    return 0.0


def _term_descriptor(
    *,
    term_index: int,
    term: Mapping[str, Any],
    parameter_values: Mapping[str, float],
    feature_names: Sequence[str],
    precision: int = 12,
) -> dict[str, Any]:
    expr = dict(term.get("expr", {}))
    feature_indices = sorted(idx for idx in _expr_collect_feature_indices(expr) if idx >= 0)
    parameter_names = sorted(_expr_collect_parameter_names(expr))
    operator_counter = _expr_operator_counter(expr)
    unary_op_count = int(sum(v for k, v in operator_counter.items() if str(k).startswith("unary:")))
    binary_op_count = int(sum(v for k, v in operator_counter.items() if str(k).startswith("binary:")))
    expr_raw = expression_to_string(expr, param_values=dict(parameter_values), precision=int(precision))
    return {
        "term_index": int(term_index),
        "term_name": str(term.get("name", f"term_{term_index}")),
        "expression_raw": expr_raw,
        "expression_named": _replace_feature_tokens(expr_raw, feature_names),
        "feature_indices": [int(v) for v in feature_indices],
        "feature_names": [_feature_label(idx, feature_names) for idx in feature_indices],
        "parameter_names": [str(v) for v in parameter_names],
        "node_count": int(_expr_node_count(expr)),
        "depth": int(_expr_depth(expr)),
        "operator_cost": float(_expr_operator_cost(expr)),
        "interaction_order": int(len(feature_indices)),
        "feature_count": int(len(feature_indices)),
        "parameter_count": int(len(parameter_names)),
        "unary_op_count": int(unary_op_count),
        "binary_op_count": int(binary_op_count),
        "operator_counts": {str(k): int(v) for k, v in sorted(operator_counter.items(), key=lambda item: item[0])},
    }


def _build_target_term_contributions(
    *,
    genome: Sequence[Mapping[str, Any]],
    parameter_values: Mapping[str, float],
    readout_weight: np.ndarray,
    feature_names: Sequence[str],
    target_names: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    weights = np.asarray(readout_weight, dtype=float)
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)

    target_labels = _target_labels(target_names, int(weights.shape[1]))
    out: dict[str, list[dict[str, Any]]] = {}
    for target_index, target_name in enumerate(target_labels):
        coeffs = np.asarray(weights[:, target_index], dtype=float).reshape(-1)
        total_abs = float(np.sum(np.abs(coeffs)))
        rows: list[dict[str, Any]] = []
        for term_index, term in enumerate(tuple(genome)):
            coeff = float(coeffs[term_index]) if term_index < coeffs.shape[0] else 0.0
            if abs(coeff) < 1e-12:
                continue
            base = _term_descriptor(
                term_index=int(term_index),
                term=dict(term),
                parameter_values=parameter_values,
                feature_names=feature_names,
            )
            rows.append(
                {
                    **base,
                    "coefficient": float(coeff),
                    "abs_coefficient": float(abs(coeff)),
                    "normalized_weight": float(0.0 if total_abs <= 1e-12 else abs(coeff) / total_abs),
                }
            )
        rows.sort(key=lambda item: (-float(item.get("abs_coefficient", 0.0)), int(item.get("term_index", 0))))
        out[str(target_name)] = rows
    return out


def _flatten_contribution_rows(target_contributions: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for target_name, payload in dict(target_contributions).items():
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            for row in payload:
                if isinstance(row, Mapping):
                    rows.append((str(target_name), dict(row)))
            continue
        if isinstance(payload, Mapping):
            for sub_name, sub_rows in dict(payload).items():
                if isinstance(sub_rows, Sequence) and not isinstance(sub_rows, (str, bytes, bytearray)):
                    for row in sub_rows:
                        if isinstance(row, Mapping):
                            rows.append((f"{target_name}:{sub_name}", dict(row)))
    return rows


def _build_feature_usage(target_contributions: Mapping[str, Any]) -> dict[str, Any]:
    flat_rows = _flatten_contribution_rows(target_contributions)
    feature_term_counts: dict[str, int] = {}
    feature_weight_magnitude: dict[str, float] = {}
    target_feature_usage: dict[str, dict[str, Any]] = {}

    for target_key, row in flat_rows:
        per_target_counts: dict[str, int] = {}
        per_target_weights: dict[str, float] = {}
        for feature_name in tuple(row.get("feature_names", ())):
            key = str(feature_name)
            feature_term_counts[key] = int(feature_term_counts.get(key, 0) + 1)
            feature_weight_magnitude[key] = float(feature_weight_magnitude.get(key, 0.0) + abs(float(row.get("coefficient", 0.0))))
            per_target_counts[key] = int(per_target_counts.get(key, 0) + 1)
            per_target_weights[key] = float(per_target_weights.get(key, 0.0) + abs(float(row.get("coefficient", 0.0))))
        bucket = target_feature_usage.setdefault(
            str(target_key),
            {
                "feature_term_counts": {},
                "feature_weight_magnitude": {},
            },
        )
        for key, value in per_target_counts.items():
            bucket["feature_term_counts"][key] = int(bucket["feature_term_counts"].get(key, 0) + int(value))
        for key, value in per_target_weights.items():
            bucket["feature_weight_magnitude"][key] = float(
                bucket["feature_weight_magnitude"].get(key, 0.0) + float(value)
            )

    used_features = tuple(sorted(feature_term_counts.keys()))
    return {
        "used_features": [str(v) for v in used_features],
        "used_feature_count": int(len(used_features)),
        "feature_term_counts": {str(k): int(v) for k, v in sorted(feature_term_counts.items(), key=lambda item: item[0])},
        "feature_weight_magnitude": {
            str(k): float(v) for k, v in sorted(feature_weight_magnitude.items(), key=lambda item: item[0])
        },
        "target_feature_usage": {
            str(target): {
                "feature_term_counts": {
                    str(k): int(v) for k, v in sorted(dict(payload.get("feature_term_counts", {})).items(), key=lambda item: item[0])
                },
                "feature_weight_magnitude": {
                    str(k): float(v)
                    for k, v in sorted(dict(payload.get("feature_weight_magnitude", {})).items(), key=lambda item: item[0])
                },
            }
            for target, payload in sorted(target_feature_usage.items(), key=lambda item: item[0])
        },
    }


def _build_complexity_metrics(genome: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    terms = tuple(dict(term) for term in tuple(genome))
    if not terms:
        return {
            "term_count": 0,
            "expression_size": 0,
            "max_depth": 0,
            "mean_depth": 0.0,
            "operator_cost": 0.0,
            "interaction_order": 0,
            "parameter_count": 0,
            "feature_count": 0,
            "feature_names": [],
            "parameter_names": [],
            "unary_op_count": 0,
            "binary_op_count": 0,
        }

    depths: list[int] = []
    node_counts: list[int] = []
    operator_costs: list[float] = []
    interaction_orders: list[int] = []
    feature_names: set[str] = set()
    parameter_names: set[str] = set()
    unary_op_count = 0
    binary_op_count = 0

    for term in terms:
        expr = dict(term.get("expr", {}))
        feature_indices = sorted(idx for idx in _expr_collect_feature_indices(expr) if idx >= 0)
        parameter_keys = _expr_collect_parameter_names(expr)
        ops = _expr_operator_counter(expr)

        depths.append(int(_expr_depth(expr)))
        node_counts.append(int(_expr_node_count(expr)))
        operator_costs.append(float(_expr_operator_cost(expr)))
        interaction_orders.append(int(len(feature_indices)))
        feature_names.update(f"x{idx}" for idx in feature_indices)
        parameter_names.update(str(v) for v in parameter_keys)
        unary_op_count += int(sum(v for k, v in ops.items() if str(k).startswith("unary:")))
        binary_op_count += int(sum(v for k, v in ops.items() if str(k).startswith("binary:")))

    return {
        "term_count": int(len(terms)),
        "expression_size": int(sum(node_counts)),
        "max_depth": int(max(depths)),
        "mean_depth": float(np.mean(np.asarray(depths, dtype=float))),
        "operator_cost": float(sum(operator_costs)),
        "interaction_order": int(max(interaction_orders)),
        "parameter_count": int(len(parameter_names)),
        "feature_count": int(len(feature_names)),
        "feature_names": [str(v) for v in sorted(feature_names)],
        "parameter_names": [str(v) for v in sorted(parameter_names)],
        "unary_op_count": int(unary_op_count),
        "binary_op_count": int(binary_op_count),
    }


def _merge_complexity_metrics(parts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    items = {str(name): dict(payload) for name, payload in dict(parts).items() if isinstance(payload, Mapping)}
    if not items:
        return _build_complexity_metrics(tuple())

    term_counts = [int(dict(payload).get("term_count", 0)) for payload in items.values()]
    mean_depth_numer = 0.0
    mean_depth_denom = 0
    feature_names: set[str] = set()
    parameter_names: set[str] = set()
    for payload in items.values():
        count = int(dict(payload).get("term_count", 0))
        mean_depth_numer += float(dict(payload).get("mean_depth", 0.0)) * float(max(1, count))
        mean_depth_denom += max(1, count)
        feature_names.update(str(v) for v in tuple(dict(payload).get("feature_names", ())))
        parameter_names.update(str(v) for v in tuple(dict(payload).get("parameter_names", ())))

    return {
        "term_count": int(sum(term_counts)),
        "expression_size": int(sum(int(dict(payload).get("expression_size", 0)) for payload in items.values())),
        "max_depth": int(max(int(dict(payload).get("max_depth", 0)) for payload in items.values())),
        "mean_depth": float(0.0 if mean_depth_denom <= 0 else mean_depth_numer / float(mean_depth_denom)),
        "operator_cost": float(sum(float(dict(payload).get("operator_cost", 0.0)) for payload in items.values())),
        "interaction_order": int(max(int(dict(payload).get("interaction_order", 0)) for payload in items.values())),
        "parameter_count": int(len(parameter_names)),
        "feature_count": int(len(feature_names)),
        "feature_names": [str(v) for v in sorted(feature_names)],
        "parameter_names": [str(v) for v in sorted(parameter_names)],
        "unary_op_count": int(sum(int(dict(payload).get("unary_op_count", 0)) for payload in items.values())),
        "binary_op_count": int(sum(int(dict(payload).get("binary_op_count", 0)) for payload in items.values())),
        "by_scope": {str(name): dict(payload) for name, payload in sorted(items.items(), key=lambda item: item[0])},
    }


def _collect_nested_values(payload: Any, key: str) -> list[Any]:
    out: list[Any] = []
    if isinstance(payload, Mapping):
        for current_key, current_value in payload.items():
            if str(current_key) == str(key):
                out.append(current_value)
            out.extend(_collect_nested_values(current_value, key))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            out.extend(_collect_nested_values(item, key))
    return out


def _coerce_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _coerce_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=float).reshape(-1)
        return [float(v) for v in arr.tolist() if np.isfinite(v)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        out: list[float] = []
        for item in value:
            val = _coerce_float_or_none(item)
            if val is not None:
                out.append(float(val))
        return out
    val = _coerce_float_or_none(value)
    return [] if val is None else [float(val)]


def _coerce_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        out: list[int] = []
        for item in value:
            val = _coerce_int_or_none(item)
            if val is not None:
                out.append(int(val))
        return out
    val = _coerce_int_or_none(value)
    return [] if val is None else [int(val)]


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [] if not text else [text]


def _normalize_fold_stability_report(
    fold_report: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    report = dict(fold_report or {})
    metric_key_map: tuple[tuple[str, str], ...] = (
        ("coverage_error", "fold_coverage_error"),
        ("pinaw", "fold_pinaw"),
        ("interval_score", "fold_interval_score"),
        ("picp", "fold_picp"),
        ("mean_width", "fold_mean_width"),
        ("rmse", "fold_rmse"),
    )
    metrics_by_name = {
        metric_name: _coerce_float_list(report.get(field_name))
        for metric_name, field_name in metric_key_map
        if _coerce_float_list(report.get(field_name))
    }
    objective_schema = _coerce_str_list(report.get("objective_schema"))
    metric_schema = [metric_name for metric_name, _ in metric_key_map if metric_name in metrics_by_name]

    fold_count_candidates = [len(values) for values in metrics_by_name.values() if values]
    explicit_subset_size = _coerce_int_or_none(report.get("subset_size"))
    fold_count = max(fold_count_candidates, default=0)

    summary = {
        "coverage_error_mean": _coerce_float_or_none(report.get("coverage_error_mean")),
        "pinaw_mean": _coerce_float_or_none(report.get("pinaw_mean")),
        "interval_score_mean": _coerce_float_or_none(report.get("interval_score_mean")),
        "picp_mean": _coerce_float_or_none(report.get("picp_mean")),
        "mean_width_mean": _coerce_float_or_none(report.get("mean_width_mean")),
        "rmse_mean": _coerce_float_or_none(report.get("rmse_mean")),
        "rmse_std": _coerce_float_or_none(report.get("rmse_std")),
        "rmse_drift": _coerce_float_or_none(report.get("rmse_drift")),
    }
    selection = {
        "coverage_error_threshold": _coerce_float_or_none(report.get("selection_coverage_error_threshold")),
        "meets_coverage_threshold": (
            None
            if report.get("selection_meets_coverage_threshold") is None
            else bool(report.get("selection_meets_coverage_threshold"))
        ),
    }
    subset = {
        "subset_size": int(explicit_subset_size if explicit_subset_size is not None else len(_coerce_int_list(report.get("subset_idx")))),
        "subset_idx": _coerce_int_list(report.get("subset_idx")),
        "subset_names": _coerce_str_list(report.get("subset_names")),
        "subset_families": _coerce_str_list(report.get("subset_families")),
    }
    complexity_context = {
        "complexity_raw": _coerce_float_or_none(report.get("complexity_raw")),
        "family_concentration": _coerce_float_or_none(report.get("family_concentration")),
        "feature_concentration": _coerce_float_or_none(report.get("feature_concentration")),
        "tuned_l2": _coerce_float_or_none(report.get("tuned_l2")),
        "strict4_min_train_ratio": _coerce_float_or_none(report.get("strict4_min_train_ratio")),
        "complexity_scale": _coerce_float_or_none(report.get("complexity_scale")),
        "family_penalty_scale": _coerce_float_or_none(report.get("family_penalty_scale")),
        "feature_penalty_scale": _coerce_float_or_none(report.get("feature_penalty_scale")),
        "drift_weight": _coerce_float_or_none(report.get("drift_weight")),
    }
    fold_branch_detail = report.get("fold_branch_detail")
    fold_interval_info = report.get("fold_interval_info")
    detail_counts = {
        "branch_detail_count": len(tuple(fold_branch_detail))
        if isinstance(fold_branch_detail, Sequence) and not isinstance(fold_branch_detail, (str, bytes, bytearray))
        else 0,
        "interval_info_count": len(tuple(fold_interval_info))
        if isinstance(fold_interval_info, Sequence) and not isinstance(fold_interval_info, (str, bytes, bytearray))
        else 0,
    }
    detail_available = {
        "branch_detail": bool(detail_counts["branch_detail_count"]),
        "interval_info": bool(detail_counts["interval_info_count"]),
        "decode_meta": isinstance(report.get("decode_meta"), Mapping),
    }

    return {
        "schema_version": int(SYMBOLIC_FOLD_STABILITY_SCHEMA_VERSION),
        "source": str(source or "metadata.fold_report"),
        "objective_schema": objective_schema,
        "metric_schema": metric_schema,
        "fold_count": int(fold_count),
        "metrics_by_name": metrics_by_name,
        "summary": {str(k): v for k, v in summary.items()},
        "selection": {str(k): v for k, v in selection.items()},
        "subset": subset,
        "complexity_context": {str(k): v for k, v in complexity_context.items()},
        "detail_counts": detail_counts,
        "detail_available": detail_available,
        "raw_report": _jsonable(report),
    }


def _build_stability_metrics(
    *,
    metadata: Mapping[str, Any],
    residual_std: Sequence[float] | np.ndarray,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    meta = dict(metadata or {})
    rs = np.asarray(residual_std, dtype=float).reshape(-1)
    search_trace = dict(meta.get("search_trace", {})) if isinstance(meta.get("search_trace"), Mapping) else {}
    iterations = tuple(search_trace.get("iterations", ()))
    grad_values = [
        float(dict(item).get("grad_stability"))
        for item in iterations
        if isinstance(item, Mapping) and dict(item).get("grad_stability") is not None
    ]
    signal_signatures = sorted(
        {
            str(value)
            for value in _collect_nested_values(meta, "signal_signature")
            if str(value).strip()
        }
    )

    fold_report: Any = {}
    fold_source = ""
    for source_name, candidate in (
        ("metadata.fold_report", meta.get("fold_report")),
        (
            "metadata.search.fold_report",
            dict(meta.get("search", {})).get("fold_report") if isinstance(meta.get("search"), Mapping) else None,
        ),
        (
            "metadata.data_metadata.fold_report",
            dict(meta.get("data_metadata", {})).get("fold_report") if isinstance(meta.get("data_metadata"), Mapping) else None,
        ),
    ):
        if isinstance(candidate, Mapping):
            fold_report = dict(candidate)
            fold_source = str(source_name)
            break

    normalized_fold = (
        _normalize_fold_stability_report(fold_report, source=fold_source)
        if isinstance(fold_report, Mapping) and fold_report
        else {}
    )
    fold_summary = dict(normalized_fold.get("summary", {})) if isinstance(normalized_fold.get("summary"), Mapping) else {}
    fold_selection = (
        dict(normalized_fold.get("selection", {})) if isinstance(normalized_fold.get("selection"), Mapping) else {}
    )
    fold_complexity = (
        dict(normalized_fold.get("complexity_context", {}))
        if isinstance(normalized_fold.get("complexity_context"), Mapping)
        else {}
    )

    out = {
        "residual_std_mean": float(np.mean(rs)) if rs.size > 0 else 0.0,
        "residual_std_max": float(np.max(rs)) if rs.size > 0 else 0.0,
        "fold_stability_available": bool(normalized_fold),
        "fold_stability": normalized_fold,
        "fold_summary": fold_summary,
        "fold_schema_version": int(normalized_fold.get("schema_version", 0) or 0),
        "fold_metric_schema": tuple(normalized_fold.get("metric_schema", tuple())),
        "fold_objective_schema": tuple(normalized_fold.get("objective_schema", tuple())),
        "fold_count": int(normalized_fold.get("fold_count", 0) or 0),
        "selection_meets_coverage_threshold": fold_selection.get("meets_coverage_threshold"),
        "selection_coverage_error_threshold": fold_selection.get("coverage_error_threshold"),
        "coverage_error_mean": fold_summary.get("coverage_error_mean"),
        "pinaw_mean": fold_summary.get("pinaw_mean"),
        "interval_score_mean": fold_summary.get("interval_score_mean"),
        "picp_mean": fold_summary.get("picp_mean"),
        "mean_width_mean": fold_summary.get("mean_width_mean"),
        "rmse_mean": fold_summary.get("rmse_mean"),
        "rmse_std": fold_summary.get("rmse_std"),
        "rmse_drift": fold_summary.get("rmse_drift"),
        "family_concentration": fold_complexity.get("family_concentration"),
        "feature_concentration": fold_complexity.get("feature_concentration"),
        "gradient_signal_signatures": [str(v) for v in signal_signatures],
        "gradient_stability_min": None if not grad_values else float(min(grad_values)),
        "gradient_stability_max": None if not grad_values else float(max(grad_values)),
        "gradient_stability_last": None if not grad_values else float(grad_values[-1]),
    }
    if extra:
        for key, value in dict(extra).items():
            out[str(key)] = value
    return out


def _build_candidate_lineage(metadata: Mapping[str, Any]) -> dict[str, Any]:
    meta = dict(metadata or {})
    symbolic_block = dict(meta.get("symbolic", {})) if isinstance(meta.get("symbolic"), Mapping) else {}
    search_block = dict(meta.get("search", {})) if isinstance(meta.get("search"), Mapping) else {}
    search_trace = dict(meta.get("search_trace", {})) if isinstance(meta.get("search_trace"), Mapping) else {}
    aggregate_manifest = dict(meta.get("aggregate_manifest", {})) if isinstance(meta.get("aggregate_manifest"), Mapping) else {}
    genome_build = dict(symbolic_block.get("genome_build", {})) if isinstance(symbolic_block.get("genome_build"), Mapping) else {}
    structure_engine = (
        dict(symbolic_block.get("structure_engine", {}))
        if isinstance(symbolic_block.get("structure_engine"), Mapping)
        else {}
    )
    if not structure_engine and isinstance(meta.get("symbolic_family"), Mapping):
        structure_engine = dict(dict(meta.get("symbolic_family", {})).get("structure_engine", {}))

    if aggregate_manifest:
        return {
            "source": "aggregate_manifest",
            "structure_engine": structure_engine,
            "selected_regime_keys": [
                str(v) for v in tuple(aggregate_manifest.get("selected_regime_keys", ()))
            ],
            "failed_regimes": dict(aggregate_manifest.get("failed_regimes", {})),
            "local_regimes": dict(aggregate_manifest.get("local_regimes", {})),
        }

    if search_trace:
        score_trace = [float(v) for v in tuple(search_trace.get("score_trace", ()))]
        iterations = [dict(v) for v in tuple(search_trace.get("iterations", ())) if isinstance(v, Mapping)]
        return {
            "source": "search_trace",
            "structure_engine": structure_engine,
            "genome_build": genome_build,
            "iteration_count": int(len(iterations)),
            "score_trace": score_trace,
            "best_score": None if not score_trace else float(min(score_trace)),
            "last_iteration": None if not iterations else dict(iterations[-1]),
        }

    if genome_build or search_block or structure_engine:
        return {
            "source": "trainer_metadata",
            "structure_engine": structure_engine,
            "genome_build": genome_build,
            "search_summary": {
                "iterations": int(search_block.get("iterations", 0)),
                "terms": int(search_block.get("terms", 0)),
                "base_metrics": dict(search_block.get("base_metrics", {})),
                "final_metrics": dict(search_block.get("final_metrics", {})),
            },
        }

    return {
        "source": "not_recorded",
        "structure_engine": {},
    }


def _build_simplification_trace(metadata: Mapping[str, Any]) -> dict[str, Any]:
    meta = dict(metadata or {})
    symbolic_block = dict(meta.get("symbolic", {})) if isinstance(meta.get("symbolic"), Mapping) else {}
    steps: Sequence[Any] = tuple()
    source = "not_recorded"
    for candidate, candidate_source in (
        (meta.get("simplification_trace"), "metadata"),
        (symbolic_block.get("simplification_trace"), "symbolic_metadata"),
    ):
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
            steps = tuple(candidate)
            source = str(candidate_source)
            break
    return {
        "available": bool(steps),
        "source": source,
        "steps": [dict(v) if isinstance(v, Mapping) else v for v in steps],
    }


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not str(value).strip():
            continue
        return value
    return None


def _symbolic_family_block(metadata: Mapping[str, Any]) -> dict[str, Any]:
    meta = dict(metadata or {})
    direct = _mapping_or_empty(meta.get("symbolic_family"))
    fallback = _mapping_or_empty(meta.get("symbolic_family_spec"))
    training_signature = _mapping_or_empty(meta.get("training_signature"))
    signature_meta = _mapping_or_empty(training_signature.get("metadata"))
    embedded = _mapping_or_empty(signature_meta.get("symbolic_family"))
    out = dict(fallback)
    out.update(direct)
    out.update(embedded)
    return out


def _symbolic_structure_engine_block(metadata: Mapping[str, Any]) -> dict[str, Any]:
    family = _symbolic_family_block(metadata)
    symbolic = _mapping_or_empty(metadata.get("symbolic"))
    family_engine = _mapping_or_empty(family.get("structure_engine"))
    symbolic_engine = _mapping_or_empty(symbolic.get("structure_engine"))
    out = dict(symbolic_engine)
    out.update(family_engine)
    return out


def _resolve_symbolic_stage_protocol(metadata: Mapping[str, Any]) -> dict[str, Any]:
    meta = dict(metadata or {})
    symbolic = _mapping_or_empty(meta.get("symbolic"))
    inner_symbolic_search = _mapping_or_empty(symbolic.get("inner_symbolic_search"))
    structure_engine = _symbolic_structure_engine_block(metadata)
    structure_engine_meta = _mapping_or_empty(structure_engine.get("metadata"))
    stage_protocols = _mapping_or_empty(meta.get("stage_head_protocols"))
    if not stage_protocols:
        stage_protocols = _mapping_or_empty(symbolic.get("stage_head_protocols"))
    assembler_stage = _mapping_or_empty(stage_protocols.get("assembler"))
    basis_stage = _mapping_or_empty(stage_protocols.get("basis_discovery"))
    if not assembler_stage:
        assembler_stage = _mapping_or_empty(inner_symbolic_search.get("stage_head_spec"))
    basis_context = _mapping_or_empty(meta.get("basis_context"))
    if not basis_context:
        basis_context = _mapping_or_empty(symbolic.get("basis_context"))
    if not basis_context:
        basis_context = _mapping_or_empty(inner_symbolic_search.get("basis_context"))
    object_gradient_pool = _mapping_or_empty(meta.get("basis_object_gradient_pool"))
    if not object_gradient_pool:
        object_gradient_pool = _mapping_or_empty(symbolic.get("basis_object_gradient_pool"))
    if not object_gradient_pool:
        object_gradient_pool = _mapping_or_empty(inner_symbolic_search.get("object_gradient_pool"))
    keys = (
        "structure_head",
        "prediction_head",
        "search_input_space",
        "pool_expansion_unit",
        "gradient_guidance_mode",
        "basis_binding_mode",
        "escape_policy",
    )
    effective: dict[str, Any] = {}
    for key in keys:
        value = _first_present(
            meta.get(key),
            symbolic.get(key),
            assembler_stage.get(key),
            structure_engine.get(key),
            structure_engine_meta.get(key),
        )
        if value is not None:
            effective[str(key)] = value
    effective["basis_source"] = _first_present(
        meta.get("basis_source"),
        symbolic.get("basis_source"),
        basis_context.get("basis_source"),
    )
    effective["orchestration_mode"] = _first_present(
        meta.get("orchestration_mode"),
        symbolic.get("orchestration_mode"),
        "basis_discovery_then_basis_conditioned_expression" if assembler_stage else None,
    )
    return {
        "effective": effective,
        "stage_protocols": stage_protocols,
        "basis_discovery_stage": basis_stage,
        "assembler_stage": assembler_stage,
        "basis_context": basis_context,
        "object_gradient_pool": object_gradient_pool,
    }


def _symbolic_structure_contracts_block(metadata: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    family = _symbolic_family_block(metadata)
    out = {
        str(key): dict(value)
        for key, value in _mapping_or_empty(family.get("structure_contracts")).items()
        if isinstance(value, Mapping)
    }
    alias_map = (
        ("regime_discovery_contract", "regime_discovery"),
        ("basis_discovery_contract", "basis_discovery"),
        ("budgeted_symbolic_assembler_contract", "budgeted_symbolic_assembler"),
    )
    for source_key, target_key in alias_map:
        if target_key in out:
            continue
        value = family.get(source_key)
        if isinstance(value, Mapping):
            out[str(target_key)] = dict(value)
    return out


def _sorted_str_int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, int] = {}
    for key, raw in dict(value).items():
        number = _coerce_int_or_none(raw)
        if number is not None:
            out[str(key)] = int(number)
    return {str(k): int(v) for k, v in sorted(out.items(), key=lambda item: item[0])}


def _gate_piecewise_blocks(metadata: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = dict(metadata or {})
    gate_piecewise = _mapping_or_empty(meta.get("gate_piecewise"))
    aggregate_manifest = _mapping_or_empty(meta.get("aggregate_manifest"))
    if not aggregate_manifest:
        symbolic = _mapping_or_empty(meta.get("symbolic"))
        aggregate_manifest = _mapping_or_empty(symbolic.get("aggregate_manifest"))
    return gate_piecewise, aggregate_manifest


def _gate_feature_payload(
    *,
    gate_feature_names: Sequence[str] | None = None,
    gate_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    names = tuple(str(v) for v in tuple(gate_feature_names or tuple()) if str(v).strip())
    indices = tuple(int(v) for v in tuple(gate_indices or tuple()))
    rows: list[dict[str, Any]] = []
    max_len = max(len(names), len(indices))
    for idx in range(max_len):
        row: dict[str, Any] = {}
        if idx < len(names):
            row["feature_name"] = str(names[idx])
        if idx < len(indices):
            row["feature_index"] = int(indices[idx])
        rows.append(row)
    return {
        "gate_features": rows,
        "gate_feature_count": int(len(rows)),
    }


def _selected_basis_payload(metadata: Mapping[str, Any]) -> Any:
    meta = dict(metadata or {})
    if meta.get("selected_basis") is not None:
        return meta.get("selected_basis")
    symbolic = _mapping_or_empty(meta.get("symbolic"))
    if symbolic.get("selected_basis") is not None:
        return symbolic.get("selected_basis")
    return None


def _collect_gate_basis_terms(metadata: Mapping[str, Any], *, local_basis_by_regime: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    meta = dict(metadata or {})
    gate_piecewise = _mapping_or_empty(meta.get("gate_piecewise"))
    aggregate_manifest = _mapping_or_empty(meta.get("aggregate_manifest"))
    explicit = gate_piecewise.get("gate_basis_terms", aggregate_manifest.get("gate_basis_terms"))
    rows: list[dict[str, Any]] = []
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes, bytearray)):
        for item in explicit:
            if isinstance(item, Mapping):
                rows.append(dict(item))
        if rows:
            return rows
    selected_basis = _selected_basis_payload(metadata)
    candidates: list[dict[str, Any]] = []
    if isinstance(selected_basis, Mapping):
        for value in dict(selected_basis).values():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                candidates.extend(dict(item) for item in value if isinstance(item, Mapping))
    elif isinstance(selected_basis, Sequence) and not isinstance(selected_basis, (str, bytes, bytearray)):
        candidates.extend(dict(item) for item in selected_basis if isinstance(item, Mapping))
    if local_basis_by_regime:
        for value in dict(local_basis_by_regime).values():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                candidates.extend(dict(item) for item in value if isinstance(item, Mapping))
    return [row for row in candidates if bool(row.get("uses_piecewise_gate"))]


def _basis_rows_from_genome(
    *,
    genome: Sequence[Mapping[str, Any]],
    parameter_values: Mapping[str, float],
    feature_names: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        _term_descriptor(
            term_index=int(term_index),
            term=dict(term),
            parameter_values=parameter_values,
            feature_names=feature_names,
        )
        for term_index, term in enumerate(tuple(genome))
    ]


def _basis_feature_union(*payloads: Any) -> list[str]:
    names: set[str] = set()
    for payload in payloads:
        for candidate in _collect_nested_values(payload, "feature_names"):
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
                for item in candidate:
                    text = str(item).strip()
                    if text:
                        names.add(text)
            else:
                text = str(candidate).strip()
                if text:
                    names.add(text)
    return [str(v) for v in sorted(names)]


def _count_basis_items(payload: Any) -> int:
    if isinstance(payload, Mapping):
        current = dict(payload)
        if "term_index" in current and ("expression_raw" in current or "expression_named" in current):
            return 1
        return sum(_count_basis_items(value) for value in current.values())
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return sum(_count_basis_items(item) for item in payload)
    return 0


def _first_mapping(*candidates: tuple[str, Any]) -> tuple[str, dict[str, Any]]:
    for source, value in candidates:
        if isinstance(value, Mapping) and dict(value):
            return str(source), dict(value)
    return "not_recorded", {}


def _extract_basis_semantics(metadata: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    meta = dict(metadata or {})
    symbolic = _mapping_or_empty(meta.get("symbolic"))
    search = _mapping_or_empty(meta.get("search"))
    return _first_mapping(
        ("metadata.basis_semantics", meta.get("basis_semantics")),
        ("metadata.symbolic.basis_semantics", symbolic.get("basis_semantics")),
        ("metadata.search.basis_semantics", search.get("basis_semantics")),
    )


def _extract_basis_overlap_report(metadata: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    meta = dict(metadata or {})
    symbolic = _mapping_or_empty(meta.get("symbolic"))
    search = _mapping_or_empty(meta.get("search"))
    return _first_mapping(
        ("metadata.basis_overlap_report", meta.get("basis_overlap_report")),
        ("metadata.symbolic.basis_overlap_report", symbolic.get("basis_overlap_report")),
        ("metadata.search.basis_overlap_report", search.get("basis_overlap_report")),
    )


def _extract_residual_complementarity_report(metadata: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    meta = dict(metadata or {})
    symbolic = _mapping_or_empty(meta.get("symbolic"))
    search = _mapping_or_empty(meta.get("search"))
    return _first_mapping(
        ("metadata.residual_complementarity_report", meta.get("residual_complementarity_report")),
        ("metadata.symbolic.residual_complementarity_report", symbolic.get("residual_complementarity_report")),
        ("metadata.search.residual_complementarity_report", search.get("residual_complementarity_report")),
    )


def _extract_semantic_dedup_report(metadata: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    meta = dict(metadata or {})
    symbolic = _mapping_or_empty(meta.get("symbolic"))
    search = _mapping_or_empty(meta.get("search"))
    return _first_mapping(
        ("metadata.semantic_dedup_report", meta.get("semantic_dedup_report")),
        ("metadata.symbolic.semantic_dedup_report", symbolic.get("semantic_dedup_report")),
        ("metadata.search.semantic_dedup_report", search.get("semantic_dedup_report")),
    )


def _build_orthogonality_status(
    *,
    metadata: Mapping[str, Any],
    basis_payload: Any,
    basis_contract: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report_source, report_payload = _extract_basis_overlap_report(metadata)
    residual_source, residual_payload = _extract_residual_complementarity_report(metadata)
    semantic_source, semantic_payload = _extract_semantic_dedup_report(metadata)
    contract_metadata = _mapping_or_empty(_mapping_or_empty(basis_contract).get("metadata"))
    out = {
        "status": "reported" if (report_payload or residual_payload or semantic_payload) else "not_recorded",
        "source": report_source,
        "basis_overlap_report": report_payload or None,
        "objectives": _coerce_str_list(contract_metadata.get("orthogonality_objectives")),
        "feature_union": _basis_feature_union(basis_payload),
        "orthogonality_score": _coerce_float_or_none(report_payload.get("orthogonality_score")),
        "pair_abs_corr_mean": _coerce_float_or_none(report_payload.get("pair_abs_corr_mean")),
        "pair_abs_corr_max": _coerce_float_or_none(report_payload.get("pair_abs_corr_max")),
        "residual_complementarity_status": "reported" if residual_payload else "not_recorded",
        "residual_complementarity_source": residual_source,
        "residual_gain_mean": _coerce_float_or_none(residual_payload.get("mean_marginal_r2_gain")),
        "residual_gain_min": _coerce_float_or_none(residual_payload.get("min_marginal_r2_gain")),
        "semantic_dedup_status": "reported" if semantic_payload else "not_recorded",
        "semantic_dedup_source": semantic_source,
        "semantic_unique_ratio": _coerce_float_or_none(semantic_payload.get("semantic_unique_ratio")),
        "piecewise_gate_term_count": _coerce_int_or_none(semantic_payload.get("piecewise_gate_term_count")),
    }
    if extra:
        for key, value in dict(extra).items():
            out[str(key)] = _jsonable(value)
    return out


def _expression_leaf_paths(payload: Any, *, prefix: str = "") -> list[str]:
    if isinstance(payload, Mapping):
        out: list[str] = []
        for key, value in dict(payload).items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_expression_leaf_paths(value, prefix=child_prefix))
        return out
    if isinstance(payload, str):
        return [str(prefix)] if str(prefix).strip() else []
    return []


def _count_expression_strings(payload: Any) -> int:
    if isinstance(payload, Mapping):
        return sum(_count_expression_strings(value) for value in dict(payload).values())
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return sum(_count_expression_strings(value) for value in payload)
    return 1 if isinstance(payload, str) else 0


def _extract_budget_values(
    metadata: Mapping[str, Any],
    *,
    declared_axes: Sequence[str],
) -> dict[str, Any]:
    meta = dict(metadata or {})
    search = _mapping_or_empty(meta.get("search"))
    search_trace = _mapping_or_empty(meta.get("search_trace"))
    training_init = _mapping_or_empty(meta.get("training_init"))
    assembler_budget = _mapping_or_empty(meta.get("assembler_budget"))
    symbolic = _mapping_or_empty(meta.get("symbolic"))
    symbolic_assembler_budget = _mapping_or_empty(symbolic.get("assembler_budget"))
    search_trace_config = _mapping_or_empty(search_trace.get("config"))
    recorded_budget = _mapping_or_empty(assembler_budget.get("recorded_values"))
    if not recorded_budget:
        recorded_budget = _mapping_or_empty(symbolic_assembler_budget.get("recorded_values"))
    out: dict[str, Any] = {}
    for axis in tuple(str(v) for v in tuple(declared_axes) if str(v).strip()):
        if axis in recorded_budget and recorded_budget.get(axis) is not None:
            out[str(axis)] = _jsonable(recorded_budget.get(axis))
            continue
        for container in (meta, search, search_trace_config, training_init):
            if axis in container and container.get(axis) is not None:
                out[str(axis)] = _jsonable(container.get(axis))
                break
    return out


def _build_regime_structure(
    *,
    metadata: Mapping[str, Any],
    piecewise_enabled: bool,
    gate_feature_names: Sequence[str] | None = None,
    gate_indices: Sequence[int] | None = None,
    selected_regime_keys: Sequence[str] | None = None,
    local_regimes: Mapping[str, Any] | None = None,
    failed_regimes: Mapping[str, Any] | None = None,
    counts_all: Mapping[str, Any] | None = None,
    counts_selected: Mapping[str, Any] | None = None,
    counts_skipped: Mapping[str, Any] | None = None,
    blend_kappa: float | None = None,
) -> dict[str, Any]:
    structure_engine = _symbolic_structure_engine_block(metadata)
    structure_contracts = _symbolic_structure_contracts_block(metadata)
    regime_contract = _mapping_or_empty(structure_contracts.get("regime_discovery"))
    gate_piecewise, aggregate_manifest = _gate_piecewise_blocks(metadata)
    selected = [str(v) for v in tuple(selected_regime_keys or aggregate_manifest.get("selected_regime_keys", tuple())) if str(v).strip()]
    local_payload = (
        dict(local_regimes)
        if isinstance(local_regimes, Mapping)
        else _mapping_or_empty(aggregate_manifest.get("local_regimes"))
    )
    failed_payload = (
        dict(failed_regimes)
        if isinstance(failed_regimes, Mapping)
        else _mapping_or_empty(aggregate_manifest.get("failed_regimes"))
    )
    counts_all_payload = (
        _sorted_str_int_mapping(counts_all)
        if isinstance(counts_all, Mapping)
        else _sorted_str_int_mapping(aggregate_manifest.get("counts_all"))
    )
    counts_selected_payload = (
        _sorted_str_int_mapping(counts_selected)
        if isinstance(counts_selected, Mapping)
        else _sorted_str_int_mapping(aggregate_manifest.get("counts_selected"))
    )
    counts_skipped_payload = (
        _sorted_str_int_mapping(counts_skipped)
        if isinstance(counts_skipped, Mapping)
        else _sorted_str_int_mapping(aggregate_manifest.get("counts_skipped"))
    )
    gate_names = _coerce_str_list(
        gate_feature_names
        if gate_feature_names is not None
        else gate_piecewise.get("gate_feature_names", aggregate_manifest.get("gate_feature_names"))
    )
    gate_idx = _coerce_int_list(
        gate_indices if gate_indices is not None else gate_piecewise.get("gate_indices", aggregate_manifest.get("gate_indices"))
    )
    enabled = bool(piecewise_enabled or selected or local_payload or gate_piecewise or aggregate_manifest)
    source = (
        "aggregate_manifest"
        if aggregate_manifest
        else "metadata.gate_piecewise"
        if gate_piecewise
        else "symbolic_family.structure_contracts"
        if regime_contract
        else "metadata.symbolic_family"
    )
    return {
        "source": source,
        "mode": (
            str(regime_contract.get("regime_mode"))
            if enabled and str(regime_contract.get("regime_mode", "")).strip()
            else "piecewise_gate"
            if enabled
            else "global_only"
        ),
        "piecewise_enabled": bool(enabled),
        "structure_mode": str(structure_engine.get("structure_mode", "unknown")),
        "search_driver": str(structure_engine.get("search_driver", "unknown")),
        "gate_feature_names": gate_names,
        "gate_indices": [int(v) for v in gate_idx],
        "gate_threshold": _coerce_float_or_none(gate_piecewise.get("gate_threshold", aggregate_manifest.get("gate_threshold"))),
        "gate_min_leaf": _coerce_int_or_none(gate_piecewise.get("gate_min_leaf", aggregate_manifest.get("gate_min_leaf"))),
        "gate_max_local_models": _coerce_int_or_none(
            gate_piecewise.get("gate_max_local_models", aggregate_manifest.get("gate_max_local_models"))
        ),
        "selected_regime_keys": selected,
        "failed_regimes": _jsonable(failed_payload),
        "local_regimes": _jsonable(local_payload),
        "counts_all": counts_all_payload,
        "counts_selected": counts_selected_payload,
        "counts_skipped": counts_skipped_payload,
        "local_regime_count": int(len(selected) if selected else len(local_payload)),
        "blend_kappa": (
            _coerce_float_or_none(blend_kappa)
            if blend_kappa is not None
            else _coerce_float_or_none(gate_piecewise.get("blend_kappa", aggregate_manifest.get("blend_kappa")))
        ),
    }


def _build_basis_structure(
    *,
    metadata: Mapping[str, Any],
    global_basis: Any,
    local_basis_by_regime: Mapping[str, Any] | None = None,
    gate_basis: Any | None = None,
    basis_scope: str,
    source: str,
    basis_semantics_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    structure_contracts = _symbolic_structure_contracts_block(metadata)
    basis_contract = _mapping_or_empty(structure_contracts.get("basis_discovery"))
    stage_payload = _resolve_symbolic_stage_protocol(metadata)
    basis_stage = _mapping_or_empty(stage_payload.get("basis_discovery_stage"))
    basis_context = _mapping_or_empty(stage_payload.get("basis_context"))
    semantics_source, semantics_payload = _extract_basis_semantics(metadata)
    residual_source, residual_payload = _extract_residual_complementarity_report(metadata)
    semantic_dedup_source, semantic_dedup_payload = _extract_semantic_dedup_report(metadata)
    local_basis = {
        str(key): _jsonable(value)
        for key, value in sorted(dict(local_basis_by_regime or {}).items(), key=lambda item: item[0])
    }
    gate_payload = _jsonable(gate_basis or {})
    basis_semantics = {
        "source": semantics_source,
        "declared_scope": str(basis_contract.get("basis_scope", basis_scope or "global")),
        "orthogonality_objectives": _coerce_str_list(
            _mapping_or_empty(basis_contract.get("metadata")).get("orthogonality_objectives")
        ),
        "recorded": semantics_payload or None,
    }
    if basis_semantics_extra:
        for key, value in dict(basis_semantics_extra).items():
            basis_semantics[str(key)] = _jsonable(value)
    return {
        "source": str(source),
        "basis_scope": str(basis_scope),
        "basis_count": int(
            _count_basis_items(global_basis)
            + _count_basis_items(local_basis)
            + _count_basis_items(gate_payload)
        ),
        "basis_feature_union": _basis_feature_union(global_basis, local_basis, gate_payload),
        "global_basis": _jsonable(global_basis),
        "local_basis_by_regime": local_basis,
        "gate_basis": gate_payload,
        "orthogonality_status": _build_orthogonality_status(
            metadata=metadata,
            basis_payload={"global_basis": global_basis, "local_basis_by_regime": local_basis, "gate_basis": gate_payload},
            basis_contract=basis_contract,
            extra={"basis_scope": basis_scope},
        ),
        "basis_semantics": basis_semantics,
        "residual_complementarity": {
            "source": str(residual_source),
            "status": "reported" if residual_payload else "not_recorded",
            "recorded": None if not residual_payload else _jsonable(residual_payload),
        },
        "semantic_deduplication": {
            "source": str(semantic_dedup_source),
            "status": "reported" if semantic_dedup_payload else "not_recorded",
            "recorded": None if not semantic_dedup_payload else _jsonable(semantic_dedup_payload),
        },
        "basis_discovery_stage": _jsonable(basis_stage) if basis_stage else None,
        "basis_context": _jsonable(basis_context) if basis_context else None,
    }


def _build_assembler_structure(
    *,
    metadata: Mapping[str, Any],
    final_expression: Any,
    piecewise_enabled: bool,
    composition_targets: Sequence[str] | None = None,
    source: str,
) -> dict[str, Any]:
    structure_contracts = _symbolic_structure_contracts_block(metadata)
    assembler_contract = _mapping_or_empty(structure_contracts.get("budgeted_symbolic_assembler"))
    assembler_metadata = _mapping_or_empty(assembler_contract.get("metadata"))
    stage_payload = _resolve_symbolic_stage_protocol(metadata)
    assembler_stage = _mapping_or_empty(stage_payload.get("assembler_stage"))
    basis_context = _mapping_or_empty(stage_payload.get("basis_context"))
    object_gradient_pool = _mapping_or_empty(stage_payload.get("object_gradient_pool"))
    explicit_budget = _mapping_or_empty(metadata.get("assembler_budget"))
    if not explicit_budget:
        explicit_budget = _mapping_or_empty(_mapping_or_empty(metadata.get("symbolic")).get("assembler_budget"))
    declared_axes = _coerce_str_list(assembler_metadata.get("budget_axes"))
    recorded_budget = _extract_budget_values(metadata, declared_axes=declared_axes)
    targets = tuple(str(v) for v in tuple(composition_targets or tuple(_expression_leaf_paths(final_expression))) if str(v).strip())
    return {
        "source": str(explicit_budget.get("source") or source),
        "assembler_mode": (
            str(explicit_budget.get("assembler_mode"))
            if str(explicit_budget.get("assembler_mode", "")).strip()
            else str(assembler_contract.get("assembler_mode"))
            if piecewise_enabled and str(assembler_contract.get("assembler_mode", "")).strip()
            else "piecewise_budgeted_symbolic_regression"
            if piecewise_enabled
            else "budgeted_symbolic_regression"
        ),
        "assembly_scope": "global+local" if piecewise_enabled else "global",
        "uses_piecewise_gate": bool(piecewise_enabled),
        "budget_recorded": bool(recorded_budget),
        "budget": {
            "budget_scale": (
                str(explicit_budget.get("budget_scale"))
                if explicit_budget.get("budget_scale") is not None
                else None if assembler_metadata.get("budget_scale") is None else str(assembler_metadata.get("budget_scale"))
            ),
            "declared_axes": declared_axes,
            "recorded_values": recorded_budget,
        },
        "output_expression_count": int(
            _coerce_int_or_none(explicit_budget.get("output_expression_count"))
            or _count_expression_strings(final_expression)
        ),
        "composition_targets": [str(v) for v in targets],
        "structure_head": assembler_stage.get("structure_head"),
        "prediction_head": assembler_stage.get("prediction_head"),
        "search_input_space": assembler_stage.get("search_input_space"),
        "pool_expansion_unit": assembler_stage.get("pool_expansion_unit"),
        "gradient_guidance_mode": assembler_stage.get("gradient_guidance_mode"),
        "basis_binding_mode": assembler_stage.get("basis_binding_mode"),
        "escape_policy": assembler_stage.get("escape_policy"),
        "basis_conditioned": bool(str(assembler_stage.get("search_input_space", "")).strip() == "basis_object_space"),
        "stage_protocol": _jsonable(assembler_stage) if assembler_stage else None,
        "basis_context": _jsonable(basis_context) if basis_context else None,
        "object_gradient_pool": _jsonable(object_gradient_pool) if object_gradient_pool else None,
    }


def _build_piecewise_gate_basis(
    *,
    metadata: Mapping[str, Any],
    gate_feature_names: Sequence[str] | None = None,
    gate_indices: Sequence[int] | None = None,
    blend_kappa: float | None = None,
    selected_regime_keys: Sequence[str] | None = None,
    failed_regimes: Mapping[str, Any] | None = None,
    counts_all: Mapping[str, Any] | None = None,
    counts_selected: Mapping[str, Any] | None = None,
    counts_skipped: Mapping[str, Any] | None = None,
    local_basis_by_regime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    gate_piecewise, aggregate_manifest = _gate_piecewise_blocks(metadata)
    gate_names = _coerce_str_list(
        gate_feature_names
        if gate_feature_names is not None
        else gate_piecewise.get("gate_feature_names", aggregate_manifest.get("gate_feature_names"))
    )
    gate_idx = _coerce_int_list(
        gate_indices if gate_indices is not None else gate_piecewise.get("gate_indices", aggregate_manifest.get("gate_indices"))
    )
    selected = [str(v) for v in tuple(selected_regime_keys or aggregate_manifest.get("selected_regime_keys", tuple())) if str(v).strip()]
    local_basis = {
        str(key): value
        for key, value in dict(local_basis_by_regime or {}).items()
    }
    gate_basis_terms = _collect_gate_basis_terms(metadata, local_basis_by_regime=local_basis_by_regime)
    available = bool(gate_piecewise or aggregate_manifest or gate_names or gate_idx or local_basis or gate_basis_terms)
    enabled = bool(selected or local_basis or gate_basis_terms)
    return {
        "available": bool(available),
        "enabled": bool(enabled),
        "status": "enabled" if enabled else "configured" if available else "not_recorded",
        "source": (
            "aggregate_manifest"
            if aggregate_manifest
            else "metadata.gate_piecewise"
            if gate_piecewise
            else "artifact_local_basis"
            if local_basis
            else "not_recorded"
        ),
        "gate_feature_names": gate_names,
        "gate_indices": [int(v) for v in gate_idx],
        "gate_threshold": _coerce_float_or_none(gate_piecewise.get("gate_threshold", aggregate_manifest.get("gate_threshold"))),
        "gate_min_leaf": _coerce_int_or_none(gate_piecewise.get("gate_min_leaf", aggregate_manifest.get("gate_min_leaf"))),
        "gate_max_local_models": _coerce_int_or_none(
            gate_piecewise.get("gate_max_local_models", aggregate_manifest.get("gate_max_local_models"))
        ),
        "blend_kappa": (
            _coerce_float_or_none(blend_kappa)
            if blend_kappa is not None
            else _coerce_float_or_none(gate_piecewise.get("blend_kappa", aggregate_manifest.get("blend_kappa")))
        ),
        "selected_regime_keys": selected,
        "failed_regimes": _jsonable(
            dict(failed_regimes)
            if isinstance(failed_regimes, Mapping)
            else _mapping_or_empty(aggregate_manifest.get("failed_regimes"))
        ),
        "counts_all": (
            _sorted_str_int_mapping(counts_all)
            if isinstance(counts_all, Mapping)
            else _sorted_str_int_mapping(aggregate_manifest.get("counts_all"))
        ),
        "counts_selected": (
            _sorted_str_int_mapping(counts_selected)
            if isinstance(counts_selected, Mapping)
            else _sorted_str_int_mapping(aggregate_manifest.get("counts_selected"))
        ),
        "counts_skipped": (
            _sorted_str_int_mapping(counts_skipped)
            if isinstance(counts_skipped, Mapping)
            else _sorted_str_int_mapping(aggregate_manifest.get("counts_skipped"))
        ),
        "local_basis_counts": {
            str(key): int(_count_basis_items(value))
            for key, value in sorted(local_basis.items(), key=lambda item: item[0])
        },
        "local_basis_keys": [str(key) for key in sorted(local_basis.keys())],
        "gate_basis_count": int(len(gate_basis_terms)),
        "gate_term_names": [str(dict(row).get("term_name", "")) for row in gate_basis_terms if str(dict(row).get("term_name", "")).strip()],
        "gate_basis_terms": _jsonable(gate_basis_terms),
        "gate_basis": _gate_feature_payload(gate_feature_names=gate_names, gate_indices=gate_idx),
    }


def _resolve_head_semantics(
    *,
    metadata: Mapping[str, Any],
    default_task: str,
    default_outputs: Sequence[str],
    default_objective_family: str,
    default_calibration_mode: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    task = str(default_task)
    outputs = tuple(str(v) for v in tuple(default_outputs))
    objective_family = str(default_objective_family)
    calibration_mode = str(default_calibration_mode)

    family_block = dict(metadata.get("symbolic_family", {})) if isinstance(metadata.get("symbolic_family"), Mapping) else {}
    head_block = dict(family_block.get("task_head", {})) if isinstance(family_block.get("task_head"), Mapping) else {}
    if head_block:
        task = str(head_block.get("task", task))
        outputs = tuple(str(v) for v in tuple(head_block.get("outputs", outputs)) if str(v))
        objective_family = str(head_block.get("objective_family", objective_family))
        calibration_mode = str(head_block.get("calibration_mode", calibration_mode))

    stage_payload = _resolve_symbolic_stage_protocol(metadata)
    effective_stage = _mapping_or_empty(stage_payload.get("effective"))
    semantics = "point_estimate" if str(task).strip().lower() == "point" else "prediction_interval"
    out = {
        "task": str(task),
        "outputs": [str(v) for v in outputs],
        "objective_family": str(objective_family),
        "calibration_mode": str(calibration_mode),
        "prediction_semantics": semantics,
        "structure_head": effective_stage.get("structure_head"),
        "prediction_head": effective_stage.get("prediction_head"),
        "search_input_space": effective_stage.get("search_input_space"),
        "pool_expansion_unit": effective_stage.get("pool_expansion_unit"),
        "gradient_guidance_mode": effective_stage.get("gradient_guidance_mode"),
        "basis_binding_mode": effective_stage.get("basis_binding_mode"),
        "escape_policy": effective_stage.get("escape_policy"),
        "basis_source": effective_stage.get("basis_source"),
        "orchestration_mode": effective_stage.get("orchestration_mode"),
        "basis_conditioned": bool(str(effective_stage.get("search_input_space", "")).strip() == "basis_object_space"),
        "stage_protocols": _jsonable(stage_payload.get("stage_protocols")),
    }
    if extra:
        for key, value in dict(extra).items():
            out[str(key)] = value
    return out


def symbolic_artifact_schema_descriptor(
    *,
    task: str,
    outputs: Sequence[str],
    objective_family: str,
    calibration_mode: str = "none",
    supports_piecewise: bool = False,
) -> dict[str, Any]:
    task_name = str(task).strip().lower() or "point"
    return {
        "schema_name": SYMBOLIC_ARTIFACT_SCHEMA_NAME,
        "schema_key": SYMBOLIC_ARTIFACT_SCHEMA_KEY,
        "schema_version": int(SYMBOLIC_ARTIFACT_SCHEMA_VERSION),
        "family": "symbolic",
        "heads": [task_name],
        "outputs": [str(v) for v in tuple(outputs)],
        "objective_family": str(objective_family),
        "calibration_mode": str(calibration_mode),
        "artifact_schema_fields": [str(v) for v in SYMBOLIC_ARTIFACT_FIELDS],
        "complexity_fields": [str(v) for v in SYMBOLIC_COMPLEXITY_FIELDS],
        "explainability_fields": [str(v) for v in SYMBOLIC_EXPLAINABILITY_FIELDS],
        "regime_fields": [str(v) for v in SYMBOLIC_REGIME_FIELDS],
        "basis_fields": [str(v) for v in SYMBOLIC_BASIS_FIELDS],
        "assembler_fields": [str(v) for v in SYMBOLIC_ASSEMBLER_FIELDS],
        "piecewise_gate_fields": [str(v) for v in SYMBOLIC_PIECEWISE_GATE_FIELDS],
        "stability_fields": [str(v) for v in SYMBOLIC_STABILITY_FIELDS],
        "truth_contract_recovery_fields": [str(v) for v in SYMBOLIC_TRUTH_CONTRACT_RECOVERY_FIELDS],
        "orthogonal_objective_fields": [str(v) for v in SYMBOLIC_ORTHOGONAL_OBJECTIVE_FIELDS],
        "supports_piecewise": bool(supports_piecewise),
    }


def merge_symbolic_artifact_schema_descriptors(
    descriptors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    items = [dict(value) for value in tuple(descriptors) if isinstance(value, Mapping)]
    if not items:
        return symbolic_artifact_schema_descriptor(
            task="point",
            outputs=("mean",),
            objective_family="regression",
            calibration_mode="none",
            supports_piecewise=False,
        )

    def _union_tuple(key: str) -> list[str]:
        return sorted({str(v) for item in items for v in tuple(item.get(key, ())) if str(v)})

    return {
        "schema_name": SYMBOLIC_ARTIFACT_SCHEMA_NAME,
        "schema_key": SYMBOLIC_ARTIFACT_SCHEMA_KEY,
        "schema_version": int(SYMBOLIC_ARTIFACT_SCHEMA_VERSION),
        "family": "symbolic",
        "heads": _union_tuple("heads"),
        "outputs": _union_tuple("outputs"),
        "objective_families": sorted({str(item.get("objective_family", "")) for item in items if str(item.get("objective_family", ""))}),
        "calibration_modes": sorted({str(item.get("calibration_mode", "")) for item in items if str(item.get("calibration_mode", ""))}),
        "artifact_schema_fields": [str(v) for v in SYMBOLIC_ARTIFACT_FIELDS],
        "complexity_fields": [str(v) for v in SYMBOLIC_COMPLEXITY_FIELDS],
        "explainability_fields": [str(v) for v in SYMBOLIC_EXPLAINABILITY_FIELDS],
        "regime_fields": [str(v) for v in SYMBOLIC_REGIME_FIELDS],
        "basis_fields": [str(v) for v in SYMBOLIC_BASIS_FIELDS],
        "assembler_fields": [str(v) for v in SYMBOLIC_ASSEMBLER_FIELDS],
        "piecewise_gate_fields": [str(v) for v in SYMBOLIC_PIECEWISE_GATE_FIELDS],
        "stability_fields": [str(v) for v in SYMBOLIC_STABILITY_FIELDS],
        "truth_contract_recovery_fields": [str(v) for v in SYMBOLIC_TRUTH_CONTRACT_RECOVERY_FIELDS],
        "orthogonal_objective_fields": [str(v) for v in SYMBOLIC_ORTHOGONAL_OBJECTIVE_FIELDS],
        "supports_piecewise": any(bool(item.get("supports_piecewise", False)) for item in items),
    }


def build_symbolic_structure_surface_payload(
    *,
    metadata: Mapping[str, Any],
    final_expression: Any,
    global_basis: Any,
    local_basis_by_regime: Mapping[str, Any] | None = None,
    gate_basis: Any | None = None,
    piecewise_enabled: bool,
    basis_scope: str,
    basis_source: str = "metadata.selected_basis",
    assembler_source: str = "metadata.assembler_budget",
    composition_targets: Sequence[str] | None = None,
    gate_feature_names: Sequence[str] | None = None,
    gate_indices: Sequence[int] | None = None,
    blend_kappa: float | None = None,
    selected_regime_keys: Sequence[str] | None = None,
    local_regimes: Mapping[str, Any] | None = None,
    failed_regimes: Mapping[str, Any] | None = None,
    counts_all: Mapping[str, Any] | None = None,
    counts_selected: Mapping[str, Any] | None = None,
    counts_skipped: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    regime_structure = _build_regime_structure(
        metadata=metadata,
        piecewise_enabled=bool(piecewise_enabled),
        gate_feature_names=gate_feature_names,
        gate_indices=gate_indices,
        selected_regime_keys=selected_regime_keys,
        local_regimes=local_regimes,
        failed_regimes=failed_regimes,
        counts_all=counts_all,
        counts_selected=counts_selected,
        counts_skipped=counts_skipped,
        blend_kappa=blend_kappa,
    )
    basis_structure = _build_basis_structure(
        metadata=metadata,
        global_basis=global_basis,
        local_basis_by_regime=local_basis_by_regime,
        gate_basis=gate_basis,
        basis_scope=str(basis_scope),
        source=str(basis_source),
    )
    assembler_structure = _build_assembler_structure(
        metadata=metadata,
        final_expression=final_expression,
        piecewise_enabled=bool(piecewise_enabled),
        composition_targets=composition_targets,
        source=str(assembler_source),
    )
    piecewise_gate_basis = _build_piecewise_gate_basis(
        metadata=metadata,
        gate_feature_names=gate_feature_names,
        gate_indices=gate_indices,
        blend_kappa=blend_kappa,
        selected_regime_keys=selected_regime_keys,
        failed_regimes=failed_regimes,
        counts_all=counts_all,
        counts_selected=counts_selected,
        counts_skipped=counts_skipped,
        local_basis_by_regime=local_basis_by_regime,
    )
    return {
        "regime_structure": regime_structure,
        "basis_structure": basis_structure,
        "assembler_structure": assembler_structure,
        "piecewise_gate_basis": piecewise_gate_basis,
    }


def build_symbolic_point_artifact_schema(
    *,
    artifact_type: str,
    artifact_id: str,
    genome: Sequence[Mapping[str, Any]],
    parameter_values: Mapping[str, float],
    readout_weight: np.ndarray,
    readout_bias: np.ndarray,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    residual_std: Sequence[float] | np.ndarray,
    metadata: Mapping[str, Any],
    final_expression: Mapping[str, str],
    normalized_expression: Mapping[str, str],
) -> dict[str, Any]:
    contributions = _build_target_term_contributions(
        genome=genome,
        parameter_values=parameter_values,
        readout_weight=readout_weight,
        feature_names=feature_names,
        target_names=target_names,
    )
    complexity = _build_complexity_metrics(genome)
    global_basis = _basis_rows_from_genome(
        genome=genome,
        parameter_values=parameter_values,
        feature_names=feature_names,
    )
    regime_structure = _build_regime_structure(
        metadata=metadata,
        piecewise_enabled=False,
    )
    basis_structure = _build_basis_structure(
        metadata=metadata,
        global_basis=global_basis,
        basis_scope="global",
        source="artifact_genome",
        basis_semantics_extra={
            "expression_targets": tuple(sorted(str(v) for v in dict(final_expression).keys())),
        },
    )
    assembler_structure = _build_assembler_structure(
        metadata=metadata,
        final_expression=final_expression,
        piecewise_enabled=False,
        composition_targets=tuple(sorted(str(v) for v in dict(final_expression).keys())),
        source="artifact_expression",
    )
    piecewise_gate_basis = _build_piecewise_gate_basis(
        metadata=metadata,
        local_basis_by_regime={},
    )
    truth_contract_recovery = _build_truth_contract_recovery(
        metadata=metadata,
        basis_structure=basis_structure,
        term_contributions=contributions,
    )
    orthogonal_search_objective = _build_orthogonal_search_objective(metadata)
    return SymbolicArtifactSchema(
        payload={
            "family": "symbolic",
            "artifact_type": str(artifact_type),
            "artifact_id": str(artifact_id),
            "final_expression": {str(k): str(v) for k, v in dict(final_expression).items()},
            "normalized_expression": {str(k): str(v) for k, v in dict(normalized_expression).items()},
            "feature_usage": _build_feature_usage(contributions),
            "term_contributions": contributions,
            "complexity_metrics": complexity,
            "stability_metrics": _build_stability_metrics(metadata=metadata, residual_std=residual_std),
            "candidate_lineage": _build_candidate_lineage(metadata),
            "simplification_trace": _build_simplification_trace(metadata),
            "truth_contract_recovery": truth_contract_recovery,
            "orthogonal_search_objective": orthogonal_search_objective,
            "heterogeneous_lane_consensus": _build_heterogeneous_lane_consensus(metadata),
            "equivalence_expression_handling": _build_equivalence_expression_handling(metadata),
            "interference_feature_handling": _build_interference_feature_handling(metadata),
            "periodic_equivalence_disambiguation": _build_periodic_equivalence_disambiguation(metadata),
            "regional_correction_basis": _build_regional_correction_basis(metadata),
            "head_semantics": _resolve_head_semantics(
                metadata=metadata,
                default_task="point",
                default_outputs=("mean",),
                default_objective_family="regression",
                default_calibration_mode="none",
                extra={
                    "target_names": [
                        str(v)
                        for v in _target_labels(target_names, np.asarray(readout_bias, dtype=float).reshape(-1).shape[0])
                    ]
                },
            ),
            "regime_structure": regime_structure,
            "basis_structure": basis_structure,
            "assembler_structure": assembler_structure,
            "piecewise_gate_basis": piecewise_gate_basis,
        }
    ).as_dict()


def build_symbolic_interval_artifact_schema(
    *,
    artifact_type: str,
    artifact_id: str,
    genome_low: Sequence[Mapping[str, Any]],
    parameter_values_low: Mapping[str, float],
    readout_weight_low: np.ndarray,
    readout_bias_low: np.ndarray,
    genome_high: Sequence[Mapping[str, Any]],
    parameter_values_high: Mapping[str, float],
    readout_weight_high: np.ndarray,
    readout_bias_high: np.ndarray,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    residual_std: Sequence[float] | np.ndarray,
    calibration_margin: Sequence[float] | np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
    metadata: Mapping[str, Any],
    final_expression: Mapping[str, Mapping[str, str]],
    normalized_expression: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    low_contributions = _build_target_term_contributions(
        genome=genome_low,
        parameter_values=parameter_values_low,
        readout_weight=readout_weight_low,
        feature_names=feature_names,
        target_names=target_names,
    )
    high_contributions = _build_target_term_contributions(
        genome=genome_high,
        parameter_values=parameter_values_high,
        readout_weight=readout_weight_high,
        feature_names=feature_names,
        target_names=target_names,
    )
    combined_contributions = {
        str(target): {
            "low": list(low_contributions.get(str(target), [])),
            "high": list(high_contributions.get(str(target), [])),
        }
        for target in sorted(
            set(tuple(low_contributions.keys())) | set(tuple(high_contributions.keys()))
        )
    }
    low_complexity = _build_complexity_metrics(genome_low)
    high_complexity = _build_complexity_metrics(genome_high)
    margin = np.asarray(calibration_margin, dtype=float).reshape(-1)
    global_basis = {
        "low": _basis_rows_from_genome(
            genome=genome_low,
            parameter_values=parameter_values_low,
            feature_names=feature_names,
        ),
        "high": _basis_rows_from_genome(
            genome=genome_high,
            parameter_values=parameter_values_high,
            feature_names=feature_names,
        ),
    }
    gate_piecewise, aggregate_manifest = _gate_piecewise_blocks(metadata)
    piecewise_enabled = bool(gate_piecewise or aggregate_manifest)
    regime_structure = _build_regime_structure(
        metadata=metadata,
        piecewise_enabled=piecewise_enabled,
    )
    basis_structure = _build_basis_structure(
        metadata=metadata,
        global_basis=global_basis,
        basis_scope="global+local" if piecewise_enabled else "global",
        source="artifact_interval_basis",
        basis_semantics_extra={
            "bound_targets": ("low", "high"),
            "target_names": tuple(sorted(str(v) for v in dict(final_expression).keys())),
        },
    )
    assembler_structure = _build_assembler_structure(
        metadata=metadata,
        final_expression=final_expression,
        piecewise_enabled=piecewise_enabled,
        composition_targets=tuple(sorted(_expression_leaf_paths(final_expression))),
        source="artifact_interval_expression",
    )
    piecewise_gate_basis = _build_piecewise_gate_basis(
        metadata=metadata,
        local_basis_by_regime={},
    )
    truth_contract_recovery = _build_truth_contract_recovery(
        metadata=metadata,
        basis_structure=basis_structure,
        term_contributions=combined_contributions,
    )
    orthogonal_search_objective = _build_orthogonal_search_objective(metadata)
    return SymbolicArtifactSchema(
        payload={
            "family": "symbolic",
            "artifact_type": str(artifact_type),
            "artifact_id": str(artifact_id),
            "final_expression": {
                str(target): {str(bound): str(expr) for bound, expr in dict(bounds).items()}
                for target, bounds in dict(final_expression).items()
            },
            "normalized_expression": {
                str(target): {str(bound): str(expr) for bound, expr in dict(bounds).items()}
                for target, bounds in dict(normalized_expression).items()
            },
            "feature_usage": _build_feature_usage(combined_contributions),
            "term_contributions": combined_contributions,
            "complexity_metrics": {
                **_merge_complexity_metrics({"low": low_complexity, "high": high_complexity}),
                "by_bound": {
                    "low": low_complexity,
                    "high": high_complexity,
                },
            },
            "stability_metrics": _build_stability_metrics(
                metadata=metadata,
                residual_std=residual_std,
                extra={
                    "calibration_margin_mean": float(np.mean(margin)) if margin.size > 0 else 0.0,
                    "calibration_margin_max": float(np.max(margin)) if margin.size > 0 else 0.0,
                },
            ),
            "candidate_lineage": _build_candidate_lineage(metadata),
            "simplification_trace": _build_simplification_trace(metadata),
            "truth_contract_recovery": truth_contract_recovery,
            "orthogonal_search_objective": orthogonal_search_objective,
            "heterogeneous_lane_consensus": _build_heterogeneous_lane_consensus(metadata),
            "equivalence_expression_handling": _build_equivalence_expression_handling(metadata),
            "interference_feature_handling": _build_interference_feature_handling(metadata),
            "periodic_equivalence_disambiguation": _build_periodic_equivalence_disambiguation(metadata),
            "regional_correction_basis": _build_regional_correction_basis(metadata),
            "head_semantics": _resolve_head_semantics(
                metadata=metadata,
                default_task="interval",
                default_outputs=("lower", "upper"),
                default_objective_family="quantile_interval",
                default_calibration_mode="none",
                extra={
                    "lower_quantile": float(lower_quantile),
                    "upper_quantile": float(upper_quantile),
                    "calibration_margin": _as_float_list(margin),
                    "prediction_semantics": "prediction_interval",
                    "supports_center_view": True,
                },
            ),
            "regime_structure": regime_structure,
            "basis_structure": basis_structure,
            "assembler_structure": assembler_structure,
            "piecewise_gate_basis": piecewise_gate_basis,
        }
    ).as_dict()


def build_piecewise_symbolic_interval_artifact_schema(
    *,
    artifact_type: str,
    artifact_id: str,
    feature_names: Sequence[str],
    gate_feature_names: Sequence[str],
    blend_kappa: float,
    regime_counts: Mapping[str, int],
    metadata: Mapping[str, Any],
    global_schema: Mapping[str, Any],
    local_schemas: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    global_complexity = dict(dict(global_schema).get("complexity_metrics", {}))
    local_complexity = {
        str(key): dict(dict(schema).get("complexity_metrics", {}))
        for key, schema in dict(local_schemas).items()
    }
    aggregate_complexity = _merge_complexity_metrics(
        {
            "global": global_complexity,
            **{f"local:{key}": value for key, value in local_complexity.items()},
        }
    )

    feature_usage_union = _build_feature_usage(
        {
            "global": dict(dict(global_schema).get("term_contributions", {})),
            **{
                f"local:{key}": dict(dict(schema).get("term_contributions", {}))
                for key, schema in dict(local_schemas).items()
            },
        }
    )

    stability_by_scope = {
        "global": dict(dict(global_schema).get("stability_metrics", {})),
        **{
            str(key): dict(dict(schema).get("stability_metrics", {}))
            for key, schema in dict(local_schemas).items()
        },
    }
    local_basis_by_regime = {
        str(key): dict(dict(schema).get("basis_structure", {})).get("global_basis", {})
        for key, schema in sorted(dict(local_schemas).items(), key=lambda item: item[0])
    }
    piecewise_gate_basis = _build_piecewise_gate_basis(
        metadata=metadata,
        gate_feature_names=gate_feature_names,
        blend_kappa=float(blend_kappa),
        selected_regime_keys=tuple(sorted(str(key) for key in dict(local_schemas).keys())),
        counts_selected=regime_counts,
        local_basis_by_regime=local_basis_by_regime,
    )
    regime_structure = _build_regime_structure(
        metadata=metadata,
        piecewise_enabled=True,
        gate_feature_names=gate_feature_names,
        selected_regime_keys=tuple(sorted(str(key) for key in dict(local_schemas).keys())),
        local_regimes=_mapping_or_empty(_mapping_or_empty(metadata.get("aggregate_manifest")).get("local_regimes")),
        failed_regimes=_mapping_or_empty(_mapping_or_empty(metadata.get("aggregate_manifest")).get("failed_regimes")),
        counts_selected=regime_counts,
        blend_kappa=float(blend_kappa),
    )
    basis_structure = _build_basis_structure(
        metadata=metadata,
        global_basis=dict(dict(global_schema).get("basis_structure", {})).get("global_basis", {}),
        local_basis_by_regime=local_basis_by_regime,
        gate_basis=dict(piecewise_gate_basis.get("gate_basis", {})),
        basis_scope="global+local",
        source="piecewise_aggregate",
        basis_semantics_extra={
            "gate_conditioned": True,
            "selected_regime_keys": tuple(sorted(str(key) for key in dict(local_schemas).keys())),
        },
    )
    assembler_structure = _build_assembler_structure(
        metadata=metadata,
        final_expression={
            "global": dict(dict(global_schema).get("final_expression", {})),
            "local_by_regime": {
                str(key): dict(dict(schema).get("final_expression", {}))
                for key, schema in sorted(dict(local_schemas).items(), key=lambda item: item[0])
            },
        },
        piecewise_enabled=True,
        composition_targets=tuple(
            sorted(
                _expression_leaf_paths(dict(dict(global_schema).get("final_expression", {})))
                + [
                    f"{key}.{path}"
                    for key, schema in sorted(dict(local_schemas).items(), key=lambda item: item[0])
                    for path in _expression_leaf_paths(dict(dict(schema).get("final_expression", {})))
                ]
            )
        ),
        source="piecewise_aggregate",
    )
    truth_contract_recovery = _build_truth_contract_recovery(
        metadata=metadata,
        basis_structure=basis_structure,
        term_contributions={
            "global": dict(dict(global_schema).get("term_contributions", {})),
            "local_by_regime": {
                str(key): dict(dict(schema).get("term_contributions", {}))
                for key, schema in sorted(dict(local_schemas).items(), key=lambda item: item[0])
            },
        },
    )
    orthogonal_search_objective = _build_orthogonal_search_objective(metadata)

    return SymbolicArtifactSchema(
        payload={
            "family": "symbolic",
            "artifact_type": str(artifact_type),
            "artifact_id": str(artifact_id),
            "final_expression": {
                "global": dict(dict(global_schema).get("final_expression", {})),
                "local_by_regime": {
                    str(key): dict(dict(schema).get("final_expression", {}))
                    for key, schema in sorted(dict(local_schemas).items(), key=lambda item: item[0])
                },
            },
            "normalized_expression": {
                "global": dict(dict(global_schema).get("normalized_expression", {})),
                "local_by_regime": {
                    str(key): dict(dict(schema).get("normalized_expression", {}))
                    for key, schema in sorted(dict(local_schemas).items(), key=lambda item: item[0])
                },
            },
            "feature_usage": {
                **feature_usage_union,
                "global": dict(dict(global_schema).get("feature_usage", {})),
                "local_by_regime": {
                    str(key): dict(dict(schema).get("feature_usage", {}))
                    for key, schema in sorted(dict(local_schemas).items(), key=lambda item: item[0])
                },
            },
            "term_contributions": {
                "global": dict(dict(global_schema).get("term_contributions", {})),
                "local_by_regime": {
                    str(key): dict(dict(schema).get("term_contributions", {}))
                    for key, schema in sorted(dict(local_schemas).items(), key=lambda item: item[0])
                },
            },
            "complexity_metrics": {
                **aggregate_complexity,
                "global": global_complexity,
                "local_by_regime": local_complexity,
                "selected_regime_count": int(len(dict(local_schemas))),
            },
            "stability_metrics": {
                "global": dict(dict(global_schema).get("stability_metrics", {})),
                "local_by_regime": {
                    str(key): dict(value)
                    for key, value in sorted(stability_by_scope.items(), key=lambda item: item[0])
                    if key != "global"
                },
                "regime_counts": {str(k): int(v) for k, v in sorted(dict(regime_counts).items(), key=lambda item: item[0])},
            },
            "candidate_lineage": {
                "source": "piecewise_aggregate",
                "aggregate_manifest": dict(metadata.get("aggregate_manifest", {}))
                if isinstance(metadata.get("aggregate_manifest"), Mapping)
                else {},
                "gate_piecewise": dict(metadata.get("gate_piecewise", {}))
                if isinstance(metadata.get("gate_piecewise"), Mapping)
                else {},
            },
            "simplification_trace": _build_simplification_trace(metadata),
            "truth_contract_recovery": truth_contract_recovery,
            "orthogonal_search_objective": orthogonal_search_objective,
            "head_semantics": {
                **dict(dict(global_schema).get("head_semantics", {})),
                "prediction_semantics": "piecewise_prediction_interval",
                "piecewise_enabled": True,
                "gate_feature_names": [str(v) for v in tuple(gate_feature_names)],
                "feature_names": [str(v) for v in tuple(feature_names)],
                "blend_kappa": float(blend_kappa),
                "selected_regime_count": int(len(dict(local_schemas))),
            },
            "regime_structure": regime_structure,
            "basis_structure": basis_structure,
            "assembler_structure": assembler_structure,
            "piecewise_gate_basis": piecewise_gate_basis,
        }
    ).as_dict()


__all__ = [
    "SYMBOLIC_ARTIFACT_FIELDS",
    "SYMBOLIC_ARTIFACT_SCHEMA_KEY",
    "SYMBOLIC_ARTIFACT_SCHEMA_NAME",
    "SYMBOLIC_ARTIFACT_SCHEMA_VERSION",
    "SYMBOLIC_COMPLEXITY_FIELDS",
    "SYMBOLIC_EXPLAINABILITY_FIELDS",
    "SYMBOLIC_REGIME_FIELDS",
    "SYMBOLIC_BASIS_FIELDS",
    "SYMBOLIC_ASSEMBLER_FIELDS",
    "SYMBOLIC_PIECEWISE_GATE_FIELDS",
    "SYMBOLIC_STABILITY_FIELDS",
    "SYMBOLIC_HETEROGENEOUS_LANE_FIELDS",
    "SymbolicArtifactSchema",
    "build_symbolic_structure_surface_payload",
    "build_piecewise_symbolic_interval_artifact_schema",
    "build_symbolic_interval_artifact_schema",
    "build_symbolic_point_artifact_schema",
    "merge_symbolic_artifact_schema_descriptors",
    "symbolic_artifact_schema_descriptor",
]
