from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.symbolic.symbolic_dsl import expression_to_string


def _expr_feature_indices(expr: Mapping[str, Any]) -> set[int]:
    kind = str(expr.get("type", "")).strip().lower()
    if kind == "feature":
        try:
            return {int(expr.get("index", -1))}
        except Exception:
            return set()
    if kind == "unary":
        return _expr_feature_indices(dict(expr.get("arg", {})))
    if kind == "binary":
        return _expr_feature_indices(dict(expr.get("left", {}))) | _expr_feature_indices(dict(expr.get("right", {})))
    return set()


def _feature_label(index: int, feature_names: Sequence[str]) -> str:
    idx = int(index)
    names = tuple(str(value) for value in tuple(feature_names))
    if 0 <= idx < len(names):
        return str(names[idx])
    return f"x{idx}"


def _expr_semantic_signature(expr: Mapping[str, Any]) -> str:
    kind = str(expr.get("type", "")).strip().lower()
    if kind == "feature":
        try:
            return f"feature:{int(expr.get('index', -1))}"
        except Exception:
            return "feature:unknown"
    if kind == "const":
        return "const"
    if kind == "param":
        name = str(expr.get("name", "")).strip()
        return f"param:{name or 'anon'}"
    if kind == "unary":
        op = str(expr.get("op", "")).strip().lower()
        return f"unary:{op}({_expr_semantic_signature(dict(expr.get('arg', {})))})"
    if kind == "binary":
        op = str(expr.get("op", "")).strip().lower()
        left = _expr_semantic_signature(dict(expr.get("left", {})))
        right = _expr_semantic_signature(dict(expr.get("right", {})))
        if op in {"add", "mul"}:
            left, right = sorted((left, right))
        return f"binary:{op}({left},{right})"
    return kind or "unknown"


def _expr_semantic_family(expr: Mapping[str, Any], *, term_name: str = "") -> str:
    name = str(term_name or "").strip().lower()
    if any(token in name for token in ("piecewise", "hinge", "gate_", "step_", "soft_gate", "gate_step")):
        return "piecewise_gate"
    kind = str(expr.get("type", "")).strip().lower()
    feature_count = int(len(_expr_feature_indices(expr)))
    if kind == "feature":
        return "linear_feature"
    if kind == "unary":
        op = str(expr.get("op", "")).strip().lower()
        if feature_count <= 1 and op in {"sin", "cos"}:
            return "single_feature_periodic"
        if feature_count <= 1 and op in {"square", "abs", "sqrt", "log", "tanh", "exp"}:
            return "single_feature_transform"
        return f"unary:{op or 'unknown'}"
    if kind == "binary":
        op = str(expr.get("op", "")).strip().lower()
        if op == "mul" and feature_count >= 2:
            return "pair_interaction"
        if op == "div":
            return "ratio_or_reciprocal"
        if op in {"add", "sub"}:
            return "additive_combo"
        return f"binary:{op or 'unknown'}"
    return "composite"


def _expr_uses_piecewise_gate(expr: Mapping[str, Any], *, term_name: str = "") -> bool:
    family = _expr_semantic_family(expr, term_name=term_name)
    if family == "piecewise_gate":
        return True
    name = str(term_name or "").strip().lower()
    return any(token in name for token in ("gate", "hinge", "piecewise", "regime"))


def build_basis_term_rows(
    genome: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
    parameter_values: Mapping[str, float] | None = None,
    expression_strings: Sequence[str] | None = None,
    scope: str = "global",
    precision: int = 12,
) -> list[dict[str, Any]]:
    params = dict(parameter_values or {})
    expressions = tuple(str(value) for value in tuple(expression_strings or tuple()) if str(value).strip())
    rows: list[dict[str, Any]] = []
    for term_index, term in enumerate(tuple(genome)):
        expr = dict(term.get("expr", {}))
        if term_index < len(expressions):
            expr_text = str(expressions[term_index])
        else:
            expr_text = expression_to_string(expr, param_values=params, precision=int(precision))
        feature_indices = sorted(idx for idx in _expr_feature_indices(expr) if idx >= 0)
        feature_labels = [_feature_label(idx, feature_names) for idx in feature_indices]
        rows.append(
            {
                "term_index": int(term_index),
                "scope": str(scope),
                "term_name": str(term.get("name", f"term_{term_index}")),
                "expression": str(expr_text),
                "feature_indices": [int(value) for value in feature_indices],
                "feature_names": [str(value) for value in feature_labels],
                "feature_count": int(len(feature_labels)),
                "semantic_signature": _expr_semantic_signature(expr),
                "semantic_family": _expr_semantic_family(expr, term_name=str(term.get("name", f"term_{term_index}"))),
                "uses_piecewise_gate": bool(
                    _expr_uses_piecewise_gate(expr, term_name=str(term.get("name", f"term_{term_index}")))
                ),
            }
        )
    return rows


def build_basis_semantics_payload(
    basis_rows: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    source: str,
    basis_scope: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(basis_rows, Mapping):
        groups = {
            str(key): [dict(row) for row in tuple(value) if isinstance(row, Mapping)]
            for key, value in dict(basis_rows).items()
        }
        flat_rows = [dict(row) for value in groups.values() for row in value]
        payload: dict[str, Any] = {
            "source": str(source),
            "basis_scope": str(basis_scope),
            "basis_count": int(len(flat_rows)),
            "basis_feature_union": sorted(
                {
                    str(name)
                    for row in flat_rows
                    for name in tuple(dict(row).get("feature_names", ()))
                    if str(name).strip()
                }
            ),
            "basis_groups": groups,
        }
    else:
        flat_rows = [dict(row) for row in tuple(basis_rows) if isinstance(row, Mapping)]
        payload = {
            "source": str(source),
            "basis_scope": str(basis_scope),
            "basis_count": int(len(flat_rows)),
            "basis_feature_union": sorted(
                {
                    str(name)
                    for row in flat_rows
                    for name in tuple(dict(row).get("feature_names", ()))
                    if str(name).strip()
                }
            ),
            "basis_terms": flat_rows,
        }
    if extra:
        payload.update({str(key): value for key, value in dict(extra).items()})
    return payload


def build_basis_overlap_report(
    basis_rows: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    source: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(basis_rows, Mapping):
        flat_rows = [dict(row) for value in dict(basis_rows).values() for row in tuple(value) if isinstance(row, Mapping)]
    else:
        flat_rows = [dict(row) for row in tuple(basis_rows) if isinstance(row, Mapping)]
    pair_count = 0
    overlap_pair_count = 0
    jaccards: list[float] = []
    feature_term_counts: dict[str, int] = {}
    expression_counts: dict[str, int] = {}
    for row in flat_rows:
        expression = str(row.get("expression", "")).strip()
        if expression:
            expression_counts[expression] = int(expression_counts.get(expression, 0) + 1)
        for feature_name in tuple(row.get("feature_names", ())):
            name = str(feature_name).strip()
            if name:
                feature_term_counts[name] = int(feature_term_counts.get(name, 0) + 1)
    for index, left in enumerate(flat_rows):
        left_features = {str(value) for value in tuple(dict(left).get("feature_names", ())) if str(value).strip()}
        for right in flat_rows[index + 1 :]:
            right_features = {str(value) for value in tuple(dict(right).get("feature_names", ())) if str(value).strip()}
            pair_count += 1
            union = left_features | right_features
            overlap = left_features & right_features
            if overlap:
                overlap_pair_count += 1
            if union:
                jaccards.append(float(len(overlap)) / float(len(union)))
    payload = {
        "source": str(source),
        "basis_count": int(len(flat_rows)),
        "pair_count": int(pair_count),
        "overlap_pair_count": int(overlap_pair_count),
        "mean_feature_jaccard": float(sum(jaccards) / len(jaccards)) if jaccards else 0.0,
        "max_feature_jaccard": float(max(jaccards)) if jaccards else 0.0,
        "duplicate_expression_count": int(sum(1 for count in expression_counts.values() if int(count) > 1)),
        "feature_term_counts": {str(key): int(value) for key, value in sorted(feature_term_counts.items(), key=lambda item: item[0])},
    }
    if extra:
        payload.update({str(key): value for key, value in dict(extra).items()})
    return payload


def build_semantic_dedup_report(
    basis_rows: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    source: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(basis_rows, Mapping):
        flat_rows = [dict(row) for value in dict(basis_rows).values() for row in tuple(value) if isinstance(row, Mapping)]
    else:
        flat_rows = [dict(row) for row in tuple(basis_rows) if isinstance(row, Mapping)]
    signature_groups: dict[str, list[str]] = {}
    family_counts: dict[str, int] = {}
    piecewise_gate_term_count = 0
    for row in flat_rows:
        signature = str(row.get("semantic_signature", "")).strip() or str(row.get("expression", "")).strip()
        term_name = str(row.get("term_name", "")).strip() or str(row.get("expression", "")).strip() or "<unnamed>"
        family = str(row.get("semantic_family", "")).strip() or "unknown"
        signature_groups.setdefault(signature, []).append(term_name)
        family_counts[family] = int(family_counts.get(family, 0) + 1)
        if bool(row.get("uses_piecewise_gate")):
            piecewise_gate_term_count += 1
    basis_count = int(len(flat_rows))
    unique_signature_count = int(sum(1 for key in signature_groups.keys() if str(key).strip()))
    duplicate_signature_count = int(sum(1 for values in signature_groups.values() if len(values) > 1))
    semantic_groups = [
        {"semantic_signature": str(key), "terms": list(values), "count": int(len(values))}
        for key, values in sorted(signature_groups.items(), key=lambda item: (-len(item[1]), str(item[0])))
        if len(values) > 1
    ]
    payload = {
        "source": str(source),
        "basis_count": basis_count,
        "semantic_unique_ratio": float(unique_signature_count / basis_count) if basis_count > 0 else 0.0,
        "duplicate_semantic_signature_count": duplicate_signature_count,
        "semantic_family_counts": {str(key): int(value) for key, value in sorted(family_counts.items(), key=lambda item: item[0])},
        "piecewise_gate_term_count": int(piecewise_gate_term_count),
        "semantic_groups": semantic_groups,
        "status": "reported" if basis_count > 0 else "not_recorded",
    }
    if extra:
        payload.update({str(key): value for key, value in dict(extra).items()})
    return payload


def build_residual_complementarity_report(
    steps: Sequence[Mapping[str, Any]],
    *,
    source: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_steps: list[dict[str, Any]] = []
    marginal_r2_gains: list[float] = []
    residual_abs_corrs: list[float] = []
    residual_ratios: list[float] = []
    for raw in tuple(steps):
        if not isinstance(raw, Mapping):
            continue
        row = {
            "term_name": str(raw.get("term_name", "")),
            "semantic_family": str(raw.get("semantic_family", "")),
            "marginal_target_abs_corr": None if raw.get("marginal_target_abs_corr") is None else float(raw.get("marginal_target_abs_corr")),
            "marginal_residual_abs_corr": None if raw.get("marginal_residual_abs_corr") is None else float(raw.get("marginal_residual_abs_corr")),
            "marginal_r2_gain": None if raw.get("marginal_r2_gain") is None else float(raw.get("marginal_r2_gain")),
            "residual_norm_before": None if raw.get("residual_norm_before") is None else float(raw.get("residual_norm_before")),
            "residual_norm_after": None if raw.get("residual_norm_after") is None else float(raw.get("residual_norm_after")),
            "residual_ratio_after": None if raw.get("residual_ratio_after") is None else float(raw.get("residual_ratio_after")),
        }
        if row["marginal_r2_gain"] is not None:
            marginal_r2_gains.append(float(row["marginal_r2_gain"]))
        if row["marginal_residual_abs_corr"] is not None:
            residual_abs_corrs.append(float(row["marginal_residual_abs_corr"]))
        if row["residual_ratio_after"] is not None:
            residual_ratios.append(float(row["residual_ratio_after"]))
        normalized_steps.append(row)
    monotone_nonnegative_gain = bool(all(float(v) >= -1e-12 for v in marginal_r2_gains)) if marginal_r2_gains else False
    payload = {
        "source": str(source),
        "status": "reported" if normalized_steps else "not_recorded",
        "step_count": int(len(normalized_steps)),
        "mean_marginal_r2_gain": float(sum(marginal_r2_gains) / len(marginal_r2_gains)) if marginal_r2_gains else 0.0,
        "min_marginal_r2_gain": float(min(marginal_r2_gains)) if marginal_r2_gains else 0.0,
        "mean_marginal_residual_abs_corr": float(sum(residual_abs_corrs) / len(residual_abs_corrs)) if residual_abs_corrs else 0.0,
        "final_residual_ratio": float(residual_ratios[-1]) if residual_ratios else None,
        "monotone_nonnegative_gain": monotone_nonnegative_gain,
        "steps": normalized_steps,
    }
    if extra:
        payload.update({str(key): value for key, value in dict(extra).items()})
    return payload


def build_assembler_budget_payload(
    *,
    source: str,
    assembler_mode: str,
    output_expression_count: int,
    selected_basis_count: int,
    budget_axes: Mapping[str, Any] | None = None,
    budget_scale: str = "small",
    uses_piecewise_gate: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    recorded_values = {
        str(key): value
        for key, value in dict(budget_axes or {}).items()
        if value is not None
    }
    payload = {
        "source": str(source),
        "assembler_mode": str(assembler_mode),
        "budget_scale": str(budget_scale),
        "budget_axes": [str(key) for key in recorded_values.keys()],
        "recorded_values": recorded_values,
        "output_expression_count": int(max(0, int(output_expression_count))),
        "selected_basis_count": int(max(0, int(selected_basis_count))),
        "uses_piecewise_gate": bool(uses_piecewise_gate),
    }
    if extra:
        payload.update({str(key): value for key, value in dict(extra).items()})
    return payload


__all__ = [
    "build_assembler_budget_payload",
    "build_basis_overlap_report",
    "build_basis_semantics_payload",
    "build_basis_term_rows",
    "build_residual_complementarity_report",
    "build_semantic_dedup_report",
]
