from __future__ import annotations

import json
import math
from typing import Any, Mapping

import numpy as np

from mlblack.models.symbolic import expression_to_string


TraceRows = list[Mapping[str, Any]]


def simplify_expression(
    expr: Mapping[str, Any],
    *,
    trace: TraceRows | None = None,
    root: str = "$",
) -> dict[str, Any]:
    """Simplify and deterministically normalize a symbolic expression.

    This is intentionally an engine-level primitive. Audit, cache, guard and
    search policy code should all call this instead of keeping local string
    heuristics.
    """

    return canonicalize_expression(expr, trace=trace, root=root)


def canonicalize_expression(
    expr: Mapping[str, Any],
    *,
    trace: TraceRows | None = None,
    root: str = "$",
) -> dict[str, Any]:
    rows = trace if trace is not None else []
    return _canonicalize(dict(expr), trace=rows, root=str(root))


def expression_canonical_payload(expr: Mapping[str, Any]) -> dict[str, Any]:
    canonical = canonicalize_expression(expr)
    return _payload(canonical)


def expression_equivalence_key(expr: Mapping[str, Any]) -> str:
    payload = expression_canonical_payload(expr)
    body = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return f"symbolic_expr_v1:{body}"


def expression_canonical_string(
    expr: Mapping[str, Any],
    *,
    precision: int = 12,
) -> str:
    return expression_to_string(canonicalize_expression(expr), precision=int(precision))


def expression_family_signature(
    expr: Mapping[str, Any],
    *,
    feature_names: tuple[str, ...] | list[str] = tuple(),
) -> dict[str, Any]:
    """Return a stable family-level signature for recovery scoring.

    Unlike ``expression_equivalence_key`` this intentionally groups broader
    families such as sin/cos phase variants, log/exp chains and ratios. It is
    report/scoring metadata, not a cache key for exact value reuse.
    """

    canonical = canonicalize_expression(expr)
    features = tuple(sorted(_collect_feature_indices(canonical)))
    names = tuple(str(v) for v in feature_names)
    labels = tuple(str(names[idx]) if 0 <= idx < len(names) else f"x{idx}" for idx in features)
    family, subfamily = _expression_family(canonical)
    phase_key = _phase_equivalence_key(canonical)
    ratio = _ratio_signature(canonical)
    return {
        "canonical_key": expression_equivalence_key(canonical),
        "canonical_expression": expression_to_string(canonical, precision=12, feature_names=names),
        "family": family,
        "subfamily": subfamily,
        "features": list(labels),
        "feature_indices": list(features),
        "phase_equivalence_key": phase_key,
        "is_periodic": bool(phase_key),
        "ratio_signature": ratio,
    }


def _canonicalize(node: Mapping[str, Any], *, trace: TraceRows, root: str) -> dict[str, Any]:
    kind = str(node.get("type", "")).strip().lower()
    if kind == "feature":
        return {"type": "feature", "index": int(node.get("index", 0))}
    if kind == "const":
        return {"type": "const", "value": _clean_number(float(node.get("value", 0.0)))}
    if kind == "param":
        return _canonical_param(node)
    if kind == "unary":
        op = str(node.get("op", "")).strip().lower()
        arg = _canonicalize(dict(node.get("arg", {}) or {}), trace=trace, root=f"{root}.arg")
        before = {"type": "unary", "op": op, "arg": arg}
        if op == "identity":
            trace.append(_trace(root, "identity_unwrap", before, arg))
            return arg
        negated = _negated_arg(arg)
        if negated is not None and op in {"cos", "square"}:
            out = _canonicalize({"type": "unary", "op": op, "arg": negated}, trace=trace, root=root)
            trace.append(_trace(root, f"even_unary_sign_{op}", before, out))
            return out
        if negated is not None and op in {"sin", "tanh"}:
            out = _canonicalize(
                {"type": "binary", "op": "mul", "left": {"type": "const", "value": -1.0}, "right": {"type": "unary", "op": op, "arg": negated}},
                trace=trace,
                root=root,
            )
            trace.append(_trace(root, f"odd_unary_sign_{op}", before, out))
            return out
        if op == "abs" and str(arg.get("type")) == "unary" and str(arg.get("op")) == "abs":
            out = dict(arg.get("arg", {}) or {})
            trace.append(_trace(root, "abs_idempotent", before, {"type": "unary", "op": "abs", "arg": out}))
            return {"type": "unary", "op": "abs", "arg": out}
        if op == "square" and str(arg.get("type")) == "unary" and str(arg.get("op")) == "abs":
            out = _canonicalize({"type": "unary", "op": "square", "arg": dict(arg.get("arg", {}) or {})}, trace=trace, root=root)
            trace.append(_trace(root, "square_abs", before, out))
            return out
        const = _const_value(arg)
        if const is not None:
            folded = {"type": "const", "value": _clean_number(_eval_unary_const(op, const))}
            trace.append(_trace(root, "const_fold_unary", before, folded))
            return folded
        return before
    if kind == "binary":
        op = str(node.get("op", "")).strip().lower()
        left = _canonicalize(dict(node.get("left", {}) or {}), trace=trace, root=f"{root}.left")
        right = _canonicalize(dict(node.get("right", {}) or {}), trace=trace, root=f"{root}.right")
        before = {"type": "binary", "op": op, "left": left, "right": right}
        folded = _const_fold_binary(op, left, right)
        if folded is not None:
            trace.append(_trace(root, "const_fold_binary", before, folded))
            return folded
        algebra = _algebraic_binary_simplify(op, left, right)
        if algebra is not None:
            trace.append(_trace(root, f"algebra_{op}", before, algebra))
            return algebra
        if op == "div":
            fraction = _constant_denominator_to_mul(left, right)
            if fraction is not None:
                out = _canonicalize(fraction, trace=trace, root=root)
                trace.append(_trace(root, "fraction_const_denominator", before, out))
                return out
        if op in {"add", "mul"}:
            return _canonicalize_associative_binary(op, left, right, before=before, trace=trace, root=root)
        return before
    return dict(node)


def _canonical_param(node: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": "param",
        "name": str(node.get("name", "")),
        "init": _clean_number(float(node.get("init", 1.0))),
        "trainable": bool(node.get("trainable", True)),
    }
    if node.get("lower") is not None:
        out["lower"] = _clean_number(float(node["lower"]))
    if node.get("upper") is not None:
        out["upper"] = _clean_number(float(node["upper"]))
    return out


def _canonicalize_associative_binary(
    op: str,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    before: Mapping[str, Any],
    trace: TraceRows,
    root: str,
) -> dict[str, Any]:
    terms = _flatten_op(op, left) + _flatten_op(op, right)
    if op == "add":
        const_sum = 0.0
        non_const: list[dict[str, Any]] = []
        for term in terms:
            value = _const_value(term)
            if value is None:
                non_const.append(dict(term))
            else:
                const_sum += float(value)
        non_const, like_const = _combine_additive_like_terms(non_const)
        const_sum += float(like_const)
        non_const, trig_added_const = _reduce_trig_pythagorean_terms(non_const)
        const_sum += float(trig_added_const)
        if abs(const_sum) > 1e-12:
            non_const.append({"type": "const", "value": _clean_number(const_sum)})
        if not non_const:
            out = {"type": "const", "value": 0.0}
        else:
            out = _build_sorted_binary("add", non_const)
    else:
        const_product = 1.0
        non_const = []
        for term in terms:
            value = _const_value(term)
            if value is None:
                non_const.append(dict(term))
            else:
                const_product *= float(value)
        if abs(const_product) <= 1e-12:
            out = {"type": "const", "value": 0.0}
        else:
            if abs(const_product - 1.0) > 1e-12:
                non_const.append({"type": "const", "value": _clean_number(const_product)})
            out = _build_sorted_mul(non_const) if non_const else {"type": "const", "value": _clean_number(const_product)}
    if expression_to_string(before, precision=12) != expression_to_string(out, precision=12):
        trace.append(_trace(root, f"canonical_associative_{op}", before, out))
    return out


def _flatten_op(op: str, expr: Mapping[str, Any]) -> list[dict[str, Any]]:
    if str(expr.get("type")) == "binary" and str(expr.get("op")) == str(op):
        return _flatten_op(op, dict(expr.get("left", {}) or {})) + _flatten_op(op, dict(expr.get("right", {}) or {}))
    return [dict(expr)]


def _build_sorted_binary(op: str, terms: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted((dict(term) for term in terms), key=expression_equivalence_key)
    if not ordered:
        return {"type": "const", "value": 0.0 if op == "add" else 1.0}
    out = ordered[0]
    for term in ordered[1:]:
        out = {"type": "binary", "op": str(op), "left": out, "right": term}
    return out


def _build_sorted_mul(terms: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted((dict(term) for term in terms), key=expression_equivalence_key)
    out_terms: list[dict[str, Any]] = []
    i = 0
    while i < len(ordered):
        current = ordered[i]
        if i + 1 < len(ordered) and expression_equivalence_key(current) == expression_equivalence_key(ordered[i + 1]):
            out_terms.append({"type": "unary", "op": "square", "arg": current})
            i += 2
        else:
            out_terms.append(current)
            i += 1
    return _build_sorted_binary("mul", out_terms)


def _reduce_trig_pythagorean_terms(terms: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    used: set[int] = set()
    added_const = 0.0
    out: list[dict[str, Any]] = []
    sin_terms: dict[str, list[int]] = {}
    cos_terms: dict[str, list[int]] = {}
    for idx, term in enumerate(terms):
        trig = _trig_square_arg(term)
        if trig is None:
            continue
        op, arg_key = trig
        if op == "sin":
            sin_terms.setdefault(arg_key, []).append(idx)
        elif op == "cos":
            cos_terms.setdefault(arg_key, []).append(idx)
    for arg_key, sin_indices in sin_terms.items():
        cos_indices = cos_terms.get(arg_key, [])
        pair_count = min(len(sin_indices), len(cos_indices))
        for pair_idx in range(pair_count):
            used.add(sin_indices[pair_idx])
            used.add(cos_indices[pair_idx])
            added_const += 1.0
    for idx, term in enumerate(terms):
        if idx not in used:
            out.append(dict(term))
    return out, added_const


def _combine_additive_like_terms(terms: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    coeffs: dict[str, float] = {}
    bases: dict[str, dict[str, Any]] = {}
    added_const = 0.0
    for term in terms:
        coeff, factors = _constant_factor(term)
        if not factors:
            added_const += float(coeff)
            continue
        base = _build_sorted_mul(factors) if len(factors) > 1 else dict(factors[0])
        key = expression_equivalence_key(base)
        coeffs[key] = float(coeffs.get(key, 0.0) + coeff)
        bases[key] = base
    out: list[dict[str, Any]] = []
    for key in sorted(coeffs):
        coeff = _clean_number(coeffs[key])
        if abs(coeff) <= 1e-12:
            continue
        base = bases[key]
        if abs(coeff - 1.0) <= 1e-12:
            out.append(base)
        else:
            out.append(
                {
                    "type": "binary",
                    "op": "mul",
                    "left": {"type": "const", "value": coeff},
                    "right": base,
                }
            )
    return out, added_const


def _trig_square_arg(expr: Mapping[str, Any]) -> tuple[str, str] | None:
    if str(expr.get("type")) != "unary" or str(expr.get("op")) != "square":
        return None
    arg = dict(expr.get("arg", {}) or {})
    if str(arg.get("type")) != "unary":
        return None
    op = str(arg.get("op"))
    if op not in {"sin", "cos"}:
        return None
    return op, expression_equivalence_key(dict(arg.get("arg", {}) or {}))


def _const_fold_binary(op: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any] | None:
    lval = _const_value(left)
    rval = _const_value(right)
    if lval is None or rval is None:
        return None
    return {"type": "const", "value": _clean_number(_eval_binary_const(op, lval, rval))}


def _algebraic_binary_simplify(op: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any] | None:
    if op == "add":
        if _is_const(left, 0.0):
            return dict(right)
        if _is_const(right, 0.0):
            return dict(left)
    if op == "sub":
        if expression_equivalence_key(left) == expression_equivalence_key(right):
            return {"type": "const", "value": 0.0}
        if _is_const(right, 0.0):
            return dict(left)
        if _is_const(left, 0.0):
            return {
                "type": "binary",
                "op": "mul",
                "left": {"type": "const", "value": -1.0},
                "right": dict(right),
            }
    if op == "mul":
        if _is_const(left, 0.0) or _is_const(right, 0.0):
            return {"type": "const", "value": 0.0}
        if _is_const(left, 1.0):
            return dict(right)
        if _is_const(right, 1.0):
            return dict(left)
    if op == "div":
        if _is_const(left, 0.0):
            return {"type": "const", "value": 0.0}
        if _is_const(right, 1.0):
            return dict(left)
    return None


def _constant_denominator_to_mul(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any] | None:
    denom = _const_value(right)
    if denom is None or abs(float(denom)) <= 1e-12:
        return None
    return {
        "type": "binary",
        "op": "mul",
        "left": dict(left),
        "right": {"type": "const", "value": _clean_number(1.0 / float(denom))},
    }


def _negated_arg(expr: Mapping[str, Any]) -> dict[str, Any] | None:
    if str(expr.get("type")) != "binary" or str(expr.get("op")) != "mul":
        return None
    terms = _flatten_op("mul", expr)
    const_product = 1.0
    non_const: list[dict[str, Any]] = []
    for term in terms:
        value = _const_value(term)
        if value is None:
            non_const.append(dict(term))
        else:
            const_product *= float(value)
    if abs(const_product + 1.0) > 1e-12 or not non_const:
        return None
    return _build_sorted_mul(non_const)


def _payload(expr: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(expr.get("type", "")).strip().lower()
    if kind == "feature":
        return {"type": "feature", "index": int(expr.get("index", 0))}
    if kind == "const":
        return {"type": "const", "value": _number_token(float(expr.get("value", 0.0)))}
    if kind == "param":
        payload: dict[str, Any] = {
            "type": "param",
            "name": str(expr.get("name", "")),
            "init": _number_token(float(expr.get("init", 1.0))),
            "trainable": bool(expr.get("trainable", True)),
        }
        if expr.get("lower") is not None:
            payload["lower"] = _number_token(float(expr["lower"]))
        if expr.get("upper") is not None:
            payload["upper"] = _number_token(float(expr["upper"]))
        return payload
    if kind == "unary":
        return {"type": "unary", "op": str(expr.get("op", "")), "arg": _payload(dict(expr.get("arg", {}) or {}))}
    if kind == "binary":
        return {
            "type": "binary",
            "op": str(expr.get("op", "")),
            "left": _payload(dict(expr.get("left", {}) or {})),
            "right": _payload(dict(expr.get("right", {}) or {})),
        }
    return {"type": kind, "raw": json.dumps(dict(expr), ensure_ascii=True, sort_keys=True, default=str)}


def _collect_feature_indices(expr: Mapping[str, Any]) -> set[int]:
    kind = str(expr.get("type", "")).strip().lower()
    if kind == "feature":
        return {int(expr.get("index", 0))}
    if kind == "unary":
        return _collect_feature_indices(dict(expr.get("arg", {}) or {}))
    if kind == "binary":
        return _collect_feature_indices(dict(expr.get("left", {}) or {})) | _collect_feature_indices(dict(expr.get("right", {}) or {}))
    return set()


def _expression_family(expr: Mapping[str, Any]) -> tuple[str, str]:
    kind = str(expr.get("type", "")).strip().lower()
    if kind == "feature":
        return "linear_feature", "feature"
    if kind == "const":
        return "constant", "const"
    if kind == "param":
        return "parameter", "param"
    if kind == "unary":
        op = str(expr.get("op", "")).strip().lower()
        arg = dict(expr.get("arg", {}) or {})
        if op in {"sin", "cos"}:
            return "trig", op
        if op == "square":
            arg_family, arg_subfamily = _expression_family(arg)
            if arg_family == "trig":
                return "trig_power", f"{arg_subfamily}_square"
            return "power", "square"
        if op in {"exp", "log"}:
            arg_family, arg_subfamily = _expression_family(arg)
            if (op, arg_family) == ("exp", "logarithmic"):
                return "exp_log_chain", f"exp_{arg_subfamily}"
            if (op, arg_family) == ("log", "exponential"):
                return "log_exp_chain", f"log_{arg_subfamily}"
            return "exponential" if op == "exp" else "logarithmic", op
        if op in {"sqrt", "abs"}:
            return "magnitude", op
        if op == "tanh":
            return "bounded", "tanh"
        return "unary", op
    if kind == "binary":
        op = str(expr.get("op", "")).strip().lower()
        if op == "div":
            return "ratio", "division"
        if op == "mul":
            terms = _flatten_op("mul", expr)
            non_const = [term for term in terms if _const_value(term) is None]
            if len(non_const) == 1:
                return _expression_family(non_const[0])
            return "product", "multiplicative"
        if op in {"add", "sub"}:
            return "additive", op
    return "expression", kind


def _phase_equivalence_key(expr: Mapping[str, Any]) -> str:
    periodic = _periodic_core(expr)
    if periodic is None:
        return ""
    family, arg = periodic
    signature = _affine_feature_signature(arg)
    return f"phase:{family}:{signature}"


def _periodic_core(expr: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]] | None:
    kind = str(expr.get("type", "")).strip().lower()
    if kind == "unary":
        op = str(expr.get("op", "")).strip().lower()
        if op in {"sin", "cos"}:
            return "trig", dict(expr.get("arg", {}) or {})
        if op == "square":
            arg = dict(expr.get("arg", {}) or {})
            if str(arg.get("type")) == "unary" and str(arg.get("op")) in {"sin", "cos"}:
                return "trig_square", dict(arg.get("arg", {}) or {})
    if kind == "binary" and str(expr.get("op")) == "mul":
        for term in _flatten_op("mul", expr):
            found = _periodic_core(term)
            if found is not None:
                return found
    return None


def _affine_feature_signature(expr: Mapping[str, Any]) -> str:
    terms = _flatten_signed_add(expr)
    rows: list[str] = []
    for sign, term in terms:
        constant, non_const = _constant_factor(term)
        if not non_const:
            continue
        if len(non_const) == 1 and str(non_const[0].get("type")) == "feature":
            rows.append(f"{_number_token(sign * constant)}*x{int(non_const[0].get('index', 0))}")
        else:
            rows.append(f"{_number_token(sign * constant)}*{expression_equivalence_key(_build_sorted_mul(non_const))}")
    return "+".join(sorted(rows)) or "constant_phase"


def _flatten_signed_add(expr: Mapping[str, Any], sign: float = 1.0) -> list[tuple[float, dict[str, Any]]]:
    if str(expr.get("type")) == "binary":
        op = str(expr.get("op"))
        if op == "add":
            return _flatten_signed_add(dict(expr.get("left", {}) or {}), sign) + _flatten_signed_add(dict(expr.get("right", {}) or {}), sign)
        if op == "sub":
            return _flatten_signed_add(dict(expr.get("left", {}) or {}), sign) + _flatten_signed_add(dict(expr.get("right", {}) or {}), -sign)
    return [(float(sign), dict(expr))]


def _constant_factor(expr: Mapping[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    if str(expr.get("type")) != "binary" or str(expr.get("op")) != "mul":
        value = _const_value(expr)
        return (float(value), []) if value is not None else (1.0, [dict(expr)])
    constant = 1.0
    non_const: list[dict[str, Any]] = []
    for term in _flatten_op("mul", expr):
        value = _const_value(term)
        if value is None:
            non_const.append(dict(term))
        else:
            constant *= float(value)
    return float(constant), non_const


def _ratio_signature(expr: Mapping[str, Any]) -> dict[str, Any]:
    if str(expr.get("type")) != "binary" or str(expr.get("op")) != "div":
        return {}
    numerator = canonicalize_expression(dict(expr.get("left", {}) or {}))
    denominator = canonicalize_expression(dict(expr.get("right", {}) or {}))
    return {
        "numerator_key": expression_equivalence_key(numerator),
        "denominator_key": expression_equivalence_key(denominator),
        "numerator_features": sorted(_collect_feature_indices(numerator)),
        "denominator_features": sorted(_collect_feature_indices(denominator)),
    }


def _trace(root: str, rule: str, before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(root),
        "rule": str(rule),
        "before": expression_to_string(before, precision=12),
        "after": expression_to_string(after, precision=12),
    }


def _const_value(expr: Mapping[str, Any]) -> float | None:
    return float(expr["value"]) if str(expr.get("type")) == "const" and expr.get("value") is not None else None


def _is_const(expr: Mapping[str, Any], value: float) -> bool:
    raw = _const_value(expr)
    return raw is not None and abs(float(raw) - float(value)) <= 1e-12


def _eval_unary_const(op: str, value: float) -> float:
    if op == "identity":
        return float(value)
    if op == "square":
        return float(value * value)
    if op == "sin":
        return float(np.sin(value))
    if op == "cos":
        return float(np.cos(value))
    if op == "tanh":
        return float(np.tanh(value))
    if op == "exp":
        return float(np.exp(np.clip(value, -30.0, 30.0)))
    if op == "log":
        return float(np.log(abs(value) + 1e-6))
    if op == "abs":
        return float(abs(value))
    if op == "sqrt":
        return float(np.sqrt(abs(value) + 1e-6))
    return float(value)


def _eval_binary_const(op: str, left: float, right: float) -> float:
    if op == "add":
        return float(left + right)
    if op == "sub":
        return float(left - right)
    if op == "mul":
        return float(left * right)
    if op == "div":
        denom = right if abs(right) > 1e-6 else (1e-6 if right >= 0.0 else -1e-6)
        return float(left / denom)
    return float(left)


def _clean_number(value: float) -> float:
    raw = float(value)
    if abs(raw) <= 1e-12:
        return 0.0
    if math.isfinite(raw):
        return float(f"{raw:.15g}")
    return raw


def _number_token(value: float) -> str:
    raw = _clean_number(float(value))
    if math.isnan(raw):
        return "nan"
    if math.isinf(raw):
        return "inf" if raw > 0 else "-inf"
    return f"{raw:.15g}"


__all__ = [
    "canonicalize_expression",
    "expression_canonical_payload",
    "expression_canonical_string",
    "expression_equivalence_key",
    "expression_family_signature",
    "simplify_expression",
]
