from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from core.symbolic.symbolic_dsl import evaluate_expression_numpy, expression_to_string


def _is_const(node: Mapping[str, Any], value: float, *, tol: float = 1e-12) -> bool:
    return str(node.get("type", "")) == "const" and abs(float(node.get("value", 0.0)) - float(value)) <= float(tol)


def _const(value: float) -> Dict[str, Any]:
    return {
        "type": "const",
        "value": float(value),
    }


def _copy_expr(node: Mapping[str, Any]) -> Dict[str, Any]:
    t = str(node.get("type", ""))
    if t in {"feature", "const", "param"}:
        return dict(node)
    if t == "unary":
        return {
            "type": "unary",
            "op": str(node["op"]),
            "arg": _copy_expr(node["arg"]),
        }
    if t == "binary":
        return {
            "type": "binary",
            "op": str(node["op"]),
            "left": _copy_expr(node["left"]),
            "right": _copy_expr(node["right"]),
        }
    raise ValueError(f"Unsupported node type: {t}")


def _unary(op: str, arg: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "type": "unary",
        "op": str(op),
        "arg": _copy_expr(arg),
    }


def _binary(op: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "type": "binary",
        "op": str(op),
        "left": _copy_expr(left),
        "right": _copy_expr(right),
    }


def _add(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_const(a, 0.0):
        return _copy_expr(b)
    if _is_const(b, 0.0):
        return _copy_expr(a)
    if str(a.get("type", "")) == "const" and str(b.get("type", "")) == "const":
        return _const(float(a["value"]) + float(b["value"]))
    return _binary("add", a, b)


def _sub(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_const(b, 0.0):
        return _copy_expr(a)
    if str(a.get("type", "")) == "const" and str(b.get("type", "")) == "const":
        return _const(float(a["value"]) - float(b["value"]))
    return _binary("sub", a, b)


def _mul(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_const(a, 0.0) or _is_const(b, 0.0):
        return _const(0.0)
    if _is_const(a, 1.0):
        return _copy_expr(b)
    if _is_const(b, 1.0):
        return _copy_expr(a)
    if str(a.get("type", "")) == "const" and str(b.get("type", "")) == "const":
        return _const(float(a["value"]) * float(b["value"]))
    return _binary("mul", a, b)


def _div(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    if _is_const(a, 0.0):
        return _const(0.0)
    if _is_const(b, 1.0):
        return _copy_expr(a)
    if str(a.get("type", "")) == "const" and str(b.get("type", "")) == "const":
        bv = float(b["value"])
        if abs(bv) <= 1e-12:
            return _binary("div", a, b)
        return _const(float(a["value"]) / bv)
    return _binary("div", a, b)


def _square(a: Mapping[str, Any]) -> Dict[str, Any]:
    if str(a.get("type", "")) == "const":
        v = float(a["value"])
        return _const(v * v)
    return _unary("square", a)


def _safe_abs(node: Mapping[str, Any]) -> Dict[str, Any]:
    return _unary("abs", node)


def _safe_sqrt(node: Mapping[str, Any]) -> Dict[str, Any]:
    return _unary("sqrt", node)


def _plus_eps(node: Mapping[str, Any], eps: float) -> Dict[str, Any]:
    return _add(node, _const(float(eps)))


def _d_expr_wrt_param(node: Mapping[str, Any], *, param_name: str, eps: float) -> Dict[str, Any]:
    t = str(node["type"])

    if t == "const":
        return _const(0.0)

    if t == "feature":
        return _const(0.0)

    if t == "param":
        return _const(1.0 if str(node["name"]) == str(param_name) else 0.0)

    if t == "unary":
        op = str(node["op"])
        u = _copy_expr(node["arg"])
        du = _d_expr_wrt_param(node["arg"], param_name=param_name, eps=eps)

        if op == "identity":
            return du
        if op == "square":
            return _mul(_mul(_const(2.0), u), du)
        if op == "sin":
            return _mul(_unary("cos", u), du)
        if op == "cos":
            return _mul(_mul(_const(-1.0), _unary("sin", u)), du)
        if op == "tanh":
            return _mul(_sub(_const(1.0), _square(_unary("tanh", u))), du)
        if op == "exp":
            return _mul(_unary("exp", u), du)
        if op == "log":
            den = _square(_plus_eps(_safe_abs(u), eps))
            return _mul(_div(u, den), du)
        if op == "abs":
            return _mul(_div(u, _plus_eps(_safe_abs(u), eps)), du)
        if op == "sqrt":
            dabs = _mul(_div(u, _plus_eps(_safe_abs(u), eps)), du)
            return _mul(_const(0.5), _div(dabs, _safe_sqrt(_plus_eps(_safe_abs(u), eps))))
        raise ValueError(f"Unsupported unary op for differentiation: {op}")

    if t == "binary":
        op = str(node["op"])
        u = _copy_expr(node["left"])
        v = _copy_expr(node["right"])
        du = _d_expr_wrt_param(node["left"], param_name=param_name, eps=eps)
        dv = _d_expr_wrt_param(node["right"], param_name=param_name, eps=eps)

        if op == "add":
            return _add(du, dv)
        if op == "sub":
            return _sub(du, dv)
        if op == "mul":
            return _add(_mul(du, v), _mul(u, dv))
        if op == "div":
            num = _sub(_mul(du, v), _mul(u, dv))
            den = _square(v)
            return _div(num, den)
        raise ValueError(f"Unsupported binary op for differentiation: {op}")

    raise ValueError(f"Unsupported node type for differentiation: {t}")


def _d_expr_wrt_feature(node: Mapping[str, Any], *, feature_index: int, eps: float) -> Dict[str, Any]:
    t = str(node["type"])

    if t == "const":
        return _const(0.0)

    if t == "param":
        return _const(0.0)

    if t == "feature":
        return _const(1.0 if int(node["index"]) == int(feature_index) else 0.0)

    if t == "unary":
        op = str(node["op"])
        u = _copy_expr(node["arg"])
        du = _d_expr_wrt_feature(node["arg"], feature_index=feature_index, eps=eps)

        if op == "identity":
            return du
        if op == "square":
            return _mul(_mul(_const(2.0), u), du)
        if op == "sin":
            return _mul(_unary("cos", u), du)
        if op == "cos":
            return _mul(_mul(_const(-1.0), _unary("sin", u)), du)
        if op == "tanh":
            return _mul(_sub(_const(1.0), _square(_unary("tanh", u))), du)
        if op == "exp":
            return _mul(_unary("exp", u), du)
        if op == "log":
            den = _square(_plus_eps(_safe_abs(u), eps))
            return _mul(_div(u, den), du)
        if op == "abs":
            return _mul(_div(u, _plus_eps(_safe_abs(u), eps)), du)
        if op == "sqrt":
            dabs = _mul(_div(u, _plus_eps(_safe_abs(u), eps)), du)
            return _mul(_const(0.5), _div(dabs, _safe_sqrt(_plus_eps(_safe_abs(u), eps))))
        raise ValueError(f"Unsupported unary op for differentiation: {op}")

    if t == "binary":
        op = str(node["op"])
        u = _copy_expr(node["left"])
        v = _copy_expr(node["right"])
        du = _d_expr_wrt_feature(node["left"], feature_index=feature_index, eps=eps)
        dv = _d_expr_wrt_feature(node["right"], feature_index=feature_index, eps=eps)

        if op == "add":
            return _add(du, dv)
        if op == "sub":
            return _sub(du, dv)
        if op == "mul":
            return _add(_mul(du, v), _mul(u, dv))
        if op == "div":
            num = _sub(_mul(du, v), _mul(u, dv))
            den = _square(v)
            return _div(num, den)
        raise ValueError(f"Unsupported binary op for differentiation: {op}")

    raise ValueError(f"Unsupported node type for differentiation: {t}")


def differentiate_expression_wrt_param(
    expr: Mapping[str, Any],
    *,
    param_name: str,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """Return symbolic derivative expression d(expr)/d(param_name)."""
    key = str(param_name).strip()
    if not key:
        raise ValueError("param_name must be non-empty")
    return _d_expr_wrt_param(expr, param_name=key, eps=float(eps))


def differentiate_expression_wrt_feature(
    expr: Mapping[str, Any],
    *,
    feature_index: int,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """Return symbolic derivative expression d(expr)/d(x_feature_index)."""
    idx = int(feature_index)
    if idx < 0:
        raise ValueError("feature_index must be >= 0")
    return _d_expr_wrt_feature(expr, feature_index=idx, eps=float(eps))


def gradient_formula_strings(
    expr: Mapping[str, Any],
    *,
    param_names: Sequence[str] = (),
    feature_indices: Sequence[int] = (),
    param_values: Mapping[str, float] | None = None,
    eps: float = 1e-6,
    precision: int = 6,
) -> Dict[str, str]:
    """Build readable gradient formulas for selected params/features."""
    out: Dict[str, str] = {}

    for name in param_names:
        key = str(name).strip()
        if not key:
            continue
        d_expr = differentiate_expression_wrt_param(expr, param_name=key, eps=float(eps))
        out[f"d/d{key}"] = expression_to_string(d_expr, param_values=param_values, precision=int(precision))

    for idx in feature_indices:
        j = int(idx)
        if j < 0:
            continue
        d_expr = differentiate_expression_wrt_feature(expr, feature_index=j, eps=float(eps))
        out[f"d/dx{j}"] = expression_to_string(d_expr, param_values=param_values, precision=int(precision))

    return out


def evaluate_gradient_numpy(
    expr: Mapping[str, Any],
    X: np.ndarray,
    *,
    param_name: str | None = None,
    feature_index: int | None = None,
    param_values: Mapping[str, float] | None = None,
    eps: float = 1e-6,
    graph_cache: Any | None = None,
    expr_key: str | None = None,
    batch_key: str | None = None,
) -> np.ndarray:
    """Evaluate selected symbolic gradient on samples."""
    choose = int(param_name is not None) + int(feature_index is not None)
    if choose != 1:
        raise ValueError("Exactly one of param_name or feature_index must be provided")

    if feature_index is not None and graph_cache is not None and hasattr(graph_cache, "evaluate_gradient"):
        try:
            out = graph_cache.evaluate_gradient(
                expr,
                X,
                feature_index=int(feature_index),
                param_values=param_values,
                eps=float(eps),
                expr_key=expr_key,
                batch_key=batch_key,
            )
            return np.asarray(out, dtype=float).reshape(-1)
        except Exception:
            pass

    if param_name is not None:
        d_expr = differentiate_expression_wrt_param(expr, param_name=str(param_name), eps=float(eps))
    else:
        d_expr = differentiate_expression_wrt_feature(expr, feature_index=int(feature_index), eps=float(eps))

    if graph_cache is not None and hasattr(graph_cache, "evaluate_expression"):
        try:
            out = graph_cache.evaluate_expression(
                d_expr,
                X,
                param_values=param_values,
                eps=float(eps),
                batch_key=batch_key,
            )
            return np.asarray(out, dtype=float).reshape(-1)
        except Exception:
            pass

    return evaluate_expression_numpy(d_expr, X, param_values=param_values, eps=float(eps))


__all__ = [
    "differentiate_expression_wrt_param",
    "differentiate_expression_wrt_feature",
    "gradient_formula_strings",
    "evaluate_gradient_numpy",
]
