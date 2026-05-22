from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.models.symbolic import (
    Expression,
    ParameterSpec,
    binary_expr,
    const_expr,
    evaluate_expression_numpy,
    param_expr,
    unary_expr,
)


@dataclass(frozen=True)
class GradientSignal:
    """Lightweight signal used by symbolic search-space expansion."""

    feature_scores: tuple[float, ...]
    feature_order: tuple[int, ...]
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_scores": list(self.feature_scores),
            "feature_order": list(self.feature_order),
            "source": self.source,
            "metadata": dict(self.metadata),
        }


def differentiate_expression(
    expr: Mapping[str, Any],
    *,
    wrt_param: str | None = None,
    wrt_feature: int | None = None,
) -> Expression:
    """Return a symbolic derivative expression for supported operators."""

    if (wrt_param is None) == (wrt_feature is None):
        raise ValueError("provide exactly one of wrt_param or wrt_feature")

    def d(node: Mapping[str, Any]) -> Expression:
        kind = str(node["type"])
        if kind == "feature":
            return const_expr(1.0 if wrt_feature is not None and int(node["index"]) == int(wrt_feature) else 0.0)
        if kind == "param":
            return const_expr(1.0 if wrt_param is not None and str(node["name"]) == str(wrt_param) else 0.0)
        if kind == "const":
            return const_expr(0.0)
        if kind == "unary":
            op = str(node["op"])
            arg = dict(node["arg"])
            dz = d(arg)
            if op == "identity":
                return dz
            if op == "square":
                return binary_expr("mul", binary_expr("mul", const_expr(2.0), arg), dz)
            if op == "sin":
                return binary_expr("mul", unary_expr("cos", arg), dz)
            if op == "cos":
                return binary_expr("mul", binary_expr("mul", const_expr(-1.0), unary_expr("sin", arg)), dz)
            if op == "tanh":
                tanh_arg = unary_expr("tanh", arg)
                return binary_expr("mul", binary_expr("sub", const_expr(1.0), unary_expr("square", tanh_arg)), dz)
            if op == "exp":
                return binary_expr("mul", unary_expr("exp", arg), dz)
            if op == "abs":
                return binary_expr("mul", binary_expr("div", arg, binary_expr("add", unary_expr("abs", arg), const_expr(1e-6))), dz)
            if op == "log":
                return binary_expr("mul", binary_expr("div", arg, binary_expr("add", unary_expr("abs", arg), const_expr(1e-6))), dz)
            if op == "sqrt":
                denom = binary_expr("mul", const_expr(2.0), unary_expr("sqrt", arg))
                return binary_expr("mul", binary_expr("div", const_expr(1.0), denom), dz)
            raise ValueError(f"unsupported unary op for derivative: {op}")
        if kind == "binary":
            op = str(node["op"])
            left = dict(node["left"])
            right = dict(node["right"])
            dl = d(left)
            dr = d(right)
            if op == "add":
                return binary_expr("add", dl, dr)
            if op == "sub":
                return binary_expr("sub", dl, dr)
            if op == "mul":
                return binary_expr("add", binary_expr("mul", dl, right), binary_expr("mul", left, dr))
            if op == "div":
                numerator = binary_expr(
                    "sub",
                    binary_expr("mul", dl, right),
                    binary_expr("mul", left, dr),
                )
                return binary_expr("div", numerator, unary_expr("square", right))
            raise ValueError(f"unsupported binary op for derivative: {op}")
        raise ValueError(f"unsupported expression node: {kind}")

    return d(expr)


def evaluate_expression_derivative_numpy(
    expr: Mapping[str, Any],
    X: np.ndarray,
    *,
    wrt_param: str | None = None,
    wrt_feature: int | None = None,
    param_values: Mapping[str, float] | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Evaluate derivative values using direct chain rule arrays."""

    if (wrt_param is None) == (wrt_feature is None):
        raise ValueError("provide exactly one of wrt_param or wrt_feature")
    x = np.asarray(X, dtype=float)
    if x.ndim != 2:
        raise ValueError("X must be 2D")
    params = dict(param_values or {})
    n = int(x.shape[0])

    def rec(node: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        kind = str(node["type"])
        if kind == "feature":
            values = np.asarray(x[:, int(node["index"])], dtype=float).reshape(-1)
            grad = np.ones(n, dtype=float) if wrt_feature is not None and int(node["index"]) == int(wrt_feature) else np.zeros(n, dtype=float)
            return values, grad
        if kind == "param":
            name = str(node["name"])
            values = np.full((n,), float(params.get(name, node.get("init", 1.0))), dtype=float)
            grad = np.ones(n, dtype=float) if wrt_param is not None and name == str(wrt_param) else np.zeros(n, dtype=float)
            return values, grad
        if kind == "const":
            return np.full((n,), float(node["value"]), dtype=float), np.zeros(n, dtype=float)
        if kind == "unary":
            z, dz = rec(node["arg"])
            op = str(node["op"])
            if op == "identity":
                return z, dz
            if op == "square":
                return z * z, 2.0 * z * dz
            if op == "sin":
                return np.sin(z), np.cos(z) * dz
            if op == "cos":
                return np.cos(z), -np.sin(z) * dz
            if op == "tanh":
                t = np.tanh(z)
                return t, (1.0 - t * t) * dz
            if op == "exp":
                clipped = np.clip(z, -30.0, 30.0)
                mask = (z >= -30.0) & (z <= 30.0)
                val = np.exp(clipped)
                return val, val * dz * mask.astype(float)
            if op == "log":
                return np.log(np.abs(z) + float(eps)), (np.sign(z) / (np.abs(z) + float(eps))) * dz
            if op == "abs":
                return np.abs(z), np.sign(z) * dz
            if op == "sqrt":
                denom = 2.0 * np.sqrt(np.abs(z) + float(eps))
                return np.sqrt(np.abs(z) + float(eps)), (np.sign(z) / denom) * dz
            raise ValueError(f"unsupported unary op: {op}")
        if kind == "binary":
            left, dleft = rec(node["left"])
            right, dright = rec(node["right"])
            op = str(node["op"])
            if op == "add":
                return left + right, dleft + dright
            if op == "sub":
                return left - right, dleft - dright
            if op == "mul":
                return left * right, dleft * right + left * dright
            if op == "div":
                denom = np.where(np.abs(right) < float(eps), np.where(right >= 0.0, float(eps), -float(eps)), right)
                return left / denom, (dleft * denom - left * dright) / (denom * denom)
            raise ValueError(f"unsupported binary op: {op}")
        raise ValueError(f"unsupported node type: {kind}")

    _values, derivative = rec(expr)
    return np.asarray(derivative, dtype=float).reshape(-1)


def parameter_jacobian_numpy(
    expr: Mapping[str, Any],
    specs: Sequence[ParameterSpec],
    X: np.ndarray,
    *,
    param_values: Mapping[str, float] | None = None,
) -> np.ndarray:
    trainable = tuple(spec for spec in specs if bool(spec.trainable))
    if not trainable:
        return np.zeros((np.asarray(X).shape[0], 0), dtype=float)
    cols = [
        evaluate_expression_derivative_numpy(expr, X, wrt_param=spec.name, param_values=param_values)
        for spec in trainable
    ]
    return np.column_stack(cols)


def symbolic_mse_parameter_gradient_numpy(
    expr: Mapping[str, Any],
    specs: Sequence[ParameterSpec],
    X: np.ndarray,
    y: np.ndarray,
    *,
    param_values: Mapping[str, float] | None = None,
    l2: float = 0.0,
) -> np.ndarray:
    trainable = tuple(spec for spec in specs if bool(spec.trainable))
    if not trainable:
        return np.zeros(0, dtype=float)
    pred = evaluate_expression_numpy(expr, X, param_values=param_values)
    target = np.asarray(y, dtype=float).reshape(-1)
    err = np.asarray(pred - target, dtype=float).reshape(-1)
    jac = parameter_jacobian_numpy(expr, trainable, X, param_values=param_values)
    grad = (2.0 / float(max(1, err.shape[0]))) * (jac.T @ err)
    if float(l2) != 0.0:
        values = np.asarray([float((param_values or {}).get(spec.name, spec.init)) for spec in trainable], dtype=float)
        grad = grad + (2.0 * float(l2) * values)
    return np.asarray(grad, dtype=float).reshape(-1)


def parse_residual_gradient_signal(
    X: np.ndarray,
    residuals: np.ndarray,
    *,
    source: str = "residual_correlation",
) -> GradientSignal:
    x = np.asarray(X, dtype=float)
    residual = np.asarray(residuals, dtype=float).reshape(-1)
    if x.ndim != 2:
        raise ValueError("X must be 2D")
    if residual.shape[0] != x.shape[0]:
        raise ValueError("residual length must match X rows")
    scores = []
    centered_residual = residual - float(np.mean(residual))
    denom_residual = float(np.linalg.norm(centered_residual)) + 1e-12
    for j in range(x.shape[1]):
        col = x[:, j] - float(np.mean(x[:, j]))
        score = float(np.dot(col, centered_residual) / ((float(np.linalg.norm(col)) + 1e-12) * denom_residual))
        scores.append(abs(score))
    order = tuple(int(i) for i in np.argsort(-np.asarray(scores, dtype=float)))
    return GradientSignal(
        feature_scores=tuple(float(v) for v in scores),
        feature_order=order,
        source=str(source),
        metadata={"n_features": int(x.shape[1])},
    )
