from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional at import time
    torch = None  # type: ignore[assignment]


_ALLOWED_EXPR_TYPES = {"feature", "const", "param", "unary", "binary"}
_ALLOWED_UNARY_OPS = {"identity", "square", "sin", "cos", "tanh", "exp", "log", "abs", "sqrt"}
_ALLOWED_BINARY_OPS = {"add", "sub", "mul", "div"}


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    init: float = 1.0
    trainable: bool = True


def default_genome(
    input_dim: int,
    *,
    ops: Sequence[str] = ("identity", "square", "sin", "cos"),
) -> tuple[Dict[str, Any], ...]:
    if int(input_dim) <= 0:
        raise ValueError("input_dim must be positive")

    out: list[Dict[str, Any]] = []
    for i in range(int(input_dim)):
        for op in ops:
            key = str(op).strip().lower()
            base: Dict[str, Any] = {"type": "feature", "index": int(i)}
            if key == "identity":
                expr = base
                name = f"x{i}"
            else:
                if key not in _ALLOWED_UNARY_OPS:
                    raise ValueError(f"Unknown default op '{op}'")
                expr = {
                    "type": "unary",
                    "op": key,
                    "arg": base,
                }
                name = f"{key}(x{i})"
            out.append({"name": name, "expr": expr})

    return tuple(out)


def _normalize_expression(expr: Mapping[str, Any], *, input_dim: int) -> Dict[str, Any]:
    if not isinstance(expr, Mapping):
        raise TypeError(f"expression must be mapping, got {type(expr).__name__}")

    t = str(expr.get("type", "")).strip().lower()
    if t not in _ALLOWED_EXPR_TYPES:
        raise ValueError(f"Unsupported expression type '{t}'. Allowed: {sorted(_ALLOWED_EXPR_TYPES)}")

    if t == "feature":
        if "index" not in expr:
            raise ValueError("feature expression requires 'index'")
        idx = int(expr["index"])
        if idx < 0 or idx >= int(input_dim):
            raise ValueError(f"feature index {idx} out of range for input_dim={input_dim}")
        return {
            "type": "feature",
            "index": idx,
        }

    if t == "const":
        if "value" not in expr:
            raise ValueError("const expression requires 'value'")
        return {
            "type": "const",
            "value": float(expr["value"]),
        }

    if t == "param":
        name = str(expr.get("name", "")).strip()
        if not name:
            raise ValueError("param expression requires non-empty 'name'")
        return {
            "type": "param",
            "name": name,
            "init": float(expr.get("init", 1.0)),
            "trainable": bool(expr.get("trainable", True)),
        }

    if t == "unary":
        op = str(expr.get("op", "")).strip().lower()
        if op not in _ALLOWED_UNARY_OPS:
            raise ValueError(f"Unsupported unary op '{op}'. Allowed: {sorted(_ALLOWED_UNARY_OPS)}")
        if "arg" not in expr:
            raise ValueError("unary expression requires 'arg'")
        return {
            "type": "unary",
            "op": op,
            "arg": _normalize_expression(expr["arg"], input_dim=input_dim),
        }

    # binary
    op = str(expr.get("op", "")).strip().lower()
    if op not in _ALLOWED_BINARY_OPS:
        raise ValueError(f"Unsupported binary op '{op}'. Allowed: {sorted(_ALLOWED_BINARY_OPS)}")
    if "left" not in expr or "right" not in expr:
        raise ValueError("binary expression requires 'left' and 'right'")
    return {
        "type": "binary",
        "op": op,
        "left": _normalize_expression(expr["left"], input_dim=input_dim),
        "right": _normalize_expression(expr["right"], input_dim=input_dim),
    }


def normalize_genome(genome: Sequence[Mapping[str, Any]], *, input_dim: int) -> tuple[Dict[str, Any], ...]:
    if not genome:
        raise ValueError("genome must not be empty")

    terms: list[Dict[str, Any]] = []
    for i, raw in enumerate(genome):
        if not isinstance(raw, Mapping):
            raise TypeError(f"genome term[{i}] must be mapping, got {type(raw).__name__}")

        if "expr" in raw:
            expr_raw = raw["expr"]
            name = str(raw.get("name", f"term_{i}")).strip() or f"term_{i}"
        else:
            expr_raw = raw
            name = f"term_{i}"

        expr = _normalize_expression(expr_raw, input_dim=input_dim)
        terms.append({
            "name": name,
            "expr": expr,
        })

    return tuple(terms)


def collect_parameter_specs(genome: Sequence[Mapping[str, Any]]) -> tuple[ParameterSpec, ...]:
    seen: Dict[str, ParameterSpec] = {}

    def visit(expr: Mapping[str, Any]) -> None:
        t = str(expr["type"]).strip().lower()

        if t == "param":
            spec = ParameterSpec(
                name=str(expr["name"]),
                init=float(expr.get("init", 1.0)),
                trainable=bool(expr.get("trainable", True)),
            )
            old = seen.get(spec.name)
            if old is None:
                seen[spec.name] = spec
            else:
                if old.trainable != spec.trainable:
                    raise ValueError(
                        f"Parameter '{spec.name}' has conflicting trainable flags: "
                        f"{old.trainable} vs {spec.trainable}"
                    )
                if abs(float(old.init) - float(spec.init)) > 1e-12:
                    raise ValueError(
                        f"Parameter '{spec.name}' has conflicting init values: {old.init} vs {spec.init}"
                    )
            return

        if t == "unary":
            visit(expr["arg"])
            return

        if t == "binary":
            visit(expr["left"])
            visit(expr["right"])
            return

    for term in genome:
        visit(term["expr"])

    return tuple(seen[k] for k in sorted(seen.keys()))


def _safe_div_numpy(a: np.ndarray, b: np.ndarray, *, eps: float) -> np.ndarray:
    denom = np.where(np.abs(b) < float(eps), np.where(b >= 0.0, float(eps), -float(eps)), b)
    return a / denom


def _safe_log_numpy(x: np.ndarray, *, eps: float) -> np.ndarray:
    return np.log(np.abs(x) + float(eps))


def _safe_sqrt_numpy(x: np.ndarray, *, eps: float) -> np.ndarray:
    return np.sqrt(np.abs(x) + float(eps))


def evaluate_expression_numpy(
    expr: Mapping[str, Any],
    X: np.ndarray,
    *,
    param_values: Mapping[str, float] | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    x = np.asarray(X, dtype=float)
    if x.ndim != 2:
        raise ValueError("X must be 2D")

    params = dict(param_values or {})
    n = int(x.shape[0])

    def rec(node: Mapping[str, Any]) -> np.ndarray:
        t = str(node["type"])

        if t == "feature":
            return x[:, int(node["index"])]

        if t == "const":
            return np.full((n,), float(node["value"]), dtype=float)

        if t == "param":
            key = str(node["name"])
            if key in params:
                val = float(params[key])
            else:
                val = float(node.get("init", 1.0))
            return np.full((n,), val, dtype=float)

        if t == "unary":
            z = rec(node["arg"])
            op = str(node["op"])
            if op == "identity":
                return z
            if op == "square":
                return z * z
            if op == "sin":
                return np.sin(z)
            if op == "cos":
                return np.cos(z)
            if op == "tanh":
                return np.tanh(z)
            if op == "exp":
                return np.exp(np.clip(z, -30.0, 30.0))
            if op == "log":
                return _safe_log_numpy(z, eps=float(eps))
            if op == "abs":
                return np.abs(z)
            if op == "sqrt":
                return _safe_sqrt_numpy(z, eps=float(eps))
            raise ValueError(f"Unsupported unary op: {op}")

        if t == "binary":
            l = rec(node["left"])
            r = rec(node["right"])
            op = str(node["op"])
            if op == "add":
                return l + r
            if op == "sub":
                return l - r
            if op == "mul":
                return l * r
            if op == "div":
                return _safe_div_numpy(l, r, eps=float(eps))
            raise ValueError(f"Unsupported binary op: {op}")

        raise ValueError(f"Unsupported node type: {t}")

    y = rec(expr)
    return np.asarray(y, dtype=float).reshape(-1)


def evaluate_genome_numpy(
    genome: Sequence[Mapping[str, Any]],
    X: np.ndarray,
    *,
    param_values: Mapping[str, float] | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    rows = [
        evaluate_expression_numpy(term["expr"], X, param_values=param_values, eps=eps)
        for term in genome
    ]
    if not rows:
        raise ValueError("genome produced no terms")
    return np.asarray(np.stack(rows, axis=1), dtype=float)


def _safe_div_torch(a, b, *, eps: float):
    if torch is None:
        raise RuntimeError("torch is not available")
    sign = torch.where(b >= 0.0, torch.ones_like(b), -torch.ones_like(b))
    denom = torch.where(torch.abs(b) < float(eps), sign * float(eps), b)
    return a / denom


def _safe_log_torch(x, *, eps: float):
    if torch is None:
        raise RuntimeError("torch is not available")
    return torch.log(torch.abs(x) + float(eps))


def _safe_sqrt_torch(x, *, eps: float):
    if torch is None:
        raise RuntimeError("torch is not available")
    return torch.sqrt(torch.abs(x) + float(eps))


def evaluate_expression_torch(
    expr: Mapping[str, Any],
    X,
    *,
    param_values: Mapping[str, Any] | None = None,
    eps: float = 1e-6,
):
    if torch is None:
        raise RuntimeError("torch is not available")
    if X.ndim != 2:
        raise ValueError("X must be 2D")

    params = dict(param_values or {})
    n = int(X.shape[0])

    def rec(node: Mapping[str, Any]):
        t = str(node["type"])

        if t == "feature":
            return X[:, int(node["index"])]

        if t == "const":
            return torch.full((n,), float(node["value"]), dtype=X.dtype, device=X.device)

        if t == "param":
            key = str(node["name"])
            if key in params:
                v = params[key]
                if not torch.is_tensor(v):
                    v = torch.as_tensor(float(v), dtype=X.dtype, device=X.device)
                else:
                    v = v.to(dtype=X.dtype, device=X.device)
            else:
                v = torch.as_tensor(float(node.get("init", 1.0)), dtype=X.dtype, device=X.device)

            if v.ndim == 0:
                return v
            if v.ndim == 1 and int(v.shape[0]) == n:
                return v
            raise ValueError(f"param '{key}' has invalid shape {tuple(v.shape)}")

        if t == "unary":
            z = rec(node["arg"])
            op = str(node["op"])
            if op == "identity":
                return z
            if op == "square":
                return z * z
            if op == "sin":
                return torch.sin(z)
            if op == "cos":
                return torch.cos(z)
            if op == "tanh":
                return torch.tanh(z)
            if op == "exp":
                return torch.exp(torch.clamp(z, min=-30.0, max=30.0))
            if op == "log":
                return _safe_log_torch(z, eps=float(eps))
            if op == "abs":
                return torch.abs(z)
            if op == "sqrt":
                return _safe_sqrt_torch(z, eps=float(eps))
            raise ValueError(f"Unsupported unary op: {op}")

        if t == "binary":
            l = rec(node["left"])
            r = rec(node["right"])
            op = str(node["op"])
            if op == "add":
                return l + r
            if op == "sub":
                return l - r
            if op == "mul":
                return l * r
            if op == "div":
                return _safe_div_torch(l, r, eps=float(eps))
            raise ValueError(f"Unsupported binary op: {op}")

        raise ValueError(f"Unsupported node type: {t}")

    return rec(expr)


def evaluate_genome_torch(
    genome: Sequence[Mapping[str, Any]],
    X,
    *,
    param_values: Mapping[str, Any] | None = None,
    eps: float = 1e-6,
):
    if torch is None:
        raise RuntimeError("torch is not available")

    cols = [
        evaluate_expression_torch(term["expr"], X, param_values=param_values, eps=eps)
        for term in genome
    ]
    if not cols:
        raise ValueError("genome produced no terms")

    fixed = []
    for col in cols:
        if col.ndim == 0:
            col = torch.full((int(X.shape[0]),), float(col.detach().cpu().item()), dtype=X.dtype, device=X.device)
        fixed.append(col.reshape(-1))

    return torch.stack(fixed, dim=1)


def _fmt_scalar(v: float, precision: int = 6) -> str:
    return f"{float(v):.{int(precision)}g}"


def expression_to_string(
    expr: Mapping[str, Any],
    *,
    param_values: Mapping[str, float] | None = None,
    precision: int = 6,
) -> str:
    params = dict(param_values or {})

    def rec(node: Mapping[str, Any]) -> str:
        t = str(node["type"])

        if t == "feature":
            return f"x{int(node['index'])}"

        if t == "const":
            return _fmt_scalar(float(node["value"]), precision=precision)

        if t == "param":
            key = str(node["name"])
            if key in params:
                return _fmt_scalar(float(params[key]), precision=precision)
            return key

        if t == "unary":
            op = str(node["op"])
            a = rec(node["arg"])
            if op == "identity":
                return f"({a})"
            if op == "square":
                return f"(({a})^2)"
            return f"{op}({a})"

        if t == "binary":
            l = rec(node["left"])
            r = rec(node["right"])
            op = str(node["op"])
            if op == "add":
                return f"(({l})+({r}))"
            if op == "sub":
                return f"(({l})-({r}))"
            if op == "mul":
                return f"(({l})*({r}))"
            if op == "div":
                return f"(({l})/({r}))"
            return f"{op}({l},{r})"

        return "<?>"

    return rec(expr)


def genome_to_strings(
    genome: Sequence[Mapping[str, Any]],
    *,
    param_values: Mapping[str, float] | None = None,
    precision: int = 6,
) -> tuple[str, ...]:
    out = []
    for term in genome:
        out.append(expression_to_string(term["expr"], param_values=param_values, precision=precision))
    return tuple(out)


def detect_binary_columns(X: np.ndarray, *, round_decimals: int = 10) -> tuple[bool, ...]:
    x = np.asarray(X, dtype=float)
    if x.ndim != 2:
        raise ValueError("X must be 2D")

    flags: list[bool] = []
    for j in range(int(x.shape[1])):
        col = np.asarray(x[:, j], dtype=float)
        uniq = np.unique(np.round(col, int(round_decimals)))
        flags.append(bool(len(uniq) <= 2))
    return tuple(flags)


def _feature_expr(index: int) -> Dict[str, Any]:
    return {
        "type": "feature",
        "index": int(index),
    }


def _const_expr(value: float) -> Dict[str, Any]:
    return {
        "type": "const",
        "value": float(value),
    }


def _unary_expr(op: str, arg: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "type": "unary",
        "op": str(op),
        "arg": dict(arg),
    }


def _binary_expr(op: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "type": "binary",
        "op": str(op),
        "left": dict(left),
        "right": dict(right),
    }


def _relu_expr(arg: Mapping[str, Any]) -> Dict[str, Any]:
    # relu(z) = 0.5 * (z + abs(z)) using existing ops (no extra runtime op required)
    z = dict(arg)
    z_abs = _unary_expr("abs", z)
    z_sum = _binary_expr("add", z, z_abs)
    return _binary_expr("mul", _const_expr(0.5), z_sum)


def _feature_scores(X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(X, dtype=float)
    n, d = x.shape

    if y is None:
        return np.std(x, axis=0)

    yt = np.asarray(y, dtype=float)
    if yt.ndim == 2:
        yt = yt[:, 0]
    yt = yt.reshape(-1)

    if yt.shape[0] != n:
        raise ValueError("y length mismatch for feature scoring")

    yc = yt - np.mean(yt)
    y_std = float(np.std(yc)) + 1e-12

    scores = np.zeros((d,), dtype=float)
    for j in range(d):
        xc = x[:, j] - np.mean(x[:, j])
        x_std = float(np.std(xc)) + 1e-12
        corr = float(np.mean(xc * yc) / (x_std * y_std))
        scores[j] = abs(corr)

    return scores


def default_genome_v2(
    X: np.ndarray,
    *,
    y: np.ndarray | None = None,
    continuous_ops: Sequence[str] = ("identity", "sin", "cos"),
    binary_ops: Sequence[str] = ("identity",),
    include_interactions: bool = True,
    max_interactions: int = 20,
    topk_features: int = 6,
    include_hinge: bool = True,
    hinge_quantiles: Sequence[float] = (0.25, 0.5, 0.75),
) -> tuple[Dict[str, Any], ...]:
    x = np.asarray(X, dtype=float)
    if x.ndim != 2:
        raise ValueError("X must be 2D")

    n, d = x.shape
    if n <= 0 or d <= 0:
        raise ValueError("X must be non-empty 2D matrix")

    cont_ops = tuple(str(o).strip().lower() for o in continuous_ops)
    bin_ops = tuple(str(o).strip().lower() for o in binary_ops)

    for op in cont_ops:
        if op not in _ALLOWED_UNARY_OPS:
            raise ValueError(f"Unsupported op in continuous_ops: {op}")
    for op in bin_ops:
        if op not in _ALLOWED_UNARY_OPS:
            raise ValueError(f"Unsupported op in binary_ops: {op}")

    is_binary = detect_binary_columns(x)

    terms: list[Dict[str, Any]] = []

    # base terms per feature
    for i in range(d):
        ops = bin_ops if is_binary[i] else cont_ops
        base = _feature_expr(i)

        for op in ops:
            if op == "identity":
                expr = base
                name = f"x{i}"
            else:
                expr = _unary_expr(op, base)
                name = f"{op}(x{i})"
            terms.append({"name": name, "expr": expr})

    # feature ranking for controlled expansion
    scores = _feature_scores(x, y=y)
    ranked = list(np.argsort(-scores))

    topk = max(2, int(topk_features))
    selected = [int(i) for i in ranked[: min(topk, d)]]

    # interaction terms: x_i * x_j
    if bool(include_interactions):
        pair_count = 0
        max_pairs = max(0, int(max_interactions))

        for ai in range(len(selected)):
            for aj in range(ai + 1, len(selected)):
                i = selected[ai]
                j = selected[aj]
                expr = _binary_expr("mul", _feature_expr(i), _feature_expr(j))
                terms.append({"name": f"x{i}*x{j}", "expr": expr})
                pair_count += 1
                if pair_count >= max_pairs:
                    break
            if pair_count >= max_pairs:
                break

    # hinge / piecewise terms: relu(x_i - q)
    if bool(include_hinge):
        qs = []
        for q in hinge_quantiles:
            qq = float(q)
            if 0.0 < qq < 1.0:
                qs.append(qq)
        qs = sorted(set(qs))

        if qs:
            # use non-binary features first for hinges
            non_binary_sel = [i for i in selected if not is_binary[i]]
            hinge_features = non_binary_sel if non_binary_sel else selected

            for i in hinge_features:
                col = x[:, int(i)]
                for q in qs:
                    threshold = float(np.quantile(col, q))
                    shifted = _binary_expr("sub", _feature_expr(i), _const_expr(threshold))
                    expr = _relu_expr(shifted)
                    terms.append({
                        "name": f"relu(x{i}-{threshold:.4g})",
                        "expr": expr,
                    })

    return tuple(terms)
