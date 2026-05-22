from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


Expression = dict[str, Any]

ALLOWED_EXPR_TYPES = frozenset({"feature", "const", "param", "unary", "binary"})
ALLOWED_UNARY_OPS = frozenset({"identity", "square", "sin", "cos", "tanh", "exp", "log", "abs", "sqrt"})
ALLOWED_BINARY_OPS = frozenset({"add", "sub", "mul", "div"})


@dataclass(frozen=True)
class ParameterSpec:
    """Numeric slot inside a fixed symbolic expression."""

    name: str
    init: float = 1.0
    trainable: bool = True
    lower: float | None = None
    upper: float | None = None

    def clamp(self, value: float) -> float:
        out = float(value)
        if self.lower is not None:
            out = max(out, float(self.lower))
        if self.upper is not None:
            out = min(out, float(self.upper))
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "init": float(self.init),
            "trainable": bool(self.trainable),
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass(frozen=True)
class SymbolicExpressionModel:
    """Decoded symbolic point model with fitted parameter values."""

    name: str
    expression: Expression
    parameter_specs: tuple[ParameterSpec, ...]
    param_values: Mapping[str, float] = field(default_factory=dict)
    input_dim: int = 0
    feature_names: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def trainable_parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return tuple(spec for spec in self.parameter_specs if bool(spec.trainable))

    def parameter_names(self, *, trainable_only: bool = True) -> tuple[str, ...]:
        specs = self.trainable_parameter_specs if trainable_only else self.parameter_specs
        return tuple(spec.name for spec in specs)

    def parameter_vector(self, *, trainable_only: bool = True) -> np.ndarray:
        specs = self.trainable_parameter_specs if trainable_only else self.parameter_specs
        return np.asarray(
            [float(self.param_values.get(spec.name, spec.init)) for spec in specs],
            dtype=float,
        ).reshape(-1)

    def with_parameter_vector(self, values: Sequence[float] | np.ndarray) -> "SymbolicExpressionModel":
        param_values = parameter_values_from_vector(self.parameter_specs, values, base_values=self.param_values)
        return SymbolicExpressionModel(
            name=self.name,
            expression=dict(self.expression),
            parameter_specs=tuple(self.parameter_specs),
            param_values=param_values,
            input_dim=int(self.input_dim),
            feature_names=tuple(self.feature_names),
            metadata=dict(self.metadata),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return evaluate_expression_numpy(self.expression, X, param_values=self.param_values)

    def parameter_gradient(self, X: np.ndarray, y: np.ndarray, *, l2: float = 0.0) -> np.ndarray:
        """MSE gradient for trainable symbolic parameters."""

        specs = self.trainable_parameter_specs
        if not specs:
            return np.zeros(0, dtype=float)
        try:
            from mlblack.models.symbolic_gradient import symbolic_mse_parameter_gradient_numpy

            return symbolic_mse_parameter_gradient_numpy(
                self.expression,
                self.parameter_specs,
                X,
                y,
                param_values=self.param_values,
                l2=float(l2),
            )
        except Exception:
            # Keep the model usable even when a derivative rule is not yet implemented.
            pass
        base = self.parameter_vector(trainable_only=True)
        target = np.asarray(y, dtype=float).reshape(-1)

        def loss_for(vec: np.ndarray) -> float:
            model = self.with_parameter_vector(vec)
            pred = np.asarray(model.predict(X), dtype=float).reshape(-1)
            err = pred - target
            return float(np.mean(err * err) + float(l2) * float(np.sum(vec * vec)))

        grad = np.zeros_like(base, dtype=float)
        for i, value in enumerate(base):
            step = 1e-5 * max(1.0, abs(float(value)))
            plus = base.copy()
            minus = base.copy()
            plus[i] = specs[i].clamp(float(value) + step)
            minus[i] = specs[i].clamp(float(value) - step)
            denom = float(plus[i] - minus[i])
            if abs(denom) <= 1e-15:
                grad[i] = 0.0
            else:
                grad[i] = (loss_for(plus) - loss_for(minus)) / denom
        return np.asarray(grad, dtype=float).reshape(-1)

    def describe(self) -> dict[str, Any]:
        from mlblack.models.symbolic_normalization import expression_canonical_string, expression_equivalence_key

        return {
            "name": self.name,
            "kind": "symbolic_expression",
            "expression": dict(self.expression),
            "expression_string": expression_to_string(self.expression, param_values=self.param_values),
            "canonical_expression_string": expression_canonical_string(self.expression),
            "equivalence_key": expression_equivalence_key(self.expression),
            "parameter_specs": [spec.as_dict() for spec in self.parameter_specs],
            "param_values": {str(k): float(v) for k, v in self.param_values.items()},
            "input_dim": int(self.input_dim),
            "feature_names": list(self.feature_names),
            "complexity": float(expression_complexity(self.expression)),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SymbolicBasisSetModel:
    """Decoded multi-expression output used by orthogonal basis search."""

    atoms: tuple[SymbolicExpressionModel, ...]
    name: str = "symbolic_basis_set"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.atoms:
            raise ValueError("basis set contains no atoms")
        cols = [np.asarray(atom.predict(X), dtype=float).reshape(-1) for atom in self.atoms]
        return np.column_stack(cols)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)

    def expression_strings(self, *, fitted: bool = True) -> tuple[str, ...]:
        return tuple(
            expression_to_string(atom.expression, param_values=atom.param_values if fitted else None)
            for atom in self.atoms
        )

    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        out: list[ParameterSpec] = []
        for atom in self.atoms:
            out.extend(atom.parameter_specs)
        return tuple(out)

    def complexity(self) -> float:
        return float(sum(expression_complexity(atom.expression) for atom in self.atoms))

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "symbolic_basis_set",
            "n_atoms": int(len(self.atoms)),
            "expressions": [atom.describe() for atom in self.atoms],
            "expression_strings": list(self.expression_strings()),
            "complexity": float(self.complexity()),
            "metadata": dict(self.metadata),
        }


def feature_expr(index: int) -> Expression:
    return {"type": "feature", "index": int(index)}


def const_expr(value: float) -> Expression:
    return {"type": "const", "value": float(value)}


def param_expr(name: str, *, init: float = 1.0, trainable: bool = True) -> Expression:
    return {"type": "param", "name": str(name), "init": float(init), "trainable": bool(trainable)}


def unary_expr(op: str, arg: Mapping[str, Any]) -> Expression:
    return {"type": "unary", "op": str(op), "arg": dict(arg)}


def binary_expr(op: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> Expression:
    return {"type": "binary", "op": str(op), "left": dict(left), "right": dict(right)}


def parameterize_expression(
    expr: Mapping[str, Any],
    *,
    prefix: str,
    output_scale: bool = True,
    output_bias: bool = True,
    unary_input_affine: bool = True,
) -> Expression:
    """Add trainable numeric slots around a fixed symbolic structure."""

    base = dict(expr)
    if bool(unary_input_affine) and str(base.get("type")) == "unary":
        arg = dict(base["arg"])
        scaled = binary_expr(
            "add",
            binary_expr("mul", param_expr(f"{prefix}_arg_scale", init=1.0), arg),
            param_expr(f"{prefix}_arg_shift", init=0.0),
        )
        base = {**base, "arg": scaled}
    out: Expression = base
    if bool(output_scale):
        out = binary_expr("mul", param_expr(f"{prefix}_out_scale", init=1.0), out)
    if bool(output_bias):
        out = binary_expr("add", param_expr(f"{prefix}_out_bias", init=0.0), out)
    return out


def normalize_expression(expr: Mapping[str, Any], *, input_dim: int) -> Expression:
    if not isinstance(expr, Mapping):
        raise TypeError(f"expression must be mapping, got {type(expr).__name__}")
    kind = str(expr.get("type", "")).strip().lower()
    if kind not in ALLOWED_EXPR_TYPES:
        raise ValueError(f"unsupported expression type '{kind}'")

    if kind == "feature":
        if "index" not in expr:
            raise ValueError("feature expression requires 'index'")
        index = int(expr["index"])
        if index < 0 or index >= int(input_dim):
            raise ValueError(f"feature index {index} out of range for input_dim={input_dim}")
        return feature_expr(index)

    if kind == "const":
        if "value" not in expr:
            raise ValueError("const expression requires 'value'")
        return const_expr(float(expr["value"]))

    if kind == "param":
        name = str(expr.get("name", "")).strip()
        if not name:
            raise ValueError("param expression requires non-empty 'name'")
        out = param_expr(name, init=float(expr.get("init", 1.0)), trainable=bool(expr.get("trainable", True)))
        if expr.get("lower") is not None:
            out["lower"] = float(expr["lower"])
        if expr.get("upper") is not None:
            out["upper"] = float(expr["upper"])
        return out

    if kind == "unary":
        op = str(expr.get("op", "")).strip().lower()
        if op not in ALLOWED_UNARY_OPS:
            raise ValueError(f"unsupported unary op '{op}'")
        return {"type": "unary", "op": op, "arg": normalize_expression(expr["arg"], input_dim=input_dim)}

    op = str(expr.get("op", "")).strip().lower()
    if op not in ALLOWED_BINARY_OPS:
        raise ValueError(f"unsupported binary op '{op}'")
    return {
        "type": "binary",
        "op": op,
        "left": normalize_expression(expr["left"], input_dim=input_dim),
        "right": normalize_expression(expr["right"], input_dim=input_dim),
    }


def normalize_genome(genome: Sequence[Mapping[str, Any]], *, input_dim: int) -> tuple[dict[str, Any], ...]:
    if not genome:
        raise ValueError("genome must not be empty")
    terms: list[dict[str, Any]] = []
    for i, raw in enumerate(genome):
        if "expr" in raw:
            expr = raw["expr"]
            name = str(raw.get("name", f"term_{i}")).strip() or f"term_{i}"
        else:
            expr = raw
            name = f"term_{i}"
        terms.append({"name": name, "expr": normalize_expression(expr, input_dim=input_dim)})
    return tuple(terms)


def collect_parameter_specs(genome: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> tuple[ParameterSpec, ...]:
    terms: Sequence[Mapping[str, Any]]
    if isinstance(genome, Mapping) and "type" in genome:
        terms = ({"name": "expr", "expr": genome},)
    else:
        terms = tuple(genome)  # type: ignore[arg-type]
    seen: dict[str, ParameterSpec] = {}

    def visit(node: Mapping[str, Any]) -> None:
        kind = str(node["type"])
        if kind == "param":
            spec = ParameterSpec(
                name=str(node["name"]),
                init=float(node.get("init", 1.0)),
                trainable=bool(node.get("trainable", True)),
                lower=None if node.get("lower") is None else float(node["lower"]),
                upper=None if node.get("upper") is None else float(node["upper"]),
            )
            old = seen.get(spec.name)
            if old is None:
                seen[spec.name] = spec
            elif old != spec:
                raise ValueError(f"parameter '{spec.name}' has conflicting specs")
        elif kind == "unary":
            visit(node["arg"])
        elif kind == "binary":
            visit(node["left"])
            visit(node["right"])

    for term in terms:
        visit(term["expr"])
    return tuple(seen[name] for name in sorted(seen))


def parameter_values_from_vector(
    specs: Sequence[ParameterSpec],
    values: Sequence[float] | np.ndarray,
    *,
    base_values: Mapping[str, float] | None = None,
) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    trainable = tuple(spec for spec in specs if bool(spec.trainable))
    if arr.shape[0] != len(trainable):
        raise ValueError(f"parameter vector has length {arr.shape[0]} but expected {len(trainable)}")
    out = {spec.name: float(base_values.get(spec.name, spec.init)) for spec in specs} if base_values else {
        spec.name: float(spec.init) for spec in specs
    }
    for spec, value in zip(trainable, arr):
        out[spec.name] = spec.clamp(float(value))
    return out


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
        kind = str(node["type"])
        if kind == "feature":
            return np.asarray(x[:, int(node["index"])], dtype=float).reshape(-1)
        if kind == "const":
            return np.full((n,), float(node["value"]), dtype=float)
        if kind == "param":
            return np.full((n,), float(params.get(str(node["name"]), node.get("init", 1.0))), dtype=float)
        if kind == "unary":
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
                return np.log(np.abs(z) + float(eps))
            if op == "abs":
                return np.abs(z)
            if op == "sqrt":
                return np.sqrt(np.abs(z) + float(eps))
            raise ValueError(f"unsupported unary op: {op}")
        if kind == "binary":
            left = rec(node["left"])
            right = rec(node["right"])
            op = str(node["op"])
            if op == "add":
                return left + right
            if op == "sub":
                return left - right
            if op == "mul":
                return left * right
            if op == "div":
                denom = np.where(np.abs(right) < float(eps), np.where(right >= 0.0, float(eps), -float(eps)), right)
                return left / denom
            raise ValueError(f"unsupported binary op: {op}")
        raise ValueError(f"unsupported node type: {kind}")

    return np.asarray(rec(expr), dtype=float).reshape(-1)


def expression_to_string(
    expr: Mapping[str, Any],
    *,
    param_values: Mapping[str, float] | None = None,
    precision: int = 6,
    feature_names: Sequence[str] | None = None,
) -> str:
    params = dict(param_values or {})
    names = tuple(feature_names or ())

    def fmt(value: float) -> str:
        return f"{float(value):.{int(precision)}g}"

    def rec(node: Mapping[str, Any]) -> str:
        kind = str(node["type"])
        if kind == "feature":
            index = int(node["index"])
            return str(names[index]) if 0 <= index < len(names) else f"x{index}"
        if kind == "const":
            return fmt(float(node["value"]))
        if kind == "param":
            key = str(node["name"])
            return fmt(float(params[key])) if key in params else key
        if kind == "unary":
            op = str(node["op"])
            arg = rec(node["arg"])
            if op == "identity":
                return f"({arg})"
            if op == "square":
                return f"(({arg})^2)"
            return f"{op}({arg})"
        if kind == "binary":
            left = rec(node["left"])
            right = rec(node["right"])
            symbols = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
            op = str(node["op"])
            if op in symbols:
                return f"(({left}){symbols[op]}({right}))"
            return f"{op}({left},{right})"
        return "<?>"

    return rec(expr)


def genome_to_strings(
    genome: Sequence[Mapping[str, Any]],
    *,
    param_values: Mapping[str, float] | None = None,
    precision: int = 6,
    feature_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    return tuple(
        expression_to_string(term["expr"], param_values=param_values, precision=precision, feature_names=feature_names)
        for term in genome
    )


def expression_complexity(expr: Mapping[str, Any]) -> float:
    kind = str(expr["type"])
    if kind in {"feature", "const", "param"}:
        return 1.0
    if kind == "unary":
        return 1.0 + expression_complexity(expr["arg"])
    if kind == "binary":
        return 1.0 + expression_complexity(expr["left"]) + expression_complexity(expr["right"])
    return 1.0


def default_symbolic_genome(
    input_dim: int,
    *,
    ops: Sequence[str] = ("identity", "square", "sin", "cos"),
) -> tuple[dict[str, Any], ...]:
    if int(input_dim) <= 0:
        raise ValueError("input_dim must be positive")
    terms: list[dict[str, Any]] = []
    for i in range(int(input_dim)):
        base = feature_expr(i)
        for raw_op in ops:
            op = str(raw_op).strip().lower()
            if op == "identity":
                expr = base
                name = f"x{i}"
            else:
                if op not in ALLOWED_UNARY_OPS:
                    raise ValueError(f"unknown default op '{raw_op}'")
                expr = unary_expr(op, base)
                name = f"{op}(x{i})"
            terms.append({"name": name, "expr": expr})
    return tuple(terms)
