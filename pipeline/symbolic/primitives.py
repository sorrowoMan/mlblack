from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from mlblack.models.symbolic import Expression, binary_expr, const_expr, feature_expr, unary_expr


UnaryExprBuilder = Callable[[Mapping[str, Any], Mapping[str, float]], Expression]
UnaryValueBuilder = Callable[[np.ndarray, Mapping[str, float]], np.ndarray]
NameBuilder = Callable[[str, Mapping[str, float]], str]
PairExprBuilder = Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, float]], Expression]
PairValueBuilder = Callable[[np.ndarray, np.ndarray, Mapping[str, float]], np.ndarray]
PairNameBuilder = Callable[[str, str, Mapping[str, float]], str]


@dataclass(frozen=True)
class UnaryPrimitiveSpec:
    key: str
    activation_family: str
    output_family: str
    complexity: float
    build_expr: UnaryExprBuilder
    evaluate_values: UnaryValueBuilder
    build_name: NameBuilder
    initial_enabled: bool = True
    dynamic_enabled: bool = True

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "activation_family": self.activation_family,
            "output_family": self.output_family,
            "complexity": float(self.complexity),
            "initial_enabled": bool(self.initial_enabled),
            "dynamic_enabled": bool(self.dynamic_enabled),
        }


@dataclass(frozen=True)
class PairGrammarRule:
    key: str
    activation_family: str
    output_family: str
    complexity: float
    build_expr: PairExprBuilder
    evaluate_values: PairValueBuilder
    build_name: PairNameBuilder
    initial_enabled: bool = True
    dynamic_enabled: bool = True

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "activation_family": self.activation_family,
            "output_family": self.output_family,
            "complexity": float(self.complexity),
            "initial_enabled": bool(self.initial_enabled),
            "dynamic_enabled": bool(self.dynamic_enabled),
        }


@dataclass(frozen=True)
class PrimitiveRegistry:
    unary_specs: tuple[UnaryPrimitiveSpec, ...]
    pair_rules: tuple[PairGrammarRule, ...]

    def iter_unary_specs(self, *, mode: str = "initial", families: Sequence[str] | None = None) -> tuple[UnaryPrimitiveSpec, ...]:
        want = None if families is None else {str(v) for v in families}
        attr = "initial_enabled" if str(mode) == "initial" else "dynamic_enabled"
        return tuple(spec for spec in self.unary_specs if bool(getattr(spec, attr)) and (want is None or spec.activation_family in want))

    def iter_pair_rules(self, *, mode: str = "initial", families: Sequence[str] | None = None) -> tuple[PairGrammarRule, ...]:
        want = None if families is None else {str(v) for v in families}
        attr = "initial_enabled" if str(mode) == "initial" else "dynamic_enabled"
        return tuple(rule for rule in self.pair_rules if bool(getattr(rule, attr)) and (want is None or rule.activation_family in want))

    def describe(self) -> dict[str, Any]:
        return {
            "unary_specs": [spec.describe() for spec in self.unary_specs],
            "pair_rules": [rule.describe() for rule in self.pair_rules],
        }


def square_expr(arg: Mapping[str, Any]) -> Expression:
    return unary_expr("square", arg)


def cube_expr(arg: Mapping[str, Any]) -> Expression:
    return binary_expr("mul", square_expr(arg), arg)


def pow4_expr(arg: Mapping[str, Any]) -> Expression:
    return square_expr(square_expr(arg))


def pow5_expr(arg: Mapping[str, Any]) -> Expression:
    return binary_expr("mul", pow4_expr(arg), arg)


def pow6_expr(arg: Mapping[str, Any]) -> Expression:
    return binary_expr("mul", square_expr(arg), pow4_expr(arg))


def log1p_abs_expr(arg: Mapping[str, Any]) -> Expression:
    return unary_expr("log", binary_expr("add", unary_expr("abs", arg), const_expr(1.0)))


def log1p_square_expr(arg: Mapping[str, Any]) -> Expression:
    return unary_expr("log", binary_expr("add", square_expr(arg), const_expr(1.0)))


def sqrt1p_abs_expr(arg: Mapping[str, Any]) -> Expression:
    return unary_expr("sqrt", binary_expr("add", unary_expr("abs", arg), const_expr(1.0)))


def reciprocal_safe_expr(arg: Mapping[str, Any], *, eps: float) -> Expression:
    return binary_expr("div", const_expr(1.0), binary_expr("add", unary_expr("abs", arg), const_expr(float(eps))))


def exp_clip_expr(arg: Mapping[str, Any], *, scale: float) -> Expression:
    scale_eff = float(max(1.0, scale))
    return unary_expr("exp", binary_expr("mul", const_expr(1.0 / scale_eff), arg))


def softsign_expr(arg: Mapping[str, Any], *, scale: float) -> Expression:
    scale_eff = float(max(1.0, scale))
    return binary_expr("div", dict(arg), binary_expr("add", unary_expr("abs", arg), const_expr(scale_eff)))


def inverse_quadratic_expr(arg: Mapping[str, Any], *, scale: float) -> Expression:
    scale_eff = float(max(1.0, scale))
    scale2 = float(scale_eff * scale_eff)
    return binary_expr("div", const_expr(scale2), binary_expr("add", square_expr(arg), const_expr(scale2)))


def gaussian_rbf_expr(arg: Mapping[str, Any], *, scale: float) -> Expression:
    scale_eff = float(max(1.0, scale))
    return unary_expr("exp", binary_expr("mul", const_expr(-1.0 / (scale_eff * scale_eff)), square_expr(arg)))


def laplace_rbf_expr(arg: Mapping[str, Any], *, scale: float) -> Expression:
    scale_eff = float(max(1.0, scale))
    return unary_expr("exp", binary_expr("mul", const_expr(-1.0 / scale_eff), unary_expr("abs", arg)))


def softplus_clip_expr(arg: Mapping[str, Any], *, scale: float) -> Expression:
    scale_eff = float(max(1.0, scale))
    scaled = binary_expr("mul", const_expr(1.0 / scale_eff), arg)
    return unary_expr("log", binary_expr("add", const_expr(1.0), unary_expr("exp", scaled)))


def _scale(params: Mapping[str, float], default: float = 8.0) -> float:
    return float(max(1.0, params.get("scale", default)))


def _eps(params: Mapping[str, float], default: float = 1e-3) -> float:
    return float(max(1e-8, params.get("eps", default)))


def _unary_specs() -> tuple[UnaryPrimitiveSpec, ...]:
    return (
        UnaryPrimitiveSpec("identity", "base", "identity", 1.0, lambda a, _p: dict(a), lambda z, _p: np.asarray(z, dtype=float), lambda l, _p: str(l)),
        UnaryPrimitiveSpec("square", "poly", "unary_square", 2.0, lambda a, _p: square_expr(a), lambda z, _p: np.asarray(z * z, dtype=float), lambda l, _p: f"square({l})"),
        UnaryPrimitiveSpec("cube", "poly", "unary_cube", 2.4, lambda a, _p: cube_expr(a), lambda z, _p: np.asarray(z * z * z, dtype=float), lambda l, _p: f"cube({l})", initial_enabled=False),
        UnaryPrimitiveSpec("pow4", "poly", "unary_pow4", 2.8, lambda a, _p: pow4_expr(a), lambda z, _p: np.asarray((z * z) * (z * z), dtype=float), lambda l, _p: f"pow4({l})", initial_enabled=False),
        UnaryPrimitiveSpec("pow5", "poly", "unary_pow5", 3.2, lambda a, _p: pow5_expr(a), lambda z, _p: np.asarray(((z * z) * (z * z)) * z, dtype=float), lambda l, _p: f"pow5({l})", initial_enabled=False),
        UnaryPrimitiveSpec("pow6", "poly", "unary_pow6", 3.4, lambda a, _p: pow6_expr(a), lambda z, _p: np.asarray((z * z) * ((z * z) * (z * z)), dtype=float), lambda l, _p: f"pow6({l})", initial_enabled=False),
        UnaryPrimitiveSpec("sin", "trig", "unary_sin", 2.0, lambda a, _p: unary_expr("sin", a), lambda z, _p: np.asarray(np.sin(z), dtype=float), lambda l, _p: f"sin({l})"),
        UnaryPrimitiveSpec("cos", "trig", "unary_cos", 2.0, lambda a, _p: unary_expr("cos", a), lambda z, _p: np.asarray(np.cos(z), dtype=float), lambda l, _p: f"cos({l})"),
        UnaryPrimitiveSpec("tanh", "bounded", "unary_tanh", 2.0, lambda a, _p: unary_expr("tanh", a), lambda z, _p: np.asarray(np.tanh(z), dtype=float), lambda l, _p: f"tanh({l})"),
        UnaryPrimitiveSpec("softsign", "saturation", "unary_softsign", 2.7, lambda a, p: softsign_expr(a, scale=_scale(p)), lambda z, p: np.asarray(z / (np.abs(z) + _scale(p)), dtype=float), lambda l, p: f"softsign_k{_scale(p):g}({l})", initial_enabled=False),
        UnaryPrimitiveSpec("inverse_quadratic", "saturation", "unary_inverse_quadratic", 3.0, lambda a, p: inverse_quadratic_expr(a, scale=_scale(p)), lambda z, p: np.asarray((_scale(p) ** 2) / ((z * z) + (_scale(p) ** 2)), dtype=float), lambda l, p: f"inverse_quad_k{_scale(p):g}({l})", initial_enabled=False),
        UnaryPrimitiveSpec("log1p_abs", "safe_log", "unary_log1p_abs", 2.5, lambda a, _p: log1p_abs_expr(a), lambda z, _p: np.asarray(np.log1p(np.abs(z)), dtype=float), lambda l, _p: f"log1p_abs({l})"),
        UnaryPrimitiveSpec("log1p_square", "safe_log", "unary_log1p_square", 2.8, lambda a, _p: log1p_square_expr(a), lambda z, _p: np.asarray(np.log1p(z * z), dtype=float), lambda l, _p: f"log1p_square({l})", initial_enabled=False),
        UnaryPrimitiveSpec("sqrt1p_abs", "safe_log", "unary_sqrt1p_abs", 2.7, lambda a, _p: sqrt1p_abs_expr(a), lambda z, _p: np.asarray(np.sqrt(np.abs(z) + 1.0), dtype=float), lambda l, _p: f"sqrt1p_abs({l})", initial_enabled=False),
        UnaryPrimitiveSpec("exp_clip", "safe_exp", "unary_exp_clip", 2.5, lambda a, p: exp_clip_expr(a, scale=_scale(p)), lambda z, p: np.asarray(np.exp(np.clip(z / _scale(p), -30.0, 30.0)), dtype=float), lambda l, p: f"exp_clip_k{_scale(p):g}({l})"),
        UnaryPrimitiveSpec("softplus_clip", "safe_exp", "unary_softplus_clip", 3.2, lambda a, p: softplus_clip_expr(a, scale=_scale(p)), lambda z, p: np.asarray(np.log1p(np.exp(np.clip(z / _scale(p), -30.0, 30.0))), dtype=float), lambda l, p: f"softplus_k{_scale(p):g}({l})", initial_enabled=False),
        UnaryPrimitiveSpec("reciprocal_safe", "safe_ratio", "unary_reciprocal_safe", 2.8, lambda a, p: reciprocal_safe_expr(a, eps=_eps(p)), lambda z, p: np.asarray(1.0 / (np.abs(z) + _eps(p)), dtype=float), lambda l, _p: f"reciprocal_safe({l})"),
        UnaryPrimitiveSpec("gaussian_rbf", "radial", "unary_gaussian_rbf", 3.0, lambda a, p: gaussian_rbf_expr(a, scale=_scale(p)), lambda z, p: np.asarray(np.exp(np.clip(-((z / _scale(p)) ** 2), -30.0, 30.0)), dtype=float), lambda l, p: f"rbf_gauss_k{_scale(p):g}({l})", initial_enabled=False),
        UnaryPrimitiveSpec("laplace_rbf", "radial", "unary_laplace_rbf", 3.0, lambda a, p: laplace_rbf_expr(a, scale=_scale(p)), lambda z, p: np.asarray(np.exp(np.clip(-np.abs(z) / _scale(p), -30.0, 30.0)), dtype=float), lambda l, p: f"rbf_laplace_k{_scale(p):g}({l})", initial_enabled=False),
    )


def _pair_rules() -> tuple[PairGrammarRule, ...]:
    return (
        PairGrammarRule("mul", "interaction_basic", "interaction", 3.0, lambda l, r, _p: binary_expr("mul", l, r), lambda a, b, _p: np.asarray(a * b, dtype=float), lambda l, r, _p: f"{l}*{r}"),
        PairGrammarRule("add", "linear_combo", "pair_add", 2.5, lambda l, r, _p: binary_expr("add", l, r), lambda a, b, _p: np.asarray(a + b, dtype=float), lambda l, r, _p: f"({l})+({r})", initial_enabled=False),
        PairGrammarRule("safe_ratio", "interaction_ratio", "interaction_ratio_safe", 3.6, lambda l, r, p: binary_expr("div", l, binary_expr("add", unary_expr("abs", r), const_expr(_eps(p)))), lambda a, b, p: np.asarray(a / (np.abs(b) + _eps(p)), dtype=float), lambda l, r, _p: f"{l}/safe({r})"),
        PairGrammarRule("safe_ratio_reverse", "interaction_ratio", "interaction_ratio_safe", 3.6, lambda l, r, p: binary_expr("div", r, binary_expr("add", unary_expr("abs", l), const_expr(_eps(p)))), lambda a, b, p: np.asarray(b / (np.abs(a) + _eps(p)), dtype=float), lambda l, r, _p: f"{r}/safe({l})"),
        PairGrammarRule("square_left_mul", "interaction_poly", "interaction_poly2", 3.8, lambda l, r, _p: binary_expr("mul", square_expr(l), r), lambda a, b, _p: np.asarray((a * a) * b, dtype=float), lambda l, r, _p: f"square({l})*{r}", initial_enabled=False),
        PairGrammarRule("square_right_mul", "interaction_poly", "interaction_poly2", 3.8, lambda l, r, _p: binary_expr("mul", l, square_expr(r)), lambda a, b, _p: np.asarray(a * (b * b), dtype=float), lambda l, r, _p: f"{l}*square({r})", initial_enabled=False),
        PairGrammarRule("mul_over_energy", "interaction_rational", "interaction_rational", 4.2, lambda l, r, p: binary_expr("div", binary_expr("mul", l, r), binary_expr("add", binary_expr("add", square_expr(l), square_expr(r)), const_expr(_scale(p) ** 2))), lambda a, b, p: np.asarray((a * b) / ((a * a) + (b * b) + (_scale(p) ** 2)), dtype=float), lambda l, r, p: f"energy_norm_k{_scale(p):g}({l},{r})", initial_enabled=False),
        PairGrammarRule("tanh_right_mul", "interaction_compose", "interaction_compose", 4.0, lambda l, r, _p: binary_expr("mul", l, unary_expr("tanh", r)), lambda a, b, _p: np.asarray(a * np.tanh(b), dtype=float), lambda l, r, _p: f"{l}*tanh({r})", initial_enabled=False),
        PairGrammarRule("softsign_right_mul", "interaction_saturation", "interaction_saturation", 4.1, lambda l, r, p: binary_expr("mul", l, softsign_expr(r, scale=_scale(p))), lambda a, b, p: np.asarray(a * (b / (np.abs(b) + _scale(p))), dtype=float), lambda l, r, p: f"{l}*softsign_k{_scale(p):g}({r})", initial_enabled=False),
        PairGrammarRule("inverse_quad_right_mul", "interaction_saturation", "interaction_saturation", 4.3, lambda l, r, p: binary_expr("mul", l, inverse_quadratic_expr(r, scale=_scale(p))), lambda a, b, p: np.asarray(a * ((_scale(p) ** 2) / ((b * b) + (_scale(p) ** 2))), dtype=float), lambda l, r, p: f"{l}*inverse_quad_k{_scale(p):g}({r})", initial_enabled=False),
        PairGrammarRule("log_right_mul", "interaction_compose", "interaction_compose", 4.4, lambda l, r, _p: binary_expr("mul", l, log1p_abs_expr(r)), lambda a, b, _p: np.asarray(a * np.log1p(np.abs(b)), dtype=float), lambda l, r, _p: f"{l}*log1p_abs({r})", initial_enabled=False),
        PairGrammarRule("exp_right_mul", "interaction_compose", "interaction_compose", 4.6, lambda l, r, p: binary_expr("mul", l, exp_clip_expr(r, scale=_scale(p))), lambda a, b, p: np.asarray(a * np.exp(np.clip(b / _scale(p), -30.0, 30.0)), dtype=float), lambda l, r, _p: f"{l}*exp_clip({r})", initial_enabled=False),
        PairGrammarRule("gaussian_right_mul", "interaction_radial", "interaction_radial", 4.4, lambda l, r, p: binary_expr("mul", l, gaussian_rbf_expr(r, scale=_scale(p))), lambda a, b, p: np.asarray(a * np.exp(np.clip(-((b / _scale(p)) ** 2), -30.0, 30.0)), dtype=float), lambda l, r, p: f"{l}*rbf_gauss_k{_scale(p):g}({r})", initial_enabled=False),
        PairGrammarRule("softsign_product", "interaction_rational", "interaction_rational", 4.5, lambda l, r, p: softsign_expr(binary_expr("mul", l, r), scale=_scale(p)), lambda a, b, p: np.asarray((a * b) / (np.abs(a * b) + _scale(p)), dtype=float), lambda l, r, p: f"softsign_prod_k{_scale(p):g}({l},{r})", initial_enabled=False),
        PairGrammarRule("sin_product", "interaction_compose", "interaction_dynamic", 4.0, lambda l, r, _p: unary_expr("sin", binary_expr("mul", l, r)), lambda a, b, _p: np.asarray(np.sin(a * b), dtype=float), lambda l, r, _p: f"sin({l}*{r})", initial_enabled=False),
        PairGrammarRule("log_abs_product", "interaction_compose", "interaction_compose", 4.6, lambda l, r, _p: log1p_abs_expr(binary_expr("mul", l, r)), lambda a, b, _p: np.asarray(np.log1p(np.abs(a * b)), dtype=float), lambda l, r, _p: f"log1p_abs({l}*{r})", initial_enabled=False),
        PairGrammarRule("sqrt_abs_product", "interaction_compose", "interaction_compose", 4.6, lambda l, r, _p: sqrt1p_abs_expr(binary_expr("mul", l, r)), lambda a, b, _p: np.asarray(np.sqrt(np.abs(a * b) + 1.0), dtype=float), lambda l, r, _p: f"sqrt1p_abs({l}*{r})", initial_enabled=False),
        PairGrammarRule("tanh_tanh_product", "interaction_compose", "interaction_compose", 4.4, lambda l, r, _p: binary_expr("mul", unary_expr("tanh", l), unary_expr("tanh", r)), lambda a, b, _p: np.asarray(np.tanh(a) * np.tanh(b), dtype=float), lambda l, r, _p: f"tanh({l})*tanh({r})", initial_enabled=False),
    )


def default_primitive_registry() -> PrimitiveRegistry:
    return PrimitiveRegistry(unary_specs=_unary_specs(), pair_rules=_pair_rules())


def seed_feature_terms(X: np.ndarray, feature_names: Sequence[str] | None = None) -> tuple[dict[str, Any], ...]:
    x = np.asarray(X, dtype=float)
    if x.ndim != 2:
        raise ValueError("X must be 2D")
    names = tuple(feature_names or tuple(f"x{i}" for i in range(x.shape[1])))
    return tuple({"name": str(names[i]), "expr": feature_expr(i), "values": x[:, i], "features": (i,)} for i in range(x.shape[1]))


__all__ = [
    "PairGrammarRule",
    "PrimitiveRegistry",
    "UnaryPrimitiveSpec",
    "binary_expr",
    "const_expr",
    "cube_expr",
    "default_primitive_registry",
    "exp_clip_expr",
    "feature_expr",
    "gaussian_rbf_expr",
    "inverse_quadratic_expr",
    "laplace_rbf_expr",
    "log1p_abs_expr",
    "log1p_square_expr",
    "pow4_expr",
    "pow5_expr",
    "pow6_expr",
    "reciprocal_safe_expr",
    "seed_feature_terms",
    "softplus_clip_expr",
    "softsign_expr",
    "sqrt1p_abs_expr",
    "square_expr",
    "unary_expr",
]
