from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


Expr = dict[str, Any]
UnaryExprBuilder = Callable[[Mapping[str, Any], Mapping[str, float]], Expr]
UnaryValueBuilder = Callable[[np.ndarray, Mapping[str, float]], np.ndarray]
NameBuilder = Callable[[str, Mapping[str, float]], str]
PairExprBuilder = Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, float]], Expr]
PairValueBuilder = Callable[[np.ndarray, np.ndarray, Mapping[str, float]], np.ndarray]
PairNameBuilder = Callable[[str, str, Mapping[str, float]], str]


def feature_expr(index: int) -> Expr:
    return {"type": "feature", "index": int(index)}


def const_expr(value: float) -> Expr:
    return {"type": "const", "value": float(value)}


def unary_expr(op: str, arg: Mapping[str, Any]) -> Expr:
    return {"type": "unary", "op": str(op), "arg": dict(arg)}


def binary_expr(op: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> Expr:
    return {"type": "binary", "op": str(op), "left": dict(left), "right": dict(right)}


def square_expr(arg: Mapping[str, Any]) -> Expr:
    return unary_expr("square", arg)


def cube_expr(arg: Mapping[str, Any]) -> Expr:
    return binary_expr("mul", square_expr(arg), arg)


def pow4_expr(arg: Mapping[str, Any]) -> Expr:
    return square_expr(square_expr(arg))


def pow5_expr(arg: Mapping[str, Any]) -> Expr:
    return binary_expr("mul", pow4_expr(arg), arg)


def pow6_expr(arg: Mapping[str, Any]) -> Expr:
    return binary_expr("mul", square_expr(arg), pow4_expr(arg))


def log1p_abs_expr(arg: Mapping[str, Any]) -> Expr:
    return unary_expr("log", binary_expr("add", unary_expr("abs", arg), const_expr(1.0)))


def log1p_square_expr(arg: Mapping[str, Any]) -> Expr:
    return unary_expr("log", binary_expr("add", square_expr(arg), const_expr(1.0)))


def sqrt1p_abs_expr(arg: Mapping[str, Any]) -> Expr:
    return unary_expr("sqrt", binary_expr("add", unary_expr("abs", arg), const_expr(1.0)))


def reciprocal_safe_expr(arg: Mapping[str, Any], *, eps: float) -> Expr:
    return binary_expr("div", const_expr(1.0), binary_expr("add", unary_expr("abs", arg), const_expr(float(eps))))


def exp_clip_expr(arg: Mapping[str, Any], *, scale: float) -> Expr:
    scale_eff = float(max(1.0, scale))
    return unary_expr("exp", binary_expr("mul", const_expr(1.0 / scale_eff), arg))


def softsign_expr(arg: Mapping[str, Any], *, scale: float) -> Expr:
    scale_eff = float(max(1.0, scale))
    return binary_expr("div", dict(arg), binary_expr("add", unary_expr("abs", arg), const_expr(scale_eff)))


def inverse_quadratic_expr(arg: Mapping[str, Any], *, scale: float) -> Expr:
    scale_eff = float(max(1.0, scale))
    scale2 = float(scale_eff * scale_eff)
    return binary_expr("div", const_expr(scale2), binary_expr("add", square_expr(arg), const_expr(scale2)))


def gaussian_rbf_expr(arg: Mapping[str, Any], *, scale: float) -> Expr:
    scale_eff = float(max(1.0, scale))
    coeff = -1.0 / float(scale_eff * scale_eff)
    return unary_expr("exp", binary_expr("mul", const_expr(coeff), square_expr(arg)))


def laplace_rbf_expr(arg: Mapping[str, Any], *, scale: float) -> Expr:
    scale_eff = float(max(1.0, scale))
    coeff = -1.0 / float(scale_eff)
    return unary_expr("exp", binary_expr("mul", const_expr(coeff), unary_expr("abs", arg)))


def softplus_clip_expr(arg: Mapping[str, Any], *, scale: float) -> Expr:
    scale_eff = float(max(1.0, scale))
    scaled = binary_expr("mul", const_expr(1.0 / scale_eff), arg)
    return unary_expr("log", binary_expr("add", const_expr(1.0), unary_expr("exp", scaled)))


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


@dataclass(frozen=True)
class PrimitiveRegistry:
    unary_specs: tuple[UnaryPrimitiveSpec, ...]
    pair_rules: tuple[PairGrammarRule, ...]

    def iter_unary_specs(
        self,
        *,
        mode: str,
        families: Sequence[str] | None = None,
    ) -> tuple[UnaryPrimitiveSpec, ...]:
        want = None if families is None else {str(v) for v in families}
        enabled_attr = "initial_enabled" if str(mode) == "initial" else "dynamic_enabled"
        out: list[UnaryPrimitiveSpec] = []
        for spec in self.unary_specs:
            if not bool(getattr(spec, enabled_attr)):
                continue
            if want is not None and str(spec.activation_family) not in want:
                continue
            out.append(spec)
        return tuple(out)

    def iter_pair_rules(
        self,
        *,
        mode: str,
        families: Sequence[str] | None = None,
    ) -> tuple[PairGrammarRule, ...]:
        want = None if families is None else {str(v) for v in families}
        enabled_attr = "initial_enabled" if str(mode) == "initial" else "dynamic_enabled"
        out: list[PairGrammarRule] = []
        for rule in self.pair_rules:
            if not bool(getattr(rule, enabled_attr)):
                continue
            if want is not None and str(rule.activation_family) not in want:
                continue
            out.append(rule)
        return tuple(out)


def _build_default_unary_specs() -> tuple[UnaryPrimitiveSpec, ...]:
    return (
        UnaryPrimitiveSpec(
            key="square",
            activation_family="poly",
            output_family="unary_square",
            complexity=2.0,
            build_expr=lambda arg, _params: square_expr(arg),
            evaluate_values=lambda z, _params: np.asarray(z * z, dtype=float),
            build_name=lambda label, _params: f"square({label})",
        ),
        UnaryPrimitiveSpec(
            key="cube",
            activation_family="poly",
            output_family="unary_cube",
            complexity=2.4,
            build_expr=lambda arg, _params: cube_expr(arg),
            evaluate_values=lambda z, _params: np.asarray(z * z * z, dtype=float),
            build_name=lambda label, _params: f"cube({label})",
            initial_enabled=False,
        ),
        UnaryPrimitiveSpec(
            key="pow4",
            activation_family="poly",
            output_family="unary_pow4",
            complexity=2.8,
            build_expr=lambda arg, _params: pow4_expr(arg),
            evaluate_values=lambda z, _params: np.asarray((z * z) * (z * z), dtype=float),
            build_name=lambda label, _params: f"pow4({label})",
            initial_enabled=False,
        ),
        UnaryPrimitiveSpec(
            key="pow5",
            activation_family="poly",
            output_family="unary_pow5",
            complexity=3.2,
            build_expr=lambda arg, _params: pow5_expr(arg),
            evaluate_values=lambda z, _params: np.asarray(((z * z) * (z * z)) * z, dtype=float),
            build_name=lambda label, _params: f"pow5({label})",
            initial_enabled=False,
        ),
        UnaryPrimitiveSpec(
            key="pow6",
            activation_family="poly",
            output_family="unary_pow6",
            complexity=3.4,
            build_expr=lambda arg, _params: pow6_expr(arg),
            evaluate_values=lambda z, _params: np.asarray((z * z) * ((z * z) * (z * z)), dtype=float),
            build_name=lambda label, _params: f"pow6({label})",
            initial_enabled=False,
        ),
        UnaryPrimitiveSpec(
            key="sin",
            activation_family="trig",
            output_family="unary_sin",
            complexity=2.0,
            build_expr=lambda arg, _params: unary_expr("sin", arg),
            evaluate_values=lambda z, _params: np.asarray(np.sin(z), dtype=float),
            build_name=lambda label, _params: f"sin({label})",
        ),
        UnaryPrimitiveSpec(
            key="cos",
            activation_family="trig",
            output_family="unary_cos",
            complexity=2.0,
            build_expr=lambda arg, _params: unary_expr("cos", arg),
            evaluate_values=lambda z, _params: np.asarray(np.cos(z), dtype=float),
            build_name=lambda label, _params: f"cos({label})",
        ),
        UnaryPrimitiveSpec(
            key="tanh",
            activation_family="bounded",
            output_family="unary_tanh",
            complexity=2.0,
            build_expr=lambda arg, _params: unary_expr("tanh", arg),
            evaluate_values=lambda z, _params: np.asarray(np.tanh(z), dtype=float),
            build_name=lambda label, _params: f"tanh({label})",
        ),
        UnaryPrimitiveSpec(
            key="softsign",
            activation_family="saturation",
            output_family="unary_softsign",
            complexity=2.7,
            build_expr=lambda arg, params: softsign_expr(arg, scale=float(params.get("scale", 8.0))),
            evaluate_values=lambda z, params: np.asarray(
                z / (np.abs(z) + float(max(1.0, params.get("scale", 8.0)))),
                dtype=float,
            ),
            build_name=lambda label, params: f"softsign_k{float(max(1.0, params.get('scale', 8.0))):g}({label})",
            initial_enabled=False,
        ),
        UnaryPrimitiveSpec(
            key="inverse_quadratic",
            activation_family="saturation",
            output_family="unary_inverse_quadratic",
            complexity=3.0,
            build_expr=lambda arg, params: inverse_quadratic_expr(arg, scale=float(params.get("scale", 8.0))),
            evaluate_values=lambda z, params: np.asarray(
                (float(max(1.0, params.get("scale", 8.0))) ** 2)
                / ((z * z) + (float(max(1.0, params.get("scale", 8.0))) ** 2)),
                dtype=float,
            ),
            build_name=lambda label, params: f"inverse_quad_k{float(max(1.0, params.get('scale', 8.0))):g}({label})",
            initial_enabled=False,
        ),
        UnaryPrimitiveSpec(
            key="log1p_abs",
            activation_family="safe_log",
            output_family="unary_log1p_abs",
            complexity=2.5,
            build_expr=lambda arg, _params: log1p_abs_expr(arg),
            evaluate_values=lambda z, _params: np.asarray(np.log1p(np.abs(z)), dtype=float),
            build_name=lambda label, _params: f"log1p_abs({label})",
        ),
        UnaryPrimitiveSpec(
            key="log1p_square",
            activation_family="safe_log",
            output_family="unary_log1p_square",
            complexity=2.8,
            build_expr=lambda arg, _params: log1p_square_expr(arg),
            evaluate_values=lambda z, _params: np.asarray(np.log1p(z * z), dtype=float),
            build_name=lambda label, _params: f"log1p_square({label})",
            initial_enabled=False,
        ),
        UnaryPrimitiveSpec(
            key="sqrt1p_abs",
            activation_family="safe_log",
            output_family="unary_sqrt1p_abs",
            complexity=2.7,
            build_expr=lambda arg, _params: sqrt1p_abs_expr(arg),
            evaluate_values=lambda z, _params: np.asarray(np.sqrt(np.abs(z) + 1.0), dtype=float),
            build_name=lambda label, _params: f"sqrt1p_abs({label})",
            initial_enabled=False,
        ),
        UnaryPrimitiveSpec(
            key="exp_clip",
            activation_family="safe_exp",
            output_family="unary_exp_clip",
            complexity=2.5,
            build_expr=lambda arg, params: exp_clip_expr(arg, scale=float(params.get("scale", 8.0))),
            evaluate_values=lambda z, params: np.asarray(
                np.exp(np.clip(z / float(max(1.0, params.get("scale", 8.0))), -30.0, 30.0)),
                dtype=float,
            ),
            build_name=lambda label, params: f"exp_clip_k{float(max(1.0, params.get('scale', 8.0))):g}({label})",
        ),
        UnaryPrimitiveSpec(
            key="softplus_clip",
            activation_family="safe_exp",
            output_family="unary_softplus_clip",
            complexity=3.2,
            build_expr=lambda arg, params: softplus_clip_expr(arg, scale=float(params.get("scale", 8.0))),
            evaluate_values=lambda z, params: np.asarray(
                np.log1p(
                    np.exp(
                        np.clip(
                            z / float(max(1.0, params.get("scale", 8.0))),
                            -30.0,
                            30.0,
                        )
                    )
                ),
                dtype=float,
            ),
            build_name=lambda label, params: f"softplus_k{float(max(1.0, params.get('scale', 8.0))):g}({label})",
            initial_enabled=False,
        ),
        UnaryPrimitiveSpec(
            key="reciprocal_safe",
            activation_family="safe_ratio",
            output_family="unary_reciprocal_safe",
            complexity=2.8,
            build_expr=lambda arg, params: reciprocal_safe_expr(arg, eps=float(params.get("eps", 1e-3))),
            evaluate_values=lambda z, params: np.asarray(
                1.0 / (np.abs(z) + float(max(1e-8, params.get("eps", 1e-3)))),
                dtype=float,
            ),
            build_name=lambda label, _params: f"reciprocal_safe({label})",
        ),
        UnaryPrimitiveSpec(
            key="gaussian_rbf",
            activation_family="radial",
            output_family="unary_gaussian_rbf",
            complexity=3.0,
            build_expr=lambda arg, params: gaussian_rbf_expr(arg, scale=float(params.get("scale", 8.0))),
            evaluate_values=lambda z, params: np.asarray(
                np.exp(
                    np.clip(
                        -((z / float(max(1.0, params.get("scale", 8.0)))) ** 2),
                        -30.0,
                        30.0,
                    )
                ),
                dtype=float,
            ),
            build_name=lambda label, params: f"rbf_gauss_k{float(max(1.0, params.get('scale', 8.0))):g}({label})",
            initial_enabled=False,
        ),
        UnaryPrimitiveSpec(
            key="laplace_rbf",
            activation_family="radial",
            output_family="unary_laplace_rbf",
            complexity=3.0,
            build_expr=lambda arg, params: laplace_rbf_expr(arg, scale=float(params.get("scale", 8.0))),
            evaluate_values=lambda z, params: np.asarray(
                np.exp(
                    np.clip(
                        -np.abs(z) / float(max(1.0, params.get("scale", 8.0))),
                        -30.0,
                        30.0,
                    )
                ),
                dtype=float,
            ),
            build_name=lambda label, params: f"rbf_laplace_k{float(max(1.0, params.get('scale', 8.0))):g}({label})",
            initial_enabled=False,
        ),
    )


def _build_default_pair_rules() -> tuple[PairGrammarRule, ...]:
    return (
        PairGrammarRule(
            key="mul",
            activation_family="interaction_basic",
            output_family="interaction",
            complexity=3.0,
            build_expr=lambda left, right, _params: binary_expr("mul", left, right),
            evaluate_values=lambda left_v, right_v, _params: np.asarray(left_v * right_v, dtype=float),
            build_name=lambda left_label, right_label, _params: f"{left_label}*{right_label}",
        ),
        PairGrammarRule(
            key="safe_ratio",
            activation_family="interaction_ratio",
            output_family="interaction_ratio_safe",
            complexity=3.6,
            build_expr=lambda left, right, params: binary_expr(
                "div",
                left,
                binary_expr("add", unary_expr("abs", right), const_expr(float(params.get("eps", 1e-3)))),
            ),
            evaluate_values=lambda left_v, right_v, params: np.asarray(
                left_v / (np.abs(right_v) + float(max(1e-8, params.get("eps", 1e-3)))),
                dtype=float,
            ),
            build_name=lambda left_label, right_label, _params: f"{left_label}/safe({right_label})",
        ),
        PairGrammarRule(
            key="safe_ratio_reverse",
            activation_family="interaction_ratio",
            output_family="interaction_ratio_safe",
            complexity=3.6,
            build_expr=lambda left, right, params: binary_expr(
                "div",
                right,
                binary_expr("add", unary_expr("abs", left), const_expr(float(params.get("eps", 1e-3)))),
            ),
            evaluate_values=lambda left_v, right_v, params: np.asarray(
                right_v / (np.abs(left_v) + float(max(1e-8, params.get("eps", 1e-3)))),
                dtype=float,
            ),
            build_name=lambda left_label, right_label, _params: f"{right_label}/safe({left_label})",
        ),
        PairGrammarRule(
            key="square_left_mul",
            activation_family="interaction_poly",
            output_family="interaction_poly2",
            complexity=3.8,
            build_expr=lambda left, right, _params: binary_expr("mul", square_expr(left), right),
            evaluate_values=lambda left_v, right_v, _params: np.asarray((left_v * left_v) * right_v, dtype=float),
            build_name=lambda left_label, right_label, _params: f"square({left_label})*{right_label}",
            initial_enabled=False,
        ),
        PairGrammarRule(
            key="square_right_mul",
            activation_family="interaction_poly",
            output_family="interaction_poly2",
            complexity=3.8,
            build_expr=lambda left, right, _params: binary_expr("mul", left, square_expr(right)),
            evaluate_values=lambda left_v, right_v, _params: np.asarray(left_v * (right_v * right_v), dtype=float),
            build_name=lambda left_label, right_label, _params: f"{left_label}*square({right_label})",
            initial_enabled=False,
        ),
        PairGrammarRule(
            key="mul_over_energy",
            activation_family="interaction_rational",
            output_family="interaction_rational",
            complexity=4.2,
            build_expr=lambda left, right, params: binary_expr(
                "div",
                binary_expr("mul", left, right),
                binary_expr(
                    "add",
                    binary_expr("add", square_expr(left), square_expr(right)),
                    const_expr(float(max(1.0, params.get("scale", 8.0))) ** 2),
                ),
            ),
            evaluate_values=lambda left_v, right_v, params: np.asarray(
                (left_v * right_v)
                / (
                    (left_v * left_v)
                    + (right_v * right_v)
                    + (float(max(1.0, params.get("scale", 8.0))) ** 2)
                ),
                dtype=float,
            ),
            build_name=lambda left_label, right_label, params: (
                f"energy_norm_k{float(max(1.0, params.get('scale', 8.0))):g}({left_label},{right_label})"
            ),
            initial_enabled=False,
        ),
        PairGrammarRule(
            key="tanh_right_mul",
            activation_family="interaction_compose",
            output_family="interaction_compose",
            complexity=4.0,
            build_expr=lambda left, right, _params: binary_expr("mul", left, unary_expr("tanh", right)),
            evaluate_values=lambda left_v, right_v, _params: np.asarray(left_v * np.tanh(right_v), dtype=float),
            build_name=lambda left_label, right_label, _params: f"{left_label}*tanh({right_label})",
            initial_enabled=False,
        ),
        PairGrammarRule(
            key="softsign_right_mul",
            activation_family="interaction_saturation",
            output_family="interaction_saturation",
            complexity=4.1,
            build_expr=lambda left, right, params: binary_expr(
                "mul",
                left,
                softsign_expr(right, scale=float(params.get("scale", 8.0))),
            ),
            evaluate_values=lambda left_v, right_v, params: np.asarray(
                left_v * (right_v / (np.abs(right_v) + float(max(1.0, params.get("scale", 8.0))))),
                dtype=float,
            ),
            build_name=lambda left_label, right_label, params: (
                f"{left_label}*softsign_k{float(max(1.0, params.get('scale', 8.0))):g}({right_label})"
            ),
            initial_enabled=False,
        ),
        PairGrammarRule(
            key="inverse_quad_right_mul",
            activation_family="interaction_saturation",
            output_family="interaction_saturation",
            complexity=4.3,
            build_expr=lambda left, right, params: binary_expr(
                "mul",
                left,
                inverse_quadratic_expr(right, scale=float(params.get("scale", 8.0))),
            ),
            evaluate_values=lambda left_v, right_v, params: np.asarray(
                left_v
                * (
                    (float(max(1.0, params.get("scale", 8.0))) ** 2)
                    / ((right_v * right_v) + (float(max(1.0, params.get("scale", 8.0))) ** 2))
                ),
                dtype=float,
            ),
            build_name=lambda left_label, right_label, params: (
                f"{left_label}*inverse_quad_k{float(max(1.0, params.get('scale', 8.0))):g}({right_label})"
            ),
            initial_enabled=False,
        ),
        PairGrammarRule(
            key="log_right_mul",
            activation_family="interaction_compose",
            output_family="interaction_compose",
            complexity=4.4,
            build_expr=lambda left, right, _params: binary_expr("mul", left, log1p_abs_expr(right)),
            evaluate_values=lambda left_v, right_v, _params: np.asarray(left_v * np.log1p(np.abs(right_v)), dtype=float),
            build_name=lambda left_label, right_label, _params: f"{left_label}*log1p_abs({right_label})",
            initial_enabled=False,
        ),
        PairGrammarRule(
            key="exp_right_mul",
            activation_family="interaction_compose",
            output_family="interaction_compose",
            complexity=4.6,
            build_expr=lambda left, right, params: binary_expr(
                "mul",
                left,
                exp_clip_expr(right, scale=float(params.get("scale", 8.0))),
            ),
            evaluate_values=lambda left_v, right_v, params: np.asarray(
                left_v
                * np.exp(
                    np.clip(
                        right_v / float(max(1.0, params.get("scale", 8.0))),
                        -30.0,
                        30.0,
                    )
                ),
                dtype=float,
            ),
            build_name=lambda left_label, right_label, _params: f"{left_label}*exp_clip({right_label})",
            initial_enabled=False,
        ),
        PairGrammarRule(
            key="gaussian_right_mul",
            activation_family="interaction_radial",
            output_family="interaction_radial",
            complexity=4.4,
            build_expr=lambda left, right, params: binary_expr(
                "mul",
                left,
                gaussian_rbf_expr(right, scale=float(params.get("scale", 8.0))),
            ),
            evaluate_values=lambda left_v, right_v, params: np.asarray(
                left_v
                * np.exp(
                    np.clip(
                        -((right_v / float(max(1.0, params.get("scale", 8.0)))) ** 2),
                        -30.0,
                        30.0,
                    )
                ),
                dtype=float,
            ),
            build_name=lambda left_label, right_label, params: (
                f"{left_label}*rbf_gauss_k{float(max(1.0, params.get('scale', 8.0))):g}({right_label})"
            ),
            initial_enabled=False,
        ),
        PairGrammarRule(
            key="softsign_product",
            activation_family="interaction_rational",
            output_family="interaction_rational",
            complexity=4.5,
            build_expr=lambda left, right, params: softsign_expr(
                binary_expr("mul", left, right),
                scale=float(params.get("scale", 8.0)),
            ),
            evaluate_values=lambda left_v, right_v, params: np.asarray(
                (left_v * right_v)
                / (np.abs(left_v * right_v) + float(max(1.0, params.get("scale", 8.0)))),
                dtype=float,
            ),
            build_name=lambda left_label, right_label, params: (
                f"softsign_prod_k{float(max(1.0, params.get('scale', 8.0))):g}({left_label},{right_label})"
            ),
            initial_enabled=False,
        ),
        PairGrammarRule(
            key="sin_product",
            activation_family="interaction_compose",
            output_family="interaction_dynamic",
            complexity=4.0,
            build_expr=lambda left, right, _params: unary_expr("sin", binary_expr("mul", left, right)),
            evaluate_values=lambda left_v, right_v, _params: np.asarray(np.sin(left_v * right_v), dtype=float),
            build_name=lambda left_label, right_label, _params: f"sin({left_label}*{right_label})",
            initial_enabled=False,
        ),
        PairGrammarRule(
            key="log_abs_product",
            activation_family="interaction_compose",
            output_family="interaction_compose",
            complexity=4.6,
            build_expr=lambda left, right, _params: log1p_abs_expr(binary_expr("mul", left, right)),
            evaluate_values=lambda left_v, right_v, _params: np.asarray(np.log1p(np.abs(left_v * right_v)), dtype=float),
            build_name=lambda left_label, right_label, _params: f"log1p_abs({left_label}*{right_label})",
            initial_enabled=False,
        ),
        PairGrammarRule(
            key="sqrt_abs_product",
            activation_family="interaction_compose",
            output_family="interaction_compose",
            complexity=4.6,
            build_expr=lambda left, right, _params: sqrt1p_abs_expr(binary_expr("mul", left, right)),
            evaluate_values=lambda left_v, right_v, _params: np.asarray(np.sqrt(np.abs(left_v * right_v) + 1.0), dtype=float),
            build_name=lambda left_label, right_label, _params: f"sqrt1p_abs({left_label}*{right_label})",
            initial_enabled=False,
        ),
        PairGrammarRule(
            key="tanh_tanh_product",
            activation_family="interaction_compose",
            output_family="interaction_compose",
            complexity=4.4,
            build_expr=lambda left, right, _params: binary_expr(
                "mul",
                unary_expr("tanh", left),
                unary_expr("tanh", right),
            ),
            evaluate_values=lambda left_v, right_v, _params: np.asarray(np.tanh(left_v) * np.tanh(right_v), dtype=float),
            build_name=lambda left_label, right_label, _params: f"tanh({left_label})*tanh({right_label})",
            initial_enabled=False,
        ),
    )


def default_primitive_registry() -> PrimitiveRegistry:
    return PrimitiveRegistry(
        unary_specs=_build_default_unary_specs(),
        pair_rules=_build_default_pair_rules(),
    )


__all__ = [
    "PrimitiveRegistry",
    "UnaryPrimitiveSpec",
    "PairGrammarRule",
    "default_primitive_registry",
    "feature_expr",
    "const_expr",
    "unary_expr",
    "binary_expr",
    "square_expr",
    "cube_expr",
    "pow4_expr",
    "pow5_expr",
    "pow6_expr",
    "log1p_abs_expr",
    "log1p_square_expr",
    "sqrt1p_abs_expr",
    "reciprocal_safe_expr",
    "exp_clip_expr",
    "softsign_expr",
    "inverse_quadratic_expr",
    "gaussian_rbf_expr",
    "laplace_rbf_expr",
    "softplus_clip_expr",
]
