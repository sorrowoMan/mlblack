from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from conditional.primitives import ConditionalPrimitiveSpec

from .primitive_registry import PrimitiveRegistry, binary_expr, const_expr, feature_expr, unary_expr


@dataclass(frozen=True)
class GrammarCandidate:
    name: str
    expr: dict[str, Any]
    complexity: float
    family: str
    activation_family: str
    features: tuple[int, ...]
    values: np.ndarray


@dataclass(frozen=True)
class ActivationPlan:
    unary_families: tuple[str, ...]
    pair_families: tuple[str, ...]
    gate_pair_families: tuple[str, ...]
    family_budgets: dict[str, int]
    family_scores: dict[str, float]


def _dedupe_ordered(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        key = str(raw)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return tuple(out)


def _merged_features(*feature_groups: Sequence[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    out: list[int] = []
    for group in feature_groups:
        for raw in group:
            idx = int(raw)
            if idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
    return tuple(out)


def make_seed_candidate(
    *,
    name: str,
    expr: Mapping[str, Any],
    values: np.ndarray,
    features: Sequence[int],
    complexity: float = 1.0,
    family: str = "seed",
    activation_family: str = "seed",
) -> GrammarCandidate:
    return GrammarCandidate(
        name=str(name),
        expr=dict(expr),
        complexity=float(complexity),
        family=str(family),
        activation_family=str(activation_family),
        features=_merged_features(features),
        values=np.asarray(values, dtype=float).reshape(-1),
    )


def _relu_expr(arg: Mapping[str, Any]) -> dict[str, Any]:
    z = dict(arg)
    return binary_expr("mul", const_expr(0.5), binary_expr("add", z, unary_expr("abs", z)))


def _soft_step_expr(feature_idx: int, threshold: float, steepness: float) -> dict[str, Any]:
    z = binary_expr("sub", feature_expr(feature_idx), const_expr(float(threshold)))
    kz = binary_expr("mul", const_expr(float(steepness)), z)
    t = unary_expr("tanh", kz)
    return binary_expr("mul", const_expr(0.5), binary_expr("add", const_expr(1.0), t))


def _binary_like(values: np.ndarray) -> bool:
    z = np.asarray(values, dtype=float).reshape(-1)
    if z.size == 0:
        return False
    mask = np.isfinite(z)
    if not np.any(mask):
        return False
    zv = z[mask]
    return bool(np.all(np.logical_or(np.isclose(zv, 0.0, atol=1e-8), np.isclose(zv, 1.0, atol=1e-8))))


def _apply_piecewise_mode_expr(mode: str, arg: Mapping[str, Any]) -> dict[str, Any]:
    mode_key = str(mode).strip().lower()
    if mode_key in {"identity", "linear"}:
        return dict(arg)
    if mode_key in {"zero", "off", "none"}:
        return const_expr(0.0)
    if mode_key in {"abs", "absolute"}:
        return unary_expr("abs", arg)
    if mode_key in {"square", "sq"}:
        return unary_expr("square", arg)
    if mode_key in {"neg_identity", "negative", "neg"}:
        return binary_expr("mul", const_expr(-1.0), arg)
    if mode_key in {"hinge", "positive_hinge", "relu"}:
        return _relu_expr(arg)
    if mode_key in {"negative_hinge", "left_hinge"}:
        return _relu_expr(binary_expr("mul", const_expr(-1.0), arg))
    return dict(arg)


def _apply_piecewise_mode_values(mode: str, values: np.ndarray) -> np.ndarray:
    z = np.asarray(values, dtype=float).reshape(-1)
    mode_key = str(mode).strip().lower()
    if mode_key in {"identity", "linear"}:
        return z
    if mode_key in {"zero", "off", "none"}:
        return np.zeros_like(z, dtype=float)
    if mode_key in {"abs", "absolute"}:
        return np.abs(z)
    if mode_key in {"square", "sq"}:
        return z * z
    if mode_key in {"neg_identity", "negative", "neg"}:
        return -z
    if mode_key in {"hinge", "positive_hinge", "relu"}:
        return np.maximum(0.0, z)
    if mode_key in {"negative_hinge", "left_hinge"}:
        return np.maximum(0.0, -z)
    return z


def lower_conditional_primitive_specs(
    specs: Sequence[ConditionalPrimitiveSpec],
    *,
    feature_names: Sequence[str],
    X: np.ndarray,
    default_gate_slope: float = 8.0,
) -> tuple[GrammarCandidate, ...]:
    if not specs:
        return tuple()
    x = np.asarray(X, dtype=float)
    name_to_idx = {str(name): int(idx) for idx, name in enumerate(feature_names)}
    seen_expr: set[str] = set()
    out: list[GrammarCandidate] = []

    def _safe_step_values(col: np.ndarray, *, cut: float, slope: float) -> np.ndarray:
        z = np.asarray(col, dtype=float).reshape(-1) - float(cut)
        return np.asarray(0.5 * (1.0 + np.tanh(float(slope) * z)), dtype=float)

    def _feature_bundle(feature_name: str) -> tuple[int, np.ndarray, dict[str, Any]] | None:
        idx = name_to_idx.get(str(feature_name))
        if idx is None or idx < 0 or idx >= x.shape[1]:
            return None
        return int(idx), np.asarray(x[:, idx], dtype=float).reshape(-1), feature_expr(int(idx))

    for spec in specs:
        family = str(spec.family)
        params = dict(spec.parameters)
        if not spec.source_features:
            continue
        base_bundle = _feature_bundle(str(spec.source_features[0]))
        if base_bundle is None:
            continue
        base_idx, base_values, base_expr = base_bundle
        features = [int(base_idx)]
        expr: dict[str, Any] | None = None
        values: np.ndarray | None = None
        complexity = 3.0

        multiplier_feature = params.get("multiplier_feature")
        multiplier_idx: int | None = None
        multiplier_expr: dict[str, Any] | None = None
        multiplier_values: np.ndarray | None = None
        if multiplier_feature is not None:
            multiplier_bundle = _feature_bundle(str(multiplier_feature))
            if multiplier_bundle is None:
                continue
            multiplier_idx, multiplier_values, multiplier_expr = multiplier_bundle
            features.append(int(multiplier_idx))

        if family == "gate_binary":
            threshold = float(params.get("threshold", 0.5))
            positive_value = float(params.get("positive_value", 1.0))
            negative_value = float(params.get("negative_value", 0.0))
            slope = float(max(1.0, params.get("slope", default_gate_slope)))
            if _binary_like(base_values) and abs(threshold - 0.5) <= 1e-8:
                if abs(positive_value - 1.0) <= 1e-8 and abs(negative_value) <= 1e-8:
                    expr = dict(base_expr)
                    values = np.asarray(base_values, dtype=float)
                else:
                    expr = binary_expr(
                        "add",
                        const_expr(float(negative_value)),
                        binary_expr("mul", const_expr(float(positive_value - negative_value)), base_expr),
                    )
                    values = np.asarray(negative_value + (positive_value - negative_value) * base_values, dtype=float)
            else:
                step_expr = _soft_step_expr(base_idx, threshold, slope)
                step_values = _safe_step_values(base_values, cut=threshold, slope=slope)
                expr = binary_expr(
                    "add",
                    const_expr(float(negative_value)),
                    binary_expr("mul", const_expr(float(positive_value - negative_value)), step_expr),
                )
                values = np.asarray(negative_value + (positive_value - negative_value) * step_values, dtype=float)
            complexity = 2.2
        elif family == "gate_onehot":
            categories = tuple(str(v) for v in params.get("categories", ()))
            if not categories:
                continue
            added_any = False
            for cat in categories:
                try:
                    cat_value = float(cat)
                except Exception:
                    continue
                slope = float(max(1.0, params.get("slope", default_gate_slope)))
                lower_expr = _soft_step_expr(base_idx, cat_value - 0.5, slope)
                upper_expr = _soft_step_expr(base_idx, cat_value + 0.5, slope)
                expr_now = binary_expr(
                    "mul",
                    lower_expr,
                    binary_expr("sub", const_expr(1.0), upper_expr),
                )
                lower_values = _safe_step_values(base_values, cut=cat_value - 0.5, slope=slope)
                upper_values = _safe_step_values(base_values, cut=cat_value + 0.5, slope=slope)
                values_now = np.asarray(lower_values * (1.0 - upper_values), dtype=float)
                key = json.dumps(expr_now, sort_keys=True)
                if key in seen_expr:
                    continue
                seen_expr.add(key)
                out.append(
                    GrammarCandidate(
                        name=f"{spec.name}:{cat}",
                        expr=expr_now,
                        complexity=3.2,
                        family=family,
                        activation_family="gate_interaction",
                        features=_merged_features((base_idx,)),
                        values=values_now,
                    )
                )
                added_any = True
            if added_any:
                continue
            continue
        elif family == "piecewise_hinge":
            cut = float(params.get("cut", 0.0))
            direction = str(params.get("direction", "positive")).strip().lower()
            if direction == "negative":
                shift_expr = binary_expr("sub", const_expr(float(cut)), base_expr)
                shift_values = np.asarray(float(cut) - base_values, dtype=float)
            else:
                shift_expr = binary_expr("sub", base_expr, const_expr(float(cut)))
                shift_values = np.asarray(base_values - float(cut), dtype=float)
            expr = _relu_expr(shift_expr)
            values = np.asarray(np.maximum(0.0, shift_values), dtype=float)
            complexity = 3.5
        elif family in {"gate_step", "gate_soft"}:
            cut = float(params.get("cut", 0.0))
            slope = float(max(1.0, params.get("slope", default_gate_slope)))
            expr = _soft_step_expr(base_idx, cut, slope)
            values = _safe_step_values(base_values, cut=cut, slope=slope)
            complexity = 4.0 if family == "gate_step" else 4.2
        elif family == "piecewise":
            cut = float(params.get("cut", 0.0))
            slope = float(max(1.0, params.get("slope", default_gate_slope)))
            left_mode = str(params.get("left_mode", "identity"))
            right_mode = str(params.get("right_mode", "identity"))
            shifted_expr = binary_expr("sub", base_expr, const_expr(float(cut)))
            shifted_values = np.asarray(base_values - float(cut), dtype=float)
            step_expr = _soft_step_expr(base_idx, cut, slope)
            step_values = _safe_step_values(base_values, cut=cut, slope=slope)
            left_expr = _apply_piecewise_mode_expr(left_mode, shifted_expr)
            right_expr = _apply_piecewise_mode_expr(right_mode, shifted_expr)
            expr = binary_expr(
                "add",
                binary_expr("mul", binary_expr("sub", const_expr(1.0), step_expr), left_expr),
                binary_expr("mul", step_expr, right_expr),
            )
            left_values = _apply_piecewise_mode_values(left_mode, shifted_values)
            right_values = _apply_piecewise_mode_values(right_mode, shifted_values)
            values = np.asarray((1.0 - step_values) * left_values + step_values * right_values, dtype=float)
            complexity = 5.0
        else:
            continue

        if expr is None or values is None:
            continue
        if multiplier_expr is not None and multiplier_values is not None:
            expr = binary_expr("mul", expr, multiplier_expr)
            values = np.asarray(values * multiplier_values, dtype=float)
            complexity += 1.0
        key = json.dumps(expr, sort_keys=True)
        if key in seen_expr:
            continue
        seen_expr.add(key)
        out.append(
            GrammarCandidate(
                name=str(spec.name),
                expr=expr,
                complexity=float(complexity),
                family=family,
                activation_family="gate_interaction",
                features=_merged_features(features),
                values=np.asarray(values, dtype=float).reshape(-1),
            )
        )
    return tuple(out)


def generate_unary_candidates(
    *,
    registry: PrimitiveRegistry,
    base_expr: Mapping[str, Any],
    base_values: np.ndarray,
    base_label: str,
    feature_ids: Sequence[int],
    params: Mapping[str, float],
    mode: str,
    active_families: Sequence[str] | None = None,
) -> tuple[GrammarCandidate, ...]:
    out: list[GrammarCandidate] = []
    for spec in registry.iter_unary_specs(mode=str(mode), families=active_families):
        values = np.asarray(spec.evaluate_values(np.asarray(base_values, dtype=float), params), dtype=float)
        out.append(
            GrammarCandidate(
                name=str(spec.build_name(str(base_label), params)),
                expr=dict(spec.build_expr(base_expr, params)),
                complexity=float(spec.complexity),
                family=str(spec.output_family),
                activation_family=str(spec.activation_family),
                features=tuple(int(v) for v in feature_ids),
                values=values.reshape(-1),
            )
        )
    return tuple(out)


def generate_pair_candidates(
    *,
    registry: PrimitiveRegistry,
    left_expr: Mapping[str, Any],
    left_values: np.ndarray,
    left_label: str,
    right_expr: Mapping[str, Any],
    right_values: np.ndarray,
    right_label: str,
    feature_ids: Sequence[int],
    params: Mapping[str, float],
    mode: str,
    active_families: Sequence[str] | None = None,
) -> tuple[GrammarCandidate, ...]:
    out: list[GrammarCandidate] = []
    lv = np.asarray(left_values, dtype=float).reshape(-1)
    rv = np.asarray(right_values, dtype=float).reshape(-1)
    for rule in registry.iter_pair_rules(mode=str(mode), families=active_families):
        values = np.asarray(rule.evaluate_values(lv, rv, params), dtype=float)
        out.append(
            GrammarCandidate(
                name=str(rule.build_name(str(left_label), str(right_label), params)),
                expr=dict(rule.build_expr(left_expr, right_expr, params)),
                complexity=float(rule.complexity),
                family=str(rule.output_family),
                activation_family=str(rule.activation_family),
                features=tuple(int(v) for v in feature_ids),
                values=values.reshape(-1),
            )
        )
    return tuple(out)


def generate_recursive_unary_candidates(
    *,
    registry: PrimitiveRegistry,
    seeds: Sequence[GrammarCandidate],
    params: Mapping[str, float],
    mode: str,
    active_families: Sequence[str] | None = None,
    max_complexity: float | None = None,
) -> tuple[GrammarCandidate, ...]:
    out: list[GrammarCandidate] = []
    seen_expr: set[str] = set()
    for seed in seeds:
        base_values = np.asarray(seed.values, dtype=float).reshape(-1)
        for spec in registry.iter_unary_specs(mode=str(mode), families=active_families):
            values = np.asarray(spec.evaluate_values(base_values, params), dtype=float).reshape(-1)
            expr = dict(spec.build_expr(seed.expr, params))
            complexity = float(seed.complexity) + float(spec.complexity) - 1.0
            if max_complexity is not None and float(complexity) > float(max_complexity):
                continue
            key = json.dumps(expr, sort_keys=True)
            if key in seen_expr:
                continue
            seen_expr.add(key)
            out.append(
                GrammarCandidate(
                    name=str(spec.build_name(str(seed.name), params)),
                    expr=expr,
                    complexity=float(complexity),
                    family=str(spec.output_family),
                    activation_family=str(spec.activation_family),
                    features=_merged_features(seed.features),
                    values=values,
                )
            )
    return tuple(out)


def generate_recursive_pair_candidates(
    *,
    registry: PrimitiveRegistry,
    left_seeds: Sequence[GrammarCandidate],
    right_seeds: Sequence[GrammarCandidate],
    params: Mapping[str, float],
    mode: str,
    active_families: Sequence[str] | None = None,
    max_complexity: float | None = None,
) -> tuple[GrammarCandidate, ...]:
    out: list[GrammarCandidate] = []
    seen_expr: set[str] = set()
    for left_seed in left_seeds:
        lv = np.asarray(left_seed.values, dtype=float).reshape(-1)
        for right_seed in right_seeds:
            rv = np.asarray(right_seed.values, dtype=float).reshape(-1)
            for rule in registry.iter_pair_rules(mode=str(mode), families=active_families):
                values = np.asarray(rule.evaluate_values(lv, rv, params), dtype=float).reshape(-1)
                expr = dict(rule.build_expr(left_seed.expr, right_seed.expr, params))
                complexity = float(left_seed.complexity) + float(right_seed.complexity) + float(rule.complexity) - 2.0
                if max_complexity is not None and float(complexity) > float(max_complexity):
                    continue
                key = json.dumps(expr, sort_keys=True)
                if key in seen_expr:
                    continue
                seen_expr.add(key)
                out.append(
                    GrammarCandidate(
                        name=str(rule.build_name(str(left_seed.name), str(right_seed.name), params)),
                        expr=expr,
                        complexity=float(complexity),
                        family=str(rule.output_family),
                        activation_family=str(rule.activation_family),
                        features=_merged_features(left_seed.features, right_seed.features),
                        values=values,
                    )
                )
    return tuple(out)


def select_activation_plan(
    *,
    registry: PrimitiveRegistry,
    feature_priority: np.ndarray,
    cross_priority: np.ndarray,
    change_scores: Mapping[int, float],
    gate_feature_count: int,
    unary_top_k: int,
    pair_top_k: int,
    gate_top_k: int,
    allow_trig: bool,
    allow_safe_exp: bool,
    allow_safe_log: bool,
    allow_safe_ratio: bool,
    family_budget: Mapping[str, int] | None = None,
) -> ActivationPlan:
    p = np.asarray(feature_priority, dtype=float).reshape(-1)
    cross = np.asarray(cross_priority, dtype=float)
    nonlinear_strength = float(np.max(np.abs(p))) if p.size > 0 else 0.0
    interaction_strength = float(np.max(np.abs(cross))) if cross.size > 0 else 0.0
    threshold_strength = float(max(change_scores.values())) if change_scores else 0.0
    gate_strength = float(threshold_strength + 0.25 * interaction_strength) if int(gate_feature_count) > 0 else 0.0

    family_scores: dict[str, float] = {
        "poly": float(nonlinear_strength),
        "bounded": float(0.90 * nonlinear_strength),
        "saturation": float(0.82 * nonlinear_strength + 0.18 * threshold_strength),
        "radial": float(0.72 * nonlinear_strength + 0.28 * threshold_strength),
        "trig": float(0.55 * nonlinear_strength) if bool(allow_trig) else 0.0,
        "safe_log": float(0.85 * nonlinear_strength + 0.15 * threshold_strength) if bool(allow_safe_log) else 0.0,
        "safe_exp": float(0.75 * nonlinear_strength + 0.25 * threshold_strength) if bool(allow_safe_exp) else 0.0,
        "safe_ratio": float(0.60 * nonlinear_strength + 0.40 * interaction_strength) if bool(allow_safe_ratio) else 0.0,
        "interaction_basic": float(interaction_strength),
        "interaction_poly": float(0.85 * interaction_strength + 0.15 * nonlinear_strength),
        "interaction_compose": float(0.70 * interaction_strength + 0.30 * nonlinear_strength),
        "interaction_ratio": float(0.65 * interaction_strength + 0.20 * threshold_strength) if bool(allow_safe_ratio) else 0.0,
        "interaction_saturation": float(0.72 * interaction_strength + 0.18 * nonlinear_strength + 0.10 * threshold_strength),
        "interaction_radial": float(0.62 * interaction_strength + 0.38 * threshold_strength),
        "interaction_rational": float(0.68 * interaction_strength + 0.22 * nonlinear_strength + 0.10 * threshold_strength),
        "gate_interaction": float(gate_strength),
    }

    unary_families_all = sorted({str(spec.activation_family) for spec in registry.unary_specs})
    pair_families_all = sorted({str(rule.activation_family) for rule in registry.pair_rules})
    family_budget_map = {str(k): int(max(0, v)) for k, v in dict(family_budget or {}).items()}

    def _budget_enabled(fam: str) -> bool:
        if fam not in family_budget_map:
            return True
        return int(family_budget_map[fam]) > 0

    unary_rank = sorted(
        [fam for fam in unary_families_all if float(family_scores.get(fam, 0.0)) > 0.0 and _budget_enabled(fam)],
        key=lambda fam: float(family_scores.get(fam, 0.0)),
        reverse=True,
    )
    pair_rank = sorted(
        [fam for fam in pair_families_all if float(family_scores.get(fam, 0.0)) > 0.0 and _budget_enabled(fam)],
        key=lambda fam: float(family_scores.get(fam, 0.0)),
        reverse=True,
    )

    unary_families = unary_rank[: int(max(1, unary_top_k))]
    pair_families = [fam for fam in pair_rank if fam != "gate_interaction"][: int(max(1, pair_top_k))]
    gate_pair_families = []
    if int(gate_feature_count) > 0:
        gate_pair_families = [
            fam
            for fam in pair_rank
            if fam in {"interaction_basic", "interaction_compose", "interaction_ratio", "gate_interaction"}
        ][: int(max(1, gate_top_k))]

    if "poly" not in unary_families and "poly" in unary_families_all:
        unary_families = ["poly", *unary_families]
    if "interaction_basic" not in pair_families and "interaction_basic" in pair_families_all:
        pair_families = ["interaction_basic", *pair_families]

    selected_budget: dict[str, int] = {}
    for fam in _dedupe_ordered([*unary_families, *pair_families, *gate_pair_families]):
        if fam in family_budget_map:
            selected_budget[str(fam)] = int(max(0, family_budget_map[fam]))

    return ActivationPlan(
        unary_families=_dedupe_ordered(unary_families[: int(max(1, unary_top_k))]),
        pair_families=_dedupe_ordered(pair_families[: int(max(1, pair_top_k))]),
        gate_pair_families=_dedupe_ordered(gate_pair_families[: int(max(1, gate_top_k))]),
        family_budgets=selected_budget,
        family_scores={str(k): float(v) for k, v in family_scores.items()},
    )


__all__ = [
    "GrammarCandidate",
    "ActivationPlan",
    "make_seed_candidate",
    "lower_conditional_primitive_specs",
    "generate_unary_candidates",
    "generate_pair_candidates",
    "generate_recursive_unary_candidates",
    "generate_recursive_pair_candidates",
    "select_activation_plan",
]
