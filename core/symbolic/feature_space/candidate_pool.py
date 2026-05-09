from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from conditional import ConditionalConfig
from conditional.primitives import ConditionalPrimitiveSpec
from core.symbolic.gradient_parser import GradientParser
from core.symbolic.feature_space.activation_config import DynamicActivationConfig, resolve_dynamic_activation_kwargs
from core.symbolic.symbolic_structure_search import evaluate_genome_with_ridge
from .generation_grammar import (
    GrammarCandidate,
    generate_recursive_pair_candidates,
    generate_recursive_unary_candidates,
    generate_pair_candidates,
    generate_unary_candidates,
    lower_conditional_primitive_specs,
    make_seed_candidate,
    select_activation_plan,
)
from .primitive_registry import default_primitive_registry


def _resolve_activation_config(activation_config: Mapping[str, Any] | None) -> dict[str, Any]:
    if activation_config:
        return dict(activation_config)
    try:
        return dict(resolve_dynamic_activation_kwargs(DynamicActivationConfig()))
    except Exception:
        return {
            "unary_top_k": 6,
            "pair_top_k": 8,
            "gate_top_k": 6,
            "recursive_depth": 2,
            "recursive_seed_top_k": 3,
            "recursive_pair_seed_top_k": 2,
            "recursive_max_complexity": 9.5,
            "allow_trig": True,
            "allow_safe_exp": True,
            "allow_safe_log": True,
            "allow_safe_ratio": True,
            "family_budget": {},
        }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    # Split the norm product to avoid overflow in sqrt(ax * ay) when both sums are large.
    nx = float(np.linalg.norm(xc))
    ny = float(np.linalg.norm(yc))
    denom = float(nx * ny) + 1e-12
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(xc, yc) / denom)


@dataclass(frozen=True)
class CandidateTerm:
    name: str
    expr: dict[str, Any]
    complexity: float
    family: str
    features: tuple[int, ...]
    prior_corr: float


def _candidate_from_grammar(item: GrammarCandidate, target: np.ndarray) -> CandidateTerm:
    values = np.asarray(item.values, dtype=float).reshape(-1)
    return CandidateTerm(
        name=str(item.name),
        expr=dict(item.expr),
        complexity=float(item.complexity),
        family=str(item.family),
        features=tuple(int(v) for v in item.features),
        prior_corr=float(abs(_safe_corr(values, np.asarray(target, dtype=float).reshape(-1)))),
    )


def _resolve_conditional_specs(
    conditional_config: ConditionalConfig | Sequence[ConditionalPrimitiveSpec] | None,
) -> tuple[ConditionalPrimitiveSpec, ...]:
    if conditional_config is None:
        return tuple()
    if isinstance(conditional_config, ConditionalConfig):
        if not bool(conditional_config.enabled):
            return tuple()
        return tuple(conditional_config.primitive_specs())
    return tuple(spec for spec in conditional_config if isinstance(spec, ConditionalPrimitiveSpec))


def build_conditional_candidate_terms(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str],
    conditional_config: ConditionalConfig | Sequence[ConditionalPrimitiveSpec] | None,
) -> list[CandidateTerm]:
    specs = _resolve_conditional_specs(conditional_config)
    if not specs:
        return []
    lowered = lower_conditional_primitive_specs(
        specs,
        feature_names=feature_names,
        X=np.asarray(X, dtype=float),
    )
    target = np.asarray(y, dtype=float).reshape(-1)
    return [_candidate_from_grammar(item, target) for item in lowered]


def augment_candidate_pool_with_conditional_config(
    pool: Sequence[CandidateTerm],
    *,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    conditional_config: ConditionalConfig | Sequence[ConditionalPrimitiveSpec] | None,
) -> list[CandidateTerm]:
    base_pool = list(pool)
    extra_terms = build_conditional_candidate_terms(
        X,
        y,
        feature_names=feature_names,
        conditional_config=conditional_config,
    )
    if not extra_terms:
        return base_pool
    seen_expr = {json.dumps(term.expr, sort_keys=True) for term in base_pool}
    for term in extra_terms:
        key = json.dumps(term.expr, sort_keys=True)
        if key in seen_expr:
            continue
        seen_expr.add(key)
        base_pool.append(term)
    return base_pool


def _feature_name_index(feature_names: Sequence[str]) -> dict[str, int]:
    return {str(name): int(idx) for idx, name in enumerate(feature_names)}


def _resolve_threshold_seed_specs(
    feature_names: Sequence[str],
    conditional_config: ConditionalConfig | Sequence[ConditionalPrimitiveSpec] | None,
) -> list[dict[str, Any]]:
    name_to_idx = _feature_name_index(feature_names)
    out: list[dict[str, Any]] = []
    for spec in _resolve_conditional_specs(conditional_config):
        family = str(spec.family)
        if family not in {"piecewise_hinge", "gate_step", "gate_soft", "piecewise"}:
            continue
        if not spec.source_features:
            continue
        feature_name = str(spec.source_features[0])
        feature_idx = name_to_idx.get(feature_name)
        if feature_idx is None:
            continue
        params = dict(spec.parameters)
        cut = float(params.get("cut", 0.0))
        if not np.isfinite(cut):
            continue
        multiplier_feature = params.get("multiplier_feature")
        multiplier_idx = name_to_idx.get(str(multiplier_feature)) if multiplier_feature is not None else None
        out.append(
            {
                "name": str(spec.name),
                "family": family,
                "feature_name": feature_name,
                "feature_idx": int(feature_idx),
                "cut": float(cut),
                "multiplier_feature": None if multiplier_feature is None else str(multiplier_feature),
                "multiplier_idx": None if multiplier_idx is None else int(multiplier_idx),
                "direction": str(params.get("direction", "positive")),
                "slope": float(params.get("slope", 4.0)),
                "left_mode": str(params.get("left_mode", "identity")),
                "right_mode": str(params.get("right_mode", "identity")),
            }
        )
    return out


def _build_dynamic_threshold_variants(
    seed_specs: Sequence[dict[str, Any]],
    *,
    X: np.ndarray,
    feature_names: Sequence[str],
    focus_features: Sequence[int],
    change_points_by_feature: Mapping[int, Sequence[tuple[float, float]]],
) -> tuple[ConditionalPrimitiveSpec, ...]:
    if not seed_specs:
        return tuple()
    x = np.asarray(X, dtype=float)
    focus_set = {int(v) for v in focus_features}
    out: list[ConditionalPrimitiveSpec] = []
    seen: set[tuple[str, int, float, str | None]] = set()
    for entry in seed_specs:
        feature_idx = int(entry["feature_idx"])
        if feature_idx not in focus_set:
            continue
        col = np.asarray(x[:, feature_idx], dtype=float).reshape(-1)
        col = col[np.isfinite(col)]
        if col.size < 8:
            continue
        spread = float(np.quantile(col, 0.9) - np.quantile(col, 0.1))
        if not np.isfinite(spread) or spread <= 1e-12:
            spread = float(np.max(col) - np.min(col))
        if not np.isfinite(spread) or spread <= 1e-12:
            continue
        delta = float(max(1e-8, 0.08 * spread))
        min_gap = float(max(1e-8, 0.05 * spread))
        base_cut = float(entry["cut"])
        raw_cuts: list[float] = [base_cut, base_cut - delta, base_cut + delta]
        for cp, _score in change_points_by_feature.get(feature_idx, ()):
            raw_cuts.append(float(cp))
        cuts: list[float] = []
        lo = float(np.min(col))
        hi = float(np.max(col))
        for cut in raw_cuts:
            if not np.isfinite(cut):
                continue
            if cut <= lo or cut >= hi:
                continue
            if any(abs(cut - prev) < min_gap for prev in cuts):
                continue
            cuts.append(float(cut))
        for cut in cuts:
            family = str(entry["family"])
            multiplier_feature = entry["multiplier_feature"]
            key = (family, feature_idx, round(float(cut), 10), None if multiplier_feature is None else str(multiplier_feature))
            if key in seen:
                continue
            seen.add(key)
            params: dict[str, Any] = {"cut": float(cut)}
            if multiplier_feature is not None:
                params["multiplier_feature"] = str(multiplier_feature)
            if family == "piecewise_hinge":
                params["direction"] = str(entry["direction"])
            if family in {"gate_step", "gate_soft"}:
                params["slope"] = float(entry["slope"])
            if family == "piecewise":
                params["left_mode"] = str(entry["left_mode"])
                params["right_mode"] = str(entry["right_mode"])
                params["slope"] = float(entry["slope"])
            out.append(
                ConditionalPrimitiveSpec(
                    name=f"{entry['name']}__dyn_{feature_idx}_{len(out)}",
                    family=family,
                    source_features=(str(entry["feature_name"]),),
                    parameters=params,
                )
            )
    return tuple(out)


def _feature_expr(j: int) -> dict[str, Any]:
    return {"type": "feature", "index": int(j)}


def _const_expr(v: float) -> dict[str, Any]:
    return {"type": "const", "value": float(v)}


def _unary_expr(op: str, arg: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "unary", "op": str(op), "arg": dict(arg)}


def _binary_expr(op: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "binary", "op": str(op), "left": dict(left), "right": dict(right)}


def _square_expr(arg: Mapping[str, Any]) -> dict[str, Any]:
    return _unary_expr("square", arg)


def _cube_expr(arg: Mapping[str, Any]) -> dict[str, Any]:
    return _binary_expr("mul", _square_expr(arg), arg)


def _pow4_expr(arg: Mapping[str, Any]) -> dict[str, Any]:
    return _square_expr(_square_expr(arg))


def _log1p_abs_expr(arg: Mapping[str, Any]) -> dict[str, Any]:
    return _unary_expr("log", _binary_expr("add", _unary_expr("abs", arg), _const_expr(1.0)))


def _sqrt1p_abs_expr(arg: Mapping[str, Any]) -> dict[str, Any]:
    return _unary_expr("sqrt", _binary_expr("add", _unary_expr("abs", arg), _const_expr(1.0)))


def _reciprocal_safe_expr(arg: Mapping[str, Any], *, eps: float) -> dict[str, Any]:
    return _binary_expr("div", _const_expr(1.0), _binary_expr("add", _unary_expr("abs", arg), _const_expr(float(eps))))


def _exp_clip_expr(arg: Mapping[str, Any], *, scale: float) -> dict[str, Any]:
    s = float(max(1.0, scale))
    return _unary_expr("exp", _binary_expr("mul", _const_expr(1.0 / s), arg))


def _relu_expr(arg: Mapping[str, Any]) -> dict[str, Any]:
    # relu(z) = 0.5 * (z + abs(z)) using existing DSL ops.
    z = dict(arg)
    return _binary_expr("mul", _const_expr(0.5), _binary_expr("add", z, _unary_expr("abs", z)))


def _soft_step_expr(feature_idx: int, threshold: float, steepness: float) -> dict[str, Any]:
    # soft-step(x>c) ~= 0.5 * (1 + tanh(k*(x-c)))
    z = _binary_expr("sub", _feature_expr(feature_idx), _const_expr(float(threshold)))
    kz = _binary_expr("mul", _const_expr(float(steepness)), z)
    t = _unary_expr("tanh", kz)
    return _binary_expr("mul", _const_expr(0.5), _binary_expr("add", _const_expr(1.0), t))


def _estimate_gradient_change_points(
    x_col: np.ndarray,
    residual: np.ndarray,
    *,
    min_bin: int,
    topk: int = 2,
) -> list[tuple[float, float]]:
    x = np.asarray(x_col, dtype=float).reshape(-1)
    r = np.asarray(residual, dtype=float).reshape(-1)
    n = int(x.size)
    if n < int(max(8, 2 * min_bin)):
        return []
    order = np.argsort(x)
    xs = x[order]
    rs = r[order]
    ps = np.concatenate(([0.0], np.cumsum(rs)))
    candidate: list[tuple[float, int]] = []
    step = max(1, n // 128)
    for t in range(int(min_bin), int(n - min_bin), int(step)):
        ln = float(t)
        rn = float(n - t)
        lm = float((ps[t] - ps[0]) / max(1.0, ln))
        rm = float((ps[n] - ps[t]) / max(1.0, rn))
        # weighted jump score on residual means
        score = float(abs(lm - rm) * np.sqrt((ln * rn) / max(1.0, float(n))))
        candidate.append((score, t))
    if not candidate:
        return []
    candidate.sort(key=lambda kv: kv[0], reverse=True)
    out: list[tuple[float, float]] = []
    used_pos: list[int] = []
    min_sep = max(4, n // 20)
    for score, t in candidate:
        if len(out) >= int(max(1, topk)):
            break
        if any(abs(int(t) - int(u)) < int(min_sep) for u in used_pos):
            continue
        if t <= 0 or t >= n:
            continue
        c = float(0.5 * (xs[t - 1] + xs[t]))
        if not np.isfinite(c):
            continue
        out.append((c, float(score)))
        used_pos.append(int(t))
    return out


def _build_candidate_pool(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str],
    topk_for_pairs: int = 6,
    include_pair_interactions: bool = True,
    include_gradient_enrich: bool = True,
    include_safe_log1p_abs: bool = True,
    include_safe_exp_clip: bool = True,
    include_safe_reciprocal: bool = True,
    safe_exp_clip_k: float = 8.0,
    safe_reciprocal_eps: float = 1e-3,
    activation_config: Mapping[str, Any] | None = None,
    conditional_config: ConditionalConfig | Sequence[ConditionalPrimitiveSpec] | None = None,
) -> list[CandidateTerm]:
    x = np.asarray(X, dtype=float)
    yt = np.asarray(y, dtype=float).reshape(-1)
    d = int(x.shape[1])
    registry = default_primitive_registry()
    eps_safe = float(max(1e-8, safe_reciprocal_eps))
    activation_cfg = _resolve_activation_config(activation_config)

    pool: list[CandidateTerm] = []
    unary_families = ["poly", "bounded", "saturation", "radial"]
    if bool(activation_cfg.get("allow_trig", True)):
        unary_families.append("trig")
    if bool(include_safe_log1p_abs) and bool(activation_cfg.get("allow_safe_log", True)):
        unary_families.append("safe_log")
    if bool(include_safe_exp_clip) and bool(activation_cfg.get("allow_safe_exp", True)):
        unary_families.append("safe_exp")
    if bool(include_safe_reciprocal) and bool(activation_cfg.get("allow_safe_ratio", True)):
        unary_families.append("safe_ratio")

    for j in range(d):
        base = _feature_expr(j)
        z0 = np.asarray(x[:, j], dtype=float)
        pool.append(
            CandidateTerm(
                name=f"x{j}:{feature_names[j]}",
                expr=base,
                complexity=1.0,
                family="linear",
                features=(int(j),),
                prior_corr=float(abs(_safe_corr(z0, yt))),
            )
        )
        spread = float(np.quantile(z0, 0.9) - np.quantile(z0, 0.1))
        param_scales = [float(max(1.0, spread, safe_exp_clip_k))]
        if bool(include_safe_exp_clip):
            for kv in (float(max(1.0, 0.5 * safe_exp_clip_k)), float(max(1.0, 2.0 * safe_exp_clip_k))):
                if not any(abs(kv - kk) <= 1e-9 for kk in param_scales):
                    param_scales.append(kv)
        for scale in param_scales:
            params = {"scale": float(scale), "eps": float(eps_safe)}
            families_now = list(unary_families)
            if float(scale) != float(param_scales[0]):
                families_now = [fam for fam in families_now if fam == "safe_exp"]
                if not families_now:
                    continue
            for item in generate_unary_candidates(
                registry=registry,
                base_expr=base,
                base_values=z0,
                base_label=f"x{j}",
                feature_ids=(int(j),),
                params=params,
                mode="initial",
                active_families=families_now,
            ):
                pool.append(_candidate_from_grammar(item, yt))

    corr = np.asarray([abs(_safe_corr(x[:, j], yt)) for j in range(d)], dtype=float)
    top_idx = list(np.argsort(-corr)[: min(int(max(2, topk_for_pairs)), d)])
    if bool(include_pair_interactions):
        pair_families = ["interaction_basic", "interaction_poly", "interaction_saturation", "interaction_radial", "interaction_rational"]
        if bool(include_safe_reciprocal) and bool(activation_cfg.get("allow_safe_ratio", True)):
            pair_families.append("interaction_ratio")
        for i in range(len(top_idx)):
            for j in range(i + 1, len(top_idx)):
                a = int(top_idx[i])
                b = int(top_idx[j])
                spread_ab = float(
                    max(
                        1.0,
                        safe_exp_clip_k,
                        np.quantile(x[:, a], 0.9) - np.quantile(x[:, a], 0.1),
                        np.quantile(x[:, b], 0.9) - np.quantile(x[:, b], 0.1),
                    )
                )
                for item in generate_pair_candidates(
                    registry=registry,
                    left_expr=_feature_expr(a),
                    left_values=x[:, a],
                    left_label=f"x{a}",
                    right_expr=_feature_expr(b),
                    right_values=x[:, b],
                    right_label=f"x{b}",
                    feature_ids=(int(a), int(b)),
                    params={"scale": float(spread_ab), "eps": float(eps_safe)},
                    mode="initial",
                    active_families=pair_families,
                ):
                    pool.append(_candidate_from_grammar(item, yt))

    # Gradient-guided interaction enrichment (function-space guidance, not constant tuning).
    if bool(include_gradient_enrich):
        try:
            seed_genome = [{"name": f"x{j}", "expr": _feature_expr(j)} for j in range(d)]
            fit_seed = evaluate_genome_with_ridge(
                seed_genome,
                X_train=x,
                y_train=yt.reshape(-1, 1),
                X_eval=x,
                y_eval=yt.reshape(-1, 1),
                l2=1e-5,
            )
            gs = GradientParser.build_signal(
                genome=seed_genome,
                weight=np.asarray(fit_seed.get("weight"), dtype=float),
                X=x,
                y=yt.reshape(-1, 1),
                slope_mode="binned_median",
                slope_bins=24,
                slope_min_bin_samples=12,
            )
            cross = np.asarray(getattr(gs, "cross_feature_priority", np.zeros((d, d), dtype=float)), dtype=float)
            p = np.asarray(getattr(gs, "feature_priority", np.zeros((d,), dtype=float)), dtype=float).reshape(-1)
            if p.size == d and cross.shape == (d, d):
                focus_idx = [int(i) for i in np.argsort(-p)[: min(4, d)]]
                existing_keys = {json.dumps(c.expr, sort_keys=True) for c in pool}
                for i in focus_idx:
                    partner_idx = [int(j) for j in np.argsort(-cross[i, :]).tolist() if int(j) != int(i)]
                    for j in partner_idx[:3]:
                        spread_ij = float(
                            max(
                                1.0,
                                safe_exp_clip_k,
                                np.quantile(x[:, i], 0.9) - np.quantile(x[:, i], 0.1),
                                np.quantile(x[:, j], 0.9) - np.quantile(x[:, j], 0.1),
                            )
                        )
                        for item in generate_pair_candidates(
                            registry=registry,
                            left_expr=_feature_expr(i),
                            left_values=x[:, i],
                            left_label=f"x{i}",
                            right_expr=_feature_expr(j),
                            right_values=x[:, j],
                            right_label=f"x{j}",
                            feature_ids=(int(i), int(j)),
                            params={"scale": float(spread_ij), "eps": float(eps_safe)},
                            mode="dynamic",
                            active_families=(
                                "interaction_basic",
                                "interaction_compose",
                                "interaction_ratio",
                                "interaction_saturation",
                                "interaction_radial",
                                "interaction_rational",
                            ),
                        ):
                            key = json.dumps(item.expr, sort_keys=True)
                            if key in existing_keys:
                                continue
                            pool.append(_candidate_from_grammar(item, yt))
                            existing_keys.add(key)
        except Exception:
            pass
    return augment_candidate_pool_with_conditional_config(
        pool,
        X=x,
        y=yt,
        feature_names=feature_names,
        conditional_config=conditional_config,
    )


def _is_gate_feature_name(name: str) -> bool:
    nm = str(name).lower()
    return nm.startswith("is_") or ("holiday" in nm) or ("weekend" in nm) or ("bad_weather" in nm) or ("aqi_high" in nm)


def _expand_candidate_pool_from_residual(
    *,
    X: np.ndarray,
    y_residual: np.ndarray,
    feature_names: Sequence[str],
    base_genome: Sequence[Mapping[str, Any]],
    base_weight: np.ndarray,
    existing: Sequence[CandidateTerm],
    max_new_terms: int,
    focus_top_features: int,
    partner_topk: int,
    activation_config: Mapping[str, Any] | None = None,
    conditional_config: ConditionalConfig | Sequence[ConditionalPrimitiveSpec] | None = None,
) -> list[CandidateTerm]:
    x = np.asarray(X, dtype=float)
    r = np.asarray(y_residual, dtype=float).reshape(-1)
    d = int(x.shape[1])
    new_terms: list[CandidateTerm] = []
    existing_keys = {json.dumps(c.expr, sort_keys=True) for c in existing}
    safe_eps = 1e-3
    registry = default_primitive_registry()

    gate_idx = [int(i) for i, nm in enumerate(feature_names) if _is_gate_feature_name(str(nm))]
    gate_set = set(gate_idx)
    name_to_idx = {str(nm): int(i) for i, nm in enumerate(feature_names)}

    try:
        gs = GradientParser.build_signal(
            genome=base_genome,
            weight=np.asarray(base_weight, dtype=float),
            X=x,
            y=r.reshape(-1, 1),
            slope_mode="binned_median",
            slope_bins=24,
            slope_min_bin_samples=12,
        )
        cross = np.asarray(getattr(gs, "cross_feature_priority", np.zeros((d, d), dtype=float)), dtype=float)
        p = np.asarray(getattr(gs, "feature_priority", np.zeros((d,), dtype=float)), dtype=float).reshape(-1)
    except Exception:
        cross = np.zeros((d, d), dtype=float)
        p = np.asarray([abs(_safe_corr(x[:, j], r)) for j in range(d)], dtype=float)

    if p.size != d:
        p = np.asarray([abs(_safe_corr(x[:, j], r)) for j in range(d)], dtype=float)
    if cross.shape != (d, d):
        cross = np.zeros((d, d), dtype=float)

    focus_idx = [int(i) for i in np.argsort(-np.abs(p))[: max(2, min(int(focus_top_features), d))]]
    cont_focus_idx = [int(i) for i in focus_idx if int(i) not in gate_set]
    change_points_by_feature: dict[int, list[tuple[float, float]]] = {}
    change_scores: dict[int, float] = {}
    for i in cont_focus_idx:
        cps = _estimate_gradient_change_points(
            x_col=x[:, i],
            residual=r,
            min_bin=max(24, int(0.06 * x.shape[0])),
            topk=2,
        )
        if cps:
            change_points_by_feature[int(i)] = list(cps)
            change_scores[int(i)] = float(max(score for _c, score in cps))
    threshold_seed_specs = _resolve_threshold_seed_specs(feature_names, conditional_config)

    activation_cfg = _resolve_activation_config(activation_config)
    activation_plan = select_activation_plan(
        registry=registry,
        feature_priority=p,
        cross_priority=cross,
        change_scores=change_scores,
        gate_feature_count=int(len(gate_idx)),
        unary_top_k=int(max(1, activation_cfg.get("unary_top_k", 4))),
        pair_top_k=int(max(1, activation_cfg.get("pair_top_k", 5))),
        gate_top_k=int(max(1, activation_cfg.get("gate_top_k", 4))),
        allow_trig=bool(activation_cfg.get("allow_trig", True)),
        allow_safe_exp=bool(activation_cfg.get("allow_safe_exp", True)),
        allow_safe_log=bool(activation_cfg.get("allow_safe_log", True)),
        allow_safe_ratio=bool(activation_cfg.get("allow_safe_ratio", True)),
        family_budget=activation_cfg.get("family_budget"),
    )
    budget = int(max(1, max_new_terms))
    family_budget = {str(k): int(max(0, v)) for k, v in dict(activation_cfg.get("family_budget", {})).items()}
    family_usage: dict[str, int] = {}
    recursive_depth = int(max(1, activation_cfg.get("recursive_depth", 1)))
    recursive_seed_top_k = int(max(1, activation_cfg.get("recursive_seed_top_k", 2)))
    recursive_pair_seed_top_k = int(max(1, activation_cfg.get("recursive_pair_seed_top_k", 2)))
    recursive_max_complexity = float(max(3.0, activation_cfg.get("recursive_max_complexity", 8.5)))
    feature_seed_cache: dict[int, tuple[GrammarCandidate, ...]] = {}

    def _try_add(
        name: str,
        expr: Mapping[str, Any],
        complexity: float,
        family: str,
        feats: Sequence[int],
        z: np.ndarray,
        *,
        activation_family: str | None = None,
    ) -> None:
        nonlocal budget
        if budget <= 0:
            return
        act_family = str(activation_family if activation_family is not None else family)
        limit = family_budget.get(act_family)
        if limit is None and act_family == "threshold_auto_cut":
            limit = family_budget.get("gate_interaction")
        used = int(family_usage.get(act_family, 0))
        if limit is not None and used >= int(limit):
            return
        key = json.dumps(expr, sort_keys=True)
        if key in existing_keys:
            return
        existing_keys.add(key)
        budget -= 1
        family_usage[act_family] = int(used + 1)
        new_terms.append(
            CandidateTerm(
                name=str(name),
                expr=dict(expr),
                complexity=float(complexity),
                family=str(family),
                features=tuple(int(v) for v in feats),
                prior_corr=float(abs(_safe_corr(np.asarray(z, dtype=float).reshape(-1), r))),
            )
        )

    def _try_add_generated(item: GrammarCandidate) -> None:
        _try_add(
            name=str(item.name),
            expr=item.expr,
            complexity=float(item.complexity),
            family=str(item.family),
            feats=item.features,
            z=item.values,
            activation_family=str(item.activation_family),
        )

    def _feature_scale(idx: int) -> float:
        zi = np.asarray(x[:, int(idx)], dtype=float)
        return float(max(1.0, np.quantile(zi, 0.9) - np.quantile(zi, 0.1)))

    def _pair_params(i: int, j: int) -> dict[str, float]:
        return {
            "scale": float(max(_feature_scale(i), _feature_scale(j))),
            "eps": float(safe_eps),
        }

    def _grammar_score(item: GrammarCandidate) -> float:
        return float(abs(_safe_corr(np.asarray(item.values, dtype=float).reshape(-1), r)))

    def _top_grammar_candidates(items: Sequence[GrammarCandidate], top_k: int) -> list[GrammarCandidate]:
        ranked = sorted(
            list(items),
            key=lambda item: (-_grammar_score(item), float(item.complexity), str(item.name)),
        )
        return ranked[: int(max(1, top_k))]

    def _base_feature_seed(idx: int) -> GrammarCandidate:
        return make_seed_candidate(
            name=f"x{int(idx)}",
            expr=_feature_expr(int(idx)),
            values=x[:, int(idx)],
            features=(int(idx),),
            complexity=1.0,
            family="linear",
            activation_family="linear",
        )

    def _recursive_unary_from_seeds(
        seeds: Sequence[GrammarCandidate],
        *,
        params: Mapping[str, float],
        active_families: Sequence[str],
    ) -> list[GrammarCandidate]:
        if int(recursive_depth) < 2:
            return []
        top_seeds = _top_grammar_candidates(seeds, int(recursive_seed_top_k))
        if not top_seeds:
            return []
        return _top_grammar_candidates(
            generate_recursive_unary_candidates(
                registry=registry,
                seeds=top_seeds,
                params=params,
                mode="dynamic",
                active_families=active_families,
                max_complexity=float(recursive_max_complexity),
            ),
            int(recursive_seed_top_k),
        )

    def _feature_seed_pool(idx: int) -> tuple[GrammarCandidate, ...]:
        key = int(idx)
        cached = feature_seed_cache.get(key)
        if cached is not None:
            return cached
        params = {"scale": float(_feature_scale(key)), "eps": float(safe_eps)}
        unary_layer = tuple(
            generate_unary_candidates(
                registry=registry,
                base_expr=_feature_expr(key),
                base_values=x[:, key],
                base_label=f"x{key}",
                feature_ids=(key,),
                params=params,
                mode="dynamic",
                active_families=activation_plan.unary_families,
            )
        )
        top_unary = _top_grammar_candidates(unary_layer, int(recursive_pair_seed_top_k))
        feature_seed_cache[key] = tuple([_base_feature_seed(key), *top_unary])
        return feature_seed_cache[key]

    def _preferred_partners(i: int, partner_idx: Sequence[int]) -> list[int]:
        fi_name = str(feature_names[i]) if int(i) < len(feature_names) else ""
        preferred: list[int] = []
        if fi_name.startswith("ci_lag"):
            lag_suffix = fi_name.replace("ci_", "")
            for base in ("avg_speed_", "avg_occ_", "total_flow_"):
                pj = name_to_idx.get(f"{base}{lag_suffix}")
                if pj is not None and int(pj) != int(i):
                    preferred.append(int(pj))
        elif "_lag" in fi_name:
            lag_suffix = fi_name.split("_")[-1]
            for base in ("ci", "avg_speed", "avg_occ", "total_flow"):
                key = f"{base}_{lag_suffix}" if base != "ci" else f"ci_{lag_suffix}"
                pj = name_to_idx.get(key)
                if pj is not None and int(pj) != int(i):
                    preferred.append(int(pj))
        seen: set[int] = set()
        out: list[int] = []
        for j in list(preferred) + [int(v) for v in partner_idx]:
            if int(j) in seen or int(j) == int(i):
                continue
            seen.add(int(j))
            out.append(int(j))
        return out

    # -1) expand around configured threshold-family seeds first so auto-cut mechanisms
    # are not crowded out by generic unary growth.
    if budget > 0 and threshold_seed_specs:
        threshold_focus_idx = sorted(
            {
                *(int(v) for v in cont_focus_idx),
                *(int(entry["feature_idx"]) for entry in threshold_seed_specs),
            }
        )
        threshold_variants = _build_dynamic_threshold_variants(
            threshold_seed_specs,
            X=x,
            feature_names=feature_names,
            focus_features=threshold_focus_idx,
            change_points_by_feature=change_points_by_feature,
        )
        if threshold_variants:
            threshold_generated = lower_conditional_primitive_specs(
                threshold_variants,
                feature_names=feature_names,
                X=x,
            )
            for item in _top_grammar_candidates(threshold_generated, int(max(2, len(threshold_generated)))):
                _try_add(
                    name=str(item.name),
                    expr=item.expr,
                    complexity=float(item.complexity),
                    family=str(item.family),
                    feats=item.features,
                    z=item.values,
                    activation_family="threshold_auto_cut",
                )
                if budget <= 0:
                    break
        if budget <= 0:
            return new_terms

    # 0) residual-guided unary enrichments on top focused features
    for i in focus_idx:
        params_i = {"scale": float(_feature_scale(i)), "eps": float(safe_eps)}
        unary_layer = tuple(
            generate_unary_candidates(
                registry=registry,
                base_expr=_feature_expr(i),
                base_values=x[:, i],
                base_label=f"x{i}",
                feature_ids=(int(i),),
                params=params_i,
                mode="dynamic",
                active_families=activation_plan.unary_families,
            )
        )
        for item in unary_layer:
            _try_add_generated(item)
            if budget <= 0:
                break
        if budget <= 0:
            break

        for item in _recursive_unary_from_seeds(
            unary_layer,
            params=params_i,
            active_families=activation_plan.unary_families,
        ):
            _try_add_generated(item)
            if budget <= 0:
                break
        if budget <= 0:
            break

    # 1) gradient change-point driven hinge / gate atoms on continuous features
    for i in cont_focus_idx:
        cps = change_points_by_feature.get(int(i), [])
        if not cps:
            continue
        xi = np.asarray(x[:, i], dtype=float)
        spread = float(np.quantile(xi, 0.9) - np.quantile(xi, 0.1))
        k = float(4.0 / max(1e-6, spread))
        base_params = {"scale": float(max(1.0, spread)), "eps": float(safe_eps)}
        for c, _score in cps:
            z_shift = np.asarray(xi - float(c), dtype=float)
            shifted_expr = _binary_expr("sub", _feature_expr(i), _const_expr(float(c)))
            ex_h_pos = _relu_expr(shifted_expr)
            z_h_pos = np.asarray(np.maximum(0.0, z_shift), dtype=float)
            _try_add(
                f"hinge+:x{i}-({c:.4g})",
                ex_h_pos,
                3.5,
                "gate_hinge_grad",
                (i,),
                z_h_pos,
                activation_family="gate_interaction",
            )
            if budget <= 0:
                break

            ex_h_neg = _relu_expr(_binary_expr("sub", _const_expr(float(c)), _feature_expr(i)))
            z_h_neg = np.asarray(np.maximum(0.0, -z_shift), dtype=float)
            _try_add(
                f"hinge-:({c:.4g})-x{i}",
                ex_h_neg,
                3.5,
                "gate_hinge_grad",
                (i,),
                z_h_neg,
                activation_family="gate_interaction",
            )
            if budget <= 0:
                break

            ex_step = _soft_step_expr(feature_idx=i, threshold=float(c), steepness=float(k))
            z_step = np.asarray(0.5 * (1.0 + np.tanh(k * z_shift)), dtype=float)
            _try_add(
                f"soft_step:x{i}>{c:.4g}",
                ex_step,
                4.0,
                "gate_step_grad",
                (i,),
                z_step,
                activation_family="gate_interaction",
            )
            if budget <= 0:
                break

            shifted_unary_layer = tuple(
                generate_unary_candidates(
                    registry=registry,
                    base_expr=shifted_expr,
                    base_values=z_shift,
                    base_label=f"x{i}-({c:.4g})",
                    feature_ids=(int(i),),
                    params=base_params,
                    mode="dynamic",
                    active_families=activation_plan.unary_families,
                )
            )
            for item in shifted_unary_layer:
                _try_add_generated(item)
                if budget <= 0:
                    break
            if budget <= 0:
                break

            for item in _recursive_unary_from_seeds(
                shifted_unary_layer,
                params=base_params,
                active_families=activation_plan.unary_families,
            ):
                _try_add_generated(item)
                if budget <= 0:
                    break
            if budget <= 0:
                break

            partner = [int(j) for j in np.argsort(-np.abs(cross[i, :])).tolist() if int(j) != int(i)]
            partner = _preferred_partners(i, partner)
            if partner:
                j = int(partner[0])
                pair_params = _pair_params(i, j)
                pair_generated: list[GrammarCandidate] = []
                pair_bases = (
                    (ex_h_pos, z_h_pos, f"hinge+(x{i}>{c:.4g})", activation_plan.pair_families),
                    (ex_h_neg, z_h_neg, f"hinge-(x{i}<{c:.4g})", activation_plan.pair_families),
                    (ex_step, z_step, f"soft_step(x{i}>{c:.4g})", activation_plan.gate_pair_families or activation_plan.pair_families),
                )
                for left_expr, left_values, left_label, fams in pair_bases:
                    for item in generate_pair_candidates(
                        registry=registry,
                        left_expr=left_expr,
                        left_values=left_values,
                        left_label=left_label,
                        right_expr=_feature_expr(j),
                        right_values=x[:, j],
                        right_label=f"x{j}",
                        feature_ids=(int(i), int(j)),
                        params=pair_params,
                        mode="dynamic",
                        active_families=fams,
                    ):
                        pair_generated.append(item)
                        _try_add_generated(item)
                        if budget <= 0:
                            break
                    if budget <= 0:
                        break
                if budget <= 0:
                    break

                for item in _recursive_unary_from_seeds(
                    pair_generated,
                    params=pair_params,
                    active_families=activation_plan.unary_families,
                ):
                    _try_add_generated(item)
                    if budget <= 0:
                        break
                if budget <= 0:
                    break
        if budget <= 0:
            break

    # 2) residual-guided pair expansions with depth-2 transformed seeds
    for i in focus_idx:
        partner_idx = [int(j) for j in np.argsort(-np.abs(cross[i, :])).tolist() if int(j) != int(i)]
        if not partner_idx:
            partner_idx = [int(j) for j in np.argsort(-np.abs(p)).tolist() if int(j) != int(i)]
        partner_idx = _preferred_partners(i, partner_idx)
        for j in partner_idx[: max(2, int(partner_topk))]:
            pair_params = _pair_params(i, j)
            pair_generated = tuple(
                generate_recursive_pair_candidates(
                    registry=registry,
                    left_seeds=_feature_seed_pool(i),
                    right_seeds=_feature_seed_pool(j),
                    params=pair_params,
                    mode="dynamic",
                    active_families=activation_plan.pair_families,
                    max_complexity=float(recursive_max_complexity),
                )
            )
            for item in _top_grammar_candidates(
                pair_generated,
                int(max(2, recursive_pair_seed_top_k * 3)),
            ):
                _try_add_generated(item)
                if budget <= 0:
                    break
            if budget <= 0:
                break

            for item in _recursive_unary_from_seeds(
                _top_grammar_candidates(pair_generated, int(recursive_pair_seed_top_k)),
                params=pair_params,
                active_families=activation_plan.unary_families,
            ):
                _try_add_generated(item)
                if budget <= 0:
                    break
            if budget <= 0:
                break
        if budget <= 0:
            break

    # 3) gate-feature interactions activated only when gate families score well
    if budget > 0 and gate_idx:
        cont_idx = [int(i) for i in focus_idx if int(i) not in gate_set]
        for i in cont_idx:
            for g in gate_idx:
                pair_params = _pair_params(i, g)
                gate_pair_generated: list[GrammarCandidate] = []
                for item in generate_pair_candidates(
                    registry=registry,
                    left_expr=_feature_expr(i),
                    left_values=x[:, i],
                    left_label=f"x{i}",
                    right_expr=_feature_expr(g),
                    right_values=x[:, g],
                    right_label=f"x{g}",
                    feature_ids=(int(i), int(g)),
                    params=pair_params,
                    mode="dynamic",
                    active_families=activation_plan.gate_pair_families or activation_plan.pair_families,
                ):
                    gate_pair_generated.append(item)
                    _try_add_generated(item)
                    if budget <= 0:
                        break
                if budget <= 0:
                    break

                for item in _recursive_unary_from_seeds(
                    gate_pair_generated,
                    params=pair_params,
                    active_families=activation_plan.unary_families,
                ):
                    _try_add_generated(item)
                    if budget <= 0:
                        break
                if budget <= 0:
                    break
            if budget <= 0:
                break

    return new_terms


def _prune_candidate_pool(
    *,
    candidates: Sequence[CandidateTerm],
    keep_expr_keys: set[str],
    feature_names: Sequence[str],
    max_pool_size: int,
) -> list[CandidateTerm]:
    pool = list(candidates)
    cap = int(max(16, max_pool_size))
    if len(pool) <= cap:
        return pool

    # Always keep foundational terms.
    anchor_keys: set[str] = set()
    for c in pool:
        if str(c.family) == "linear":
            anchor_keys.add(json.dumps(c.expr, sort_keys=True))
        if str(c.family).startswith("unary_") and len(c.features) == 1 and int(c.features[0]) < int(len(feature_names)):
            anchor_keys.add(json.dumps(c.expr, sort_keys=True))

    keep_all = set(keep_expr_keys) | anchor_keys
    keep_terms = [c for c in pool if json.dumps(c.expr, sort_keys=True) in keep_all]
    rest = [c for c in pool if json.dumps(c.expr, sort_keys=True) not in keep_all]
    rest.sort(key=lambda c: float(c.prior_corr), reverse=True)
    budget = max(0, cap - len(keep_terms))
    return keep_terms + rest[:budget]


