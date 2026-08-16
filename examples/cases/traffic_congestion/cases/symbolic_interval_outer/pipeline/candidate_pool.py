from __future__ import annotations

from ..runtime.legacy_imports import *
from ..evaluation.metrics import *
from .splits import *
from .regimes import *

@dataclass(frozen=True)
class CandidateTerm:
    name: str
    expr: dict[str, Any]
    complexity: float
    family: str
    features: tuple[int, ...]
    prior_corr: float

def _feature_expr(j: int) -> dict[str, Any]:
    return {"type": "feature", "index": int(j)}

def _const_expr(v: float) -> dict[str, Any]:
    return {"type": "const", "value": float(v)}

def _unary_expr(op: str, arg: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "unary", "op": str(op), "arg": dict(arg)}

def _binary_expr(op: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "binary", "op": str(op), "left": dict(left), "right": dict(right)}

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
) -> list[CandidateTerm]:
    x = np.asarray(X, dtype=float)
    yt = np.asarray(y, dtype=float).reshape(-1)
    d = int(x.shape[1])

    pool: list[CandidateTerm] = []
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
        for op in ("square", "sin", "cos", "tanh"):
            if op == "square":
                z = np.asarray(z0 * z0, dtype=float)
            elif op == "sin":
                z = np.asarray(np.sin(z0), dtype=float)
            elif op == "cos":
                z = np.asarray(np.cos(z0), dtype=float)
            else:
                z = np.asarray(np.tanh(z0), dtype=float)
            pool.append(
                CandidateTerm(
                    name=f"{op}(x{j})",
                    expr=_unary_expr(op, base),
                    complexity=2.0,
                    family=f"unary_{op}",
                    features=(int(j),),
                    prior_corr=float(abs(_safe_corr(z, yt))),
                )
            )
        if bool(include_safe_log1p_abs):
            ex_log1p_abs = _unary_expr("log", _binary_expr("add", _unary_expr("abs", base), _const_expr(1.0)))
            z_log1p_abs = np.asarray(np.log1p(np.abs(z0)), dtype=float)
            pool.append(
                CandidateTerm(
                    name=f"log1p_abs(x{j})",
                    expr=ex_log1p_abs,
                    complexity=2.5,
                    family="unary_log1p_abs",
                    features=(int(j),),
                    prior_corr=float(abs(_safe_corr(z_log1p_abs, yt))),
                )
            )
        if bool(include_safe_exp_clip):
            k = float(max(1.0, safe_exp_clip_k))
            ex_exp_clip = _unary_expr("exp", _binary_expr("mul", _const_expr(1.0 / k), base))
            z_exp_clip = np.asarray(np.exp(np.clip(z0 / k, -30.0, 30.0)), dtype=float)
            pool.append(
                CandidateTerm(
                    name=f"exp_clip(x{j})",
                    expr=ex_exp_clip,
                    complexity=2.5,
                    family="unary_exp_clip",
                    features=(int(j),),
                    prior_corr=float(abs(_safe_corr(z_exp_clip, yt))),
                )
            )
        if bool(include_safe_reciprocal):
            epsv = float(max(1e-8, safe_reciprocal_eps))
            ex_recip = _binary_expr("div", _const_expr(1.0), _binary_expr("add", _unary_expr("abs", base), _const_expr(epsv)))
            z_recip = np.asarray(1.0 / (np.abs(z0) + epsv), dtype=float)
            pool.append(
                CandidateTerm(
                    name=f"reciprocal_safe(x{j})",
                    expr=ex_recip,
                    complexity=2.8,
                    family="unary_reciprocal_safe",
                    features=(int(j),),
                    prior_corr=float(abs(_safe_corr(z_recip, yt))),
                )
            )

    corr = np.asarray([abs(_safe_corr(x[:, j], yt)) for j in range(d)], dtype=float)
    top_idx = list(np.argsort(-corr)[: min(int(max(2, topk_for_pairs)), d)])
    if bool(include_pair_interactions):
        for i in range(len(top_idx)):
            for j in range(i + 1, len(top_idx)):
                a = int(top_idx[i])
                b = int(top_idx[j])
                z = np.asarray(x[:, a] * x[:, b], dtype=float)
                pool.append(
                    CandidateTerm(
                        name=f"x{a}*x{b}",
                        expr=_binary_expr("mul", _feature_expr(a), _feature_expr(b)),
                        complexity=3.0,
                        family="interaction",
                        features=(int(a), int(b)),
                        prior_corr=float(abs(_safe_corr(z, yt))),
                    )
                )

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
                        ex1 = _binary_expr("mul", _feature_expr(i), _feature_expr(j))
                        k1 = json.dumps(ex1, sort_keys=True)
                        if k1 not in existing_keys:
                            z1 = np.asarray(x[:, i] * x[:, j], dtype=float)
                            pool.append(
                                CandidateTerm(
                                    name=f"grad:x{i}*x{j}",
                                    expr=ex1,
                                    complexity=3.0,
                                    family="interaction_grad",
                                    features=(int(i), int(j)),
                                    prior_corr=float(abs(_safe_corr(z1, yt))),
                                )
                            )
                            existing_keys.add(k1)
                        ex2 = _binary_expr("mul", _unary_expr("tanh", _feature_expr(i)), _feature_expr(j))
                        k2 = json.dumps(ex2, sort_keys=True)
                        if k2 not in existing_keys:
                            z2 = np.asarray(np.tanh(x[:, i]) * x[:, j], dtype=float)
                            pool.append(
                                CandidateTerm(
                                    name=f"grad:tanh(x{i})*x{j}",
                                    expr=ex2,
                                    complexity=4.0,
                                    family="interaction_grad",
                                    features=(int(i), int(j)),
                                    prior_corr=float(abs(_safe_corr(z2, yt))),
                                )
                            )
                            existing_keys.add(k2)
        except Exception:
            pass
    return pool

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
) -> list[CandidateTerm]:
    x = np.asarray(X, dtype=float)
    r = np.asarray(y_residual, dtype=float).reshape(-1)
    d = int(x.shape[1])
    new_terms: list[CandidateTerm] = []
    existing_keys = {json.dumps(c.expr, sort_keys=True) for c in existing}

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
    budget = int(max(1, max_new_terms))

    def _try_add(name: str, expr: Mapping[str, Any], complexity: float, family: str, feats: Sequence[int], z: np.ndarray) -> None:
        nonlocal budget
        if budget <= 0:
            return
        key = json.dumps(expr, sort_keys=True)
        if key in existing_keys:
            return
        existing_keys.add(key)
        budget -= 1
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

    # 0) gradient change-point driven hinge / gate atoms on continuous features
    cont_focus_idx = [int(i) for i in focus_idx if int(i) not in gate_set]
    for i in cont_focus_idx:
        cps = _estimate_gradient_change_points(
            x_col=x[:, i],
            residual=r,
            min_bin=max(24, int(0.06 * x.shape[0])),
            topk=2,
        )
        if not cps:
            continue
        xi = np.asarray(x[:, i], dtype=float)
        # derive steepness from robust spread
        spread = float(np.quantile(xi, 0.9) - np.quantile(xi, 0.1))
        k = float(4.0 / max(1e-6, spread))
        for c, _score in cps:
            z_shift = np.asarray(xi - float(c), dtype=float)
            ex_h_pos = _relu_expr(_binary_expr("sub", _feature_expr(i), _const_expr(float(c))))
            z_h_pos = np.asarray(np.maximum(0.0, z_shift), dtype=float)
            _try_add(f"hinge+:x{i}-({c:.4g})", ex_h_pos, 3.5, "gate_hinge_grad", (i,), z_h_pos)
            if budget <= 0:
                break

            ex_h_neg = _relu_expr(_binary_expr("sub", _const_expr(float(c)), _feature_expr(i)))
            z_h_neg = np.asarray(np.maximum(0.0, -z_shift), dtype=float)
            _try_add(f"hinge-:({c:.4g})-x{i}", ex_h_neg, 3.5, "gate_hinge_grad", (i,), z_h_neg)
            if budget <= 0:
                break

            # Symmetry + slope-change modeling: hinge(x,c) * original feature.
            ex_h_pos_self = _binary_expr("mul", ex_h_pos, _feature_expr(i))
            z_h_pos_self = np.asarray(z_h_pos * xi, dtype=float)
            _try_add(f"hinge+:x{i}-({c:.4g})*x{i}", ex_h_pos_self, 4.5, "interaction_hinge_grad", (i, i), z_h_pos_self)
            if budget <= 0:
                break

            ex_h_neg_self = _binary_expr("mul", ex_h_neg, _feature_expr(i))
            z_h_neg_self = np.asarray(z_h_neg * xi, dtype=float)
            _try_add(f"hinge-:({c:.4g})-x{i}*x{i}", ex_h_neg_self, 4.5, "interaction_hinge_grad", (i, i), z_h_neg_self)
            if budget <= 0:
                break

            ex_step = _soft_step_expr(feature_idx=i, threshold=float(c), steepness=float(k))
            z_step = np.asarray(0.5 * (1.0 + np.tanh(k * z_shift)), dtype=float)
            _try_add(f"soft_step:x{i}>{c:.4g}", ex_step, 4.0, "gate_step_grad", (i,), z_step)
            if budget <= 0:
                break

            # targeted gated interaction with strongest partner for current residual map
            partner = [int(j) for j in np.argsort(-np.abs(cross[i, :])).tolist() if int(j) != int(i)]
            # Lag-aware partner prioritization: explicitly favor lag-cross terms.
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
            if preferred:
                seen = set()
                merged = []
                for j in preferred + partner:
                    if int(j) not in seen:
                        seen.add(int(j))
                        merged.append(int(j))
                partner = merged
            if partner:
                j = int(partner[0])
                ex_h_pos_mul = _binary_expr("mul", ex_h_pos, _feature_expr(j))
                z_h_pos_mul = np.asarray(z_h_pos * x[:, j], dtype=float)
                _try_add(f"hinge+(x{i}>{c:.4g})*x{j}", ex_h_pos_mul, 5.0, "interaction_hinge_grad", (i, j), z_h_pos_mul)
                if budget <= 0:
                    break
                ex_h_neg_mul = _binary_expr("mul", ex_h_neg, _feature_expr(j))
                z_h_neg_mul = np.asarray(z_h_neg * x[:, j], dtype=float)
                _try_add(f"hinge-(x{i}<{c:.4g})*x{j}", ex_h_neg_mul, 5.0, "interaction_hinge_grad", (i, j), z_h_neg_mul)
                if budget <= 0:
                    break
                ex_gate_mul = _binary_expr("mul", ex_step, _feature_expr(j))
                z_gate_mul = np.asarray(z_step * x[:, j], dtype=float)
                fam = "interaction_gate_step" if (j in gate_set) else "interaction_step"
                _try_add(f"soft_step(x{i}>{c:.4g})*x{j}", ex_gate_mul, 5.0, fam, (i, j), z_gate_mul)
                if budget <= 0:
                    break
        if budget <= 0:
            break

    # 1) residual-guided interaction expansions
    for i in focus_idx:
        partner_idx = [int(j) for j in np.argsort(-np.abs(cross[i, :])).tolist() if int(j) != int(i)]
        if not partner_idx:
            partner_idx = [int(j) for j in np.argsort(-np.abs(p)).tolist() if int(j) != int(i)]
        for j in partner_idx[: max(2, int(partner_topk))]:
            ex1 = _binary_expr("mul", _feature_expr(i), _feature_expr(j))
            z1 = np.asarray(x[:, i] * x[:, j], dtype=float)
            fam = "interaction_gate" if (int(i) in gate_set or int(j) in gate_set) else "interaction_dynamic"
            _try_add(f"dyn:x{i}*x{j}", ex1, 3.0, fam, (i, j), z1)
            if budget <= 0:
                break
            ex2 = _binary_expr("mul", _unary_expr("tanh", _feature_expr(i)), _feature_expr(j))
            z2 = np.asarray(np.tanh(x[:, i]) * x[:, j], dtype=float)
            _try_add(f"dyn:tanh(x{i})*x{j}", ex2, 4.0, fam, (i, j), z2)
            if budget <= 0:
                break
            ex3 = _unary_expr("sin", ex1)
            z3 = np.asarray(np.sin(z1), dtype=float)
            _try_add(f"dyn:sin(x{i}*x{j})", ex3, 4.0, "interaction_dynamic", (i, j), z3)
            if budget <= 0:
                break
        if budget <= 0:
            break

    # 2) gate-feature injections to mimic piecewise behavior
    if budget > 0 and gate_idx:
        cont_idx = [int(i) for i in focus_idx if int(i) not in gate_set]
        for i in cont_idx:
            for g in gate_idx:
                exg = _binary_expr("mul", _feature_expr(i), _feature_expr(g))
                zg = np.asarray(x[:, i] * x[:, g], dtype=float)
                _try_add(f"gate:x{i}*x{g}", exg, 3.0, "interaction_gate", (i, g), zg)
                if budget <= 0:
                    break
                ext = _binary_expr("mul", _unary_expr("tanh", _feature_expr(i)), _feature_expr(g))
                zt = np.asarray(np.tanh(x[:, i]) * x[:, g], dtype=float)
                _try_add(f"gate:tanh(x{i})*x{g}", ext, 4.0, "interaction_gate", (i, g), zt)
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

__all__ = ['CandidateTerm', '_feature_expr', '_const_expr', '_unary_expr', '_binary_expr', '_relu_expr', '_soft_step_expr', '_estimate_gradient_change_points', '_build_candidate_pool', '_is_gate_feature_name', '_expand_candidate_pool_from_residual', '_prune_candidate_pool']
