from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from core.symbolic.symbolic_dsl import (
    detect_binary_columns,
    evaluate_expression_numpy,
    evaluate_genome_numpy,
    expression_to_string,
    normalize_genome,
)
from core.symbolic.gradient_correction import GradientCorrection, GradientCorrectionConfig
from core.symbolic.expression_graph_cache import ExpressionGraphCache
from core.symbolic.gradient_parser import GradientParser
from core.symbolic.path_memory import SymbolicPathMemory
from core.symbolic.structure_optimizer import StructureOptimizer, StructureScoreConfig
from core.symbolic.symbolic_gradient import gradient_formula_strings
from training.inner_runtime import (
    InnerRuntimeDispatcher,
    InnerRuntimeErrorPayload,
    InnerRuntimeFinishPayload,
    InnerRuntimeRoundPayload,
    InnerRuntimeStartPayload,
)


_ALLOWED_NESTED_UNARY_OPS = frozenset({"identity", "square", "sin", "cos", "tanh", "exp", "log", "abs", "sqrt"})


def _as_2d_float(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError("array must be 2D")
    return out


def _feature_label(feature_names: Sequence[str] | None, index: int) -> str:
    if feature_names is not None and not isinstance(feature_names, (str, bytes, bytearray)):
        values = tuple(str(v) for v in tuple(feature_names))
        if 0 <= int(index) < len(values):
            label = str(values[int(index)]).strip()
            if label:
                return label
    return f"x{int(index)}"


def _feature_labels(indices: Sequence[int], feature_names: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(_feature_label(feature_names, int(index)) for index in tuple(indices))


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
    # relu(z) = 0.5 * (z + abs(z))
    z = dict(arg)
    z_abs = _unary_expr("abs", z)
    return _binary_expr("mul", _const_expr(0.5), _binary_expr("add", z, z_abs))


def _expr_key(expr: Mapping[str, Any]) -> str:
    return expression_to_string(expr, param_values=None, precision=12)


def _expr_node_count(expr: Mapping[str, Any]) -> int:
    t = str(expr.get("type", ""))
    if t in {"feature", "const", "param"}:
        return 1
    if t == "unary":
        return 1 + _expr_node_count(expr["arg"])
    if t == "binary":
        return 1 + _expr_node_count(expr["left"]) + _expr_node_count(expr["right"])
    return 1


def _expr_depth(expr: Mapping[str, Any]) -> int:
    t = str(expr.get("type", ""))
    if t in {"feature", "const", "param"}:
        return 1
    if t == "unary":
        return 1 + _expr_depth(expr["arg"])
    if t == "binary":
        return 1 + max(_expr_depth(expr["left"]), _expr_depth(expr["right"]))
    return 1


def _expr_features(expr: Mapping[str, Any]) -> tuple[int, ...]:
    used: set[int] = set()

    def rec(node: Mapping[str, Any]) -> None:
        t = str(node.get("type", ""))
        if t == "feature":
            used.add(int(node["index"]))
            return
        if t == "unary":
            rec(node["arg"])
            return
        if t == "binary":
            rec(node["left"])
            rec(node["right"])
            return

    rec(expr)
    return tuple(sorted(used))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("corr shape mismatch")

    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    denom = float(np.sqrt(np.dot(xc, xc) * np.dot(yc, yc))) + 1e-12
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(xc, yc) / denom)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = _as_2d_float(y_true)
    yp = _as_2d_float(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch y_true={yt.shape} y_pred={yp.shape}")

    err = yp - yt
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    yt_flat = yt.reshape(-1)
    yp_flat = yp.reshape(-1)
    ss_tot = float(np.sum((yt_flat - np.mean(yt_flat)) ** 2))
    ss_res = float(np.sum((yp_flat - yt_flat) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def _fit_ridge_readout(phi: np.ndarray, y: np.ndarray, *, l2: float) -> dict[str, np.ndarray]:
    X = _as_2d_float(phi)
    Y = _as_2d_float(y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("phi and y row mismatch")

    n, t = X.shape
    M = Y.shape[1]

    A = np.hstack([X, np.ones((n, 1), dtype=float)])
    reg = np.eye(t + 1, dtype=float)
    reg[-1, -1] = 0.0

    lhs = A.T @ A + float(l2) * reg
    rhs = A.T @ Y
    try:
        W = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(lhs) @ rhs

    pred = A @ W
    weight = W[:-1, :].reshape(t, M)
    bias = W[-1, :].reshape(M)

    return {
        "weight": np.asarray(weight, dtype=float),
        "bias": np.asarray(bias, dtype=float),
        "pred": np.asarray(pred, dtype=float),
    }

def _design_matrix_from_genome(
    genome: Sequence[Mapping[str, Any]],
    X: np.ndarray,
    *,
    graph_cache: ExpressionGraphCache | None = None,
    batch_key: str | None = None,
) -> np.ndarray:
    x = _as_2d_float(X)
    terms = list(genome)
    if not terms:
        return np.zeros((int(x.shape[0]), 0), dtype=float)
    if graph_cache is None:
        return evaluate_genome_numpy(terms, x)

    cols: list[np.ndarray] = []
    for term in terms:
        expr = term["expr"]
        expr_key = str(term.get("expr_key", _expr_key(expr)))
        col = graph_cache.evaluate_expression(
            expr,
            x,
            expr_key=expr_key,
            batch_key=batch_key,
        ).reshape(-1)
        cols.append(np.asarray(col, dtype=float))
    return np.asarray(np.stack(cols, axis=1), dtype=float)

def _feature_residual_scores(X: np.ndarray, residual: np.ndarray) -> np.ndarray:
    x = _as_2d_float(X)
    r = _as_2d_float(residual)
    if x.shape[0] != r.shape[0]:
        raise ValueError("X and residual row mismatch")

    d = int(x.shape[1])
    m = int(r.shape[1])
    scores = np.zeros((d,), dtype=float)

    for j in range(d):
        xj = x[:, j]
        corr_sum = 0.0
        for k in range(m):
            corr_sum += abs(_safe_corr(xj, r[:, k]))
        scores[j] = corr_sum / max(1, m)

    return scores


def _default_seed_genome(input_dim: int) -> tuple[Dict[str, Any], ...]:
    out: list[Dict[str, Any]] = []
    for i in range(int(input_dim)):
        out.append(
            {
                "name": f"x{i}",
                "expr": _feature_expr(i),
            }
        )
    return tuple(out)


def _genome_expr_keys(genome: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    out: list[str] = []
    for term in genome:
        out.append(_expr_key(term["expr"]))
    return tuple(out)


def _genome_signature(genome: Sequence[Mapping[str, Any]]) -> str:
    return SymbolicPathMemory.genome_signature(_genome_expr_keys(genome))


@dataclass(frozen=True)
class StructureSearchConfig:
    max_added_terms: int = 10
    topk_features: int = 8
    max_pair_terms: int = 16
    max_candidates_per_iter: int = 500
    candidate_keep_top: int = 12
    candidate_pool_mode: str = "legacy"  # legacy | shared_full

    ridge_l2: float = 1e-4
    min_score: float = 1e-6
    min_projected_gain: float = 1e-7
    score_complexity_penalty: float = 7e-4
    score_corr_bonus: float = 0.04
    score_grad_guidance_bonus: float = 0.08
    min_actual_rmse_gain: float = 0.0

    # validation-driven overfit guard (accept/rollback/replace/tabu)
    overfit_guard_enabled: bool = False
    overfit_guard_val_ratio: float = 0.2
    overfit_guard_min_val_samples: int = 64
    overfit_guard_random_seed: int = 42
    overfit_guard_min_val_rmse_gain: float = 0.0
    overfit_guard_max_gap_increase: float = 0.05
    overfit_guard_patience: int = 3
    overfit_guard_snapshot_min_improve: float = 0.0
    overfit_guard_tabu_rounds: int = 2
    overfit_guard_replace_topk: int = 3
    overfit_guard_replace_drop_topk: int = 3

    grad_focus_topk: int = 3
    grad_min_priority: float = 1e-4
    grad_slope_mode: str = "central_diff"
    grad_slope_bins: int = 24
    grad_slope_min_bin_samples: int = 12

    grad_adv_check: bool = False
    grad_adv_trials: int = 3
    grad_adv_noise_std: float = 0.02
    grad_adv_min_stability: float = 0.0
    grad_adv_random_seed: int = 42

    include_hinge: bool = True
    hinge_quantiles: Sequence[float] = (0.25, 0.5, 0.75)

    # Keep defaults stable for downstream torch training.
    unary_ops: Sequence[str] = ("square", "sin", "cos", "tanh")
    nested_mode: str = "auto"  # manual | auto | hybrid
    nested_unary_patterns: Sequence[str] = (
        "sin(square)",
        "cos(square)",
    )
    auto_nested_allowed_ops: Sequence[str] = ("square", "sin", "cos", "tanh")
    auto_nested_min_depth: int = 2
    auto_nested_max_depth: int = 3
    auto_nested_beam_width: int = 8
    auto_nested_max_patterns_per_feature: int = 16
    shared_pool_safe_log1p_abs: bool = True
    shared_pool_safe_exp_clip: bool = True
    shared_pool_safe_reciprocal: bool = True
    # candidate-budget policy: fixed | interaction_first
    interaction_budget_mode: str = "fixed"
    interaction_diag_threshold: float = 1.15
    interaction_diag_topk_features: int = 6
    interaction_pair_budget_boost: float = 2.0
    interaction_grad_projection_budget_boost: float = 1.5
    max_arity: int = 3
    max_expr_depth: int = 8

    # gradient-residual projection (u_i ~ pr_u) for higher-order interaction hints
    enable_grad_residual_projection: bool = True
    grad_projection_topk_focus: int = 3
    grad_projection_partner_pool: int = 8
    grad_projection_topk_partners: int = 3
    grad_projection_topk_unary: int = 2
    # whether focus feature can also use unary/nested transforms in projection candidates:
    # x_i*phi(x_j,...) -> g(x_i)*phi(x_j,...)
    grad_projection_focus_include_transforms: bool = True
    grad_projection_focus_topk_transforms: int = 2
    # human-defined partner orders: k means x_i multiplied by k projected partner terms.
    # Example: (1, 2, 3) -> generate x_i*phi(xj), x_i*phi(xj)*psi(xp), x_i*phi(xj)*psi(xp)*chi(xq)
    grad_projection_partner_orders: Sequence[int] = (1, 2)
    grad_projection_enable_pair_dictionary: bool = True
    grad_projection_min_abs_corr: float = 0.05
    grad_projection_max_generated: int = 120

    # structural pruning ("砍挂件")
    enable_prune: bool = True
    prune_rmse_tolerance: float = 1e-8
    prune_max_removed_per_iter: int = 1

    # persistent path memory (cross-task)
    path_memory_enabled: bool = True
    path_memory_db_path: str = ""
    path_memory_namespace: str = "global"
    path_memory_prior_bonus: float = 0.03
    path_memory_tabu_penalty: float = 0.06
    path_memory_min_outcomes: int = 3
    path_memory_hard_tabu: bool = False
    path_memory_hard_tabu_accept_rate: float = 0.1

    # reusable compute-graph cache
    graph_cache_enabled: bool = True
    graph_cache_max_value_entries: int = 20000
    graph_cache_max_derivative_entries: int = 50000
    graph_cache_backend: str = "memory"
    graph_cache_db_path: str = ""
    graph_cache_namespace: str = "global"
    graph_cache_persist_values: bool = False

    # joint bundle selection (approximate L0/L1 subset on shortlist)
    joint_bundle_enabled: bool = False
    joint_bundle_max_terms: int = 3
    joint_bundle_preselect_topk: int = 8
    joint_bundle_max_combos: int = 48
    joint_bundle_l1_alpha: float = 1e-3
    joint_bundle_l1_iters: int = 20


@dataclass(frozen=True)
class StructureSearchResult:
    genome: tuple[Dict[str, Any], ...]
    base_metrics: dict[str, float]
    final_metrics: dict[str, float]
    iterations: tuple[dict[str, Any], ...]
    weight: np.ndarray
    bias: np.ndarray
    score_trace: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_metrics": dict(self.base_metrics),
            "final_metrics": dict(self.final_metrics),
            "n_terms": int(len(self.genome)),
            "score_trace": [float(v) for v in self.score_trace],
            "iterations": [dict(item) for item in self.iterations],
            "genome": list(self.genome),
            "readout_weight": np.asarray(self.weight, dtype=float).tolist(),
            "readout_bias": np.asarray(self.bias, dtype=float).tolist(),
        }


def _split_outer_call(raw: str) -> tuple[str, str] | None:
    txt = str(raw).strip()
    if not txt:
        return None
    p = txt.find("(")
    if p <= 0 or not txt.endswith(")"):
        return None

    depth = 0
    close_at = -1
    for i, ch in enumerate(txt):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return None
            if depth == 0:
                close_at = i
                break
    if depth != 0 or close_at != int(len(txt) - 1):
        return None

    op = txt[:p].strip().lower()
    inner = txt[p + 1 : close_at].strip().lower()
    if not op:
        return None
    return op, inner


def _compose_nested_pattern(pattern: str, base: Mapping[str, Any]) -> Dict[str, Any] | None:
    txt = str(pattern).strip().lower().replace(" ", "")
    if not txt:
        return None

    def rec(node: str) -> Dict[str, Any] | None:
        s = str(node).strip().lower()
        if not s:
            return None
        if s == "x":
            return dict(base)

        call = _split_outer_call(s)
        if call is None:
            if s not in _ALLOWED_NESTED_UNARY_OPS:
                return None
            return _unary_expr(s, base)

        op, inner = call
        if op not in _ALLOWED_NESTED_UNARY_OPS:
            return None
        arg_expr = rec(inner)
        if arg_expr is None:
            return None
        return _unary_expr(op, arg_expr)

    return rec(txt)


def _build_nested_expr(pattern: str, base: Mapping[str, Any]) -> Dict[str, Any] | None:
    return _compose_nested_pattern(pattern, base)


def _normalize_nested_mode(mode: Any) -> str:
    if isinstance(mode, bool):
        return "auto" if bool(mode) else "manual"
    key = str(mode).strip().lower()
    if key in {"manual", "auto", "hybrid"}:
        return key
    raise ValueError("nested_mode must be one of: manual | auto | hybrid (or bool true/false)")


def _normalize_interaction_budget_mode(mode: Any) -> str:
    key = str(mode).strip().lower()
    if key in {"fixed", "interaction_first"}:
        return key
    raise ValueError("interaction_budget_mode must be one of: fixed | interaction_first")


def _normalize_candidate_pool_mode(mode: Any) -> str:
    key = str(mode).strip().lower()
    if key in {"shared", "shared_full", "full", "feature_space_full", "feature-space-full"}:
        return "shared_full"
    return "legacy"


def _collapse_residual_target_for_shared_pool(residual: np.ndarray) -> np.ndarray:
    target = _as_2d_float(residual)
    if int(target.shape[1]) == 1:
        return np.asarray(target[:, 0], dtype=float).reshape(-1)
    collapsed = np.asarray(np.mean(target, axis=1), dtype=float).reshape(-1)
    if not np.all(np.isfinite(collapsed)) or float(np.std(collapsed)) <= 1e-12:
        collapsed = np.asarray(target[:, 0], dtype=float).reshape(-1)
    return collapsed


def _build_shared_full_candidates(
    X: np.ndarray,
    residual: np.ndarray,
    *,
    cfg: StructureSearchConfig,
    feature_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    from core.symbolic.feature_space.activation_config import DynamicActivationConfig
    from core.symbolic.feature_space.builder import CandidatePoolConfig, build_full_candidate_pool
    from core.symbolic.feature_space.feature_bundle import FeatureBundle

    x = _as_2d_float(X)
    d = int(x.shape[1])
    names = tuple(feature_names or tuple(f"x{i}" for i in range(d)))
    if len(names) != d:
        names = tuple(f"x{i}" for i in range(d))

    residual_target = _collapse_residual_target_for_shared_pool(residual)
    bundle = FeatureBundle(
        X_train=np.asarray(x, dtype=float),
        y_train=np.asarray(residual_target, dtype=float).reshape(-1, 1),
        X_test=np.asarray(x, dtype=float),
        y_test=np.asarray(residual_target, dtype=float).reshape(-1, 1),
        feature_names=tuple(str(value) for value in names),
        n_features_raw=int(d),
        feature_names_raw=tuple(str(value) for value in names),
        lag_added_features=tuple(),
        lag_cross_added_features=tuple(),
        dropped_features=tuple(),
    )
    pool_cfg = CandidatePoolConfig(
        dynamic_pool_enabled=True,
        dynamic_init_minimal=False,
        safe_log1p_abs=bool(cfg.shared_pool_safe_log1p_abs),
        safe_exp_clip=bool(cfg.shared_pool_safe_exp_clip),
        safe_reciprocal=bool(cfg.shared_pool_safe_reciprocal),
        dynamic_activation=DynamicActivationConfig(),
        conditional_config=None,
    )
    candidates = list(build_full_candidate_pool(bundle, pool_cfg))
    candidates.sort(key=lambda item: (-float(item.prior_corr), float(item.complexity), str(item.name)))

    out: list[dict[str, Any]] = []
    for item in candidates:
        features = tuple(sorted(int(v) for v in tuple(item.features)))
        out.append(
            {
                "name": str(item.name),
                "expr": dict(item.expr),
                "family": str(item.family),
                "features": features,
                "feature_labels": _feature_labels(features, names),
                "prior_corr": float(item.prior_corr),
                "candidate_pool_mode": "shared_full",
            }
        )
    return out


def _ops_chain_to_pattern(ops: Sequence[str]) -> str:
    s = "x"
    for op in reversed(tuple(str(v) for v in ops)):
        s = f"{op}({s})"
    return s


def _auto_nested_expr_specs(
    *,
    base: Mapping[str, Any],
    cfg: StructureSearchConfig,
) -> list[dict[str, Any]]:
    max_patterns = int(max(0, cfg.auto_nested_max_patterns_per_feature))
    if max_patterns <= 0:
        return []

    allowed_ops: list[str] = []
    seen_ops: set[str] = set()
    for raw in tuple(cfg.auto_nested_allowed_ops):
        op = str(raw).strip().lower()
        if not op or op == "identity":
            continue
        if op not in _ALLOWED_NESTED_UNARY_OPS:
            continue
        if op in seen_ops:
            continue
        seen_ops.add(op)
        allowed_ops.append(op)
    if not allowed_ops:
        return []

    min_depth = int(max(2, cfg.auto_nested_min_depth))
    max_depth = int(max(min_depth, cfg.auto_nested_max_depth))
    max_ops_by_expr_depth = int(max(1, cfg.max_expr_depth - 1))
    max_depth = int(min(max_depth, max_ops_by_expr_depth))
    if max_depth < min_depth:
        return []

    beam = int(max(1, cfg.auto_nested_beam_width))
    out: list[dict[str, Any]] = []
    seen_expr: set[str] = set()

    # depth=1 layer (single unary wrapper)
    layer: list[dict[str, Any]] = []
    for op in allowed_ops:
        expr = _unary_expr(op, base)
        if not _candidate_allowed(expr, cfg=cfg):
            continue
        layer.append({"ops": (op,), "expr": expr})

    depth = 1
    while layer and depth <= max_depth:
        if depth >= min_depth:
            for node in layer:
                expr = dict(node["expr"])
                key = _expr_key(expr)
                if key in seen_expr:
                    continue
                out.append(
                    {
                        "pattern": _ops_chain_to_pattern(tuple(node["ops"])),
                        "expr": expr,
                    }
                )
                seen_expr.add(key)
                if len(out) >= max_patterns:
                    return out

        if depth >= max_depth:
            break

        next_layer_raw: list[dict[str, Any]] = []
        for node in layer:
            ops_old = tuple(str(v) for v in node["ops"])
            expr_old = dict(node["expr"])
            for op in allowed_ops:
                expr_new = _unary_expr(op, expr_old)
                if not _candidate_allowed(expr_new, cfg=cfg):
                    continue
                next_layer_raw.append(
                    {
                        "ops": (op, *ops_old),
                        "expr": expr_new,
                    }
                )

        dedup: list[dict[str, Any]] = []
        layer_seen: set[str] = set()
        for node in next_layer_raw:
            key = _expr_key(node["expr"])
            if key in layer_seen:
                continue
            dedup.append(node)
            layer_seen.add(key)
        layer = dedup[:beam]
        depth += 1

    return out


def _iter_nested_expr_specs(
    *,
    base: Mapping[str, Any],
    cfg: StructureSearchConfig,
) -> list[dict[str, Any]]:
    mode = _normalize_nested_mode(cfg.nested_mode)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def push(name: str, family: str, expr: Mapping[str, Any]) -> None:
        key = _expr_key(expr)
        if key in seen:
            return
        if not _candidate_allowed(expr, cfg=cfg):
            return
        out.append(
            {
                "name": str(name),
                "family": str(family),
                "expr": dict(expr),
            }
        )
        seen.add(key)

    if mode in {"manual", "hybrid"}:
        for pattern in cfg.nested_unary_patterns:
            expr = _build_nested_expr(str(pattern), base)
            if expr is None:
                continue
            push(name=f"{pattern}", family=f"nested:manual:{str(pattern)}", expr=expr)

    if mode in {"auto", "hybrid"}:
        auto_specs = _auto_nested_expr_specs(base=base, cfg=cfg)
        for spec in auto_specs:
            pattern = str(spec["pattern"])
            expr = dict(spec["expr"])
            push(name=pattern, family=f"nested:auto:{pattern}", expr=expr)

    return out


def _residual_1d_proxy(residual: np.ndarray) -> np.ndarray:
    r = _as_2d_float(residual)
    if int(r.shape[1]) == 1:
        return np.asarray(r[:, 0], dtype=float)
    return np.asarray(np.mean(r, axis=1), dtype=float)


def _interaction_dominance_diag(
    *,
    X: np.ndarray,
    residual: np.ndarray,
    selected: Sequence[int],
    cfg: StructureSearchConfig,
    is_binary: np.ndarray,
    feature_names: Sequence[str] | None = None,
    graph_cache: ExpressionGraphCache | None = None,
    batch_key: str | None = None,
) -> dict[str, Any]:
    x = _as_2d_float(X)
    r = _residual_1d_proxy(residual)
    if int(x.shape[0]) != int(r.shape[0]):
        raise ValueError("X and residual row mismatch in interaction diagnostic")

    d = int(x.shape[1])
    topk_diag = int(max(2, cfg.interaction_diag_topk_features))
    sel = [int(i) for i in selected if 0 <= int(i) < d][:topk_diag]
    if len(sel) < 2:
        return {
            "enabled": True,
            "mode": str(cfg.interaction_budget_mode),
            "selected_features": tuple(sel),
            "main_proxy": 0.0,
            "interaction_proxy": 0.0,
            "ratio": 0.0,
            "dominant_interaction": False,
            "threshold": float(cfg.interaction_diag_threshold),
        }

    main_best = 0.0
    for i in sel:
        main_best = max(main_best, abs(_safe_corr_masked(x[:, i], r)))
        lib = _feature_transform_library(
            X=x,
            feature_idx=int(i),
            feature_names=feature_names,
            cfg=cfg,
            is_binary=is_binary,
            graph_cache=graph_cache,
            batch_key=batch_key,
        )
        for item in lib:
            vec = np.asarray(item.get("vector", np.zeros((x.shape[0],), dtype=float)), dtype=float).reshape(-1)
            if vec.shape[0] != x.shape[0]:
                continue
            main_best = max(main_best, abs(_safe_corr_masked(vec, r)))

    interaction_best = 0.0
    best_pair: tuple[int, int] | None = None
    for i, j in combinations(sel, 2):
        z = np.asarray(x[:, int(i)] * x[:, int(j)], dtype=float)
        corr = abs(_safe_corr_masked(z, r))
        if corr > interaction_best:
            interaction_best = float(corr)
            best_pair = (int(i), int(j))

    ratio = float(interaction_best / (main_best + 1e-12))
    threshold = float(max(0.0, cfg.interaction_diag_threshold))
    dominant = bool(ratio >= threshold)
    return {
        "enabled": True,
        "mode": str(cfg.interaction_budget_mode),
        "selected_features": tuple(sel),
        "main_proxy": float(main_best),
        "interaction_proxy": float(interaction_best),
        "ratio": float(ratio),
        "dominant_interaction": bool(dominant),
        "threshold": float(threshold),
        "best_pair": tuple(best_pair) if best_pair is not None else None,
    }


def _safe_corr_masked(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("corr shape mismatch")
    valid = np.isfinite(x) & np.isfinite(y)
    if not bool(np.any(valid)):
        return 0.0
    return float(_safe_corr(x[valid], y[valid]))


def _candidate_allowed(expr: Mapping[str, Any], *, cfg: StructureSearchConfig) -> bool:
    arity = int(len(_expr_features(expr)))
    depth = int(_expr_depth(expr))
    if arity > int(max(1, cfg.max_arity)):
        return False
    if depth > int(max(1, cfg.max_expr_depth)):
        return False
    return True


def _feature_transform_library(
    *,
    X: np.ndarray,
    feature_idx: int,
    cfg: StructureSearchConfig,
    is_binary: np.ndarray,
    feature_names: Sequence[str] | None = None,
    graph_cache: ExpressionGraphCache | None = None,
    batch_key: str | None = None,
) -> list[dict[str, Any]]:
    x = _as_2d_float(X)
    j = int(feature_idx)
    if j < 0 or j >= int(x.shape[1]):
        return []

    base = _feature_expr(j)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    base_label = _feature_label(feature_names, j)

    def push(name: str, family: str, expr: Mapping[str, Any]) -> None:
        key = _expr_key(expr)
        if key in seen:
            return
        if graph_cache is None:
            vec = evaluate_expression_numpy(expr, x).reshape(-1)
        else:
            vec = graph_cache.evaluate_expression(
                expr,
                x,
                expr_key=key,
                batch_key=batch_key,
            ).reshape(-1)
        if not np.all(np.isfinite(vec)):
            return
        out.append(
            {
                "name": str(name),
                "family": str(family),
                "expr": dict(expr),
                "expr_key": str(key),
                "features": tuple(_expr_features(expr)),
                "feature_labels": _feature_labels(tuple(_expr_features(expr)), feature_names),
                "vector": np.asarray(vec, dtype=float),
            }
        )
        seen.add(key)

    push(name=base_label, family="identity", expr=base)

    if bool(is_binary[j]):
        return out

    for op in cfg.unary_ops:
        op_key = str(op).strip().lower()
        expr = _unary_expr(op_key, base)
        push(name=f"{op_key}({base_label})", family=f"unary:{op_key}", expr=expr)

    nested_specs = _iter_nested_expr_specs(base=base, cfg=cfg)
    for spec in nested_specs:
        nm = str(spec["name"])
        fam = str(spec["family"])
        expr = dict(spec["expr"])
        push(name=f"{nm}({base_label})", family=fam, expr=expr)

    return out


def _build_grad_projection_candidates(
    *,
    X: np.ndarray,
    cfg: StructureSearchConfig,
    gradient_signal: Any | None,
    residual_selected: Sequence[int],
    is_binary: np.ndarray,
    feature_names: Sequence[str] | None = None,
    max_generated_override: int | None = None,
    graph_cache: ExpressionGraphCache | None = None,
    batch_key: str | None = None,
) -> list[dict[str, Any]]:
    if gradient_signal is None:
        return []
    if not bool(cfg.enable_grad_residual_projection):
        return []
    if int(cfg.grad_projection_max_generated) <= 0 and max_generated_override is None:
        return []

    x = _as_2d_float(X)
    n, d = int(x.shape[0]), int(x.shape[1])
    if n <= 2 or d <= 1:
        return []

    p = np.asarray(getattr(gradient_signal, "feature_priority", np.zeros((d,), dtype=float)), dtype=float).reshape(-1)
    if p.size != d:
        p = np.zeros((d,), dtype=float)
    p_ms = np.asarray(
        getattr(gradient_signal, "feature_priority_multiscale", np.zeros((d,), dtype=float)),
        dtype=float,
    ).reshape(-1)
    if p_ms.size != d:
        p_ms = p.copy()
    st = np.asarray(getattr(gradient_signal, "feature_stability", np.ones((d,), dtype=float)), dtype=float).reshape(-1)
    if st.size != d:
        st = np.ones((d,), dtype=float)
    cross = np.asarray(getattr(gradient_signal, "cross_feature_priority", np.zeros((d, d), dtype=float)), dtype=float)
    if cross.shape != (d, d):
        cross = np.zeros((d, d), dtype=float)

    # Multi-scale + stability adjusted focus ranking.
    p_eff = np.maximum(0.0, 0.6 * p + 0.4 * p_ms) * np.clip(st, 0.2, 1.0)
    ranked_focus = list(np.argsort(-p))
    focus_topk = int(max(1, cfg.grad_projection_topk_focus))
    focus_idx = [int(i) for i in list(np.argsort(-p_eff))[:focus_topk]]

    partner_pool_size = int(max(2, cfg.grad_projection_partner_pool))
    partner_ranked_global = [int(i) for i in ranked_focus[:partner_pool_size]]
    for i in residual_selected:
        if int(i) not in partner_ranked_global:
            partner_ranked_global.append(int(i))
    partner_ranked_global = partner_ranked_global[: max(partner_pool_size, len(residual_selected))]

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    min_abs_corr = float(max(0.0, cfg.grad_projection_min_abs_corr))
    if max_generated_override is None:
        max_generated = int(max(1, cfg.grad_projection_max_generated))
    else:
        max_generated = int(max(1, max_generated_override))
    per_partner_topk = int(max(1, cfg.grad_projection_topk_unary))
    cross_topk = int(max(1, cfg.grad_projection_topk_partners))
    focus_topk = int(max(1, cfg.grad_projection_focus_topk_transforms))
    focus_include_transforms = bool(cfg.grad_projection_focus_include_transforms)
    max_partner_k = max(1, int(max(1, cfg.max_arity)) - 1)
    raw_orders = tuple(int(v) for v in cfg.grad_projection_partner_orders)
    partner_orders = sorted({int(v) for v in raw_orders if int(v) >= 1 and int(v) <= max_partner_k})
    if not partner_orders:
        partner_orders = [1]
    enable_pair_dictionary = bool(cfg.grad_projection_enable_pair_dictionary)

    for i in focus_idx:
        if i < 0 or i >= d:
            continue
        gaps = np.asarray(gradient_signal.gap_by_feature[i], dtype=float)
        if gaps.ndim != 2 or gaps.shape[0] != n:
            continue
        # u_i: row-level gradient mismatch mean across targets.
        valid_gap = np.isfinite(gaps)
        valid_count = np.sum(valid_gap, axis=1)
        u = np.full((n,), np.nan, dtype=float)
        has_valid = valid_count > 0
        if bool(np.any(has_valid)):
            num = np.sum(np.where(valid_gap, gaps, 0.0), axis=1)
            u[has_valid] = num[has_valid] / valid_count[has_valid]
        if not np.any(np.isfinite(u)):
            continue

        # Focus-side transform library: x_i -> g(x_i). Keeps legacy x_i when disabled.
        focus_items: list[dict[str, Any]] = [
            {
                "name": _feature_label(feature_names, int(i)),
                "family": "identity",
                "expr": _feature_expr(i),
                "features": (i,),
                "feature_labels": (_feature_label(feature_names, int(i)),),
                "abs_corr_u": abs(_safe_corr_masked(u, x[:, i])),
                "focus_transformed": False,
            }
        ]
        if focus_include_transforms:
            focus_lib = _feature_transform_library(
                X=x,
                feature_idx=int(i),
                feature_names=feature_names,
                cfg=cfg,
                is_binary=is_binary,
                graph_cache=graph_cache,
                batch_key=batch_key,
            )
            focus_scored: list[dict[str, Any]] = []
            for item in focus_lib:
                if int(len(item.get("features", ()))) <= 0:
                    continue
                corr_u = abs(_safe_corr_masked(u, np.asarray(item["vector"], dtype=float)))
                if corr_u < (0.5 * min_abs_corr) and str(item.get("family", "")) != "identity":
                    continue
                focus_scored.append(
                    {
                        "name": str(item["name"]),
                        "family": str(item["family"]),
                        "expr": dict(item["expr"]),
                        "features": tuple(int(v) for v in item.get("features", ())),
                        "feature_labels": tuple(str(v) for v in tuple(item.get("feature_labels", ()))),
                        "abs_corr_u": float(corr_u),
                        "focus_transformed": str(item.get("family", "")) != "identity",
                    }
                )
            focus_scored.sort(key=lambda v: float(v["abs_corr_u"]), reverse=True)
            keep: list[dict[str, Any]] = []
            seen_focus: set[str] = set()
            for it in focus_scored:
                kf = _expr_key(it["expr"])
                if kf in seen_focus:
                    continue
                keep.append(it)
                seen_focus.add(kf)
                if len(keep) >= focus_topk:
                    break
            # Ensure legacy identity path always remains available.
            if not any(str(v.get("family", "")) == "identity" for v in keep):
                keep.append(
                    {
                        "name": _feature_label(feature_names, int(i)),
                        "family": "identity",
                        "expr": _feature_expr(i),
                        "features": (i,),
                        "feature_labels": (_feature_label(feature_names, int(i)),),
                        "abs_corr_u": abs(_safe_corr_masked(u, x[:, i])),
                        "focus_transformed": False,
                    }
                )
            focus_items = keep

        # Partner ranking is focus-conditional: prioritize high cross-feature gradient relevance.
        if int(i) >= 0 and int(i) < cross.shape[0]:
            local_partner_rank = np.argsort(-cross[int(i), :]).tolist()
            partner_ranked = [int(j) for j in local_partner_rank if int(j) != int(i)]
            for j in partner_ranked_global:
                if int(j) not in partner_ranked:
                    partner_ranked.append(int(j))
        else:
            partner_ranked = list(partner_ranked_global)
        partner_ranked = partner_ranked[: max(partner_pool_size, len(residual_selected))]

        hits: list[dict[str, Any]] = []
        for j in partner_ranked:
            if j == i:
                continue
            if j < 0 or j >= d:
                continue
            lib = _feature_transform_library(
                X=x,
                feature_idx=int(j),
                feature_names=feature_names,
                cfg=cfg,
                is_binary=is_binary,
                graph_cache=graph_cache,
                batch_key=batch_key,
            )
            if not lib:
                continue

            scored: list[dict[str, Any]] = []
            for item in lib:
                corr = abs(_safe_corr_masked(u, np.asarray(item["vector"], dtype=float)))
                if corr < min_abs_corr:
                    continue
                scored.append(
                    {
                        **item,
                        "abs_corr_u": float(corr),
                        "partner": int(j),
                        "cross_priority": float(cross[int(i), int(j)]) if 0 <= int(i) < d and 0 <= int(j) < d else 0.0,
                    }
                )
            scored.sort(
                key=lambda v: (
                    0.85 * float(v.get("abs_corr_u", 0.0)) + 0.15 * float(v.get("cross_priority", 0.0))
                ),
                reverse=True,
            )
            hits.extend(scored[:per_partner_topk])

        if not hits:
            continue

        hits.sort(key=lambda v: float(v["abs_corr_u"]), reverse=True)
        top_hits = hits[: max(1, cross_topk)]
        if not top_hits:
            continue

        for order_k in partner_orders:
            if order_k >= 2 and not enable_pair_dictionary:
                continue
            need_n = max(order_k, cross_topk)
            seed_hits = hits[: max(order_k, need_n)]
            if len(seed_hits) < order_k:
                continue
            for combo in combinations(seed_hits, order_k):
                partners = [int(h["partner"]) for h in combo]
                if len(set(partners)) != len(partners):
                    continue

                prod_expr = dict(combo[0]["expr"])
                for h in combo[1:]:
                    prod_expr = _binary_expr("mul", prod_expr, h["expr"])
                combo_names = [str(h["name"]) for h in combo]
                combo_corr = [float(h["abs_corr_u"]) for h in combo]
                combo_corr_proxy = float(np.prod(np.asarray(combo_corr, dtype=float)))

                for focus in focus_items:
                    expr = _binary_expr("mul", dict(focus["expr"]), prod_expr)
                    if not _candidate_allowed(expr, cfg=cfg):
                        continue

                    key = _expr_key(expr)
                    if key in seen:
                        continue

                    feature_union: set[int] = set(int(v) for v in focus.get("features", (int(i),)))
                    for h in combo:
                        feature_union.update(int(v) for v in h.get("features", ()))

                    out.append(
                        {
                            "name": f"{str(focus['name'])}*{'*'.join(f'({n})' for n in combo_names)}",
                            "expr": expr,
                            "family": (
                                f"interaction:grad_projected_focus_order{order_k + 1}"
                                if bool(focus.get("focus_transformed", False))
                                else f"interaction:grad_projected_order{order_k + 1}"
                            ),
                            "features": tuple(sorted(feature_union)),
                            "feature_labels": _feature_labels(tuple(sorted(feature_union)), feature_names),
                            "grad_projection": {
                                "focus_feature": int(i),
                                "focus_transform": str(focus["name"]),
                                "focus_family": str(focus.get("family", "identity")),
                                "focus_transformed": bool(focus.get("focus_transformed", False)),
                                "focus_abs_corr_u": float(focus.get("abs_corr_u", 0.0)),
                                "partner_order": int(order_k),
                                "partner_features": partners,
                                "partner_transforms": combo_names,
                                "abs_corr_u_combo_proxy": float(combo_corr_proxy),
                            },
                        }
                    )
                    seen.add(key)
                    if len(out) >= max_generated:
                        return out

    return out


def _finalize_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    cfg: StructureSearchConfig,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cand in candidates:
        expr = cand["expr"]
        if not _candidate_allowed(expr, cfg=cfg):
            continue
        key = _expr_key(expr)
        if key in seen:
            continue
        c = dict(cand)
        c["features"] = tuple(int(i) for i in _expr_features(expr))
        c["expr_key"] = str(key)
        c["arity"] = int(len(c["features"]))
        c["expr_depth"] = int(_expr_depth(expr))
        out.append(c)
        seen.add(key)
        if len(out) >= int(cfg.max_candidates_per_iter):
            break
    return out


def _build_candidates(
    X: np.ndarray,
    residual: np.ndarray,
    *,
    cfg: StructureSearchConfig,
    feature_names: Sequence[str] | None = None,
    gradient_signal: Any | None = None,
    graph_cache: ExpressionGraphCache | None = None,
    batch_key: str | None = None,
) -> list[dict[str, Any]]:
    x = _as_2d_float(X)
    d = int(x.shape[1])
    scores = _feature_residual_scores(x, residual)
    ranked = list(np.argsort(-scores))
    candidate_pool_mode = _normalize_candidate_pool_mode(cfg.candidate_pool_mode)

    topk = max(1, min(int(cfg.topk_features), d))
    selected = [int(i) for i in ranked[:topk]]

    is_binary = detect_binary_columns(x)
    budget_mode = _normalize_interaction_budget_mode(cfg.interaction_budget_mode)
    interaction_diag = {
        "enabled": False,
        "mode": str(budget_mode),
        "dominant_interaction": False,
        "ratio": 0.0,
        "threshold": float(cfg.interaction_diag_threshold),
        "main_proxy": 0.0,
        "interaction_proxy": 0.0,
        "selected_features": tuple(selected),
    }
    effective_max_pair_terms = int(max(0, cfg.max_pair_terms))
    effective_grad_projection_max_generated = int(max(0, cfg.grad_projection_max_generated))
    interaction_dominant = False
    if budget_mode == "interaction_first":
        interaction_diag = _interaction_dominance_diag(
            X=x,
            residual=residual,
            selected=tuple(selected),
            cfg=cfg,
            is_binary=is_binary,
            feature_names=feature_names,
            graph_cache=graph_cache,
            batch_key=batch_key,
        )
        interaction_dominant = bool(interaction_diag.get("dominant_interaction", False))
        if interaction_dominant:
            effective_max_pair_terms = int(
                max(
                    1,
                    round(float(max(1, cfg.max_pair_terms)) * float(max(1.0, cfg.interaction_pair_budget_boost))),
                )
            )
            effective_grad_projection_max_generated = int(
                max(
                    1,
                    round(
                        float(max(1, cfg.grad_projection_max_generated))
                        * float(max(1.0, cfg.interaction_grad_projection_budget_boost))
                    ),
                )
            )
    out: list[dict[str, Any]] = []

    if candidate_pool_mode == "shared_full":
        out.extend(
            _build_shared_full_candidates(
                x,
                residual,
                cfg=cfg,
                feature_names=feature_names,
            )
        )
    else:
        # unary terms
        for i in selected:
            base = _feature_expr(i)
            base_label = _feature_label(feature_names, int(i))
            for op in cfg.unary_ops:
                op_key = str(op).strip().lower()
                expr = _unary_expr(op_key, base)
                out.append(
                    {
                        "name": f"{op_key}({base_label})",
                        "expr": expr,
                        "family": f"unary:{op_key}",
                        "features": (i,),
                        "feature_labels": (base_label,),
                    }
                )

            nested_specs = _iter_nested_expr_specs(base=base, cfg=cfg)
            for spec in nested_specs:
                nm = str(spec["name"])
                fam = str(spec["family"])
                expr = dict(spec["expr"])
                out.append(
                    {
                        "name": f"{nm}({base_label})",
                        "expr": expr,
                        "family": fam,
                        "features": (i,),
                        "feature_labels": (base_label,),
                    }
                )

        # pair interactions
        pair_terms: list[tuple[float, int, int]] = []
        for ai in range(len(selected)):
            for aj in range(ai + 1, len(selected)):
                i = selected[ai]
                j = selected[aj]
                rank = float(scores[i] * scores[j])
                pair_terms.append((rank, i, j))
        pair_terms.sort(key=lambda x: x[0], reverse=True)

        for _, i, j in pair_terms[: max(0, int(effective_max_pair_terms))]:
            left_label = _feature_label(feature_names, int(i))
            right_label = _feature_label(feature_names, int(j))
            out.append(
                {
                    "name": f"({left_label})*({right_label})",
                    "expr": _binary_expr("mul", _feature_expr(i), _feature_expr(j)),
                    "family": "interaction:mul",
                    "features": tuple(sorted((i, j))),
                    "feature_labels": tuple(sorted((left_label, right_label))),
                }
            )

    # hinge terms
    if bool(cfg.include_hinge):
        qs = sorted({float(q) for q in cfg.hinge_quantiles if 0.0 < float(q) < 1.0})
        if qs:
            for i in selected:
                if bool(is_binary[i]):
                    continue
                col = x[:, i]
                base_label = _feature_label(feature_names, int(i))
                for q in qs:
                    thr = float(np.quantile(col, q))
                    shifted = _binary_expr("sub", _feature_expr(i), _const_expr(thr))
                    out.append(
                        {
                            "name": f"relu({base_label}-{thr:.4g})",
                            "expr": _relu_expr(shifted),
                            "family": "piecewise:hinge",
                            "features": (i,),
                            "feature_labels": (base_label,),
                        }
                    )

    grad_projected = _build_grad_projection_candidates(
        X=x,
        cfg=cfg,
        gradient_signal=gradient_signal,
        residual_selected=tuple(selected),
        feature_names=feature_names,
        is_binary=is_binary,
        max_generated_override=effective_grad_projection_max_generated,
        graph_cache=graph_cache,
        batch_key=batch_key,
    )
    if grad_projected:
        out.extend(grad_projected)

    if interaction_dominant:
        inter = [dict(c) for c in out if str(c.get("family", "")).startswith("interaction:")]
        non_inter = [dict(c) for c in out if not str(c.get("family", "")).startswith("interaction:")]
        out = inter + non_inter

    for c in out:
        c["budget_policy"] = {
            "mode": str(budget_mode),
            "candidate_pool_mode": str(candidate_pool_mode),
            "interaction_dominant": bool(interaction_dominant),
            "interaction_diag": dict(interaction_diag),
            "effective_max_pair_terms": int(effective_max_pair_terms),
            "effective_grad_projection_max_generated": int(effective_grad_projection_max_generated),
        }

    return _finalize_candidates(out, cfg=cfg)


def _score_candidate(
    candidate: Mapping[str, Any],
    *,
    X: np.ndarray,
    residual: np.ndarray,
    cfg: StructureSearchConfig,
    gradient_correction: GradientCorrection | None = None,
    structure_optimizer: StructureOptimizer | None = None,
    x_scale: np.ndarray | None = None,
    grad_adv_config: Mapping[str, Any] | None = None,
    rng: np.random.Generator | None = None,
    path_memory: SymbolicPathMemory | None = None,
    graph_cache: ExpressionGraphCache | None = None,
    batch_key: str | None = None,
) -> dict[str, Any] | None:
    expr = candidate["expr"]
    expr_key = str(candidate.get("expr_key", _expr_key(expr)))

    if graph_cache is None:
        z = evaluate_expression_numpy(expr, X).reshape(-1)
    else:
        z = graph_cache.evaluate_expression(
            expr,
            X,
            expr_key=expr_key,
            batch_key=batch_key,
        ).reshape(-1)
    if not np.all(np.isfinite(z)):
        return None

    zc = z - float(np.mean(z))
    ztz = float(np.dot(zc, zc))
    if ztz <= 1e-12:
        return None

    R = _as_2d_float(residual)
    dots = np.asarray(zc.reshape(1, -1) @ R, dtype=float).reshape(-1)
    projected_gain = float(np.sum((dots**2) / (ztz + 1e-12)) / max(1, int(R.shape[0])))

    corrs: list[float] = []
    for k in range(int(R.shape[1])):
        corrs.append(_safe_corr(zc, R[:, k]))
    abs_corr = float(np.mean(np.abs(corrs)))
    signed_corr = float(np.mean(corrs))

    if projected_gain < float(cfg.min_projected_gain):
        return None

    complexity = float(_expr_node_count(expr) + 0.5 * _expr_depth(expr))
    coeff_vector = np.asarray(dots / (ztz + 1e-12), dtype=float)

    grad_alignment = 0.0
    grad_details: dict[str, Any] = {
        "grad_alignment": 0.0,
        "used_features": [],
        "per_feature": [],
    }
    grad_adversarial: dict[str, Any] = {
        "enabled": False,
        "applied": False,
    }

    if gradient_correction is not None:
        grad_details = gradient_correction.score_candidate(
            expr=expr,
            X=X,
            coeff_vector=coeff_vector,
            feature_indices=tuple(int(i) for i in candidate.get("features", _expr_features(expr))),
        )
        grad_alignment = float(grad_details.get("grad_alignment", 0.0))

        if grad_adv_config is not None and bool(grad_adv_config.get("enabled", False)):
            trials = int(max(0, grad_adv_config.get("trials", 0)))
            noise_std = float(max(0.0, grad_adv_config.get("noise_std", 0.0)))
            min_stability = float(max(0.0, grad_adv_config.get("min_stability", 0.0)))
            seed = int(grad_adv_config.get("seed", 42))

            grad_adversarial["enabled"] = True
            grad_adversarial.update(
                {
                    "trials": int(trials),
                    "noise_std": float(noise_std),
                    "min_stability": float(min_stability),
                    "seed": int(seed),
                }
            )

            if trials > 0 and noise_std > 0.0:
                base_x = _as_2d_float(np.asarray(X, dtype=float))
                scale = (
                    np.asarray(x_scale, dtype=float).reshape(1, -1)
                    if x_scale is not None
                    else (np.std(base_x, axis=0, keepdims=True) + 1e-8)
                )
                local_rng = rng if rng is not None else np.random.default_rng(int(seed))

                aligns = [float(grad_alignment)]
                for _ in range(trials):
                    noise = local_rng.normal(0.0, noise_std, size=base_x.shape) * scale
                    x_adv = base_x + noise
                    detail_adv = gradient_correction.score_candidate(
                        expr=expr,
                        X=x_adv,
                        coeff_vector=coeff_vector,
                        feature_indices=tuple(int(i) for i in candidate.get("features", _expr_features(expr))),
                    )
                    aligns.append(float(detail_adv.get("grad_alignment", 0.0)))

                align_arr = np.asarray(aligns, dtype=float)
                align_median = float(np.median(align_arr))
                align_std = float(np.std(align_arr))
                stability = float(1.0 / (1.0 + align_std))

                grad_adversarial.update(
                    {
                        "applied": True,
                        "samples": [float(v) for v in align_arr.tolist()],
                        "median": float(align_median),
                        "std": float(align_std),
                        "stability": float(stability),
                    }
                )

                if stability < min_stability:
                    return None

                grad_alignment = float(align_median * stability)

    optimizer = structure_optimizer or StructureOptimizer(
        StructureScoreConfig(
            score_corr_bonus=float(cfg.score_corr_bonus),
            score_complexity_penalty=float(cfg.score_complexity_penalty),
            score_grad_guidance_bonus=float(cfg.score_grad_guidance_bonus),
        )
    )
    combined = optimizer.combine(
        projected_gain=float(projected_gain),
        abs_corr=float(abs_corr),
        complexity=float(complexity),
        grad_alignment=float(grad_alignment),
    )

    base_score = float(combined["score"])
    prior_bonus = 0.0
    tabu_penalty = 0.0
    hard_tabu = False
    prior_info: dict[str, Any] = {
        "enabled": False,
    }

    if path_memory is not None:
        prior = path_memory.get_expr_prior(expr_key)
        outcomes = int(prior.outcomes)
        prior_info = {
            "enabled": True,
            **prior.to_dict(),
        }

        min_outcomes = int(max(0, cfg.path_memory_min_outcomes))
        if outcomes >= min_outcomes:
            accept_rate = float(prior.accept_rate)
            strength = float(np.log1p(outcomes))
            prior_bonus = float(cfg.path_memory_prior_bonus) * accept_rate * strength
            tabu_penalty = float(cfg.path_memory_tabu_penalty) * (1.0 - accept_rate) * strength
            hard_tabu = bool(cfg.path_memory_hard_tabu) and (accept_rate <= float(cfg.path_memory_hard_tabu_accept_rate))

        prior_info.update(
            {
                "min_outcomes": int(min_outcomes),
                "prior_bonus": float(prior_bonus),
                "tabu_penalty": float(tabu_penalty),
                "hard_tabu": bool(hard_tabu),
            }
        )

        if hard_tabu:
            return None

    score = float(base_score + prior_bonus - tabu_penalty)

    return {
        "name": str(candidate["name"]),
        "family": str(candidate["family"]),
        "features": tuple(int(i) for i in candidate.get("features", _expr_features(expr))),
        "feature_labels": tuple(
            str(v)
            for v in tuple(
                candidate.get(
                    "feature_labels",
                    _feature_labels(tuple(int(i) for i in candidate.get("features", _expr_features(expr))), None),
                )
            )
        ),
        "expr": dict(expr),
        "expr_key": str(expr_key),
        "projected_gain": float(projected_gain),
        "abs_corr": float(abs_corr),
        "signed_corr": float(signed_corr),
        "complexity": float(complexity),
        "score": float(score),
        "score_parts": {
            **dict(combined["score_parts"]),
            "path_prior_bonus": float(prior_bonus),
            "path_tabu_penalty": float(tabu_penalty),
        },
        "score_raw": float(base_score),
        "grad_alignment": float(grad_alignment),
        "grad_details": dict(grad_details),
        "grad_adversarial": dict(grad_adversarial),
        "path_prior": dict(prior_info),
        "coeff_vector": coeff_vector.tolist(),
    }


def evaluate_genome_with_ridge(
    genome: Sequence[Mapping[str, Any]],
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
    l2: float = 1e-4,
    graph_cache: ExpressionGraphCache | None = None,
    train_batch_key: str | None = None,
    eval_batch_key: str | None = None,
) -> dict[str, Any]:
    xtr = _as_2d_float(X_train)
    ytr = _as_2d_float(y_train)
    if xtr.shape[0] != ytr.shape[0]:
        raise ValueError("X_train and y_train row mismatch")

    terms = list(genome)
    if terms:
        normalized = normalize_genome(terms, input_dim=int(xtr.shape[1]))
    else:
        normalized = []

    phi_tr = _design_matrix_from_genome(
        normalized,
        xtr,
        graph_cache=graph_cache,
        batch_key=train_batch_key,
    )
    fit = _fit_ridge_readout(phi_tr, ytr, l2=float(l2))

    out: dict[str, Any] = {
        "weight": np.asarray(fit["weight"], dtype=float),
        "bias": np.asarray(fit["bias"], dtype=float),
        "pred_train": np.asarray(fit["pred"], dtype=float),
        "metrics_train": _regression_metrics(ytr, fit["pred"]),
    }

    if X_eval is not None:
        xev = _as_2d_float(X_eval)
        if xev.shape[1] != xtr.shape[1]:
            raise ValueError("X_eval feature dim mismatch")
        phi_ev = _design_matrix_from_genome(
            normalized,
            xev,
            graph_cache=graph_cache,
            batch_key=eval_batch_key,
        )
        pred_ev = phi_ev @ np.asarray(fit["weight"], dtype=float) + np.asarray(fit["bias"], dtype=float)
        out["pred_eval"] = np.asarray(pred_ev, dtype=float)
        if y_eval is not None:
            yev = _as_2d_float(y_eval)
            out["metrics_eval"] = _regression_metrics(yev, pred_ev)

    return out


def _candidate_log_row(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": str(item["name"]),
        "family": str(item["family"]),
        "feature_indices": [int(v) for v in tuple(item.get("features", ()))],
        "feature_labels": [str(v) for v in tuple(item.get("feature_labels", ()))],
        "score": float(item["score"]),
        "score_raw": float(item.get("score_raw", item["score"])),
        "projected_gain": float(item["projected_gain"]),
        "abs_corr": float(item["abs_corr"]),
        "signed_corr": float(item["signed_corr"]),
        "complexity": float(item["complexity"]),
        "grad_alignment": float(item.get("grad_alignment", 0.0)),
        "grad_stability": float(item.get("grad_adversarial", {}).get("stability", 1.0)),
        "score_parts": dict(item.get("score_parts", {})),
        "path_prior": dict(item.get("path_prior", {})),
    }


def _soft_threshold(v: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(v, dtype=float)
    a = float(max(0.0, alpha))
    return np.sign(x) * np.maximum(np.abs(x) - a, 0.0)


def _l1_cd_proxy(
    A: np.ndarray,
    b: np.ndarray,
    *,
    alpha: float,
    iters: int,
) -> np.ndarray:
    X = np.asarray(A, dtype=float)
    y = np.asarray(b, dtype=float).reshape(-1)
    n, p = X.shape
    if n <= 0 or p <= 0:
        return np.zeros((p,), dtype=float)
    beta = np.zeros((p,), dtype=float)
    r = y.copy()
    col_norm = np.sum(X * X, axis=0) + 1e-12
    for _ in range(int(max(1, iters))):
        for j in range(p):
            xj = X[:, j]
            rho = float(np.dot(xj, r) + col_norm[j] * beta[j])
            bj = float(_soft_threshold(np.asarray([rho]), alpha=float(alpha))[0] / col_norm[j])
            delta = bj - float(beta[j])
            if abs(delta) > 0.0:
                r -= xj * delta
                beta[j] = bj
    return np.asarray(beta, dtype=float)


def _build_joint_bundle_candidates(
    shortlisted: Sequence[Mapping[str, Any]],
    *,
    X: np.ndarray,
    residual: np.ndarray,
    cfg: StructureSearchConfig,
    graph_cache: ExpressionGraphCache | None = None,
    batch_key: str | None = None,
) -> list[dict[str, Any]]:
    if not bool(cfg.joint_bundle_enabled):
        return []
    items = [dict(v) for v in shortlisted]
    if len(items) < 2:
        return []

    pre_k = int(max(2, cfg.joint_bundle_preselect_topk))
    items = items[: min(len(items), pre_k)]

    r1 = _residual_1d_proxy(residual)
    cols: list[np.ndarray] = []
    valid_items: list[dict[str, Any]] = []
    for it in items:
        expr = dict(it["expr"])
        expr_key = str(it.get("expr_key", _expr_key(expr)))
        if graph_cache is None:
            z = evaluate_expression_numpy(expr, X).reshape(-1)
        else:
            z = graph_cache.evaluate_expression(expr, X, expr_key=expr_key, batch_key=batch_key).reshape(-1)
        z = np.asarray(z, dtype=float)
        if not np.all(np.isfinite(z)):
            continue
        zc = z - float(np.mean(z))
        if float(np.dot(zc, zc)) <= 1e-12:
            continue
        cols.append(zc)
        valid_items.append(it)

    if len(valid_items) < 2:
        return []

    A = np.asarray(np.stack(cols, axis=1), dtype=float)
    scale = np.std(A, axis=0, ddof=0) + 1e-8
    A_std = A / scale.reshape(1, -1)
    beta = _l1_cd_proxy(
        A_std,
        r1,
        alpha=float(max(0.0, cfg.joint_bundle_l1_alpha)),
        iters=int(max(1, cfg.joint_bundle_l1_iters)),
    )
    beta_raw = beta / scale
    active_rank = list(np.argsort(-np.abs(beta_raw)))
    active = [int(i) for i in active_rank if abs(float(beta_raw[int(i)])) > 0.0]
    if len(active) < 2:
        # fallback: use score top-k
        active = list(range(min(len(valid_items), pre_k)))
    active = active[: min(len(active), pre_k)]
    if len(active) < 2:
        return []

    max_terms = int(max(2, cfg.joint_bundle_max_terms))
    max_terms = min(max_terms, len(active))
    max_combos = int(max(1, cfg.joint_bundle_max_combos))
    all_combos: list[tuple[int, ...]] = []
    for k in range(2, max_terms + 1):
        for cb in combinations(active, k):
            all_combos.append(tuple(int(i) for i in cb))
            if len(all_combos) >= max_combos:
                break
        if len(all_combos) >= max_combos:
            break

    if not all_combos:
        return []

    base_proxy_rmse = float(np.sqrt(np.mean(r1**2)))
    out: list[dict[str, Any]] = []
    for cb in all_combos:
        phi = np.asarray(A[:, list(cb)], dtype=float)
        fit = _fit_ridge_readout(phi, r1.reshape(-1, 1), l2=float(cfg.ridge_l2))
        pred = np.asarray(fit["pred"], dtype=float).reshape(-1)
        rmse = float(np.sqrt(np.mean((pred - r1) ** 2)))
        gain = float(base_proxy_rmse - rmse)
        if gain < float(cfg.min_projected_gain):
            continue
        members = [dict(valid_items[int(i)]) for i in cb]
        fam = sorted({str(v.get("family", "unknown")) for v in members})
        complexity = float(sum(float(v.get("complexity", 1.0)) for v in members))
        score_sum = float(sum(float(v.get("score", 0.0)) for v in members))
        out.append(
            {
                "name": "bundle(" + " + ".join(str(v.get("name", "?")) for v in members) + ")",
                "family": "joint_bundle:" + "|".join(fam),
                "projected_gain": float(gain),
                "abs_corr": float(np.mean([float(v.get("abs_corr", 0.0)) for v in members])),
                "signed_corr": float(np.mean([float(v.get("signed_corr", 0.0)) for v in members])),
                "complexity": float(complexity),
                "score_raw": float(score_sum),
                "score": float(score_sum + float(gain)),
                "score_parts": {
                    "bundle_score_sum": float(score_sum),
                    "bundle_proxy_gain": float(gain),
                },
                "bundle_items": members,
                "bundle_size": int(len(members)),
                "features": tuple(sorted({int(i) for v in members for i in v.get("features", ())})),
                "feature_labels": tuple(
                    dict.fromkeys(
                        str(label)
                        for v in members
                        for label in tuple(v.get("feature_labels", ()))
                        if str(label).strip()
                    ).keys()
                ),
                "expr_key": "bundle::" + "||".join(str(v.get("expr_key", "")) for v in members),
                "path_prior": {"enabled": False, "mode": "bundle"},
                "grad_alignment": float(np.mean([float(v.get("grad_alignment", 0.0)) for v in members])),
            }
        )

    out.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    return out[:max_combos]


def _fit_with_metrics(
    genome: Sequence[Mapping[str, Any]],
    *,
    X: np.ndarray,
    y: np.ndarray,
    l2: float,
    graph_cache: ExpressionGraphCache | None = None,
    batch_key: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    phi = _design_matrix_from_genome(
        genome,
        X,
        graph_cache=graph_cache,
        batch_key=batch_key,
    )
    fit = _fit_ridge_readout(phi, y, l2=float(l2))
    metrics = _regression_metrics(y, fit["pred"])
    return fit, metrics


def _fit_readout_stats(
    genome: Sequence[Mapping[str, Any]],
    fit: Mapping[str, np.ndarray],
    *,
    topk_terms: int = 8,
) -> dict[str, Any]:
    w = _as_2d_float(np.asarray(fit["weight"], dtype=float))
    b = np.asarray(fit["bias"], dtype=float).reshape(-1)

    n_terms = min(int(len(genome)), int(w.shape[0]))
    if n_terms <= 0:
        return {
            "n_terms": 0,
            "weight_l2": 0.0,
            "weight_max_abs": 0.0,
            "bias_l2": float(np.linalg.norm(b)),
            "top_terms": [],
        }

    ww = np.asarray(w[:n_terms, :], dtype=float)
    coeff_l2 = np.sqrt(np.sum(ww**2, axis=1))
    coeff_max = np.max(np.abs(ww), axis=1)
    rank = list(np.argsort(-coeff_l2))
    k = max(1, int(topk_terms))
    top_idx = rank[: min(len(rank), k)]

    top_terms: list[dict[str, Any]] = []
    for idx in top_idx:
        term = dict(genome[int(idx)])
        coeff_vec = np.asarray(ww[int(idx), :], dtype=float).reshape(-1)
        top_terms.append(
            {
                "index": int(idx),
                "name": str(term.get("name", f"term_{int(idx)}")),
                "expr": expression_to_string(term.get("expr", {}), precision=10),
                "coeff_l2": float(coeff_l2[int(idx)]),
                "coeff_max_abs": float(coeff_max[int(idx)]),
                "coeff_target": [float(v) for v in coeff_vec],
            }
        )

    return {
        "n_terms": int(n_terms),
        "weight_l2": float(np.linalg.norm(ww)),
        "weight_max_abs": float(np.max(np.abs(ww))),
        "bias_l2": float(np.linalg.norm(b)),
        "top_terms": top_terms,
    }


def _predict_with_fit(
    genome: Sequence[Mapping[str, Any]],
    *,
    X_eval: np.ndarray,
    fit: Mapping[str, np.ndarray],
    graph_cache: ExpressionGraphCache | None = None,
    batch_key: str | None = None,
) -> np.ndarray:
    phi = _design_matrix_from_genome(
        genome,
        X_eval,
        graph_cache=graph_cache,
        batch_key=batch_key,
    )
    w = _as_2d_float(np.asarray(fit["weight"], dtype=float))
    b = np.asarray(fit["bias"], dtype=float).reshape(1, -1)
    return np.asarray(phi @ w + b, dtype=float)


def _split_fit_val(
    X: np.ndarray,
    y: np.ndarray,
    *,
    val_ratio: float,
    min_val_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    x = _as_2d_float(X)
    t = _as_2d_float(y)
    if x.shape[0] != t.shape[0]:
        raise ValueError("X and y row mismatch for fit/val split")

    n = int(x.shape[0])
    if n <= 16:
        return {
            "active": False,
            "reason": "too_few_samples",
            "X_fit": x,
            "y_fit": t,
            "X_val": None,
            "y_val": None,
            "n_fit": int(n),
            "n_val": 0,
        }

    raw_val = int(round(float(np.clip(val_ratio, 0.0, 0.9)) * float(n)))
    want_val = max(int(min_val_samples), int(raw_val))
    max_val = max(0, n - 8)
    n_val = int(min(max(want_val, 0), max_val))

    if n_val <= 0 or (n - n_val) < 8:
        return {
            "active": False,
            "reason": "split_infeasible",
            "X_fit": x,
            "y_fit": t,
            "X_val": None,
            "y_val": None,
            "n_fit": int(n),
            "n_val": 0,
        }

    rng = np.random.default_rng(int(random_seed))
    perm = np.asarray(rng.permutation(n), dtype=int)
    idx_val = perm[:n_val]
    idx_fit = perm[n_val:]
    return {
        "active": True,
        "reason": "ok",
        "X_fit": np.asarray(x[idx_fit], dtype=float),
        "y_fit": np.asarray(t[idx_fit], dtype=float),
        "X_val": np.asarray(x[idx_val], dtype=float),
        "y_val": np.asarray(t[idx_val], dtype=float),
        "n_fit": int(idx_fit.size),
        "n_val": int(idx_val.size),
    }


def _prune_terms_once(
    genome: list[Dict[str, Any]],
    *,
    X: np.ndarray,
    y: np.ndarray,
    cfg: StructureSearchConfig,
    graph_cache: ExpressionGraphCache | None = None,
    batch_key: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, float]]:
    removed: list[dict[str, Any]] = []
    current_fit, current_metrics = _fit_with_metrics(
        genome,
        X=X,
        y=y,
        l2=float(cfg.ridge_l2),
        graph_cache=graph_cache,
        batch_key=batch_key,
    )

    if not bool(cfg.enable_prune):
        return removed, current_fit, current_metrics

    max_drop = int(max(0, cfg.prune_max_removed_per_iter))
    tol = float(max(0.0, cfg.prune_rmse_tolerance))
    if max_drop <= 0:
        return removed, current_fit, current_metrics

    for _ in range(max_drop):
        if len(genome) <= 1:
            break

        base_rmse = float(current_metrics["rmse"])
        best_drop: dict[str, Any] | None = None

        for idx in range(len(genome)):
            trial = genome[:idx] + genome[idx + 1 :]
            trial_fit, trial_metrics = _fit_with_metrics(
                trial,
                X=X,
                y=y,
                l2=float(cfg.ridge_l2),
                graph_cache=graph_cache,
                batch_key=batch_key,
            )
            trial_rmse = float(trial_metrics["rmse"])
            delta_rmse = float(base_rmse - trial_rmse)

            if trial_rmse <= base_rmse + tol:
                if best_drop is None or delta_rmse > float(best_drop["delta_rmse"]):
                    best_drop = {
                        "idx": int(idx),
                        "term": dict(genome[idx]),
                        "fit": trial_fit,
                        "metrics": trial_metrics,
                        "delta_rmse": float(delta_rmse),
                    }

        if best_drop is None:
            break

        idx = int(best_drop["idx"])
        removed_term = genome.pop(idx)
        removed.append(
            {
                "name": str(removed_term.get("name", "")),
                "expr": expression_to_string(removed_term["expr"], precision=10),
                "expr_key": str(_expr_key(removed_term["expr"])),
                "delta_rmse": float(best_drop["delta_rmse"]),
            }
        )
        current_fit = dict(best_drop["fit"])
        current_metrics = dict(best_drop["metrics"])

    return removed, current_fit, current_metrics


def residual_guided_structure_search(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    seed_genome: Sequence[Mapping[str, Any]] | None = None,
    config: StructureSearchConfig | None = None,
    inner_runtime_dispatcher: InnerRuntimeDispatcher | None = None,
    inner_runtime_context: Mapping[str, Any] | None = None,
) -> StructureSearchResult:
    x = _as_2d_float(X)
    target = _as_2d_float(y)
    if x.shape[0] != target.shape[0]:
        raise ValueError("X and y row mismatch")

    cfg = config or StructureSearchConfig()
    d = int(x.shape[1])
    names = tuple(feature_names or tuple(f"x{i}" for i in range(d)))
    if len(names) != d:
        names = tuple(f"x{i}" for i in range(d))
    dispatcher = inner_runtime_dispatcher
    runtime_context = {} if inner_runtime_context is None else dict(inner_runtime_context)
    run_id = str(runtime_context.get("run_id", runtime_context.get("task_id", "symbolic_structure_search")))
    runtime_key = str(runtime_context.get("runtime_key", "symbolic_structure_search"))
    trainer_name = str(runtime_context.get("trainer_name", "symbolic_structure_search"))
    runtime_metadata = {
        "search_driver": runtime_context.get("search_driver"),
        "training_mode": runtime_context.get("training_mode"),
        "max_added_terms": int(cfg.max_added_terms),
        "candidate_keep_top": int(cfg.candidate_keep_top),
    }

    if seed_genome is None:
        genome: list[Dict[str, Any]] = list(_default_seed_genome(d))
    else:
        raw_seed = list(seed_genome)
        if raw_seed:
            genome = [dict(t) for t in normalize_genome(raw_seed, input_dim=d)]
        else:
            genome = []

    existing = {_expr_key(term["expr"]) for term in genome}

    graph_cache: ExpressionGraphCache | None = None
    graph_cache_status: dict[str, Any] = {
        "enabled": bool(cfg.graph_cache_enabled),
        "active": False,
        "max_value_entries": int(max(1, cfg.graph_cache_max_value_entries)),
        "max_derivative_entries": int(max(1, cfg.graph_cache_max_derivative_entries)),
        "backend": str(cfg.graph_cache_backend),
        "db_path": str(cfg.graph_cache_db_path),
        "namespace": str(cfg.graph_cache_namespace),
        "persist_values": bool(cfg.graph_cache_persist_values),
    }
    if bool(cfg.graph_cache_enabled):
        try:
            graph_cache = ExpressionGraphCache(
                enabled=True,
                max_value_entries=int(max(1, cfg.graph_cache_max_value_entries)),
                max_derivative_entries=int(max(1, cfg.graph_cache_max_derivative_entries)),
                backend=str(cfg.graph_cache_backend),
                db_path=str(cfg.graph_cache_db_path),
                namespace=str(cfg.graph_cache_namespace),
                persist_values=bool(cfg.graph_cache_persist_values),
            )
            graph_cache_status["active"] = True
        except Exception as exc:
            graph_cache = None
            graph_cache_status["error"] = f"{type(exc).__name__}: {exc}"

    # Base metrics are reported on the full data for stable external comparisons.
    base_phi = _design_matrix_from_genome(
        genome,
        x,
        graph_cache=graph_cache,
        batch_key="full",
    )
    base_fit = _fit_ridge_readout(base_phi, target, l2=float(cfg.ridge_l2))
    base_metrics = _regression_metrics(target, base_fit["pred"])

    split = _split_fit_val(
        x,
        target,
        val_ratio=float(cfg.overfit_guard_val_ratio),
        min_val_samples=int(max(1, cfg.overfit_guard_min_val_samples)),
        random_seed=int(cfg.overfit_guard_random_seed),
    )
    overfit_guard_active = bool(cfg.overfit_guard_enabled) and bool(split["active"])
    if overfit_guard_active:
        x_fit = np.asarray(split["X_fit"], dtype=float)
        y_fit = np.asarray(split["y_fit"], dtype=float)
        x_val = np.asarray(split["X_val"], dtype=float) if split["X_val"] is not None else None
        y_val = np.asarray(split["y_val"], dtype=float) if split["y_val"] is not None else None
    else:
        x_fit = np.asarray(x, dtype=float)
        y_fit = np.asarray(target, dtype=float)
        x_val = None
        y_val = None

    history: list[dict[str, Any]] = []
    score_trace: list[float] = []
    tabu_until_step: dict[str, int] = {}
    best_snapshot_genome: list[Dict[str, Any]] = [dict(t) for t in genome]
    best_snapshot_val_rmse = float("inf")
    no_improve_rounds = 0
    last_round_index: int | None = None

    def _emit_round(history_entry: Mapping[str, Any], *, round_index: int) -> None:
        nonlocal last_round_index
        last_round_index = int(round_index)
        if dispatcher is None or not dispatcher.enabled:
            return
        dispatcher.emit_round_end(
            InnerRuntimeRoundPayload(
                run_id=run_id,
                runtime_key=runtime_key,
                trainer_name=trainer_name,
                round_index=int(round_index),
                total_rounds=int(cfg.max_added_terms),
                genome_size=int(len(genome)),
                score_trace=tuple(float(v) for v in score_trace),
                history_entry=dict(history_entry),
                context=runtime_context,
                metadata=runtime_metadata,
            )
        )

    path_memory: SymbolicPathMemory | None = None
    path_memory_status: dict[str, Any] = {
        "enabled": bool(cfg.path_memory_enabled),
        "active": False,
        "namespace": str(cfg.path_memory_namespace),
    }
    if bool(cfg.path_memory_enabled):
        try:
            db_path = str(cfg.path_memory_db_path).strip()
            path_memory = SymbolicPathMemory(
                db_path=(db_path if db_path else None),
                namespace=str(cfg.path_memory_namespace),
            )
            path_memory_status["active"] = True
            path_memory_status["db_path"] = str(path_memory.path)
        except Exception as exc:
            path_memory_status["error"] = f"{type(exc).__name__}: {exc}"
            path_memory = None

    structure_optimizer = StructureOptimizer(
        StructureScoreConfig(
            score_corr_bonus=float(cfg.score_corr_bonus),
            score_complexity_penalty=float(cfg.score_complexity_penalty),
            score_grad_guidance_bonus=float(cfg.score_grad_guidance_bonus),
        )
    )

    x_scale = np.std(x_fit, axis=0) + 1e-8
    grad_adv_cfg = {
        "enabled": bool(cfg.grad_adv_check),
        "trials": int(max(0, cfg.grad_adv_trials)),
        "noise_std": float(max(0.0, cfg.grad_adv_noise_std)),
        "min_stability": float(max(0.0, cfg.grad_adv_min_stability)),
        "seed": int(cfg.grad_adv_random_seed),
    }
    overfit_guard_status: dict[str, Any] = {
        "enabled": bool(cfg.overfit_guard_enabled),
        "active": bool(overfit_guard_active),
        "reason": (str(split.get("reason", "disabled")) if overfit_guard_active else "disabled"),
        "n_fit": int(split.get("n_fit", int(x.shape[0]))) if overfit_guard_active else int(x.shape[0]),
        "n_val": int(split.get("n_val", 0)) if overfit_guard_active else 0,
        "val_ratio": float(cfg.overfit_guard_val_ratio),
        "min_val_samples": int(max(1, cfg.overfit_guard_min_val_samples)),
        "min_val_rmse_gain": float(cfg.overfit_guard_min_val_rmse_gain),
        "max_gap_increase": float(cfg.overfit_guard_max_gap_increase),
        "patience": int(max(0, cfg.overfit_guard_patience)),
        "snapshot_min_improve": float(max(0.0, cfg.overfit_guard_snapshot_min_improve)),
        "tabu_rounds": int(max(0, cfg.overfit_guard_tabu_rounds)),
        "replace_topk": int(max(0, cfg.overfit_guard_replace_topk)),
        "replace_drop_topk": int(max(0, cfg.overfit_guard_replace_drop_topk)),
    }

    try:
        if dispatcher is not None and dispatcher.enabled:
            dispatcher.emit_start(
                InnerRuntimeStartPayload(
                    run_id=run_id,
                    runtime_key=runtime_key,
                    trainer_name=trainer_name,
                    total_rounds=int(cfg.max_added_terms),
                    input_shape=(int(x.shape[0]), int(x.shape[1])),
                    feature_names=tuple(str(v) for v in names),
                    seed_terms=int(len(genome)),
                    context=runtime_context,
                    metadata=runtime_metadata,
                )
            )
        for step in range(int(cfg.max_added_terms)):
            fit_now, metrics_before = _fit_with_metrics(
                genome,
                X=x_fit,
                y=y_fit,
                l2=float(cfg.ridge_l2),
                graph_cache=graph_cache,
                batch_key="fit",
            )
            readout_before = _fit_readout_stats(genome, fit_now, topk_terms=8)
            metrics_before_val: dict[str, float] | None = None
            if overfit_guard_active and x_val is not None and y_val is not None:
                pred_before_val = _predict_with_fit(
                    genome,
                    X_eval=x_val,
                    fit=fit_now,
                    graph_cache=graph_cache,
                    batch_key="val",
                )
                metrics_before_val = _regression_metrics(y_val, pred_before_val)
                snapshot_tol = float(max(0.0, cfg.overfit_guard_snapshot_min_improve))
                if float(metrics_before_val["rmse"]) + snapshot_tol < float(best_snapshot_val_rmse):
                    best_snapshot_val_rmse = float(metrics_before_val["rmse"])
                    best_snapshot_genome = [dict(t) for t in genome]
                    no_improve_rounds = 0

            pred_now = np.asarray(fit_now["pred"], dtype=float)
            residual = y_fit - pred_now

            gradient_signal = GradientParser.build_signal(
                genome=genome,
                weight=np.asarray(fit_now["weight"], dtype=float),
                X=x_fit,
                y=y_fit,
                slope_mode=str(cfg.grad_slope_mode),
                slope_bins=int(cfg.grad_slope_bins),
                slope_min_bin_samples=int(cfg.grad_slope_min_bin_samples),
                graph_cache=graph_cache,
                batch_key="fit",
            )
            gradient_correction: GradientCorrection | None = None
            if float(cfg.score_grad_guidance_bonus) > 0.0:
                gradient_correction = GradientCorrection(
                    gradient_signal,
                    config=GradientCorrectionConfig(
                        focus_topk_features=int(max(1, cfg.grad_focus_topk)),
                        min_priority=float(max(0.0, cfg.grad_min_priority)),
                    ),
                )

            top_grad_idx = np.argsort(-gradient_signal.feature_priority)[: min(5, int(d))]
            grad_summary = {
                "overall_mismatch": float(gradient_signal.overall_mismatch),
                "signal_signature": str(getattr(gradient_signal, "signal_signature", "")),
                "slope_mode": str(cfg.grad_slope_mode),
                "slope_bins": int(cfg.grad_slope_bins),
                "slope_min_bin_samples": int(cfg.grad_slope_min_bin_samples),
                "adversarial_check": bool(grad_adv_cfg["enabled"]),
                "active_features": (
                    [int(i) for i in gradient_correction.active_features]
                    if gradient_correction is not None
                    else []
                ),
                "top_feature_priority": [
                    {
                        "feature": names[int(i)],
                        "priority": float(gradient_signal.feature_priority[int(i)]),
                        "priority_multiscale": float(
                            np.asarray(
                                getattr(gradient_signal, "feature_priority_multiscale", gradient_signal.feature_priority),
                                dtype=float,
                            ).reshape(-1)[int(i)]
                        ),
                        "stability": float(
                            np.asarray(
                                getattr(gradient_signal, "feature_stability", np.ones_like(gradient_signal.feature_priority)),
                                dtype=float,
                            ).reshape(-1)[int(i)]
                        ),
                        "mismatch": float(gradient_signal.feature_mismatch[int(i)]),
                        "abs_gap_mean": float(gradient_signal.feature_gap_abs_mean[int(i)]),
                    }
                    for i in top_grad_idx
                ],
            }

            raw_candidates = _build_candidates(
                x_fit,
                residual,
                cfg=cfg,
                feature_names=names,
                gradient_signal=gradient_signal,
                graph_cache=graph_cache,
                batch_key="fit",
            )
            scored: list[dict[str, Any]] = []
            step_rng = np.random.default_rng(int(grad_adv_cfg["seed"]) + 10007 * int(step + 1))

            for cand in raw_candidates:
                key = _expr_key(cand["expr"])
                if key in existing:
                    continue
                until = int(tabu_until_step.get(str(key), -1))
                if until > int(step):
                    continue
                rec = _score_candidate(
                    cand,
                    X=x_fit,
                    residual=residual,
                    cfg=cfg,
                    gradient_correction=gradient_correction,
                    structure_optimizer=structure_optimizer,
                    x_scale=x_scale,
                    grad_adv_config=grad_adv_cfg,
                    rng=step_rng,
                    path_memory=path_memory,
                    graph_cache=graph_cache,
                    batch_key="fit",
                )
                if rec is not None:
                    scored.append(rec)

            scored.sort(key=lambda r: float(r["score"]), reverse=True)
            if not scored:
                history_row = {
                    "iteration": int(step + 1),
                    "stop_reason": "no_valid_candidate",
                    "metrics_before": dict(metrics_before),
                    "metrics_before_val": dict(metrics_before_val) if metrics_before_val is not None else None,
                    "n_terms": int(len(genome)),
                    "gradient_summary": dict(grad_summary),
                    "path_memory": dict(path_memory_status),
                    "overfit_guard": {**dict(overfit_guard_status), "tabu_size": int(len(tabu_until_step))},
                }
                history.append(history_row)
                _emit_round(history_row, round_index=int(step + 1))
                break

            top = scored[: max(1, int(cfg.candidate_keep_top))]
            joint_bundles = _build_joint_bundle_candidates(
                top,
                X=x_fit,
                residual=residual,
                cfg=cfg,
                graph_cache=graph_cache,
                batch_key="fit",
            )
            top_pool = list(top) + list(joint_bundles)
            top_pool.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
            ranked_best = top_pool[0]
            if float(ranked_best["score"]) < float(cfg.min_score):
                if path_memory is not None:
                    path_memory.record_expr_outcome(
                        str(ranked_best["expr_key"]),
                        selected_score=float(ranked_best["score"]),
                        delta_rmse=0.0,
                        success=False,
                    )

                history.append(
                    {
                        "iteration": int(step + 1),
                        "stop_reason": "score_below_threshold",
                        "best_score": float(ranked_best["score"]),
                        "metrics_before": dict(metrics_before),
                        "metrics_before_val": dict(metrics_before_val) if metrics_before_val is not None else None,
                        "n_terms": int(len(genome)),
                        "top_candidates": [
                            _candidate_log_row(item) for item in top_pool[: max(1, int(cfg.candidate_keep_top))]
                        ],
                        "gradient_summary": dict(grad_summary),
                        "path_memory": dict(path_memory_status),
                        "overfit_guard": {**dict(overfit_guard_status), "tabu_size": int(len(tabu_until_step))},
                    }
                )
                break

            # verify top-k with real rmse gain before final add
            base_rmse = float(metrics_before["rmse"])
            base_val_rmse = float(metrics_before_val["rmse"]) if metrics_before_val is not None else float(base_rmse)
            base_gap = float(base_val_rmse - base_rmse) if overfit_guard_active else 0.0
            selected_bundle: dict[str, Any] | None = None
            selection_trials: list[dict[str, Any]] = []
            best_actual_utility = -float("inf")
            tabu_rounds = int(max(0, cfg.overfit_guard_tabu_rounds))

            for item in top_pool:
                bundle_items = [dict(v) for v in item.get("bundle_items", [])]
                is_bundle = len(bundle_items) > 0
                trial_terms = (
                    [{"name": str(v["name"]), "expr": dict(v["expr"])} for v in bundle_items]
                    if is_bundle
                    else [{"name": str(item["name"]), "expr": dict(item["expr"])}]
                )
                trial_genome = [dict(t) for t in genome] + [dict(t) for t in trial_terms]
                trial_fit, trial_metrics = _fit_with_metrics(
                    trial_genome,
                    X=x_fit,
                    y=y_fit,
                    l2=float(cfg.ridge_l2),
                    graph_cache=graph_cache,
                    batch_key="fit",
                )
                delta_rmse = float(base_rmse - float(trial_metrics["rmse"]))
                trial_metrics_val: dict[str, float] | None = None
                delta_val_rmse = float(delta_rmse)
                gap_increase = 0.0
                guard_ok = True
                guard_reason = "ok"
                if overfit_guard_active and x_val is not None and y_val is not None:
                    pred_val = _predict_with_fit(
                        trial_genome,
                        X_eval=x_val,
                        fit=trial_fit,
                        graph_cache=graph_cache,
                        batch_key="val",
                    )
                    trial_metrics_val = _regression_metrics(y_val, pred_val)
                    delta_val_rmse = float(base_val_rmse - float(trial_metrics_val["rmse"]))
                    gap_after = float(trial_metrics_val["rmse"] - float(trial_metrics["rmse"]))
                    gap_increase = float(gap_after - base_gap)
                    if delta_val_rmse < float(cfg.overfit_guard_min_val_rmse_gain):
                        guard_ok = False
                        guard_reason = "val_gain_too_small"
                    elif gap_increase > float(cfg.overfit_guard_max_gap_increase):
                        guard_ok = False
                        guard_reason = "gap_increase_too_large"

                utility_basis = float(delta_val_rmse) if overfit_guard_active else float(delta_rmse)
                actual_utility = float(utility_basis - float(cfg.score_complexity_penalty) * float(item["complexity"]))

                selection_trials.append(
                    {
                        "operation": "add_bundle" if is_bundle else "add",
                        "name": str(item["name"]),
                        "family": str(item["family"]),
                        "bundle_size": int(len(trial_terms)),
                        "bundle_members": [str(v["name"]) for v in trial_terms],
                        "rank_score": float(item["score"]),
                        "delta_rmse": float(delta_rmse),
                        "delta_val_rmse": float(delta_val_rmse),
                        "gap_increase": float(gap_increase),
                        "actual_utility": float(actual_utility),
                        "guard_ok": bool(guard_ok),
                        "guard_reason": str(guard_reason),
                        "path_prior": dict(item.get("path_prior", {})),
                    }
                )

                if not bool(guard_ok):
                    if tabu_rounds > 0:
                        tabu_until_step[str(item["expr_key"])] = int(step + tabu_rounds)
                    continue

                if actual_utility > best_actual_utility:
                    best_actual_utility = float(actual_utility)
                    selected_bundle = {
                        "operation": "add_bundle" if is_bundle else "add",
                        "candidate": dict(item),
                        "terms": [dict(v) for v in trial_terms],
                        "fit": trial_fit,
                        "metrics": trial_metrics,
                        "metrics_val": dict(trial_metrics_val) if trial_metrics_val is not None else None,
                        "delta_rmse": float(delta_rmse),
                        "delta_val_rmse": float(delta_val_rmse),
                        "gap_increase": float(gap_increase),
                        "actual_utility": float(actual_utility),
                    }

            replace_topk = int(max(0, cfg.overfit_guard_replace_topk))
            if (
                selected_bundle is None
                and overfit_guard_active
                and replace_topk > 0
                and len(genome) > 0
            ):
                w_now = _as_2d_float(np.asarray(fit_now["weight"], dtype=float))
                if int(w_now.shape[0]) == int(len(genome)):
                    coeff_strength = np.mean(np.abs(w_now), axis=1)
                else:
                    coeff_strength = np.ones((len(genome),), dtype=float)
                drop_topk = int(max(1, cfg.overfit_guard_replace_drop_topk))
                drop_order = list(np.argsort(coeff_strength)[: min(len(genome), drop_topk)])
                replace_pool = top[: max(1, replace_topk)]

                for item in replace_pool:
                    for drop_idx in drop_order:
                        replaced_term = dict(genome[int(drop_idx)])
                        replaced_expr_key = str(_expr_key(replaced_term["expr"]))
                        if replaced_expr_key == str(item["expr_key"]):
                            continue

                        term = {
                            "name": str(item["name"]),
                            "expr": dict(item["expr"]),
                        }
                        trial_genome = [dict(t) for t in genome]
                        trial_genome[int(drop_idx)] = dict(term)
                        trial_fit, trial_metrics = _fit_with_metrics(
                            trial_genome,
                            X=x_fit,
                            y=y_fit,
                            l2=float(cfg.ridge_l2),
                            graph_cache=graph_cache,
                            batch_key="fit",
                        )
                        delta_rmse = float(base_rmse - float(trial_metrics["rmse"]))
                        trial_metrics_val: dict[str, float] | None = None
                        delta_val_rmse = float(delta_rmse)
                        gap_increase = 0.0
                        guard_ok = True
                        guard_reason = "ok"
                        if overfit_guard_active and x_val is not None and y_val is not None:
                            pred_val = _predict_with_fit(
                                trial_genome,
                                X_eval=x_val,
                                fit=trial_fit,
                                graph_cache=graph_cache,
                                batch_key="val",
                            )
                            trial_metrics_val = _regression_metrics(y_val, pred_val)
                            delta_val_rmse = float(base_val_rmse - float(trial_metrics_val["rmse"]))
                            gap_after = float(trial_metrics_val["rmse"] - float(trial_metrics["rmse"]))
                            gap_increase = float(gap_after - base_gap)
                            if delta_val_rmse < float(cfg.overfit_guard_min_val_rmse_gain):
                                guard_ok = False
                                guard_reason = "val_gain_too_small"
                            elif gap_increase > float(cfg.overfit_guard_max_gap_increase):
                                guard_ok = False
                                guard_reason = "gap_increase_too_large"

                        utility_basis = float(delta_val_rmse) if overfit_guard_active else float(delta_rmse)
                        actual_utility = float(
                            utility_basis - float(cfg.score_complexity_penalty) * float(item["complexity"])
                        )
                        selection_trials.append(
                            {
                                "operation": "replace",
                                "name": str(item["name"]),
                                "family": str(item["family"]),
                                "rank_score": float(item["score"]),
                                "delta_rmse": float(delta_rmse),
                                "delta_val_rmse": float(delta_val_rmse),
                                "gap_increase": float(gap_increase),
                                "actual_utility": float(actual_utility),
                                "guard_ok": bool(guard_ok),
                                "guard_reason": str(guard_reason),
                                "replace_drop_index": int(drop_idx),
                                "replace_drop_name": str(replaced_term.get("name", "")),
                                "path_prior": dict(item.get("path_prior", {})),
                            }
                        )

                        if not bool(guard_ok):
                            if tabu_rounds > 0:
                                tabu_until_step[str(item["expr_key"])] = int(step + tabu_rounds)
                            continue

                        if actual_utility > best_actual_utility:
                            best_actual_utility = float(actual_utility)
                            selected_bundle = {
                                "operation": "replace",
                                "replace_drop_index": int(drop_idx),
                                "replace_drop_term": dict(replaced_term),
                                "replace_drop_expr_key": str(replaced_expr_key),
                                "candidate": dict(item),
                                "terms": [dict(term)],
                                "fit": trial_fit,
                                "metrics": trial_metrics,
                                "metrics_val": dict(trial_metrics_val) if trial_metrics_val is not None else None,
                                "delta_rmse": float(delta_rmse),
                                "delta_val_rmse": float(delta_val_rmse),
                                "gap_increase": float(gap_increase),
                                "actual_utility": float(actual_utility),
                            }

            if selected_bundle is None:
                history.append(
                    {
                        "iteration": int(step + 1),
                        "stop_reason": "selection_failed",
                        "metrics_before": dict(metrics_before),
                        "metrics_before_val": dict(metrics_before_val) if metrics_before_val is not None else None,
                        "n_terms": int(len(genome)),
                        "top_candidates": [
                            _candidate_log_row(item) for item in top_pool[: max(1, int(cfg.candidate_keep_top))]
                        ],
                        "selection_trials": [dict(item) for item in selection_trials],
                        "gradient_summary": dict(grad_summary),
                        "path_memory": dict(path_memory_status),
                        "overfit_guard": {**dict(overfit_guard_status), "tabu_size": int(len(tabu_until_step))},
                    }
                )
                break

            best = dict(selected_bundle["candidate"])
            delta_add = float(selected_bundle["delta_rmse"])
            delta_val_add = float(selected_bundle.get("delta_val_rmse", delta_add))
            min_actual_gain = float(cfg.min_actual_rmse_gain)

            if delta_add < min_actual_gain:
                if tabu_rounds > 0:
                    tabu_until_step[str(best["expr_key"])] = int(step + tabu_rounds)
                if path_memory is not None:
                    path_memory.record_expr_outcome(
                        str(best["expr_key"]),
                        selected_score=float(best["score"]),
                        delta_rmse=float(delta_add),
                        success=False,
                    )

                history.append(
                    {
                        "iteration": int(step + 1),
                        "stop_reason": "no_actual_gain",
                        "actual_delta_rmse": float(delta_add),
                        "actual_delta_val_rmse": float(delta_val_add),
                        "min_actual_rmse_gain": float(min_actual_gain),
                        "metrics_before": dict(metrics_before),
                        "metrics_before_val": dict(metrics_before_val) if metrics_before_val is not None else None,
                        "n_terms": int(len(genome)),
                        "top_candidates": [
                            _candidate_log_row(item) for item in top_pool[: max(1, int(cfg.candidate_keep_top))]
                        ],
                        "selection_trials": [dict(item) for item in selection_trials],
                        "gradient_summary": dict(grad_summary),
                        "path_memory": dict(path_memory_status),
                        "overfit_guard": {**dict(overfit_guard_status), "tabu_size": int(len(tabu_until_step))},
                    }
                )
                break

            n_terms_before_step = int(len(genome))
            selection_operation = str(selected_bundle.get("operation", "add"))
            selected_terms = [dict(v) for v in selected_bundle.get("terms", [])]
            if not selected_terms:
                history.append(
                    {
                        "iteration": int(step + 1),
                        "stop_reason": "empty_selected_terms",
                        "metrics_before": dict(metrics_before),
                        "n_terms": int(len(genome)),
                    }
                )
                break
            src_sig = _genome_signature(genome)
            replaced_term_info: dict[str, Any] | None = None
            if selection_operation == "replace":
                drop_idx = int(selected_bundle.get("replace_drop_index", -1))
                if drop_idx < 0 or drop_idx >= len(genome):
                    history.append(
                        {
                            "iteration": int(step + 1),
                            "stop_reason": "replace_index_invalid",
                            "metrics_before": dict(metrics_before),
                            "metrics_before_val": dict(metrics_before_val) if metrics_before_val is not None else None,
                            "n_terms": int(len(genome)),
                            "top_candidates": [
                                _candidate_log_row(item) for item in top_pool[: max(1, int(cfg.candidate_keep_top))]
                            ],
                            "selection_trials": [dict(item) for item in selection_trials],
                            "gradient_summary": dict(grad_summary),
                            "path_memory": dict(path_memory_status),
                            "overfit_guard": {**dict(overfit_guard_status), "tabu_size": int(len(tabu_until_step))},
                        }
                    )
                    break
                replaced_term = dict(genome[drop_idx])
                replaced_key = str(_expr_key(replaced_term["expr"]))
                genome[drop_idx] = dict(selected_terms[0])
                existing.discard(replaced_key)
                replaced_term_info = {
                    "index": int(drop_idx),
                    "name": str(replaced_term.get("name", "")),
                    "expr": expression_to_string(replaced_term["expr"], precision=10),
                    "expr_key": str(replaced_key),
                }
            else:
                for t_add in selected_terms:
                    k_add = str(_expr_key(t_add["expr"]))
                    if k_add in existing:
                        continue
                    genome.append(dict(t_add))
                    existing.add(k_add)
            if selection_operation == "replace":
                existing.add(str(best["expr_key"]))
            score_trace.append(float(best["score"]))
            fit_after = dict(selected_bundle["fit"])
            metrics_after = dict(selected_bundle["metrics"])
            metrics_after_val = dict(selected_bundle.get("metrics_val", {})) if selected_bundle.get("metrics_val") else None
            dst_sig = _genome_signature(genome)

            if path_memory is not None:
                bundle_items = [dict(v) for v in best.get("bundle_items", [])]
                if bundle_items:
                    for it in bundle_items:
                        expr_k = str(it.get("expr_key", ""))
                        if not expr_k:
                            continue
                        path_memory.record_expr_outcome(
                            expr_k,
                            selected_score=float(it.get("score", best["score"])),
                            delta_rmse=float(delta_add) / max(1, len(bundle_items)),
                            success=True,
                        )
                else:
                    path_memory.record_expr_outcome(
                        str(best["expr_key"]),
                        selected_score=float(best["score"]),
                        delta_rmse=float(delta_add),
                        success=True,
                    )
                if selection_operation == "replace":
                    old_expr_key = str(selected_bundle.get("replace_drop_expr_key", ""))
                    if old_expr_key:
                        path_memory.record_expr_outcome(
                            old_expr_key,
                            selected_score=0.0,
                            delta_rmse=-abs(float(delta_add)),
                            success=False,
                        )
                        path_memory.record_edge(
                            src_sig=str(src_sig),
                            op="replace_drop",
                            expr_key=old_expr_key,
                            dst_sig=str(src_sig),
                            delta_rmse=0.0,
                            success=True,
                        )
                    path_memory.record_edge(
                        src_sig=str(src_sig),
                        op="replace_add",
                        expr_key=str(best.get("expr_key", "")),
                        dst_sig=str(dst_sig),
                        delta_rmse=float(delta_add),
                        success=True,
                    )
                else:
                    bundle_items = [dict(v) for v in best.get("bundle_items", [])]
                    if bundle_items:
                        for it in bundle_items:
                            expr_k = str(it.get("expr_key", ""))
                            if not expr_k:
                                continue
                            path_memory.record_edge(
                                src_sig=str(src_sig),
                                op="add",
                                expr_key=expr_k,
                                dst_sig=str(dst_sig),
                                delta_rmse=float(delta_add) / max(1, len(bundle_items)),
                                success=True,
                            )
                    else:
                        path_memory.record_edge(
                            src_sig=str(src_sig),
                            op="add",
                            expr_key=str(best["expr_key"]),
                            dst_sig=str(dst_sig),
                            delta_rmse=float(delta_add),
                            success=True,
                        )

            prune_src_sig = _genome_signature(genome)
            removed_terms, fit_after_prune, metrics_after_prune = _prune_terms_once(
                genome,
                X=x_fit,
                y=y_fit,
                cfg=cfg,
                graph_cache=graph_cache,
                batch_key="fit",
            )
            if removed_terms:
                fit_after = dict(fit_after_prune)
                metrics_after = dict(metrics_after_prune)
                existing = set(_genome_expr_keys(genome))
                prune_dst_sig = _genome_signature(genome)
                if path_memory is not None:
                    for removed in removed_terms:
                        expr_key = str(removed.get("expr_key", ""))
                        if not expr_key:
                            continue
                        path_memory.record_expr_outcome(
                            expr_key,
                            selected_score=0.0,
                            delta_rmse=-abs(float(removed.get("delta_rmse", 0.0))),
                            success=False,
                        )
                        path_memory.record_edge(
                            src_sig=str(prune_src_sig),
                            op="drop",
                            expr_key=expr_key,
                            dst_sig=str(prune_dst_sig),
                            delta_rmse=float(removed.get("delta_rmse", 0.0)),
                            success=bool(float(removed.get("delta_rmse", 0.0)) >= 0.0),
                        )

            if overfit_guard_active and x_val is not None and y_val is not None:
                pred_after_val = _predict_with_fit(
                    genome,
                    X_eval=x_val,
                    fit=fit_after,
                    graph_cache=graph_cache,
                    batch_key="val",
                )
                metrics_after_val = _regression_metrics(y_val, pred_after_val)
            readout_after = _fit_readout_stats(genome, fit_after, topk_terms=8)

            rollback_triggered = False
            if overfit_guard_active and metrics_after_val is not None:
                snapshot_tol = float(max(0.0, cfg.overfit_guard_snapshot_min_improve))
                current_val_rmse = float(metrics_after_val["rmse"])
                if current_val_rmse + snapshot_tol < float(best_snapshot_val_rmse):
                    best_snapshot_val_rmse = float(current_val_rmse)
                    best_snapshot_genome = [dict(t) for t in genome]
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1

                patience = int(max(0, cfg.overfit_guard_patience))
                if patience > 0 and no_improve_rounds >= patience:
                    genome = [dict(t) for t in best_snapshot_genome]
                    existing = set(_genome_expr_keys(genome))
                    fit_after, metrics_after = _fit_with_metrics(
                        genome,
                        X=x_fit,
                        y=y_fit,
                        l2=float(cfg.ridge_l2),
                        graph_cache=graph_cache,
                        batch_key="fit",
                    )
                    if x_val is not None and y_val is not None:
                        pred_after_val = _predict_with_fit(
                            genome,
                            X_eval=x_val,
                            fit=fit_after,
                            graph_cache=graph_cache,
                            batch_key="val",
                        )
                        metrics_after_val = _regression_metrics(y_val, pred_after_val)
                    rollback_triggered = True

            if "expr" in best:
                grad_formulas = gradient_formula_strings(
                    best["expr"],
                    feature_indices=tuple(int(i) for i in best.get("features", ())),
                    precision=8,
                )
            else:
                grad_formulas = {}
                for it in best.get("bundle_items", []):
                    if "expr" not in it:
                        continue
                    grad_formulas[str(it.get("name", "term"))] = gradient_formula_strings(
                        it["expr"],
                        feature_indices=tuple(int(i) for i in it.get("features", ())),
                        precision=8,
                    )

            history_row = {
                "iteration": int(step + 1),
                "selected": {
                    "operation": str(selection_operation),
                    "name": str(best["name"]),
                    "family": str(best["family"]),
                    "expr": (
                        expression_to_string(best["expr"], precision=10)
                        if "expr" in best
                        else str(best.get("name", "bundle"))
                    ),
                    "bundle_size": int(len(best.get("bundle_items", []))),
                    "bundle_members": [str(v.get("name", "")) for v in best.get("bundle_items", [])],
                    "score": float(best["score"]),
                    "score_raw": float(best.get("score_raw", best["score"])),
                    "projected_gain": float(best["projected_gain"]),
                    "abs_corr": float(best["abs_corr"]),
                    "signed_corr": float(best["signed_corr"]),
                    "complexity": float(best["complexity"]),
                    "grad_alignment": float(best.get("grad_alignment", 0.0)),
                    "score_parts": dict(best.get("score_parts", {})),
                    "features": [names[int(i)] for i in best.get("features", ())],
                    "gradient_formulas": dict(grad_formulas),
                    "gradient_correction": {
                        "used_features": [int(i) for i in best.get("grad_details", {}).get("used_features", [])],
                        "per_feature": list(best.get("grad_details", {}).get("per_feature", [])),
                        "adversarial": dict(best.get("grad_adversarial", {})),
                    },
                    "path_prior": dict(best.get("path_prior", {})),
                    "actual_delta_rmse": float(delta_add),
                    "actual_delta_val_rmse": float(delta_val_add),
                    "actual_utility": float(selected_bundle["actual_utility"]),
                    "replace_drop": dict(replaced_term_info) if replaced_term_info is not None else None,
                },
                "metrics_before": dict(metrics_before),
                "metrics_after": dict(metrics_after),
                "metrics_before_val": dict(metrics_before_val) if metrics_before_val is not None else None,
                "metrics_after_val": dict(metrics_after_val) if metrics_after_val is not None else None,
                "n_terms_before": int(n_terms_before_step),
                "n_terms_after": int(len(genome)),
                "pruning": {
                    "enabled": bool(cfg.enable_prune),
                    "removed_terms": [dict(item) for item in removed_terms],
                },
                "selection_trials": [dict(item) for item in selection_trials],
                "gradient_summary": dict(grad_summary),
                "readout": {
                    "before": dict(readout_before),
                    "after": dict(readout_after),
                },
                "top_candidates": [
                    _candidate_log_row(item) for item in top_pool[: max(1, int(cfg.candidate_keep_top))]
                ],
                "path_memory": dict(path_memory_status),
                "graph_cache": (graph_cache.snapshot() if graph_cache is not None else dict(graph_cache_status)),
                "overfit_guard": {
                    **dict(overfit_guard_status),
                    "best_snapshot_val_rmse": (
                        float(best_snapshot_val_rmse) if np.isfinite(best_snapshot_val_rmse) else None
                    ),
                    "no_improve_rounds": int(no_improve_rounds),
                    "rollback_triggered": bool(rollback_triggered),
                    "tabu_size": int(len(tabu_until_step)),
                },
            }
            history.append(history_row)
            _emit_round(history_row, round_index=int(step + 1))
            if rollback_triggered:
                rollback_row = {
                    "iteration": int(step + 1),
                    "stop_reason": "rollback_to_best_snapshot",
                    "metrics_after_rollback": dict(metrics_after),
                    "metrics_after_val_rollback": (
                        dict(metrics_after_val) if metrics_after_val is not None else None
                    ),
                    "n_terms_after_rollback": int(len(genome)),
                    "path_memory": dict(path_memory_status),
                    "graph_cache": (graph_cache.snapshot() if graph_cache is not None else dict(graph_cache_status)),
                    "overfit_guard": {
                        **dict(overfit_guard_status),
                        "best_snapshot_val_rmse": (
                            float(best_snapshot_val_rmse) if np.isfinite(best_snapshot_val_rmse) else None
                        ),
                        "no_improve_rounds": int(no_improve_rounds),
                        "rollback_triggered": True,
                        "tabu_size": int(len(tabu_until_step)),
                    },
                }
                history.append(rollback_row)
                _emit_round(rollback_row, round_index=int(step + 1))
                break

        final_fit, final_metrics = _fit_with_metrics(
            genome,
            X=x,
            y=target,
            l2=float(cfg.ridge_l2),
            graph_cache=graph_cache,
            batch_key="full",
        )

        result = StructureSearchResult(
            genome=tuple(genome),
            base_metrics=dict(base_metrics),
            final_metrics=dict(final_metrics),
            iterations=tuple(history),
            weight=np.asarray(final_fit["weight"], dtype=float),
            bias=np.asarray(final_fit["bias"], dtype=float),
            score_trace=tuple(float(v) for v in score_trace),
        )
        if dispatcher is not None and dispatcher.enabled:
            dispatcher.emit_finish(
                InnerRuntimeFinishPayload(
                    run_id=run_id,
                    runtime_key=runtime_key,
                    trainer_name=trainer_name,
                    total_rounds=int(cfg.max_added_terms),
                    completed_rounds=int(last_round_index or len(history)),
                    genome_size=int(len(genome)),
                    final_metrics=dict(final_metrics),
                    context=runtime_context,
                    metadata=runtime_metadata,
                )
            )
        return result
    except Exception as exc:
        if dispatcher is not None and dispatcher.enabled:
            dispatcher.emit_error(
                InnerRuntimeErrorPayload(
                    run_id=run_id,
                    runtime_key=runtime_key,
                    trainer_name=trainer_name,
                    error=f"{type(exc).__name__}: {exc}",
                    round_index=last_round_index,
                    context=runtime_context,
                    metadata=runtime_metadata,
                )
            )
        raise
    finally:
        if graph_cache is not None:
            try:
                graph_cache.close()
            except Exception:
                pass
        if path_memory is not None:
            path_memory.close()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return _regression_metrics(y_true, y_pred)


__all__ = [
    "StructureSearchConfig",
    "StructureSearchResult",
    "residual_guided_structure_search",
    "evaluate_genome_with_ridge",
    "regression_metrics",
]















