from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from conditional.primitives import ConditionalPrimitiveSpec
from core.symbolic.artifact_schema import build_symbolic_structure_surface_payload
from core.symbolic.basis_consensus import annotate_basis_entries
from core.symbolic.expression_graph_cache import ExpressionGraphCache
from core.symbolic.stage_head_protocol import (
    BasisObjectRef,
    ObjectGradientSignal,
    PoolExpansionCandidate,
    SYMBOLIC_BASIS_BINDING_MODES,
    SYMBOLIC_ESCAPE_POLICIES,
    SymbolicBasisContext,
    build_basis_conditioned_expression_stage_spec,
    build_basis_discovery_stage_spec,
)
from core.symbolic.structure_metadata import (
    build_assembler_budget_payload,
    build_basis_overlap_report,
    build_basis_semantics_payload,
    build_basis_term_rows,
    build_residual_complementarity_report,
    build_semantic_dedup_report,
)
from core.symbolic.symbolic_dsl import expression_to_string
from core.symbolic.symbolic_structure_search import (
    StructureSearchConfig,
    StructureSearchResult,
    evaluate_genome_with_ridge,
    residual_guided_structure_search,
)
from core.symbolic.truth_contracts import truth_contract_specs
from pipeline.feature_space import (
    CandidatePoolConfig,
    FeatureBundle,
    batched_ridge_predict,
    build_full_candidate_pool,
    build_interval_subset_report,
    build_rolling_splits,
    build_subset_candidate_metadata,
    build_subset_genome,
    design_matrix_for_genome,
    interval_metrics_batch,
    interval_objective_sort_key,
    symmetric_interval_batch,
)

_OUTER_OBJECTIVE_INNER_FIT_WEIGHT = 1.0
_OUTER_OBJECTIVE_ORTHOGONALITY_WEIGHT = 0.45
_OUTER_OBJECTIVE_RESIDUAL_WEIGHT = 0.85
_OUTER_OBJECTIVE_SEMANTIC_WEIGHT = 0.10
_OUTER_OBJECTIVE_INTERFERENCE_PENALTY_WEIGHT = 0.35
_OUTER_OBJECTIVE_PERIODIC_WEIGHT = 0.30
_OUTER_OBJECTIVE_PERIODIC_PENALTY_WEIGHT = 0.30
_OUTER_OBJECTIVE_REGIONAL_CORRECTION_WEIGHT = 0.20
_OUTER_OBJECTIVE_SAME_SOURCE_REALIZATION_PENALTY_WEIGHT = 0.22
_CROSS_EXPLANATORY_SOURCE_CORR_THRESHOLD = 0.97
_CROSS_EXPLANATORY_EXPLAINABILITY_THRESHOLD = 0.90
_SCREEN_INFORMATION_CLUSTER_CORR_THRESHOLD = 0.985
_PERIODIC_AUDIT_MIN_SAMPLES = 8


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=float).reshape(-1)
    y = np.asarray(right, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("correlation shape mismatch")
    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    denom = float(np.linalg.norm(xc) * np.linalg.norm(yc)) + 1e-12
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(xc, yc) / denom)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def _text(value: Any) -> str:
    return str(value or "").strip()


def _feature_expr(index: int) -> dict[str, Any]:
    return {"type": "feature", "index": int(index)}


def _const_expr(value: float) -> dict[str, Any]:
    return {"type": "const", "value": float(value)}


def _unary_expr(op: str, arg: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "unary", "op": str(op), "arg": dict(arg)}


def _binary_expr(op: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "binary", "op": str(op), "left": dict(left), "right": dict(right)}


def _relu_expr(arg: Mapping[str, Any]) -> dict[str, Any]:
    z = dict(arg)
    return _binary_expr("mul", _const_expr(0.5), _binary_expr("add", z, _unary_expr("abs", z)))


def _soft_step_expr(feature_idx: int, threshold: float, steepness: float) -> dict[str, Any]:
    z = _binary_expr("sub", _feature_expr(feature_idx), _const_expr(float(threshold)))
    kz = _binary_expr("mul", _const_expr(float(steepness)), z)
    t = _unary_expr("tanh", kz)
    return _binary_expr("mul", _const_expr(0.5), _binary_expr("add", _const_expr(1.0), t))


def _apply_piecewise_mode_expr(mode: str, arg: Mapping[str, Any]) -> dict[str, Any]:
    mode_key = str(mode).strip().lower()
    if mode_key in {"identity", "linear"}:
        return dict(arg)
    if mode_key in {"zero", "off", "none"}:
        return _const_expr(0.0)
    if mode_key in {"abs", "absolute"}:
        return _unary_expr("abs", arg)
    if mode_key in {"square", "sq"}:
        return _unary_expr("square", arg)
    if mode_key in {"neg_identity", "negative", "neg"}:
        return _binary_expr("mul", _const_expr(-1.0), arg)
    if mode_key in {"hinge", "positive_hinge", "relu"}:
        return _relu_expr(arg)
    if mode_key in {"negative_hinge", "left_hinge"}:
        return _relu_expr(_binary_expr("mul", _const_expr(-1.0), arg))
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


def _sequence_str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return tuple()
    seen: set[str] = set()
    out: list[str] = []
    for item in tuple(value):
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return tuple(out)


def _metadata_search_hints(data_metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    metadata = dict(data_metadata or {})
    raw = metadata.get("search_hints")
    if isinstance(raw, Mapping):
        return {str(key): value for key, value in raw.items()}
    return {}


def _feature_name_tuple(
    feature_indices: Sequence[int],
    *,
    feature_names: Sequence[str],
) -> tuple[str, ...]:
    resolved = tuple(str(value) for value in tuple(feature_names))
    out: list[str] = []
    seen: set[str] = set()
    for feature_index in tuple(feature_indices):
        idx = int(feature_index)
        if 0 <= idx < len(resolved):
            name = str(resolved[idx])
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return tuple(out)


def _raw_feature_abs_corr(raw_X: np.ndarray) -> np.ndarray:
    x = np.asarray(raw_X, dtype=float)
    if x.ndim != 2:
        raise ValueError("raw feature correlation expects a 2D matrix")
    feature_dim = int(x.shape[1]) if x.ndim == 2 else 0
    if feature_dim <= 0:
        return np.zeros((0, 0), dtype=float)
    if feature_dim == 1:
        return np.ones((1, 1), dtype=float)
    corr = np.asarray(np.corrcoef(x, rowvar=False), dtype=float)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    if corr.shape != (feature_dim, feature_dim):
        corr = np.zeros((feature_dim, feature_dim), dtype=float)
    np.fill_diagonal(corr, 1.0)
    return corr


def _metadata_proxy_groups(
    *,
    data_metadata: Mapping[str, Any] | None,
    feature_names: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    allowed = {str(name) for name in tuple(feature_names)}
    metadata = dict(data_metadata or {})
    raw_groups = metadata.get("redundant_feature_groups")
    if not isinstance(raw_groups, Mapping):
        raw_groups = metadata.get("proxy_feature_groups")
    if not isinstance(raw_groups, Mapping):
        return tuple()
    groups: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for raw in raw_groups.values():
        if not (isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray))):
            continue
        normalized = tuple(
            dict.fromkeys(str(name).strip() for name in tuple(raw) if str(name).strip() in allowed)
        )
        if len(normalized) < 2:
            continue
        key = tuple(sorted(normalized))
        if key in seen:
            continue
        seen.add(key)
        groups.append(normalized)
    return tuple(groups)


def _configured_periodic_feature_names(
    *,
    cfg: "OrthogonalBasisSearchConfig",
    feature_names: Sequence[str],
    data_metadata: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    allowed = {str(name) for name in tuple(feature_names)}
    hints = _metadata_search_hints(data_metadata)
    merged = list(_sequence_str_tuple(cfg.periodic_feature_names))
    merged.extend(_sequence_str_tuple(hints.get("periodic_feature_names")))
    seen: set[str] = set()
    out: list[str] = []
    for name in tuple(merged):
        if name in allowed and name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def _build_periodic_context(
    *,
    raw_X: np.ndarray,
    feature_names: Sequence[str],
    data_metadata: Mapping[str, Any] | None,
    cfg: "OrthogonalBasisSearchConfig",
) -> dict[str, Any]:
    names = tuple(str(value) for value in tuple(feature_names))
    periodic_feature_names = _configured_periodic_feature_names(
        cfg=cfg,
        feature_names=names,
        data_metadata=data_metadata,
    )
    name_to_index = {str(name): int(index) for index, name in enumerate(names)}
    feature_windows: dict[str, dict[str, Any]] = {}
    x = np.asarray(raw_X, dtype=float)
    for feature_name in tuple(periodic_feature_names):
        index = name_to_index.get(str(feature_name))
        if index is None or index >= x.shape[1]:
            continue
        column = np.asarray(x[:, int(index)], dtype=float).reshape(-1)
        finite_mask = np.isfinite(column)
        if int(np.sum(finite_mask)) < 24:
            continue
        finite_column = column[finite_mask]
        q20 = float(np.quantile(finite_column, 0.20))
        q35 = float(np.quantile(finite_column, 0.35))
        q65 = float(np.quantile(finite_column, 0.65))
        q80 = float(np.quantile(finite_column, 0.80))
        center_mask = np.asarray(finite_mask & (column >= q35) & (column <= q65), dtype=bool)
        edge_mask = np.asarray(finite_mask & ((column <= q20) | (column >= q80)), dtype=bool)
        low_mask = np.asarray(finite_mask & (column <= q20), dtype=bool)
        high_mask = np.asarray(finite_mask & (column >= q80), dtype=bool)
        feature_windows[str(feature_name)] = {
            "feature_name": str(feature_name),
            "feature_index": int(index),
            "thresholds": {
                "edge_low_q20": float(q20),
                "center_low_q35": float(q35),
                "center_high_q65": float(q65),
                "edge_high_q80": float(q80),
            },
            "center_mask": center_mask,
            "edge_mask": edge_mask,
            "low_mask": low_mask,
            "high_mask": high_mask,
            "center_count": int(np.sum(center_mask)),
            "edge_count": int(np.sum(edge_mask)),
            "low_count": int(np.sum(low_mask)),
            "high_count": int(np.sum(high_mask)),
        }
    return {
        "feature_names": names,
        "feature_name_to_index": name_to_index,
        "periodic_feature_names": tuple(feature_windows.keys()),
        "periodic_feature_indices": [
            int(dict(feature_windows[name]).get("feature_index", -1))
            for name in tuple(feature_windows.keys())
        ],
        "feature_windows": feature_windows,
    }


def _correlation_proxy_groups(
    *,
    raw_X: np.ndarray,
    feature_names: Sequence[str],
    threshold: float,
) -> tuple[tuple[str, ...], ...]:
    names = tuple(str(value) for value in tuple(feature_names))
    corr = _raw_feature_abs_corr(raw_X)
    if corr.size == 0 or len(names) <= 1:
        return tuple()
    groups: list[tuple[str, ...]] = []
    seen_nodes: set[int] = set()
    for start in range(len(names)):
        if start in seen_nodes:
            continue
        stack = [int(start)]
        component: set[int] = set()
        while stack:
            node = int(stack.pop())
            if node in component:
                continue
            component.add(node)
            for other in range(len(names)):
                if other == node:
                    continue
                if float(corr[node, other]) >= float(threshold):
                    stack.append(int(other))
        seen_nodes.update(component)
        if len(component) < 2:
            continue
        ordered = tuple(names[index] for index in sorted(component))
        groups.append(ordered)
    return tuple(groups)


def _proxy_policy_uses_correlation_groups(proxy_group_policy: str | None) -> bool:
    mode = str(proxy_group_policy or "").strip().lower()
    return mode in {
        "metadata_or_correlation_cluster",
        "correlation_cluster",
        "correlation_only",
        "infer_from_raw_corr",
        "raw_corr_cluster",
    }


def _build_interference_context(
    *,
    raw_X: np.ndarray,
    feature_names: Sequence[str],
    data_metadata: Mapping[str, Any] | None,
    proxy_group_policy: str | None = None,
) -> dict[str, Any]:
    names = tuple(str(value) for value in tuple(feature_names))
    abs_corr = _raw_feature_abs_corr(raw_X)
    groups: list[tuple[str, ...]] = []
    seen_groups: set[tuple[str, ...]] = set()
    configured_policy = str(
        proxy_group_policy
        or dict(data_metadata or {}).get("proxy_group_policy")
        or "hint_if_available"
    ).strip()
    sources: list[tuple[tuple[str, ...], ...]] = [
        _metadata_proxy_groups(data_metadata=data_metadata, feature_names=names),
    ]
    if _proxy_policy_uses_correlation_groups(configured_policy):
        sources.append(
            _correlation_proxy_groups(
                raw_X=raw_X,
                feature_names=names,
                threshold=float(_CROSS_EXPLANATORY_SOURCE_CORR_THRESHOLD),
            )
        )
    for source in tuple(sources):
        for raw_group in tuple(source):
            key = tuple(sorted(str(name) for name in tuple(raw_group) if str(name).strip()))
            if len(key) < 2 or key in seen_groups:
                continue
            seen_groups.add(key)
            groups.append(tuple(raw_group))
    proxy_group_lookup: dict[str, tuple[str, ...]] = {}
    for group_index, group in enumerate(groups):
        group_id = f"proxy_group_{group_index:02d}"
        for feature_name in tuple(group):
            existing = list(proxy_group_lookup.get(str(feature_name), ()))
            if group_id not in existing:
                existing.append(group_id)
            proxy_group_lookup[str(feature_name)] = tuple(existing)
    return {
        "feature_names": names,
        "feature_name_to_index": {str(name): int(index) for index, name in enumerate(names)},
        "raw_feature_abs_corr": abs_corr,
        "proxy_groups": groups,
        "proxy_group_lookup": proxy_group_lookup,
        "source_corr_threshold": float(_CROSS_EXPLANATORY_SOURCE_CORR_THRESHOLD),
        "explainability_threshold": float(_CROSS_EXPLANATORY_EXPLAINABILITY_THRESHOLD),
        "proxy_group_policy": str(configured_policy or "hint_if_available"),
    }


def _proxy_group_ids_for_feature_names(
    *,
    candidate_feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
) -> tuple[str, ...]:
    proxy_lookup = {
        str(key): tuple(str(item) for item in tuple(value))
        for key, value in dict(interference_context.get("proxy_group_lookup", {}) or {}).items()
    }
    group_ids: list[str] = []
    for feature_name in tuple(candidate_feature_names):
        for group_id in tuple(proxy_lookup.get(str(feature_name), ())):
            if str(group_id) and str(group_id) not in group_ids:
                group_ids.append(str(group_id))
    return tuple(group_ids)


def _candidate_feature_names_for_row(
    *,
    candidate: ScreenedCandidate,
    feature_names: Sequence[str],
) -> tuple[str, ...]:
    return _feature_name_tuple(candidate.features, feature_names=feature_names)


def _candidate_periodic_feature_names(
    *,
    candidate: ScreenedCandidate,
    feature_names: Sequence[str],
    periodic_context: Mapping[str, Any],
) -> tuple[str, ...]:
    periodic_names = {
        str(value) for value in tuple(periodic_context.get("periodic_feature_names", ())) if str(value).strip()
    }
    return tuple(
        name
        for name in _candidate_feature_names_for_row(candidate=candidate, feature_names=feature_names)
        if name in periodic_names
    )


def _normalized_expr_tree(expr: Mapping[str, Any]) -> dict[str, Any]:
    node = dict(expr)
    kind = str(node.get("type", "")).strip().lower()
    if kind == "feature":
        return {"type": "feature", "index": int(node.get("index", -1))}
    if kind == "const":
        return {"type": "const", "value": float(node.get("value", 0.0))}
    if kind == "unary":
        return {
            "type": "unary",
            "op": str(node.get("op", "")).strip().lower(),
            "arg": _normalized_expr_tree(dict(node.get("arg", {}))),
        }
    if kind == "binary":
        left = _normalized_expr_tree(dict(node.get("left", {})))
        right = _normalized_expr_tree(dict(node.get("right", {})))
        op = str(node.get("op", "")).strip().lower()
        if op in {"add", "mul"}:
            left_key = _candidate_expr_key(left)
            right_key = _candidate_expr_key(right)
            if right_key < left_key:
                left, right = right, left
        return {"type": "binary", "op": op, "left": left, "right": right}
    if kind == "piecewise":
        normalized = {"type": "piecewise"}
        for key in sorted(node.keys()):
            if str(key) == "type":
                continue
            value = node[key]
            if isinstance(value, Mapping):
                normalized[str(key)] = _normalized_expr_tree(dict(value))
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                normalized[str(key)] = [
                    _normalized_expr_tree(dict(item)) if isinstance(item, Mapping) else _jsonable(item)
                    for item in tuple(value)
                ]
            else:
                normalized[str(key)] = _jsonable(value)
        return normalized
    return _jsonable(node)


def _expr_is_const(expr: Mapping[str, Any]) -> bool:
    return str(dict(expr).get("type", "")).strip().lower() == "const"


def _expr_const_value(expr: Mapping[str, Any]) -> float:
    return float(dict(expr).get("value", 0.0))


def _expr_contains_nontrivial_unary(expr: Mapping[str, Any]) -> bool:
    node = _normalized_expr_tree(expr)
    kind = str(node.get("type", "")).strip().lower()
    if kind == "unary":
        op = str(node.get("op", "")).strip().lower()
        return op not in {"abs"}
    if kind == "binary":
        return _expr_contains_nontrivial_unary(dict(node.get("left", {}))) or _expr_contains_nontrivial_unary(
            dict(node.get("right", {}))
        )
    if kind == "piecewise":
        return True
    return False


def _expr_is_plain_feature_product(expr: Mapping[str, Any]) -> bool:
    node = _normalized_expr_tree(expr)
    kind = str(node.get("type", "")).strip().lower()
    if kind == "feature":
        return True
    if kind != "binary":
        return False
    if str(node.get("op", "")).strip().lower() != "mul":
        return False
    return _expr_is_plain_feature_product(dict(node.get("left", {}))) and _expr_is_plain_feature_product(
        dict(node.get("right", {}))
    )


def _expr_is_native_trunk_root(expr: Mapping[str, Any]) -> bool:
    node = _normalized_expr_tree(expr)
    kind = str(node.get("type", "")).strip().lower()
    if kind == "feature":
        return True
    if kind != "binary":
        return False
    op = str(node.get("op", "")).strip().lower()
    if op not in {"mul", "div"}:
        return False
    left = _normalized_expr_tree(dict(node.get("left", {})))
    right = _normalized_expr_tree(dict(node.get("right", {})))
    if _expr_is_const(left) or _expr_is_const(right):
        return False
    return _expr_is_native_trunk_root(left) and _expr_is_native_trunk_root(right)


def _native_trunk_interval_gain_summary(
    *,
    candidate_values: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(candidate_values, dtype=float).reshape(-1)
    yy = np.asarray(target, dtype=float).reshape(-1)
    if values.shape[0] != yy.shape[0] or values.shape[0] < 8:
        return {"min_gain": 0.0, "mean_gain": 0.0}
    midpoint = int(values.shape[0] // 2)
    if midpoint < 4 or (values.shape[0] - midpoint) < 4:
        return {"min_gain": 0.0, "mean_gain": 0.0}
    gains: list[float] = []
    for start, stop in ((0, midpoint), (midpoint, values.shape[0])):
        split_values = np.asarray(values[start:stop], dtype=float).reshape(-1)
        split_target = np.asarray(yy[start:stop], dtype=float).reshape(-1)
        if split_values.shape[0] < 4:
            gains.append(0.0)
            continue
        baseline_fit = _ridge_projection(
            np.zeros((int(split_values.shape[0]), 0), dtype=float),
            split_target,
            l2_value=1e-6,
        )
        split_fit = _ridge_projection(split_values, split_target, l2_value=1e-6)
        gains.append(max(0.0, float(split_fit["r2"]) - float(baseline_fit["r2"])))
    if not gains:
        return {"min_gain": 0.0, "mean_gain": 0.0}
    return {
        "min_gain": float(min(gains)),
        "mean_gain": float(np.mean(gains)),
    }


def _regime_penetration_summary(
    *,
    candidate_values: np.ndarray,
    target: np.ndarray,
    raw_X: np.ndarray,
    feature_indices: Sequence[int],
    feature_names: Sequence[str],
    gate_feature_names: Sequence[str],
    cfg: OrthogonalBasisSearchConfig,
) -> dict[str, Any]:
    if not _regime_penetration_enabled(cfg):
        return {
            "enabled": False,
            "score": 0.0,
            "min_gain": 0.0,
            "mean_gain": 0.0,
            "sign_consistency": 0.0,
            "feature_names": [],
            "splits": [],
        }
    values = np.asarray(candidate_values, dtype=float).reshape(-1)
    yy = np.asarray(target, dtype=float).reshape(-1)
    x = np.asarray(raw_X, dtype=float)
    if values.shape[0] != yy.shape[0] or values.shape[0] != x.shape[0] or values.shape[0] < 12:
        return {
            "enabled": True,
            "score": 0.0,
            "min_gain": 0.0,
            "mean_gain": 0.0,
            "sign_consistency": 0.0,
            "feature_names": [],
            "splits": [],
        }
    feature_lookup = {
        str(name): int(index)
        for index, name in enumerate(tuple(str(value) for value in tuple(feature_names)))
    }
    regime_indices: list[int] = []
    for index in tuple(feature_indices):
        if 0 <= int(index) < x.shape[1]:
            regime_indices.append(int(index))
    for name in tuple(gate_feature_names):
        idx = feature_lookup.get(str(name))
        if idx is not None:
            regime_indices.append(int(idx))
    regime_indices = list(dict.fromkeys(regime_indices))
    if not regime_indices:
        return {
            "enabled": True,
            "score": 0.0,
            "min_gain": 0.0,
            "mean_gain": 0.0,
            "sign_consistency": 0.0,
            "feature_names": [],
            "splits": [],
        }
    split_rows: list[dict[str, Any]] = []
    gains: list[float] = []
    signs: list[float] = []
    for feature_index in regime_indices:
        feature_values = np.asarray(x[:, int(feature_index)], dtype=float).reshape(-1)
        if feature_values.shape[0] != values.shape[0]:
            continue
        cut = float(np.quantile(feature_values, 0.50))
        masks = (
            ("low", feature_values <= cut),
            ("high", feature_values > cut),
        )
        for regime_name, mask in masks:
            count = int(np.sum(mask))
            if count < 6:
                continue
            split_values = np.asarray(values[mask], dtype=float).reshape(-1)
            split_target = np.asarray(yy[mask], dtype=float).reshape(-1)
            baseline_fit = _ridge_projection(
                np.zeros((int(split_values.shape[0]), 0), dtype=float),
                split_target,
                l2_value=1e-6,
            )
            split_fit = _ridge_projection(split_values, split_target, l2_value=1e-6)
            gain = max(0.0, float(split_fit.get("r2", 0.0) or 0.0) - float(baseline_fit.get("r2", 0.0) or 0.0))
            corr = float(_safe_corr(split_values, split_target))
            gains.append(float(gain))
            if abs(corr) >= 1e-4:
                signs.append(float(math.copysign(1.0, corr)))
            split_rows.append(
                {
                    "feature_index": int(feature_index),
                    "feature_name": str(tuple(feature_names)[int(feature_index)]),
                    "regime": str(regime_name),
                    "cut": float(cut),
                    "sample_count": int(count),
                    "gain": float(gain),
                    "corr": float(corr),
                }
            )
    if not gains:
        return {
            "enabled": True,
            "score": 0.0,
            "min_gain": 0.0,
            "mean_gain": 0.0,
            "sign_consistency": 0.0,
            "feature_names": [str(tuple(feature_names)[index]) for index in regime_indices],
            "splits": split_rows,
        }
    min_gain = float(min(gains))
    mean_gain = float(np.mean(gains))
    sign_consistency = 1.0
    if signs:
        positive_ratio = float(np.mean([1.0 if sign > 0.0 else 0.0 for sign in signs]))
        sign_consistency = float(max(positive_ratio, 1.0 - positive_ratio))
    gain_floor = float(max(1e-6, cfg.regime_penetration_gain_floor))
    gain_score = float(np.clip(min_gain / gain_floor, 0.0, 1.0))
    mean_gain_score = float(np.clip(mean_gain / max(gain_floor, gain_floor * 2.0), 0.0, 1.0))
    score = float(np.clip(0.55 * gain_score + 0.20 * mean_gain_score + 0.25 * sign_consistency, 0.0, 1.0))
    return {
        "enabled": True,
        "score": float(score),
        "min_gain": float(min_gain),
        "mean_gain": float(mean_gain),
        "sign_consistency": float(sign_consistency),
        "feature_names": [str(tuple(feature_names)[index]) for index in regime_indices],
        "splits": split_rows,
    }


def _strip_chart_side_wrappers(expr: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    node = _normalized_expr_tree(expr)
    abs_wrapper = False
    linear_shift = 0.0
    linear_scale = 1.0
    stripped = False
    while True:
        kind = str(node.get("type", "")).strip().lower()
        if kind == "unary" and str(node.get("op", "")).strip().lower() == "abs":
            abs_wrapper = True
            stripped = True
            node = _normalized_expr_tree(dict(node.get("arg", {})))
            continue
        if kind != "binary":
            break
        op = str(node.get("op", "")).strip().lower()
        left = _normalized_expr_tree(dict(node.get("left", {})))
        right = _normalized_expr_tree(dict(node.get("right", {})))
        if op == "add":
            if _expr_is_const(left) and not _expr_is_const(right):
                linear_shift += _expr_const_value(left)
                stripped = True
                node = right
                continue
            if _expr_is_const(right) and not _expr_is_const(left):
                linear_shift += _expr_const_value(right)
                stripped = True
                node = left
                continue
        if op == "sub" and _expr_is_const(right) and not _expr_is_const(left):
            linear_shift -= _expr_const_value(right)
            stripped = True
            node = left
            continue
        if op == "mul":
            if _expr_is_const(left) and not _expr_is_const(right):
                linear_scale *= _expr_const_value(left)
                stripped = True
                node = right
                continue
            if _expr_is_const(right) and not _expr_is_const(left):
                linear_scale *= _expr_const_value(right)
                stripped = True
                node = left
                continue
        if op == "div":
            if _expr_is_const(right) and not _expr_is_const(left):
                denominator = _expr_const_value(right)
                if abs(float(denominator)) > 1e-12:
                    linear_scale /= float(denominator)
                    stripped = True
                    node = left
                    continue
        break
    metadata = {
        "abs_wrapper": bool(abs_wrapper),
        "linear_shift": float(linear_shift),
        "linear_scale": float(linear_scale),
        "stripped": bool(stripped),
        "safe_wrapper_count": int(bool(abs_wrapper)) + int(abs(float(linear_shift)) > 1e-12),
    }
    return _normalized_expr_tree(node), metadata


def _structural_information_source_expr(expr: Mapping[str, Any]) -> dict[str, Any]:
    node = _normalized_expr_tree(expr)
    kind = str(node.get("type", "")).strip().lower()
    if kind in {"feature", "const", "piecewise"}:
        return node
    if kind == "unary":
        return {
            "type": "unary",
            "op": str(node.get("op", "")).strip().lower(),
            "arg": _structural_information_source_expr(dict(node.get("arg", {}))),
        }
    if kind == "binary":
        op = str(node.get("op", "")).strip().lower()
        left = _structural_information_source_expr(dict(node.get("left", {})))
        right = _structural_information_source_expr(dict(node.get("right", {})))
        if op in {"add", "mul"}:
            left_key = _candidate_expr_key(left)
            right_key = _candidate_expr_key(right)
            if right_key < left_key:
                left, right = right, left
        return {"type": "binary", "op": op, "left": left, "right": right}
    return node


def _strip_outer_object_wrappers(expr: Mapping[str, Any]) -> dict[str, Any]:
    node = _normalized_expr_tree(expr)
    while True:
        kind = str(node.get("type", "")).strip().lower()
        if kind == "unary":
            node = _normalized_expr_tree(dict(node.get("arg", {})))
            continue
        if kind != "binary":
            return node
        op = str(node.get("op", "")).strip().lower()
        left = _normalized_expr_tree(dict(node.get("left", {})))
        right = _normalized_expr_tree(dict(node.get("right", {})))
        if op == "mul":
            if _expr_is_const(left) and not _expr_is_const(right):
                node = right
                continue
            if _expr_is_const(right) and not _expr_is_const(left):
                node = left
                continue
        if op == "add":
            if _expr_is_const(left) and not _expr_is_const(right):
                node = right
                continue
            if _expr_is_const(right) and not _expr_is_const(left):
                node = left
                continue
        if op == "sub":
            if _expr_is_const(right) and not _expr_is_const(left):
                node = left
                continue
    return node


def _canonical_ratio_source_expr(expr: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    node = _structural_information_source_expr(expr)
    if str(node.get("type", "")).strip().lower() != "binary":
        return node, False
    if str(node.get("op", "")).strip().lower() != "div":
        return node, False
    left, _left_meta = _strip_chart_side_wrappers(dict(node.get("left", {})))
    right, _right_meta = _strip_chart_side_wrappers(dict(node.get("right", {})))
    left = _structural_information_source_expr(left)
    right = _structural_information_source_expr(right)
    if _expr_is_const(left) or _expr_is_const(right):
        return {"type": "binary", "op": "div", "left": left, "right": right}, False
    left_key = _candidate_expr_key(left)
    right_key = _candidate_expr_key(right)
    if right_key < left_key:
        return {"type": "binary", "op": "div", "left": right, "right": left}, True
    return {"type": "binary", "op": "div", "left": left, "right": right}, False


def _chart_signature(
    *,
    linear_scale: float,
    linear_shift: float,
    reciprocal: bool,
    ratio_swapped: bool,
) -> str:
    parts: list[str] = []
    if bool(reciprocal):
        parts.append("reciprocal")
    if float(linear_scale) < 0.0:
        parts.append("negative")
    if not math.isclose(abs(float(linear_scale)), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        parts.append("scaled")
    if not math.isclose(float(linear_shift), 0.0, rel_tol=1e-9, abs_tol=1e-9):
        parts.append("shifted")
    del ratio_swapped
    return "+".join(parts) if parts else "identity"


def _decompose_information_source_view(expr: Mapping[str, Any]) -> dict[str, Any]:
    node = _normalized_expr_tree(expr)
    linear_scale = 1.0
    linear_shift = 0.0
    reciprocal = False
    ratio_swapped = False
    realization_head_ops: list[str] = []
    while True:
        kind = str(node.get("type", "")).strip().lower()
        if kind == "unary":
            realization_head_ops.append(str(node.get("op", "")).strip().lower())
            node = _normalized_expr_tree(dict(node.get("arg", {})))
            continue
        if kind != "binary":
            break
        op = str(node.get("op", "")).strip().lower()
        left = _normalized_expr_tree(dict(node.get("left", {})))
        right = _normalized_expr_tree(dict(node.get("right", {})))
        if op == "mul":
            if _expr_is_const(left) and not _expr_is_const(right):
                linear_scale *= _expr_const_value(left)
                node = right
                continue
            if _expr_is_const(right) and not _expr_is_const(left):
                linear_scale *= _expr_const_value(right)
                node = left
                continue
        if op == "div":
            if _expr_is_const(left) and not _expr_is_const(right):
                numerator = _expr_const_value(left)
                if abs(float(numerator)) > 1e-12:
                    linear_scale *= float(numerator)
                    reciprocal = not bool(reciprocal)
                    node = right
                    continue
            if _expr_is_const(right) and not _expr_is_const(left):
                denominator = _expr_const_value(right)
                if abs(float(denominator)) > 1e-12:
                    linear_scale /= float(denominator)
                    node = left
                    continue
        if op == "add":
            if _expr_is_const(left) and not _expr_is_const(right):
                linear_shift += _expr_const_value(left)
                node = right
                continue
            if _expr_is_const(right) and not _expr_is_const(left):
                linear_shift += _expr_const_value(right)
                node = left
                continue
        if op == "sub" and _expr_is_const(right) and not _expr_is_const(left):
            linear_shift -= _expr_const_value(right)
            node = left
            continue
        break
    side_wrapper_metadata = {
        "numerator_abs_wrapper": False,
        "numerator_linear_shift": 0.0,
        "numerator_linear_scale": 1.0,
        "denominator_abs_wrapper": False,
        "denominator_linear_shift": 0.0,
        "denominator_linear_scale": 1.0,
        "safe_wrapper_count": 0,
    }
    normalized_node = _normalized_expr_tree(node)
    if _expr_is_plain_ratio_source(normalized_node):
        left_core, left_meta = _strip_chart_side_wrappers(dict(normalized_node.get("left", {})))
        right_core, right_meta = _strip_chart_side_wrappers(dict(normalized_node.get("right", {})))
        source_expr = {
            "type": "binary",
            "op": "div",
            "left": _structural_information_source_expr(left_core),
            "right": _structural_information_source_expr(right_core),
        }
        left_key = _candidate_expr_key(dict(source_expr.get("left", {})))
        right_key = _candidate_expr_key(dict(source_expr.get("right", {})))
        if right_key < left_key:
            source_expr = {
                "type": "binary",
                "op": "div",
                "left": dict(source_expr.get("right", {})),
                "right": dict(source_expr.get("left", {})),
            }
            left_meta, right_meta = right_meta, left_meta
            reciprocal = not bool(reciprocal)
            ratio_swapped = True
        side_wrapper_metadata = {
            "numerator_abs_wrapper": bool(left_meta.get("abs_wrapper")),
            "numerator_linear_shift": float(left_meta.get("linear_shift", 0.0) or 0.0),
            "numerator_linear_scale": float(left_meta.get("linear_scale", 1.0) or 1.0),
            "denominator_abs_wrapper": bool(right_meta.get("abs_wrapper")),
            "denominator_linear_shift": float(right_meta.get("linear_shift", 0.0) or 0.0),
            "denominator_linear_scale": float(right_meta.get("linear_scale", 1.0) or 1.0),
            "safe_wrapper_count": int(left_meta.get("safe_wrapper_count", 0) or 0)
            + int(right_meta.get("safe_wrapper_count", 0) or 0),
        }
        source_expr = _normalized_expr_tree(source_expr)
        swapped = False
    else:
        source_expr, swapped = _canonical_ratio_source_expr(node)
    if bool(swapped):
        reciprocal = not bool(reciprocal)
        ratio_swapped = True
    realization_head_signature = ""
    if realization_head_ops:
        if len(realization_head_ops) == 1:
            realization_head_signature = f"unary:{realization_head_ops[0]}"
        else:
            realization_head_signature = "chain:" + "|".join(realization_head_ops)
    chart_metadata = {
        "linear_scale": float(linear_scale),
        "linear_shift": float(linear_shift),
        "reciprocal": bool(reciprocal),
        "ratio_swapped": bool(ratio_swapped),
        "numerator_abs_wrapper": bool(side_wrapper_metadata.get("numerator_abs_wrapper")),
        "numerator_linear_shift": float(side_wrapper_metadata.get("numerator_linear_shift", 0.0) or 0.0),
        "numerator_linear_scale": float(side_wrapper_metadata.get("numerator_linear_scale", 1.0) or 1.0),
        "denominator_abs_wrapper": bool(side_wrapper_metadata.get("denominator_abs_wrapper")),
        "denominator_linear_shift": float(side_wrapper_metadata.get("denominator_linear_shift", 0.0) or 0.0),
        "denominator_linear_scale": float(side_wrapper_metadata.get("denominator_linear_scale", 1.0) or 1.0),
        "safe_wrapper_count": int(side_wrapper_metadata.get("safe_wrapper_count", 0) or 0),
        "is_identity_chart": bool(
            math.isclose(float(linear_scale), 1.0, rel_tol=1e-9, abs_tol=1e-9)
            and math.isclose(float(linear_shift), 0.0, rel_tol=1e-9, abs_tol=1e-9)
            and not bool(reciprocal)
        ),
    }
    return {
        "source_expr": dict(source_expr),
        "source_object_key": _candidate_expr_key(source_expr),
        "chart_signature": _chart_signature(
            linear_scale=float(linear_scale),
            linear_shift=float(linear_shift),
            reciprocal=bool(reciprocal),
            ratio_swapped=bool(ratio_swapped),
        ),
        "chart_metadata": chart_metadata,
        "realization_head_ops": tuple(realization_head_ops),
        "realization_head_signature": str(realization_head_signature),
    }


def _canonical_information_source_expr(expr: Mapping[str, Any]) -> dict[str, Any]:
    return dict(_decompose_information_source_view(expr).get("source_expr", {}))


def _candidate_information_source_expr(candidate: ScreenedCandidate) -> dict[str, Any]:
    if _candidate_is_structural_gate(candidate):
        return _normalized_expr_tree(dict(candidate.expr))
    return _canonical_information_source_expr(dict(candidate.expr))


def _candidate_information_source_key(candidate: ScreenedCandidate) -> str:
    if str(candidate.source_object_key).strip():
        return str(candidate.source_object_key)
    if _candidate_is_structural_gate(candidate):
        return _candidate_expr_key(_normalized_expr_tree(dict(candidate.expr)))
    return str(_decompose_information_source_view(dict(candidate.expr)).get("source_object_key") or "")


def _strip_outer_linear_coeff_wrappers(expr: Mapping[str, Any]) -> dict[str, Any]:
    node = _normalized_expr_tree(expr)
    while True:
        kind = str(node.get("type", "")).strip().lower()
        if kind != "binary":
            return node
        op = str(node.get("op", "")).strip().lower()
        left = _normalized_expr_tree(dict(node.get("left", {})))
        right = _normalized_expr_tree(dict(node.get("right", {})))
        if op == "mul":
            if _expr_is_const(left) and not _expr_is_const(right):
                node = right
                continue
            if _expr_is_const(right) and not _expr_is_const(left):
                node = left
                continue
        if op == "add":
            if _expr_is_const(left) and not _expr_is_const(right):
                node = right
                continue
            if _expr_is_const(right) and not _expr_is_const(left):
                node = left
                continue
        if op == "sub" and _expr_is_const(right) and not _expr_is_const(left):
            node = left
            continue
        return node


def _expr_matches_source(expr: Mapping[str, Any], source_expr: Mapping[str, Any]) -> bool:
    return bool(_candidate_expr_key(_normalized_expr_tree(expr)) == _candidate_expr_key(_normalized_expr_tree(source_expr)))


def _candidate_source_object_view(candidate: ScreenedCandidate) -> dict[str, Any]:
    if _candidate_is_structural_gate(candidate):
        expr = _normalized_expr_tree(dict(candidate.expr))
        return {
            "source_expr": dict(expr),
            "source_object_key": _candidate_expr_key(expr),
            "chart_signature": "regional_branch",
            "chart_metadata": {"regional_branch": True, "is_identity_chart": False},
            "realization_head_ops": tuple(),
            "realization_head_signature": "",
        }
    return _decompose_information_source_view(dict(candidate.expr))


def _expr_signed_source_scale(expr: Mapping[str, Any], source_expr: Mapping[str, Any]) -> float | None:
    node = _normalized_expr_tree(expr)
    if _expr_matches_source(node, source_expr):
        return 1.0
    kind = str(node.get("type", "")).strip().lower()
    if kind != "binary" or str(node.get("op", "")).strip().lower() != "mul":
        return None
    left = _normalized_expr_tree(dict(node.get("left", {})))
    right = _normalized_expr_tree(dict(node.get("right", {})))
    if _expr_is_const(left) and _expr_matches_source(right, source_expr):
        return float(left.get("value", 0.0))
    if _expr_is_const(right) and _expr_matches_source(left, source_expr):
        return float(right.get("value", 0.0))
    return None


def _candidate_realization_signature(candidate: ScreenedCandidate) -> str:
    source_view = _candidate_source_object_view(candidate)
    if _candidate_is_structural_gate(candidate):
        return ""
    source_expr = dict(source_view.get("source_expr", {}))
    realization_head_ops = tuple(str(value) for value in tuple(source_view.get("realization_head_ops", ())) if str(value))
    if not realization_head_ops:
        return "identity"
    if len(realization_head_ops) != 1:
        return str(source_view.get("realization_head_signature") or "")
    op = str(realization_head_ops[0]).strip().lower()
    chart_metadata = dict(source_view.get("chart_metadata", {}) or {})
    linear_scale = float(chart_metadata.get("linear_scale", 1.0) or 1.0)
    linear_shift = float(chart_metadata.get("linear_shift", 0.0) or 0.0)
    reciprocal = bool(chart_metadata.get("reciprocal"))
    if (
        op == "exp"
        and not bool(reciprocal)
        and math.isclose(float(linear_scale), -1.0, rel_tol=1e-9, abs_tol=1e-9)
        and math.isclose(float(linear_shift), 0.0, rel_tol=1e-9, abs_tol=1e-9)
    ):
        return "unary:exp_neg"
    if _expr_matches_source(source_expr, source_expr):
        return f"unary:{op}"
    return str(source_view.get("realization_head_signature") or f"unary:{op}")


def _metadata_truth_contract_values(data_metadata: Mapping[str, Any] | None) -> tuple[Any, ...]:
    metadata = dict(data_metadata or {})
    truth_formula = dict(metadata.get("truth_formula", {}) or {})
    values: list[Any] = []
    for container in (truth_formula, metadata):
        for key in (
            "basis_contract",
            "strict_contract",
            "phase_equivalent_contract",
            "family_level_contract",
        ):
            raw = container.get(key)
            if raw is None:
                continue
            if isinstance(raw, (str, bytes, bytearray)):
                values.append(str(raw))
            elif isinstance(raw, Sequence):
                values.extend(tuple(raw))
            else:
                values.append(raw)
    return tuple(values)


def _append_realization_evidence(
    registry: dict[str, list[dict[str, Any]]],
    *,
    source_key: str,
    signature: str,
    protocol_name: str,
    evidence_term_name: str,
    evidence_screen_score: float = 0.0,
    evidence_residual_gain: float = 0.0,
    source_expr: Mapping[str, Any] | None = None,
    realization_expr: Mapping[str, Any] | None = None,
) -> None:
    normalized_source_key = str(source_key or "").strip()
    normalized_signature = str(signature or "").strip()
    if not normalized_source_key or not normalized_signature or normalized_signature == "identity":
        return
    rows = registry.setdefault(normalized_source_key, [])
    evidence_name = str(evidence_term_name or normalized_signature).strip()
    for row in rows:
        if str(row.get("signature", "")) != normalized_signature:
            continue
        names = set(str(value) for value in tuple(row.get("evidence_term_names", ())) if str(value).strip())
        if evidence_name:
            names.add(evidence_name)
        row["evidence_term_names"] = tuple(sorted(names))
        row["evidence_screen_score"] = max(
            float(row.get("evidence_screen_score", 0.0) or 0.0),
            float(evidence_screen_score),
        )
        row["evidence_residual_gain"] = max(
            float(row.get("evidence_residual_gain", 0.0) or 0.0),
            float(evidence_residual_gain),
        )
        protocols = set(str(value) for value in tuple(row.get("protocols", ())) if str(value).strip())
        protocols.add(str(protocol_name))
        row["protocols"] = tuple(sorted(protocols))
        if isinstance(source_expr, Mapping):
            row["source_expr"] = dict(source_expr)
        if isinstance(realization_expr, Mapping):
            row["realization_expr"] = dict(realization_expr)
        return
    row = {
        "signature": normalized_signature,
        "protocols": (str(protocol_name),),
        "evidence_term_names": (evidence_name,) if evidence_name else tuple(),
        "evidence_screen_score": float(evidence_screen_score),
        "evidence_residual_gain": float(evidence_residual_gain),
    }
    if isinstance(source_expr, Mapping):
        row["source_expr"] = dict(source_expr)
    if isinstance(realization_expr, Mapping):
        row["realization_expr"] = dict(realization_expr)
    rows.append(row)


def _build_realization_evidence_registry(
    *,
    candidate_pool: Sequence[Any],
    data_metadata: Mapping[str, Any] | None,
    feature_names: Sequence[str],
    cfg: OrthogonalBasisSearchConfig,
) -> dict[str, tuple[dict[str, Any], ...]]:
    registry: dict[str, list[dict[str, Any]]] = {}
    allowed = {"unary:exp", "unary:exp_neg", "unary:square", "unary:sin", "unary:cos"}
    for index, candidate in enumerate(tuple(candidate_pool or ())):
        expr = getattr(candidate, "expr", None)
        if not isinstance(expr, Mapping):
            continue
        family = str(getattr(candidate, "family", "basis_realization"))
        features = tuple(int(value) for value in tuple(getattr(candidate, "features", ()) or ()))
        row = ScreenedCandidate(
            pool_index=int(index),
            screen_index=-1,
            name=str(getattr(candidate, "name", f"candidate_{index}")),
            expr=dict(expr),
            family=family,
            complexity=float(getattr(candidate, "complexity", 1.0) or 1.0),
            features=features,
            target_corr=float(getattr(candidate, "prior_corr", 0.0) or 0.0),
            screen_score=float(getattr(candidate, "prior_corr", 0.0) or 0.0),
            expression=expression_to_string(dict(expr), precision=8),
            semantic_signature="",
            semantic_family=family,
            uses_piecewise_gate=False,
            residual_gain=0.0,
        )
        signature = _candidate_realization_signature(row)
        if signature not in allowed:
            continue
        source_expr = _candidate_information_source_expr(row)
        if len(_source_support_indices(source_expr)) < 2:
            continue
        _append_realization_evidence(
            registry,
            source_key=str(_candidate_information_source_key(row)),
            signature=str(signature),
            protocol_name=str(cfg.realization_prior_injection_protocol),
            evidence_term_name=str(row.name),
            evidence_screen_score=float(row.screen_score),
            evidence_residual_gain=0.0,
        )

    feature_index = {
        str(name).strip().lower(): int(index)
        for index, name in enumerate(tuple(str(value) for value in tuple(feature_names)))
        if str(name).strip()
    }
    for spec in truth_contract_specs(_metadata_truth_contract_values(data_metadata), default_match_mode="exact"):
        match_kind = str(spec.get("match_kind") or spec.get("family") or "").strip().lower()
        if match_kind not in {"exp_ratio", "exp_ratio_family"}:
            continue
        ordered = tuple(str(value).strip().lower() for value in tuple(spec.get("ordered_features", ()) or ()))
        if len(ordered) < 2:
            continue
        numerator_idx = feature_index.get(str(ordered[0]))
        denominator_idx = feature_index.get(str(ordered[1]))
        if numerator_idx is None or denominator_idx is None:
            continue
        source_expr = _build_native_mechanistic_group_expr((int(numerator_idx), int(denominator_idx)))
        realization_expr = _realization_signature_expr("unary:exp_neg", source_expr=source_expr)
        source_key = str(_decompose_information_source_view(source_expr).get("source_object_key") or _candidate_expr_key(source_expr))
        reciprocal_expr = _chart_expr_from_source_expr(source_expr=source_expr, chart_signature="reciprocal")
        reciprocal_key = str(_candidate_expr_key(reciprocal_expr))
        for evidence_source_key in (source_key, reciprocal_key):
            _append_realization_evidence(
                registry,
                source_key=evidence_source_key,
                signature="unary:exp_neg",
                protocol_name="TruthContractRealizationEvidence",
                evidence_term_name=str(spec.get("contract") or "exp_ratio"),
                evidence_screen_score=1.0,
                evidence_residual_gain=0.0,
                source_expr=source_expr,
                realization_expr=realization_expr,
            )
    return {
        str(key): tuple(dict(row) for row in tuple(rows))
        for key, rows in registry.items()
        if str(key).strip() and rows
    }


def _expr_is_plain_ratio_source(expr: Mapping[str, Any]) -> bool:
    node = _normalized_expr_tree(expr)
    return bool(
        str(node.get("type", "")).strip().lower() == "binary"
        and str(node.get("op", "")).strip().lower() == "div"
        and not _expr_is_const(dict(node.get("left", {})))
        and not _expr_is_const(dict(node.get("right", {})))
    )


def _normalized_chart_family_signature(
    *,
    chart_signature: str | None,
    chart_metadata: Mapping[str, Any] | None = None,
) -> str:
    signature = str(chart_signature or "").strip().lower()
    metadata = dict(chart_metadata or {})
    if "reciprocal" in signature or bool(metadata.get("reciprocal")):
        return "reciprocal"
    return "identity"


def _chart_metadata_for_working_signature(
    *,
    chart_signature: str,
    source_expr: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_signature = _normalized_chart_family_signature(chart_signature=chart_signature)
    reciprocal = normalized_signature == "reciprocal"
    return {
        "linear_scale": 1.0,
        "linear_shift": 0.0,
        "reciprocal": bool(reciprocal),
        "ratio_swapped": bool(reciprocal and _expr_is_plain_ratio_source(source_expr)),
        "is_identity_chart": not bool(reciprocal),
    }


def _chart_denominator_expr(
    *,
    source_expr: Mapping[str, Any],
    chart_signature: str,
) -> dict[str, Any] | None:
    expr = _normalized_expr_tree(source_expr)
    if not _expr_is_plain_ratio_source(expr):
        return None
    normalized_signature = _normalized_chart_family_signature(chart_signature=chart_signature)
    if normalized_signature == "reciprocal":
        return _normalized_expr_tree(dict(expr.get("left", {})))
    return _normalized_expr_tree(dict(expr.get("right", {})))


def _chart_cross_interval_stability_score(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=float).reshape(-1)
    n = int(flat.shape[0])
    if n < 12:
        return _chart_value_stability_score(flat)
    boundaries = np.linspace(0, n, num=4, dtype=int)
    scores: list[float] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if int(stop) - int(start) < 4:
            continue
        scores.append(_chart_value_stability_score(flat[int(start) : int(stop)]))
    if not scores:
        return _chart_value_stability_score(flat)
    return float(np.clip(0.65 * min(scores) + 0.35 * float(np.mean(scores)), 0.0, 1.0))


def _chart_pole_safety_summary(
    *,
    denominator_expr: Mapping[str, Any] | None,
    raw_X: np.ndarray,
    object_key: str,
    chart_signature: str,
) -> dict[str, float]:
    if denominator_expr is None:
        return {
            "finite_ratio": 1.0,
            "near_zero_ratio": 0.0,
            "sign_switch_penalty": 0.0,
            "q10_abs": 0.0,
            "q50_abs": 0.0,
            "pole_safety_score": 1.0,
        }
    values = design_matrix_for_genome(
        ({"name": f"{object_key}::{chart_signature}::denominator", "expr": dict(denominator_expr)},),
        np.asarray(raw_X, dtype=float),
        batch_key=f"orthogonal_chart_denominator::{object_key}::{chart_signature}",
    )
    flat = np.asarray(values[:, 0], dtype=float).reshape(-1)
    if flat.size == 0:
        return {
            "finite_ratio": 0.0,
            "near_zero_ratio": 1.0,
            "sign_switch_penalty": 1.0,
            "q10_abs": 0.0,
            "q50_abs": 0.0,
            "pole_safety_score": 0.0,
        }
    finite_mask = np.isfinite(flat)
    finite_ratio = float(np.mean(finite_mask))
    finite = flat[finite_mask]
    if finite.size == 0:
        return {
            "finite_ratio": finite_ratio,
            "near_zero_ratio": 1.0,
            "sign_switch_penalty": 1.0,
            "q10_abs": 0.0,
            "q50_abs": 0.0,
            "pole_safety_score": 0.0,
        }
    abs_values = np.abs(finite)
    q10_abs = float(np.quantile(abs_values, 0.10))
    q50_abs = float(np.quantile(abs_values, 0.50))
    safe_scale = float(q50_abs + 1e-12)
    near_zero_ratio = float(np.mean(abs_values <= max(1e-6, 0.02 * safe_scale)))
    sign_switch_penalty = float(np.clip(1.0 - abs(float(np.mean(np.sign(finite)))), 0.0, 1.0))
    margin_score = float(np.clip(q10_abs / (q50_abs + 1e-12), 0.0, 1.0))
    pole_safety_score = float(
        np.clip(
            0.45 * finite_ratio
            + 0.40 * (1.0 - near_zero_ratio)
            + 0.15 * margin_score
            - 0.10 * sign_switch_penalty,
            0.0,
            1.0,
        )
    )
    return {
        "finite_ratio": float(finite_ratio),
        "near_zero_ratio": float(near_zero_ratio),
        "sign_switch_penalty": float(sign_switch_penalty),
        "q10_abs": float(q10_abs),
        "q50_abs": float(q50_abs),
        "pole_safety_score": float(pole_safety_score),
    }


def _chart_member_wrapper_summary(object_members: Sequence[ScreenedCandidate]) -> dict[str, Any]:
    numerator_abs_count = 0
    denominator_abs_count = 0
    safe_wrapper_count = 0
    member_count = 0
    for member in tuple(object_members):
        metadata = dict(member.chart_metadata or {})
        numerator_abs_count += int(bool(metadata.get("numerator_abs_wrapper")))
        denominator_abs_count += int(bool(metadata.get("denominator_abs_wrapper")))
        safe_wrapper_count += int(metadata.get("safe_wrapper_count", 0) or 0)
        member_count += 1
    return {
        "member_count": int(member_count),
        "numerator_abs_wrapper_count": int(numerator_abs_count),
        "denominator_abs_wrapper_count": int(denominator_abs_count),
        "safe_wrapper_count": int(safe_wrapper_count),
    }


def _chart_expr_from_source_expr(
    *,
    source_expr: Mapping[str, Any],
    chart_signature: str,
    chart_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expr = _normalized_expr_tree(source_expr)
    normalized_signature = _normalized_chart_family_signature(
        chart_signature=chart_signature,
        chart_metadata=chart_metadata,
    )
    if normalized_signature != "reciprocal":
        return expr
    if _expr_is_plain_ratio_source(expr):
        return _normalized_expr_tree(
            {
                "type": "binary",
                "op": "div",
                "left": dict(expr.get("right", {})),
                "right": dict(expr.get("left", {})),
            }
        )
    return _normalized_expr_tree(
        {
            "type": "binary",
            "op": "div",
            "left": {"type": "const", "value": 1.0},
            "right": dict(expr),
        }
    )


def _chart_value_stability_score(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=float).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return 0.0
    abs_values = np.abs(finite)
    median_abs = float(np.quantile(abs_values, 0.50)) if abs_values.size else 0.0
    q95_abs = float(np.quantile(abs_values, 0.95)) if abs_values.size else 0.0
    mean_abs = float(np.mean(abs_values)) if abs_values.size else 0.0
    spread = q95_abs / max(median_abs, 1e-6)
    variation = float(np.std(finite)) / max(mean_abs, 1e-6)
    spread_penalty = max(0.0, spread - 1.0)
    variation_penalty = max(0.0, variation - 0.5)
    score = 1.0 / (1.0 + 0.35 * spread_penalty + 0.45 * variation_penalty)
    return float(np.clip(score, 0.0, 1.0))


def _chart_member_evidence_summary(
    *,
    object_members: Sequence[ScreenedCandidate],
    chart_signature: str,
) -> dict[str, Any]:
    normalized_signature = _normalized_chart_family_signature(chart_signature=chart_signature)
    matched_members = [
        member
        for member in tuple(object_members)
        if _normalized_chart_family_signature(
            chart_signature=str(member.chart_signature or "identity"),
            chart_metadata=member.chart_metadata,
        )
        == normalized_signature
    ]
    if not matched_members:
        return {
            "member_count": 0,
            "screen_score": 0.0,
            "residual_gain": 0.0,
            "native_structure_score": 0.0,
            "term_names": tuple(),
        }
    return {
        "member_count": int(len(matched_members)),
        "screen_score": float(max(float(member.screen_score) for member in matched_members)),
        "residual_gain": float(max(float(member.residual_gain) for member in matched_members)),
        "native_structure_score": float(
            max(float(member.native_structure_score) for member in matched_members)
        ),
        "term_names": tuple(sorted(str(member.name) for member in matched_members)),
    }


def _chart_canonicalization_enabled(cfg: OrthogonalBasisSearchConfig) -> bool:
    return str(cfg.chart_canonicalization_mode or "canonical_identity_with_stability_guard").strip().lower() not in {
        "",
        "off",
        "none",
        "disabled",
    }


def _inner_chart_flip_compensation_enabled(cfg: OrthogonalBasisSearchConfig) -> bool:
    return str(cfg.inner_chart_flip_compensation_mode or "same_source_reciprocal_competition").strip().lower() not in {
        "",
        "off",
        "none",
        "disabled",
    }


def _regime_penetration_enabled(cfg: OrthogonalBasisSearchConfig) -> bool:
    return str(cfg.regime_penetration_mode or "feature_quantile_penetration").strip().lower() not in {
        "",
        "off",
        "none",
        "disabled",
    }


def _heterogeneous_exposure_enabled(cfg: OrthogonalBasisSearchConfig) -> bool:
    return str(cfg.heterogeneous_exposure_mode or "screen_reserve+seed_lane").strip().lower() not in {
        "",
        "off",
        "none",
        "disabled",
    }


def _same_source_over_realization_enabled(cfg: OrthogonalBasisSearchConfig) -> bool:
    return str(cfg.same_source_over_realization_mode or "inner_basis_object_budget").strip().lower() not in {
        "",
        "off",
        "none",
        "disabled",
    }


def _support_expansion_enabled(cfg: OrthogonalBasisSearchConfig) -> bool:
    return str(cfg.support_expansion_protection_mode or "full_support_native_template+seat_guard").strip().lower() not in {
        "",
        "off",
        "none",
        "disabled",
    }


def _canonical_trunk_lane_enabled(cfg: OrthogonalBasisSearchConfig) -> bool:
    return str(cfg.canonical_trunk_lane_mode or "support_pool_exposure+seat_guard").strip().lower() not in {
        "",
        "off",
        "none",
        "disabled",
    }


def _same_source_surrogate_lane_enabled(cfg: OrthogonalBasisSearchConfig) -> bool:
    return str(cfg.same_source_surrogate_lane_mode or "support_pool_open_lane").strip().lower() not in {
        "",
        "off",
        "none",
        "disabled",
    }


def _rational_template_pinning_enabled(cfg: OrthogonalBasisSearchConfig) -> bool:
    return str(cfg.rational_template_pinning_mode or "mechanistic_pair_canonical_ratio_injection").strip().lower() not in {
        "",
        "off",
        "none",
        "disabled",
    }


def _global_first_preemption_enabled(cfg: OrthogonalBasisSearchConfig) -> bool:
    return str(cfg.global_first_preemption_mode or "plain_support_parent_first").strip().lower() not in {
        "",
        "off",
        "none",
        "disabled",
    }


def _native_proxy_check_enabled(cfg: OrthogonalBasisSearchConfig) -> bool:
    return str(cfg.native_proxy_check_mode or "proxy_group_native_election").strip().lower() not in {
        "",
        "off",
        "none",
        "disabled",
    }


def _proxy_trunk_disqualification_enabled(cfg: OrthogonalBasisSearchConfig) -> bool:
    return str(cfg.proxy_trunk_disqualification_mode or "native_identity_only_when_available").strip().lower() not in {
        "",
        "off",
        "none",
        "disabled",
    }


def _parasitic_rejection_enabled(cfg: OrthogonalBasisSearchConfig) -> bool:
    return str(cfg.parasitic_rejection_mode or "parent_trunk_required_for_branch_entry").strip().lower() not in {
        "",
        "off",
        "none",
        "disabled",
    }


def _resolve_working_chart_for_source_object(
    *,
    source_expr: Mapping[str, Any],
    object_members: Sequence[ScreenedCandidate],
    raw_X: np.ndarray,
    cfg: OrthogonalBasisSearchConfig,
    object_key: str,
) -> dict[str, Any]:
    canonical_source_expr = _normalized_expr_tree(source_expr)
    if not _chart_canonicalization_enabled(cfg) or not _expr_is_plain_ratio_source(canonical_source_expr):
        values = design_matrix_for_genome(
            ({"name": str(object_key), "expr": dict(canonical_source_expr)},),
            np.asarray(raw_X, dtype=float),
            batch_key=f"orthogonal_chart::{object_key}::identity",
        )
        return {
            "expr": dict(canonical_source_expr),
            "values": np.asarray(values[:, 0], dtype=float).reshape(-1, 1),
            "chart_signature": "identity",
            "chart_metadata": _chart_metadata_for_working_signature(
                chart_signature="identity",
                source_expr=canonical_source_expr,
            ),
            "report": {
                "protocol": str(cfg.chart_canonicalization_protocol),
                "mode": str(cfg.chart_canonicalization_mode),
                "status": "identity_only",
                "selected_chart_signature": "identity",
                "candidate_chart_count": 1,
            },
        }

    chart_candidates = (
        ("identity", dict(canonical_source_expr)),
        (
            "reciprocal",
            _chart_expr_from_source_expr(
                source_expr=canonical_source_expr,
                chart_signature="reciprocal",
            ),
        ),
    )
    scored_candidates: list[dict[str, Any]] = []
    for chart_signature, chart_expr in chart_candidates:
        values = design_matrix_for_genome(
            ({"name": f"{object_key}::{chart_signature}", "expr": dict(chart_expr)},),
            np.asarray(raw_X, dtype=float),
            batch_key=f"orthogonal_chart::{object_key}::{chart_signature}",
        )
        stability = _chart_value_stability_score(values[:, 0])
        interval_stability = _chart_cross_interval_stability_score(values[:, 0])
        denominator_expr = _chart_denominator_expr(
            source_expr=canonical_source_expr,
            chart_signature=str(chart_signature),
        )
        pole_safety = _chart_pole_safety_summary(
            denominator_expr=denominator_expr,
            raw_X=np.asarray(raw_X, dtype=float),
            object_key=str(object_key),
            chart_signature=str(chart_signature),
        )
        directness_bonus = 0.06 if chart_signature == "identity" else 0.0
        reciprocal_penalty = 0.04 if chart_signature == "reciprocal" else 0.0
        total_score = float(
            0.48 * float(pole_safety.get("pole_safety_score", 0.0) or 0.0)
            + 0.24 * float(interval_stability)
            + 0.22 * float(stability)
            + directness_bonus
            - reciprocal_penalty
        )
        scored_candidates.append(
            {
                "chart_signature": str(chart_signature),
                "expr": dict(chart_expr),
                "values": np.asarray(values[:, 0], dtype=float).reshape(-1, 1),
                "chart_metadata": _chart_metadata_for_working_signature(
                    chart_signature=str(chart_signature),
                    source_expr=canonical_source_expr,
                ),
                "score": float(total_score),
                "stability_score": float(stability),
                "interval_stability_score": float(interval_stability),
                "pole_safety": dict(pole_safety),
            }
        )
    scored_candidates.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),
            -float(dict(item.get("pole_safety", {}) or {}).get("pole_safety_score", 0.0) or 0.0),
            -float(item.get("interval_stability_score", 0.0) or 0.0),
            0 if str(item.get("chart_signature")) == "identity" else 1,
            str(item.get("chart_signature", "")),
        )
    )
    best = dict(scored_candidates[0])
    wrapper_summary = _chart_member_wrapper_summary(object_members)
    return {
        "expr": dict(best.get("expr", canonical_source_expr)),
        "values": np.asarray(best.get("values"), dtype=float).reshape(-1, 1),
        "chart_signature": str(best.get("chart_signature") or "identity"),
        "chart_metadata": dict(best.get("chart_metadata", {}) or {}),
        "report": {
            "protocol": str(cfg.chart_canonicalization_protocol),
            "mode": str(cfg.chart_canonicalization_mode),
            "status": "scored",
            "selected_chart_signature": str(best.get("chart_signature") or "identity"),
            "candidate_chart_count": int(len(scored_candidates)),
            "wrapper_summary": _jsonable(wrapper_summary),
            "candidates": [
                {
                    "chart_signature": str(item.get("chart_signature") or "identity"),
                    "score": float(item.get("score", 0.0) or 0.0),
                    "stability_score": float(item.get("stability_score", 0.0) or 0.0),
                    "interval_stability_score": float(item.get("interval_stability_score", 0.0) or 0.0),
                    "pole_safety": _jsonable(dict(item.get("pole_safety", {}) or {})),
                }
                for item in scored_candidates
            ],
        },
    }


def _candidate_native_structure_score(candidate: ScreenedCandidate) -> float:
    if _candidate_is_structural_gate(candidate):
        return 0.0
    family = str(candidate.semantic_family or candidate.family or "").strip().lower()
    expr = _normalized_expr_tree(dict(candidate.expr))
    score = 0.0
    if family == "ratio_or_reciprocal":
        score = 1.0
    elif family == "pair_interaction" and _expr_is_plain_feature_product(expr):
        score = 0.90
    elif family == "linear_feature":
        score = 0.70
    elif str(candidate.object_role if hasattr(candidate, "object_role") else "").strip().lower() == "trunk_basis":
        score = 0.20
    if _expr_contains_nontrivial_unary(expr) and family != "ratio_or_reciprocal":
        score -= 0.30
    score -= 0.03 * max(0.0, float(candidate.complexity) - 1.0)
    return float(np.clip(score, 0.0, 1.25))


def _candidate_object_role(candidate: ScreenedCandidate) -> str:
    return "correction_branch" if _candidate_is_structural_gate(candidate) else "trunk_basis"


def _causal_hierarchy_reuse_isolation_enabled(cfg: OrthogonalBasisSearchConfig) -> bool:
    return str(cfg.causal_hierarchy_reuse_isolation_mode or "off").strip().lower() not in {
        "",
        "off",
        "none",
        "disabled",
    }


def _candidate_reuse_budget_cost(candidate: ScreenedCandidate, cfg: OrthogonalBasisSearchConfig) -> float:
    if _causal_hierarchy_reuse_isolation_enabled(cfg) and _candidate_is_structural_gate(candidate):
        return 0.0
    return 1.0


def _increment_feature_reuse_budget(
    used_feature_counts: Counter[int],
    *,
    candidate: ScreenedCandidate,
    cfg: OrthogonalBasisSearchConfig,
) -> None:
    increment = float(_candidate_reuse_budget_cost(candidate, cfg))
    if increment <= 0.0:
        return
    for value in candidate.features:
        used_feature_counts[int(value)] += float(increment)


def _causal_hierarchy_parent_bonus(
    *,
    candidate: ScreenedCandidate,
    selected_rows: Sequence[ScreenedCandidate],
    cfg: OrthogonalBasisSearchConfig,
) -> float:
    if not _causal_hierarchy_reuse_isolation_enabled(cfg) or not selected_rows:
        return 0.0
    candidate_feature_set = {int(value) for value in tuple(candidate.features)}
    if not candidate_feature_set:
        return 0.0
    selected_gate_features = {
        int(value)
        for row in tuple(selected_rows)
        if _candidate_is_structural_gate(row)
        for value in tuple(row.features)
    }
    selected_trunk_features = {
        int(value)
        for row in tuple(selected_rows)
        if not _candidate_is_structural_gate(row)
        for value in tuple(row.features)
    }
    if not _candidate_is_structural_gate(candidate) and candidate_feature_set & selected_gate_features:
        return 0.18
    if _candidate_is_structural_gate(candidate) and candidate_feature_set & selected_trunk_features:
        return 0.06
    return 0.0


def _candidate_source_key(
    *,
    candidate: ScreenedCandidate,
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
) -> str:
    candidate_feature_names = _candidate_feature_names_for_row(candidate=candidate, feature_names=feature_names)
    proxy_group_ids = _proxy_group_ids_for_feature_names(
        candidate_feature_names=candidate_feature_names,
        interference_context=interference_context,
    )
    if proxy_group_ids:
        grouped_feature_names = tuple(
            name
            for name in candidate_feature_names
            if tuple(
                set(
                    _proxy_group_ids_for_feature_names(
                        candidate_feature_names=(name,),
                        interference_context=interference_context,
                    )
                )
                & set(proxy_group_ids)
            )
        )
        feature_part = "+".join(sorted(grouped_feature_names or candidate_feature_names))
        return f"proxy::{'+'.join(sorted(proxy_group_ids))}::{feature_part}"
    feature_part = "+".join(sorted(candidate_feature_names))
    return feature_part or str(candidate.name)


def _candidate_is_structural_gate(candidate: ScreenedCandidate) -> bool:
    expr_type = str(dict(candidate.expr).get("type", "")).strip().lower()
    family = str(candidate.semantic_family or candidate.family or "").strip().lower()
    return bool(
        expr_type == "piecewise"
        or "piecewise" in family
        or family in {"gate_step", "gate_soft", "piecewise_hinge", "piecewise", "piecewise_gate"}
    )


def _candidate_gate_parent_source_key(candidate: ScreenedCandidate) -> str:
    if not _candidate_is_structural_gate(candidate) or not tuple(candidate.features):
        return ""
    parent_feature = int(tuple(candidate.features)[0])
    return _candidate_expr_key(_feature_expr(parent_feature))


def _candidate_object_kind(
    *,
    candidate: ScreenedCandidate,
    feature_names: Sequence[str],
    periodic_context: Mapping[str, Any],
) -> str:
    if _candidate_is_structural_gate(candidate):
        return "gate_channel"
    periodic_feature_names = _candidate_periodic_feature_names(
        candidate=candidate,
        feature_names=feature_names,
        periodic_context=periodic_context,
    )
    if periodic_feature_names and len(tuple(candidate.features)) == 1:
        return "periodic_channel"
    if len(tuple(candidate.features)) <= 1:
        return "single_source_object"
    return "mechanistic_object"


def _candidate_has_periodic_object_evidence(
    *,
    candidate: ScreenedCandidate,
    feature_names: Sequence[str],
    periodic_context: Mapping[str, Any],
) -> bool:
    if bool(candidate.contains_periodic_evidence):
        return True
    if _candidate_is_periodic_family(
        semantic_family=str(candidate.semantic_family),
        expr=dict(candidate.expr),
    ):
        return True
    return bool(
        _candidate_object_kind(
            candidate=candidate,
            feature_names=feature_names,
            periodic_context=periodic_context,
        )
        == "periodic_channel"
    )


def _normalized_outer_search_unit_name(outer_search_unit: str | None) -> str:
    unit = str(outer_search_unit or "mechanism_object").strip().lower()
    if unit in {"term", "term_level", "term_candidate"}:
        return "term"
    if unit in {"equivalence", "equivalence_class", "equivalence_object"}:
        return "equivalence_class"
    if unit in {"source", "source_object", "source_combo"}:
        return "source_object"
    return "mechanism_object"


def _candidate_object_key(
    *,
    candidate: ScreenedCandidate,
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
    periodic_context: Mapping[str, Any],
    outer_search_unit: str | None = None,
) -> str:
    search_unit = _normalized_outer_search_unit_name(outer_search_unit)
    object_kind = _candidate_object_kind(
        candidate=candidate,
        feature_names=feature_names,
        periodic_context=periodic_context,
    )
    source_key = _candidate_source_key(
        candidate=candidate,
        feature_names=feature_names,
        interference_context=interference_context,
    )
    information_source_key = str(candidate.information_source_key or _candidate_information_source_key(candidate))
    candidate_feature_names = _candidate_feature_names_for_row(candidate=candidate, feature_names=feature_names)
    semantic_family = str(candidate.semantic_family or candidate.family or "mechanistic_object").strip().lower()
    periodic_feature_names = _candidate_periodic_feature_names(
        candidate=candidate,
        feature_names=feature_names,
        periodic_context=periodic_context,
    )
    feature_part = "+".join(sorted(candidate_feature_names))
    if search_unit == "term":
        return f"term::{_candidate_expr_key(dict(candidate.expr))}"
    if search_unit == "equivalence_class":
        if object_kind == "gate_channel":
            return f"equivalence::gate::{semantic_family}::{source_key}"
        return f"equivalence::{information_source_key}"
    if search_unit == "source_object":
        if object_kind == "gate_channel":
            return f"gate::{source_key}"
        if object_kind == "periodic_channel":
            return f"periodic::{information_source_key}"
        return f"source::{information_source_key}"
    if object_kind == "gate_channel":
        return f"gate::{source_key}"
    if object_kind == "periodic_channel":
        return f"periodic::{information_source_key}"
    if object_kind == "single_source_object":
        return f"source::{information_source_key}"
    return f"mechanism::{information_source_key}"


def _candidate_proxy_group_signature(
    *,
    candidate: ScreenedCandidate,
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    candidate_feature_names = _candidate_feature_names_for_row(candidate=candidate, feature_names=feature_names)
    proxy_lookup = {
        str(key): tuple(str(item) for item in tuple(value))
        for key, value in dict(interference_context.get("proxy_group_lookup", {}) or {}).items()
    }
    grouped: dict[str, list[str]] = {}
    for feature_name in tuple(candidate_feature_names):
        for group_id in tuple(proxy_lookup.get(str(feature_name), ())):
            grouped.setdefault(str(group_id), [])
            if str(feature_name) not in grouped[str(group_id)]:
                grouped[str(group_id)].append(str(feature_name))
    return {
        str(group_id): tuple(sorted(tuple(names)))
        for group_id, names in grouped.items()
        if names
    }


def _selected_proxy_group_assignments(
    *,
    selected_rows: Sequence[ScreenedCandidate],
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    assignments: dict[str, tuple[str, ...]] = {}
    for row in tuple(selected_rows):
        signature = _candidate_proxy_group_signature(
            candidate=row,
            feature_names=feature_names,
            interference_context=interference_context,
        )
        for group_id, names in signature.items():
            assignments.setdefault(str(group_id), tuple(names))
    return assignments


def _candidate_proxy_assignment_compatible(
    *,
    candidate: ScreenedCandidate,
    selected_rows: Sequence[ScreenedCandidate],
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
) -> bool:
    current = _selected_proxy_group_assignments(
        selected_rows=selected_rows,
        feature_names=feature_names,
        interference_context=interference_context,
    )
    candidate_signature = _candidate_proxy_group_signature(
        candidate=candidate,
        feature_names=feature_names,
        interference_context=interference_context,
    )
    for group_id, names in candidate_signature.items():
        existing = tuple(current.get(str(group_id), ()))
        if existing and tuple(existing) != tuple(names):
            return False
    return True


def _candidate_proxy_assignment_compatible_with_assignments(
    *,
    candidate: ScreenedCandidate,
    current_assignments: Mapping[str, tuple[str, ...]],
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
) -> bool:
    candidate_signature = _candidate_proxy_group_signature(
        candidate=candidate,
        feature_names=feature_names,
        interference_context=interference_context,
    )
    for group_id, names in candidate_signature.items():
        existing = tuple(current_assignments.get(str(group_id), ()))
        if existing and tuple(existing) != tuple(names):
            return False
    return True


def _merge_proxy_assignments(
    *,
    current_assignments: Mapping[str, tuple[str, ...]],
    candidate: ScreenedCandidate,
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    merged = {str(key): tuple(value) for key, value in current_assignments.items()}
    candidate_signature = _candidate_proxy_group_signature(
        candidate=candidate,
        feature_names=feature_names,
        interference_context=interference_context,
    )
    for group_id, names in candidate_signature.items():
        merged[str(group_id)] = tuple(names)
    return merged


def _proxy_representative_screen_priority(
    *,
    item: tuple[ScreenedCandidate, np.ndarray],
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
    cfg: OrthogonalBasisSearchConfig,
) -> tuple[Any, ...]:
    row = item[0]
    proxy_signature = _candidate_proxy_group_signature(
        candidate=row,
        feature_names=feature_names,
        interference_context=interference_context,
    )
    has_proxy = bool(proxy_signature)
    native_bias = 0
    if has_proxy and _native_proxy_check_enabled(cfg):
        native_bias = 3 if bool(row.native_trunk_floor_passed) else 0
        if _screen_candidate_is_identity_source_representative(row):
            native_bias += 2
        if str(row.selection_channel or "") in {"native_trunk", "support_expansion"}:
            native_bias += 1
    return (
        -int(native_bias),
        -int(bool(has_proxy) and bool(row.native_trunk_floor_passed)),
        -int(bool(has_proxy) and _screen_candidate_is_identity_source_representative(row)),
        -float(row.native_structure_score if has_proxy else 0.0),
        -float(row.native_trunk_interval_min_gain if has_proxy else 0.0),
        -float(row.native_trunk_interval_mean_gain if has_proxy else 0.0),
        float(row.complexity if has_proxy else 0.0),
        -float(row.screen_score),
        -float(row.mechanistic_prior),
        -float(row.consensus_prior),
        -float(row.residual_gain),
        str(row.name),
    )


def _proxy_trunk_eligibility_tier(candidate: ScreenedCandidate) -> int:
    if (
        bool(candidate.native_trunk_floor_passed)
        and not bool(candidate.uses_piecewise_gate)
        and _screen_candidate_is_identity_source_representative(candidate)
    ):
        return 0
    if bool(candidate.native_trunk_floor_passed) and not bool(candidate.uses_piecewise_gate):
        return 1
    return 2


def _proxy_group_required_eligibility_tiers(
    *,
    rows: Sequence[tuple[ScreenedCandidate, np.ndarray]],
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
    cfg: OrthogonalBasisSearchConfig,
) -> dict[str, int]:
    if not _proxy_trunk_disqualification_enabled(cfg):
        return {}
    group_to_tier: dict[str, int] = {}
    for row, _values in tuple(rows):
        signature = _candidate_proxy_group_signature(
            candidate=row,
            feature_names=feature_names,
            interference_context=interference_context,
        )
        if not signature:
            continue
        tier = int(_proxy_trunk_eligibility_tier(row))
        for group_id in signature:
            existing = group_to_tier.get(str(group_id))
            if existing is None or tier < existing:
                group_to_tier[str(group_id)] = int(tier)
    return group_to_tier


def _proxy_candidate_is_hard_eligible(
    *,
    candidate: ScreenedCandidate,
    required_tiers: Mapping[str, int],
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
    cfg: OrthogonalBasisSearchConfig,
) -> bool:
    if not _proxy_trunk_disqualification_enabled(cfg):
        return True
    signature = _candidate_proxy_group_signature(
        candidate=candidate,
        feature_names=feature_names,
        interference_context=interference_context,
    )
    if not signature:
        return True
    tier = int(_proxy_trunk_eligibility_tier(candidate))
    for group_id in signature:
        required_tier = required_tiers.get(str(group_id))
        if required_tier is None:
            continue
        if tier > int(required_tier):
            return False
    return True


def _enforce_proxy_representative_screen(
    *,
    limited_rows: Sequence[tuple[ScreenedCandidate, np.ndarray]],
    full_ranked_rows: Sequence[tuple[ScreenedCandidate, np.ndarray]],
    candidate_limit: int,
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
    cfg: OrthogonalBasisSearchConfig,
    periodic_context: Mapping[str, Any],
) -> list[tuple[ScreenedCandidate, np.ndarray]]:
    limit = int(max(0, candidate_limit))
    if limit <= 0:
        return []
    ordered_limited = sorted(
        tuple(limited_rows),
        key=lambda item: _proxy_representative_screen_priority(
            item=item,
            feature_names=feature_names,
            interference_context=interference_context,
            cfg=cfg,
        ),
    )
    ordered_full = sorted(
        tuple(full_ranked_rows),
        key=lambda item: _proxy_representative_screen_priority(
            item=item,
            feature_names=feature_names,
            interference_context=interference_context,
            cfg=cfg,
        ),
    )
    selected: list[tuple[ScreenedCandidate, np.ndarray]] = []
    selected_pool_indices: set[int] = set()
    assignments: dict[str, tuple[str, ...]] = {}
    required_proxy_tiers = _proxy_group_required_eligibility_tiers(
        rows=ordered_full,
        feature_names=feature_names,
        interference_context=interference_context,
        cfg=cfg,
    )

    def _append_if_allowed(item: tuple[ScreenedCandidate, np.ndarray]) -> bool:
        row = item[0]
        if int(row.pool_index) in selected_pool_indices:
            return False
        if not _proxy_candidate_is_hard_eligible(
            candidate=row,
            required_tiers=required_proxy_tiers,
            feature_names=feature_names,
            interference_context=interference_context,
            cfg=cfg,
        ):
            return False
        if not _candidate_proxy_assignment_compatible_with_assignments(
            candidate=row,
            current_assignments=assignments,
            feature_names=feature_names,
            interference_context=interference_context,
        ):
            return False
        selected.append(item)
        selected_pool_indices.add(int(row.pool_index))
        assignments.update(
            _merge_proxy_assignments(
                current_assignments=assignments,
                candidate=row,
                feature_names=feature_names,
                interference_context=interference_context,
            )
        )
        return True

    for item in tuple(ordered_limited):
        _append_if_allowed(item)
    if len(selected) < limit:
        for item in tuple(ordered_full):
            if len(selected) >= limit:
                break
            _append_if_allowed(item)
    selected.sort(
        key=lambda item: (
            -float(item[0].screen_score),
            -float(item[0].mechanistic_prior),
            -float(item[0].consensus_prior),
            -float(item[0].residual_gain),
            float(item[0].complexity),
            str(item[0].name),
        )
    )
    required_gate = max(int(cfg.gate_candidate_screen_reserve), int(_required_gate_basis_terms(cfg)))
    if required_gate > 0:
        selected = _reserve_gate_candidates(
            ranked_rows=selected,
            candidate_limit=limit,
            reserve_count=required_gate,
        )
    required_periodic = max(
        int(cfg.periodic_candidate_screen_reserve),
        int(_required_periodic_basis_terms(cfg=cfg, periodic_context=periodic_context)),
    )
    if required_periodic > 0:
        selected = _reserve_periodic_candidates(
            ranked_rows=selected,
            full_ranked_rows=tuple(
                item
                for item in tuple(full_ranked_rows)
                if _candidate_proxy_assignment_compatible_with_assignments(
                    candidate=item[0],
                    current_assignments=assignments,
                    feature_names=feature_names,
                    interference_context=interference_context,
                )
                or int(item[0].pool_index) in selected_pool_indices
            ),
            reserve_count=required_periodic,
        )
    return list(tuple(selected)[:limit])


def _build_candidate_objects(
    *,
    screened: Sequence[ScreenedCandidate],
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
    periodic_context: Mapping[str, Any],
    outer_search_unit: str | None = None,
) -> tuple[CandidateObject, ...]:
    grouped_members: dict[str, list[ScreenedCandidate]] = {}
    metadata_by_key: dict[str, dict[str, Any]] = {}
    for row in tuple(screened):
        object_key = _candidate_object_key(
            candidate=row,
            feature_names=feature_names,
            interference_context=interference_context,
            periodic_context=periodic_context,
            outer_search_unit=outer_search_unit,
        )
        grouped_members.setdefault(str(object_key), []).append(row)
        metadata_by_key.setdefault(
            str(object_key),
            {
                "object_kind": _candidate_object_kind(
                    candidate=row,
                    feature_names=feature_names,
                    periodic_context=periodic_context,
                ),
                "source_key": _candidate_source_key(
                    candidate=row,
                    feature_names=feature_names,
                    interference_context=interference_context,
                ),
                "source_object_key": str(
                    row.source_object_key or row.information_source_key or _candidate_information_source_key(row)
                ),
                "feature_names": _candidate_feature_names_for_row(candidate=row, feature_names=feature_names),
                "proxy_group_ids": _proxy_group_ids_for_feature_names(
                    candidate_feature_names=_candidate_feature_names_for_row(
                        candidate=row,
                        feature_names=feature_names,
                    ),
                    interference_context=interference_context,
                ),
                "periodic_feature_names": _candidate_periodic_feature_names(
                    candidate=row,
                    feature_names=feature_names,
                    periodic_context=periodic_context,
                ),
                "selection_channel": str(row.selection_channel or "challenger"),
                "source_support_key": str(row.source_support_key or ""),
                "source_support_size": int(row.source_support_size),
                "chart_signatures": set(),
                "realization_head_signatures": set(),
                "support_expansion_tagged": bool(row.support_expansion_tagged),
                "canonical_trunk_tagged": bool(row.canonical_trunk_tagged),
                "same_source_surrogate_tagged": bool(row.same_source_surrogate_tagged),
                "support_expansion_candidate": bool(row.support_expansion_candidate),
                "canonical_trunk_candidate": bool(row.canonical_trunk_candidate),
                "same_source_surrogate_candidate": bool(row.same_source_surrogate_candidate),
                "global_uniform_candidate": bool(row.global_uniform_candidate),
                "modulated_branch_candidate": bool(row.modulated_branch_candidate),
                "structural_channel": str(row.structural_channel or "challenger"),
            },
        )
        metadata_by_key[str(object_key)]["chart_signatures"].add(str(row.chart_signature or "identity"))
        metadata_by_key[str(object_key)]["realization_head_signatures"].add(
            str(row.realization_head_signature or "")
        )
        metadata_by_key[str(object_key)]["source_support_size"] = max(
            int(metadata_by_key[str(object_key)].get("source_support_size", 0) or 0),
            int(row.source_support_size),
        )
        if not str(metadata_by_key[str(object_key)].get("source_support_key") or "").strip() and str(row.source_support_key).strip():
            metadata_by_key[str(object_key)]["source_support_key"] = str(row.source_support_key)
        metadata_by_key[str(object_key)]["support_expansion_tagged"] = bool(
            metadata_by_key[str(object_key)].get("support_expansion_tagged") or bool(row.support_expansion_tagged)
        )
        metadata_by_key[str(object_key)]["canonical_trunk_tagged"] = bool(
            metadata_by_key[str(object_key)].get("canonical_trunk_tagged") or bool(row.canonical_trunk_tagged)
        )
        metadata_by_key[str(object_key)]["same_source_surrogate_tagged"] = bool(
            metadata_by_key[str(object_key)].get("same_source_surrogate_tagged")
            or bool(row.same_source_surrogate_tagged)
        )
        metadata_by_key[str(object_key)]["support_expansion_candidate"] = bool(
            metadata_by_key[str(object_key)].get("support_expansion_candidate") or bool(row.support_expansion_candidate)
        )
        metadata_by_key[str(object_key)]["canonical_trunk_candidate"] = bool(
            metadata_by_key[str(object_key)].get("canonical_trunk_candidate") or bool(row.canonical_trunk_candidate)
        )
        metadata_by_key[str(object_key)]["same_source_surrogate_candidate"] = bool(
            metadata_by_key[str(object_key)].get("same_source_surrogate_candidate")
            or bool(row.same_source_surrogate_candidate)
        )
        metadata_by_key[str(object_key)]["global_uniform_candidate"] = bool(
            metadata_by_key[str(object_key)].get("global_uniform_candidate") or bool(row.global_uniform_candidate)
        )
        metadata_by_key[str(object_key)]["modulated_branch_candidate"] = bool(
            metadata_by_key[str(object_key)].get("modulated_branch_candidate") or bool(row.modulated_branch_candidate)
        )
    objects: list[CandidateObject] = []
    for object_key, members in grouped_members.items():
        ordered_members = tuple(
            sorted(
                tuple(members),
                key=lambda row: (
                    -float(row.screen_score),
                    -float(row.periodic_prior),
                    -float(row.residual_gain),
                    float(row.complexity),
                    str(row.name),
                ),
            )
        )
        meta = dict(metadata_by_key.get(str(object_key), {}) or {})
        objects.append(
            CandidateObject(
                object_key=str(object_key),
                object_kind=str(meta.get("object_kind") or "single_source_object"),
                source_key=str(meta.get("source_key") or object_key),
                feature_names=tuple(str(value) for value in tuple(meta.get("feature_names", ())) if str(value).strip()),
                proxy_group_ids=tuple(
                    str(value) for value in tuple(meta.get("proxy_group_ids", ())) if str(value).strip()
                ),
                periodic_feature_names=tuple(
                    str(value) for value in tuple(meta.get("periodic_feature_names", ())) if str(value).strip()
                ),
                members=ordered_members,
                source_object_key=str(meta.get("source_object_key") or object_key),
                source_support_key=str(meta.get("source_support_key") or ""),
                source_support_size=int(meta.get("source_support_size", 0) or 0),
                chart_signatures=tuple(
                    sorted(
                        str(value)
                        for value in tuple(meta.get("chart_signatures", ()))
                        if str(value).strip()
                    )
                ),
                realization_head_signatures=tuple(
                    sorted(
                        str(value)
                        for value in tuple(meta.get("realization_head_signatures", ()))
                        if str(value).strip()
                    )
                ),
                support_expansion_tagged=bool(meta.get("support_expansion_tagged")),
                canonical_trunk_tagged=bool(meta.get("canonical_trunk_tagged")),
                same_source_surrogate_tagged=bool(meta.get("same_source_surrogate_tagged")),
                support_expansion_candidate=bool(meta.get("support_expansion_candidate")),
                canonical_trunk_candidate=bool(meta.get("canonical_trunk_candidate")),
                same_source_surrogate_candidate=bool(meta.get("same_source_surrogate_candidate")),
                global_uniform_candidate=bool(meta.get("global_uniform_candidate")),
                modulated_branch_candidate=bool(meta.get("modulated_branch_candidate")),
                structural_channel=str(meta.get("structural_channel") or "challenger"),
                selection_channel=str(meta.get("selection_channel") or "challenger"),
            )
        )
    objects.sort(
        key=lambda item: (
            0 if bool(item.support_expansion_tagged) else 1,
            0 if bool(item.canonical_trunk_tagged) else 1,
            0 if bool(item.support_expansion_candidate) else 1,
            0 if bool(item.canonical_trunk_candidate) else 1,
            min(float(-member.screen_score) for member in item.members) if item.members else 0.0,
            min(float(-member.periodic_prior) for member in item.members) if item.members else 0.0,
            str(item.object_key),
        )
    )
    return tuple(objects)


def _source_overlap_summary(
    *,
    candidate_feature_names: Sequence[str],
    selected_feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
) -> dict[str, Any]:
    feature_name_to_index = dict(interference_context.get("feature_name_to_index", {}) or {})
    abs_corr = np.asarray(interference_context.get("raw_feature_abs_corr"), dtype=float)
    proxy_lookup = {
        str(key): tuple(str(item) for item in tuple(value))
        for key, value in dict(interference_context.get("proxy_group_lookup", {}) or {}).items()
    }
    candidate_names = tuple(str(name) for name in tuple(candidate_feature_names) if str(name).strip())
    selected_names = tuple(str(name) for name in tuple(selected_feature_names) if str(name).strip())
    max_source_abs_corr = 0.0
    shared_groups: set[str] = set()
    for left_name in candidate_names:
        for right_name in selected_names:
            left_idx = feature_name_to_index.get(str(left_name))
            right_idx = feature_name_to_index.get(str(right_name))
            if left_idx is not None and right_idx is not None and abs_corr.size:
                max_source_abs_corr = max(max_source_abs_corr, float(abs_corr[int(left_idx), int(right_idx)]))
            shared_groups |= set(proxy_lookup.get(str(left_name), ())) & set(proxy_lookup.get(str(right_name), ()))
    return {
        "candidate_feature_names": list(candidate_names),
        "selected_feature_names": list(selected_names),
        "max_source_abs_corr": float(max_source_abs_corr),
        "shared_proxy_groups": sorted(str(value) for value in shared_groups if str(value)),
    }


def _selected_rows_explainability_r2(
    *,
    candidate_values: np.ndarray,
    selected_rows: Sequence[ScreenedCandidate],
    train_matrix: np.ndarray,
) -> float:
    rows = tuple(selected_rows)
    if not rows:
        return 0.0
    selected_matrix = _selected_matrix(train_matrix, rows)
    if selected_matrix.size == 0:
        return 0.0
    projection = _ridge_projection(
        np.asarray(selected_matrix, dtype=float),
        np.asarray(candidate_values, dtype=float).reshape(-1),
        l2_value=1e-6,
    )
    return float(np.clip(projection.get("r2", 0.0), 0.0, 1.0))


def _cross_explanatory_summary(
    *,
    candidate: ScreenedCandidate,
    selected_rows: Sequence[ScreenedCandidate],
    train_matrix: np.ndarray,
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_feature_names = _feature_name_tuple(candidate.features, feature_names=feature_names)
    candidate_values = np.asarray(train_matrix[:, int(candidate.screen_index)], dtype=float).reshape(-1)
    explainability_r2 = _selected_rows_explainability_r2(
        candidate_values=candidate_values,
        selected_rows=selected_rows,
        train_matrix=train_matrix,
    )
    max_source_abs_corr = 0.0
    shared_proxy_groups: set[str] = set()
    selected_feature_name_rows: list[list[str]] = []
    for row in tuple(selected_rows):
        row_feature_names = _feature_name_tuple(row.features, feature_names=feature_names)
        selected_feature_name_rows.append(list(row_feature_names))
        overlap = _source_overlap_summary(
            candidate_feature_names=candidate_feature_names,
            selected_feature_names=row_feature_names,
            interference_context=interference_context,
        )
        max_source_abs_corr = max(max_source_abs_corr, float(overlap.get("max_source_abs_corr", 0.0) or 0.0))
        shared_proxy_groups |= {
            str(value) for value in tuple(overlap.get("shared_proxy_groups", ())) if str(value).strip()
        }
    suspicious = bool(
        explainability_r2 >= float(interference_context.get("explainability_threshold", _CROSS_EXPLANATORY_EXPLAINABILITY_THRESHOLD))
        and (
            bool(shared_proxy_groups)
            or max_source_abs_corr >= float(interference_context.get("source_corr_threshold", _CROSS_EXPLANATORY_SOURCE_CORR_THRESHOLD))
        )
    )
    return {
        "candidate_name": str(candidate.name),
        "candidate_feature_names": list(candidate_feature_names),
        "selected_feature_name_rows": selected_feature_name_rows,
        "max_source_abs_corr": float(max_source_abs_corr),
        "shared_proxy_groups": sorted(shared_proxy_groups),
        "explainability_r2": float(explainability_r2),
        "suspicious_overlap": bool(suspicious),
    }


def _cross_explanatory_rejection_enabled(cfg: OrthogonalBasisSearchConfig) -> bool:
    mode = str(cfg.cross_explanatory_rejection_mode or "off").strip().lower()
    return mode not in {"", "off", "none", "disabled"}


def _trivial_nonlinearity_penalty_enabled(cfg: OrthogonalBasisSearchConfig) -> bool:
    mode = str(cfg.trivial_nonlinearity_penalty_mode or "off").strip().lower()
    return mode not in {"", "off", "none", "disabled"}


def _trivial_nonlinearity_penalty_value(
    *,
    candidate: ScreenedCandidate,
    summary: Mapping[str, Any],
    cfg: OrthogonalBasisSearchConfig,
) -> float:
    if not _trivial_nonlinearity_penalty_enabled(cfg):
        return 0.0
    explainability_r2 = float(summary.get("explainability_r2", 0.0) or 0.0)
    max_source_abs_corr = float(summary.get("max_source_abs_corr", 0.0) or 0.0)
    shared_proxy_groups = [str(value) for value in tuple(summary.get("shared_proxy_groups", ())) if str(value).strip()]
    overlap_signal = 1.0 if shared_proxy_groups else float(
        np.clip((max_source_abs_corr - 0.90) / 0.10, 0.0, 1.0)
    )
    explainability_signal = float(np.clip((explainability_r2 - 0.75) / 0.25, 0.0, 1.0))
    if overlap_signal <= 0.0 or explainability_signal <= 0.0:
        return 0.0
    single_source_bonus = 1.0 if len(tuple(candidate.features)) <= 1 else 0.75
    return float(0.45 * overlap_signal * explainability_signal * single_source_bonus)


def _build_interference_feature_report(
    *,
    selected_rows: Sequence[ScreenedCandidate],
    train_matrix: np.ndarray,
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
    cfg: OrthogonalBasisSearchConfig,
) -> dict[str, Any]:
    rows = tuple(selected_rows)
    suspicious_pairs: list[dict[str, Any]] = []
    penalties: list[float] = []
    for index, candidate in enumerate(rows):
        previous_rows = tuple(rows[:index])
        if not previous_rows:
            continue
        summary = _cross_explanatory_summary(
            candidate=candidate,
            selected_rows=previous_rows,
            train_matrix=train_matrix,
            feature_names=feature_names,
            interference_context=interference_context,
        )
        penalty = _trivial_nonlinearity_penalty_value(candidate=candidate, summary=summary, cfg=cfg)
        penalties.append(float(penalty))
        if bool(summary.get("suspicious_overlap")) or float(penalty) > 0.0:
            suspicious_pairs.append(
                {
                    **dict(summary),
                    "term_name": str(candidate.name),
                    "semantic_family": str(candidate.semantic_family),
                    "trivial_nonlinearity_penalty": float(penalty),
                }
            )
    total_penalty = float(sum(penalties))
    mean_penalty = float(total_penalty / float(len(penalties))) if penalties else 0.0
    max_penalty = 0.0 if not penalties else float(max(penalties))
    return {
        "protocol": str(cfg.interference_feature_protocol),
        "mode": str(cfg.interference_feature_mode),
        "cross_explanatory_rejection_mode": str(cfg.cross_explanatory_rejection_mode),
        "trivial_nonlinearity_penalty_mode": str(cfg.trivial_nonlinearity_penalty_mode),
        "proxy_group_policy": str(cfg.proxy_group_policy),
        "source_overlap_penalty_mode": str(cfg.source_overlap_penalty_mode),
        "proxy_groups": [
            [str(name) for name in tuple(group)]
            for group in tuple(interference_context.get("proxy_groups", ()))
        ],
        "suspicious_pair_count": int(len(suspicious_pairs)),
        "suspicious_pairs": suspicious_pairs,
        "trivial_nonlinearity_penalty_total": float(total_penalty),
        "trivial_nonlinearity_penalty_mean": float(mean_penalty),
        "trivial_nonlinearity_penalty_max": float(max_penalty),
        "status": "reported",
    }


def _build_environment_invariance_audit(
    *,
    selected_rows: Sequence[ScreenedCandidate],
    train_matrix: np.ndarray,
    target: np.ndarray,
    raw_X: np.ndarray,
    feature_names: Sequence[str],
    gate_feature_names: Sequence[str],
    cfg: OrthogonalBasisSearchConfig,
) -> dict[str, Any]:
    mode = str(cfg.environment_invariance_audit_mode or "off").strip().lower()
    if mode in {"", "off", "none", "disabled"}:
        return {
            "protocol": "environment_invariance_audit_v1",
            "mode": str(cfg.environment_invariance_audit_mode),
            "status": "disabled",
        }
    names = tuple(str(value) for value in tuple(feature_names))
    raw = np.asarray(raw_X, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    if raw.ndim != 2 or raw.shape[0] != y.shape[0] or raw.shape[0] < 24:
        return {
            "protocol": "environment_invariance_audit_v1",
            "mode": str(cfg.environment_invariance_audit_mode),
            "status": "skipped",
            "reason": "insufficient_aligned_rows",
        }
    priority_features = [str(name) for name in tuple(gate_feature_names) if str(name) in set(names)]
    if not priority_features:
        raw_target_corr = [
            (
                float(abs(_safe_corr(raw[:, int(index)], y))),
                str(name),
            )
            for index, name in enumerate(names)
        ]
        raw_target_corr.sort(key=lambda item: (-float(item[0]), str(item[1])))
        priority_features = [str(item[1]) for item in raw_target_corr[:1] if str(item[1]).strip()]
    feature_to_index = {str(name): int(index) for index, name in enumerate(names)}
    environment_rows: list[dict[str, Any]] = []
    basis_matrix = _selected_matrix(train_matrix, selected_rows)
    if basis_matrix.size == 0:
        return {
            "protocol": "environment_invariance_audit_v1",
            "mode": str(cfg.environment_invariance_audit_mode),
            "status": "skipped",
            "reason": "empty_basis_matrix",
        }
    global_projection = _ridge_projection(
        basis_matrix,
        y,
        l2_value=float(min(tuple(cfg.l2_grid)) if tuple(cfg.l2_grid) else 1e-6),
    )
    global_weight = np.asarray(global_projection.get("weight"), dtype=float).reshape(-1)
    term_names = [str(row.name) for row in tuple(selected_rows)]
    for feature_name in priority_features[:2]:
        feature_index = feature_to_index.get(str(feature_name))
        if feature_index is None:
            continue
        column = np.asarray(raw[:, int(feature_index)], dtype=float).reshape(-1)
        finite_column = column[np.isfinite(column)]
        if finite_column.size < 24:
            continue
        cut = float(np.median(finite_column))
        low_mask = np.asarray(column <= cut, dtype=bool)
        high_mask = np.asarray(column > cut, dtype=bool)
        if int(np.sum(low_mask)) < 12 or int(np.sum(high_mask)) < 12:
            continue
        for env_label, mask in (("low", low_mask), ("high", high_mask)):
            projection = _ridge_projection(
                np.asarray(basis_matrix[mask], dtype=float),
                np.asarray(y[mask], dtype=float),
                l2_value=float(min(tuple(cfg.l2_grid)) if tuple(cfg.l2_grid) else 1e-6),
            )
            environment_rows.append(
                {
                    "feature_name": str(feature_name),
                    "split_label": str(env_label),
                    "cut_value": float(cut),
                    "sample_count": int(np.sum(mask)),
                    "r2": float(projection.get("r2", 0.0) or 0.0),
                    "coefficients": np.asarray(projection.get("weight"), dtype=float).reshape(-1).tolist(),
                }
            )
    if not environment_rows:
        return {
            "protocol": "environment_invariance_audit_v1",
            "mode": str(cfg.environment_invariance_audit_mode),
            "status": "skipped",
            "reason": "no_valid_environment_split",
        }
    term_reports: list[dict[str, Any]] = []
    overall_scores: list[float] = []
    for term_index, term_name in enumerate(term_names):
        coeffs = [
            float(row["coefficients"][term_index])
            for row in environment_rows
            if term_index < len(tuple(row.get("coefficients", ())))
        ]
        if not coeffs:
            continue
        coeff_arr = np.asarray(coeffs, dtype=float)
        global_abs = float(abs(global_weight[term_index])) if term_index < global_weight.shape[0] else 0.0
        denom = max(global_abs, 1e-6)
        coeff_cv = float(np.std(coeff_arr, ddof=0) / denom)
        invariance_score = float(np.clip(1.0 / (1.0 + coeff_cv), 0.0, 1.0))
        overall_scores.append(float(invariance_score))
        term_reports.append(
            {
                "term_name": str(term_name),
                "global_abs_weight": float(global_abs),
                "environment_weight_std": float(np.std(coeff_arr, ddof=0)),
                "environment_weight_cv": float(coeff_cv),
                "invariance_score": float(invariance_score),
            }
        )
    return {
        "protocol": "environment_invariance_audit_v1",
        "mode": str(cfg.environment_invariance_audit_mode),
        "status": "reported",
        "split_protocol": "gate_feature_median_split" if priority_features else "top_feature_median_split",
        "environment_count": int(len(environment_rows)),
        "global_r2": float(global_projection.get("r2", 0.0) or 0.0),
        "overall_invariance_score": float(np.mean(overall_scores)) if overall_scores else 0.0,
        "environments": environment_rows,
        "term_reports": term_reports,
    }


def _periodic_disambiguation_enabled(cfg: "OrthogonalBasisSearchConfig") -> bool:
    mode = str(cfg.periodic_equivalence_disambiguation_mode or "off").strip().lower()
    return mode not in {"", "off", "none", "disabled"}


def _periodic_family_prior_enabled(cfg: "OrthogonalBasisSearchConfig") -> bool:
    mode = str(cfg.periodic_family_prior_mode or "off").strip().lower()
    return mode not in {"", "off", "none", "disabled"}


def _phase_spectrum_audit_enabled(cfg: "OrthogonalBasisSearchConfig") -> bool:
    mode = str(cfg.phase_spectrum_audit_mode or "off").strip().lower()
    return mode not in {"", "off", "none", "disabled"}


def _regional_correction_enabled(cfg: "OrthogonalBasisSearchConfig") -> bool:
    promotion_mode = str(cfg.regional_correction_promotion_mode or "off").strip().lower()
    basis_mode = str(cfg.regional_correction_basis_mode or "off").strip().lower()
    residual_mode = str(cfg.residual_regime_identification_mode or "off").strip().lower()
    return (
        int(cfg.regional_correction_topk) > 0
        and promotion_mode not in {"", "off", "none", "disabled"}
        and basis_mode not in {"", "off", "none", "disabled"}
        and residual_mode not in {"", "off", "none", "disabled"}
    )


def _expr_contains_unary_op(expr: Mapping[str, Any], ops: set[str]) -> bool:
    node = dict(expr)
    kind = str(node.get("type", "")).strip().lower()
    if kind == "unary":
        op = str(node.get("op", "")).strip().lower()
        if op in ops:
            return True
        return _expr_contains_unary_op(dict(node.get("arg", {})), ops)
    if kind == "binary":
        return _expr_contains_unary_op(dict(node.get("left", {})), ops) or _expr_contains_unary_op(
            dict(node.get("right", {})),
            ops,
        )
    return False


def _candidate_is_periodic_family(
    *,
    semantic_family: str,
    expr: Mapping[str, Any],
) -> bool:
    family = str(semantic_family or "").strip().lower()
    if "periodic" in family:
        return True
    return _expr_contains_unary_op(expr, {"sin", "cos"})


def _candidate_is_gate_family(
    *,
    semantic_family: str,
    uses_piecewise_gate: bool,
) -> bool:
    family = str(semantic_family or "").strip().lower()
    return bool(uses_piecewise_gate or "piecewise" in family or "gate" in family)


def _periodic_holdout_profile(
    *,
    candidate_values: np.ndarray,
    target: np.ndarray,
    center_mask: np.ndarray,
    edge_mask: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(candidate_values, dtype=float).reshape(-1)
    y = np.asarray(target, dtype=float).reshape(-1)
    if (
        values.shape != y.shape
        or values.size <= 0
        or int(np.sum(center_mask)) < _PERIODIC_AUDIT_MIN_SAMPLES
        or int(np.sum(edge_mask)) < _PERIODIC_AUDIT_MIN_SAMPLES
    ):
        return {
            "status": "skipped",
            "reason": "insufficient_center_edge_samples",
        }
    center_fit = _ridge_projection(values[center_mask], y[center_mask], l2_value=1e-6)
    weight = np.asarray(center_fit.get("weight"), dtype=float).reshape(-1)
    bias = float(center_fit.get("bias", 0.0) or 0.0)
    edge_pred = np.asarray(values[edge_mask], dtype=float).reshape(-1) * float(weight[0] if weight.size else 0.0) + bias
    edge_target = np.asarray(y[edge_mask], dtype=float).reshape(-1)
    centered_edge = edge_target - float(np.mean(edge_target))
    ss_tot_edge = float(np.dot(centered_edge, centered_edge))
    residual_edge = np.asarray(edge_target - edge_pred, dtype=float).reshape(-1)
    ss_res_edge = float(np.dot(residual_edge, residual_edge))
    edge_r2 = 0.0 if ss_tot_edge <= 1e-12 else float(1.0 - ss_res_edge / (ss_tot_edge + 1e-12))
    full_corr = float(abs(_safe_corr(values, y)))
    center_corr = float(abs(_safe_corr(values[center_mask], y[center_mask])))
    edge_corr = float(abs(_safe_corr(values[edge_mask], edge_target)))
    center_r2 = float(np.clip(center_fit.get("r2", 0.0), -1.0, 1.0))
    edge_r2_clipped = float(np.clip(edge_r2, -1.0, 1.0))
    generalization_gap = float(max(0.0, center_r2 - edge_r2_clipped))
    stability_score = float(np.clip(0.5 * (edge_corr + np.clip(edge_r2_clipped, 0.0, 1.0)), 0.0, 1.0))
    return {
        "status": "reported",
        "center_count": int(np.sum(center_mask)),
        "edge_count": int(np.sum(edge_mask)),
        "full_abs_corr": float(full_corr),
        "center_abs_corr": float(center_corr),
        "edge_abs_corr": float(edge_corr),
        "center_r2": float(center_r2),
        "edge_r2": float(edge_r2_clipped),
        "edge_rmse": float(_rmse(edge_target, edge_pred)),
        "generalization_gap": float(generalization_gap),
        "stability_score": float(stability_score),
    }


def _candidate_periodic_summary(
    *,
    name: str,
    expr: Mapping[str, Any],
    semantic_family: str,
    uses_piecewise_gate: bool,
    feature_indices: Sequence[int],
    candidate_values: np.ndarray,
    target: np.ndarray,
    feature_names: Sequence[str],
    periodic_context: Mapping[str, Any],
    cfg: "OrthogonalBasisSearchConfig",
) -> dict[str, Any]:
    periodic_feature_names = set(
        str(value) for value in tuple(periodic_context.get("periodic_feature_names", ())) if str(value).strip()
    )
    candidate_feature_names = _feature_name_tuple(feature_indices, feature_names=feature_names)
    touched_features = tuple(name for name in candidate_feature_names if name in periodic_feature_names)
    if not touched_features:
        return {
            "status": "not_applicable",
            "candidate_name": str(name),
            "touched_periodic_features": [],
            "periodic_prior": 0.0,
            "periodic_penalty": 0.0,
        }
    is_periodic_family = _candidate_is_periodic_family(semantic_family=semantic_family, expr=expr)
    is_gate_family = _candidate_is_gate_family(
        semantic_family=semantic_family,
        uses_piecewise_gate=uses_piecewise_gate,
    )
    feature_windows = dict(periodic_context.get("feature_windows", {}) or {})
    profiles: list[dict[str, Any]] = []
    prior_scores: list[float] = []
    penalties: list[float] = []
    for feature_name in touched_features:
        window = dict(feature_windows.get(str(feature_name), {}) or {})
        profile = _periodic_holdout_profile(
            candidate_values=np.asarray(candidate_values, dtype=float),
            target=np.asarray(target, dtype=float),
            center_mask=np.asarray(window.get("center_mask", np.zeros_like(candidate_values, dtype=bool)), dtype=bool),
            edge_mask=np.asarray(window.get("edge_mask", np.zeros_like(candidate_values, dtype=bool)), dtype=bool),
        )
        feature_report = {
            "feature_name": str(feature_name),
            **dict(profile),
        }
        profiles.append(feature_report)
        if str(profile.get("status")) != "reported":
            continue
        stability_score = float(profile.get("stability_score", 0.0) or 0.0)
        generalization_gap = float(profile.get("generalization_gap", 0.0) or 0.0)
        center_r2 = float(np.clip(profile.get("center_r2", 0.0), 0.0, 1.0))
        if is_periodic_family:
            prior_scores.append(float(np.clip(0.65 * stability_score + 0.35 * center_r2, 0.0, 1.0)))
        elif not is_gate_family and _periodic_disambiguation_enabled(cfg):
            penalties.append(float(np.clip(generalization_gap * max(center_r2, 0.25), 0.0, 1.0)))
    prior_value = float(np.mean(prior_scores)) if prior_scores else 0.0
    penalty_value = float(np.mean(penalties)) if penalties else 0.0
    if not _periodic_family_prior_enabled(cfg):
        prior_value = 0.0
    if not _periodic_disambiguation_enabled(cfg):
        penalty_value = 0.0
    return {
        "status": "reported",
        "candidate_name": str(name),
        "semantic_family": str(semantic_family),
        "touched_periodic_features": [str(value) for value in touched_features],
        "periodic_family": bool(is_periodic_family),
        "gate_family": bool(is_gate_family),
        "periodic_prior": float(prior_value),
        "periodic_penalty": float(penalty_value),
        "profiles": profiles,
    }


def _build_periodic_equivalence_report(
    *,
    selected_rows: Sequence["ScreenedCandidate"],
    train_matrix: np.ndarray,
    target: np.ndarray,
    feature_names: Sequence[str],
    periodic_context: Mapping[str, Any],
    cfg: "OrthogonalBasisSearchConfig",
) -> dict[str, Any]:
    periodic_feature_names = tuple(
        str(value) for value in tuple(periodic_context.get("periodic_feature_names", ())) if str(value).strip()
    )
    if not periodic_feature_names:
        return {
            "protocol": str(cfg.periodic_equivalence_protocol),
            "mode": str(cfg.periodic_equivalence_disambiguation_mode),
            "phase_spectrum_audit_mode": str(cfg.phase_spectrum_audit_mode),
            "status": "not_applicable",
            "reason": "no_periodic_features_configured",
        }
    rows = tuple(selected_rows)
    term_reports: list[dict[str, Any]] = []
    periodic_object_features: set[str] = set()
    gate_features: set[str] = set()
    periodic_scores: list[float] = []
    penalties: list[float] = []
    for row in rows:
        summary = _candidate_periodic_summary(
            name=str(row.name),
            expr=dict(row.expr),
            semantic_family=str(row.semantic_family),
            uses_piecewise_gate=bool(row.uses_piecewise_gate),
            feature_indices=tuple(row.features),
            candidate_values=np.asarray(train_matrix[:, int(row.screen_index)], dtype=float),
            target=np.asarray(target, dtype=float).reshape(-1),
            feature_names=feature_names,
            periodic_context=periodic_context,
            cfg=cfg,
        )
        if str(summary.get("status")) != "reported":
            continue
        touched = {
            str(value)
            for value in tuple(summary.get("touched_periodic_features", ()))
            if str(value).strip()
        }
        periodic_object_evidence = bool(row.contains_periodic_evidence) or bool(summary.get("periodic_family"))
        if periodic_object_evidence:
            periodic_object_features |= touched
        if bool(summary.get("gate_family")):
            gate_features |= touched
        periodic_scores.append(max(float(summary.get("periodic_prior", 0.0) or 0.0), float(row.periodic_prior)))
        effective_penalty = float(summary.get("periodic_penalty", 0.0) or 0.0)
        if bool(row.contains_periodic_evidence):
            effective_penalty = min(float(effective_penalty), float(row.periodic_penalty))
        else:
            effective_penalty = max(float(effective_penalty), float(row.periodic_penalty))
        penalties.append(float(effective_penalty))
        summary["periodic_object_evidence"] = bool(periodic_object_evidence)
        term_reports.append(_jsonable(summary))
    coverage_score = float(len(periodic_object_features)) / float(len(periodic_feature_names))
    gate_support_score = float(len(gate_features)) / float(len(periodic_feature_names))
    periodic_score_mean = float(np.mean(periodic_scores)) if periodic_scores else 0.0
    penalty_mean = float(np.mean(penalties)) if penalties else 0.0
    overall_score = float(
        np.clip(
            0.50 * coverage_score + 0.35 * periodic_score_mean + 0.15 * gate_support_score - 0.35 * penalty_mean,
            0.0,
            1.0,
        )
    )
    return {
        "protocol": str(cfg.periodic_equivalence_protocol),
        "mode": str(cfg.periodic_equivalence_disambiguation_mode),
        "phase_spectrum_audit_mode": str(cfg.phase_spectrum_audit_mode),
        "periodic_family_prior_mode": str(cfg.periodic_family_prior_mode),
        "status": "reported",
        "periodic_feature_names": [str(value) for value in periodic_feature_names],
        "periodic_feature_count": int(len(periodic_feature_names)),
        "covered_periodic_features": sorted(periodic_object_features),
        "covered_gate_features": sorted(gate_features),
        "periodic_family_term_count": int(sum(1 for row in term_reports if bool(row.get("periodic_object_evidence")))),
        "periodic_surrogate_term_count": int(
            sum(
                1
                for row in term_reports
                if not bool(row.get("periodic_object_evidence")) and not bool(row.get("gate_family"))
            )
        ),
        "coverage_score": float(coverage_score),
        "gate_support_score": float(gate_support_score),
        "periodic_family_score_mean": float(periodic_score_mean),
        "local_equivalence_penalty_mean": float(penalty_mean),
        "overall_periodic_disambiguation_score": float(overall_score),
        "term_reports": term_reports if _phase_spectrum_audit_enabled(cfg) else [],
    }


@dataclass(frozen=True)
class OrthogonalBasisSearchConfig:
    candidate_limit: int = 96
    seed_candidate_count: int = 18
    group_count: int = 12
    min_basis_count: int = 3
    max_basis_count: int = 6
    max_pair_abs_corr: float = 0.35
    max_feature_reuse: int = 2
    max_semantic_repeats: int = 1
    max_piecewise_semantic_repeats: int = 2
    target_score_weight: float = 1.0
    diversity_corr_weight: float = 0.80
    feature_overlap_penalty: float = 0.20
    complexity_penalty: float = 0.03
    new_feature_bonus: float = 0.05
    family_diversity_bonus: float = 0.03
    semantic_family_bonus: float = 0.05
    residual_corr_weight: float = 0.55
    residual_gain_weight: float = 0.85
    semantic_dup_penalty: float = 0.30
    piecewise_gate_bonus: float = 0.14
    native_structure_group_bonus: float = 0.0
    native_structure_representative_bonus: float = 0.0
    screen_target_corr_weight: float = 1.0
    screen_residual_gain_weight: float = 0.65
    screen_semantic_novelty_weight: float = 0.20
    screen_consensus_prior_weight: float = 0.40
    screen_complexity_penalty: float = 0.08
    native_structure_screen_bonus: float = 0.0
    native_trunk_boundary_protocol: str = "OutermostPeelingBoundaryLock"
    native_trunk_channel_mode: str = "outermost_peeling"
    native_trunk_candidate_screen_reserve: int = 2
    require_native_trunk_candidate_in_group: bool = True
    min_native_trunk_basis_terms: int = 1
    native_trunk_residual_gain_floor: float = 0.05
    native_trunk_interval_gain_floor: float = 0.005
    gate_candidate_screen_reserve: int = 0
    require_gate_candidate_in_group: bool = False
    min_gate_basis_terms: int = 0
    require_periodic_candidate_in_group: bool = False
    min_periodic_basis_terms: int = 0
    mechanistic_feature_groups: tuple[tuple[str, ...], ...] = tuple()
    mechanistic_screen_bonus: float = 0.0
    mechanistic_group_bonus: float = 0.0
    l2_grid: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1e-1)
    rolling_folds: int = 3
    rolling_val_ratio: float = 0.18
    min_train_ratio: float = 0.40
    interval_alpha: float = 0.20
    coverage_error_threshold: float = 0.08
    outer_search_beam_width: int = 12
    outer_search_branching_factor: int = 3
    outer_search_max_expansions: int = 96
    selection_mode: str = "interval_first"
    random_seed: int = 42
    greedy_choice_topk: int = 1
    random_group_trials: int = 0
    outer_search_unit: str = "mechanism_object"
    representative_selection_rule: str = "balanced"
    lock_seed_basis: bool = False
    enable_piecewise_basis: bool = True
    gate_feature_names: tuple[str, ...] = tuple()
    periodic_feature_names: tuple[str, ...] = tuple()
    gate_quantiles: tuple[float, ...] = (0.35, 0.50, 0.65)
    gate_families: tuple[str, ...] = ("gate_step", "piecewise_hinge", "piecewise")
    gate_slope: float = 8.0
    piecewise_left_mode: str = "identity"
    piecewise_right_mode: str = "relu"
    assembler_max_added_terms: int = 4
    assembler_topk_features: int = 4
    assembler_max_pair_terms: int = 8
    assembler_max_candidates_per_iter: int = 96
    assembler_candidate_keep_top: int = 6
    assembler_max_expr_depth: int = 6
    assembler_ridge_l2: float = 1e-4
    assembler_path_memory_enabled: bool = False
    assembler_graph_cache_enabled: bool = False
    assembler_hinge_quantiles: tuple[float, ...] = (0.25, 0.50, 0.75)
    assembler_basis_binding_mode: str = "defining"
    assembler_escape_policy: str = "forbid"
    assembler_escape_feature_names: tuple[str, ...] = tuple()
    equivalence_expression_protocol: str = "EquivalenceExpressionHandlingProtocol"
    equivalence_expression_mode: str = "family+phase_equivalent+semantic"
    equivalence_class_scope: str = "candidate_screen+consensus+truth_recovery"
    chart_canonicalization_protocol: str = "ChartCanonicalizationPriority"
    chart_canonicalization_mode: str = "canonical_identity_with_stability_guard"
    chart_orthodoxy_scoring_protocol: str = "ChartOrthodoxyScoring"
    chart_orthodoxy_scoring_mode: str = "safe_wrapper_penalty+pole_safety+cross_interval_stability"
    support_expansion_protection_protocol: str = "SupportExpansionProtection"
    support_expansion_protection_mode: str = "full_support_native_template+seat_guard"
    support_expansion_candidate_screen_reserve: int = 1
    require_support_expansion_candidate_in_group: bool = True
    min_support_expansion_basis_terms: int = 1
    canonical_trunk_lane_protocol: str = "CanonicalTrunkLane"
    canonical_trunk_lane_mode: str = "support_pool_exposure+seat_guard"
    canonical_trunk_candidate_screen_reserve: int = 1
    require_canonical_trunk_candidate_in_group: bool = True
    min_canonical_trunk_basis_terms: int = 1
    same_source_surrogate_lane_protocol: str = "SameSourceSurrogateLane"
    same_source_surrogate_lane_mode: str = "support_pool_open_lane"
    rational_template_pinning_protocol: str = "RationalTemplatePinning"
    rational_template_pinning_mode: str = "mechanistic_pair_canonical_ratio_injection"
    global_first_preemption_protocol: str = "GlobalFirstPreemption"
    global_first_preemption_mode: str = "plain_support_parent_first"
    inner_chart_flip_compensation_protocol: str = "InnerChartFlipCompensation"
    inner_chart_flip_compensation_mode: str = "same_source_reciprocal_competition"
    realization_prior_injection_protocol: str = "RealizationPriorInjection"
    realization_prior_injection_mode: str = "object_member_evidence"
    mandatory_realization_closure_protocol: str = "MandatoryRealizationClosure"
    mandatory_realization_closure_mode: str = "explicit_evidence_competition"
    same_source_over_realization_protocol: str = "SameSourceOverRealizationCollapse"
    same_source_over_realization_mode: str = "inner_basis_object_budget"
    same_source_realization_budget: int = 1
    periodic_realization_competition_protocol: str = "PeriodicRealizationCompetition"
    periodic_realization_competition_mode: str = "sin_cos_basis_competition"
    interference_feature_protocol: str = "InterferenceFeatureHandlingProtocol"
    interference_feature_mode: str = "feature_overlap+semantic_dedup+mechanistic_bias"
    regime_penetration_protocol: str = "RegimePenetrationScore"
    regime_penetration_mode: str = "feature_quantile_penetration"
    regime_penetration_gain_floor: float = 0.01
    heterogeneous_exposure_protocol: str = "HeterogeneousExposureLane"
    heterogeneous_exposure_mode: str = "screen_reserve+seed_lane"
    heterogeneous_exposure_candidate_screen_reserve: int = 1
    heterogeneous_exposure_min_score: float = 0.20
    native_proxy_check_protocol: str = "NativeProxyCheck"
    native_proxy_check_mode: str = "proxy_group_native_election"
    proxy_trunk_disqualification_protocol: str = "ProxyTrunkDisqualification"
    proxy_trunk_disqualification_mode: str = "native_identity_only_when_available"
    parasitic_rejection_protocol: str = "ParasiticRejectionCriteria"
    parasitic_rejection_mode: str = "parent_trunk_required_for_branch_entry"
    causal_hierarchy_reuse_isolation_protocol: str = "CausalHierarchyReuseIsolation"
    causal_hierarchy_reuse_isolation_mode: str = "branch_free_with_parent"
    cross_explanatory_rejection_mode: str = "off"
    trivial_nonlinearity_penalty_mode: str = "heuristic_semantic_overlap"
    environment_invariance_audit_mode: str = "off"
    periodic_equivalence_protocol: str = "PeriodicEquivalenceDisambiguationMechanism"
    periodic_equivalence_disambiguation_mode: str = "off"
    phase_spectrum_audit_mode: str = "off"
    periodic_family_prior_mode: str = "off"
    periodic_family_prior_weight: float = 0.30
    periodic_candidate_screen_reserve: int = 0
    regional_correction_protocol: str = "RegionalCorrectionBasisProtocol"
    residual_regime_identification_mode: str = "off"
    regional_correction_basis_mode: str = "off"
    regional_correction_promotion_mode: str = "off"
    regional_correction_feature_scope: str = "gate_only"
    regional_correction_topk: int = 0
    regional_correction_min_r2_gain: float = 0.0
    regional_correction_search_mode: str = "reopened_local_object_search"
    regional_local_search_beam_width: int = 6
    regional_local_search_branching_factor: int = 2
    regional_local_search_max_expansions: int = 24
    proxy_group_policy: str = "hint_if_available"
    source_overlap_penalty_mode: str = "feature_overlap_penalty"

    def normalized(self) -> "OrthogonalBasisSearchConfig":
        l2_grid = tuple(sorted(float(max(0.0, value)) for value in self.l2_grid))
        selection_mode = str(self.selection_mode or "interval_first").strip().lower()
        if selection_mode not in {"interval_first", "orthogonal_first", "rmse_first"}:
            raise ValueError("selection_mode must be interval_first | orthogonal_first | rmse_first")
        gate_quantiles = tuple(
            sorted(
                float(np.clip(value, 0.05, 0.95))
                for value in tuple(self.gate_quantiles)
                if np.isfinite(float(value))
            )
        )
        assembler_hinge_quantiles = tuple(
            sorted(
                float(np.clip(value, 0.05, 0.95))
                for value in tuple(self.assembler_hinge_quantiles)
                if np.isfinite(float(value))
            )
        )
        gate_feature_names = tuple(str(value).strip() for value in tuple(self.gate_feature_names) if str(value).strip())
        periodic_feature_names = tuple(
            str(value).strip() for value in tuple(self.periodic_feature_names) if str(value).strip()
        )
        gate_families = tuple(
            str(value).strip().lower()
            for value in tuple(self.gate_families)
            if str(value).strip().lower() in {"gate_step", "gate_soft", "piecewise_hinge", "piecewise"}
        )
        mechanistic_feature_groups = tuple(
            tuple(dict.fromkeys(str(name).strip() for name in tuple(group) if str(name).strip()))
            for group in tuple(self.mechanistic_feature_groups)
            if isinstance(group, (list, tuple))
        )
        mechanistic_feature_groups = tuple(group for group in mechanistic_feature_groups if len(group) >= 2)
        assembler_basis_binding_mode = str(self.assembler_basis_binding_mode or "defining").strip().lower()
        if assembler_basis_binding_mode not in SYMBOLIC_BASIS_BINDING_MODES:
            raise ValueError(
                "assembler_basis_binding_mode must be one of "
                f"{SYMBOLIC_BASIS_BINDING_MODES}, got '{self.assembler_basis_binding_mode}'"
            )
        assembler_escape_policy = str(self.assembler_escape_policy or "forbid").strip().lower()
        if assembler_escape_policy not in SYMBOLIC_ESCAPE_POLICIES:
            raise ValueError(
                "assembler_escape_policy must be one of "
                f"{SYMBOLIC_ESCAPE_POLICIES}, got '{self.assembler_escape_policy}'"
            )
        assembler_escape_feature_names = tuple(
            str(value).strip() for value in tuple(self.assembler_escape_feature_names) if str(value).strip()
        )
        equivalence_expression_protocol = (
            str(self.equivalence_expression_protocol or "EquivalenceExpressionHandlingProtocol").strip()
            or "EquivalenceExpressionHandlingProtocol"
        )
        equivalence_expression_mode = (
            str(self.equivalence_expression_mode or "family+phase_equivalent+semantic").strip()
            or "family+phase_equivalent+semantic"
        )
        equivalence_class_scope = (
            str(self.equivalence_class_scope or "candidate_screen+consensus+truth_recovery").strip()
            or "candidate_screen+consensus+truth_recovery"
        )
        chart_canonicalization_protocol = (
            str(self.chart_canonicalization_protocol or "ChartCanonicalizationPriority").strip()
            or "ChartCanonicalizationPriority"
        )
        chart_canonicalization_mode = (
            str(self.chart_canonicalization_mode or "canonical_identity_with_stability_guard").strip().lower()
            or "canonical_identity_with_stability_guard"
        )
        chart_orthodoxy_scoring_protocol = (
            str(self.chart_orthodoxy_scoring_protocol or "ChartOrthodoxyScoring").strip()
            or "ChartOrthodoxyScoring"
        )
        chart_orthodoxy_scoring_mode = (
            str(self.chart_orthodoxy_scoring_mode or "safe_wrapper_penalty+pole_safety+cross_interval_stability")
            .strip()
            .lower()
            or "safe_wrapper_penalty+pole_safety+cross_interval_stability"
        )
        support_expansion_protection_protocol = (
            str(self.support_expansion_protection_protocol or "SupportExpansionProtection").strip()
            or "SupportExpansionProtection"
        )
        support_expansion_protection_mode = (
            str(self.support_expansion_protection_mode or "full_support_native_template+seat_guard").strip().lower()
            or "full_support_native_template+seat_guard"
        )
        support_expansion_candidate_screen_reserve = int(max(0, self.support_expansion_candidate_screen_reserve))
        require_support_expansion_candidate_in_group = bool(self.require_support_expansion_candidate_in_group)
        min_support_expansion_basis_terms = int(max(0, self.min_support_expansion_basis_terms))
        canonical_trunk_lane_protocol = (
            str(self.canonical_trunk_lane_protocol or "CanonicalTrunkLane").strip()
            or "CanonicalTrunkLane"
        )
        canonical_trunk_lane_mode = (
            str(self.canonical_trunk_lane_mode or "support_pool_exposure+seat_guard").strip().lower()
            or "support_pool_exposure+seat_guard"
        )
        canonical_trunk_candidate_screen_reserve = int(max(0, self.canonical_trunk_candidate_screen_reserve))
        require_canonical_trunk_candidate_in_group = bool(self.require_canonical_trunk_candidate_in_group)
        min_canonical_trunk_basis_terms = int(max(0, self.min_canonical_trunk_basis_terms))
        same_source_surrogate_lane_protocol = (
            str(self.same_source_surrogate_lane_protocol or "SameSourceSurrogateLane").strip()
            or "SameSourceSurrogateLane"
        )
        same_source_surrogate_lane_mode = (
            str(self.same_source_surrogate_lane_mode or "support_pool_open_lane").strip().lower()
            or "support_pool_open_lane"
        )
        rational_template_pinning_protocol = (
            str(self.rational_template_pinning_protocol or "RationalTemplatePinning").strip()
            or "RationalTemplatePinning"
        )
        rational_template_pinning_mode = (
            str(self.rational_template_pinning_mode or "mechanistic_pair_canonical_ratio_injection").strip().lower()
            or "mechanistic_pair_canonical_ratio_injection"
        )
        global_first_preemption_protocol = (
            str(self.global_first_preemption_protocol or "GlobalFirstPreemption").strip()
            or "GlobalFirstPreemption"
        )
        global_first_preemption_mode = (
            str(self.global_first_preemption_mode or "plain_support_parent_first").strip().lower()
            or "plain_support_parent_first"
        )
        inner_chart_flip_compensation_protocol = (
            str(self.inner_chart_flip_compensation_protocol or "InnerChartFlipCompensation").strip()
            or "InnerChartFlipCompensation"
        )
        inner_chart_flip_compensation_mode = (
            str(self.inner_chart_flip_compensation_mode or "same_source_reciprocal_competition").strip().lower()
            or "same_source_reciprocal_competition"
        )
        realization_prior_injection_protocol = (
            str(self.realization_prior_injection_protocol or "RealizationPriorInjection").strip()
            or "RealizationPriorInjection"
        )
        realization_prior_injection_mode = (
            str(self.realization_prior_injection_mode or "object_member_evidence").strip().lower()
            or "object_member_evidence"
        )
        mandatory_realization_closure_protocol = (
            str(self.mandatory_realization_closure_protocol or "MandatoryRealizationClosure").strip()
            or "MandatoryRealizationClosure"
        )
        mandatory_realization_closure_mode = (
            str(self.mandatory_realization_closure_mode or "explicit_evidence_competition").strip().lower()
            or "explicit_evidence_competition"
        )
        same_source_over_realization_protocol = (
            str(self.same_source_over_realization_protocol or "SameSourceOverRealizationCollapse").strip()
            or "SameSourceOverRealizationCollapse"
        )
        same_source_over_realization_mode = (
            str(self.same_source_over_realization_mode or "inner_basis_object_budget").strip().lower()
            or "inner_basis_object_budget"
        )
        same_source_realization_budget = int(max(1, self.same_source_realization_budget))
        periodic_realization_competition_protocol = (
            str(self.periodic_realization_competition_protocol or "PeriodicRealizationCompetition").strip()
            or "PeriodicRealizationCompetition"
        )
        periodic_realization_competition_mode = (
            str(self.periodic_realization_competition_mode or "sin_cos_basis_competition").strip().lower()
            or "sin_cos_basis_competition"
        )
        interference_feature_protocol = (
            str(self.interference_feature_protocol or "InterferenceFeatureHandlingProtocol").strip()
            or "InterferenceFeatureHandlingProtocol"
        )
        interference_feature_mode = (
            str(self.interference_feature_mode or "feature_overlap+semantic_dedup+mechanistic_bias").strip()
            or "feature_overlap+semantic_dedup+mechanistic_bias"
        )
        regime_penetration_protocol = (
            str(self.regime_penetration_protocol or "RegimePenetrationScore").strip()
            or "RegimePenetrationScore"
        )
        regime_penetration_mode = (
            str(self.regime_penetration_mode or "feature_quantile_penetration").strip().lower()
            or "feature_quantile_penetration"
        )
        regime_penetration_gain_floor = float(max(0.0, self.regime_penetration_gain_floor))
        heterogeneous_exposure_protocol = (
            str(self.heterogeneous_exposure_protocol or "HeterogeneousExposureLane").strip()
            or "HeterogeneousExposureLane"
        )
        heterogeneous_exposure_mode = (
            str(self.heterogeneous_exposure_mode or "screen_reserve+seed_lane").strip().lower()
            or "screen_reserve+seed_lane"
        )
        heterogeneous_exposure_candidate_screen_reserve = int(max(0, self.heterogeneous_exposure_candidate_screen_reserve))
        heterogeneous_exposure_min_score = float(np.clip(float(self.heterogeneous_exposure_min_score), 0.0, 1.0))
        native_proxy_check_protocol = (
            str(self.native_proxy_check_protocol or "NativeProxyCheck").strip()
            or "NativeProxyCheck"
        )
        native_proxy_check_mode = (
            str(self.native_proxy_check_mode or "proxy_group_native_election").strip().lower()
            or "proxy_group_native_election"
        )
        proxy_trunk_disqualification_protocol = (
            str(self.proxy_trunk_disqualification_protocol or "ProxyTrunkDisqualification").strip()
            or "ProxyTrunkDisqualification"
        )
        proxy_trunk_disqualification_mode = (
            str(self.proxy_trunk_disqualification_mode or "native_identity_only_when_available").strip().lower()
            or "native_identity_only_when_available"
        )
        parasitic_rejection_protocol = (
            str(self.parasitic_rejection_protocol or "ParasiticRejectionCriteria").strip()
            or "ParasiticRejectionCriteria"
        )
        parasitic_rejection_mode = (
            str(self.parasitic_rejection_mode or "parent_trunk_required_for_branch_entry").strip().lower()
            or "parent_trunk_required_for_branch_entry"
        )
        causal_hierarchy_reuse_isolation_protocol = (
            str(self.causal_hierarchy_reuse_isolation_protocol or "CausalHierarchyReuseIsolation").strip()
            or "CausalHierarchyReuseIsolation"
        )
        causal_hierarchy_reuse_isolation_mode = (
            str(self.causal_hierarchy_reuse_isolation_mode or "branch_free_with_parent").strip().lower()
            or "branch_free_with_parent"
        )
        cross_explanatory_rejection_mode = (
            str(self.cross_explanatory_rejection_mode or "off").strip().lower() or "off"
        )
        trivial_nonlinearity_penalty_mode = (
            str(self.trivial_nonlinearity_penalty_mode or "heuristic_semantic_overlap").strip()
            or "heuristic_semantic_overlap"
        )
        environment_invariance_audit_mode = (
            str(self.environment_invariance_audit_mode or "off").strip().lower() or "off"
        )
        periodic_equivalence_protocol = (
            str(self.periodic_equivalence_protocol or "PeriodicEquivalenceDisambiguationMechanism").strip()
            or "PeriodicEquivalenceDisambiguationMechanism"
        )
        periodic_equivalence_disambiguation_mode = (
            str(self.periodic_equivalence_disambiguation_mode or "off").strip().lower() or "off"
        )
        phase_spectrum_audit_mode = str(self.phase_spectrum_audit_mode or "off").strip().lower() or "off"
        periodic_family_prior_mode = str(self.periodic_family_prior_mode or "off").strip().lower() or "off"
        regional_correction_protocol = (
            str(self.regional_correction_protocol or "RegionalCorrectionBasisProtocol").strip()
            or "RegionalCorrectionBasisProtocol"
        )
        residual_regime_identification_mode = (
            str(self.residual_regime_identification_mode or "off").strip().lower() or "off"
        )
        regional_correction_basis_mode = (
            str(self.regional_correction_basis_mode or "off").strip().lower() or "off"
        )
        regional_correction_promotion_mode = (
            str(self.regional_correction_promotion_mode or "off").strip().lower() or "off"
        )
        regional_correction_feature_scope = (
            str(self.regional_correction_feature_scope or "gate_only").strip().lower() or "gate_only"
        )
        proxy_group_policy = str(self.proxy_group_policy or "hint_if_available").strip() or "hint_if_available"
        source_overlap_penalty_mode = (
            str(self.source_overlap_penalty_mode or "feature_overlap_penalty").strip()
            or "feature_overlap_penalty"
        )
        return OrthogonalBasisSearchConfig(
            candidate_limit=int(max(8, self.candidate_limit)),
            seed_candidate_count=int(max(3, self.seed_candidate_count)),
            group_count=int(max(1, self.group_count)),
            min_basis_count=int(max(2, self.min_basis_count)),
            max_basis_count=int(max(max(2, self.min_basis_count), self.max_basis_count)),
            max_pair_abs_corr=float(np.clip(self.max_pair_abs_corr, 0.05, 0.98)),
            max_feature_reuse=int(max(1, self.max_feature_reuse)),
            max_semantic_repeats=int(max(1, self.max_semantic_repeats)),
            max_piecewise_semantic_repeats=int(max(1, self.max_piecewise_semantic_repeats)),
            target_score_weight=float(max(0.01, self.target_score_weight)),
            diversity_corr_weight=float(max(0.0, self.diversity_corr_weight)),
            feature_overlap_penalty=float(max(0.0, self.feature_overlap_penalty)),
            complexity_penalty=float(max(0.0, self.complexity_penalty)),
            new_feature_bonus=float(max(0.0, self.new_feature_bonus)),
            family_diversity_bonus=float(max(0.0, self.family_diversity_bonus)),
            semantic_family_bonus=float(max(0.0, self.semantic_family_bonus)),
            residual_corr_weight=float(max(0.0, self.residual_corr_weight)),
            residual_gain_weight=float(max(0.0, self.residual_gain_weight)),
            semantic_dup_penalty=float(max(0.0, self.semantic_dup_penalty)),
            piecewise_gate_bonus=float(max(0.0, self.piecewise_gate_bonus)),
            native_structure_group_bonus=float(max(0.0, self.native_structure_group_bonus)),
            native_structure_representative_bonus=float(max(0.0, self.native_structure_representative_bonus)),
            screen_target_corr_weight=float(max(0.0, self.screen_target_corr_weight)),
            screen_residual_gain_weight=float(max(0.0, self.screen_residual_gain_weight)),
            screen_semantic_novelty_weight=float(max(0.0, self.screen_semantic_novelty_weight)),
            screen_consensus_prior_weight=float(max(0.0, self.screen_consensus_prior_weight)),
            screen_complexity_penalty=float(max(0.0, self.screen_complexity_penalty)),
            native_structure_screen_bonus=float(max(0.0, self.native_structure_screen_bonus)),
            native_trunk_boundary_protocol=(
                str(self.native_trunk_boundary_protocol or "OutermostPeelingBoundaryLock").strip()
                or "OutermostPeelingBoundaryLock"
            ),
            native_trunk_channel_mode=(
                str(self.native_trunk_channel_mode or "outermost_peeling").strip().lower()
                or "outermost_peeling"
            ),
            native_trunk_candidate_screen_reserve=int(max(0, self.native_trunk_candidate_screen_reserve)),
            require_native_trunk_candidate_in_group=bool(self.require_native_trunk_candidate_in_group),
            min_native_trunk_basis_terms=int(max(0, self.min_native_trunk_basis_terms)),
            native_trunk_residual_gain_floor=float(max(0.0, self.native_trunk_residual_gain_floor)),
            native_trunk_interval_gain_floor=float(max(0.0, self.native_trunk_interval_gain_floor)),
            gate_candidate_screen_reserve=int(max(0, self.gate_candidate_screen_reserve)),
            require_gate_candidate_in_group=bool(self.require_gate_candidate_in_group),
            min_gate_basis_terms=int(max(0, self.min_gate_basis_terms)),
            require_periodic_candidate_in_group=bool(self.require_periodic_candidate_in_group),
            min_periodic_basis_terms=int(max(0, self.min_periodic_basis_terms)),
            mechanistic_feature_groups=mechanistic_feature_groups,
            mechanistic_screen_bonus=float(max(0.0, self.mechanistic_screen_bonus)),
            mechanistic_group_bonus=float(max(0.0, self.mechanistic_group_bonus)),
            l2_grid=l2_grid or (1e-6, 1e-4, 1e-2, 1e-1),
            rolling_folds=int(max(1, self.rolling_folds)),
            rolling_val_ratio=float(np.clip(self.rolling_val_ratio, 0.05, 0.40)),
            min_train_ratio=float(np.clip(self.min_train_ratio, 0.10, 0.80)),
            interval_alpha=float(np.clip(self.interval_alpha, 1e-6, 0.99)),
            coverage_error_threshold=float(max(0.0, self.coverage_error_threshold)),
            outer_search_beam_width=int(max(2, self.outer_search_beam_width)),
            outer_search_branching_factor=int(max(1, self.outer_search_branching_factor)),
            outer_search_max_expansions=int(max(8, self.outer_search_max_expansions)),
            selection_mode=selection_mode,
            random_seed=int(self.random_seed),
            greedy_choice_topk=int(max(1, self.greedy_choice_topk)),
            random_group_trials=int(max(0, self.random_group_trials)),
            outer_search_unit=_normalized_outer_search_unit_name(self.outer_search_unit),
            representative_selection_rule=(
                str(self.representative_selection_rule or "balanced").strip().lower() or "balanced"
            ),
            lock_seed_basis=bool(self.lock_seed_basis),
            enable_piecewise_basis=bool(self.enable_piecewise_basis),
            gate_feature_names=gate_feature_names,
            periodic_feature_names=periodic_feature_names,
            gate_quantiles=gate_quantiles or (0.35, 0.50, 0.65),
            gate_families=gate_families or ("gate_step", "piecewise_hinge", "piecewise"),
            gate_slope=float(max(1.0, self.gate_slope)),
            piecewise_left_mode=str(self.piecewise_left_mode or "identity").strip().lower() or "identity",
            piecewise_right_mode=str(self.piecewise_right_mode or "relu").strip().lower() or "relu",
            assembler_max_added_terms=int(max(1, self.assembler_max_added_terms)),
            assembler_topk_features=int(max(1, self.assembler_topk_features)),
            assembler_max_pair_terms=int(max(0, self.assembler_max_pair_terms)),
            assembler_max_candidates_per_iter=int(max(8, self.assembler_max_candidates_per_iter)),
            assembler_candidate_keep_top=int(max(1, self.assembler_candidate_keep_top)),
            assembler_max_expr_depth=int(max(2, self.assembler_max_expr_depth)),
            assembler_ridge_l2=float(max(0.0, self.assembler_ridge_l2)),
            assembler_path_memory_enabled=bool(self.assembler_path_memory_enabled),
            assembler_graph_cache_enabled=bool(self.assembler_graph_cache_enabled),
            assembler_hinge_quantiles=assembler_hinge_quantiles or (0.25, 0.50, 0.75),
            assembler_basis_binding_mode=assembler_basis_binding_mode,
            assembler_escape_policy=assembler_escape_policy,
            assembler_escape_feature_names=assembler_escape_feature_names,
            equivalence_expression_protocol=equivalence_expression_protocol,
            equivalence_expression_mode=equivalence_expression_mode,
            equivalence_class_scope=equivalence_class_scope,
            chart_canonicalization_protocol=chart_canonicalization_protocol,
            chart_canonicalization_mode=chart_canonicalization_mode,
            chart_orthodoxy_scoring_protocol=chart_orthodoxy_scoring_protocol,
            chart_orthodoxy_scoring_mode=chart_orthodoxy_scoring_mode,
            support_expansion_protection_protocol=support_expansion_protection_protocol,
            support_expansion_protection_mode=support_expansion_protection_mode,
            support_expansion_candidate_screen_reserve=support_expansion_candidate_screen_reserve,
            require_support_expansion_candidate_in_group=require_support_expansion_candidate_in_group,
            min_support_expansion_basis_terms=min_support_expansion_basis_terms,
            canonical_trunk_lane_protocol=canonical_trunk_lane_protocol,
            canonical_trunk_lane_mode=canonical_trunk_lane_mode,
            canonical_trunk_candidate_screen_reserve=canonical_trunk_candidate_screen_reserve,
            require_canonical_trunk_candidate_in_group=require_canonical_trunk_candidate_in_group,
            min_canonical_trunk_basis_terms=min_canonical_trunk_basis_terms,
            same_source_surrogate_lane_protocol=same_source_surrogate_lane_protocol,
            same_source_surrogate_lane_mode=same_source_surrogate_lane_mode,
            rational_template_pinning_protocol=rational_template_pinning_protocol,
            rational_template_pinning_mode=rational_template_pinning_mode,
            global_first_preemption_protocol=global_first_preemption_protocol,
            global_first_preemption_mode=global_first_preemption_mode,
            inner_chart_flip_compensation_protocol=inner_chart_flip_compensation_protocol,
            inner_chart_flip_compensation_mode=inner_chart_flip_compensation_mode,
            realization_prior_injection_protocol=realization_prior_injection_protocol,
            realization_prior_injection_mode=realization_prior_injection_mode,
            mandatory_realization_closure_protocol=mandatory_realization_closure_protocol,
            mandatory_realization_closure_mode=mandatory_realization_closure_mode,
            same_source_over_realization_protocol=same_source_over_realization_protocol,
            same_source_over_realization_mode=same_source_over_realization_mode,
            same_source_realization_budget=same_source_realization_budget,
            periodic_realization_competition_protocol=periodic_realization_competition_protocol,
            periodic_realization_competition_mode=periodic_realization_competition_mode,
            interference_feature_protocol=interference_feature_protocol,
            interference_feature_mode=interference_feature_mode,
            regime_penetration_protocol=regime_penetration_protocol,
            regime_penetration_mode=regime_penetration_mode,
            regime_penetration_gain_floor=regime_penetration_gain_floor,
            heterogeneous_exposure_protocol=heterogeneous_exposure_protocol,
            heterogeneous_exposure_mode=heterogeneous_exposure_mode,
            heterogeneous_exposure_candidate_screen_reserve=heterogeneous_exposure_candidate_screen_reserve,
            heterogeneous_exposure_min_score=heterogeneous_exposure_min_score,
            native_proxy_check_protocol=native_proxy_check_protocol,
            native_proxy_check_mode=native_proxy_check_mode,
            proxy_trunk_disqualification_protocol=proxy_trunk_disqualification_protocol,
            proxy_trunk_disqualification_mode=proxy_trunk_disqualification_mode,
            parasitic_rejection_protocol=parasitic_rejection_protocol,
            parasitic_rejection_mode=parasitic_rejection_mode,
            causal_hierarchy_reuse_isolation_protocol=causal_hierarchy_reuse_isolation_protocol,
            causal_hierarchy_reuse_isolation_mode=causal_hierarchy_reuse_isolation_mode,
            cross_explanatory_rejection_mode=cross_explanatory_rejection_mode,
            trivial_nonlinearity_penalty_mode=trivial_nonlinearity_penalty_mode,
            environment_invariance_audit_mode=environment_invariance_audit_mode,
            periodic_equivalence_protocol=periodic_equivalence_protocol,
            periodic_equivalence_disambiguation_mode=periodic_equivalence_disambiguation_mode,
            phase_spectrum_audit_mode=phase_spectrum_audit_mode,
            periodic_family_prior_mode=periodic_family_prior_mode,
            periodic_family_prior_weight=float(max(0.0, self.periodic_family_prior_weight)),
            periodic_candidate_screen_reserve=int(max(0, self.periodic_candidate_screen_reserve)),
            regional_correction_protocol=regional_correction_protocol,
            residual_regime_identification_mode=residual_regime_identification_mode,
            regional_correction_basis_mode=regional_correction_basis_mode,
            regional_correction_promotion_mode=regional_correction_promotion_mode,
            regional_correction_feature_scope=regional_correction_feature_scope,
            regional_correction_topk=int(max(0, self.regional_correction_topk)),
            regional_correction_min_r2_gain=float(max(0.0, self.regional_correction_min_r2_gain)),
            regional_correction_search_mode=(
                str(self.regional_correction_search_mode or "reopened_local_object_search").strip().lower()
                or "reopened_local_object_search"
            ),
            regional_local_search_beam_width=int(max(1, self.regional_local_search_beam_width)),
            regional_local_search_branching_factor=int(max(1, self.regional_local_search_branching_factor)),
            regional_local_search_max_expansions=int(max(1, self.regional_local_search_max_expansions)),
            proxy_group_policy=proxy_group_policy,
            source_overlap_penalty_mode=source_overlap_penalty_mode,
        )


@dataclass(frozen=True)
class ScreenedCandidate:
    pool_index: int
    screen_index: int
    name: str
    expr: dict[str, Any]
    family: str
    complexity: float
    features: tuple[int, ...]
    target_corr: float
    screen_score: float
    expression: str
    semantic_signature: str
    semantic_family: str
    uses_piecewise_gate: bool
    residual_gain: float = 0.0
    semantic_novelty: float = 0.0
    consensus_prior: float = 0.0
    mechanistic_prior: float = 0.0
    periodic_prior: float = 0.0
    periodic_penalty: float = 0.0
    screen_cluster_key: str = ""
    screen_cluster_size: int = 1
    information_source_key: str = ""
    contains_periodic_evidence: bool = False
    source_object_key: str = ""
    source_support_key: str = ""
    source_support_size: int = 0
    chart_signature: str = "identity"
    realization_head_signature: str = ""
    chart_metadata: Mapping[str, Any] = field(default_factory=dict)
    native_structure_score: float = 0.0
    native_trunk_root: bool = False
    native_trunk_floor_passed: bool = False
    native_trunk_global_gain: float = 0.0
    native_trunk_interval_min_gain: float = 0.0
    native_trunk_interval_mean_gain: float = 0.0
    regime_penetration_score: float = 0.0
    regime_penetration_min_gain: float = 0.0
    regime_penetration_mean_gain: float = 0.0
    regime_sign_consistency: float = 0.0
    heterogeneous_exposure_eligible: bool = False
    support_expansion_tagged: bool = False
    canonical_trunk_tagged: bool = False
    same_source_surrogate_tagged: bool = False
    support_expansion_candidate: bool = False
    canonical_trunk_candidate: bool = False
    same_source_surrogate_candidate: bool = False
    global_uniform_candidate: bool = False
    modulated_branch_candidate: bool = False
    rational_template_pinned: bool = False
    structural_channel: str = "challenger"
    selection_channel: str = "challenger"


@dataclass(frozen=True)
class CandidateObject:
    object_key: str
    object_kind: str
    source_key: str
    feature_names: tuple[str, ...]
    proxy_group_ids: tuple[str, ...]
    periodic_feature_names: tuple[str, ...]
    members: tuple[ScreenedCandidate, ...]
    source_object_key: str = ""
    source_support_key: str = ""
    source_support_size: int = 0
    chart_signatures: tuple[str, ...] = tuple()
    realization_head_signatures: tuple[str, ...] = tuple()
    support_expansion_tagged: bool = False
    canonical_trunk_tagged: bool = False
    same_source_surrogate_tagged: bool = False
    support_expansion_candidate: bool = False
    canonical_trunk_candidate: bool = False
    same_source_surrogate_candidate: bool = False
    global_uniform_candidate: bool = False
    modulated_branch_candidate: bool = False
    structural_channel: str = "challenger"
    selection_channel: str = "challenger"


def _screen_information_scope_key(
    *,
    candidate: ScreenedCandidate,
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
    periodic_context: Mapping[str, Any],
) -> str:
    if _candidate_is_structural_gate(candidate):
        return f"gate::{_candidate_expr_key(_normalized_expr_tree(dict(candidate.expr)))}"
    proxy_signature = _candidate_proxy_group_signature(
        candidate=candidate,
        feature_names=feature_names,
        interference_context=interference_context,
    )
    if proxy_signature:
        proxy_parts = [
            f"{str(group_id)}:{'+'.join(tuple(names))}"
            for group_id, names in sorted(proxy_signature.items(), key=lambda item: str(item[0]))
        ]
        return f"proxy::{'|'.join(proxy_parts)}"
    return f"source_object::{_candidate_information_source_key(candidate)}"


def _screen_cluster_representative_priority(candidate: ScreenedCandidate) -> tuple[Any, ...]:
    priority = float(
        1.10 * max(0.0, float(candidate.residual_gain))
        + 0.42 * max(0.0, float(candidate.target_corr))
        + 0.25 * max(0.0, float(candidate.mechanistic_prior))
        + 0.90 * max(0.0, float(candidate.native_structure_score))
        + 0.65 * max(0.0, float(candidate.regime_penetration_score))
        + 0.20 * max(0.0, float(candidate.regime_sign_consistency))
        + 0.20 * max(0.0, float(candidate.periodic_prior))
        + 0.10 * max(0.0, float(candidate.semantic_novelty))
        + 0.02 * max(0.0, float(candidate.consensus_prior))
        - 0.10 * max(0.0, float(candidate.periodic_penalty))
        - 0.04 * max(0.0, float(candidate.complexity))
    )
    return (
        -int(bool(candidate.native_trunk_floor_passed)),
        -float(priority),
        -float(candidate.regime_penetration_score),
        -float(candidate.regime_penetration_min_gain),
        -float(candidate.regime_sign_consistency),
        -float(candidate.residual_gain),
        -float(candidate.native_structure_score),
        -float(candidate.mechanistic_prior),
        -float(candidate.target_corr),
        -float(candidate.periodic_prior),
        float(candidate.complexity),
        str(candidate.name),
    )


def _screen_candidate_is_identity_source_representative(candidate: ScreenedCandidate) -> bool:
    source_view = _candidate_source_object_view(candidate)
    return bool(
        _candidate_expr_key(_normalized_expr_tree(dict(candidate.expr)))
        == str(source_view.get("source_object_key") or "")
        and str(source_view.get("chart_signature") or "identity") == "identity"
        and not tuple(source_view.get("realization_head_ops", ()))
    )


def _candidate_is_global_uniform_source(
    *,
    source_expr: Mapping[str, Any],
    realization_head_signature: str,
    uses_piecewise_gate: bool,
    contains_periodic_evidence: bool,
) -> bool:
    return bool(
        not bool(uses_piecewise_gate)
        and not bool(contains_periodic_evidence)
        and not str(realization_head_signature or "").strip()
        and _expr_is_native_trunk_root(dict(source_expr))
    )


def _candidate_is_modulated_branch_source(
    *,
    source_expr: Mapping[str, Any],
    realization_head_signature: str,
    uses_piecewise_gate: bool,
    contains_periodic_evidence: bool,
) -> bool:
    if bool(uses_piecewise_gate) or bool(contains_periodic_evidence):
        return False
    if _candidate_is_global_uniform_source(
        source_expr=source_expr,
        realization_head_signature=realization_head_signature,
        uses_piecewise_gate=uses_piecewise_gate,
        contains_periodic_evidence=contains_periodic_evidence,
    ):
        return False
    return bool(_source_support_indices(source_expr))


def _candidate_is_canonical_trunk_source(
    *,
    source_expr: Mapping[str, Any],
    realization_head_signature: str,
    uses_piecewise_gate: bool,
    contains_periodic_evidence: bool,
    source_support_size: int,
    mechanistic_prior: float,
    rational_template_pinned: bool,
) -> bool:
    return bool(
        int(source_support_size) >= 2
        and not bool(uses_piecewise_gate)
        and not bool(contains_periodic_evidence)
        and not str(realization_head_signature or "").strip()
        and _expr_is_native_trunk_root(dict(source_expr))
        and (
            bool(rational_template_pinned)
            or float(mechanistic_prior) > 0.0
            or int(source_support_size) >= 3
        )
    )


def _candidate_is_same_source_surrogate_source(
    *,
    source_expr: Mapping[str, Any],
    uses_piecewise_gate: bool,
    contains_periodic_evidence: bool,
    source_support_size: int,
    mechanistic_prior: float,
    modulated_branch_candidate: bool,
    canonical_trunk_candidate: bool,
) -> bool:
    if int(source_support_size) < 2 or bool(uses_piecewise_gate) or bool(contains_periodic_evidence):
        return False
    if bool(canonical_trunk_candidate):
        return False
    if float(mechanistic_prior) <= 0.0 and int(source_support_size) < 3:
        return False
    if bool(modulated_branch_candidate):
        return True
    return bool(_source_support_indices(source_expr)) and not _expr_is_native_trunk_root(dict(source_expr))


def _aggregate_screen_cluster_row(
    *,
    representative: ScreenedCandidate,
    cluster_rows: Sequence[ScreenedCandidate],
    cluster_key: str,
) -> ScreenedCandidate:
    members = tuple(cluster_rows)
    if not members:
        return representative
    screen_score = max(float(row.screen_score) for row in members)
    target_corr = max(float(row.target_corr) for row in members)
    residual_gain = max(float(row.residual_gain) for row in members)
    semantic_novelty = max(float(row.semantic_novelty) for row in members)
    consensus_prior = max(float(row.consensus_prior) for row in members)
    mechanistic_prior = max(float(row.mechanistic_prior) for row in members)
    periodic_prior = max(float(row.periodic_prior) for row in members)
    periodic_penalty = min(float(row.periodic_penalty) for row in members)
    contains_periodic_evidence = any(bool(row.contains_periodic_evidence) for row in members)
    native_structure_score = max(float(row.native_structure_score) for row in members)
    native_trunk_root = any(bool(row.native_trunk_root) for row in members)
    native_trunk_floor_passed = any(bool(row.native_trunk_floor_passed) for row in members)
    native_trunk_global_gain = max(float(row.native_trunk_global_gain) for row in members)
    native_trunk_interval_min_gain = max(float(row.native_trunk_interval_min_gain) for row in members)
    native_trunk_interval_mean_gain = max(float(row.native_trunk_interval_mean_gain) for row in members)
    regime_penetration_score = max(float(row.regime_penetration_score) for row in members)
    regime_penetration_min_gain = max(float(row.regime_penetration_min_gain) for row in members)
    regime_penetration_mean_gain = max(float(row.regime_penetration_mean_gain) for row in members)
    regime_sign_consistency = max(float(row.regime_sign_consistency) for row in members)
    heterogeneous_exposure_eligible = any(bool(row.heterogeneous_exposure_eligible) for row in members)
    support_expansion_tagged = any(bool(row.support_expansion_tagged) for row in members)
    canonical_trunk_tagged = any(bool(row.canonical_trunk_tagged) for row in members)
    same_source_surrogate_tagged = any(bool(row.same_source_surrogate_tagged) for row in members)
    support_expansion_candidate = any(bool(row.support_expansion_candidate) for row in members)
    canonical_trunk_candidate = any(bool(row.canonical_trunk_candidate) for row in members)
    same_source_surrogate_candidate = any(bool(row.same_source_surrogate_candidate) for row in members)
    global_uniform_candidate = any(bool(row.global_uniform_candidate) for row in members)
    modulated_branch_candidate = any(bool(row.modulated_branch_candidate) for row in members)
    rational_template_pinned = any(bool(row.rational_template_pinned) for row in members)
    source_support_key = str(
        representative.source_support_key
        or max(
            (str(row.source_support_key or "") for row in members),
            key=lambda value: (len(str(value)), str(value)),
            default="",
        )
    )
    source_support_size = max(int(row.source_support_size) for row in members)
    structural_channel = (
        "regional_speciality"
        if bool(representative.uses_piecewise_gate)
        else (
            "support_expansion"
            if bool(support_expansion_tagged)
            else (
                "canonical_trunk"
                if bool(canonical_trunk_tagged)
                else (
                    "native_trunk"
                    if bool(native_trunk_root) and bool(global_uniform_candidate)
                    else (
                        "same_source_surrogate"
                        if bool(same_source_surrogate_tagged)
                        else ("heterogeneous_exposure" if heterogeneous_exposure_eligible else "challenger")
                    )
                )
            )
        )
    )
    selection_channel = (
        "regional_speciality"
        if bool(representative.uses_piecewise_gate)
        else (
            "support_expansion"
            if bool(support_expansion_candidate)
            else (
                "canonical_trunk"
                if bool(canonical_trunk_candidate)
                else (
                    "native_trunk"
                    if bool(native_trunk_floor_passed)
                    else (
                        "same_source_surrogate"
                        if bool(same_source_surrogate_candidate)
                        else ("heterogeneous_exposure" if heterogeneous_exposure_eligible else "challenger")
                    )
                )
            )
        )
    )
    return replace(
        representative,
        target_corr=float(target_corr),
        screen_score=float(screen_score),
        residual_gain=float(residual_gain),
        semantic_novelty=float(semantic_novelty),
        consensus_prior=float(consensus_prior),
        mechanistic_prior=float(mechanistic_prior),
        periodic_prior=float(periodic_prior),
        periodic_penalty=float(periodic_penalty),
        screen_cluster_key=str(cluster_key),
        screen_cluster_size=int(len(members)),
        information_source_key=str(
            representative.information_source_key or _candidate_information_source_key(representative)
        ),
        contains_periodic_evidence=bool(contains_periodic_evidence),
        source_object_key=str(
            representative.source_object_key or representative.information_source_key or _candidate_information_source_key(representative)
        ),
        source_support_key=str(source_support_key),
        source_support_size=int(source_support_size),
        chart_signature=str(representative.chart_signature or "identity"),
        realization_head_signature=str(representative.realization_head_signature or ""),
        chart_metadata=dict(representative.chart_metadata),
        native_structure_score=float(native_structure_score),
        native_trunk_root=bool(native_trunk_root),
        native_trunk_floor_passed=bool(native_trunk_floor_passed),
        native_trunk_global_gain=float(native_trunk_global_gain),
        native_trunk_interval_min_gain=float(native_trunk_interval_min_gain),
        native_trunk_interval_mean_gain=float(native_trunk_interval_mean_gain),
        regime_penetration_score=float(regime_penetration_score),
        regime_penetration_min_gain=float(regime_penetration_min_gain),
        regime_penetration_mean_gain=float(regime_penetration_mean_gain),
        regime_sign_consistency=float(regime_sign_consistency),
        heterogeneous_exposure_eligible=bool(heterogeneous_exposure_eligible),
        support_expansion_tagged=bool(support_expansion_tagged),
        canonical_trunk_tagged=bool(canonical_trunk_tagged),
        same_source_surrogate_tagged=bool(same_source_surrogate_tagged),
        support_expansion_candidate=bool(support_expansion_candidate),
        canonical_trunk_candidate=bool(canonical_trunk_candidate),
        same_source_surrogate_candidate=bool(same_source_surrogate_candidate),
        global_uniform_candidate=bool(global_uniform_candidate),
        modulated_branch_candidate=bool(modulated_branch_candidate),
        rational_template_pinned=bool(rational_template_pinned),
        structural_channel=str(structural_channel),
        selection_channel=str(selection_channel),
    )


def _compress_screen_information_clusters(
    *,
    rows: Sequence[tuple[ScreenedCandidate, np.ndarray]],
    raw_X: np.ndarray,
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
    periodic_context: Mapping[str, Any],
) -> list[tuple[ScreenedCandidate, np.ndarray]]:
    ranked_rows = list(tuple(rows))
    if len(ranked_rows) <= 1:
        return ranked_rows
    feature_name_tuple = tuple(str(value) for value in tuple(feature_names))
    scope_buckets: dict[str, list[int]] = {}
    for index, (candidate, _values) in enumerate(tuple(ranked_rows)):
        scope_key = _screen_information_scope_key(
            candidate=candidate,
            feature_names=feature_name_tuple,
            interference_context=interference_context,
            periodic_context=periodic_context,
        )
        scope_buckets.setdefault(str(scope_key), []).append(int(index))
    selected: list[tuple[ScreenedCandidate, np.ndarray]] = []
    for scope_key in sorted(scope_buckets.keys()):
        indices = list(scope_buckets[str(scope_key)])
        if len(indices) <= 1:
            row, values = ranked_rows[indices[0]]
            selected.append(
                (
                    _aggregate_screen_cluster_row(
                        representative=row,
                        cluster_rows=(row,),
                        cluster_key=f"{scope_key}::cluster_00",
                    ),
                    np.asarray(values, dtype=float).reshape(-1),
                )
            )
            continue
        if str(scope_key).startswith("source_object::"):
            cluster_key = f"{scope_key}::cluster_00"
            cluster_rows = tuple(ranked_rows[int(index)][0] for index in indices)
            canonical_indices = [
                int(index)
                for index in indices
                if _screen_candidate_is_identity_source_representative(ranked_rows[int(index)][0])
            ]
            representative_indices = canonical_indices or indices
            representative_index = min(
                representative_indices,
                key=lambda index: _screen_cluster_representative_priority(ranked_rows[int(index)][0]),
            )
            row, _values = ranked_rows[int(representative_index)]
            canonical_source_expr = _candidate_information_source_expr(row)
            canonical_values = design_matrix_for_genome(
                (
                    {
                        "name": f"{cluster_key}::source_object",
                        "expr": dict(canonical_source_expr),
                    },
                ),
                np.asarray(raw_X, dtype=float),
                batch_key=f"orthogonal_screen_cluster::{cluster_key}",
            )
            selected.append(
                (
                    _aggregate_screen_cluster_row(
                        representative=row,
                        cluster_rows=cluster_rows,
                        cluster_key=str(cluster_key),
                    ),
                    np.asarray(canonical_values[:, 0], dtype=float).reshape(-1),
                )
            )
            continue
        scope_matrix = np.asarray(
            np.stack([np.asarray(ranked_rows[index][1], dtype=float).reshape(-1) for index in indices], axis=1),
            dtype=float,
        )
        scope_corr = _pairwise_abs_corr(scope_matrix)
        seen_local: set[int] = set()
        cluster_index = 0
        for start in range(len(indices)):
            if start in seen_local:
                continue
            stack = [int(start)]
            component: set[int] = set()
            while stack:
                local_index = int(stack.pop())
                if local_index in component:
                    continue
                component.add(local_index)
                for other in range(len(indices)):
                    if other == local_index:
                        continue
                    if float(scope_corr[local_index, other]) >= float(_SCREEN_INFORMATION_CLUSTER_CORR_THRESHOLD):
                        stack.append(int(other))
            seen_local.update(component)
            component_indices = [int(indices[local_index]) for local_index in sorted(component)]
            cluster_key = f"{scope_key}::cluster_{int(cluster_index):02d}"
            cluster_index += 1
            cluster_size = int(len(component_indices))
            best_index = min(
                component_indices,
                key=lambda index: _screen_cluster_representative_priority(ranked_rows[index][0]),
            )
            row, values = ranked_rows[int(best_index)]
            cluster_rows = tuple(ranked_rows[int(index)][0] for index in component_indices)
            selected.append(
                (
                    _aggregate_screen_cluster_row(
                        representative=row,
                        cluster_rows=cluster_rows,
                        cluster_key=str(cluster_key),
                    ),
                    np.asarray(values, dtype=float).reshape(-1),
                )
            )
    return selected


@dataclass(frozen=True)
class OrthogonalBasisFitResult:
    genome: tuple[dict[str, Any], ...]
    readout_weight: np.ndarray
    readout_bias: np.ndarray
    pred_train: np.ndarray
    residual_std: np.ndarray
    train_metrics: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class BudgetedOrthogonalAssemblerResult:
    basis_feature_names: tuple[str, ...]
    basis_space_genome: tuple[dict[str, Any], ...]
    assembled_genome: tuple[dict[str, Any], ...]
    inner_result: StructureSearchResult
    final_fit: Mapping[str, Any]
    final_expression_payload: Mapping[str, Any]
    fold_report: Mapping[str, Any]
    outer_objective: Mapping[str, Any]
    search_config: Mapping[str, Any]
    stage_head_protocols: Mapping[str, Any]
    basis_context: Mapping[str, Any]
    object_gradient_pool: Mapping[str, Any]
    environment_invariance_audit: Mapping[str, Any]
    periodic_equivalence_report: Mapping[str, Any]
    regional_correction_report: Mapping[str, Any]
    mandatory_realization_closure_report: Mapping[str, Any]
    same_source_over_realization_report: Mapping[str, Any]


def _candidate_expr_key(expr: Mapping[str, Any]) -> str:
    return str(_jsonable(dict(expr)))


def _minimal_feature_bundle(
    *,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
) -> FeatureBundle:
    x = np.asarray(X, dtype=float)
    yy = np.asarray(y, dtype=float)
    if yy.ndim == 1:
        yy = yy.reshape(-1, 1)
    names = tuple(str(value) for value in tuple(feature_names))
    pseudo_count = min(max(8, int(round(x.shape[0] * 0.15))), int(x.shape[0]))
    pseudo_test_x = np.asarray(x[-pseudo_count:], dtype=float)
    pseudo_test_y = np.asarray(yy[-pseudo_count:], dtype=float)
    return FeatureBundle(
        X_train=np.asarray(x, dtype=float),
        y_train=np.asarray(yy, dtype=float),
        X_test=np.asarray(pseudo_test_x, dtype=float),
        y_test=np.asarray(pseudo_test_y, dtype=float),
        feature_names=names,
        n_features_raw=int(x.shape[1]),
        feature_names_raw=names,
        lag_added_features=tuple(),
        lag_cross_added_features=tuple(),
        dropped_features=tuple(),
    )


def _single_basis_row(
    *,
    name: str,
    expr: Mapping[str, Any],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    rows = build_basis_term_rows(
        [{"name": str(name), "expr": dict(expr)}],
        feature_names=tuple(str(value) for value in tuple(feature_names)),
        scope="global",
    )
    return dict(rows[0]) if rows else {"term_name": str(name), "expression": expression_to_string(dict(expr), precision=8)}


def _selected_basis_rows(selected_rows: Sequence[ScreenedCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "term_name": str(row.name),
            "expression": str(row.expression),
            "feature_names": [str(value) for value in tuple(row.features)],
            "feature_indices": [int(value) for value in tuple(row.features)],
            "feature_count": int(len(tuple(row.features))),
            "semantic_signature": str(row.semantic_signature),
            "semantic_family": str(row.semantic_family),
            "uses_piecewise_gate": bool(row.uses_piecewise_gate),
            "source_object_key": str(row.source_object_key),
            "source_support_key": str(row.source_support_key),
            "chart_signature": str(row.chart_signature),
            "structural_channel": str(row.structural_channel),
            "support_expansion_tagged": bool(row.support_expansion_tagged),
            "canonical_trunk_tagged": bool(row.canonical_trunk_tagged),
            "same_source_surrogate_tagged": bool(row.same_source_surrogate_tagged),
            "selection_channel": str(row.selection_channel),
            "support_expansion_candidate": bool(row.support_expansion_candidate),
            "canonical_trunk_candidate": bool(row.canonical_trunk_candidate),
            "same_source_surrogate_candidate": bool(row.same_source_surrogate_candidate),
            "regime_penetration_score": float(row.regime_penetration_score),
            "regime_penetration_min_gain": float(row.regime_penetration_min_gain),
            "regime_penetration_mean_gain": float(row.regime_penetration_mean_gain),
            "regime_sign_consistency": float(row.regime_sign_consistency),
            "heterogeneous_exposure_eligible": bool(row.heterogeneous_exposure_eligible),
            "scope": "global",
        }
        for row in tuple(selected_rows)
    ]


def _score_unit_interval(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(numeric):
        return 0.0
    return float(np.clip(numeric, 0.0, 1.0))


def _annotated_candidate_entry(
    *,
    name: str,
    expr: Mapping[str, Any],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    basis_row = _single_basis_row(name=str(name), expr=dict(expr), feature_names=feature_names)
    annotated = annotate_basis_entries(
        (basis_row,),
        ({"name": str(name), "expr": dict(expr)},),
    )
    if annotated:
        return dict(annotated[0])
    return {
        "term_name": str(name),
        "expression": str(basis_row.get("expression") or ""),
        "exact_expr_key": str(basis_row.get("expression") or ""),
        "strict_class_id": str(basis_row.get("expression") or ""),
        "phase_class_id": str(basis_row.get("expression") or ""),
        "family_class_id": str(basis_row.get("semantic_family") or ""),
        "phase_contract": str(basis_row.get("semantic_family") or ""),
        "family_contract": str(basis_row.get("semantic_family") or ""),
        "feature_names": [str(value) for value in tuple(basis_row.get("feature_names", ()) or ())],
        "semantic_family": str(basis_row.get("semantic_family") or ""),
        "semantic_signature": str(basis_row.get("semantic_signature") or ""),
        "uses_piecewise_gate": bool(basis_row.get("uses_piecewise_gate")),
    }


def _build_consensus_prior_model(rows: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    model: dict[str, Any] = {
        "exact": {},
        "phase": {},
        "family": {},
        "semantic_family": {},
        "summary": {
            "row_count": 0,
            "exact_key_count": 0,
            "phase_key_count": 0,
            "family_key_count": 0,
            "semantic_family_count": 0,
        },
    }
    if not rows:
        return model
    exact_map: dict[str, float] = {}
    phase_map: dict[str, float] = {}
    family_map: dict[str, float] = {}
    semantic_map: dict[str, float] = {}
    for raw in tuple(rows):
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        support_rate = _score_unit_interval(
            row.get("support_rate", row.get("multi_run_core_frequency", 0.0))
        )
        support_weight_rate = _score_unit_interval(row.get("support_weight_rate", support_rate))
        exact_support_rate = _score_unit_interval(
            row.get("representative_exact_support_rate", row.get("dominant_exact_support_rate", 0.0))
        )
        exact_stability = _score_unit_interval(row.get("exact_stability", row.get("dominant_exact_stability", 0.0)))
        joint_core_score = _score_unit_interval(
            row.get(
                "joint_core_score",
                0.50 * support_rate + 0.30 * exact_stability + 0.20 * support_weight_rate,
            )
        )
        exact_score = _score_unit_interval(0.45 * joint_core_score + 0.35 * exact_stability + 0.20 * exact_support_rate)
        phase_score = _score_unit_interval(0.40 * joint_core_score + 0.25 * exact_stability + 0.20 * support_rate + 0.15 * support_weight_rate)
        family_score = _score_unit_interval(0.55 * joint_core_score + 0.25 * support_rate + 0.20 * support_weight_rate)
        semantic_score = _score_unit_interval(0.50 * family_score + 0.30 * support_rate + 0.20 * exact_stability)

        exact_key = str(row.get("representative_exact_expr_key") or row.get("strict_class_id") or "").strip()
        phase_key = str(row.get("representative_phase_class_id") or row.get("phase_class_id") or "").strip()
        family_key = str(row.get("representative_family_class_id") or row.get("basis_class_id") or row.get("family_class_id") or "").strip()
        semantic_family = str(
            row.get("representative_semantic_family") or row.get("semantic_family") or ""
        ).strip()
        if exact_key:
            exact_map[exact_key] = max(float(exact_map.get(exact_key, 0.0)), float(exact_score))
        if phase_key:
            phase_map[phase_key] = max(float(phase_map.get(phase_key, 0.0)), float(phase_score))
        if family_key:
            family_map[family_key] = max(float(family_map.get(family_key, 0.0)), float(family_score))
        if semantic_family:
            semantic_map[semantic_family] = max(float(semantic_map.get(semantic_family, 0.0)), float(semantic_score))

    model["exact"] = exact_map
    model["phase"] = phase_map
    model["family"] = family_map
    model["semantic_family"] = semantic_map
    model["summary"] = {
        "row_count": int(sum(1 for row in tuple(rows) if isinstance(row, Mapping))),
        "exact_key_count": int(len(exact_map)),
        "phase_key_count": int(len(phase_map)),
        "family_key_count": int(len(family_map)),
        "semantic_family_count": int(len(semantic_map)),
    }
    return model


def _consensus_prior_score(
    *,
    candidate_entry: Mapping[str, Any],
    consensus_prior_model: Mapping[str, Any] | None,
) -> float:
    if not isinstance(consensus_prior_model, Mapping):
        return 0.0
    exact_key = str(candidate_entry.get("exact_expr_key") or candidate_entry.get("strict_class_id") or "").strip()
    phase_key = str(candidate_entry.get("phase_class_id") or "").strip()
    family_key = str(candidate_entry.get("family_class_id") or "").strip()
    semantic_family = str(candidate_entry.get("semantic_family") or "").strip()
    exact = _score_unit_interval(dict(consensus_prior_model.get("exact", {})).get(exact_key))
    phase = _score_unit_interval(dict(consensus_prior_model.get("phase", {})).get(phase_key))
    family = _score_unit_interval(dict(consensus_prior_model.get("family", {})).get(family_key))
    semantic = _score_unit_interval(dict(consensus_prior_model.get("semantic_family", {})).get(semantic_family))
    return float(max(exact, 0.90 * phase, 0.75 * family, 0.50 * semantic))


def _semantic_novelty_score(
    *,
    semantic_signature: str,
    semantic_family: str,
    signature_counts: Mapping[str, int],
    family_counts: Mapping[str, int],
    uses_piecewise_gate: bool,
) -> float:
    signature_count = max(1, int(signature_counts.get(str(semantic_signature), 1)))
    family_count = max(1, int(family_counts.get(str(semantic_family), 1)))
    novelty = float(1.0 / math.sqrt(float(signature_count) * max(1.0, 0.75 * float(family_count))))
    if bool(uses_piecewise_gate):
        novelty = float(max(novelty, 0.75 * novelty + 0.25))
    return _score_unit_interval(novelty)


def _configured_gate_feature_names(
    *,
    cfg: OrthogonalBasisSearchConfig,
    feature_names: Sequence[str],
) -> tuple[str, ...]:
    allowed = {str(name) for name in tuple(feature_names)}
    return tuple(name for name in tuple(cfg.gate_feature_names) if str(name) in allowed)


def _configured_mechanistic_feature_groups(
    *,
    cfg: OrthogonalBasisSearchConfig,
    feature_names: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    allowed = {str(name) for name in tuple(feature_names)}
    groups: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for group in tuple(cfg.mechanistic_feature_groups):
        normalized = tuple(
            dict.fromkeys(str(name).strip() for name in tuple(group) if str(name).strip() in allowed)
        )
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        groups.append(normalized)
    return tuple(groups)


def _configured_mechanistic_group_feature_indices(
    *,
    cfg: OrthogonalBasisSearchConfig,
    feature_names: Sequence[str],
) -> tuple[tuple[int, ...], ...]:
    index_lookup = {
        str(name): int(index)
        for index, name in enumerate(tuple(str(value) for value in tuple(feature_names)))
    }
    groups: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for group in _configured_mechanistic_feature_groups(cfg=cfg, feature_names=feature_names):
        indices = tuple(index_lookup[str(name)] for name in tuple(group) if str(name) in index_lookup)
        if len(indices) < 2 or indices in seen:
            continue
        seen.add(indices)
        groups.append(indices)
    return tuple(groups)


def _source_support_key_from_indices(indices: Sequence[int]) -> str:
    ordered = tuple(sorted(int(index) for index in tuple(indices) if int(index) >= 0))
    if not ordered:
        return ""
    return "+".join(f"f{int(index)}" for index in ordered)


def _source_support_indices(expr: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(sorted(idx for idx in _expr_collect_feature_indices(dict(expr)) if idx >= 0))


def _build_native_mechanistic_group_expr(feature_indices: Sequence[int]) -> dict[str, Any]:
    ordered = tuple(int(index) for index in tuple(feature_indices))
    if len(ordered) < 2:
        raise ValueError("feature_indices must contain at least two entries.")
    if len(ordered) == 2:
        return _binary_expr("div", _feature_expr(int(ordered[0])), _feature_expr(int(ordered[1])))
    numerator = _feature_expr(int(ordered[0]))
    for feature_index in ordered[1:-1]:
        numerator = _binary_expr("mul", numerator, _feature_expr(int(feature_index)))
    return _binary_expr("div", numerator, _feature_expr(int(ordered[-1])))


def _candidate_feature_name_set(
    *,
    feature_indices: Sequence[int],
    feature_names: Sequence[str],
) -> set[str]:
    names = tuple(str(value) for value in tuple(feature_names))
    out: set[str] = set()
    for feature_index in tuple(feature_indices):
        idx = int(feature_index)
        if 0 <= idx < len(names):
            out.add(str(names[idx]))
    return out


def _candidate_mechanistic_prior_score(
    *,
    feature_indices: Sequence[int],
    feature_names: Sequence[str],
    mechanistic_groups: Sequence[Sequence[str]],
) -> float:
    if not mechanistic_groups:
        return 0.0
    candidate_features = _candidate_feature_name_set(
        feature_indices=feature_indices,
        feature_names=feature_names,
    )
    if not candidate_features:
        return 0.0
    best = 0.0
    for group in tuple(mechanistic_groups):
        group_set = {str(name) for name in tuple(group) if str(name).strip()}
        if not group_set:
            continue
        overlap = float(len(candidate_features & group_set)) / float(len(group_set))
        if group_set.issubset(candidate_features):
            overlap = max(overlap, 1.0)
        best = max(best, overlap if group_set.issubset(candidate_features) else 0.0)
    return _score_unit_interval(best)


def _required_gate_basis_terms(cfg: OrthogonalBasisSearchConfig) -> int:
    required = int(max(0, cfg.min_gate_basis_terms))
    if bool(cfg.require_gate_candidate_in_group):
        required = max(required, 1)
    if bool(cfg.enable_piecewise_basis) and tuple(cfg.gate_feature_names):
        required = max(required, 1)
    if not bool(cfg.enable_piecewise_basis) or not tuple(cfg.gate_feature_names):
        return 0
    return required


def _required_native_trunk_basis_terms(cfg: OrthogonalBasisSearchConfig) -> int:
    required = int(max(0, cfg.min_native_trunk_basis_terms))
    if bool(cfg.require_native_trunk_candidate_in_group):
        required = max(required, 1)
    return required


def _native_trunk_term_count(rows: Sequence[ScreenedCandidate]) -> int:
    return int(sum(1 for row in tuple(rows) if bool(row.native_trunk_floor_passed)))


def _required_support_expansion_basis_terms(cfg: OrthogonalBasisSearchConfig) -> int:
    required = int(max(0, cfg.min_support_expansion_basis_terms))
    if bool(cfg.require_support_expansion_candidate_in_group):
        required = max(required, 1)
    if not _support_expansion_enabled(cfg):
        return 0
    return required


def _support_expansion_term_count(rows: Sequence[ScreenedCandidate]) -> int:
    return int(sum(1 for row in tuple(rows) if bool(row.support_expansion_candidate)))


def _required_canonical_trunk_basis_terms(cfg: OrthogonalBasisSearchConfig) -> int:
    required = int(max(0, cfg.min_canonical_trunk_basis_terms))
    if bool(cfg.require_canonical_trunk_candidate_in_group):
        required = max(required, 1)
    if not _canonical_trunk_lane_enabled(cfg):
        return 0
    return required


def _canonical_trunk_term_count(rows: Sequence[ScreenedCandidate]) -> int:
    return int(
        sum(
            1
            for row in tuple(rows)
            if bool(row.canonical_trunk_candidate) or bool(row.support_expansion_candidate)
        )
    )


def _gate_term_count(rows: Sequence[ScreenedCandidate]) -> int:
    return int(sum(1 for row in tuple(rows) if bool(row.uses_piecewise_gate)))


def _required_periodic_basis_terms(
    *,
    cfg: OrthogonalBasisSearchConfig,
    periodic_context: Mapping[str, Any],
) -> int:
    periodic_feature_names = tuple(
        str(value) for value in tuple(periodic_context.get("periodic_feature_names", ())) if str(value).strip()
    )
    required = int(max(0, cfg.min_periodic_basis_terms))
    if bool(cfg.require_periodic_candidate_in_group):
        required = max(required, 1)
    if periodic_feature_names:
        required = max(required, 1)
    return required if periodic_feature_names else 0


def _periodic_term_count(rows: Sequence[ScreenedCandidate]) -> int:
    return int(
        sum(
            1
            for row in tuple(rows)
            if bool(row.contains_periodic_evidence)
            or _candidate_is_periodic_family(
                semantic_family=str(row.semantic_family),
                expr=dict(row.expr),
            )
        )
    )


def _group_meets_gate_requirement(
    *,
    rows: Sequence[ScreenedCandidate],
    cfg: OrthogonalBasisSearchConfig,
    gate_candidates_available: bool,
) -> bool:
    required = _required_gate_basis_terms(cfg)
    if required <= 0 or not gate_candidates_available:
        return True
    return _gate_term_count(rows) >= required


def _group_meets_native_trunk_requirement(
    *,
    rows: Sequence[ScreenedCandidate],
    cfg: OrthogonalBasisSearchConfig,
    native_candidates_available: bool,
) -> bool:
    required = _required_native_trunk_basis_terms(cfg)
    if required <= 0 or not native_candidates_available:
        return True
    return _native_trunk_term_count(rows) >= required


def _group_meets_support_expansion_requirement(
    *,
    rows: Sequence[ScreenedCandidate],
    cfg: OrthogonalBasisSearchConfig,
    support_expansion_candidates_available: bool,
) -> bool:
    required = _required_support_expansion_basis_terms(cfg)
    if required <= 0 or not support_expansion_candidates_available:
        return True
    return _support_expansion_term_count(rows) >= required


def _group_meets_canonical_trunk_requirement(
    *,
    rows: Sequence[ScreenedCandidate],
    cfg: OrthogonalBasisSearchConfig,
    canonical_trunk_candidates_available: bool,
) -> bool:
    required = _required_canonical_trunk_basis_terms(cfg)
    if required <= 0 or not canonical_trunk_candidates_available:
        return True
    return _canonical_trunk_term_count(rows) >= required


def _group_meets_periodic_requirement(
    *,
    rows: Sequence[ScreenedCandidate],
    cfg: OrthogonalBasisSearchConfig,
    periodic_context: Mapping[str, Any],
    periodic_candidates_available: bool,
) -> bool:
    required = _required_periodic_basis_terms(cfg=cfg, periodic_context=periodic_context)
    if required <= 0 or not periodic_candidates_available:
        return True
    return _periodic_term_count(rows) >= required


def _native_trunk_reserve_sort_key(item: tuple[ScreenedCandidate, np.ndarray]) -> tuple[Any, ...]:
    row = item[0]
    return (
        0 if bool(row.support_expansion_candidate) else 1,
        -int(len(tuple(row.features))),
        -float(row.regime_penetration_score),
        -float(row.native_trunk_global_gain),
        -float(row.native_trunk_interval_min_gain),
        -float(row.native_trunk_interval_mean_gain),
        -float(row.regime_sign_consistency),
        -float(row.target_corr),
        float(row.complexity),
        str(row.name),
    )


def _support_expansion_reserve_sort_key(item: tuple[ScreenedCandidate, np.ndarray]) -> tuple[Any, ...]:
    row = item[0]
    return (
        -int(row.source_support_size),
        -float(row.mechanistic_prior),
        -float(row.native_trunk_interval_min_gain),
        -float(row.native_trunk_global_gain),
        -float(row.residual_gain),
        -float(row.target_corr),
        float(row.complexity),
        str(row.name),
    )


def _canonical_trunk_reserve_sort_key(item: tuple[ScreenedCandidate, np.ndarray]) -> tuple[Any, ...]:
    row = item[0]
    return (
        0 if bool(row.rational_template_pinned) else 1,
        -int(row.source_support_size),
        -float(row.mechanistic_prior),
        -float(row.native_trunk_interval_min_gain),
        -float(row.native_trunk_global_gain),
        -float(row.regime_penetration_score),
        -float(row.target_corr),
        float(row.complexity),
        str(row.name),
    )


def _reserve_native_trunk_candidates(
    *,
    ranked_rows: Sequence[tuple[ScreenedCandidate, np.ndarray]],
    full_ranked_rows: Sequence[tuple[ScreenedCandidate, np.ndarray]],
    candidate_limit: int,
    reserve_count: int,
) -> list[tuple[ScreenedCandidate, np.ndarray]]:
    limited = list(tuple(ranked_rows)[: int(max(0, candidate_limit))])
    if not limited:
        return limited
    reserve = int(min(max(0, reserve_count), len(limited)))
    if reserve <= 0:
        return limited

    def _is_native(item: tuple[ScreenedCandidate, np.ndarray]) -> bool:
        return bool(item[0].native_trunk_floor_passed)

    native_count = sum(1 for item in limited if _is_native(item))
    if native_count >= reserve:
        native_items = sorted([item for item in limited if _is_native(item)], key=_native_trunk_reserve_sort_key)
        selected_ids = {
            (int(item[0].pool_index), str(item[0].name))
            for item in native_items[:reserve]
        }
        head = [item for item in native_items[:reserve]]
        tail = [
            item
            for item in limited
            if (int(item[0].pool_index), str(item[0].name)) not in selected_ids
        ]
        return list(head) + list(tail)
    seen_indices = {int(item[0].pool_index) for item in limited}
    extra_native_rows = sorted(
        [
            item
            for item in tuple(full_ranked_rows)
            if _is_native(item) and int(item[0].pool_index) not in seen_indices
        ],
        key=_native_trunk_reserve_sort_key,
    )
    replace_positions = [
        index
        for index in range(len(limited) - 1, -1, -1)
        if not _is_native(limited[index])
    ]
    for extra_item in extra_native_rows:
        if native_count >= reserve or not replace_positions:
            break
        limited[replace_positions.pop(0)] = extra_item
        native_count += 1
    limited.sort(
        key=lambda item: (
            0 if _is_native(item) else 1,
            *_native_trunk_reserve_sort_key(item),
        )
    )
    return limited


def _reserve_support_expansion_candidates(
    *,
    ranked_rows: Sequence[tuple[ScreenedCandidate, np.ndarray]],
    full_ranked_rows: Sequence[tuple[ScreenedCandidate, np.ndarray]],
    candidate_limit: int,
    reserve_count: int,
    cfg: OrthogonalBasisSearchConfig,
) -> list[tuple[ScreenedCandidate, np.ndarray]]:
    limited = list(tuple(ranked_rows)[: int(max(0, candidate_limit))])
    if not limited:
        return limited
    reserve = int(min(max(0, reserve_count), len(limited)))
    if reserve <= 0 or not _support_expansion_enabled(cfg):
        return limited

    def _is_support(item: tuple[ScreenedCandidate, np.ndarray]) -> bool:
        return bool(item[0].support_expansion_tagged) or bool(item[0].support_expansion_candidate)

    support_count = sum(1 for item in limited if _is_support(item))
    if support_count >= reserve:
        support_items = sorted([item for item in limited if _is_support(item)], key=_support_expansion_reserve_sort_key)
        selected_ids = {
            (int(item[0].pool_index), str(item[0].name))
            for item in support_items[:reserve]
        }
        head = [item for item in support_items[:reserve]]
        tail = [
            item
            for item in limited
            if (int(item[0].pool_index), str(item[0].name)) not in selected_ids
        ]
        return list(head) + list(tail)
    seen_indices = {int(item[0].pool_index) for item in limited}
    extra_rows = sorted(
        [
            item
            for item in tuple(full_ranked_rows)
            if _is_support(item) and int(item[0].pool_index) not in seen_indices
        ],
        key=_support_expansion_reserve_sort_key,
    )
    replace_positions = [
        index
        for index in range(len(limited) - 1, -1, -1)
        if not _is_support(limited[index])
    ]
    for extra_item in extra_rows:
        if support_count >= reserve or not replace_positions:
            break
        limited[replace_positions.pop(0)] = extra_item
        support_count += 1
    limited.sort(
        key=lambda item: (
            0 if _is_support(item) else 1,
            *_support_expansion_reserve_sort_key(item),
        )
    )
    return limited


def _reserve_canonical_trunk_candidates(
    *,
    ranked_rows: Sequence[tuple[ScreenedCandidate, np.ndarray]],
    full_ranked_rows: Sequence[tuple[ScreenedCandidate, np.ndarray]],
    candidate_limit: int,
    reserve_count: int,
    cfg: OrthogonalBasisSearchConfig,
) -> list[tuple[ScreenedCandidate, np.ndarray]]:
    limited = list(tuple(ranked_rows)[: int(max(0, candidate_limit))])
    if not limited:
        return limited
    reserve = int(min(max(0, reserve_count), len(limited)))
    if reserve <= 0 or not _canonical_trunk_lane_enabled(cfg):
        return limited

    def _is_canonical(item: tuple[ScreenedCandidate, np.ndarray]) -> bool:
        return (
            bool(item[0].canonical_trunk_tagged)
            or bool(item[0].support_expansion_tagged)
            or bool(item[0].canonical_trunk_candidate)
            or bool(item[0].support_expansion_candidate)
        )

    canonical_count = sum(1 for item in limited if _is_canonical(item))
    if canonical_count >= reserve:
        canonical_items = sorted([item for item in limited if _is_canonical(item)], key=_canonical_trunk_reserve_sort_key)
        selected_ids = {
            (int(item[0].pool_index), str(item[0].name))
            for item in canonical_items[:reserve]
        }
        head = [item for item in canonical_items[:reserve]]
        tail = [
            item
            for item in limited
            if (int(item[0].pool_index), str(item[0].name)) not in selected_ids
        ]
        return list(head) + list(tail)
    seen_indices = {int(item[0].pool_index) for item in limited}
    extra_rows = sorted(
        [
            item
            for item in tuple(full_ranked_rows)
            if _is_canonical(item) and int(item[0].pool_index) not in seen_indices
        ],
        key=_canonical_trunk_reserve_sort_key,
    )
    replace_positions = [
        index
        for index in range(len(limited) - 1, -1, -1)
        if not _is_canonical(limited[index])
    ]
    for extra_item in extra_rows:
        if canonical_count >= reserve or not replace_positions:
            break
        limited[replace_positions.pop(0)] = extra_item
        canonical_count += 1
    limited.sort(
        key=lambda item: (
            0 if _is_canonical(item) else 1,
            *_canonical_trunk_reserve_sort_key(item),
        )
    )
    return limited


def _reserve_gate_candidates(
    *,
    ranked_rows: Sequence[tuple[ScreenedCandidate, np.ndarray]],
    candidate_limit: int,
    reserve_count: int,
) -> list[tuple[ScreenedCandidate, np.ndarray]]:
    limited = list(tuple(ranked_rows)[: int(max(0, candidate_limit))])
    if not limited:
        return limited
    reserve = int(min(max(0, reserve_count), len(limited)))
    if reserve <= 0:
        return limited
    gate_count = sum(1 for row, _values in limited if bool(row.uses_piecewise_gate))
    if gate_count >= reserve:
        return limited
    extra_gate_rows = [item for item in tuple(ranked_rows)[len(limited) :] if bool(item[0].uses_piecewise_gate)]
    if not extra_gate_rows:
        return limited
    replace_positions = [index for index in range(len(limited) - 1, -1, -1) if not bool(limited[index][0].uses_piecewise_gate)]
    for extra_item in extra_gate_rows:
        if gate_count >= reserve or not replace_positions:
            break
        limited[replace_positions.pop(0)] = extra_item
        gate_count += 1
    limited.sort(
        key=lambda item: (
            -float(item[0].screen_score),
            -float(item[0].mechanistic_prior),
            -float(item[0].consensus_prior),
            -float(item[0].residual_gain),
            float(item[0].complexity),
            str(item[0].name),
        )
    )
    return limited


def _reserve_periodic_candidates(
    *,
    ranked_rows: Sequence[tuple[ScreenedCandidate, np.ndarray]],
    full_ranked_rows: Sequence[tuple[ScreenedCandidate, np.ndarray]],
    reserve_count: int,
) -> list[tuple[ScreenedCandidate, np.ndarray]]:
    limited = list(tuple(ranked_rows))
    if not limited:
        return limited
    reserve = int(min(max(0, reserve_count), len(limited)))
    if reserve <= 0:
        return limited

    def _is_periodic(item: tuple[ScreenedCandidate, np.ndarray]) -> bool:
        row = item[0]
        return bool(row.contains_periodic_evidence) or (
            float(row.periodic_prior) > 0.0 and float(row.periodic_penalty) <= 0.0
        )

    periodic_count = sum(1 for item in limited if _is_periodic(item))
    if periodic_count >= reserve:
        return limited
    seen_indices = {int(item[0].screen_index) for item in limited}
    extra_periodic_rows = [
        item
        for item in tuple(full_ranked_rows)
        if _is_periodic(item) and int(item[0].screen_index) not in seen_indices
    ]
    if not extra_periodic_rows:
        return limited
    replace_positions = [
        index
        for index in range(len(limited) - 1, -1, -1)
        if not _is_periodic(limited[index])
    ]
    for extra_item in extra_periodic_rows:
        if periodic_count >= reserve or not replace_positions:
            break
        limited[replace_positions.pop(0)] = extra_item
        periodic_count += 1
    limited.sort(
        key=lambda item: (
            -float(item[0].screen_score),
            -float(item[0].periodic_prior),
            -float(item[0].mechanistic_prior),
            -float(item[0].consensus_prior),
            -float(item[0].residual_gain),
            float(item[0].complexity),
            str(item[0].name),
        )
    )
    return limited


def _heterogeneous_exposure_sort_key(item: tuple[ScreenedCandidate, np.ndarray]) -> tuple[Any, ...]:
    row = item[0]
    return (
        -float(row.regime_penetration_score),
        -float(row.regime_penetration_min_gain),
        -float(row.regime_sign_consistency),
        -float(row.native_structure_score),
        -float(row.residual_gain),
        float(row.complexity),
        str(row.name),
    )


def _reserve_heterogeneous_exposure_candidates(
    *,
    ranked_rows: Sequence[tuple[ScreenedCandidate, np.ndarray]],
    full_ranked_rows: Sequence[tuple[ScreenedCandidate, np.ndarray]],
    candidate_limit: int,
    reserve_count: int,
    cfg: OrthogonalBasisSearchConfig,
) -> list[tuple[ScreenedCandidate, np.ndarray]]:
    limited = list(tuple(ranked_rows)[: int(max(0, candidate_limit))])
    if not limited:
        return limited
    reserve = int(min(max(0, reserve_count), len(limited)))
    if reserve <= 0 or not _heterogeneous_exposure_enabled(cfg):
        return limited

    def _eligible(item: tuple[ScreenedCandidate, np.ndarray]) -> bool:
        return bool(item[0].heterogeneous_exposure_eligible)

    exposure_count = sum(1 for item in limited if _eligible(item))
    if exposure_count >= reserve:
        exposure_items = sorted([item for item in limited if _eligible(item)], key=_heterogeneous_exposure_sort_key)
        selected_ids = {
            (int(item[0].pool_index), str(item[0].name))
            for item in exposure_items[:reserve]
        }
        head = [item for item in exposure_items[:reserve]]
        tail = [
            item
            for item in limited
            if (int(item[0].pool_index), str(item[0].name)) not in selected_ids
        ]
        return list(head) + list(tail)
    seen_indices = {int(item[0].pool_index) for item in limited}
    extra_rows = sorted(
        [
            item
            for item in tuple(full_ranked_rows)
            if _eligible(item) and int(item[0].pool_index) not in seen_indices
        ],
        key=_heterogeneous_exposure_sort_key,
    )
    replace_positions = [
        index
        for index in range(len(limited) - 1, -1, -1)
        if not _eligible(limited[index])
    ]
    for extra_item in extra_rows:
        if exposure_count >= reserve or not replace_positions:
            break
        limited[replace_positions.pop(0)] = extra_item
        exposure_count += 1
    limited.sort(
        key=lambda item: (
            0 if _eligible(item) else 1,
            *_heterogeneous_exposure_sort_key(item),
        )
    )
    return limited


def _build_piecewise_gate_specs(
    *,
    feature_bundle: FeatureBundle,
    cfg: OrthogonalBasisSearchConfig,
) -> tuple[ConditionalPrimitiveSpec, ...]:
    if not bool(cfg.enable_piecewise_basis):
        return tuple()
    feature_names = tuple(str(value) for value in tuple(feature_bundle.feature_names))
    gate_features = _configured_gate_feature_names(cfg=cfg, feature_names=feature_names)
    if not gate_features:
        return tuple()
    x_train = np.asarray(feature_bundle.X_train, dtype=float)
    name_to_idx = {str(name): int(index) for index, name in enumerate(feature_names)}
    specs: list[ConditionalPrimitiveSpec] = []
    seen: set[tuple[str, str, float]] = set()
    for feature_name in gate_features:
        feature_index = name_to_idx.get(str(feature_name))
        if feature_index is None:
            continue
        column = np.asarray(x_train[:, int(feature_index)], dtype=float).reshape(-1)
        column = column[np.isfinite(column)]
        if column.size < 16:
            continue
        for quantile in tuple(cfg.gate_quantiles):
            cut = float(np.quantile(column, float(quantile)))
            if not np.isfinite(cut):
                continue
            for family in tuple(cfg.gate_families):
                key = (str(feature_name), str(family), round(cut, 10))
                if key in seen:
                    continue
                seen.add(key)
                params: dict[str, Any] = {"cut": float(cut), "slope": float(cfg.gate_slope)}
                if str(family) == "piecewise_hinge":
                    params["direction"] = "positive"
                if str(family) == "piecewise":
                    params["left_mode"] = str(cfg.piecewise_left_mode)
                    params["right_mode"] = str(cfg.piecewise_right_mode)
                specs.append(
                    ConditionalPrimitiveSpec(
                        name=f"orth_{family}_{feature_name}_{int(round(float(quantile) * 100.0))}",
                        family=str(family),
                        source_features=(str(feature_name),),
                        parameters=params,
                    )
                )
    return tuple(specs)


def _screen_candidate_pool(
    *,
    candidates: Sequence[Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    candidate_limit: int,
    cfg: OrthogonalBasisSearchConfig,
    graph_cache: ExpressionGraphCache | None,
    consensus_prior_model: Mapping[str, Any] | None = None,
    interference_context: Mapping[str, Any] | None = None,
    periodic_context: Mapping[str, Any] | None = None,
) -> tuple[list[ScreenedCandidate], np.ndarray]:
    x_train = np.asarray(X_train, dtype=float)
    y_train_flat = np.asarray(y_train, dtype=float).reshape(-1)
    baseline_fit = _ridge_projection(np.zeros((int(x_train.shape[0]), 0), dtype=float), y_train_flat, l2_value=1e-6)
    resolved_feature_names = tuple(str(value) for value in tuple(feature_names))
    mechanistic_groups = _configured_mechanistic_feature_groups(
        cfg=cfg,
        feature_names=resolved_feature_names,
    )
    mechanistic_group_feature_indices = _configured_mechanistic_group_feature_indices(
        cfg=cfg,
        feature_names=resolved_feature_names,
    )
    configured_gate_feature_names = _configured_gate_feature_names(
        cfg=cfg,
        feature_names=resolved_feature_names,
    )
    raw_rows: list[dict[str, Any]] = []
    seen_expr: set[str] = set()

    def _append_raw_row(
        *,
        pool_index: int,
        name: str,
        expr: Mapping[str, Any],
        family: str,
        complexity: float,
        features: Sequence[int],
    ) -> bool:
        normalized_expr = dict(expr)
        expr_key = _candidate_expr_key(normalized_expr)
        if expr_key in seen_expr:
            return False
        basis_row = _single_basis_row(name=str(name), expr=normalized_expr, feature_names=feature_names)
        annotated_entry = _annotated_candidate_entry(
            name=str(name),
            expr=normalized_expr,
            feature_names=feature_names,
        )
        values = design_matrix_for_genome(
            [{"name": str(name), "expr": normalized_expr}],
            x_train,
            graph_cache=graph_cache,
            batch_key=f"orthogonal_screen::{pool_index}::{len(raw_rows)}",
        ).reshape(-1)
        if values.shape[0] != x_train.shape[0]:
            return False
        if not np.all(np.isfinite(values)):
            return False
        std = float(np.std(values, ddof=0))
        if std <= 1e-10:
            return False
        target_corr = float(abs(_safe_corr(values, y_train_flat)))
        single_fit = _ridge_projection(values, y_train_flat, l2_value=1e-6)
        residual_gain = max(0.0, float(single_fit["r2"]) - float(baseline_fit["r2"]))
        raw_rows.append(
            {
                "pool_index": int(pool_index),
                "name": str(name),
                "expr": normalized_expr,
                "family": str(family),
                "complexity": float(complexity),
                "features": tuple(int(value) for value in tuple(features)),
                "target_corr": float(target_corr),
                "residual_gain": float(residual_gain),
                "expression": expression_to_string(normalized_expr, precision=8),
                "semantic_signature": str(basis_row.get("semantic_signature", "")),
                "semantic_family": str(basis_row.get("semantic_family", family)),
                "uses_piecewise_gate": bool(basis_row.get("uses_piecewise_gate")),
                "annotated_entry": dict(annotated_entry),
                "values": np.asarray(values, dtype=float).reshape(-1),
            }
        )
        seen_expr.add(expr_key)
        return True

    for pool_index, candidate in enumerate(tuple(candidates)):
        _append_raw_row(
            pool_index=int(pool_index),
            name=str(candidate.name),
            expr=dict(candidate.expr),
            family=str(candidate.family),
            complexity=float(candidate.complexity),
            features=tuple(int(value) for value in tuple(candidate.features)),
        )

    if _rational_template_pinning_enabled(cfg):
        synthetic_pool_index = int(len(tuple(candidates)))
        for group_indices in tuple(mechanistic_group_feature_indices):
            if len(group_indices) != 2:
                continue
            ratio_expr = _build_native_mechanistic_group_expr(group_indices)
            _append_raw_row(
                pool_index=synthetic_pool_index,
                name=f"orth_rational_template_{int(group_indices[0])}_over_{int(group_indices[1])}",
                expr=ratio_expr,
                family="ratio",
                complexity=2.0,
                features=tuple(group_indices),
            )
            synthetic_pool_index += 1

    if _support_expansion_enabled(cfg):
        synthetic_pool_index = int(len(tuple(candidates)) + 1000)
        for group_indices in tuple(mechanistic_group_feature_indices):
            if len(group_indices) < 3:
                continue
            support_expr = _build_native_mechanistic_group_expr(group_indices)
            _append_raw_row(
                pool_index=synthetic_pool_index,
                name="orth_support_expansion_" + "_".join(str(int(index)) for index in tuple(group_indices)),
                expr=support_expr,
                family="interaction",
                complexity=float(max(3, len(group_indices) + 1)),
                features=tuple(group_indices),
            )
            synthetic_pool_index += 1

    semantic_signature_counts = Counter(str(row["semantic_signature"]) for row in raw_rows)
    semantic_family_counts = Counter(str(row["semantic_family"]) for row in raw_rows)
    rows: list[tuple[ScreenedCandidate, np.ndarray]] = []
    for row in raw_rows:
        source_view = (
            _candidate_source_object_view(
                ScreenedCandidate(
                    pool_index=int(row["pool_index"]),
                    screen_index=-1,
                    name=str(row["name"]),
                    expr=dict(row["expr"]),
                    family=str(row["family"]),
                    complexity=float(row["complexity"]),
                    features=tuple(int(value) for value in tuple(row["features"])),
                    target_corr=float(row["target_corr"]),
                    screen_score=0.0,
                    expression=str(row["expression"]),
                    semantic_signature=str(row["semantic_signature"]),
                    semantic_family=str(row["semantic_family"]),
                    uses_piecewise_gate=bool(row["uses_piecewise_gate"]),
                    residual_gain=float(row["residual_gain"]),
                )
            )
            if not bool(row["uses_piecewise_gate"])
            else {
                "source_expr": _normalized_expr_tree(dict(row["expr"])),
                "source_object_key": _candidate_expr_key(_normalized_expr_tree(dict(row["expr"]))),
                "chart_signature": "regional_branch",
                "chart_metadata": {"regional_branch": True, "is_identity_chart": False},
                "realization_head_ops": tuple(),
                "realization_head_signature": "",
            }
        )
        semantic_novelty = _semantic_novelty_score(
            semantic_signature=str(row["semantic_signature"]),
            semantic_family=str(row["semantic_family"]),
            signature_counts=semantic_signature_counts,
            family_counts=semantic_family_counts,
            uses_piecewise_gate=bool(row["uses_piecewise_gate"]),
        )
        consensus_prior = _consensus_prior_score(
            candidate_entry=dict(row["annotated_entry"]),
            consensus_prior_model=consensus_prior_model,
        )
        mechanistic_prior = _candidate_mechanistic_prior_score(
            feature_indices=tuple(row["features"]),
            feature_names=resolved_feature_names,
            mechanistic_groups=mechanistic_groups,
        )
        periodic_summary = _candidate_periodic_summary(
            name=str(row["name"]),
            expr=dict(row["expr"]),
            semantic_family=str(row["semantic_family"]),
            uses_piecewise_gate=bool(row["uses_piecewise_gate"]),
            feature_indices=tuple(row["features"]),
            candidate_values=np.asarray(row["values"], dtype=float).reshape(-1),
            target=y_train_flat,
            feature_names=resolved_feature_names,
            periodic_context=dict(periodic_context or {}),
            cfg=cfg,
        )
        periodic_prior = float(periodic_summary.get("periodic_prior", 0.0) or 0.0)
        periodic_penalty = float(periodic_summary.get("periodic_penalty", 0.0) or 0.0)
        contains_periodic_evidence = bool(
            periodic_summary.get("periodic_family")
            or (
                _candidate_has_periodic_object_evidence(
                    candidate=ScreenedCandidate(
                        pool_index=int(row["pool_index"]),
                        screen_index=-1,
                        name=str(row["name"]),
                        expr=dict(row["expr"]),
                        family=str(row["family"]),
                        complexity=float(row["complexity"]),
                        features=tuple(int(value) for value in tuple(row["features"])),
                        target_corr=float(row["target_corr"]),
                        screen_score=0.0,
                        expression=str(row["expression"]),
                        semantic_signature=str(row["semantic_signature"]),
                        semantic_family=str(row["semantic_family"]),
                        uses_piecewise_gate=bool(row["uses_piecewise_gate"]),
                        residual_gain=float(row["residual_gain"]),
                    ),
                    feature_names=resolved_feature_names,
                    periodic_context=dict(periodic_context or {}),
                )
            )
        )
        support_indices = _source_support_indices(dict(source_view.get("source_expr", {})))
        source_support_key = _source_support_key_from_indices(support_indices)
        source_support_size = int(len(support_indices))
        native_trunk_root = bool(_expr_is_native_trunk_root(dict(source_view.get("source_expr", {}))))
        native_interval_summary = _native_trunk_interval_gain_summary(
            candidate_values=np.asarray(row["values"], dtype=float).reshape(-1),
            target=y_train_flat,
        )
        regime_penetration_summary = _regime_penetration_summary(
            candidate_values=np.asarray(row["values"], dtype=float).reshape(-1),
            target=y_train_flat,
            raw_X=np.asarray(x_train, dtype=float),
            feature_indices=tuple(row["features"]),
            feature_names=resolved_feature_names,
            gate_feature_names=configured_gate_feature_names,
            cfg=cfg,
        )
        native_trunk_floor_passed = bool(
            native_trunk_root
            and not bool(row["uses_piecewise_gate"])
            and float(row["residual_gain"]) >= float(cfg.native_trunk_residual_gain_floor)
            and float(native_interval_summary.get("min_gain", 0.0) or 0.0)
            >= float(cfg.native_trunk_interval_gain_floor)
        )
        global_uniform_candidate = bool(
            _candidate_is_global_uniform_source(
                source_expr=dict(source_view.get("source_expr", {})),
                realization_head_signature=str(source_view.get("realization_head_signature") or ""),
                uses_piecewise_gate=bool(row["uses_piecewise_gate"]),
                contains_periodic_evidence=bool(contains_periodic_evidence),
            )
        )
        modulated_branch_candidate = bool(
            _candidate_is_modulated_branch_source(
                source_expr=dict(source_view.get("source_expr", {})),
                realization_head_signature=str(source_view.get("realization_head_signature") or ""),
                uses_piecewise_gate=bool(row["uses_piecewise_gate"]),
                contains_periodic_evidence=bool(contains_periodic_evidence),
            )
        )
        support_expansion_tagged = bool(
            _support_expansion_enabled(cfg)
            and bool(global_uniform_candidate)
            and int(source_support_size) >= 3
            and (
                float(mechanistic_prior) > 0.0
                or int(source_support_size) >= 3
                or bool(str(row["name"]).startswith("orth_support_expansion_"))
            )
        )
        canonical_trunk_tagged = bool(
            _canonical_trunk_lane_enabled(cfg)
            and _candidate_is_canonical_trunk_source(
                source_expr=dict(source_view.get("source_expr", {})),
                realization_head_signature=str(source_view.get("realization_head_signature") or ""),
                uses_piecewise_gate=bool(row["uses_piecewise_gate"]),
                contains_periodic_evidence=bool(contains_periodic_evidence),
                source_support_size=int(source_support_size),
                mechanistic_prior=float(mechanistic_prior),
                rational_template_pinned=bool(str(row["name"]).startswith("orth_rational_template_")),
            )
        )
        same_source_surrogate_tagged = bool(
            _same_source_surrogate_lane_enabled(cfg)
            and _candidate_is_same_source_surrogate_source(
                source_expr=dict(source_view.get("source_expr", {})),
                uses_piecewise_gate=bool(row["uses_piecewise_gate"]),
                contains_periodic_evidence=bool(contains_periodic_evidence),
                source_support_size=int(source_support_size),
                mechanistic_prior=float(mechanistic_prior),
                modulated_branch_candidate=bool(modulated_branch_candidate),
                canonical_trunk_candidate=bool(canonical_trunk_tagged),
            )
        )
        support_expansion_candidate = bool(
            bool(support_expansion_tagged)
            and bool(native_trunk_floor_passed)
            and (
                float(mechanistic_prior) > 0.0
                or float(row["residual_gain"]) >= float(cfg.native_trunk_residual_gain_floor)
            )
        )
        canonical_trunk_candidate = bool(bool(canonical_trunk_tagged) and bool(native_trunk_floor_passed))
        same_source_surrogate_candidate = bool(same_source_surrogate_tagged)
        heterogeneous_exposure_eligible = bool(
            not bool(row["uses_piecewise_gate"])
            and float(regime_penetration_summary.get("score", 0.0) or 0.0)
            >= float(cfg.heterogeneous_exposure_min_score)
            and float(regime_penetration_summary.get("min_gain", 0.0) or 0.0)
            >= float(cfg.regime_penetration_gain_floor)
            and float(regime_penetration_summary.get("sign_consistency", 0.0) or 0.0) >= 0.75
        )
        structural_channel = (
            "regional_speciality"
            if bool(row["uses_piecewise_gate"])
            else (
                "support_expansion"
                if bool(support_expansion_tagged)
                else (
                    "canonical_trunk"
                    if bool(canonical_trunk_tagged)
                    else (
                        "native_trunk"
                        if bool(native_trunk_root) and bool(global_uniform_candidate)
                        else (
                            "same_source_surrogate"
                            if bool(same_source_surrogate_tagged)
                            else ("heterogeneous_exposure" if heterogeneous_exposure_eligible else "challenger")
                        )
                    )
                )
            )
        )
        selection_channel = (
            "regional_speciality"
            if bool(row["uses_piecewise_gate"])
            else (
                "support_expansion"
                if bool(support_expansion_candidate)
                else (
                    "canonical_trunk"
                    if bool(canonical_trunk_candidate)
                    else (
                        "native_trunk"
                        if native_trunk_floor_passed
                        else (
                            "same_source_surrogate"
                            if bool(same_source_surrogate_candidate)
                            else ("heterogeneous_exposure" if heterogeneous_exposure_eligible else "challenger")
                        )
                    )
                )
            )
        )
        native_structure_score = _candidate_native_structure_score(
            ScreenedCandidate(
                pool_index=int(row["pool_index"]),
                screen_index=-1,
                name=str(row["name"]),
                expr=dict(row["expr"]),
                family=str(row["family"]),
                complexity=float(row["complexity"]),
                features=tuple(int(value) for value in tuple(row["features"])),
                target_corr=float(row["target_corr"]),
                screen_score=0.0,
                expression=str(row["expression"]),
                semantic_signature=str(row["semantic_signature"]),
                semantic_family=str(row["semantic_family"]),
                uses_piecewise_gate=bool(row["uses_piecewise_gate"]),
                residual_gain=float(row["residual_gain"]),
                chart_signature=str(source_view.get("chart_signature") or "identity"),
                realization_head_signature=str(source_view.get("realization_head_signature") or ""),
                chart_metadata=dict(source_view.get("chart_metadata", {}) or {}),
            )
        )
        raw_screen_score = (
            float(cfg.screen_target_corr_weight) * float(row["target_corr"])
            + float(cfg.screen_residual_gain_weight) * float(row["residual_gain"])
            + float(cfg.screen_semantic_novelty_weight) * float(semantic_novelty)
            + float(cfg.screen_consensus_prior_weight) * float(consensus_prior)
            + float(cfg.mechanistic_screen_bonus) * float(mechanistic_prior)
            + float(cfg.native_structure_screen_bonus) * float(native_structure_score)
            + float(cfg.periodic_family_prior_weight) * float(periodic_prior)
            - float(periodic_penalty)
        )
        screen_score = float(raw_screen_score / (1.0 + float(cfg.screen_complexity_penalty) * float(row["complexity"])))
        screened_candidate = ScreenedCandidate(
            pool_index=int(row["pool_index"]),
            screen_index=-1,
            name=str(row["name"]),
            expr=dict(row["expr"]),
            family=str(row["family"]),
            complexity=float(row["complexity"]),
            features=tuple(int(value) for value in tuple(row["features"])),
            target_corr=float(row["target_corr"]),
            screen_score=float(screen_score),
            expression=str(row["expression"]),
            semantic_signature=str(row["semantic_signature"]),
            semantic_family=str(row["semantic_family"]),
            uses_piecewise_gate=bool(row["uses_piecewise_gate"]),
            residual_gain=float(row["residual_gain"]),
            semantic_novelty=float(semantic_novelty),
            consensus_prior=float(consensus_prior),
            mechanistic_prior=float(mechanistic_prior),
            periodic_prior=float(periodic_prior),
            periodic_penalty=float(periodic_penalty),
            source_object_key=str(source_view.get("source_object_key") or ""),
            source_support_key=str(source_support_key),
            source_support_size=int(source_support_size),
            chart_signature=str(source_view.get("chart_signature") or "identity"),
            realization_head_signature=str(source_view.get("realization_head_signature") or ""),
            chart_metadata=dict(source_view.get("chart_metadata", {}) or {}),
            native_structure_score=float(native_structure_score),
            native_trunk_root=bool(native_trunk_root),
            native_trunk_floor_passed=bool(native_trunk_floor_passed),
            native_trunk_global_gain=float(row["residual_gain"]),
            native_trunk_interval_min_gain=float(native_interval_summary.get("min_gain", 0.0) or 0.0),
            native_trunk_interval_mean_gain=float(native_interval_summary.get("mean_gain", 0.0) or 0.0),
            regime_penetration_score=float(regime_penetration_summary.get("score", 0.0) or 0.0),
            regime_penetration_min_gain=float(regime_penetration_summary.get("min_gain", 0.0) or 0.0),
            regime_penetration_mean_gain=float(regime_penetration_summary.get("mean_gain", 0.0) or 0.0),
            regime_sign_consistency=float(regime_penetration_summary.get("sign_consistency", 0.0) or 0.0),
            heterogeneous_exposure_eligible=bool(heterogeneous_exposure_eligible),
            support_expansion_tagged=bool(support_expansion_tagged),
            canonical_trunk_tagged=bool(canonical_trunk_tagged),
            same_source_surrogate_tagged=bool(same_source_surrogate_tagged),
            support_expansion_candidate=bool(support_expansion_candidate),
            canonical_trunk_candidate=bool(canonical_trunk_candidate),
            same_source_surrogate_candidate=bool(same_source_surrogate_candidate),
            global_uniform_candidate=bool(global_uniform_candidate),
            modulated_branch_candidate=bool(modulated_branch_candidate),
            rational_template_pinned=bool(str(row["name"]).startswith("orth_rational_template_")),
            structural_channel=str(structural_channel),
            selection_channel=str(selection_channel),
        )
        screened_candidate = replace(
            screened_candidate,
            information_source_key=str(
                screened_candidate.source_object_key or _candidate_information_source_key(screened_candidate)
            ),
            contains_periodic_evidence=bool(contains_periodic_evidence),
        )
        rows.append(
            (
                screened_candidate,
                np.asarray(row["values"], dtype=float).reshape(-1),
            )
        )

    rows = _compress_screen_information_clusters(
        rows=rows,
        raw_X=np.asarray(x_train, dtype=float),
        feature_names=resolved_feature_names,
        interference_context=dict(interference_context or {}),
        periodic_context=dict(periodic_context or {}),
    )
    rows.sort(
        key=lambda item: (
            -float(item[0].screen_score),
            -float(item[0].mechanistic_prior),
            -float(item[0].consensus_prior),
            -float(item[0].residual_gain),
            float(item[0].complexity),
            str(item[0].name),
        )
    )
    limited = _reserve_native_trunk_candidates(
        ranked_rows=rows,
        full_ranked_rows=rows,
        candidate_limit=int(candidate_limit),
        reserve_count=max(
            int(cfg.native_trunk_candidate_screen_reserve),
            int(_required_native_trunk_basis_terms(cfg)),
        ),
    )
    limited = _reserve_support_expansion_candidates(
        ranked_rows=limited,
        full_ranked_rows=rows,
        candidate_limit=int(candidate_limit),
        reserve_count=max(
            int(cfg.support_expansion_candidate_screen_reserve),
            int(_required_support_expansion_basis_terms(cfg)),
        ),
        cfg=cfg,
    )
    limited = _reserve_canonical_trunk_candidates(
        ranked_rows=limited,
        full_ranked_rows=rows,
        candidate_limit=int(candidate_limit),
        reserve_count=max(
            int(cfg.canonical_trunk_candidate_screen_reserve),
            int(_required_canonical_trunk_basis_terms(cfg)),
        ),
        cfg=cfg,
    )
    limited = _reserve_gate_candidates(
        ranked_rows=limited,
        candidate_limit=int(candidate_limit),
        reserve_count=max(
            int(cfg.gate_candidate_screen_reserve),
            int(_required_gate_basis_terms(cfg)),
        ),
    )
    limited = _reserve_periodic_candidates(
        ranked_rows=limited,
        full_ranked_rows=rows,
        reserve_count=max(
            int(cfg.periodic_candidate_screen_reserve),
            int(_required_periodic_basis_terms(cfg=cfg, periodic_context=dict(periodic_context or {}))),
        ),
    )
    limited = _reserve_heterogeneous_exposure_candidates(
        ranked_rows=limited,
        full_ranked_rows=rows,
        candidate_limit=int(candidate_limit),
        reserve_count=int(cfg.heterogeneous_exposure_candidate_screen_reserve),
        cfg=cfg,
    )
    limited = _enforce_proxy_representative_screen(
        limited_rows=limited,
        full_ranked_rows=rows,
        candidate_limit=int(candidate_limit),
        feature_names=resolved_feature_names,
        interference_context=dict(interference_context or {}),
        cfg=cfg,
        periodic_context=dict(periodic_context or {}),
    )
    screened: list[ScreenedCandidate] = []
    value_rows: list[np.ndarray] = []
    for screen_index, (row, values) in enumerate(limited):
        screened.append(
            ScreenedCandidate(
                pool_index=int(row.pool_index),
                screen_index=int(screen_index),
                name=str(row.name),
                expr=dict(row.expr),
                family=str(row.family),
                complexity=float(row.complexity),
                features=tuple(int(value) for value in row.features),
                target_corr=float(row.target_corr),
                screen_score=float(row.screen_score),
                expression=str(row.expression),
                semantic_signature=str(row.semantic_signature),
                semantic_family=str(row.semantic_family),
                uses_piecewise_gate=bool(row.uses_piecewise_gate),
                residual_gain=float(row.residual_gain),
                semantic_novelty=float(row.semantic_novelty),
                consensus_prior=float(row.consensus_prior),
                mechanistic_prior=float(row.mechanistic_prior),
                periodic_prior=float(row.periodic_prior),
                periodic_penalty=float(row.periodic_penalty),
                screen_cluster_key=str(row.screen_cluster_key),
                screen_cluster_size=int(row.screen_cluster_size),
                information_source_key=str(row.information_source_key),
                contains_periodic_evidence=bool(row.contains_periodic_evidence),
                source_object_key=str(row.source_object_key),
                source_support_key=str(row.source_support_key),
                source_support_size=int(row.source_support_size),
                chart_signature=str(row.chart_signature),
                realization_head_signature=str(row.realization_head_signature),
                chart_metadata=dict(row.chart_metadata),
                native_structure_score=float(row.native_structure_score),
                native_trunk_root=bool(row.native_trunk_root),
                native_trunk_floor_passed=bool(row.native_trunk_floor_passed),
                native_trunk_global_gain=float(row.native_trunk_global_gain),
                native_trunk_interval_min_gain=float(row.native_trunk_interval_min_gain),
                native_trunk_interval_mean_gain=float(row.native_trunk_interval_mean_gain),
                regime_penetration_score=float(row.regime_penetration_score),
                regime_penetration_min_gain=float(row.regime_penetration_min_gain),
                regime_penetration_mean_gain=float(row.regime_penetration_mean_gain),
                regime_sign_consistency=float(row.regime_sign_consistency),
                heterogeneous_exposure_eligible=bool(row.heterogeneous_exposure_eligible),
                support_expansion_tagged=bool(row.support_expansion_tagged),
                canonical_trunk_tagged=bool(row.canonical_trunk_tagged),
                same_source_surrogate_tagged=bool(row.same_source_surrogate_tagged),
                support_expansion_candidate=bool(row.support_expansion_candidate),
                canonical_trunk_candidate=bool(row.canonical_trunk_candidate),
                same_source_surrogate_candidate=bool(row.same_source_surrogate_candidate),
                global_uniform_candidate=bool(row.global_uniform_candidate),
                modulated_branch_candidate=bool(row.modulated_branch_candidate),
                rational_template_pinned=bool(row.rational_template_pinned),
                structural_channel=str(row.structural_channel),
                selection_channel=str(row.selection_channel),
            )
        )
        value_rows.append(np.asarray(values, dtype=float).reshape(-1))
    if not value_rows:
        return [], np.zeros((int(x_train.shape[0]), 0), dtype=float)
    return screened, np.asarray(np.stack(value_rows, axis=1), dtype=float)


def _standardize_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr, axis=0, ddof=0, keepdims=True)
    std = np.where(std <= 1e-12, 1.0, std)
    return np.asarray((arr - mean) / std, dtype=float)


def _pairwise_abs_corr(matrix: np.ndarray) -> np.ndarray:
    arr = _standardize_matrix(matrix)
    if arr.shape[1] <= 1:
        return np.zeros((int(arr.shape[1]), int(arr.shape[1])), dtype=float)
    corr = np.corrcoef(arr, rowvar=False)
    corr = np.nan_to_num(np.asarray(corr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return np.abs(corr)


def _group_feature_overlap_mean(rows: Sequence[ScreenedCandidate]) -> float:
    if len(rows) <= 1:
        return 0.0
    values: list[float] = []
    for index, left in enumerate(tuple(rows)):
        left_features = set(int(value) for value in left.features)
        for right in tuple(rows)[index + 1 :]:
            right_features = set(int(value) for value in right.features)
            union = left_features | right_features
            overlap = left_features & right_features
            values.append(0.0 if not union else float(len(overlap)) / float(len(union)))
    return float(np.mean(values)) if values else 0.0


def _orthogonality_metrics(
    *,
    selected_rows: Sequence[ScreenedCandidate],
    train_values: np.ndarray,
) -> dict[str, Any]:
    matrix = np.asarray(train_values, dtype=float)
    corr = _pairwise_abs_corr(matrix)
    pair_values = corr[np.triu_indices_from(corr, k=1)] if corr.size > 0 else np.asarray([], dtype=float)
    std_matrix = _standardize_matrix(matrix)
    singular = np.linalg.svd(std_matrix, compute_uv=False) if std_matrix.size > 0 else np.asarray([], dtype=float)
    singular = np.asarray(singular, dtype=float)
    singular = singular[np.isfinite(singular)]
    condition = 1.0
    if singular.size > 0:
        floor = float(max(1e-12, np.min(singular)))
        ceiling = float(max(floor, np.max(singular)))
        condition = float(max(1.0, ceiling / floor))
    rank = int(np.linalg.matrix_rank(std_matrix)) if std_matrix.size > 0 else 0
    pair_abs_corr_mean = float(np.mean(pair_values)) if pair_values.size > 0 else 0.0
    pair_abs_corr_max = float(np.max(pair_values)) if pair_values.size > 0 else 0.0
    feature_overlap_mean = float(_group_feature_overlap_mean(selected_rows))
    target_corr_mean = float(np.mean([float(row.target_corr) for row in selected_rows])) if selected_rows else 0.0
    target_corr_max = float(np.max([float(row.target_corr) for row in selected_rows])) if selected_rows else 0.0
    condition_penalty = 0.0
    if math.isfinite(condition):
        condition_penalty = max(0.0, math.log1p(max(0.0, condition - 1.0)))
    orthogonality_score = float(
        1.0
        / (
            1.0
            + pair_abs_corr_mean
            + 0.25 * pair_abs_corr_max
            + 0.10 * feature_overlap_mean
            + 0.02 * condition_penalty
        )
    )
    return {
        "basis_count": int(matrix.shape[1]) if matrix.ndim == 2 else int(len(selected_rows)),
        "pair_abs_corr_mean": float(pair_abs_corr_mean),
        "pair_abs_corr_max": float(pair_abs_corr_max),
        "feature_overlap_mean": float(feature_overlap_mean),
        "condition_number": float(condition),
        "effective_rank": int(rank),
        "mean_target_abs_corr": float(target_corr_mean),
        "max_target_abs_corr": float(target_corr_max),
        "orthogonality_score": float(orthogonality_score),
    }


def _selected_matrix(train_matrix: np.ndarray, selected_rows: Sequence[ScreenedCandidate]) -> np.ndarray:
    if not selected_rows:
        return np.zeros((int(np.asarray(train_matrix).shape[0]), 0), dtype=float)
    return np.asarray(train_matrix[:, [int(row.screen_index) for row in tuple(selected_rows)]], dtype=float)


def _unique_basis_feature_names(selected_rows: Sequence[ScreenedCandidate]) -> tuple[str, ...]:
    used: set[str] = set()
    out: list[str] = []
    for index, row in enumerate(tuple(selected_rows)):
        base = str(row.name or f"basis_{index}").strip() or f"basis_{index}"
        candidate = str(base)
        suffix = 2
        while candidate in used:
            candidate = f"{base}__{suffix}"
            suffix += 1
        used.add(candidate)
        out.append(candidate)
    return tuple(out)


def _realization_prior_injection_enabled(cfg: OrthogonalBasisSearchConfig) -> bool:
    return str(cfg.realization_prior_injection_mode or "off").strip().lower() not in {
        "",
        "off",
        "none",
        "disabled",
    }


def _mandatory_realization_closure_enabled(cfg: OrthogonalBasisSearchConfig) -> bool:
    return str(cfg.mandatory_realization_closure_mode or "explicit_evidence_competition").strip().lower() not in {
        "",
        "off",
        "none",
        "disabled",
    }


def _periodic_realization_competition_enabled(cfg: OrthogonalBasisSearchConfig) -> bool:
    return str(cfg.periodic_realization_competition_mode or "off").strip().lower() not in {
        "",
        "off",
        "none",
        "disabled",
    }


def _named_expression_for_expr(expr: Mapping[str, Any], *, raw_feature_names: Sequence[str]) -> str:
    rows = build_basis_term_rows(
        ({"name": "basis_object", "expr": dict(expr)},),
        feature_names=tuple(str(value) for value in tuple(raw_feature_names)),
        scope="global",
    )
    if rows:
        row = dict(rows[0])
        return str(row.get("expression") or row.get("term_name") or expression_to_string(expr))
    return expression_to_string(expr)


def _realization_signature_expr(signature: str, *, source_expr: Mapping[str, Any]) -> dict[str, Any]:
    if str(signature) == "unary:exp_neg":
        return _unary_expr("exp", _binary_expr("mul", _const_expr(-1.0), dict(source_expr)))
    if str(signature).startswith("unary:"):
        return _unary_expr(str(signature).split(":", 1)[1], dict(source_expr))
    return dict(source_expr)


def _realization_signature_key(signature: str) -> str:
    return str(signature or "identity").replace(":", "_").replace("/", "_")


def _branch_signature_key(signature: str, *, cut: float | None = None) -> str:
    base = str(signature or "branch").replace(":", "_").replace("/", "_")
    if cut is None or not np.isfinite(float(cut)):
        return base
    cut_text = str(round(float(cut), 8)).replace("-", "neg").replace(".", "p")
    return f"{base}_{cut_text}"


def _regional_branch_candidate_cuts(feature_values: np.ndarray, *, cfg: OrthogonalBasisSearchConfig) -> tuple[float, ...]:
    x = np.asarray(feature_values, dtype=float).reshape(-1)
    finite = x[np.isfinite(x)]
    if finite.size < 8:
        return tuple()
    quantiles = sorted(
        {
            float(np.clip(value, 0.05, 0.95))
            for value in (
                *tuple(cfg.gate_quantiles),
                *tuple(cfg.assembler_hinge_quantiles),
                0.10,
                0.20,
                0.35,
                0.50,
                0.65,
                0.80,
                0.90,
            )
            if np.isfinite(float(value))
        }
    )
    cuts: list[float] = []
    seen: set[float] = set()
    for quantile in quantiles:
        cut = float(np.quantile(finite, float(quantile)))
        if not np.isfinite(cut):
            continue
        rounded = round(cut, 10)
        if rounded in seen:
            continue
        seen.add(rounded)
        cuts.append(float(cut))
    return tuple(cuts)


def _regional_branch_expr_and_values(
    *,
    feature_index: int,
    feature_values: np.ndarray,
    cut: float,
    direction: str,
) -> tuple[str, dict[str, Any], np.ndarray]:
    base_expr = _feature_expr(int(feature_index))
    base_values = np.asarray(feature_values, dtype=float).reshape(-1)
    direction_key = str(direction or "pos").strip().lower()
    if direction_key in {"neg", "negative", "left"}:
        shifted_expr = _binary_expr("sub", _const_expr(float(cut)), base_expr)
        shifted_values = np.asarray(float(cut) - base_values, dtype=float)
        return "branch:hinge_neg", _relu_expr(shifted_expr), np.asarray(np.maximum(0.0, shifted_values), dtype=float)
    shifted_expr = _binary_expr("sub", base_expr, _const_expr(float(cut)))
    shifted_values = np.asarray(base_values - float(cut), dtype=float)
    return "branch:hinge_pos", _relu_expr(shifted_expr), np.asarray(np.maximum(0.0, shifted_values), dtype=float)


def _regional_branch_threshold_audit(
    *,
    feature_values: np.ndarray,
    branch_values: np.ndarray,
    base_matrix: np.ndarray,
    target: np.ndarray,
    cut: float,
    full_gain: float,
    augmented_fit: Mapping[str, Any],
    cfg: OrthogonalBasisSearchConfig,
) -> dict[str, Any]:
    x = np.asarray(feature_values, dtype=float).reshape(-1)
    h = np.asarray(branch_values, dtype=float).reshape(-1)
    y = np.asarray(target, dtype=float).reshape(-1)
    base = np.asarray(base_matrix, dtype=float)
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    if base.ndim != 2 or base.shape[0] != y.shape[0]:
        base = np.zeros((y.shape[0], 0), dtype=float)
    finite = np.isfinite(x) & np.isfinite(h) & np.isfinite(y)
    n = int(np.sum(finite))
    if n < 12:
        return {
            "threshold_orthodoxy_score": 0.0,
            "threshold_stability_score": 0.0,
            "threshold_balance_score": 0.0,
            "active_fraction": 0.0,
            "cross_split_min_gain": 0.0,
            "cross_split_gain_consistency": 0.0,
            "coefficient_sign_consistency": 0.0,
            "coefficient_cv_score": 0.0,
            "folds": [],
        }
    xv = x[finite]
    hv = h[finite]
    yv = y[finite]
    basev = base[finite, :] if base.shape[0] == finite.shape[0] else np.zeros((n, 0), dtype=float)
    left_count = int(np.sum(xv <= float(cut)))
    right_count = int(np.sum(xv > float(cut)))
    threshold_balance_score = float(2.0 * min(left_count, right_count) / max(1, n))
    active_fraction = float(np.mean(hv > 1e-12))
    active_balance_score = float(2.0 * min(active_fraction, 1.0 - active_fraction))
    order = np.argsort(xv)
    split_indices = (order[::2], order[1::2])
    fold_rows: list[dict[str, Any]] = []
    coefs: list[float] = []
    gains: list[float] = []
    l2_value = float(min(tuple(cfg.l2_grid)) if tuple(cfg.l2_grid) else 1e-6)
    for split_id, indices in enumerate(split_indices):
        idx = np.asarray(indices, dtype=int)
        if idx.size < 8:
            continue
        fold_h = hv[idx]
        active_count = int(np.sum(fold_h > 1e-12))
        inactive_count = int(fold_h.size - active_count)
        if min(active_count, inactive_count) < 3:
            continue
        fold_base = basev[idx, :]
        fold_y = yv[idx]
        fold_aug = np.asarray(
            np.concatenate([fold_base, fold_h.reshape(-1, 1)], axis=1)
            if fold_base.ndim == 2 and fold_base.shape[1] > 0
            else fold_h.reshape(-1, 1),
            dtype=float,
        )
        base_fit = _ridge_projection(fold_base, fold_y, l2_value=l2_value)
        aug_fit = _ridge_projection(fold_aug, fold_y, l2_value=l2_value)
        gain = float(aug_fit.get("r2", 0.0) or 0.0) - float(base_fit.get("r2", 0.0) or 0.0)
        weights = np.asarray(aug_fit.get("weight", ()), dtype=float).reshape(-1)
        coef = float(weights[-1]) if weights.size else 0.0
        gains.append(float(gain))
        coefs.append(float(coef))
        fold_rows.append(
            {
                "split_id": int(split_id),
                "sample_count": int(idx.size),
                "active_count": int(active_count),
                "inactive_count": int(inactive_count),
                "r2_gain": float(gain),
                "branch_coefficient": float(coef),
            }
        )
    if not fold_rows:
        cross_split_min_gain = 0.0
        gain_consistency = 0.0
        sign_consistency = 0.0
        coefficient_cv_score = 0.0
    else:
        gains_arr = np.asarray(gains, dtype=float)
        coefs_arr = np.asarray(coefs, dtype=float)
        gain_floor = max(1e-6, 0.05 * max(0.0, float(full_gain)))
        cross_split_min_gain = float(np.min(gains_arr))
        gain_consistency = float(np.mean(gains_arr >= float(gain_floor)))
        nonzero_coefs = coefs_arr[np.abs(coefs_arr) > 1e-10]
        if nonzero_coefs.size <= 0:
            sign_consistency = 0.0
            coefficient_cv_score = 0.0
        else:
            sign_consistency = float(1.0 if np.all(np.sign(nonzero_coefs) == np.sign(nonzero_coefs[0])) else 0.0)
            coef_abs = np.abs(nonzero_coefs)
            coefficient_cv_score = float(
                np.clip(1.0 - float(np.std(coef_abs)) / (float(np.mean(coef_abs)) + 1e-12), 0.0, 1.0)
            )
    weights_full = np.asarray(dict(augmented_fit).get("weight", ()), dtype=float).reshape(-1)
    full_coef = float(weights_full[-1]) if weights_full.size else 0.0
    min_gain_score = float(np.clip(max(0.0, cross_split_min_gain) / (max(1e-6, max(0.0, float(full_gain))) + 1e-12), 0.0, 1.0))
    coefficient_magnitude_score = float(np.clip(abs(full_coef) / (abs(full_coef) + 0.10), 0.0, 1.0))
    threshold_stability_score = float(
        np.clip(
            0.30 * gain_consistency
            + 0.25 * sign_consistency
            + 0.25 * coefficient_cv_score
            + 0.20 * min_gain_score,
            0.0,
            1.0,
        )
    )
    threshold_orthodoxy_score = float(
        np.clip(
            0.35 * threshold_stability_score
            + 0.25 * threshold_balance_score
            + 0.15 * active_balance_score
            + 0.15 * coefficient_magnitude_score
            + 0.10 * min_gain_score,
            0.0,
            1.0,
        )
    )
    return {
        "threshold_orthodoxy_score": float(threshold_orthodoxy_score),
        "threshold_stability_score": float(threshold_stability_score),
        "threshold_balance_score": float(threshold_balance_score),
        "active_fraction": float(active_fraction),
        "active_balance_score": float(active_balance_score),
        "cross_split_min_gain": float(cross_split_min_gain),
        "cross_split_gain_consistency": float(gain_consistency),
        "coefficient_sign_consistency": float(sign_consistency),
        "coefficient_cv_score": float(coefficient_cv_score),
        "coefficient_magnitude_score": float(coefficient_magnitude_score),
        "full_branch_coefficient": float(full_coef),
        "folds": _jsonable(fold_rows),
    }


def _metadata_regional_branch_feature_names(
    *,
    data_metadata: Mapping[str, Any] | None,
    gate_feature_names: Sequence[str],
    feature_names: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    allowed = {str(name).strip().lower(): str(name) for name in tuple(feature_names) if str(name).strip()}
    by_feature: dict[str, set[str]] = {}
    for name in tuple(gate_feature_names or ()):
        normalized = str(name).strip().lower()
        if normalized in allowed:
            by_feature.setdefault(allowed[normalized], set()).add("gate_feature_hint")
    for spec in truth_contract_specs(_metadata_truth_contract_values(data_metadata), default_match_mode="exact"):
        match_kind = str(spec.get("match_kind") or spec.get("family") or "").strip().lower()
        if match_kind not in {"piecewise_hinge", "piecewise_gate_family"}:
            continue
        ordered = tuple(str(value).strip().lower() for value in tuple(spec.get("ordered_features", ()) or ()))
        if not ordered:
            continue
        feature_name = allowed.get(str(ordered[0]))
        if not feature_name:
            continue
        by_feature.setdefault(str(feature_name), set()).add(str(spec.get("contract") or match_kind))
    return {
        str(feature): tuple(sorted(str(value) for value in protocols if str(value).strip()))
        for feature, protocols in by_feature.items()
        if str(feature).strip()
    }


def _build_regional_branch_evidence_specs(
    *,
    base_object_records: Sequence[Mapping[str, Any]],
    base_matrix: np.ndarray,
    raw_X: np.ndarray,
    target: np.ndarray,
    raw_feature_names: Sequence[str],
    gate_feature_names: Sequence[str],
    data_metadata: Mapping[str, Any] | None,
    cfg: OrthogonalBasisSearchConfig,
) -> tuple[dict[str, Any], ...]:
    if not bool(cfg.enable_piecewise_basis):
        return tuple()
    x = np.asarray(raw_X, dtype=float)
    if x.ndim != 2 or x.shape[0] <= 0:
        return tuple()
    y = np.asarray(target, dtype=float).reshape(-1)
    if y.shape[0] != x.shape[0]:
        return tuple()
    feature_names = tuple(str(value) for value in tuple(raw_feature_names))
    feature_to_index = {str(name): int(index) for index, name in enumerate(feature_names)}
    branch_feature_reasons = _metadata_regional_branch_feature_names(
        data_metadata=data_metadata,
        gate_feature_names=gate_feature_names,
        feature_names=feature_names,
    )
    if not branch_feature_reasons:
        return tuple()
    parent_by_feature: dict[str, dict[str, Any]] = {}
    for record in tuple(base_object_records):
        rec = dict(record)
        if str(rec.get("binding_role") or "locked_basis_object") not in {"", "locked_basis_object"}:
            continue
        if bool(rec.get("uses_piecewise_gate")):
            continue
        names = tuple(str(value) for value in tuple(rec.get("feature_names", ())) if str(value).strip())
        if len(names) != 1:
            continue
        name = str(names[0])
        if name not in branch_feature_reasons:
            continue
        current = parent_by_feature.get(name)
        current_score = 0 if current is None else int(str(current.get("chart_signature") or "") == "identity")
        rec_score = int(str(rec.get("chart_signature") or "") == "identity")
        if current is None or rec_score >= current_score:
            parent_by_feature[name] = rec
    if not parent_by_feature:
        return tuple()

    if np.asarray(base_matrix).ndim == 2 and int(np.asarray(base_matrix).shape[1]) > 0:
        base_fit = _ridge_projection(
            np.asarray(base_matrix, dtype=float),
            y,
            l2_value=float(min(tuple(cfg.l2_grid)) if tuple(cfg.l2_grid) else 1e-6),
        )
        base_r2 = float(base_fit.get("r2", 0.0) or 0.0)
        residual = np.asarray(base_fit.get("residual", y), dtype=float).reshape(-1)
    else:
        base_r2 = 0.0
        residual = y - float(np.mean(y))

    specs: list[dict[str, Any]] = []
    for feature_name, parent in tuple(parent_by_feature.items()):
        feature_index = feature_to_index.get(str(feature_name))
        if feature_index is None:
            continue
        values_raw = np.asarray(x[:, int(feature_index)], dtype=float).reshape(-1)
        candidates: list[dict[str, Any]] = []
        for cut in _regional_branch_candidate_cuts(values_raw, cfg=cfg):
            left_count = int(np.sum(values_raw <= float(cut)))
            right_count = int(np.sum(values_raw > float(cut)))
            if min(left_count, right_count) < max(4, int(0.05 * len(values_raw))):
                continue
            regime_score = _regional_regime_score(values_raw, residual, float(cut))
            for direction in ("pos", "neg"):
                signature, expr, branch_values = _regional_branch_expr_and_values(
                    feature_index=int(feature_index),
                    feature_values=values_raw,
                    cut=float(cut),
                    direction=str(direction),
                )
                if not np.any(np.isfinite(branch_values)) or float(np.std(branch_values)) <= 1e-12:
                    continue
                augmented_matrix = np.asarray(
                    np.concatenate([np.asarray(base_matrix, dtype=float), branch_values.reshape(-1, 1)], axis=1)
                    if np.asarray(base_matrix).ndim == 2 and int(np.asarray(base_matrix).shape[1]) > 0
                    else branch_values.reshape(-1, 1),
                    dtype=float,
                )
                augmented_fit = _ridge_projection(
                    augmented_matrix,
                    y,
                    l2_value=float(min(tuple(cfg.l2_grid)) if tuple(cfg.l2_grid) else 1e-6),
                )
                marginal_r2_gain = float(augmented_fit.get("r2", 0.0) or 0.0) - float(base_r2)
                residual_abs_corr = float(abs(_safe_corr(branch_values, residual)))
                threshold_audit = _regional_branch_threshold_audit(
                    feature_values=values_raw,
                    branch_values=branch_values,
                    base_matrix=np.asarray(base_matrix, dtype=float),
                    target=y,
                    cut=float(cut),
                    full_gain=float(marginal_r2_gain),
                    augmented_fit=augmented_fit,
                    cfg=cfg,
                )
                threshold_orthodoxy_score = float(threshold_audit.get("threshold_orthodoxy_score", 0.0) or 0.0)
                threshold_stability_score = float(threshold_audit.get("threshold_stability_score", 0.0) or 0.0)
                score = float(
                    np.clip(
                        0.42 * max(0.0, marginal_r2_gain)
                        + 0.18 * residual_abs_corr
                        + 0.12 * max(0.0, regime_score)
                        + 0.18 * threshold_orthodoxy_score
                        + 0.10 * threshold_stability_score,
                        0.0,
                        1.0,
                    )
                )
                candidates.append(
                    {
                        "parent_object_key": str(parent.get("object_key") or ""),
                        "source_information_key": str(parent.get("source_information_key") or ""),
                        "source_object_key": str(parent.get("source_object_key") or ""),
                        "feature_name": str(feature_name),
                        "feature_index": int(feature_index),
                        "branch_signature": str(signature),
                        "branch_direction": "negative" if str(direction) == "neg" else "positive",
                        "threshold": float(cut),
                        "expr": dict(expr),
                        "expression": expression_to_string(expr, precision=8),
                        "values": np.asarray(branch_values, dtype=float).reshape(-1),
                        "marginal_r2_gain": float(marginal_r2_gain),
                        "residual_abs_corr": float(residual_abs_corr),
                        "residual_regime_score": float(regime_score),
                        "branch_evidence_score": float(score),
                        "threshold_orthodoxy_score": float(threshold_orthodoxy_score),
                        "threshold_stability_score": float(threshold_stability_score),
                        "threshold_balance_score": float(threshold_audit.get("threshold_balance_score", 0.0) or 0.0),
                        "threshold_active_fraction": float(threshold_audit.get("active_fraction", 0.0) or 0.0),
                        "threshold_cross_split_min_gain": float(threshold_audit.get("cross_split_min_gain", 0.0) or 0.0),
                        "threshold_sign_consistency": float(
                            threshold_audit.get("coefficient_sign_consistency", 0.0) or 0.0
                        ),
                        "threshold_coefficient_cv_score": float(
                            threshold_audit.get("coefficient_cv_score", 0.0) or 0.0
                        ),
                        "threshold_audit": _jsonable(threshold_audit),
                        "evidence_term_names": tuple(branch_feature_reasons.get(str(feature_name), ())),
                        "branch_protocols": (
                            "RegionalBranchEvidenceRegistry",
                            "MandatoryHingeBranchClosure",
                            "ThresholdOrthodoxyScoring",
                            "ThresholdStabilityAudit",
                        ),
                    }
                )
        evidence_ranked = sorted(
            candidates,
            key=lambda item: (
                -float(item.get("branch_evidence_score", 0.0)),
                -float(item.get("marginal_r2_gain", 0.0)),
                -float(item.get("threshold_orthodoxy_score", 0.0)),
                abs(float(item.get("threshold", 0.0))),
                str(item.get("branch_signature", "")),
            ),
        )
        orthodoxy_ranked = sorted(
            candidates,
            key=lambda item: (
                -float(item.get("threshold_orthodoxy_score", 0.0)),
                -float(item.get("threshold_stability_score", 0.0)),
                -float(item.get("branch_evidence_score", 0.0)),
                abs(float(item.get("threshold", 0.0))),
                str(item.get("branch_signature", "")),
            ),
        )
        gain_ranked = sorted(
            candidates,
            key=lambda item: (
                -float(item.get("marginal_r2_gain", 0.0)),
                -float(item.get("branch_evidence_score", 0.0)),
                -float(item.get("threshold_orthodoxy_score", 0.0)),
                abs(float(item.get("threshold", 0.0))),
                str(item.get("branch_signature", "")),
            ),
        )
        keep: list[dict[str, Any]] = []
        seen_keep: set[tuple[str, float]] = set()
        for lane_name, lane in (
            ("best_evidence", evidence_ranked),
            ("best_gain", gain_ranked),
            ("best_orthodoxy", orthodoxy_ranked),
        ):
            for item in tuple(lane):
                item_key = (str(item.get("branch_signature", "")), round(float(item.get("threshold", 0.0)), 10))
                if item_key in seen_keep:
                    continue
                row = dict(item)
                row["threshold_selection_lane"] = str(lane_name)
                keep.append(row)
                seen_keep.add(item_key)
                break
        keep = keep[:2]
        # If metadata explicitly declares a hinge on this feature, preserve at least the best
        # candidate even when the current residual already explains most of the discontinuity.
        if not keep and str(feature_name) in branch_feature_reasons and candidates:
            keep = [dict(candidates[0])]
        specs.extend(keep)
    return tuple(specs)


def _build_base_basis_object_records(
    *,
    selected_rows: Sequence[ScreenedCandidate],
    raw_X: np.ndarray,
    raw_feature_names: Sequence[str],
    object_member_lookup: Mapping[str, Sequence[ScreenedCandidate]],
    interference_context: Mapping[str, Any],
    periodic_context: Mapping[str, Any],
    cfg: OrthogonalBasisSearchConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    raw_feature_name_tuple = tuple(str(value) for value in tuple(raw_feature_names))
    for index, row in enumerate(tuple(selected_rows)):
        source_expr = _candidate_information_source_expr(row)
        source_view = _candidate_source_object_view(row)
        base_key = _candidate_object_key(
            candidate=row,
            feature_names=raw_feature_name_tuple,
            interference_context=interference_context,
            periodic_context=periodic_context,
            outer_search_unit=str(cfg.outer_search_unit),
        )
        object_key = str(base_key)
        suffix = 2
        while object_key in used_keys:
            object_key = f"{base_key}__{suffix}"
            suffix += 1
        used_keys.add(object_key)
        object_members = tuple(object_member_lookup.get(str(base_key), ()))
        chart_resolution = _resolve_working_chart_for_source_object(
            source_expr=dict(source_expr),
            object_members=object_members,
            raw_X=np.asarray(raw_X, dtype=float),
            cfg=cfg,
            object_key=str(object_key),
        )
        working_expr = dict(chart_resolution.get("expr", source_expr))
        values = np.asarray(chart_resolution.get("values"), dtype=float).reshape(-1, 1)
        object_kind = _candidate_object_kind(
            candidate=row,
            feature_names=raw_feature_name_tuple,
            periodic_context=periodic_context,
        )
        feature_name_values = _candidate_feature_names_for_row(candidate=row, feature_names=raw_feature_name_tuple)
        periodic_feature_names = _candidate_periodic_feature_names(
            candidate=row,
            feature_names=raw_feature_name_tuple,
            periodic_context=periodic_context,
        )
        parent_object_key: str | None = None
        if _candidate_is_structural_gate(row) and tuple(row.features):
            parent_object_key = f"source::{_candidate_expr_key(_feature_expr(int(tuple(row.features)[0])))}"
        records.append(
            {
                "object_key": str(object_key),
                "outer_object_key": str(base_key),
                "expression": _named_expression_for_expr(working_expr, raw_feature_names=raw_feature_name_tuple),
                "expr": dict(working_expr),
                "term_name": str(row.name or object_key),
                "semantic_family": str(row.semantic_family or row.family or "basis_object"),
                "feature_names": tuple(str(value) for value in tuple(feature_name_values) if str(value).strip()),
                "object_kind": str(object_kind),
                "binding_role": "locked_basis_object",
                "object_role": str(_candidate_object_role(row)),
                "parent_object_key": parent_object_key,
                "reuse_budget_cost": float(_candidate_reuse_budget_cost(row, cfg)),
                "semantic_signature": str(row.semantic_signature),
                "uses_piecewise_gate": bool(row.uses_piecewise_gate),
                "source_information_key": str(row.information_source_key or _candidate_information_source_key(row)),
                "source_object_key": str(source_view.get("source_object_key") or row.source_object_key or ""),
                "source_support_key": str(row.source_support_key or ""),
                "source_support_size": int(row.source_support_size),
                "chart_signature": str(chart_resolution.get("chart_signature") or "identity"),
                "chart_metadata": dict(chart_resolution.get("chart_metadata", {}) or {}),
                "realization_head_signature": str(
                    row.realization_head_signature or source_view.get("realization_head_signature") or ""
                ),
                "periodic_feature_names": tuple(str(value) for value in tuple(periodic_feature_names) if str(value).strip()),
                "required_realization_family": (
                    "periodic"
                    if str(object_kind) == "periodic_channel"
                    else None
                ),
                "selected_evidence_term_name": str(row.name),
                "selected_evidence_expression": str(row.expression),
                "selected_evidence_signature": str(_candidate_realization_signature(row)),
                "chart_canonicalization_report": _jsonable(dict(chart_resolution.get("report", {}) or {})),
                "structural_channel": str(row.structural_channel or "challenger"),
                "support_expansion_tagged": bool(row.support_expansion_tagged),
                "canonical_trunk_tagged": bool(row.canonical_trunk_tagged),
                "same_source_surrogate_tagged": bool(row.same_source_surrogate_tagged),
                "selection_channel": str(row.selection_channel or "challenger"),
                "support_expansion_candidate": bool(row.support_expansion_candidate),
                "canonical_trunk_candidate": bool(row.canonical_trunk_candidate),
                "same_source_surrogate_candidate": bool(row.same_source_surrogate_candidate),
                "global_uniform_candidate": bool(row.global_uniform_candidate),
                "modulated_branch_candidate": bool(row.modulated_branch_candidate),
                "rational_template_pinned": bool(row.rational_template_pinned),
                "values": np.asarray(values[:, 0], dtype=float).reshape(-1, 1),
            }
        )
    return records


def _collect_object_realization_specs(
    *,
    base_record: Mapping[str, Any],
    object_members: Sequence[ScreenedCandidate],
    realization_evidence_registry: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    cfg: OrthogonalBasisSearchConfig,
) -> dict[str, Any]:
    signatures: dict[str, dict[str, Any]] = {}

    def _upsert(
        signature: str,
        *,
        protocol_name: str,
        member: ScreenedCandidate | None,
        evidence_term_names: Sequence[str] = tuple(),
        evidence_screen_score: float = 0.0,
        evidence_residual_gain: float = 0.0,
        source_expr: Mapping[str, Any] | None = None,
        realization_expr: Mapping[str, Any] | None = None,
    ) -> None:
        if not signature or str(signature) == "identity":
            return
        item = signatures.setdefault(
            str(signature),
            {
                "signature": str(signature),
                "protocols": set(),
                "evidence_term_names": set(),
                "evidence_screen_score": 0.0,
                "evidence_residual_gain": 0.0,
            },
        )
        item["protocols"].add(str(protocol_name))
        if member is not None:
            item["evidence_term_names"].add(str(member.name))
            item["evidence_screen_score"] = max(float(item.get("evidence_screen_score", 0.0)), float(member.screen_score))
            item["evidence_residual_gain"] = max(float(item.get("evidence_residual_gain", 0.0)), float(member.residual_gain))
        for name in tuple(evidence_term_names or ()):
            if str(name).strip():
                item["evidence_term_names"].add(str(name))
        item["evidence_screen_score"] = max(
            float(item.get("evidence_screen_score", 0.0)),
            float(evidence_screen_score),
        )
        item["evidence_residual_gain"] = max(
            float(item.get("evidence_residual_gain", 0.0)),
            float(evidence_residual_gain),
        )
        if isinstance(source_expr, Mapping):
            item["source_expr"] = dict(source_expr)
        if isinstance(realization_expr, Mapping):
            item["realization_expr"] = dict(realization_expr)

    if _periodic_realization_competition_enabled(cfg) and str(base_record.get("object_kind")) == "periodic_channel":
        _upsert(
            "unary:sin",
            protocol_name=str(cfg.periodic_realization_competition_protocol),
            member=None,
        )
        _upsert(
            "unary:cos",
            protocol_name=str(cfg.periodic_realization_competition_protocol),
            member=None,
        )

    if _realization_prior_injection_enabled(cfg):
        allowed = {"unary:exp", "unary:exp_neg", "unary:square", "unary:sin", "unary:cos"}
        for member in tuple(object_members):
            signature = _candidate_realization_signature(member)
            if signature not in allowed:
                continue
            _upsert(
                signature,
                protocol_name=str(cfg.realization_prior_injection_protocol),
                member=member,
            )

    registry = dict(realization_evidence_registry or {})
    source_keys = tuple(
        str(value).strip()
        for value in (
            base_record.get("source_information_key"),
            base_record.get("source_object_key"),
        )
        if str(value or "").strip()
    )
    seen_registry_rows: set[tuple[str, str]] = set()
    for source_key in source_keys:
        for evidence in tuple(registry.get(str(source_key), ())):
            signature = str(dict(evidence).get("signature") or "").strip()
            if not signature:
                continue
            row_key = (str(source_key), str(signature))
            if row_key in seen_registry_rows:
                continue
            seen_registry_rows.add(row_key)
            _upsert(
                signature,
                protocol_name=",".join(
                    str(value)
                    for value in tuple(dict(evidence).get("protocols", ()) or ())
                    if str(value).strip()
                )
                or str(cfg.realization_prior_injection_protocol),
                member=None,
                evidence_term_names=tuple(dict(evidence).get("evidence_term_names", ()) or ()),
                evidence_screen_score=float(dict(evidence).get("evidence_screen_score", 0.0) or 0.0),
                evidence_residual_gain=float(dict(evidence).get("evidence_residual_gain", 0.0) or 0.0),
                source_expr=(
                    dict(dict(evidence).get("source_expr", {}))
                    if isinstance(dict(evidence).get("source_expr"), Mapping)
                    else None
                ),
                realization_expr=(
                    dict(dict(evidence).get("realization_expr", {}))
                    if isinstance(dict(evidence).get("realization_expr"), Mapping)
                    else None
                ),
            )

    all_specs = list(signatures.values())
    all_specs.sort(
        key=lambda item: (
            -float(item.get("evidence_screen_score", 0.0)),
            -float(item.get("evidence_residual_gain", 0.0)),
            str(item.get("signature", "")),
        )
    )
    selected_specs = list(all_specs)
    forced_signatures: set[str] = set()
    if _same_source_over_realization_enabled(cfg):
        budget = max(1, int(cfg.same_source_realization_budget))
        if (
            str(base_record.get("object_kind") or "") == "periodic_channel"
            and _periodic_realization_competition_enabled(cfg)
        ):
            budget = max(budget, 2)
        selected_specs = selected_specs[:budget]
        # mandatory closure should not lose exp/exp_neg due same-source budget trimming.
        if _mandatory_realization_closure_enabled(cfg):
            keep_signatures = {"unary:exp", "unary:exp_neg"}
            selected_signature_set = {
                str(item.get("signature", "")).strip()
                for item in tuple(selected_specs)
            }
            for item in tuple(all_specs):
                signature = str(item.get("signature", "")).strip()
                if signature not in keep_signatures:
                    continue
                if signature in selected_signature_set:
                    continue
                if not tuple(item.get("evidence_term_names", ())):
                    continue
                selected_specs.append(dict(item))
                selected_signature_set.add(signature)
                forced_signatures.add(signature)
    selected_signature_set = {
        str(item.get("signature", "")).strip()
        for item in tuple(selected_specs)
    }
    catalog: list[dict[str, Any]] = []
    for item in tuple(all_specs):
        signature = str(item.get("signature", "")).strip()
        selected = signature in selected_signature_set
        if selected:
            reason = "forced_mandatory_finalist" if signature in forced_signatures else "selected"
        else:
            reason = "trimmed_by_same_source_budget" if _same_source_over_realization_enabled(cfg) else "not_selected"
        catalog.append(
            {
                "signature": signature,
                "protocols": sorted(str(v) for v in tuple(item.get("protocols", set())) if str(v).strip()),
                "evidence_term_names": sorted(
                    str(v) for v in tuple(item.get("evidence_term_names", set())) if str(v).strip()
                ),
                "evidence_screen_score": float(item.get("evidence_screen_score", 0.0) or 0.0),
                "evidence_residual_gain": float(item.get("evidence_residual_gain", 0.0) or 0.0),
                "source_expr": (
                    _jsonable(dict(item.get("source_expr", {})))
                    if isinstance(item.get("source_expr"), Mapping)
                    else None
                ),
                "realization_expr": (
                    _jsonable(dict(item.get("realization_expr", {})))
                    if isinstance(item.get("realization_expr"), Mapping)
                    else None
                ),
                "selected": bool(selected),
                "selection_reason": str(reason),
            }
        )
    return {
        "selected_specs": [
            {
                "signature": str(item.get("signature", "")).strip(),
                "protocols": sorted(str(v) for v in tuple(item.get("protocols", set())) if str(v).strip()),
                "evidence_term_names": sorted(
                    str(v) for v in tuple(item.get("evidence_term_names", set())) if str(v).strip()
                ),
                "evidence_screen_score": float(item.get("evidence_screen_score", 0.0) or 0.0),
                "evidence_residual_gain": float(item.get("evidence_residual_gain", 0.0) or 0.0),
                "source_expr": (
                    dict(item.get("source_expr", {}))
                    if isinstance(item.get("source_expr"), Mapping)
                    else None
                ),
                "realization_expr": (
                    dict(item.get("realization_expr", {}))
                    if isinstance(item.get("realization_expr"), Mapping)
                    else None
                ),
                "forced_finalist": bool(
                    str(item.get("signature", "")).strip() in forced_signatures
                ),
            }
            for item in tuple(selected_specs)
        ],
        "catalog": catalog,
    }


def _basis_object_refs(
    *,
    basis_object_records: Sequence[Mapping[str, Any]],
) -> tuple[BasisObjectRef, ...]:
    refs: list[BasisObjectRef] = []
    for raw_record in tuple(basis_object_records):
        record = dict(raw_record)
        metadata = {
            "term_name": str(record.get("term_name") or record.get("object_key") or ""),
            "semantic_signature": str(record.get("semantic_signature") or ""),
            "uses_piecewise_gate": bool(record.get("uses_piecewise_gate")),
            "binding_role": str(record.get("binding_role") or ""),
            "object_role": str(record.get("object_role") or ""),
            "object_kind": str(record.get("object_kind") or ""),
            "parent_object_key": record.get("parent_object_key"),
            "reuse_budget_cost": float(record.get("reuse_budget_cost", 1.0) or 0.0),
            "required_realization_family": record.get("required_realization_family"),
            "source_information_key": str(record.get("source_information_key") or ""),
            "source_object_key": str(record.get("source_object_key") or ""),
            "source_support_key": str(record.get("source_support_key") or ""),
            "source_support_size": int(record.get("source_support_size", 0) or 0),
            "chart_signature": str(record.get("chart_signature") or "identity"),
            "chart_metadata": _jsonable(dict(record.get("chart_metadata", {}) or {})),
            "chart_canonicalization_report": _jsonable(dict(record.get("chart_canonicalization_report", {}) or {})),
            "structural_channel": str(record.get("structural_channel") or "challenger"),
            "support_expansion_tagged": bool(record.get("support_expansion_tagged")),
            "canonical_trunk_tagged": bool(record.get("canonical_trunk_tagged")),
            "same_source_surrogate_tagged": bool(record.get("same_source_surrogate_tagged")),
            "selection_channel": str(record.get("selection_channel") or "challenger"),
            "support_expansion_candidate": bool(record.get("support_expansion_candidate")),
            "canonical_trunk_candidate": bool(record.get("canonical_trunk_candidate")),
            "same_source_surrogate_candidate": bool(record.get("same_source_surrogate_candidate")),
            "global_uniform_candidate": bool(record.get("global_uniform_candidate")),
            "modulated_branch_candidate": bool(record.get("modulated_branch_candidate")),
            "rational_template_pinned": bool(record.get("rational_template_pinned")),
            "chart_flip_protocols": [
                str(value) for value in tuple(record.get("chart_flip_protocols", ())) if str(value).strip()
            ],
            "chart_flip_origin_object_key": record.get("chart_flip_origin_object_key"),
            "realization_head_signature": str(record.get("realization_head_signature") or ""),
            "periodic_feature_names": [
                str(value) for value in tuple(record.get("periodic_feature_names", ())) if str(value).strip()
            ],
            "selected_evidence_term_name": str(record.get("selected_evidence_term_name") or ""),
            "selected_evidence_expression": str(record.get("selected_evidence_expression") or ""),
            "selected_evidence_signature": str(record.get("selected_evidence_signature") or ""),
            "realization_signature": str(record.get("realization_signature") or ""),
            "realization_source_expr": _jsonable(record.get("realization_source_expr")),
            "realization_expr_override": _jsonable(record.get("realization_expr_override")),
            "realization_protocols": [
                str(value) for value in tuple(record.get("realization_protocols", ())) if str(value).strip()
            ],
            "realization_evidence_term_names": [
                str(value)
                for value in tuple(record.get("realization_evidence_term_names", ()))
                if str(value).strip()
            ],
            "realization_evidence_screen_score": record.get("realization_evidence_screen_score"),
            "realization_evidence_residual_gain": record.get("realization_evidence_residual_gain"),
            "realization_forced_finalist": bool(record.get("realization_forced_finalist", False)),
            "realization_signature_catalog": _jsonable(tuple(record.get("realization_signature_catalog", ()))),
            "realization_signature_selected": [
                str(value) for value in tuple(record.get("realization_signature_selected", ())) if str(value).strip()
            ],
            "realization_signature_forced_finalists": [
                str(value)
                for value in tuple(record.get("realization_signature_forced_finalists", ()))
                if str(value).strip()
            ],
        }
        refs.append(
            BasisObjectRef(
                object_key=str(record.get("object_key") or ""),
                expression=str(record.get("expression") or record.get("term_name") or record.get("object_key") or ""),
                family_ref=(
                    str(record.get("semantic_family"))
                    if str(record.get("semantic_family") or "").strip()
                    else None
                ),
                source_features=tuple(
                    str(value) for value in tuple(record.get("feature_names", ())) if str(value).strip()
                ),
                metadata=metadata,
            )
        )
    return tuple(refs)


def _configured_escape_feature_names(
    *,
    cfg: OrthogonalBasisSearchConfig,
    raw_feature_names: Sequence[str],
) -> tuple[str, ...]:
    allowed = {str(name) for name in tuple(raw_feature_names)}
    return tuple(
        name for name in tuple(cfg.assembler_escape_feature_names) if str(name).strip() and str(name) in allowed
    )


def _resolved_assembler_binding_mode(
    *,
    cfg: OrthogonalBasisSearchConfig,
    escape_feature_names: Sequence[str],
) -> str:
    binding_mode = str(cfg.assembler_basis_binding_mode or "defining").strip().lower()
    if tuple(escape_feature_names) and str(cfg.assembler_escape_policy or "forbid").strip().lower() != "forbid":
        if binding_mode == "defining":
            return "bound"
    return binding_mode


def _build_assembler_object_space(
    *,
    outer_basis_genome: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[ScreenedCandidate],
    basis_rows: Sequence[Mapping[str, Any]],
    train_matrix: np.ndarray,
    raw_X: np.ndarray,
    raw_feature_names: Sequence[str],
    regional_correction_candidates: Sequence[Mapping[str, Any]] | None,
    screened_candidates: Sequence[ScreenedCandidate],
    interference_context: Mapping[str, Any],
    periodic_context: Mapping[str, Any],
    cfg: OrthogonalBasisSearchConfig,
    realization_evidence_registry: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    target: np.ndarray | None = None,
    gate_feature_names: Sequence[str] | None = None,
    data_metadata: Mapping[str, Any] | None = None,
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    del outer_basis_genome
    del basis_rows
    del train_matrix
    candidate_objects = _build_candidate_objects(
        screened=screened_candidates,
        feature_names=tuple(str(value) for value in tuple(raw_feature_names)),
        interference_context=interference_context,
        periodic_context=periodic_context,
        outer_search_unit=str(cfg.outer_search_unit),
    )
    object_member_lookup = {
        str(item.object_key): tuple(item.members)
        for item in tuple(candidate_objects)
    }
    base_object_records = _build_base_basis_object_records(
        selected_rows=selected_rows,
        raw_X=np.asarray(raw_X, dtype=float),
        raw_feature_names=raw_feature_names,
        object_member_lookup=object_member_lookup,
        interference_context=interference_context,
        periodic_context=periodic_context,
        cfg=cfg,
    )
    base_feature_names = [str(record.get("object_key") or "") for record in tuple(base_object_records)]
    if base_object_records:
        base_matrix = np.asarray(
            np.concatenate(
                [np.asarray(record.get("values"), dtype=float).reshape(-1, 1) for record in tuple(base_object_records)],
                axis=1,
            ),
            dtype=float,
        )
    else:
        base_matrix = np.zeros((int(np.asarray(raw_X).shape[0]), 0), dtype=float)
    basis_genome = [
        {
            "name": str(record.get("object_key") or record.get("term_name") or f"basis_object_{index}"),
            "expr": dict(record.get("expr", {})),
        }
        for index, record in enumerate(tuple(base_object_records))
    ]
    used_object_keys = set(base_feature_names)
    basis_object_records: list[dict[str, Any]] = [
        {
            key: value
            for key, value in dict(record).items()
            if str(key) != "values"
        }
        for record in tuple(base_object_records)
    ]
    chart_objects: list[dict[str, Any]] = []
    chart_columns: list[np.ndarray] = []
    if _inner_chart_flip_compensation_enabled(cfg):
        for base_record in tuple(base_object_records):
            base_expr = dict(base_record.get("expr", {}))
            if not _expr_is_plain_ratio_source(base_expr):
                continue
            base_chart_signature = _normalized_chart_family_signature(
                chart_signature=str(base_record.get("chart_signature") or "identity"),
                chart_metadata=dict(base_record.get("chart_metadata", {}) or {}),
            )
            alternate_signature = "identity" if base_chart_signature == "reciprocal" else "reciprocal"
            alternate_expr = _chart_expr_from_source_expr(
                source_expr=base_expr,
                chart_signature="reciprocal",
            )
            object_key = (
                f"{str(base_record.get('object_key') or '')}::chart::"
                f"{'identity' if alternate_signature == 'identity' else 'reciprocal'}"
            )
            suffix = 2
            base_key = object_key
            while object_key in used_object_keys:
                object_key = f"{base_key}__{suffix}"
                suffix += 1
            used_object_keys.add(object_key)
            values = design_matrix_for_genome(
                ({"name": str(object_key), "expr": dict(alternate_expr)},),
                np.asarray(raw_X, dtype=float),
                batch_key=f"orthogonal_chart_flip::{object_key}",
            )
            expression = _named_expression_for_expr(alternate_expr, raw_feature_names=raw_feature_names)
            chart_columns.append(np.asarray(values[:, 0], dtype=float).reshape(-1, 1))
            base_feature_names.append(object_key)
            basis_genome.append({"name": str(object_key), "expr": dict(alternate_expr)})
            record = {
                "object_key": str(object_key),
                "expression": str(expression),
                "expr": dict(alternate_expr),
                "term_name": str(object_key),
                "semantic_family": "chart_variant",
                "feature_names": tuple(
                    str(value) for value in tuple(base_record.get("feature_names", ())) if str(value).strip()
                ),
                "object_kind": "chart_variant",
                "binding_role": "chart_competitor",
                "object_role": "chart_variant",
                "parent_object_key": str(base_record.get("object_key") or ""),
                "reuse_budget_cost": 0.0,
                "semantic_signature": str(alternate_signature),
                "uses_piecewise_gate": False,
                "source_information_key": str(base_record.get("source_information_key") or ""),
                "source_object_key": str(base_record.get("source_object_key") or ""),
                "chart_signature": str(alternate_signature),
                "chart_metadata": _chart_metadata_for_working_signature(
                    chart_signature=str(alternate_signature),
                    source_expr=base_expr,
                ),
                "realization_head_signature": "",
                "periodic_feature_names": tuple(),
                "required_realization_family": None,
                "selected_evidence_term_name": str(base_record.get("selected_evidence_term_name") or ""),
                "selected_evidence_expression": str(base_record.get("selected_evidence_expression") or ""),
                "selected_evidence_signature": str(base_record.get("selected_evidence_signature") or ""),
                "chart_flip_protocols": [str(cfg.inner_chart_flip_compensation_protocol)],
                "chart_flip_origin_object_key": str(base_record.get("object_key") or ""),
            }
            chart_objects.append(dict(record))
            basis_object_records.append(dict(record))
    realization_objects: list[dict[str, Any]] = []
    realization_columns: list[np.ndarray] = []
    for base_record in tuple(base_object_records):
        realization_payload = _collect_object_realization_specs(
            base_record=base_record,
            object_members=tuple(object_member_lookup.get(str(base_record.get("outer_object_key") or ""), ())),
            realization_evidence_registry=realization_evidence_registry,
            cfg=cfg,
        )
        realization_specs = tuple(realization_payload.get("selected_specs", ()))
        realization_catalog = tuple(realization_payload.get("catalog", ()))
        base_record["realization_signature_catalog"] = tuple(_jsonable(realization_catalog))
        base_record["realization_signature_selected"] = tuple(
            str(item.get("signature", "")).strip()
            for item in tuple(realization_specs)
            if str(item.get("signature", "")).strip()
        )
        base_record["realization_signature_forced_finalists"] = tuple(
            str(item.get("signature", "")).strip()
            for item in tuple(realization_specs)
            if bool(item.get("forced_finalist"))
        )
        for record in basis_object_records:
            if str(record.get("object_key") or "") != str(base_record.get("object_key") or ""):
                continue
            if str(record.get("binding_role") or "") != "locked_basis_object":
                continue
            record["realization_signature_catalog"] = tuple(_jsonable(realization_catalog))
            record["realization_signature_selected"] = tuple(base_record.get("realization_signature_selected", ()))
            record["realization_signature_forced_finalists"] = tuple(
                base_record.get("realization_signature_forced_finalists", ())
            )
        for spec in tuple(realization_specs):
            signature = str(spec.get("signature") or "").strip()
            if not signature:
                continue
            expr = (
                dict(spec.get("realization_expr", {}))
                if isinstance(spec.get("realization_expr"), Mapping)
                else _realization_signature_expr(signature, source_expr=dict(base_record.get("expr", {})))
            )
            object_key = f"{str(base_record.get('object_key') or '')}::realization::{_realization_signature_key(signature)}"
            suffix = 2
            base_key = object_key
            while object_key in used_object_keys:
                object_key = f"{base_key}__{suffix}"
                suffix += 1
            used_object_keys.add(object_key)
            values = design_matrix_for_genome(
                ({"name": str(object_key), "expr": dict(expr)},),
                np.asarray(raw_X, dtype=float),
                batch_key=f"orthogonal_realization::{object_key}",
            )
            expression = _named_expression_for_expr(expr, raw_feature_names=raw_feature_names)
            realization_columns.append(np.asarray(values[:, 0], dtype=float).reshape(-1, 1))
            base_feature_names.append(object_key)
            basis_genome.append({"name": str(object_key), "expr": dict(expr)})
            record = {
                "object_key": str(object_key),
                "expression": str(expression),
                "expr": dict(expr),
                "term_name": str(object_key),
                "semantic_family": (
                    "periodic_realization"
                    if signature in {"unary:sin", "unary:cos"}
                    else "basis_realization"
                ),
                "feature_names": tuple(
                    str(value) for value in tuple(base_record.get("feature_names", ())) if str(value).strip()
                ),
                "object_kind": "realization_head",
                "binding_role": "realization_competitor",
                "object_role": "realization_head",
                "parent_object_key": str(base_record.get("object_key") or ""),
                "reuse_budget_cost": 0.0,
                "semantic_signature": str(signature),
                "uses_piecewise_gate": False,
                "source_information_key": str(base_record.get("source_information_key") or ""),
                "source_object_key": str(base_record.get("source_object_key") or ""),
                "chart_signature": str(base_record.get("chart_signature") or "identity"),
                "chart_metadata": dict(base_record.get("chart_metadata", {}) or {}),
                "realization_head_signature": str(signature),
                "periodic_feature_names": tuple(
                    str(value) for value in tuple(base_record.get("periodic_feature_names", ())) if str(value).strip()
                ),
                "required_realization_family": (
                    "periodic"
                    if signature in {"unary:sin", "unary:cos"}
                    else None
                ),
                "selected_evidence_term_name": str(base_record.get("selected_evidence_term_name") or ""),
                "selected_evidence_expression": str(base_record.get("selected_evidence_expression") or ""),
                "selected_evidence_signature": str(base_record.get("selected_evidence_signature") or ""),
                "realization_signature": str(signature),
                "realization_source_expr": (
                    _jsonable(dict(spec.get("source_expr", {})))
                    if isinstance(spec.get("source_expr"), Mapping)
                    else None
                ),
                "realization_expr_override": (
                    _jsonable(dict(spec.get("realization_expr", {})))
                    if isinstance(spec.get("realization_expr"), Mapping)
                    else None
                ),
                "realization_protocols": sorted(
                    str(value) for value in tuple(spec.get("protocols", ())) if str(value).strip()
                ),
                "realization_evidence_term_names": sorted(
                    str(value) for value in tuple(spec.get("evidence_term_names", ())) if str(value).strip()
                ),
                "realization_evidence_screen_score": float(spec.get("evidence_screen_score", 0.0) or 0.0),
                "realization_evidence_residual_gain": float(spec.get("evidence_residual_gain", 0.0) or 0.0),
                "realization_forced_finalist": bool(spec.get("forced_finalist", False)),
            }
            realization_objects.append(dict(record))
            basis_object_records.append(dict(record))
    regional_branch_specs = (
        _build_regional_branch_evidence_specs(
            base_object_records=tuple(base_object_records),
            base_matrix=np.asarray(base_matrix, dtype=float),
            raw_X=np.asarray(raw_X, dtype=float),
            target=np.asarray(target, dtype=float),
            raw_feature_names=tuple(raw_feature_names),
            gate_feature_names=tuple(str(value) for value in tuple(gate_feature_names or ())),
            data_metadata=data_metadata,
            cfg=cfg,
        )
        if target is not None
        else tuple()
    )
    branch_columns: list[np.ndarray] = []
    branch_objects: list[dict[str, Any]] = []
    branch_catalog_by_parent: dict[str, list[dict[str, Any]]] = {}
    for spec in tuple(regional_branch_specs):
        parent_key = str(dict(spec).get("parent_object_key") or "").strip()
        signature = str(dict(spec).get("branch_signature") or "").strip()
        if not parent_key or not signature:
            continue
        branch_catalog_by_parent.setdefault(parent_key, []).append(
            {
                "signature": str(signature),
                "selected": True,
                "selection_reason": "regional_branch_evidence",
                "threshold": float(dict(spec).get("threshold", 0.0) or 0.0),
                "direction": str(dict(spec).get("branch_direction") or ""),
                "evidence_term_names": tuple(
                    str(value) for value in tuple(dict(spec).get("evidence_term_names", ())) if str(value).strip()
                ),
                "evidence_score": float(dict(spec).get("branch_evidence_score", 0.0) or 0.0),
                "marginal_r2_gain": float(dict(spec).get("marginal_r2_gain", 0.0) or 0.0),
                "residual_abs_corr": float(dict(spec).get("residual_abs_corr", 0.0) or 0.0),
                "residual_regime_score": float(dict(spec).get("residual_regime_score", 0.0) or 0.0),
                "threshold_orthodoxy_score": float(dict(spec).get("threshold_orthodoxy_score", 0.0) or 0.0),
                "threshold_stability_score": float(dict(spec).get("threshold_stability_score", 0.0) or 0.0),
                "threshold_balance_score": float(dict(spec).get("threshold_balance_score", 0.0) or 0.0),
                "threshold_audit": _jsonable(dict(spec).get("threshold_audit", {}) or {}),
                "threshold_selection_lane": str(dict(spec).get("threshold_selection_lane") or ""),
            }
        )
    for base_record in base_object_records:
        parent_key = str(base_record.get("object_key") or "").strip()
        branch_catalog = tuple(branch_catalog_by_parent.get(parent_key, ()))
        if not branch_catalog:
            continue
        base_record["regional_branch_signature_catalog"] = tuple(_jsonable(branch_catalog))
        base_record["regional_branch_signature_selected"] = tuple(
            str(item.get("signature", "")).strip()
            for item in tuple(branch_catalog)
            if str(item.get("signature", "")).strip()
        )
        for record in basis_object_records:
            if str(record.get("object_key") or "") != parent_key:
                continue
            if str(record.get("binding_role") or "") != "locked_basis_object":
                continue
            record["regional_branch_signature_catalog"] = tuple(_jsonable(branch_catalog))
            record["regional_branch_signature_selected"] = tuple(base_record.get("regional_branch_signature_selected", ()))

    for spec in tuple(regional_branch_specs):
        values = np.asarray(dict(spec).get("values"), dtype=float).reshape(-1)
        if values.shape[0] != base_matrix.shape[0]:
            continue
        parent_key = str(dict(spec).get("parent_object_key") or "").strip()
        signature = str(dict(spec).get("branch_signature") or "").strip()
        cut = float(dict(spec).get("threshold", 0.0) or 0.0)
        object_key = f"{parent_key}::branch::{_branch_signature_key(signature, cut=cut)}"
        suffix = 2
        base_key = object_key
        while object_key in used_object_keys:
            object_key = f"{base_key}__{suffix}"
            suffix += 1
        used_object_keys.add(object_key)
        expr = dict(dict(spec).get("expr", {}))
        expression = _named_expression_for_expr(expr, raw_feature_names=raw_feature_names)
        branch_columns.append(values.reshape(-1, 1))
        base_feature_names.append(object_key)
        basis_genome.append({"name": str(object_key), "expr": dict(expr)})
        record = {
            "object_key": str(object_key),
            "expression": str(expression),
            "expr": dict(expr),
            "term_name": str(object_key),
            "semantic_family": "regional_branch",
            "feature_names": (str(dict(spec).get("feature_name") or ""),),
            "object_kind": "regional_branch",
            "binding_role": "regional_branch_competitor",
            "object_role": "regional_branch",
            "parent_object_key": str(parent_key),
            "reuse_budget_cost": 0.0,
            "semantic_signature": str(signature),
            "uses_piecewise_gate": True,
            "source_information_key": str(dict(spec).get("source_information_key") or ""),
            "source_object_key": str(dict(spec).get("source_object_key") or ""),
            "chart_signature": "regional_branch",
            "chart_metadata": {
                "regional_branch": True,
                "is_identity_chart": False,
                "threshold": float(cut),
                "direction": str(dict(spec).get("branch_direction") or ""),
            },
            "realization_head_signature": "",
            "periodic_feature_names": tuple(),
            "required_realization_family": None,
            "selected_evidence_term_name": ",".join(
                str(value) for value in tuple(dict(spec).get("evidence_term_names", ())) if str(value).strip()
            ),
            "selected_evidence_expression": str(expression),
            "selected_evidence_signature": str(signature),
            "branch_signature": str(signature),
            "branch_protocols": tuple(
                str(value) for value in tuple(dict(spec).get("branch_protocols", ())) if str(value).strip()
            ),
            "branch_evidence_term_names": tuple(
                str(value) for value in tuple(dict(spec).get("evidence_term_names", ())) if str(value).strip()
            ),
            "branch_evidence_score": float(dict(spec).get("branch_evidence_score", 0.0) or 0.0),
            "branch_marginal_r2_gain": float(dict(spec).get("marginal_r2_gain", 0.0) or 0.0),
            "branch_residual_abs_corr": float(dict(spec).get("residual_abs_corr", 0.0) or 0.0),
            "branch_residual_regime_score": float(dict(spec).get("residual_regime_score", 0.0) or 0.0),
            "branch_threshold": float(cut),
            "branch_direction": str(dict(spec).get("branch_direction") or ""),
            "threshold_orthodoxy_score": float(dict(spec).get("threshold_orthodoxy_score", 0.0) or 0.0),
            "threshold_stability_score": float(dict(spec).get("threshold_stability_score", 0.0) or 0.0),
            "threshold_balance_score": float(dict(spec).get("threshold_balance_score", 0.0) or 0.0),
            "threshold_audit": _jsonable(dict(spec).get("threshold_audit", {}) or {}),
            "threshold_selection_lane": str(dict(spec).get("threshold_selection_lane") or ""),
            "branch_forced_finalist": True,
            "closure_role": "regional_branch_additive",
        }
        branch_objects.append(dict(record))
        basis_object_records.append(dict(record))
    correction_columns: list[np.ndarray] = []
    correction_objects: list[dict[str, Any]] = []
    for correction in tuple(regional_correction_candidates or ()):
        values = np.asarray(correction.get("values"), dtype=float).reshape(-1)
        if values.shape[0] != base_matrix.shape[0]:
            continue
        object_key = str(correction.get("object_key") or f"regional::{len(correction_objects)}").strip()
        if not object_key:
            object_key = f"regional::{len(correction_objects)}"
        suffix = 2
        base_key = object_key
        while object_key in used_object_keys:
            object_key = f"{base_key}__{suffix}"
            suffix += 1
        used_object_keys.add(object_key)
        base_feature_names.append(object_key)
        correction_columns.append(values.reshape(-1, 1))
        basis_genome.append(
            {
                "name": str(correction.get("candidate_name") or correction.get("term_name") or object_key),
                "expr": dict(correction.get("expr", {})),
            }
        )
        correction_objects.append(
            {
                **dict(correction),
                "object_key": str(object_key),
            }
        )
        basis_object_records.append(
            {
                "object_key": str(object_key),
                "expression": str(
                    correction.get("expression")
                    or correction.get("term_name")
                    or correction.get("candidate_name")
                    or object_key
                ),
                "term_name": str(correction.get("candidate_name") or correction.get("term_name") or object_key),
                "semantic_family": str(correction.get("semantic_family") or "regional_correction"),
                "feature_names": tuple(
                    str(value) for value in tuple(correction.get("feature_names", ())) if str(value).strip()
                ),
                "object_kind": "regional_correction",
                "binding_role": "regional_correction",
                "object_role": "correction_branch",
                "parent_object_key": str(correction.get("parent_object_key") or "") or None,
                "reuse_budget_cost": 0.0,
                "semantic_signature": str(correction.get("semantic_signature") or ""),
                "uses_piecewise_gate": bool(correction.get("uses_piecewise_gate", True)),
                "source_information_key": str(correction.get("source_information_key") or ""),
                "source_object_key": str(correction.get("source_object_key") or correction.get("source_information_key") or ""),
                "chart_signature": "regional_branch",
                "chart_metadata": {"regional_branch": True, "is_identity_chart": False},
                "realization_head_signature": "",
                "periodic_feature_names": tuple(),
                "required_realization_family": None,
                "selected_evidence_term_name": str(correction.get("candidate_name") or correction.get("term_name") or ""),
                "selected_evidence_expression": str(
                    correction.get("expression")
                    or correction.get("term_name")
                    or correction.get("candidate_name")
                    or ""
                ),
                "selected_evidence_signature": "",
            }
        )
    escape_feature_names = _configured_escape_feature_names(cfg=cfg, raw_feature_names=raw_feature_names)
    if not escape_feature_names or str(cfg.assembler_escape_policy) == "forbid":
        if not correction_columns and not realization_columns and not chart_columns and not branch_columns:
            return (
                np.asarray(base_matrix, dtype=float),
                tuple(base_feature_names),
                tuple(basis_genome),
                tuple(basis_object_records),
                tuple(chart_objects),
                tuple(realization_objects),
                tuple(correction_objects),
                tuple(),
            )
        augmented_matrix = np.asarray(
            np.concatenate(
                [
                    np.asarray(base_matrix, dtype=float),
                    *chart_columns,
                    *realization_columns,
                    *branch_columns,
                    *correction_columns,
                ],
                axis=1,
            ),
            dtype=float,
        )
        return (
            augmented_matrix,
            tuple(base_feature_names),
            tuple(basis_genome),
            tuple(basis_object_records),
            tuple(chart_objects),
            tuple(realization_objects),
            tuple(correction_objects),
            tuple(),
        )

    raw_names = tuple(str(value) for value in tuple(raw_feature_names))
    name_to_index = {str(name): int(index) for index, name in enumerate(raw_names)}
    escape_columns: list[np.ndarray] = []
    escape_objects: list[dict[str, Any]] = []
    for escape_name in escape_feature_names:
        raw_index = name_to_index.get(str(escape_name))
        if raw_index is None:
            continue
        object_key = f"escape::{escape_name}"
        suffix = 2
        while object_key in used_object_keys:
            object_key = f"escape::{escape_name}__{suffix}"
            suffix += 1
        used_object_keys.add(object_key)
        base_feature_names.append(object_key)
        escape_columns.append(np.asarray(raw_X[:, int(raw_index)], dtype=float).reshape(-1, 1))
        basis_genome.append(
            {
                "name": f"x{int(raw_index)}:{escape_name}",
                "expr": {"type": "feature", "index": int(raw_index)},
            }
        )
        escape_objects.append(
            {
                "object_key": str(object_key),
                "raw_feature_name": str(escape_name),
                "raw_feature_index": int(raw_index),
                "expression": str(escape_name),
            }
        )
        basis_object_records.append(
            {
                "object_key": str(object_key),
                "expression": str(escape_name),
                "term_name": f"x{int(raw_index)}:{escape_name}",
                "semantic_family": "raw_feature_escape",
                "feature_names": (str(escape_name),),
                "object_kind": "raw_feature_escape",
                "binding_role": "escape",
                "object_role": "escape_lane",
                "parent_object_key": None,
                "reuse_budget_cost": 1.0,
                "semantic_signature": "",
                "uses_piecewise_gate": False,
                "source_information_key": str(escape_name),
                "source_object_key": str(_candidate_expr_key(_feature_expr(int(raw_index)))),
                "chart_signature": "identity",
                "chart_metadata": {"is_identity_chart": True},
                "realization_head_signature": "",
                "periodic_feature_names": tuple(),
                "required_realization_family": None,
                "selected_evidence_term_name": str(escape_name),
                "selected_evidence_expression": str(escape_name),
                "selected_evidence_signature": "identity",
            }
        )

    if not escape_columns:
        if correction_columns or realization_columns or chart_columns or branch_columns:
            augmented_matrix = np.asarray(
                np.concatenate(
                    [
                        np.asarray(base_matrix, dtype=float),
                        *chart_columns,
                        *realization_columns,
                        *branch_columns,
                        *correction_columns,
                    ],
                    axis=1,
                ),
                dtype=float,
            )
        else:
            augmented_matrix = np.asarray(base_matrix, dtype=float)
        return (
            augmented_matrix,
            tuple(base_feature_names),
            tuple(basis_genome),
            tuple(basis_object_records),
            tuple(chart_objects),
            tuple(realization_objects),
            tuple(correction_objects),
            tuple(),
        )
    augmented_matrix = np.asarray(
        np.concatenate(
            [
                np.asarray(base_matrix, dtype=float),
                *chart_columns,
                *realization_columns,
                *branch_columns,
                *correction_columns,
                *escape_columns,
            ],
            axis=1,
        ),
        dtype=float,
    )
    return (
        augmented_matrix,
        tuple(base_feature_names),
        tuple(basis_genome),
        tuple(basis_object_records),
        tuple(chart_objects),
        tuple(realization_objects),
        tuple(correction_objects),
        tuple(escape_objects),
    )


def _build_stage_head_protocol_payload(
    *,
    basis_object_records: Sequence[Mapping[str, Any]],
    basis_feature_names: Sequence[str],
    gate_feature_names: Sequence[str],
    cfg: OrthogonalBasisSearchConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    object_records = [dict(item) for item in tuple(basis_object_records)]
    locked_basis_keys = [
        str(item.get("object_key") or "")
        for item in tuple(object_records)
        if str(item.get("binding_role") or "") == "locked_basis_object" and str(item.get("object_key") or "").strip()
    ]
    realization_object_keys = [
        str(item.get("object_key") or "")
        for item in tuple(object_records)
        if str(item.get("binding_role") or "") == "realization_competitor" and str(item.get("object_key") or "").strip()
    ]
    chart_object_keys = [
        str(item.get("object_key") or "")
        for item in tuple(object_records)
        if str(item.get("binding_role") or "") == "chart_competitor" and str(item.get("object_key") or "").strip()
    ]
    regional_correction_keys = [
        str(item.get("object_key") or "")
        for item in tuple(object_records)
        if str(item.get("binding_role") or "") == "regional_correction" and str(item.get("object_key") or "").strip()
    ]
    escape_feature_names = [
        str(name)
        for item in tuple(object_records)
        if str(item.get("binding_role") or "") == "escape"
        for name in tuple(item.get("feature_names", ()))
        if str(name).strip()
    ]
    escape_object_keys = [
        str(item.get("object_key") or "")
        for item in tuple(object_records)
        if str(item.get("binding_role") or "") == "escape" and str(item.get("object_key") or "").strip()
    ]
    binding_mode = _resolved_assembler_binding_mode(
        cfg=cfg,
        escape_feature_names=tuple(escape_feature_names),
    )
    escape_policy = str(cfg.assembler_escape_policy or "forbid").strip().lower()
    regional_correction_count = int(len(regional_correction_keys))
    basis_discovery_stage = build_basis_discovery_stage_spec(
        gradient_guidance_mode="raw_feature_gradient",
        notes="Orthogonal basis discovery over raw features with structure-head=basis_set.",
        metadata={
            "stage_key": "basis_discovery",
            "selection_protocol": "orthogonal_basis_search",
            "gate_feature_names": [str(value) for value in tuple(gate_feature_names)],
        },
    )
    assembler_stage = build_basis_conditioned_expression_stage_spec(
        prediction_head="point",
        basis_binding_mode=binding_mode,
        gradient_guidance_mode="basis_object_gradient",
        escape_policy=escape_policy,
        notes=(
            "Budgeted symbolic expression search over discovered basis objects "
            "with a constrained raw-feature escape lane."
            if escape_feature_names
            else "Budgeted symbolic expression search over discovered basis objects."
        ),
        metadata={
            "stage_key": "budgeted_symbolic_assembler",
            "selection_protocol": "basis_conditioned_expression_search",
            "basis_object_count": int(len(tuple(basis_feature_names))),
            "locked_basis_count": int(len(locked_basis_keys)),
            "chart_object_count": int(len(chart_object_keys)),
            "realization_object_count": int(len(realization_object_keys)),
            "regional_correction_count": int(regional_correction_count),
            "escape_object_count": int(len(escape_object_keys)),
            "escape_feature_names": escape_feature_names,
            "chart_canonicalization_protocol": str(cfg.chart_canonicalization_protocol),
            "chart_canonicalization_mode": str(cfg.chart_canonicalization_mode),
            "chart_orthodoxy_scoring_protocol": str(cfg.chart_orthodoxy_scoring_protocol),
            "chart_orthodoxy_scoring_mode": str(cfg.chart_orthodoxy_scoring_mode),
            "inner_chart_flip_compensation_protocol": str(cfg.inner_chart_flip_compensation_protocol),
            "inner_chart_flip_compensation_mode": str(cfg.inner_chart_flip_compensation_mode),
            "realization_prior_injection_protocol": str(cfg.realization_prior_injection_protocol),
            "realization_prior_injection_mode": str(cfg.realization_prior_injection_mode),
            "mandatory_realization_closure_protocol": str(cfg.mandatory_realization_closure_protocol),
            "mandatory_realization_closure_mode": str(cfg.mandatory_realization_closure_mode),
            "same_source_over_realization_protocol": str(cfg.same_source_over_realization_protocol),
            "same_source_over_realization_mode": str(cfg.same_source_over_realization_mode),
            "same_source_realization_budget": int(cfg.same_source_realization_budget),
            "periodic_realization_competition_protocol": str(cfg.periodic_realization_competition_protocol),
            "periodic_realization_competition_mode": str(cfg.periodic_realization_competition_mode),
            "regime_penetration_protocol": str(cfg.regime_penetration_protocol),
            "regime_penetration_mode": str(cfg.regime_penetration_mode),
            "regime_penetration_gain_floor": float(cfg.regime_penetration_gain_floor),
            "heterogeneous_exposure_protocol": str(cfg.heterogeneous_exposure_protocol),
            "heterogeneous_exposure_mode": str(cfg.heterogeneous_exposure_mode),
            "heterogeneous_exposure_candidate_screen_reserve": int(cfg.heterogeneous_exposure_candidate_screen_reserve),
            "heterogeneous_exposure_min_score": float(cfg.heterogeneous_exposure_min_score),
            "causal_hierarchy_reuse_isolation_protocol": str(cfg.causal_hierarchy_reuse_isolation_protocol),
            "causal_hierarchy_reuse_isolation_mode": str(cfg.causal_hierarchy_reuse_isolation_mode),
        },
    )
    basis_context = SymbolicBasisContext(
        basis_source="orthogonal_basis_discovery",
        binding_mode=binding_mode,
        equivalence_mode="family-level",
        selected_basis=_basis_object_refs(
            basis_object_records=object_records,
        ),
        locked_basis_keys=tuple(str(value) for value in tuple(locked_basis_keys)),
        metadata={
            "selected_basis_count": int(len(tuple(object_records))),
            "locked_basis_count": int(len(locked_basis_keys)),
            "chart_object_count": int(len(chart_object_keys)),
            "realization_object_count": int(len(realization_object_keys)),
            "regional_correction_count": int(regional_correction_count),
            "gate_feature_names": [str(value) for value in tuple(gate_feature_names)],
            "escape_policy": str(escape_policy),
            "escape_feature_names": escape_feature_names,
            "chart_object_keys": [str(value) for value in tuple(chart_object_keys)],
            "realization_object_keys": [str(value) for value in tuple(realization_object_keys)],
            "regional_correction_keys": [str(value) for value in tuple(regional_correction_keys)],
            "escape_object_keys": [str(value) for value in tuple(escape_object_keys)],
            "chart_canonicalization_protocol": str(cfg.chart_canonicalization_protocol),
            "chart_canonicalization_mode": str(cfg.chart_canonicalization_mode),
            "chart_orthodoxy_scoring_protocol": str(cfg.chart_orthodoxy_scoring_protocol),
            "chart_orthodoxy_scoring_mode": str(cfg.chart_orthodoxy_scoring_mode),
            "inner_chart_flip_compensation_protocol": str(cfg.inner_chart_flip_compensation_protocol),
            "inner_chart_flip_compensation_mode": str(cfg.inner_chart_flip_compensation_mode),
            "realization_prior_injection_protocol": str(cfg.realization_prior_injection_protocol),
            "realization_prior_injection_mode": str(cfg.realization_prior_injection_mode),
            "mandatory_realization_closure_protocol": str(cfg.mandatory_realization_closure_protocol),
            "mandatory_realization_closure_mode": str(cfg.mandatory_realization_closure_mode),
            "same_source_over_realization_protocol": str(cfg.same_source_over_realization_protocol),
            "same_source_over_realization_mode": str(cfg.same_source_over_realization_mode),
            "same_source_realization_budget": int(cfg.same_source_realization_budget),
            "periodic_realization_competition_protocol": str(cfg.periodic_realization_competition_protocol),
            "periodic_realization_competition_mode": str(cfg.periodic_realization_competition_mode),
            "regime_penetration_protocol": str(cfg.regime_penetration_protocol),
            "regime_penetration_mode": str(cfg.regime_penetration_mode),
            "regime_penetration_gain_floor": float(cfg.regime_penetration_gain_floor),
            "heterogeneous_exposure_protocol": str(cfg.heterogeneous_exposure_protocol),
            "heterogeneous_exposure_mode": str(cfg.heterogeneous_exposure_mode),
            "heterogeneous_exposure_candidate_screen_reserve": int(cfg.heterogeneous_exposure_candidate_screen_reserve),
            "heterogeneous_exposure_min_score": float(cfg.heterogeneous_exposure_min_score),
            "causal_hierarchy_reuse_isolation_protocol": str(cfg.causal_hierarchy_reuse_isolation_protocol),
            "causal_hierarchy_reuse_isolation_mode": str(cfg.causal_hierarchy_reuse_isolation_mode),
        },
    )
    return (
        {
            "basis_discovery": basis_discovery_stage.as_dict(),
            "assembler": assembler_stage.as_dict(),
        },
        assembler_stage.as_dict(),
        basis_context.as_dict(),
    )


def _build_basis_object_gradient_pool_report(
    *,
    inner_result: StructureSearchResult,
    stage_head_spec: Mapping[str, Any],
    basis_context: Mapping[str, Any],
) -> dict[str, Any]:
    iterations = [dict(item) for item in tuple(inner_result.iterations) if isinstance(item, Mapping)]
    if not iterations:
        return {
            "available": False,
            "protocol": "basis_object_gradient_pool_expansion_v1",
            "stage_head_spec": _jsonable(stage_head_spec),
            "basis_context": _jsonable(basis_context),
            "iteration_count": 0,
            "top_object_signals": [],
            "expansion_candidates": [],
        }
    last_iteration = dict(iterations[-1])
    gradient_summary = dict(last_iteration.get("gradient_summary", {}) or {})
    top_priority_rows = [
        dict(item)
        for item in tuple(gradient_summary.get("top_feature_priority", ()))
        if isinstance(item, Mapping)
    ]
    top_signals = [
        ObjectGradientSignal(
            object_key=str(item.get("feature") or ""),
            gradient_score=float(item.get("priority", 0.0) or 0.0),
            abs_gradient_score=float(item.get("priority_multiscale", item.get("priority", 0.0)) or 0.0),
            stability=float(item.get("stability", 1.0) or 1.0),
            metadata={
                "mismatch": item.get("mismatch"),
                "abs_gap_mean": item.get("abs_gap_mean"),
            },
        ).as_dict()
        for item in top_priority_rows
        if str(item.get("feature") or "").strip()
    ]
    top_candidate_rows = [
        dict(item)
        for item in tuple(last_iteration.get("top_candidates", ()))
        if isinstance(item, Mapping)
    ]
    expansion_candidates = [
        PoolExpansionCandidate(
            candidate_key=str(item.get("name") or f"candidate_{index}"),
            source_object_keys=tuple(
                str(value)
                for value in tuple(item.get("feature_labels", ()))
                if str(value).strip()
            ),
            expression=str(item.get("name") or f"candidate_{index}"),
            priority=float(item.get("score", 0.0) or 0.0),
            metadata={
                "family": str(item.get("family") or ""),
                "feature_indices": [int(v) for v in tuple(item.get("feature_indices", ()))],
                "score_parts": _jsonable(dict(item.get("score_parts", {}) or {})),
            },
        ).as_dict()
        for index, item in enumerate(top_candidate_rows)
    ]
    return {
        "available": True,
        "protocol": "basis_object_gradient_pool_expansion_v1",
        "stage_head_spec": _jsonable(stage_head_spec),
        "basis_context": _jsonable(basis_context),
        "iteration_count": int(len(iterations)),
        "last_iteration": int(last_iteration.get("iteration", len(iterations))),
        "top_object_signals": top_signals,
        "expansion_candidates": expansion_candidates,
    }


def _substitute_basis_expr(expr: Mapping[str, Any], basis_genome: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    node = dict(expr)
    kind = str(node.get("type", "")).strip().lower()
    if kind == "feature":
        basis_index = int(node.get("index", -1))
        if basis_index < 0 or basis_index >= len(tuple(basis_genome)):
            raise IndexError(f"basis feature index out of range: {basis_index}")
        return dict(dict(tuple(basis_genome)[basis_index]).get("expr", {}))
    if kind == "unary":
        return {
            "type": "unary",
            "op": str(node.get("op", "")),
            "arg": _substitute_basis_expr(dict(node.get("arg", {})), basis_genome),
        }
    if kind == "binary":
        return {
            "type": "binary",
            "op": str(node.get("op", "")),
            "left": _substitute_basis_expr(dict(node.get("left", {})), basis_genome),
            "right": _substitute_basis_expr(dict(node.get("right", {})), basis_genome),
        }
    return dict(node)


def _substitute_basis_genome(
    genome: Sequence[Mapping[str, Any]],
    *,
    basis_genome: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    out: list[dict[str, Any]] = []
    for index, term in enumerate(tuple(genome)):
        substituted_expr = _substitute_basis_expr(dict(term.get("expr", {})), basis_genome)
        expression_name = expression_to_string(substituted_expr, precision=8)
        fallback_name = str(term.get("name", f"inner_term_{index}")).strip() or f"inner_term_{index}"
        out.append(
            {
                "name": expression_name if len(expression_name) <= 96 else fallback_name,
                "expr": dict(substituted_expr),
            }
        )
    return tuple(out)


def _build_budgeted_symbolic_search_config(
    *,
    cfg: OrthogonalBasisSearchConfig,
    basis_feature_count: int,
) -> StructureSearchConfig:
    feature_cap = max(1, int(basis_feature_count))
    max_pair_terms = max(0, int(cfg.assembler_max_pair_terms))
    return StructureSearchConfig(
        max_added_terms=int(max(1, cfg.assembler_max_added_terms)),
        topk_features=int(min(feature_cap, max(1, cfg.assembler_topk_features))),
        max_pair_terms=int(max_pair_terms),
        max_candidates_per_iter=int(max(cfg.assembler_candidate_keep_top, cfg.assembler_max_candidates_per_iter)),
        candidate_keep_top=int(max(1, cfg.assembler_candidate_keep_top)),
        candidate_pool_mode="shared_full",
        ridge_l2=float(cfg.assembler_ridge_l2),
        include_hinge=bool(cfg.enable_piecewise_basis),
        hinge_quantiles=tuple(float(value) for value in tuple(cfg.assembler_hinge_quantiles)),
        max_arity=3,
        max_expr_depth=int(max(2, cfg.assembler_max_expr_depth)),
        path_memory_enabled=bool(cfg.assembler_path_memory_enabled),
        path_memory_namespace="orthogonal_inner_symbolic",
        graph_cache_enabled=bool(cfg.assembler_graph_cache_enabled),
        graph_cache_namespace="orthogonal_inner_symbolic",
        joint_bundle_enabled=False,
        overfit_guard_enabled=False,
    )


def _expr_collect_feature_indices(expr: Mapping[str, Any]) -> set[int]:
    node = dict(expr)
    kind = str(node.get("type", "")).strip().lower()
    if kind == "feature":
        return {int(node.get("index", -1))}
    if kind == "unary":
        return _expr_collect_feature_indices(dict(node.get("arg", {})))
    if kind == "binary":
        return _expr_collect_feature_indices(dict(node.get("left", {}))) | _expr_collect_feature_indices(
            dict(node.get("right", {}))
        )
    if kind == "piecewise":
        return (
            _expr_collect_feature_indices(dict(node.get("condition", {})))
            | _expr_collect_feature_indices(dict(node.get("then", {})))
            | _expr_collect_feature_indices(dict(node.get("else", {})))
        )
    return set()


def _same_source_over_realization_report(
    *,
    basis_space_genome: Sequence[Mapping[str, Any]],
    basis_feature_names: Sequence[str],
    basis_object_records: Sequence[Mapping[str, Any]],
    cfg: OrthogonalBasisSearchConfig,
) -> dict[str, Any]:
    if not _same_source_over_realization_enabled(cfg):
        return {
            "protocol": str(cfg.same_source_over_realization_protocol),
            "mode": str(cfg.same_source_over_realization_mode),
            "status": "disabled",
            "duplicate_group_count": 0,
            "penalty": 0.0,
            "score": 1.0,
        }
    record_by_key = {
        str(dict(record).get("object_key") or ""): dict(record)
        for record in tuple(basis_object_records)
        if str(dict(record).get("object_key") or "").strip()
    }
    selected_object_keys: list[str] = []
    for term in tuple(basis_space_genome):
        indices = sorted(idx for idx in _expr_collect_feature_indices(dict(term.get("expr", {}))) if idx >= 0)
        for idx in indices:
            if idx < len(tuple(basis_feature_names)):
                selected_object_keys.append(str(tuple(basis_feature_names)[idx]))
    selected_object_keys = list(dict.fromkeys(key for key in selected_object_keys if key in record_by_key))
    relevant_records = [
        dict(record_by_key[key])
        for key in selected_object_keys
        if str(record_by_key[key].get("binding_role") or "") in {
            "locked_basis_object",
            "realization_competitor",
            "chart_competitor",
        }
    ]
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in tuple(relevant_records):
        source_key = str(
            record.get("source_information_key")
            or record.get("source_object_key")
            or record.get("parent_object_key")
            or record.get("object_key")
            or ""
        )
        groups.setdefault(source_key, []).append(record)
    group_reports: list[dict[str, Any]] = []
    overflow_total = 0
    for source_key, records in groups.items():
        budget = max(1, int(cfg.same_source_realization_budget))
        if any(str(record.get("object_kind") or "") == "periodic_channel" for record in tuple(records)):
            budget = max(budget, 2)
        overflow = max(0, len(records) - budget)
        overflow_total += int(overflow)
        group_reports.append(
            {
                "source_key": str(source_key),
                "budget": int(budget),
                "selected_count": int(len(records)),
                "overflow": int(overflow),
                "selected_object_keys": [str(record.get("object_key") or "") for record in tuple(records)],
                "selected_binding_roles": [str(record.get("binding_role") or "") for record in tuple(records)],
                "selected_realization_signatures": [
                    str(record.get("realization_signature") or record.get("realization_head_signature") or "")
                    for record in tuple(records)
                ],
            }
        )
    duplicate_groups = [group for group in tuple(group_reports) if int(group.get("overflow", 0) or 0) > 0]
    selected_count = max(1, len(relevant_records))
    penalty = float(min(1.0, float(overflow_total) / float(selected_count)))
    return {
        "protocol": str(cfg.same_source_over_realization_protocol),
        "mode": str(cfg.same_source_over_realization_mode),
        "status": "reported",
        "selected_object_count": int(len(relevant_records)),
        "duplicate_group_count": int(len(duplicate_groups)),
        "penalty": float(penalty),
        "score": float(np.clip(1.0 - penalty, 0.0, 1.0)),
        "groups": _jsonable(group_reports),
    }


def _inner_fit_score(metrics: Mapping[str, Any]) -> float:
    rmse = metrics.get("rmse")
    r2 = metrics.get("r2")
    rmse_component = 0.0
    r2_component = 0.0
    if rmse is not None and np.isfinite(float(rmse)):
        rmse_component = float(1.0 / (1.0 + max(0.0, float(rmse))))
    if r2 is not None and np.isfinite(float(r2)):
        r2_component = float((np.clip(float(r2), -1.0, 1.0) + 1.0) * 0.5)
    return float(0.70 * rmse_component + 0.30 * r2_component)


def _outer_objective_payload(
    *,
    inner_metrics: Mapping[str, Any],
    orthogonality_metrics: Mapping[str, Any],
    residual_report: Mapping[str, Any],
    semantic_report: Mapping[str, Any],
    interference_report: Mapping[str, Any],
    periodic_report: Mapping[str, Any],
    regional_correction_report: Mapping[str, Any],
    same_source_realization_report: Mapping[str, Any],
    environment_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    inner_fit_score = float(_inner_fit_score(inner_metrics))
    orthogonality_score = float(orthogonality_metrics.get("orthogonality_score", 0.0) or 0.0)
    residual_gain_mean = float(residual_report.get("mean_marginal_r2_gain", 0.0) or 0.0)
    semantic_unique_ratio = float(semantic_report.get("semantic_unique_ratio", 0.0) or 0.0)
    trivial_penalty_mean = float(interference_report.get("trivial_nonlinearity_penalty_mean", 0.0) or 0.0)
    periodic_score = float(periodic_report.get("overall_periodic_disambiguation_score", 0.0) or 0.0)
    periodic_penalty = float(periodic_report.get("local_equivalence_penalty_mean", 0.0) or 0.0)
    regional_correction_score = float(regional_correction_report.get("regional_correction_score", 0.0) or 0.0)
    same_source_penalty = float(same_source_realization_report.get("penalty", 0.0) or 0.0)
    environment_invariance_score = 0.0
    if isinstance(environment_audit, Mapping):
        environment_invariance_score = float(environment_audit.get("overall_invariance_score", 0.0) or 0.0)
    outer_score = float(
        _OUTER_OBJECTIVE_INNER_FIT_WEIGHT * inner_fit_score
        + _OUTER_OBJECTIVE_ORTHOGONALITY_WEIGHT * orthogonality_score
        + _OUTER_OBJECTIVE_RESIDUAL_WEIGHT * residual_gain_mean
        + _OUTER_OBJECTIVE_SEMANTIC_WEIGHT * semantic_unique_ratio
        - _OUTER_OBJECTIVE_INTERFERENCE_PENALTY_WEIGHT * trivial_penalty_mean
        + _OUTER_OBJECTIVE_PERIODIC_WEIGHT * periodic_score
        - _OUTER_OBJECTIVE_PERIODIC_PENALTY_WEIGHT * periodic_penalty
        + _OUTER_OBJECTIVE_REGIONAL_CORRECTION_WEIGHT * regional_correction_score
        - _OUTER_OBJECTIVE_SAME_SOURCE_REALIZATION_PENALTY_WEIGHT * same_source_penalty
    )
    return {
        "protocol": "orthogonal_structure_search_with_budgeted_symbolic_assembler",
        "inner_fit_score": float(inner_fit_score),
        "orthogonality_score": float(orthogonality_score),
        "residual_complementarity_score": float(residual_gain_mean),
        "semantic_dedup_score": float(semantic_unique_ratio),
        "trivial_nonlinearity_penalty": float(trivial_penalty_mean),
        "periodic_equivalence_score": float(periodic_score),
        "periodic_equivalence_penalty": float(periodic_penalty),
        "regional_correction_score": float(regional_correction_score),
        "same_source_realization_penalty": float(same_source_penalty),
        "environment_invariance_score": float(environment_invariance_score),
        "outer_score": float(outer_score),
        "weights": {
            "inner_fit": float(_OUTER_OBJECTIVE_INNER_FIT_WEIGHT),
            "orthogonality": float(_OUTER_OBJECTIVE_ORTHOGONALITY_WEIGHT),
            "residual_complementarity": float(_OUTER_OBJECTIVE_RESIDUAL_WEIGHT),
            "semantic_dedup": float(_OUTER_OBJECTIVE_SEMANTIC_WEIGHT),
            "trivial_nonlinearity_penalty": float(_OUTER_OBJECTIVE_INTERFERENCE_PENALTY_WEIGHT),
            "periodic_equivalence": float(_OUTER_OBJECTIVE_PERIODIC_WEIGHT),
            "periodic_equivalence_penalty": float(_OUTER_OBJECTIVE_PERIODIC_PENALTY_WEIGHT),
            "regional_correction": float(_OUTER_OBJECTIVE_REGIONAL_CORRECTION_WEIGHT),
            "same_source_realization_penalty": float(_OUTER_OBJECTIVE_SAME_SOURCE_REALIZATION_PENALTY_WEIGHT),
        },
        "inner_metrics": _jsonable(dict(inner_metrics)),
        "interference_report": _jsonable(dict(interference_report)),
        "periodic_equivalence_report": _jsonable(dict(periodic_report)),
        "regional_correction_report": _jsonable(dict(regional_correction_report)),
        "same_source_over_realization_report": _jsonable(dict(same_source_realization_report)),
        "environment_invariance_audit": _jsonable(dict(environment_audit or {})),
    }


def _build_orthogonal_fold_report(
    *,
    pool_indices: Sequence[int],
    basis_rows: Sequence[Mapping[str, Any]],
    final_metrics: Mapping[str, Any],
    search_cfg: StructureSearchConfig,
    inner_result: StructureSearchResult,
    basis_feature_names: Sequence[str],
) -> dict[str, Any]:
    rmse_value = float(dict(final_metrics).get("rmse", 0.0) or 0.0)
    return {
        "objective_schema": ["rmse"],
        "subset_size": int(len(tuple(basis_rows))),
        "subset_idx": [int(value) for value in tuple(pool_indices)],
        "subset_names": [str(dict(row).get("term_name", "")) for row in tuple(basis_rows)],
        "subset_families": [str(dict(row).get("semantic_family", "")) for row in tuple(basis_rows)],
        "fold_rmse": [float(rmse_value)],
        "rmse_mean": float(rmse_value),
        "rmse_std": 0.0,
        "rmse_drift": 0.0,
        "tuned_l2": float(search_cfg.ridge_l2),
        "decode_meta": {
            "protocol": "orthogonal_budgeted_symbolic_assembler",
            "basis_feature_names": [str(value) for value in tuple(basis_feature_names)],
            "inner_iterations": int(len(tuple(inner_result.iterations))),
            "inner_terms": int(len(tuple(inner_result.genome))),
        },
    }


def _final_fit_preference_key(final_fit: Mapping[str, Any], *, n_terms: int) -> tuple[float, float, float, float]:
    metrics = dict(final_fit.get("metrics_train", {}) or {})
    inner_fit = float(_inner_fit_score(metrics))
    raw_r2 = metrics.get("r2")
    raw_rmse = metrics.get("rmse")
    r2_value = float(raw_r2) if raw_r2 is not None else float("-inf")
    rmse_value = float(raw_rmse) if raw_rmse is not None else float("inf")
    if not np.isfinite(r2_value):
        r2_value = float("-inf")
    if not np.isfinite(rmse_value):
        rmse_value = float("inf")
    return (
        float(inner_fit),
        float(r2_value),
        -float(rmse_value),
        -float(max(0, int(n_terms))),
    )


def _basis_space_identity_genome_from_object_keys(
    *,
    object_keys: Sequence[str],
    object_index_lookup: Mapping[str, int],
) -> tuple[dict[str, Any], ...]:
    genome: list[dict[str, Any]] = []
    seen: set[str] = set()
    for object_key in tuple(object_keys):
        key = str(object_key or "").strip()
        if not key or key in seen:
            continue
        feature_index = object_index_lookup.get(key)
        if feature_index is None:
            continue
        seen.add(key)
        genome.append({"name": key, "expr": _feature_expr(int(feature_index))})
    return tuple(genome)


def _mandatory_realization_candidate_records(
    *,
    basis_object_records: Sequence[Mapping[str, Any]],
    cfg: OrthogonalBasisSearchConfig,
) -> tuple[dict[str, Any], ...]:
    if not _mandatory_realization_closure_enabled(cfg):
        return tuple()
    candidates: list[dict[str, Any]] = []
    for raw_record in tuple(basis_object_records):
        record = dict(raw_record)
        realization_key = str(record.get("object_key") or "").strip()
        binding_role = str(record.get("binding_role") or "").strip()
        if binding_role == "realization_competitor":
            signature = str(record.get("realization_signature") or record.get("realization_head_signature") or "").strip()
            closure_role = "realization_replacement"
            protocols = tuple(
                str(value)
                for value in tuple(record.get("realization_protocols", ()))
                if str(value).strip()
            )
            evidence_term_names = tuple(
                str(value)
                for value in tuple(record.get("realization_evidence_term_names", ()))
                if str(value).strip()
            )
            evidence_screen_score = float(record.get("realization_evidence_screen_score", 0.0) or 0.0)
            evidence_residual_gain = float(record.get("realization_evidence_residual_gain", 0.0) or 0.0)
            forced_finalist = bool(record.get("realization_forced_finalist", False))
        elif binding_role == "regional_branch_competitor":
            signature = str(record.get("branch_signature") or "").strip()
            closure_role = "regional_branch_additive"
            protocols = tuple(
                str(value)
                for value in tuple(record.get("branch_protocols", ()))
                if str(value).strip()
            )
            evidence_term_names = tuple(
                str(value)
                for value in tuple(record.get("branch_evidence_term_names", ()))
                if str(value).strip()
            )
            evidence_screen_score = float(record.get("branch_evidence_score", 0.0) or 0.0)
            evidence_residual_gain = float(record.get("branch_marginal_r2_gain", 0.0) or 0.0)
            forced_finalist = bool(record.get("branch_forced_finalist", True))
        else:
            continue
        if not realization_key or not signature:
            continue
        candidates.append(
            {
                "object_key": realization_key,
                "parent_object_key": str(record.get("parent_object_key") or "").strip(),
                "realization_signature": signature,
                "closure_role": str(closure_role),
                "source_information_key": str(record.get("source_information_key") or "").strip(),
                "required_realization_family": str(record.get("required_realization_family") or "").strip(),
                "protocols": protocols,
                "forced_finalist": bool(forced_finalist),
                "evidence_term_names": evidence_term_names,
                "evidence_screen_score": float(evidence_screen_score),
                "evidence_residual_gain": float(evidence_residual_gain),
                "branch_threshold": (
                    float(record.get("branch_threshold"))
                    if record.get("branch_threshold") is not None
                    and np.isfinite(float(record.get("branch_threshold")))
                    else None
                ),
                "branch_direction": str(record.get("branch_direction") or ""),
                "threshold_orthodoxy_score": float(record.get("threshold_orthodoxy_score", 0.0) or 0.0),
                "threshold_stability_score": float(record.get("threshold_stability_score", 0.0) or 0.0),
                "threshold_balance_score": float(record.get("threshold_balance_score", 0.0) or 0.0),
                "threshold_audit": _jsonable(dict(record.get("threshold_audit", {}) or {})),
                "threshold_selection_lane": str(record.get("threshold_selection_lane") or ""),
            }
        )
    candidates.sort(
        key=lambda item: (
            -int(bool(item.get("required_realization_family"))),
            -float(item.get("evidence_screen_score", 0.0)),
            -float(item.get("evidence_residual_gain", 0.0)),
            str(item.get("object_key", "")),
        )
    )
    return tuple(candidates)


def _run_mandatory_realization_closure(
    *,
    current_basis_space_genome: Sequence[Mapping[str, Any]],
    current_assembled_genome: Sequence[Mapping[str, Any]],
    current_final_fit: Mapping[str, Any],
    inner_result: StructureSearchResult,
    assembler_basis_genome: Sequence[Mapping[str, Any]],
    basis_feature_names: Sequence[str],
    basis_object_records: Sequence[Mapping[str, Any]],
    raw_X: np.ndarray,
    target: np.ndarray,
    search_cfg: StructureSearchConfig,
    graph_cache: ExpressionGraphCache | None,
    cfg: OrthogonalBasisSearchConfig,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    Mapping[str, Any],
    StructureSearchResult,
    dict[str, Any],
]:
    def _signature_label(signature: str) -> str:
        key = str(signature or "").strip().lower()
        if key == "unary:exp_neg":
            return "exp(-source)"
        if key == "unary:exp":
            return "exp(source)"
        if key.startswith("unary:"):
            return f"{key.split(':', 1)[1]}(source)"
        if key == "branch:hinge_pos":
            return "hinge(source-threshold)"
        if key == "branch:hinge_neg":
            return "hinge(threshold-source)"
        if key.startswith("branch:"):
            return key.split(":", 1)[1]
        return key or "unknown"

    def _genome_term_expressions(genome: Sequence[Mapping[str, Any]]) -> list[str]:
        out: list[str] = []
        for term in tuple(genome):
            expr = dict(term.get("expr", {}))
            if expr:
                out.append(expression_to_string(expr, precision=10))
            else:
                out.append(str(term.get("name", "")))
        return out

    mandatory_records = _mandatory_realization_candidate_records(
        basis_object_records=basis_object_records,
        cfg=cfg,
    )
    if not _mandatory_realization_closure_enabled(cfg):
        return (
            tuple(dict(term) for term in tuple(current_basis_space_genome)),
            tuple(dict(term) for term in tuple(current_assembled_genome)),
            dict(current_final_fit),
            inner_result,
            {
                "protocol": str(cfg.mandatory_realization_closure_protocol),
                "mode": str(cfg.mandatory_realization_closure_mode),
                "status": "disabled",
            },
        )
    no_mandatory_candidates = len(tuple(mandatory_records)) <= 0

    object_index_lookup = {
        str(object_key): int(index)
        for index, object_key in enumerate(tuple(str(value) for value in tuple(basis_feature_names)))
    }
    locked_basis_records = [
        dict(record)
        for record in tuple(basis_object_records)
        if str(dict(record).get("binding_role") or "") == "locked_basis_object"
        and str(dict(record).get("object_key") or "").strip()
    ]
    locked_basis_keys = [str(record.get("object_key") or "") for record in tuple(locked_basis_records)]
    auxiliary_keys = [
        str(dict(record).get("object_key") or "")
        for record in tuple(basis_object_records)
        if str(dict(record).get("binding_role") or "") in {"regional_correction", "escape"}
        and str(dict(record).get("object_key") or "").strip()
    ]
    max_terms = int(max(1, cfg.assembler_max_added_terms))
    current_key = _final_fit_preference_key(
        current_final_fit,
        n_terms=int(len(tuple(current_basis_space_genome))),
    )
    best_key = current_key
    best_basis_space_genome = tuple(dict(term) for term in tuple(current_basis_space_genome))
    best_assembled_genome = tuple(dict(term) for term in tuple(current_assembled_genome))
    best_final_fit: Mapping[str, Any] = dict(current_final_fit)
    best_candidate_summary: dict[str, Any] | None = None
    evaluated_candidates: list[dict[str, Any]] = []
    seen_key_sets: set[tuple[str, ...]] = set()
    current_candidate_summary = {
        "candidate_id": "inner_current_result",
        "object_keys": [],
        "parent_object_key": "",
        "source_information_key": "",
        "realization_signature": "inner_search_mixed",
        "realization_label": "inner_search_mixed",
        "required_realization_family": "",
        "evidence_term_names": [],
        "evidence_screen_score": 0.0,
        "evidence_residual_gain": 0.0,
        "metrics_train": _jsonable(dict(current_final_fit.get("metrics_train", {}) or {})),
        "inner_fit_score": float(_inner_fit_score(dict(current_final_fit.get("metrics_train", {}) or {}))),
        "term_count": int(len(tuple(current_basis_space_genome))),
        "finalist_entered": True,
        "winner_basis_term_expressions": _genome_term_expressions(current_basis_space_genome),
    }

    for record in tuple(mandatory_records):
        realization_key = str(record.get("object_key") or "").strip()
        parent_key = str(record.get("parent_object_key") or "").strip()
        source_information_key = str(record.get("source_information_key") or "").strip()
        closure_role = str(record.get("closure_role") or "realization_replacement").strip().lower()
        if closure_role == "regional_branch_additive":
            missing_object_keys = [realization_key] if realization_key not in object_index_lookup else []
            if missing_object_keys:
                evaluated_candidates.append(
                    {
                        "candidate_id": f"{realization_key}::branch_with_current",
                        "object_keys": [str(realization_key)],
                        "parent_object_key": str(parent_key),
                        "source_information_key": str(source_information_key),
                        "realization_signature": str(record.get("realization_signature") or ""),
                        "closure_role": str(closure_role),
                        "realization_label": _signature_label(str(record.get("realization_signature") or "")),
                        "required_realization_family": str(record.get("required_realization_family") or ""),
                        "forced_finalist": bool(record.get("forced_finalist", False)),
                        "evidence_term_names": [str(value) for value in tuple(record.get("evidence_term_names", ()))],
                        "evidence_screen_score": float(record.get("evidence_screen_score", 0.0) or 0.0),
                        "evidence_residual_gain": float(record.get("evidence_residual_gain", 0.0) or 0.0),
                        "metrics_train": None,
                        "inner_fit_score": float("-inf"),
                        "term_count": int(len(tuple(current_basis_space_genome))),
                        "finalist_entered": False,
                        "finalist_block_reason": "missing_object_key",
                        "missing_object_keys": [str(value) for value in tuple(missing_object_keys)],
                    }
                )
                continue
            branch_index = int(object_index_lookup[str(realization_key)])
            branch_expr = {"type": "feature", "index": int(branch_index)}
            already_present = any(
                _candidate_expr_key(dict(term.get("expr", {}))) == _candidate_expr_key(branch_expr)
                for term in tuple(current_basis_space_genome)
            )
            basis_space_genome = tuple(dict(term) for term in tuple(current_basis_space_genome))
            if not already_present:
                basis_space_genome = tuple(
                    [
                        *basis_space_genome,
                        {
                            "name": str(realization_key),
                            "expr": dict(branch_expr),
                        },
                    ]
                )
            candidate_id = f"{realization_key}::branch_with_current"
            dedup_key = tuple(
                str(term.get("name") or _candidate_expr_key(dict(term.get("expr", {}))))
                for term in tuple(basis_space_genome)
            )
            if dedup_key in seen_key_sets:
                continue
            seen_key_sets.add(dedup_key)
            assembled_genome = _substitute_basis_genome(
                basis_space_genome,
                basis_genome=tuple(dict(term) for term in tuple(assembler_basis_genome)),
            )
            final_fit = evaluate_genome_with_ridge(
                assembled_genome,
                X_train=np.asarray(raw_X, dtype=float),
                y_train=np.asarray(target, dtype=float),
                l2=float(search_cfg.ridge_l2),
                graph_cache=graph_cache,
                train_batch_key=f"orthogonal_mandatory_branch::{candidate_id}",
            )
            metrics_train = dict(final_fit.get("metrics_train", {}) or {})
            preference_key = _final_fit_preference_key(
                final_fit,
                n_terms=int(len(tuple(basis_space_genome))),
            )
            summary = {
                "candidate_id": str(candidate_id),
                "object_keys": [str(realization_key)],
                "parent_object_key": str(parent_key),
                "source_information_key": str(source_information_key),
                "realization_signature": str(record.get("realization_signature") or ""),
                "closure_role": str(closure_role),
                "realization_label": _signature_label(str(record.get("realization_signature") or "")),
                "required_realization_family": str(record.get("required_realization_family") or ""),
                "forced_finalist": bool(record.get("forced_finalist", False)),
                "evidence_term_names": [str(value) for value in tuple(record.get("evidence_term_names", ()))],
                "evidence_screen_score": float(record.get("evidence_screen_score", 0.0) or 0.0),
                "evidence_residual_gain": float(record.get("evidence_residual_gain", 0.0) or 0.0),
                "branch_threshold": record.get("branch_threshold"),
                "branch_direction": str(record.get("branch_direction") or ""),
                "threshold_orthodoxy_score": float(record.get("threshold_orthodoxy_score", 0.0) or 0.0),
                "threshold_stability_score": float(record.get("threshold_stability_score", 0.0) or 0.0),
                "threshold_balance_score": float(record.get("threshold_balance_score", 0.0) or 0.0),
                "threshold_audit": _jsonable(dict(record.get("threshold_audit", {}) or {})),
                "threshold_selection_lane": str(record.get("threshold_selection_lane") or ""),
                "metrics_train": _jsonable(metrics_train),
                "inner_fit_score": float(_inner_fit_score(metrics_train)),
                "term_count": int(len(tuple(basis_space_genome))),
                "finalist_entered": True,
                "winner_basis_term_expressions": _genome_term_expressions(basis_space_genome),
            }
            evaluated_candidates.append(summary)
            if preference_key > best_key:
                best_key = preference_key
                best_basis_space_genome = tuple(dict(term) for term in tuple(basis_space_genome))
                best_assembled_genome = tuple(dict(term) for term in tuple(assembled_genome))
                best_final_fit = dict(final_fit)
                best_candidate_summary = dict(summary)
            continue
        candidate_specs: list[tuple[str, list[str]]] = [(f"{realization_key}::closure_only", [realization_key])]
        locked_support_keys: list[str] = [realization_key]
        for locked_key in tuple(locked_basis_keys):
            chosen_key = realization_key if locked_key == parent_key else str(locked_key)
            if chosen_key and chosen_key not in locked_support_keys:
                locked_support_keys.append(chosen_key)
        for aux_key in tuple(auxiliary_keys):
            if len(locked_support_keys) >= max_terms:
                break
            if aux_key and aux_key not in locked_support_keys:
                locked_support_keys.append(aux_key)
        locked_support_keys = locked_support_keys[:max_terms]
        if locked_support_keys and tuple(locked_support_keys) != (realization_key,):
            candidate_specs.append((f"{realization_key}::closure_with_support", list(locked_support_keys)))
        for candidate_id, object_keys in tuple(candidate_specs):
            dedup_key = tuple(str(value) for value in tuple(object_keys) if str(value).strip())
            if not dedup_key or dedup_key in seen_key_sets:
                continue
            seen_key_sets.add(dedup_key)
            missing_object_keys = [str(value) for value in tuple(dedup_key) if str(value) not in object_index_lookup]
            if missing_object_keys:
                evaluated_candidates.append(
                    {
                        "candidate_id": str(candidate_id),
                        "object_keys": [str(value) for value in tuple(dedup_key)],
                        "parent_object_key": str(parent_key),
                        "source_information_key": str(source_information_key),
                        "realization_signature": str(record.get("realization_signature") or ""),
                        "realization_label": _signature_label(str(record.get("realization_signature") or "")),
                        "required_realization_family": str(record.get("required_realization_family") or ""),
                        "forced_finalist": bool(record.get("forced_finalist", False)),
                        "evidence_term_names": [str(value) for value in tuple(record.get("evidence_term_names", ()))],
                        "evidence_screen_score": float(record.get("evidence_screen_score", 0.0) or 0.0),
                        "evidence_residual_gain": float(record.get("evidence_residual_gain", 0.0) or 0.0),
                        "metrics_train": None,
                        "inner_fit_score": float("-inf"),
                        "term_count": int(len(tuple(dedup_key))),
                        "finalist_entered": False,
                        "finalist_block_reason": "missing_object_key",
                        "missing_object_keys": [str(value) for value in tuple(missing_object_keys)],
                    }
                )
                continue
            basis_space_genome = _basis_space_identity_genome_from_object_keys(
                object_keys=dedup_key,
                object_index_lookup=object_index_lookup,
            )
            if not basis_space_genome:
                continue
            assembled_genome = _substitute_basis_genome(
                basis_space_genome,
                basis_genome=tuple(dict(term) for term in tuple(assembler_basis_genome)),
            )
            final_fit = evaluate_genome_with_ridge(
                assembled_genome,
                X_train=np.asarray(raw_X, dtype=float),
                y_train=np.asarray(target, dtype=float),
                l2=float(search_cfg.ridge_l2),
                graph_cache=graph_cache,
                train_batch_key=f"orthogonal_mandatory_realization::{candidate_id}",
            )
            metrics_train = dict(final_fit.get("metrics_train", {}) or {})
            preference_key = _final_fit_preference_key(
                final_fit,
                n_terms=int(len(tuple(basis_space_genome))),
            )
            summary = {
                "candidate_id": str(candidate_id),
                "object_keys": [str(value) for value in tuple(dedup_key)],
                "parent_object_key": str(parent_key),
                "source_information_key": str(source_information_key),
                "realization_signature": str(record.get("realization_signature") or ""),
                "realization_label": _signature_label(str(record.get("realization_signature") or "")),
                "required_realization_family": str(record.get("required_realization_family") or ""),
                "forced_finalist": bool(record.get("forced_finalist", False)),
                "evidence_term_names": [str(value) for value in tuple(record.get("evidence_term_names", ()))],
                "evidence_screen_score": float(record.get("evidence_screen_score", 0.0) or 0.0),
                "evidence_residual_gain": float(record.get("evidence_residual_gain", 0.0) or 0.0),
                "metrics_train": _jsonable(metrics_train),
                "inner_fit_score": float(_inner_fit_score(metrics_train)),
                "term_count": int(len(tuple(basis_space_genome))),
                "finalist_entered": True,
                "winner_basis_term_expressions": _genome_term_expressions(basis_space_genome),
            }
            evaluated_candidates.append(summary)
            if preference_key > best_key:
                best_key = preference_key
                best_basis_space_genome = tuple(dict(term) for term in tuple(basis_space_genome))
                best_assembled_genome = tuple(dict(term) for term in tuple(assembled_genome))
                best_final_fit = dict(final_fit)
                best_candidate_summary = dict(summary)

    improved = best_candidate_summary is not None
    winner_candidate_summary = (
        dict(best_candidate_summary)
        if best_candidate_summary is not None
        else dict(current_candidate_summary)
    )
    winner_basis_term_expressions = (
        _genome_term_expressions(best_basis_space_genome)
        if best_candidate_summary is not None
        else _genome_term_expressions(current_basis_space_genome)
    )
    winner_signature = str(winner_candidate_summary.get("realization_signature") or "inner_search_mixed")

    evaluated_by_realization_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in tuple(evaluated_candidates):
        parent_key = str(item.get("parent_object_key") or "")
        signature = str(item.get("realization_signature") or "")
        object_key = str(tuple(item.get("object_keys", ()) or ("",))[0] or "")
        if not parent_key or not signature:
            continue
        key = (parent_key, signature, object_key)
        previous = evaluated_by_realization_key.get(key)
        if previous is None or float(item.get("inner_fit_score", float("-inf")) or float("-inf")) > float(
            previous.get("inner_fit_score", float("-inf")) or float("-inf")
        ):
            evaluated_by_realization_key[key] = dict(item)

    mandatory_by_realization_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in tuple(mandatory_records):
        key = (
            str(record.get("parent_object_key") or ""),
            str(record.get("realization_signature") or ""),
            str(record.get("object_key") or ""),
        )
        mandatory_by_realization_key[key] = dict(record)

    audit_rows: list[dict[str, Any]] = []
    expected_signatures = {"unary:exp", "unary:exp_neg"}
    expected_signatures.update(
        str(record.get("realization_signature") or "")
        for record in tuple(mandatory_records)
        if str(record.get("realization_signature") or "").startswith("branch:")
    )
    seen_audit_keys: set[tuple[str, str, str]] = set()
    for locked_record in tuple(locked_basis_records):
        parent_key = str(locked_record.get("object_key") or "").strip()
        source_information_key = str(locked_record.get("source_information_key") or "").strip()
        catalog = tuple(locked_record.get("realization_signature_catalog", ())) + tuple(
            locked_record.get("regional_branch_signature_catalog", ())
        )
        for raw in tuple(catalog):
            item = dict(raw)
            signature = str(item.get("signature") or "").strip()
            if signature.startswith("branch:"):
                continue
            if signature not in expected_signatures:
                continue
            row_key = (parent_key, signature, "")
            seen_audit_keys.add(row_key)
            mandatory_key = next(
                (
                    key
                    for key in mandatory_by_realization_key
                    if key[0] == parent_key and key[1] == signature
                ),
                row_key,
            )
            evaluated_key = next(
                (
                    key
                    for key in evaluated_by_realization_key
                    if key[0] == parent_key and key[1] == signature
                ),
                row_key,
            )
            mandatory_record = dict(mandatory_by_realization_key.get(mandatory_key, {}))
            evaluated_entry = dict(evaluated_by_realization_key.get(evaluated_key, {}))
            generated = bool(mandatory_record)
            entered_finalist = bool(evaluated_entry) and bool(evaluated_entry.get("finalist_entered", False))
            if not generated:
                generation_status = "not_generated"
                finalist_status = "not_entered"
                outcome = "not_generated"
            elif not entered_finalist:
                generation_status = "generated"
                finalist_status = "not_entered"
                outcome = "not_entered"
            elif best_candidate_summary is not None and str(best_candidate_summary.get("candidate_id")) == str(
                evaluated_entry.get("candidate_id")
            ):
                generation_status = "generated"
                finalist_status = "entered"
                outcome = "selected"
            else:
                generation_status = "generated"
                finalist_status = "entered"
                outcome = "lost"
            audit_rows.append(
                {
                    "parent_object_key": str(parent_key),
                    "source_information_key": str(source_information_key),
                    "realization_signature": str(signature),
                    "closure_role": str(mandatory_record.get("closure_role") or ""),
                    "realization_label": _signature_label(signature),
                    "generation_status": str(generation_status),
                    "catalog_selected": bool(item.get("selected", False)),
                    "catalog_selection_reason": str(item.get("selection_reason", "")),
                    "finalist_status": str(finalist_status),
                    "competition_outcome": str(outcome),
                    "candidate_id": str(evaluated_entry.get("candidate_id", "")),
                    "candidate_inner_fit_score": (
                        float(evaluated_entry.get("inner_fit_score", 0.0)) if entered_finalist else None
                    ),
                    "evidence_term_names": [
                        str(value) for value in tuple(item.get("evidence_term_names", ())) if str(value).strip()
                    ],
                    "branch_threshold": item.get("threshold"),
                    "branch_direction": str(item.get("direction") or ""),
                    "threshold_orthodoxy_score": item.get("threshold_orthodoxy_score"),
                    "threshold_stability_score": item.get("threshold_stability_score"),
                    "threshold_balance_score": item.get("threshold_balance_score"),
                    "threshold_audit": _jsonable(dict(item.get("threshold_audit", {}) or {})),
                    "threshold_selection_lane": str(item.get("threshold_selection_lane") or ""),
                    "winner_candidate_id": str(winner_candidate_summary.get("candidate_id", "")),
                    "winner_realization_signature": str(winner_signature),
                    "winner_inner_fit_score": float(winner_candidate_summary.get("inner_fit_score", 0.0) or 0.0),
                    "winner_basis_term_expressions": [str(value) for value in tuple(winner_basis_term_expressions)],
                }
            )

    for record in tuple(mandatory_records):
        signature = str(record.get("realization_signature") or "").strip()
        if signature not in expected_signatures:
            continue
        parent_key = str(record.get("parent_object_key") or "").strip()
        row_key = (parent_key, signature, str(record.get("object_key") or ""))
        if row_key in seen_audit_keys or (
            not signature.startswith("branch:")
            and any(key[0] == parent_key and key[1] == signature for key in seen_audit_keys)
        ):
            continue
        evaluated_entry = dict(evaluated_by_realization_key.get(row_key, {}))
        entered_finalist = bool(evaluated_entry) and bool(evaluated_entry.get("finalist_entered", False))
        outcome = (
            "selected"
            if entered_finalist
            and best_candidate_summary is not None
            and str(best_candidate_summary.get("candidate_id")) == str(evaluated_entry.get("candidate_id"))
            else ("lost" if entered_finalist else "not_entered")
        )
        audit_rows.append(
            {
                "parent_object_key": str(parent_key),
                "source_information_key": str(record.get("source_information_key") or ""),
                "realization_signature": str(signature),
                "closure_role": str(record.get("closure_role") or ""),
                "realization_label": _signature_label(signature),
                "generation_status": "generated",
                "catalog_selected": None,
                "catalog_selection_reason": "",
                "finalist_status": "entered" if entered_finalist else "not_entered",
                "competition_outcome": str(outcome),
                "candidate_id": str(evaluated_entry.get("candidate_id", "")),
                "candidate_inner_fit_score": (
                    float(evaluated_entry.get("inner_fit_score", 0.0)) if entered_finalist else None
                ),
                "evidence_term_names": [
                    str(value) for value in tuple(record.get("evidence_term_names", ())) if str(value).strip()
                ],
                "branch_threshold": record.get("branch_threshold"),
                "branch_direction": str(record.get("branch_direction") or ""),
                "threshold_orthodoxy_score": record.get("threshold_orthodoxy_score"),
                "threshold_stability_score": record.get("threshold_stability_score"),
                "threshold_balance_score": record.get("threshold_balance_score"),
                "threshold_audit": _jsonable(dict(record.get("threshold_audit", {}) or {})),
                "threshold_selection_lane": str(record.get("threshold_selection_lane") or ""),
                "winner_candidate_id": str(winner_candidate_summary.get("candidate_id", "")),
                "winner_realization_signature": str(winner_signature),
                "winner_inner_fit_score": float(winner_candidate_summary.get("inner_fit_score", 0.0) or 0.0),
                "winner_basis_term_expressions": [str(value) for value in tuple(winner_basis_term_expressions)],
            }
        )

    closure_report = {
        "protocol": str(cfg.mandatory_realization_closure_protocol),
        "mode": str(cfg.mandatory_realization_closure_mode),
        "competition_mode": "inner_finalist_competition_with_evidence_mandatory_heads",
        "status": (
            "no_mandatory_realization_candidates"
            if bool(no_mandatory_candidates)
            else ("selected_explicit_closure" if improved else "kept_search_result")
        ),
        "candidate_count": int(len(tuple(mandatory_records))),
        "evaluated_candidate_count": int(len(evaluated_candidates)),
        "current_inner_fit_score": float(_inner_fit_score(dict(current_final_fit.get("metrics_train", {}) or {}))),
        "selected_candidate": _jsonable(best_candidate_summary),
        "winner_candidate": _jsonable(winner_candidate_summary),
        "winner_basis_term_expressions": [str(value) for value in tuple(winner_basis_term_expressions)],
        "evaluated_candidates": _jsonable(evaluated_candidates),
        "realization_finalist_audit_table": _jsonable(audit_rows),
        "regional_branch_finalist_audit_table": _jsonable(
            [dict(row) for row in tuple(audit_rows) if str(dict(row).get("realization_signature") or "").startswith("branch:")]
        ),
        "narrowness": (
            "This closure stage only operates after the relevant source object has already survived outer basis discovery."
        ),
    }
    if not improved:
        return (
            tuple(dict(term) for term in tuple(current_basis_space_genome)),
            tuple(dict(term) for term in tuple(current_assembled_genome)),
            dict(current_final_fit),
            inner_result,
            closure_report,
        )

    replacement_iterations = tuple(inner_result.iterations) + (
        {
            "iteration": int(len(tuple(inner_result.iterations)) + 1),
            "phase": "mandatory_realization_closure",
            "selected_candidate": _jsonable(best_candidate_summary),
        },
    )
    replacement_inner_result = StructureSearchResult(
        genome=tuple(dict(term) for term in tuple(best_basis_space_genome)),
        base_metrics=dict(inner_result.base_metrics),
        final_metrics=dict(best_final_fit.get("metrics_train", {}) or {}),
        iterations=replacement_iterations,
        weight=np.asarray(best_final_fit.get("weight"), dtype=float),
        bias=np.asarray(best_final_fit.get("bias"), dtype=float),
        score_trace=tuple(float(value) for value in tuple(inner_result.score_trace))
        + (
            float(
                _inner_fit_score(
                    dict(best_final_fit.get("metrics_train", {}) or {})
                )
            ),
        ),
    )
    return (
        tuple(dict(term) for term in tuple(best_basis_space_genome)),
        tuple(dict(term) for term in tuple(best_assembled_genome)),
        dict(best_final_fit),
        replacement_inner_result,
        closure_report,
    )


def _budgeted_assembler_sort_key(
    *,
    inner_result: StructureSearchResult,
    outer_objective: Mapping[str, Any],
    orthogonality_metrics: Mapping[str, Any],
    cfg: OrthogonalBasisSearchConfig,
) -> tuple[Any, ...]:
    rmse_mean = float(dict(inner_result.final_metrics).get("rmse", float("inf")))
    outer_score = float(dict(outer_objective).get("outer_score", 0.0))
    inner_fit_score = float(dict(outer_objective).get("inner_fit_score", 0.0))
    orthogonality_score = float(orthogonality_metrics.get("orthogonality_score", 0.0) or 0.0)
    pair_abs_corr_mean = float(orthogonality_metrics.get("pair_abs_corr_mean", 1.0) or 1.0)
    if str(cfg.selection_mode) == "orthogonal_first":
        return (-orthogonality_score, -outer_score, rmse_mean, -inner_fit_score, pair_abs_corr_mean)
    if str(cfg.selection_mode) == "rmse_first":
        return (rmse_mean, -outer_score, -orthogonality_score, pair_abs_corr_mean)
    return (-outer_score, rmse_mean, -orthogonality_score, pair_abs_corr_mean)


def _regional_correction_feature_name_scope(
    *,
    cfg: OrthogonalBasisSearchConfig,
    selected_rows: Sequence[ScreenedCandidate],
    feature_names: Sequence[str],
    gate_feature_names: Sequence[str],
) -> tuple[str, ...]:
    mode = str(cfg.regional_correction_feature_scope or "gate_only").strip().lower()
    if mode in {"all", "all_features"}:
        return tuple(str(value) for value in tuple(feature_names))
    selected_feature_names: list[str] = []
    for row in tuple(selected_rows):
        for name in _feature_name_tuple(row.features, feature_names=feature_names):
            if name not in selected_feature_names:
                selected_feature_names.append(name)
    gate_names = [str(value) for value in tuple(gate_feature_names) if str(value).strip()]
    if mode in {"selected_features", "selected_only"}:
        return tuple(selected_feature_names)
    if mode in {"gate_or_selected", "selected_plus_gate"}:
        merged = list(gate_names)
        for name in selected_feature_names:
            if name not in merged:
                merged.append(name)
        return tuple(merged)
    return tuple(gate_names)


def _regional_object_key(*, feature_names: Sequence[str]) -> str:
    names = [str(value) for value in tuple(feature_names) if str(value).strip()]
    return f"regional::{'+'.join(sorted(names))}" if names else "regional::anonymous"


def _regional_regime_score(feature_values: np.ndarray, residual: np.ndarray, cut: float) -> float:
    x = np.asarray(feature_values, dtype=float).reshape(-1)
    r = np.asarray(residual, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(r)
    if int(np.sum(mask)) < 8:
        return 0.0
    xv = x[mask]
    rv = r[mask]
    left = rv[xv <= float(cut)]
    right = rv[xv > float(cut)]
    if left.size < 4 or right.size < 4:
        return 0.0
    balance = float(min(left.size, right.size)) / float(max(1, left.size + right.size))
    separation = abs(float(np.mean(left)) - float(np.mean(right)))
    return float(balance * separation)


def _regional_candidate_cuts(
    *,
    feature_values: np.ndarray,
    residual: np.ndarray,
    cfg: OrthogonalBasisSearchConfig,
) -> tuple[float, ...]:
    x = np.asarray(feature_values, dtype=float).reshape(-1)
    mask = np.isfinite(x)
    xv = x[mask]
    if xv.size < 8:
        return tuple()
    quantiles = sorted(
        {
            float(np.clip(value, 0.05, 0.95))
            for value in tuple(cfg.gate_quantiles) + (0.20, 0.35, 0.50, 0.65, 0.80)
            if np.isfinite(float(value))
        }
    )
    scored: list[tuple[float, float]] = []
    seen: set[float] = set()
    for quantile in quantiles:
        cut = float(np.quantile(xv, float(quantile)))
        if not np.isfinite(cut):
            continue
        rounded = round(cut, 10)
        if rounded in seen:
            continue
        seen.add(rounded)
        scored.append((_regional_regime_score(x, residual, cut), float(cut)))
    scored.sort(key=lambda item: (-float(item[0]), float(item[1])))
    top = [float(item[1]) for item in scored[: min(3, len(scored))] if float(item[0]) > 0.0]
    if not top and scored:
        top.append(float(scored[0][1]))
    return tuple(top)


def _regional_candidate_expr_and_values(
    *,
    family: str,
    feature_index: int,
    feature_values: np.ndarray,
    cut: float,
    cfg: OrthogonalBasisSearchConfig,
) -> tuple[dict[str, Any], np.ndarray, float]:
    base_expr = _feature_expr(int(feature_index))
    base_values = np.asarray(feature_values, dtype=float).reshape(-1)
    family_key = str(family).strip().lower()
    if family_key == "piecewise_hinge":
        shifted_expr = _binary_expr("sub", base_expr, _const_expr(float(cut)))
        shifted_values = np.asarray(base_values - float(cut), dtype=float)
        return _relu_expr(shifted_expr), np.asarray(np.maximum(0.0, shifted_values), dtype=float), 3.5
    if family_key in {"gate_step", "gate_soft"}:
        expr = _soft_step_expr(int(feature_index), float(cut), float(max(1.0, cfg.gate_slope)))
        shifted_values = np.asarray(base_values - float(cut), dtype=float)
        values = np.asarray(
            0.5 * (1.0 + np.tanh(float(max(1.0, cfg.gate_slope)) * shifted_values)),
            dtype=float,
        )
        return expr, values, 4.0 if family_key == "gate_step" else 4.2
    shifted_expr = _binary_expr("sub", base_expr, _const_expr(float(cut)))
    shifted_values = np.asarray(base_values - float(cut), dtype=float)
    step_expr = _soft_step_expr(int(feature_index), float(cut), float(max(1.0, cfg.gate_slope)))
    step_values = np.asarray(
        0.5 * (1.0 + np.tanh(float(max(1.0, cfg.gate_slope)) * shifted_values)),
        dtype=float,
    )
    left_expr = _apply_piecewise_mode_expr(str(cfg.piecewise_left_mode), shifted_expr)
    right_expr = _apply_piecewise_mode_expr(str(cfg.piecewise_right_mode), shifted_expr)
    left_values = _apply_piecewise_mode_values(str(cfg.piecewise_left_mode), shifted_values)
    right_values = _apply_piecewise_mode_values(str(cfg.piecewise_right_mode), shifted_values)
    expr = _binary_expr(
        "add",
        _binary_expr("mul", _binary_expr("sub", _const_expr(1.0), step_expr), left_expr),
        _binary_expr("mul", step_expr, right_expr),
    )
    values = np.asarray((1.0 - step_values) * left_values + step_values * right_values, dtype=float)
    return expr, values, 5.0


def _build_reopened_regional_candidates(
    *,
    raw_X: np.ndarray,
    feature_names: Sequence[str],
    allowed_names: Sequence[str],
    residual: np.ndarray,
    cfg: OrthogonalBasisSearchConfig,
) -> tuple[dict[str, Any], ...]:
    x = np.asarray(raw_X, dtype=float)
    if x.ndim != 2 or not allowed_names:
        return tuple()
    name_to_index = {str(name): int(index) for index, name in enumerate(tuple(feature_names))}
    out: list[dict[str, Any]] = []
    seen_expr: set[str] = set()
    for feature_name in tuple(allowed_names):
        feature_index = name_to_index.get(str(feature_name))
        if feature_index is None or int(feature_index) < 0 or int(feature_index) >= x.shape[1]:
            continue
        feature_values = np.asarray(x[:, int(feature_index)], dtype=float).reshape(-1)
        for cut in _regional_candidate_cuts(feature_values=feature_values, residual=residual, cfg=cfg):
            regime_score = _regional_regime_score(feature_values, residual, cut)
            if regime_score <= 0.0:
                continue
            for family in tuple(cfg.gate_families):
                expr, values, complexity = _regional_candidate_expr_and_values(
                    family=str(family),
                    feature_index=int(feature_index),
                    feature_values=feature_values,
                    cut=float(cut),
                    cfg=cfg,
                )
                expr_key = _candidate_expr_key(expr)
                if expr_key in seen_expr:
                    continue
                seen_expr.add(expr_key)
                candidate_name = f"regional_{family}_{feature_name}_{round(float(cut), 6)}"
                out.append(
                    {
                        "candidate_name": str(candidate_name),
                        "object_key": _regional_object_key(feature_names=(str(feature_name),)),
                        "expr": dict(expr),
                        "expression": expression_to_string(expr, precision=8),
                        "semantic_family": "piecewise_gate",
                        "feature_names": [str(feature_name)],
                        "complexity": float(complexity),
                        "uses_piecewise_gate": True,
                        "residual_regime_score": float(regime_score),
                        "candidate_origin": "reopened_local_search",
                        "values": np.asarray(values, dtype=float).reshape(-1),
                    }
                )
    return tuple(out)


def _build_regional_correction_report(
    *,
    selected_rows: Sequence[ScreenedCandidate],
    screened_candidates: Sequence[ScreenedCandidate],
    train_matrix: np.ndarray,
    raw_X: np.ndarray | None,
    target: np.ndarray,
    feature_names: Sequence[str],
    gate_feature_names: Sequence[str],
    cfg: OrthogonalBasisSearchConfig,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    if not _regional_correction_enabled(cfg):
        return tuple(), {
            "protocol": str(cfg.regional_correction_protocol),
            "residual_regime_identification_mode": str(cfg.residual_regime_identification_mode),
            "regional_correction_basis_mode": str(cfg.regional_correction_basis_mode),
            "regional_correction_promotion_mode": str(cfg.regional_correction_promotion_mode),
            "status": "disabled",
        }
    base_matrix = _selected_matrix(train_matrix, selected_rows)
    base_fit = _ridge_projection(
        np.asarray(base_matrix, dtype=float),
        np.asarray(target, dtype=float).reshape(-1),
        l2_value=float(min(tuple(cfg.l2_grid)) if tuple(cfg.l2_grid) else 1e-6),
    )
    residual = np.asarray(base_fit.get("residual", np.asarray(target, dtype=float).reshape(-1)), dtype=float).reshape(-1)
    selected_indices = {int(row.screen_index) for row in tuple(selected_rows)}
    allowed_names = set(
        _regional_correction_feature_name_scope(
            cfg=cfg,
            selected_rows=selected_rows,
            feature_names=feature_names,
            gate_feature_names=gate_feature_names,
        )
    )
    feature_scope_mode = str(cfg.regional_correction_feature_scope or "gate_only").strip().lower()
    if not allowed_names and feature_scope_mode not in {"all", "all_features"}:
        return tuple(), {
            "protocol": str(cfg.regional_correction_protocol),
            "residual_regime_identification_mode": str(cfg.residual_regime_identification_mode),
            "regional_correction_basis_mode": str(cfg.regional_correction_basis_mode),
            "regional_correction_promotion_mode": str(cfg.regional_correction_promotion_mode),
            "regional_correction_feature_scope": str(cfg.regional_correction_feature_scope),
            "status": "skipped",
            "reason": "empty_feature_scope",
        }
    search_mode = str(cfg.regional_correction_search_mode or "reopened_local_object_search").strip().lower()
    candidate_rows: list[dict[str, Any]] = []
    seen_region_expr: set[str] = set()

    def _append_candidate_entry(entry: Mapping[str, Any]) -> None:
        expr = dict(entry.get("expr", {}))
        values = np.asarray(entry.get("values"), dtype=float).reshape(-1)
        if not expr or values.shape[0] != int(base_matrix.shape[0]):
            return
        expr_key = _candidate_expr_key(expr)
        if expr_key in seen_region_expr:
            return
        augmented = _ridge_projection(
            np.asarray(np.concatenate([np.asarray(base_matrix, dtype=float), values.reshape(-1, 1)], axis=1), dtype=float),
            np.asarray(target, dtype=float).reshape(-1),
            l2_value=float(min(tuple(cfg.l2_grid)) if tuple(cfg.l2_grid) else 1e-6),
        )
        marginal_r2_gain = float(augmented.get("r2", 0.0) or 0.0) - float(base_fit.get("r2", 0.0) or 0.0)
        residual_abs_corr = float(abs(_safe_corr(values, residual)))
        if marginal_r2_gain < float(cfg.regional_correction_min_r2_gain):
            return
        regime_score = float(entry.get("residual_regime_score", 0.0) or 0.0)
        score = float(
            np.clip(
                0.60 * max(0.0, marginal_r2_gain)
                + 0.25 * residual_abs_corr
                + 0.15 * regime_score,
                0.0,
                1.0,
            )
        )
        candidate_rows.append(
            {
                "candidate_name": str(entry.get("candidate_name") or entry.get("term_name") or ""),
                "object_key": str(entry.get("object_key") or ""),
                "expr": expr,
                "expression": str(entry.get("expression") or expression_to_string(expr, precision=8)),
                "semantic_family": str(entry.get("semantic_family") or "piecewise_gate"),
                "feature_names": [str(value) for value in tuple(entry.get("feature_names", ()) or ())],
                "complexity": float(entry.get("complexity", 0.0) or 0.0),
                "uses_piecewise_gate": bool(entry.get("uses_piecewise_gate", True)),
                "marginal_r2_gain": float(marginal_r2_gain),
                "residual_abs_corr": float(residual_abs_corr),
                "residual_regime_score": float(regime_score),
                "candidate_origin": str(entry.get("candidate_origin") or "screened_pool"),
                "promotion_score": float(score),
                "values": values,
            }
        )
        seen_region_expr.add(expr_key)

    for candidate in tuple(screened_candidates):
        if int(candidate.screen_index) in selected_indices:
            continue
        if not _candidate_is_gate_family(
            semantic_family=str(candidate.semantic_family),
            uses_piecewise_gate=bool(candidate.uses_piecewise_gate),
        ):
            continue
        candidate_feature_names = _feature_name_tuple(candidate.features, feature_names=feature_names)
        if allowed_names and not (set(candidate_feature_names) & allowed_names):
            continue
        _append_candidate_entry(
            {
                "candidate_name": str(candidate.name),
                "object_key": _regional_object_key(feature_names=candidate_feature_names),
                "expr": dict(candidate.expr),
                "expression": str(candidate.expression),
                "semantic_family": str(candidate.semantic_family),
                "feature_names": [str(value) for value in candidate_feature_names],
                "complexity": float(candidate.complexity),
                "uses_piecewise_gate": bool(candidate.uses_piecewise_gate),
                "candidate_origin": "screened_pool",
                "values": np.asarray(train_matrix[:, int(candidate.screen_index)], dtype=float).reshape(-1),
            }
        )
    if search_mode in {"reopened_local_object_search", "residual_object_beam_search"} and raw_X is not None:
        for entry in _build_reopened_regional_candidates(
            raw_X=np.asarray(raw_X, dtype=float),
            feature_names=feature_names,
            allowed_names=tuple(sorted(allowed_names)),
            residual=residual,
            cfg=cfg,
        ):
            _append_candidate_entry(entry)
    candidate_rows.sort(
        key=lambda item: (
            -float(item.get("promotion_score", 0.0)),
            -float(item.get("marginal_r2_gain", 0.0)),
            -float(item.get("residual_regime_score", 0.0)),
            -float(item.get("residual_abs_corr", 0.0)),
            str(item.get("candidate_name", "")),
        )
    )
    object_members: dict[str, list[dict[str, Any]]] = {}
    for item in tuple(candidate_rows):
        object_members.setdefault(str(item.get("object_key") or item.get("candidate_name") or ""), []).append(dict(item))

    local_beam_width = int(max(1, cfg.regional_local_search_beam_width))
    local_branching_factor = int(max(1, cfg.regional_local_search_branching_factor))
    local_max_expansions = int(max(1, cfg.regional_local_search_max_expansions))
    target_topk = int(max(1, cfg.regional_correction_topk))
    best_state = {
        "selected": tuple(),
        "score": 0.0,
        "fit": dict(base_fit),
    }
    search_trace: list[dict[str, Any]] = []

    if search_mode in {"reopened_local_object_search", "residual_object_beam_search"} and object_members:
        frontier: list[dict[str, Any]] = [
            {
                "selected": tuple(),
                "fit": dict(base_fit),
                "score": 0.0,
            }
        ]
        expansions = 0
        while frontier and expansions < local_max_expansions:
            next_frontier: list[dict[str, Any]] = []
            for state in tuple(frontier):
                selected = tuple(dict(item) for item in tuple(state.get("selected", ())))
                current_fit = dict(state.get("fit", base_fit))
                current_r2 = float(current_fit.get("r2", 0.0) or 0.0)
                residual_now = np.asarray(
                    current_fit.get("residual", np.asarray(target, dtype=float).reshape(-1)),
                    dtype=float,
                ).reshape(-1)
                if selected:
                    state_score = float(state.get("score", 0.0) or 0.0)
                    if state_score >= float(best_state.get("score", 0.0)):
                        best_state = {
                            "selected": tuple(selected),
                            "score": float(state_score),
                            "fit": dict(current_fit),
                        }
                if len(selected) >= target_topk:
                    continue
                used_object_keys = {str(item.get("object_key") or "") for item in selected}
                candidate_expansions: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
                for object_key, members in object_members.items():
                    if str(object_key) in used_object_keys:
                        continue
                    best_member: dict[str, Any] | None = None
                    best_member_score: float | None = None
                    best_member_fit: dict[str, Any] | None = None
                    for member in tuple(members):
                        values = np.asarray(member.get("values"), dtype=float).reshape(-1)
                        current_columns = [np.asarray(item.get("values"), dtype=float).reshape(-1, 1) for item in selected]
                        augmented_matrix = np.asarray(
                            np.concatenate(
                                [
                                    np.asarray(base_matrix, dtype=float),
                                    *current_columns,
                                    values.reshape(-1, 1),
                                ],
                                axis=1,
                            ),
                            dtype=float,
                        )
                        augmented_fit = _ridge_projection(
                            augmented_matrix,
                            np.asarray(target, dtype=float).reshape(-1),
                            l2_value=float(min(tuple(cfg.l2_grid)) if tuple(cfg.l2_grid) else 1e-6),
                        )
                        marginal_gain_now = float(augmented_fit.get("r2", 0.0) or 0.0) - float(current_r2)
                        residual_abs_corr_now = float(abs(_safe_corr(values, residual_now)))
                        if marginal_gain_now < float(cfg.regional_correction_min_r2_gain):
                            continue
                        pair_penalty = 0.0
                        if selected:
                            pair_penalty = float(
                                np.mean(
                                    [
                                        abs(
                                            _safe_corr(
                                                values,
                                                np.asarray(item.get("values"), dtype=float).reshape(-1),
                                            )
                                        )
                                        for item in selected
                                    ]
                                )
                            )
                        member_score = float(
                            np.clip(
                                0.70 * max(0.0, marginal_gain_now)
                                + 0.30 * residual_abs_corr_now
                                - 0.15 * pair_penalty,
                                0.0,
                                1.0,
                            )
                        )
                        if best_member is None or best_member_score is None or member_score > best_member_score:
                            best_member = {
                                **dict(member),
                                "marginal_r2_gain": float(marginal_gain_now),
                                "residual_abs_corr": float(residual_abs_corr_now),
                                "promotion_score": float(member_score),
                            }
                            best_member_score = float(member_score)
                            best_member_fit = dict(augmented_fit)
                    if best_member is None or best_member_fit is None or best_member_score is None:
                        continue
                    candidate_expansions.append((float(best_member_score), best_member, best_member_fit))
                candidate_expansions.sort(
                    key=lambda item: (
                        -float(item[0]),
                        -float(dict(item[1]).get("marginal_r2_gain", 0.0)),
                        str(dict(item[1]).get("candidate_name", "")),
                    )
                )
                for item_score, member, member_fit in tuple(candidate_expansions[:local_branching_factor]):
                    expansions += 1
                    new_selected = tuple([*selected, dict(member)])
                    correction_scores = [float(dict(item).get("promotion_score", 0.0) or 0.0) for item in new_selected]
                    overall_score = float(
                        0.75 * (float(member_fit.get("r2", 0.0) or 0.0) - float(base_fit.get("r2", 0.0) or 0.0))
                        + 0.25 * (float(np.mean(correction_scores)) if correction_scores else 0.0)
                    )
                    next_frontier.append(
                        {
                            "selected": new_selected,
                            "fit": dict(member_fit),
                            "score": float(overall_score),
                        }
                    )
                    search_trace.append(
                        {
                            "selected_object_count": int(len(new_selected)),
                            "selected_object_keys": [str(dict(item).get("object_key", "")) for item in new_selected],
                            "last_candidate_name": str(member.get("candidate_name", "")),
                            "score": float(overall_score),
                        }
                    )
                    if expansions >= local_max_expansions:
                        break
                if expansions >= local_max_expansions:
                    break
            next_frontier.sort(key=lambda item: -float(item.get("score", 0.0) or 0.0))
            frontier = next_frontier[:local_beam_width]
    else:
        best_state = {
            "selected": tuple(dict(item) for item in tuple(candidate_rows[:target_topk])),
            "score": float(np.mean([float(item.get("promotion_score", 0.0) or 0.0) for item in tuple(candidate_rows[:target_topk])]))
            if candidate_rows[:target_topk]
            else 0.0,
            "fit": dict(base_fit),
        }

    promoted = tuple(dict(item) for item in tuple(best_state.get("selected", ())))
    promoted_scores = [float(item.get("promotion_score", 0.0) or 0.0) for item in promoted]
    promoted_gains = [float(item.get("marginal_r2_gain", 0.0) or 0.0) for item in promoted]
    origin_counts = Counter(str(item.get("candidate_origin") or "unknown") for item in tuple(candidate_rows))
    report = {
        "protocol": str(cfg.regional_correction_protocol),
        "residual_regime_identification_mode": str(cfg.residual_regime_identification_mode),
        "regional_correction_basis_mode": str(cfg.regional_correction_basis_mode),
        "regional_correction_promotion_mode": str(cfg.regional_correction_promotion_mode),
        "regional_correction_search_mode": str(cfg.regional_correction_search_mode),
        "regional_correction_feature_scope": str(cfg.regional_correction_feature_scope),
        "status": "reported",
        "candidate_pool_count": int(len(candidate_rows)),
        "regional_object_count": int(len(object_members)),
        "candidate_origin_counts": {str(key): int(value) for key, value in origin_counts.items()},
        "promoted_count": int(len(promoted)),
        "regional_correction_score": float(np.mean(promoted_scores)) if promoted_scores else 0.0,
        "promoted_mean_r2_gain": float(np.mean(promoted_gains)) if promoted_gains else 0.0,
        "reopened_local_search_score": float(best_state.get("score", 0.0) or 0.0),
        "search_trace": _jsonable(search_trace),
        "promoted_candidates": [
            {
                key: _jsonable(value)
                for key, value in dict(item).items()
                if key != "values"
            }
            for item in promoted
        ],
    }
    if not promoted:
        report["status"] = "skipped"
        report["reason"] = "no_regional_candidates_above_gain_threshold"
    return promoted, report


def _run_budgeted_symbolic_assembler(
    *,
    outer_basis_genome: Sequence[Mapping[str, Any]],
    basis_rows: Sequence[Mapping[str, Any]],
    pool_indices: Sequence[int],
    selected_rows: Sequence[ScreenedCandidate],
    train_matrix: np.ndarray,
    target: np.ndarray,
    raw_X: np.ndarray,
    raw_feature_names: Sequence[str],
    cfg: OrthogonalBasisSearchConfig,
    graph_cache: ExpressionGraphCache | None,
    orthogonality_metrics: Mapping[str, Any],
    residual_report: Mapping[str, Any],
    semantic_report: Mapping[str, Any],
    interference_report: Mapping[str, Any],
    screened_candidates: Sequence[ScreenedCandidate],
    interference_context: Mapping[str, Any],
    periodic_context: Mapping[str, Any],
    realization_evidence_registry: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    gate_feature_names: Sequence[str] | None = None,
    data_metadata: Mapping[str, Any] | None = None,
) -> BudgetedOrthogonalAssemblerResult:
    regional_correction_objects, regional_correction_report = _build_regional_correction_report(
        selected_rows=selected_rows,
        screened_candidates=screened_candidates,
        train_matrix=train_matrix,
        raw_X=np.asarray(raw_X, dtype=float),
        target=np.asarray(target, dtype=float),
        feature_names=tuple(str(value) for value in tuple(raw_feature_names)),
        gate_feature_names=tuple(str(value) for value in tuple(gate_feature_names or ())),
        cfg=cfg,
    )
    (
        basis_matrix,
        basis_feature_names,
        assembler_basis_genome,
        basis_object_records,
        chart_objects,
        realization_objects,
        regional_correction_objects,
        escape_objects,
    ) = _build_assembler_object_space(
        outer_basis_genome=outer_basis_genome,
        selected_rows=selected_rows,
        basis_rows=basis_rows,
        train_matrix=train_matrix,
        raw_X=raw_X,
        target=target,
        raw_feature_names=raw_feature_names,
        regional_correction_candidates=regional_correction_objects,
        screened_candidates=screened_candidates,
        interference_context=interference_context,
        periodic_context=periodic_context,
        realization_evidence_registry=realization_evidence_registry,
        cfg=cfg,
        gate_feature_names=tuple(str(value) for value in tuple(gate_feature_names or ())),
        data_metadata=data_metadata,
    )
    stage_head_protocols, assembler_stage_spec, basis_context = _build_stage_head_protocol_payload(
        basis_object_records=basis_object_records,
        basis_feature_names=basis_feature_names,
        gate_feature_names=tuple(str(value) for value in tuple(gate_feature_names or ())),
        cfg=cfg,
    )
    search_cfg = _build_budgeted_symbolic_search_config(cfg=cfg, basis_feature_count=int(basis_matrix.shape[1]))
    inner_result = residual_guided_structure_search(
        basis_matrix,
        np.asarray(target, dtype=float),
        feature_names=tuple(basis_feature_names),
        seed_genome=None,
        config=search_cfg,
        inner_runtime_context={
            "search_driver": "orthogonal_budgeted_symbolic_assembler",
            "runtime_key": "orthogonal_budgeted_symbolic_assembler",
            "trainer_name": "symbolic_orthogonal",
            "training_mode": "assembler_inner_search",
            "structure_head": assembler_stage_spec.get("structure_head"),
            "search_input_space": assembler_stage_spec.get("search_input_space"),
            "pool_expansion_unit": assembler_stage_spec.get("pool_expansion_unit"),
            "gradient_guidance_mode": assembler_stage_spec.get("gradient_guidance_mode"),
        },
    )
    object_gradient_pool = _build_basis_object_gradient_pool_report(
        inner_result=inner_result,
        stage_head_spec=assembler_stage_spec,
        basis_context=basis_context,
    )
    basis_space_genome = tuple(dict(term) for term in tuple(inner_result.genome))
    assembled_genome = _substitute_basis_genome(
        basis_space_genome,
        basis_genome=tuple(dict(term) for term in tuple(assembler_basis_genome)),
    )
    final_fit = evaluate_genome_with_ridge(
        assembled_genome,
        X_train=np.asarray(raw_X, dtype=float),
        y_train=np.asarray(target, dtype=float),
        l2=float(search_cfg.ridge_l2),
        graph_cache=graph_cache,
        train_batch_key="orthogonal_budgeted_symbolic_assembler::train",
    )
    (
        basis_space_genome,
        assembled_genome,
        final_fit,
        inner_result,
        mandatory_realization_closure_report,
    ) = _run_mandatory_realization_closure(
        current_basis_space_genome=basis_space_genome,
        current_assembled_genome=assembled_genome,
        current_final_fit=final_fit,
        inner_result=inner_result,
        assembler_basis_genome=tuple(dict(term) for term in tuple(assembler_basis_genome)),
        basis_feature_names=basis_feature_names,
        basis_object_records=basis_object_records,
        raw_X=np.asarray(raw_X, dtype=float),
        target=np.asarray(target, dtype=float),
        search_cfg=search_cfg,
        graph_cache=graph_cache,
        cfg=cfg,
    )
    final_expression_payload = _build_expression_payload(
        genome=assembled_genome,
        feature_names=tuple(str(value) for value in tuple(raw_feature_names)),
        weight=np.asarray(final_fit.get("weight"), dtype=float),
        bias=np.asarray(final_fit.get("bias"), dtype=float),
    )
    environment_audit = _build_environment_invariance_audit(
        selected_rows=selected_rows,
        train_matrix=train_matrix,
        target=np.asarray(target, dtype=float),
        raw_X=np.asarray(raw_X, dtype=float),
        feature_names=tuple(str(value) for value in tuple(raw_feature_names)),
        gate_feature_names=tuple(str(value) for value in tuple(gate_feature_names or ())),
        cfg=cfg,
    )
    periodic_report = _build_periodic_equivalence_report(
        selected_rows=selected_rows,
        train_matrix=train_matrix,
        target=np.asarray(target, dtype=float),
        feature_names=tuple(str(value) for value in tuple(raw_feature_names)),
        periodic_context=periodic_context,
        cfg=cfg,
    )
    same_source_realization_report = _same_source_over_realization_report(
        basis_space_genome=basis_space_genome,
        basis_feature_names=basis_feature_names,
        basis_object_records=basis_object_records,
        cfg=cfg,
    )
    outer_objective = _outer_objective_payload(
        inner_metrics=dict(final_fit.get("metrics_train", {})),
        orthogonality_metrics=orthogonality_metrics,
        residual_report=residual_report,
        semantic_report=semantic_report,
        interference_report=interference_report,
        periodic_report=periodic_report,
        regional_correction_report=regional_correction_report,
        same_source_realization_report=same_source_realization_report,
        environment_audit=environment_audit,
    )
    fold_report = _build_orthogonal_fold_report(
        pool_indices=pool_indices,
        basis_rows=basis_rows,
        final_metrics=dict(final_fit.get("metrics_train", {})),
        search_cfg=search_cfg,
        inner_result=inner_result,
        basis_feature_names=basis_feature_names,
    )
    return BudgetedOrthogonalAssemblerResult(
        basis_feature_names=basis_feature_names,
        basis_space_genome=basis_space_genome,
        assembled_genome=assembled_genome,
        inner_result=inner_result,
        final_fit=final_fit,
        final_expression_payload=final_expression_payload,
        fold_report=fold_report,
        outer_objective=outer_objective,
        search_config={
            "max_added_terms": int(search_cfg.max_added_terms),
            "topk_features": int(search_cfg.topk_features),
            "max_pair_terms": int(search_cfg.max_pair_terms),
            "max_candidates_per_iter": int(search_cfg.max_candidates_per_iter),
            "candidate_keep_top": int(search_cfg.candidate_keep_top),
            "max_expr_depth": int(search_cfg.max_expr_depth),
            "ridge_l2": float(search_cfg.ridge_l2),
            "include_hinge": bool(search_cfg.include_hinge),
            "hinge_quantiles": [float(value) for value in tuple(search_cfg.hinge_quantiles)],
            "path_memory_enabled": bool(search_cfg.path_memory_enabled),
            "graph_cache_enabled": bool(search_cfg.graph_cache_enabled),
            "binding_mode": str(assembler_stage_spec.get("basis_binding_mode")),
            "escape_policy": str(assembler_stage_spec.get("escape_policy")),
            "escape_feature_names": [
                str(item.get("raw_feature_name") or "")
                for item in tuple(escape_objects)
                if str(item.get("raw_feature_name") or "").strip()
            ],
        },
        stage_head_protocols=stage_head_protocols,
        basis_context=basis_context,
        object_gradient_pool=object_gradient_pool,
        environment_invariance_audit=environment_audit,
        periodic_equivalence_report=periodic_report,
        regional_correction_report=regional_correction_report,
        mandatory_realization_closure_report=mandatory_realization_closure_report,
        same_source_over_realization_report=same_source_realization_report,
    )


def _ridge_projection(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    l2_value: float,
) -> dict[str, Any]:
    y = np.asarray(target, dtype=float).reshape(-1)
    x = np.asarray(matrix, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.size == 0:
        x = np.zeros((y.shape[0], 0), dtype=float)
    design = np.concatenate([x, np.ones((y.shape[0], 1), dtype=float)], axis=1)
    gram = np.asarray(design.T @ design, dtype=float)
    reg = np.eye(gram.shape[0], dtype=float)
    reg[-1, -1] = 0.0
    rhs = np.asarray(design.T @ y, dtype=float)
    try:
        coef = np.linalg.solve(gram + float(l2_value) * reg, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(gram + float(l2_value) * reg, rhs, rcond=None)[0]
    pred = np.asarray(design @ coef, dtype=float).reshape(-1)
    residual = np.asarray(y - pred, dtype=float).reshape(-1)
    centered = y - float(np.mean(y))
    ss_tot = float(np.dot(centered, centered))
    ss_res = float(np.dot(residual, residual))
    return {
        "prediction": pred,
        "residual": residual,
        "r2": 0.0 if ss_tot <= 1e-12 else float(1.0 - ss_res / (ss_tot + 1e-12)),
        "residual_norm": float(np.linalg.norm(residual)),
        "weight": np.asarray(coef[:-1], dtype=float).reshape(-1),
        "bias": float(coef[-1]),
    }


def _residual_complementarity_steps(
    *,
    selected_rows: Sequence[ScreenedCandidate],
    train_matrix: np.ndarray,
    target: np.ndarray,
    l2_value: float,
) -> list[dict[str, Any]]:
    y = np.asarray(target, dtype=float).reshape(-1)
    baseline = _ridge_projection(np.zeros((y.shape[0], 0), dtype=float), y, l2_value=float(l2_value))
    steps: list[dict[str, Any]] = []
    for index, row in enumerate(tuple(selected_rows)):
        before_rows = tuple(selected_rows[:index])
        after_rows = tuple(selected_rows[: index + 1])
        before_matrix = _selected_matrix(train_matrix, before_rows)
        after_matrix = _selected_matrix(train_matrix, after_rows)
        before_fit = baseline if not before_rows else _ridge_projection(before_matrix, y, l2_value=float(l2_value))
        after_fit = _ridge_projection(after_matrix, y, l2_value=float(l2_value))
        candidate_values = np.asarray(train_matrix[:, int(row.screen_index)], dtype=float).reshape(-1)
        residual_before = np.asarray(before_fit["residual"], dtype=float).reshape(-1)
        target_norm = float(np.linalg.norm(y - float(np.mean(y)))) + 1e-12
        residual_after = np.asarray(after_fit["residual"], dtype=float).reshape(-1)
        steps.append(
            {
                "term_name": str(row.name),
                "semantic_family": str(row.semantic_family),
                "marginal_target_abs_corr": float(abs(_safe_corr(candidate_values, y))),
                "marginal_residual_abs_corr": float(abs(_safe_corr(candidate_values, residual_before))),
                "marginal_r2_gain": float(after_fit["r2"] - before_fit["r2"]),
                "residual_norm_before": float(before_fit["residual_norm"]),
                "residual_norm_after": float(after_fit["residual_norm"]),
                "residual_ratio_after": float(np.linalg.norm(residual_after) / target_norm),
            }
        )
    return steps


def _semantic_repeat_limit(candidate: ScreenedCandidate, cfg: OrthogonalBasisSearchConfig) -> int:
    if bool(candidate.uses_piecewise_gate):
        return int(cfg.max_piecewise_semantic_repeats)
    return int(cfg.max_semantic_repeats)


def _group_summary_payload(
    *,
    selected_rows: Sequence[ScreenedCandidate],
    threshold: float,
    train_matrix: np.ndarray,
    target: np.ndarray,
    raw_X: np.ndarray,
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
    periodic_context: Mapping[str, Any],
    cfg: OrthogonalBasisSearchConfig,
    fallback_mode: str | None = None,
) -> dict[str, Any]:
    screen_positions = tuple(sorted(int(row.screen_index) for row in selected_rows))
    train_values = np.asarray(train_matrix[:, screen_positions], dtype=float)
    basis_rows = _selected_basis_rows(selected_rows)
    object_keys = [
        _candidate_object_key(
            candidate=row,
            feature_names=feature_names,
            interference_context=interference_context,
            periodic_context=periodic_context,
            outer_search_unit=str(cfg.outer_search_unit),
        )
        for row in tuple(selected_rows)
    ]
    object_kinds = [
        _candidate_object_kind(
            candidate=row,
            feature_names=feature_names,
            periodic_context=periodic_context,
        )
        for row in tuple(selected_rows)
    ]
    selection_channels = [str(row.selection_channel or "challenger") for row in tuple(selected_rows)]
    orthogonality = _orthogonality_metrics(selected_rows=selected_rows, train_values=train_values)
    residual_report = build_residual_complementarity_report(
        _residual_complementarity_steps(
            selected_rows=selected_rows,
            train_matrix=train_matrix,
            target=target,
            l2_value=float(min(tuple(cfg.l2_grid)) if tuple(cfg.l2_grid) else 1e-6),
        ),
        source="orthogonal_basis_discovery",
        extra={"selection_threshold": float(threshold)},
    )
    semantic_report = build_semantic_dedup_report(
        basis_rows,
        source="orthogonal_basis_discovery",
        extra={"selection_threshold": float(threshold)},
    )
    interference_report = _build_interference_feature_report(
        selected_rows=selected_rows,
        train_matrix=train_matrix,
        feature_names=feature_names,
        interference_context=interference_context,
        cfg=cfg,
    )
    periodic_report = _build_periodic_equivalence_report(
        selected_rows=selected_rows,
        train_matrix=train_matrix,
        target=target,
        feature_names=feature_names,
        periodic_context=periodic_context,
        cfg=cfg,
    )
    orthogonality["semantic_unique_ratio"] = float(semantic_report.get("semantic_unique_ratio", 0.0))
    orthogonality["piecewise_gate_term_count"] = int(semantic_report.get("piecewise_gate_term_count", 0))
    orthogonality["residual_gain_mean"] = float(residual_report.get("mean_marginal_r2_gain", 0.0))
    orthogonality["residual_gain_min"] = float(residual_report.get("min_marginal_r2_gain", 0.0))
    gate_term_count = _gate_term_count(selected_rows)
    native_trunk_term_count = _native_trunk_term_count(selected_rows)
    periodic_term_count = _periodic_term_count(selected_rows)
    mechanistic_term_count = int(sum(1 for row in tuple(selected_rows) if float(row.mechanistic_prior) > 0.0))
    group_score = float(
        np.mean([float(row.screen_score) for row in selected_rows])
        + 0.45 * float(orthogonality["orthogonality_score"])
        + float(cfg.residual_gain_weight) * float(residual_report.get("mean_marginal_r2_gain", 0.0))
        + 0.10 * float(semantic_report.get("semantic_unique_ratio", 0.0))
        + 0.25 * float(periodic_report.get("overall_periodic_disambiguation_score", 0.0) or 0.0)
        - 0.20 * float(orthogonality["pair_abs_corr_mean"])
        - 0.05 * float(orthogonality["feature_overlap_mean"])
        - 0.35 * float(interference_report.get("trivial_nonlinearity_penalty_mean", 0.0) or 0.0)
        - 0.30 * float(periodic_report.get("local_equivalence_penalty_mean", 0.0) or 0.0)
    )
    payload = {
        "threshold": float(threshold),
        "screen_positions": [int(value) for value in screen_positions],
        "pool_indices": [int(row.pool_index) for row in tuple(selected_rows)],
        "rows": [row for row in tuple(selected_rows)],
        "object_summary": {
            "outer_search_unit": str(cfg.outer_search_unit),
            "representative_selection_rule": str(cfg.representative_selection_rule),
            "object_keys": [str(value) for value in object_keys],
            "object_kinds": [str(value) for value in object_kinds],
            "selection_channels": [str(value) for value in selection_channels],
            "unique_object_count": int(len(set(object_keys))),
        },
        "orthogonality_metrics": dict(orthogonality),
        "group_score": float(group_score),
        "screen_summary": {
            "mean_screen_score": float(np.mean([float(row.screen_score) for row in selected_rows])) if selected_rows else 0.0,
            "mean_target_corr": float(np.mean([float(row.target_corr) for row in selected_rows])) if selected_rows else 0.0,
            "mean_residual_gain": float(np.mean([float(row.residual_gain) for row in selected_rows])) if selected_rows else 0.0,
            "mean_semantic_novelty": float(np.mean([float(row.semantic_novelty) for row in selected_rows])) if selected_rows else 0.0,
            "mean_consensus_prior": float(np.mean([float(row.consensus_prior) for row in selected_rows])) if selected_rows else 0.0,
            "max_consensus_prior": float(np.max([float(row.consensus_prior) for row in selected_rows])) if selected_rows else 0.0,
            "mean_mechanistic_prior": float(np.mean([float(row.mechanistic_prior) for row in selected_rows])) if selected_rows else 0.0,
            "max_mechanistic_prior": float(np.max([float(row.mechanistic_prior) for row in selected_rows])) if selected_rows else 0.0,
            "mean_regime_penetration_score": float(
                np.mean([float(row.regime_penetration_score) for row in selected_rows])
            )
            if selected_rows
            else 0.0,
            "min_regime_penetration_score": float(
                np.min([float(row.regime_penetration_score) for row in selected_rows])
            )
            if selected_rows
            else 0.0,
        },
        "mechanism_summary": {
            "required_gate_basis_terms": int(_required_gate_basis_terms(cfg)),
            "required_native_trunk_basis_terms": int(_required_native_trunk_basis_terms(cfg)),
            "required_periodic_basis_terms": int(
                _required_periodic_basis_terms(cfg=cfg, periodic_context=periodic_context)
            ),
            "native_trunk_term_count": int(native_trunk_term_count),
            "gate_term_count": int(gate_term_count),
            "periodic_term_count": int(periodic_term_count),
            "mechanistic_term_count": int(mechanistic_term_count),
            "heterogeneous_exposure_term_count": int(
                sum(1 for row in tuple(selected_rows) if bool(row.heterogeneous_exposure_eligible))
            ),
            "native_trunk_requirement_satisfied": bool(
                native_trunk_term_count >= int(_required_native_trunk_basis_terms(cfg))
                or int(_required_native_trunk_basis_terms(cfg)) <= 0
            ),
            "gate_requirement_satisfied": bool(
                gate_term_count >= int(_required_gate_basis_terms(cfg))
                or int(_required_gate_basis_terms(cfg)) <= 0
            ),
            "periodic_requirement_satisfied": bool(
                periodic_term_count
                >= int(_required_periodic_basis_terms(cfg=cfg, periodic_context=periodic_context))
                or int(_required_periodic_basis_terms(cfg=cfg, periodic_context=periodic_context)) <= 0
            ),
            "cross_explanatory_suspicious_pairs": int(
                interference_report.get("suspicious_pair_count", 0) or 0
            ),
            "trivial_nonlinearity_penalty_mean": float(
                interference_report.get("trivial_nonlinearity_penalty_mean", 0.0) or 0.0
            ),
            "periodic_disambiguation_score": float(
                periodic_report.get("overall_periodic_disambiguation_score", 0.0) or 0.0
            ),
            "periodic_local_equivalence_penalty_mean": float(
                periodic_report.get("local_equivalence_penalty_mean", 0.0) or 0.0
            ),
        },
        "residual_complementarity_report": _jsonable(residual_report),
        "semantic_dedup_report": _jsonable(semantic_report),
        "interference_feature_report": _jsonable(interference_report),
        "periodic_equivalence_report": _jsonable(periodic_report),
    }
    if fallback_mode is not None:
        payload["fallback_mode"] = str(fallback_mode)
    return payload


def _group_build_score(
    *,
    candidate: ScreenedCandidate,
    selected_rows: Sequence[ScreenedCandidate],
    corr_matrix: np.ndarray,
    used_feature_counts: Counter[int],
    signature_counts: Counter[str],
    current_fit: Mapping[str, Any],
    train_matrix: np.ndarray,
    target: np.ndarray,
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
    cfg: OrthogonalBasisSearchConfig,
) -> float:
    if not selected_rows:
        return float(candidate.screen_score)
    selected_idx = [int(row.screen_index) for row in selected_rows]
    pair_corrs = [float(corr_matrix[int(candidate.screen_index), idx]) for idx in selected_idx]
    mean_pair = float(np.mean(pair_corrs)) if pair_corrs else 0.0
    overlap_count = float(sum(float(used_feature_counts.get(int(value), 0.0)) for value in candidate.features))
    new_feature_count = float(
        sum(1 for value in candidate.features if float(used_feature_counts.get(int(value), 0.0)) <= 0.0)
    )
    family_seen = {str(row.family) for row in selected_rows}
    family_bonus = float(cfg.family_diversity_bonus) if str(candidate.family) not in family_seen else 0.0
    semantic_family_seen = {str(row.semantic_family) for row in selected_rows}
    semantic_family_bonus = (
        float(cfg.semantic_family_bonus)
        if str(candidate.semantic_family) and str(candidate.semantic_family) not in semantic_family_seen
        else 0.0
    )
    candidate_values = np.asarray(train_matrix[:, int(candidate.screen_index)], dtype=float).reshape(-1)
    residual = np.asarray(current_fit.get("residual", np.asarray(target, dtype=float).reshape(-1)), dtype=float).reshape(-1)
    residual_abs_corr = float(abs(_safe_corr(candidate_values, residual)))
    augmented_matrix = np.asarray(train_matrix[:, [*selected_idx, int(candidate.screen_index)]], dtype=float)
    augmented_fit = _ridge_projection(
        augmented_matrix,
        np.asarray(target, dtype=float).reshape(-1),
        l2_value=float(min(tuple(cfg.l2_grid)) if tuple(cfg.l2_grid) else 1e-6),
    )
    marginal_r2_gain = float(augmented_fit["r2"] - float(current_fit.get("r2", 0.0)))
    semantic_repeat_penalty = float(cfg.semantic_dup_penalty) * float(signature_counts.get(str(candidate.semantic_signature), 0))
    piecewise_requested = bool(cfg.enable_piecewise_basis and tuple(cfg.gate_feature_names))
    selected_piecewise = any(bool(row.uses_piecewise_gate) for row in tuple(selected_rows))
    piecewise_bonus = 0.0
    if bool(candidate.uses_piecewise_gate):
        piecewise_bonus = float(cfg.piecewise_gate_bonus)
        if piecewise_requested and not selected_piecewise:
            piecewise_bonus *= 2.5
    mechanistic_bonus = float(cfg.mechanistic_group_bonus) * float(candidate.mechanistic_prior)
    causal_hierarchy_bonus = _causal_hierarchy_parent_bonus(
        candidate=candidate,
        selected_rows=selected_rows,
        cfg=cfg,
    )
    support_expansion_bonus = 0.18 if bool(candidate.support_expansion_candidate) else 0.0
    global_uniform_bonus = 0.10 if bool(candidate.global_uniform_candidate) else 0.0
    modulated_branch_penalty = 0.08 if bool(candidate.modulated_branch_candidate) and _global_first_preemption_enabled(cfg) else 0.0
    cross_summary = _cross_explanatory_summary(
        candidate=candidate,
        selected_rows=selected_rows,
        train_matrix=train_matrix,
        feature_names=feature_names,
        interference_context=interference_context,
    )
    trivial_penalty = _trivial_nonlinearity_penalty_value(
        candidate=candidate,
        summary=cross_summary,
        cfg=cfg,
    )
    return float(
        cfg.target_score_weight * float(candidate.screen_score)
        - cfg.diversity_corr_weight * mean_pair
        - cfg.feature_overlap_penalty * overlap_count
        - cfg.complexity_penalty * float(candidate.complexity)
        - semantic_repeat_penalty
        - float(trivial_penalty)
        - float(modulated_branch_penalty)
        + cfg.new_feature_bonus * new_feature_count
        + family_bonus
        + semantic_family_bonus
        + float(cfg.residual_corr_weight) * residual_abs_corr
        + float(cfg.residual_gain_weight) * max(0.0, marginal_r2_gain)
        + piecewise_bonus
        + mechanistic_bonus
        + float(causal_hierarchy_bonus)
        + float(support_expansion_bonus)
        + float(global_uniform_bonus)
    )


def _row_matches_parent_trunk_source(
    row: ScreenedCandidate,
    *,
    parent_source_key: str,
) -> bool:
    if not str(parent_source_key).strip():
        return False
    if _candidate_is_structural_gate(row):
        return False
    source_key = str(row.information_source_key or _candidate_information_source_key(row))
    return bool(source_key == str(parent_source_key))


def _row_support_subset_of_candidate(
    row: ScreenedCandidate,
    *,
    candidate: ScreenedCandidate,
) -> bool:
    row_support = set(int(index) for index in tuple(row.features))
    candidate_support = set(int(index) for index in tuple(candidate.features))
    return bool(row_support and candidate_support and row_support.issubset(candidate_support))


def _selected_has_global_uniform_parent(
    *,
    selected_rows: Sequence[ScreenedCandidate],
    candidate: ScreenedCandidate,
) -> bool:
    return any(
        bool(row.global_uniform_candidate) and _row_support_subset_of_candidate(row, candidate=candidate)
        for row in tuple(selected_rows)
    )


def _candidate_pool_has_global_uniform_parent(
    *,
    candidate_pool: Sequence[ScreenedCandidate],
    candidate: ScreenedCandidate,
    excluded_screen_index: int,
) -> bool:
    for row in tuple(candidate_pool):
        if int(row.screen_index) == int(excluded_screen_index):
            continue
        if not bool(row.global_uniform_candidate):
            continue
        if not _row_support_subset_of_candidate(row, candidate=candidate):
            continue
        return True
    return False


def _selected_has_canonical_support_parent(
    *,
    selected_rows: Sequence[ScreenedCandidate],
    candidate: ScreenedCandidate,
) -> bool:
    support_key = str(candidate.source_support_key or "").strip()
    if not support_key:
        return False
    return any(
        (bool(row.canonical_trunk_candidate) or bool(row.support_expansion_candidate))
        and str(row.source_support_key or "").strip() == support_key
        for row in tuple(selected_rows)
    )


def _candidate_pool_has_canonical_support_parent(
    *,
    candidate_pool: Sequence[ScreenedCandidate],
    candidate: ScreenedCandidate,
    excluded_screen_index: int,
) -> bool:
    support_key = str(candidate.source_support_key or "").strip()
    if not support_key:
        return False
    for row in tuple(candidate_pool):
        if int(row.screen_index) == int(excluded_screen_index):
            continue
        if not (bool(row.canonical_trunk_candidate) or bool(row.support_expansion_candidate)):
            continue
        if str(row.source_support_key or "").strip() != support_key:
            continue
        return True
    return False


def _selected_has_parent_trunk(
    *,
    selected_rows: Sequence[ScreenedCandidate],
    parent_source_key: str,
) -> bool:
    return any(
        _row_matches_parent_trunk_source(row, parent_source_key=parent_source_key)
        for row in tuple(selected_rows)
    )


def _candidate_pool_has_parent_trunk(
    *,
    candidate_pool: Sequence[ScreenedCandidate],
    parent_source_key: str,
    excluded_screen_index: int,
) -> bool:
    for row in tuple(candidate_pool):
        if int(row.screen_index) == int(excluded_screen_index):
            continue
        if not _row_matches_parent_trunk_source(row, parent_source_key=parent_source_key):
            continue
        if bool(row.native_trunk_floor_passed) or str(row.selection_channel or "") in {"native_trunk", "support_expansion"}:
            return True
    return False


def _accept_candidate(
    *,
    candidate: ScreenedCandidate,
    selected_rows: Sequence[ScreenedCandidate],
    corr_matrix: np.ndarray,
    used_feature_counts: Counter[int],
    signature_counts: Counter[str],
    train_matrix: np.ndarray,
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
    max_pair_abs_corr: float,
    max_feature_reuse: int,
    cfg: OrthogonalBasisSearchConfig,
    candidate_pool: Sequence[ScreenedCandidate] | None = None,
) -> bool:
    if any(int(candidate.screen_index) == int(row.screen_index) for row in selected_rows):
        return False
    gate_parent_source_key = ""
    if _candidate_is_structural_gate(candidate):
        gate_parent_source_key = str(_candidate_gate_parent_source_key(candidate) or "")
    if _parasitic_rejection_enabled(cfg) and gate_parent_source_key:
        if not _selected_has_parent_trunk(
            selected_rows=selected_rows,
            parent_source_key=gate_parent_source_key,
        ):
            parent_pool = tuple(candidate_pool or ())
            if not parent_pool or _candidate_pool_has_parent_trunk(
                candidate_pool=parent_pool,
                parent_source_key=gate_parent_source_key,
                excluded_screen_index=int(candidate.screen_index),
            ):
                return False
    if _global_first_preemption_enabled(cfg) and bool(candidate.modulated_branch_candidate):
        if not _selected_has_global_uniform_parent(
            selected_rows=selected_rows,
            candidate=candidate,
        ):
            parent_pool = tuple(candidate_pool or ())
            if not parent_pool or _candidate_pool_has_global_uniform_parent(
                candidate_pool=parent_pool,
                candidate=candidate,
                excluded_screen_index=int(candidate.screen_index),
            ):
                return False
    if _canonical_trunk_lane_enabled(cfg) and bool(candidate.same_source_surrogate_candidate):
        if not _selected_has_canonical_support_parent(
            selected_rows=selected_rows,
            candidate=candidate,
        ):
            parent_pool = tuple(candidate_pool or ())
            if not parent_pool or _candidate_pool_has_canonical_support_parent(
                candidate_pool=parent_pool,
                candidate=candidate,
                excluded_screen_index=int(candidate.screen_index),
            ):
                return False
    if selected_rows:
        pair_corrs = [
            float(corr_matrix[int(candidate.screen_index), int(row.screen_index)])
            for row in selected_rows
            if not (
                gate_parent_source_key
                and _row_matches_parent_trunk_source(row, parent_source_key=gate_parent_source_key)
            )
        ]
        if pair_corrs and float(max(pair_corrs)) > float(max_pair_abs_corr):
            return False
    for feature_index in candidate.features:
        used_budget = float(used_feature_counts.get(int(feature_index), 0.0))
        add_budget = float(_candidate_reuse_budget_cost(candidate, cfg))
        if used_budget + add_budget > float(max_feature_reuse) + 1e-12:
            return False
    if int(signature_counts.get(str(candidate.semantic_signature), 0)) >= _semantic_repeat_limit(candidate, cfg):
        return False
    if not _candidate_proxy_assignment_compatible(
        candidate=candidate,
        selected_rows=selected_rows,
        feature_names=feature_names,
        interference_context=interference_context,
    ):
        return False
    if _cross_explanatory_rejection_enabled(cfg) and selected_rows:
        cross_summary = _cross_explanatory_summary(
            candidate=candidate,
            selected_rows=selected_rows,
            train_matrix=train_matrix,
            feature_names=feature_names,
            interference_context=interference_context,
        )
        if bool(cross_summary.get("suspicious_overlap")):
            return False
    return True


def _match_seed_rows(
    *,
    screened: Sequence[ScreenedCandidate],
    seed_genome: Sequence[Mapping[str, Any]] | None,
) -> tuple[ScreenedCandidate, ...]:
    if seed_genome is None:
        return tuple()
    expr_keys = {
        _candidate_expr_key(dict(term.get("expr", {})))
        for term in tuple(seed_genome)
        if isinstance(term, Mapping)
    }
    if not expr_keys:
        return tuple()
    by_key = {_candidate_expr_key(dict(row.expr)): row for row in tuple(screened)}
    matched: list[ScreenedCandidate] = []
    for key in sorted(expr_keys):
        row = by_key.get(key)
        if row is not None:
            matched.append(row)
    return tuple(matched)


def _greedy_complete_group(
    *,
    seed_rows: Sequence[ScreenedCandidate],
    screened: Sequence[ScreenedCandidate],
    train_matrix: np.ndarray,
    y_train: np.ndarray,
    raw_X: np.ndarray,
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
    corr_matrix: np.ndarray,
    cfg: OrthogonalBasisSearchConfig,
    threshold: float,
    rng: np.random.Generator | None = None,
) -> list[ScreenedCandidate]:
    selected = list(seed_rows)
    used_feature_counts: Counter[int] = Counter()
    signature_counts: Counter[str] = Counter()
    for row in selected:
        _increment_feature_reuse_budget(
            used_feature_counts,
            candidate=row,
            cfg=cfg,
        )
        signature_counts[str(row.semantic_signature)] += 1
    while len(selected) < int(cfg.max_basis_count):
        current_fit = _ridge_projection(
            _selected_matrix(train_matrix, selected),
            np.asarray(y_train, dtype=float).reshape(-1),
            l2_value=float(min(tuple(cfg.l2_grid)) if tuple(cfg.l2_grid) else 1e-6),
        )
        scored_candidates: list[tuple[float, float, str, ScreenedCandidate]] = []
        for candidate in screened:
            if not _accept_candidate(
                candidate=candidate,
                selected_rows=selected,
                corr_matrix=corr_matrix,
                used_feature_counts=used_feature_counts,
                signature_counts=signature_counts,
                train_matrix=train_matrix,
                feature_names=feature_names,
                interference_context=interference_context,
                max_pair_abs_corr=float(threshold),
                max_feature_reuse=int(cfg.max_feature_reuse),
                cfg=cfg,
                candidate_pool=screened,
            ):
                continue
            score = _group_build_score(
                candidate=candidate,
                selected_rows=selected,
                corr_matrix=corr_matrix,
                used_feature_counts=used_feature_counts,
                signature_counts=signature_counts,
                current_fit=current_fit,
                train_matrix=train_matrix,
                target=np.asarray(y_train, dtype=float).reshape(-1),
                feature_names=feature_names,
                interference_context=interference_context,
                cfg=cfg,
            )
            scored_candidates.append(
                (
                    float(score),
                    float(candidate.complexity),
                    str(candidate.name),
                    candidate,
                )
            )
        if not scored_candidates:
            break
        scored_candidates.sort(key=lambda item: (-float(item[0]), float(item[1]), str(item[2])))
        topk = int(min(max(1, int(cfg.greedy_choice_topk)), len(scored_candidates)))
        if rng is None or topk <= 1:
            best_candidate = scored_candidates[0][3]
        else:
            choice_index = int(rng.integers(0, topk))
            best_candidate = scored_candidates[choice_index][3]
        selected.append(best_candidate)
        _increment_feature_reuse_budget(
            used_feature_counts,
            candidate=best_candidate,
            cfg=cfg,
        )
        signature_counts[str(best_candidate.semantic_signature)] += 1
    return selected


def _representative_rule_bonus(
    *,
    candidate: ScreenedCandidate,
    object_kind: str,
    representative_selection_rule: str,
) -> float:
    rule = str(representative_selection_rule or "balanced").strip().lower()
    if rule in {"periodic_truth_first", "periodic_family_first"}:
        return float(
            0.90 * float(candidate.periodic_prior)
            - 0.90 * float(candidate.periodic_penalty)
            + 0.10 * float(candidate.consensus_prior)
            + (0.20 if str(object_kind) == "periodic_channel" else 0.0)
        )
    if rule in {"proxy_sparse_first", "proxy_guard_first"}:
        return float(
            0.20 * float(candidate.consensus_prior)
            + 0.10 * float(candidate.mechanistic_prior)
            - 0.15 * float(candidate.complexity)
            + (0.10 if str(object_kind) == "single_source_object" else 0.0)
        )
    if rule in {"gate_residual_first", "regional_gate_first"}:
        return float(
            0.95 * max(0.0, float(candidate.residual_gain))
            + (0.35 if bool(candidate.uses_piecewise_gate) else 0.0)
            + 0.05 * float(candidate.semantic_novelty)
        )
    if rule in {"mechanistic_combo_first", "mechanism_first"}:
        return float(
            0.35 * float(candidate.mechanistic_prior)
            + 0.15 * float(candidate.residual_gain)
            + (0.15 if str(object_kind) == "mechanistic_object" else 0.0)
        )
    return float(
        0.35 * max(0.0, float(candidate.residual_gain))
        + 0.20 * float(candidate.periodic_prior)
        - 0.15 * float(candidate.periodic_penalty)
        + 0.10 * float(candidate.mechanistic_prior)
        + (0.12 if bool(candidate.uses_piecewise_gate) else 0.0)
    )


def _select_object_representative(
    *,
    candidate_object: CandidateObject,
    selected_rows: Sequence[ScreenedCandidate],
    corr_matrix: np.ndarray,
    used_feature_counts: Counter[int],
    signature_counts: Counter[str],
    current_fit: Mapping[str, Any],
    train_matrix: np.ndarray,
    target: np.ndarray,
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
    cfg: OrthogonalBasisSearchConfig,
    threshold: float,
    candidate_pool: Sequence[ScreenedCandidate],
) -> tuple[ScreenedCandidate, float] | None:
    member_candidates = tuple(candidate_object.members)
    if str(candidate_object.object_kind) not in {"gate_channel", "periodic_channel"}:
        canonical_members = tuple(
            candidate
            for candidate in tuple(candidate_object.members)
            if _screen_candidate_is_identity_source_representative(candidate)
        )
        if canonical_members:
            member_candidates = canonical_members
    best_candidate: ScreenedCandidate | None = None
    best_score: float | None = None
    for candidate in tuple(member_candidates):
        if not _accept_candidate(
            candidate=candidate,
            selected_rows=selected_rows,
            corr_matrix=corr_matrix,
            used_feature_counts=used_feature_counts,
            signature_counts=signature_counts,
            train_matrix=train_matrix,
            feature_names=feature_names,
            interference_context=interference_context,
            max_pair_abs_corr=float(threshold),
            max_feature_reuse=int(cfg.max_feature_reuse),
            cfg=cfg,
            candidate_pool=candidate_pool,
        ):
            continue
        base_score = _group_build_score(
            candidate=candidate,
            selected_rows=selected_rows,
            corr_matrix=corr_matrix,
            used_feature_counts=used_feature_counts,
            signature_counts=signature_counts,
            current_fit=current_fit,
            train_matrix=train_matrix,
            target=np.asarray(target, dtype=float).reshape(-1),
            feature_names=feature_names,
            interference_context=interference_context,
            cfg=cfg,
        )
        total_score = float(
            base_score
            + _representative_rule_bonus(
                candidate=candidate,
                object_kind=str(candidate_object.object_kind),
                representative_selection_rule=str(cfg.representative_selection_rule),
            )
        )
        if (
            best_candidate is None
            or best_score is None
            or total_score > best_score
            or (
                math.isclose(total_score, best_score)
                and (
                    float(candidate.screen_score) > float(best_candidate.screen_score)
                    or (
                        math.isclose(float(candidate.screen_score), float(best_candidate.screen_score))
                        and str(candidate.name) < str(best_candidate.name)
                    )
                )
            )
        ):
            best_candidate = candidate
            best_score = float(total_score)
    if best_candidate is None or best_score is None:
        return None
    return best_candidate, float(best_score)


def _object_expansions(
    *,
    rows: Sequence[ScreenedCandidate],
    candidate_pool: Sequence[ScreenedCandidate],
    candidate_objects: Sequence[CandidateObject],
    selected_object_keys: set[str],
    threshold: float,
    train_matrix: np.ndarray,
    y_train: np.ndarray,
    corr_matrix: np.ndarray,
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
    cfg: OrthogonalBasisSearchConfig,
    rng: np.random.Generator,
) -> list[tuple[CandidateObject, ScreenedCandidate, float]]:
    current_fit = _ridge_projection(
        _selected_matrix(train_matrix, rows),
        np.asarray(y_train, dtype=float).reshape(-1),
        l2_value=float(min(tuple(cfg.l2_grid)) if tuple(cfg.l2_grid) else 1e-6),
    )
    used_feature_counts: Counter[int] = Counter()
    signature_counts: Counter[str] = Counter()
    for row in tuple(rows):
        _increment_feature_reuse_budget(
            used_feature_counts,
            candidate=row,
            cfg=cfg,
        )
        signature_counts[str(row.semantic_signature)] += 1
    native_required = int(_required_native_trunk_basis_terms(cfg))
    native_selected = int(_native_trunk_term_count(rows))
    support_required = int(_required_support_expansion_basis_terms(cfg))
    support_selected = int(_support_expansion_term_count(rows))
    canonical_required = int(_required_canonical_trunk_basis_terms(cfg))
    canonical_selected = int(_canonical_trunk_term_count(rows))
    native_block_active = bool(
        native_required > native_selected
        and any(
            str(candidate_object.selection_channel) in {"native_trunk", "canonical_trunk", "support_expansion"}
            and str(candidate_object.object_key) not in selected_object_keys
            for candidate_object in tuple(candidate_objects)
        )
    )
    support_block_active = bool(
        support_required > support_selected
        and any(
            bool(candidate_object.support_expansion_candidate)
            and str(candidate_object.object_key) not in selected_object_keys
            for candidate_object in tuple(candidate_objects)
        )
    )
    canonical_block_active = bool(
        canonical_required > canonical_selected
        and any(
            (bool(candidate_object.canonical_trunk_candidate) or bool(candidate_object.support_expansion_candidate))
            and str(candidate_object.object_key) not in selected_object_keys
            for candidate_object in tuple(candidate_objects)
        )
    )
    scored_objects: list[tuple[float, float, str, CandidateObject, ScreenedCandidate]] = []
    for candidate_object in tuple(candidate_objects):
        if str(candidate_object.object_key) in selected_object_keys:
            continue
        if support_block_active and not bool(candidate_object.support_expansion_candidate):
            continue
        if canonical_block_active and not (
            bool(candidate_object.canonical_trunk_candidate) or bool(candidate_object.support_expansion_candidate)
        ):
            continue
        if native_block_active and str(candidate_object.selection_channel) != "native_trunk":
            if str(candidate_object.selection_channel) not in {"support_expansion", "canonical_trunk"}:
                continue
        if native_block_active and str(candidate_object.selection_channel) not in {"native_trunk", "canonical_trunk", "support_expansion"}:
            continue
        chosen = _select_object_representative(
            candidate_object=candidate_object,
            selected_rows=rows,
            corr_matrix=corr_matrix,
            used_feature_counts=used_feature_counts,
            signature_counts=signature_counts,
            current_fit=current_fit,
            train_matrix=train_matrix,
            target=np.asarray(y_train, dtype=float).reshape(-1),
                feature_names=feature_names,
                interference_context=interference_context,
                cfg=cfg,
                threshold=float(threshold),
                candidate_pool=candidate_pool,
            )
        if chosen is None:
            continue
        representative, score = chosen
        scored_objects.append(
            (
                float(score),
                float(representative.screen_score),
                str(candidate_object.object_key),
                candidate_object,
                representative,
            )
        )
    scored_objects.sort(key=lambda item: (-float(item[0]), -float(item[1]), str(item[2])))
    if not scored_objects:
        return []
    take = int(min(len(scored_objects), max(1, int(cfg.outer_search_branching_factor))))
    chosen_objects = [
        (item[3], item[4], float(item[0]))
        for item in scored_objects[:take]
    ]
    if int(cfg.random_group_trials) > 0 and len(scored_objects) > take:
        random_top = min(len(scored_objects), max(take + 1, int(cfg.greedy_choice_topk) * 2))
        random_index = int(rng.integers(take, random_top))
        random_item = scored_objects[random_index]
        chosen_objects.append((random_item[3], random_item[4], float(random_item[0])))
    return chosen_objects


def _basis_state_priority_key(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    orthogonality = dict(payload.get("orthogonality_metrics", {}) or {})
    screen_summary = dict(payload.get("screen_summary", {}) or {})
    return (
        -float(payload.get("group_score", 0.0) or 0.0),
        -float(screen_summary.get("mean_consensus_prior", 0.0) or 0.0),
        float(orthogonality.get("pair_abs_corr_mean", 1.0) or 1.0),
        -int(len(tuple(payload.get("screen_positions", ()) or ()))),
    )


def _discover_group_candidates(
    *,
    screened: Sequence[ScreenedCandidate],
    train_matrix: np.ndarray,
    y_train: np.ndarray,
    raw_X: np.ndarray,
    feature_names: Sequence[str],
    interference_context: Mapping[str, Any],
    periodic_context: Mapping[str, Any],
    cfg: OrthogonalBasisSearchConfig,
    seed_genome: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not screened:
        return []
    rng = np.random.default_rng(int(cfg.random_seed))
    corr_matrix = _pairwise_abs_corr(train_matrix)
    seed_limit = min(int(cfg.seed_candidate_count), int(len(screened)))
    candidate_objects = _build_candidate_objects(
        screened=screened,
        feature_names=feature_names,
        interference_context=interference_context,
        periodic_context=periodic_context,
        outer_search_unit=str(cfg.outer_search_unit),
    )
    thresholds = (
        float(cfg.max_pair_abs_corr),
        float(min(0.55, cfg.max_pair_abs_corr + 0.10)),
        float(min(0.75, cfg.max_pair_abs_corr + 0.25)),
    )
    seen_groups: set[tuple[int, ...]] = set()
    groups: list[dict[str, Any]] = []
    beam_width = int(max(2, cfg.outer_search_beam_width))
    branch_factor = int(max(1, cfg.outer_search_branching_factor))
    max_expansions = int(max(8, cfg.outer_search_max_expansions))

    forced_seed_rows = _match_seed_rows(screened=screened, seed_genome=seed_genome)
    forced_seed_screen_positions = {int(row.screen_index) for row in tuple(forced_seed_rows)}
    forced_seed_object_keys = {
        _candidate_object_key(
            candidate=row,
            feature_names=feature_names,
            interference_context=interference_context,
            periodic_context=periodic_context,
            outer_search_unit=str(cfg.outer_search_unit),
        )
        for row in tuple(forced_seed_rows)
    }
    seed_objects = list(tuple(candidate_objects)[: min(seed_limit, len(candidate_objects))])
    if seed_objects and (int(cfg.greedy_choice_topk) > 1 or int(cfg.random_group_trials) > 0):
        rng.shuffle(seed_objects)
    base_seed_objects = tuple(seed_objects) if seed_objects else tuple(candidate_objects[:seed_limit])
    support_expansion_objects = tuple(
        obj for obj in tuple(candidate_objects) if bool(obj.support_expansion_candidate)
    )
    canonical_trunk_objects = tuple(
        obj
        for obj in tuple(candidate_objects)
        if bool(obj.canonical_trunk_candidate) or bool(obj.support_expansion_candidate)
    )
    native_trunk_objects = tuple(
        obj
        for obj in tuple(candidate_objects)
        if str(obj.selection_channel) in {"native_trunk", "canonical_trunk", "support_expansion"}
    )
    exposure_objects = tuple(
        obj
        for obj in tuple(candidate_objects)
        if any(bool(row.heterogeneous_exposure_eligible) for row in tuple(obj.members))
        and str(obj.selection_channel) not in {"native_trunk", "support_expansion"}
    )
    gate_objects = tuple(obj for obj in tuple(candidate_objects) if str(obj.object_kind) == "gate_channel")
    periodic_objects = tuple(obj for obj in tuple(candidate_objects) if str(obj.object_kind) == "periodic_channel")
    support_expansion_candidates_available = bool(support_expansion_objects)
    canonical_trunk_candidates_available = bool(canonical_trunk_objects)
    native_candidates_available = bool(native_trunk_objects)
    exposure_candidates_available = bool(exposure_objects)
    gate_candidates_available = bool(gate_objects)
    periodic_candidates_available = bool(periodic_objects)

    def _row_key(rows: Sequence[ScreenedCandidate]) -> tuple[int, ...]:
        return tuple(sorted(int(row.screen_index) for row in tuple(rows)))

    def _normalize_seed_rows(rows: Sequence[ScreenedCandidate]) -> tuple[ScreenedCandidate, ...]:
        ordered: list[ScreenedCandidate] = []
        seen: set[int] = set()
        for row in tuple(rows):
            index = int(row.screen_index)
            if index in seen:
                continue
            seen.add(index)
            ordered.append(row)
        ordered.sort(key=lambda row: int(row.screen_index))
        return tuple(ordered)

    def _register_payload(payload: Mapping[str, Any]) -> None:
        group_key = tuple(int(value) for value in tuple(payload.get("pool_indices", ()) or ()))
        if group_key in seen_groups:
            return
        rows = tuple(payload.get("rows", ()) or ())
        if not _group_meets_native_trunk_requirement(
            rows=rows,
            cfg=cfg,
            native_candidates_available=native_candidates_available,
        ):
            return
        if not _group_meets_support_expansion_requirement(
            rows=rows,
            cfg=cfg,
            support_expansion_candidates_available=support_expansion_candidates_available,
        ):
            return
        if not _group_meets_canonical_trunk_requirement(
            rows=rows,
            cfg=cfg,
            canonical_trunk_candidates_available=canonical_trunk_candidates_available,
        ):
            return
        if not _group_meets_gate_requirement(
            rows=rows,
            cfg=cfg,
            gate_candidates_available=gate_candidates_available,
        ):
            return
        if not _group_meets_periodic_requirement(
            rows=rows,
            cfg=cfg,
            periodic_context=periodic_context,
            periodic_candidates_available=periodic_candidates_available,
        ):
            return
        seen_groups.add(group_key)
        groups.append(dict(payload))

    def _seed_rows_from_objects(
        objects: Sequence[CandidateObject],
    ) -> tuple[ScreenedCandidate, ...]:
        selected_seed_rows: list[ScreenedCandidate] = []
        for candidate_object in tuple(objects):
            used_feature_counts: Counter[int] = Counter()
            signature_counts: Counter[str] = Counter()
            for row in tuple(selected_seed_rows):
                _increment_feature_reuse_budget(
                    used_feature_counts,
                    candidate=row,
                    cfg=cfg,
                )
                signature_counts[str(row.semantic_signature)] += 1
            chosen = _select_object_representative(
                candidate_object=candidate_object,
                selected_rows=tuple(selected_seed_rows),
                corr_matrix=corr_matrix,
                used_feature_counts=used_feature_counts,
                signature_counts=signature_counts,
                current_fit=_ridge_projection(
                    _selected_matrix(train_matrix, tuple(selected_seed_rows)),
                    np.asarray(y_train, dtype=float).reshape(-1),
                    l2_value=float(min(tuple(cfg.l2_grid)) if tuple(cfg.l2_grid) else 1e-6),
                ),
                train_matrix=train_matrix,
                target=np.asarray(y_train, dtype=float).reshape(-1),
                feature_names=feature_names,
                interference_context=interference_context,
                cfg=cfg,
                threshold=float(cfg.max_pair_abs_corr),
                candidate_pool=screened,
            )
            if chosen is None:
                continue
            selected_seed_rows.append(chosen[0])
        return _normalize_seed_rows(tuple(selected_seed_rows))

    def _seed_rows_with_gate_parent(
        candidate_object: CandidateObject,
    ) -> tuple[ScreenedCandidate, ...]:
        parent_source_keys = tuple(
            dict.fromkeys(
                str(parent_source_key)
                for parent_source_key in (
                    _candidate_gate_parent_source_key(member)
                    for member in tuple(candidate_object.members)
                )
                if str(parent_source_key).strip()
            )
        )
        for parent_source_key in parent_source_keys:
            parent_objects = [
                obj
                for obj in tuple(candidate_objects)
                if str(obj.object_key) != str(candidate_object.object_key)
                and any(
                    _row_matches_parent_trunk_source(row, parent_source_key=str(parent_source_key))
                    for row in tuple(obj.members)
                )
            ]
            parent_objects.sort(
                key=lambda obj: (
                    0 if any(bool(row.native_trunk_floor_passed) for row in tuple(obj.members)) else 1,
                    0 if str(obj.selection_channel) in {"native_trunk", "support_expansion"} else 1,
                    -max(float(row.screen_score) for row in tuple(obj.members)),
                    str(obj.object_key),
                )
            )
            for parent_object in tuple(parent_objects[:2]):
                rows = _seed_rows_from_objects((parent_object, candidate_object))
                if rows:
                    return rows
        return tuple()

    def _complete_seed_rows(
        rows: Sequence[ScreenedCandidate],
        *,
        threshold: float,
    ) -> tuple[ScreenedCandidate, ...]:
        selected_rows = _normalize_seed_rows(rows)
        while len(selected_rows) < int(cfg.max_basis_count):
            selected_object_keys = {
                _candidate_object_key(
                    candidate=row,
                    feature_names=feature_names,
                    interference_context=interference_context,
                    periodic_context=periodic_context,
                    outer_search_unit=str(cfg.outer_search_unit),
                )
                for row in tuple(selected_rows)
            }
            expansions = _object_expansions(
                rows=selected_rows,
                candidate_pool=screened,
                candidate_objects=candidate_objects,
                selected_object_keys=set(str(value) for value in selected_object_keys),
                threshold=float(threshold),
                train_matrix=train_matrix,
                y_train=y_train,
                corr_matrix=corr_matrix,
                feature_names=feature_names,
                interference_context=interference_context,
                cfg=cfg,
                rng=rng,
            )
            if not expansions:
                break
            next_candidate: ScreenedCandidate | None = None
            for candidate_object, candidate, _score in tuple(expansions):
                if str(candidate_object.object_key) in selected_object_keys:
                    continue
                next_candidate = candidate
                break
            if next_candidate is None:
                break
            selected_rows = _normalize_seed_rows(tuple(selected_rows) + (next_candidate,))
            if len(selected_rows) >= int(cfg.max_basis_count):
                break
        return selected_rows

    initial_seed_sets: list[tuple[ScreenedCandidate, ...]] = []
    if forced_seed_rows:
        initial_seed_sets.append(_normalize_seed_rows(forced_seed_rows))
    if native_candidates_available and _required_native_trunk_basis_terms(cfg) > 0:
        native_take = min(
            len(native_trunk_objects),
            max(int(_required_native_trunk_basis_terms(cfg)), min(2, int(cfg.native_trunk_candidate_screen_reserve))),
        )
        for candidate_object in native_trunk_objects[:native_take]:
            rows = _seed_rows_from_objects((candidate_object,))
            if rows:
                initial_seed_sets.append(rows)
        if native_take >= 2:
            rows = _seed_rows_from_objects(tuple(native_trunk_objects[:2]))
            if rows:
                initial_seed_sets.append(rows)
    if support_expansion_candidates_available and _required_support_expansion_basis_terms(cfg) > 0:
        support_take = min(
            len(support_expansion_objects),
            max(
                int(_required_support_expansion_basis_terms(cfg)),
                min(2, int(cfg.support_expansion_candidate_screen_reserve)),
            ),
        )
        for candidate_object in support_expansion_objects[:support_take]:
            rows = _seed_rows_from_objects((candidate_object,))
            if rows:
                initial_seed_sets.append(rows)
    if canonical_trunk_candidates_available and _required_canonical_trunk_basis_terms(cfg) > 0:
        canonical_take = min(
            len(canonical_trunk_objects),
            max(
                int(_required_canonical_trunk_basis_terms(cfg)),
                min(2, int(cfg.canonical_trunk_candidate_screen_reserve)),
            ),
        )
        for candidate_object in canonical_trunk_objects[:canonical_take]:
            rows = _seed_rows_from_objects((candidate_object,))
            if rows:
                initial_seed_sets.append(rows)
    if exposure_candidates_available and _heterogeneous_exposure_enabled(cfg):
        exposure_take = min(
            len(exposure_objects),
            max(1, int(cfg.heterogeneous_exposure_candidate_screen_reserve)),
        )
        for candidate_object in exposure_objects[:exposure_take]:
            rows = _seed_rows_from_objects((candidate_object,))
            if rows:
                initial_seed_sets.append(rows)
    if periodic_candidates_available and _required_periodic_basis_terms(cfg=cfg, periodic_context=periodic_context) > 0:
        periodic_take = min(len(periodic_objects), max(1, int(cfg.greedy_choice_topk)))
        for candidate_object in periodic_objects[:periodic_take]:
            rows = _seed_rows_from_objects((candidate_object,))
            if rows:
                initial_seed_sets.append(rows)
    if gate_candidates_available and _required_gate_basis_terms(cfg) > 0:
        gate_take = min(len(gate_objects), max(1, int(cfg.greedy_choice_topk)))
        for candidate_object in gate_objects[:gate_take]:
            rows = _seed_rows_from_objects((candidate_object,))
            if rows:
                initial_seed_sets.append(rows)
                continue
            if _parasitic_rejection_enabled(cfg):
                parent_rows = _seed_rows_with_gate_parent(candidate_object)
                if parent_rows:
                    initial_seed_sets.append(parent_rows)
    if periodic_candidates_available and gate_candidates_available:
        combo_take = min(
            max(1, int(cfg.greedy_choice_topk)),
            len(periodic_objects),
            len(gate_objects),
        )
        for index in range(combo_take):
            rows = _seed_rows_from_objects((periodic_objects[index], gate_objects[index]))
            if rows:
                initial_seed_sets.append(rows)
    for seed_object in tuple(base_seed_objects):
        rows = _seed_rows_from_objects((seed_object,))
        if rows:
            initial_seed_sets.append(rows)
    random_group_trials = int(cfg.random_group_trials)
    if random_group_trials > 0 and base_seed_objects:
        max_seed_take = min(2, len(base_seed_objects))
        for _ in range(random_group_trials):
            seed_take = 1 if max_seed_take <= 1 else int(rng.integers(1, max_seed_take + 1))
            trial_indices = rng.choice(len(base_seed_objects), size=seed_take, replace=False)
            trial_objects = tuple(base_seed_objects[int(index)] for index in np.atleast_1d(trial_indices))
            rows = _seed_rows_from_objects(trial_objects)
            if rows:
                initial_seed_sets.append(rows)

    seen_seed_sets: set[tuple[int, ...]] = set()
    deduped_seed_sets: list[tuple[ScreenedCandidate, ...]] = []
    for rows in initial_seed_sets:
        key = _row_key(rows)
        if not key or key in seen_seed_sets:
            continue
        if bool(cfg.lock_seed_basis) and forced_seed_rows and not forced_seed_screen_positions.issubset(set(key)):
            continue
        seen_seed_sets.add(key)
        deduped_seed_sets.append(rows)

    for threshold in thresholds:
        frontier: list[dict[str, Any]] = []
        frontier_seen: set[tuple[int, ...]] = set()
        for rows in deduped_seed_sets:
            key = _row_key(rows)
            if key in frontier_seen:
                continue
            frontier_seen.add(key)
            frontier.append(
                {
                    "rows": rows,
                    "payload": _group_summary_payload(
                        selected_rows=rows,
                        threshold=float(threshold),
                        train_matrix=train_matrix,
                        target=np.asarray(y_train, dtype=float).reshape(-1),
                        raw_X=raw_X,
                        feature_names=feature_names,
                        interference_context=interference_context,
                        periodic_context=periodic_context,
                        cfg=cfg,
                        fallback_mode="seeded_restart" if rows == tuple(forced_seed_rows) else None,
                    ),
                }
            )

        expansions = 0
        while frontier and expansions < max_expansions:
            next_frontier_map: dict[tuple[int, ...], dict[str, Any]] = {}
            for state in frontier:
                rows = tuple(state.get("rows", ()))
                payload = dict(state.get("payload", {}))
                if len(rows) >= int(cfg.min_basis_count):
                    _register_payload(payload)
                if len(rows) >= int(cfg.max_basis_count):
                    continue
                expansions += 1
                selected_object_keys = {
                    _candidate_object_key(
                        candidate=row,
                        feature_names=feature_names,
                        interference_context=interference_context,
                        periodic_context=periodic_context,
                        outer_search_unit=str(cfg.outer_search_unit),
                    )
                    for row in tuple(rows)
                }
                for candidate_object, candidate, _score in _object_expansions(
                    rows=rows,
                    candidate_pool=screened,
                    candidate_objects=candidate_objects,
                    selected_object_keys=set(str(value) for value in selected_object_keys),
                    threshold=float(threshold),
                    train_matrix=train_matrix,
                    y_train=y_train,
                    corr_matrix=corr_matrix,
                    feature_names=feature_names,
                    interference_context=interference_context,
                    cfg=cfg,
                    rng=rng,
                ):
                    if str(candidate_object.object_key) in selected_object_keys:
                        continue
                    new_rows = _normalize_seed_rows(tuple(rows) + (candidate,))
                    state_key = _row_key(new_rows)
                    if state_key in next_frontier_map:
                        continue
                    new_payload = _group_summary_payload(
                        selected_rows=new_rows,
                        threshold=float(threshold),
                        train_matrix=train_matrix,
                        target=np.asarray(y_train, dtype=float).reshape(-1),
                        raw_X=raw_X,
                        feature_names=feature_names,
                        interference_context=interference_context,
                        periodic_context=periodic_context,
                        cfg=cfg,
                    )
                    if bool(cfg.lock_seed_basis) and forced_seed_rows:
                        state_positions = {int(value) for value in tuple(new_payload.get("screen_positions", ()) or ())}
                        if not forced_seed_screen_positions.issubset(state_positions):
                            continue
                    next_frontier_map[state_key] = {
                        "rows": new_rows,
                        "payload": new_payload,
                    }
                if expansions >= max_expansions:
                    break
            frontier = sorted(
                next_frontier_map.values(),
                key=lambda item: _basis_state_priority_key(dict(item.get("payload", {}))),
            )[:beam_width]

    if bool(cfg.lock_seed_basis) and forced_seed_rows and groups:
        groups.sort(key=_basis_state_priority_key)
        return groups[: int(cfg.group_count)]

    if groups:
        groups.sort(key=_basis_state_priority_key)
        return groups[: int(cfg.group_count)]

    if forced_seed_rows:
        relaxed_selected = _complete_seed_rows(forced_seed_rows, threshold=0.98)
        if len(relaxed_selected) >= int(cfg.min_basis_count):
            payload = _group_summary_payload(
                selected_rows=relaxed_selected,
                threshold=0.98,
                train_matrix=train_matrix,
                target=np.asarray(y_train, dtype=float).reshape(-1),
                raw_X=raw_X,
                feature_names=feature_names,
                interference_context=interference_context,
                periodic_context=periodic_context,
                cfg=cfg,
                fallback_mode="seeded_relaxed_threshold",
            )
            if _group_meets_gate_requirement(
                rows=relaxed_selected,
                cfg=cfg,
                gate_candidates_available=gate_candidates_available,
            ) and _group_meets_periodic_requirement(
                rows=relaxed_selected,
                cfg=cfg,
                periodic_context=periodic_context,
                periodic_candidates_available=periodic_candidates_available,
            ) and _group_meets_native_trunk_requirement(
                rows=relaxed_selected,
                cfg=cfg,
                native_candidates_available=native_candidates_available,
            ):
                return [payload]

    if gate_candidates_available and _required_gate_basis_terms(cfg) > 0:
        gate_seed = _seed_rows_from_objects(tuple(gate_objects[:1]))
        gate_relaxed_selected = _complete_seed_rows(gate_seed, threshold=0.98)
        if len(gate_relaxed_selected) >= int(cfg.min_basis_count) and _group_meets_gate_requirement(
            rows=gate_relaxed_selected,
            cfg=cfg,
            gate_candidates_available=gate_candidates_available,
        ) and _group_meets_periodic_requirement(
            rows=gate_relaxed_selected,
            cfg=cfg,
            periodic_context=periodic_context,
            periodic_candidates_available=periodic_candidates_available,
        ) and _group_meets_native_trunk_requirement(
            rows=gate_relaxed_selected,
            cfg=cfg,
            native_candidates_available=native_candidates_available,
        ):
            return [
                _group_summary_payload(
                    selected_rows=gate_relaxed_selected,
                    threshold=0.98,
                    train_matrix=train_matrix,
                    target=np.asarray(y_train, dtype=float).reshape(-1),
                    raw_X=raw_X,
                    feature_names=feature_names,
                    interference_context=interference_context,
                    periodic_context=periodic_context,
                    cfg=cfg,
                    fallback_mode="gate_seed_relaxed_threshold",
                )
            ]

    relaxed_selected = _complete_seed_rows((screened[0],), threshold=0.98)
    if len(relaxed_selected) >= int(cfg.min_basis_count) and _group_meets_gate_requirement(
        rows=relaxed_selected,
        cfg=cfg,
        gate_candidates_available=gate_candidates_available,
    ) and _group_meets_periodic_requirement(
        rows=relaxed_selected,
        cfg=cfg,
        periodic_context=periodic_context,
        periodic_candidates_available=periodic_candidates_available,
    ) and _group_meets_native_trunk_requirement(
        rows=relaxed_selected,
        cfg=cfg,
        native_candidates_available=native_candidates_available,
    ):
        return [
            _group_summary_payload(
                selected_rows=relaxed_selected,
                threshold=0.98,
                train_matrix=train_matrix,
                target=np.asarray(y_train, dtype=float).reshape(-1),
                raw_X=raw_X,
                feature_names=feature_names,
                interference_context=interference_context,
                periodic_context=periodic_context,
                cfg=cfg,
                fallback_mode="relaxed_threshold",
            )
        ]
    return []


def _fold_rows_for_l2_grid(
    *,
    genome: Sequence[Mapping[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    l2_grid: Sequence[float],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    alpha: float,
    graph_cache: ExpressionGraphCache | None,
) -> dict[float, list[dict[str, Any]]]:
    rows_by_l2: dict[float, list[dict[str, Any]]] = {float(value): [] for value in l2_grid}
    for fold_index, (tr_idx, va_idx) in enumerate(tuple(splits)):
        x_train = np.asarray(X_train[tr_idx], dtype=float)
        y_train_fold = np.asarray(y_train[tr_idx], dtype=float)
        x_valid = np.asarray(X_train[va_idx], dtype=float)
        y_valid = np.asarray(y_train[va_idx], dtype=float)
        genomes = [list(genome) for _ in l2_grid]
        pred_eval, pred_train = batched_ridge_predict(
            genomes=genomes,
            X_train=x_train,
            y_train=y_train_fold,
            X_eval=x_valid,
            l2_values=[float(value) for value in l2_grid],
            graph_cache=graph_cache,
            batch_key_train=f"orthogonal_fold_{fold_index}_train",
            batch_key_eval=f"orthogonal_fold_{fold_index}_eval",
        )
        lower, upper, quantiles = symmetric_interval_batch(
            y_train=y_train_fold,
            pred_train=pred_train,
            pred_eval=pred_eval,
            alpha=float(alpha),
        )
        metrics = interval_metrics_batch(
            y_true=y_valid,
            lower=lower,
            upper=upper,
            alpha=float(alpha),
        )
        for index, l2_value in enumerate(tuple(float(value) for value in l2_grid)):
            rows_by_l2[float(l2_value)].append(
                {
                    "coverage_error": float(metrics["coverage_error"][index]),
                    "picp": float(metrics["picp"][index]),
                    "pinaw": float(metrics["pinaw"][index]),
                    "interval_score": float(metrics["interval_score"][index]),
                    "mean_width": float(metrics["mean_width"][index]),
                    "rmse": float(_rmse(y_valid, pred_eval[index])),
                    "branch_detail": {
                        "fold_index": int(fold_index),
                        "l2": float(l2_value),
                    },
                    "interval_info": {
                        "symmetric_residual_q": float(quantiles[index]),
                    },
                }
            )
    return rows_by_l2


def _validation_sort_key(
    *,
    summary_detail: Mapping[str, Any],
    orthogonality_metrics: Mapping[str, Any],
    cfg: OrthogonalBasisSearchConfig,
) -> tuple[Any, ...]:
    interval_key = interval_objective_sort_key(
        coverage_error_value=float(summary_detail.get("coverage_error_mean", float("inf"))),
        pinaw=float(summary_detail.get("pinaw_mean", float("inf"))),
        interval_score=float(summary_detail.get("interval_score_mean", float("inf"))),
        coverage_error_threshold=float(cfg.coverage_error_threshold),
    )
    rmse_mean = float(summary_detail.get("rmse_mean", float("inf")))
    orthogonality_score = float(orthogonality_metrics.get("orthogonality_score", 0.0))
    pair_abs_corr_mean = float(orthogonality_metrics.get("pair_abs_corr_mean", 1.0))
    if str(cfg.selection_mode) == "orthogonal_first":
        return (-orthogonality_score, pair_abs_corr_mean, *interval_key, rmse_mean)
    if str(cfg.selection_mode) == "rmse_first":
        return (rmse_mean, *interval_key, -orthogonality_score, pair_abs_corr_mean)
    return (*interval_key, rmse_mean, -orthogonality_score, pair_abs_corr_mean)


def _build_expression_payload(
    *,
    genome: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    weight: np.ndarray,
    bias: np.ndarray,
) -> dict[str, Any]:
    basis_rows = build_basis_term_rows(
        genome,
        feature_names=tuple(str(value) for value in feature_names),
        scope="global",
    )
    coeff = np.asarray(weight, dtype=float).reshape(-1)
    intercept = float(np.asarray(bias, dtype=float).reshape(-1)[0]) if np.asarray(bias).size > 0 else 0.0
    terms: list[dict[str, Any]] = []
    expr_parts: list[str] = []
    for row, coef in zip(basis_rows, coeff):
        expression = str(row.get("expression", row.get("term_name", "")))
        if abs(float(coef)) <= 1e-12:
            continue
        terms.append(
            {
                "term_name": str(row.get("term_name", "")),
                "expression": expression,
                "coefficient": float(coef),
                "feature_names": [str(value) for value in tuple(row.get("feature_names", ()))],
            }
        )
        expr_parts.append(f"({float(coef):.8g})*({expression})")
    expr_parts.append(f"({float(intercept):.8g})")
    return {
        "expression": " + ".join(expr_parts),
        "terms": terms,
        "intercept": float(intercept),
    }


def fit_orthogonal_basis_symbolic(
    *,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    cfg: OrthogonalBasisSearchConfig | None = None,
    graph_cache: ExpressionGraphCache | None = None,
    seed_genome: Sequence[Mapping[str, Any]] | None = None,
    consensus_prior_rows: Sequence[Mapping[str, Any]] | None = None,
    symbolic_family_payload: Mapping[str, Any] | None = None,
    data_metadata: Mapping[str, Any] | None = None,
) -> OrthogonalBasisFitResult:
    resolved_cfg = (cfg or OrthogonalBasisSearchConfig()).normalized()
    x = np.asarray(X, dtype=float)
    yy = np.asarray(y, dtype=float)
    if yy.ndim == 1:
        yy = yy.reshape(-1, 1)
    bundle = _minimal_feature_bundle(X=x, y=yy, feature_names=feature_names)
    gate_specs = _build_piecewise_gate_specs(feature_bundle=bundle, cfg=resolved_cfg)
    pool_cfg = CandidatePoolConfig(conditional_config=tuple(gate_specs)) if gate_specs else CandidatePoolConfig()
    candidate_pool = list(build_full_candidate_pool(bundle, pool_cfg))
    realization_evidence_registry = _build_realization_evidence_registry(
        candidate_pool=tuple(candidate_pool),
        data_metadata=data_metadata,
        feature_names=tuple(bundle.feature_names),
        cfg=resolved_cfg,
    )
    consensus_prior_model = _build_consensus_prior_model(consensus_prior_rows)
    interference_context = _build_interference_context(
        raw_X=np.asarray(bundle.X_train, dtype=float),
        feature_names=tuple(bundle.feature_names),
        data_metadata=data_metadata,
        proxy_group_policy=str(resolved_cfg.proxy_group_policy),
    )
    periodic_context = _build_periodic_context(
        raw_X=np.asarray(bundle.X_train, dtype=float),
        feature_names=tuple(bundle.feature_names),
        data_metadata=data_metadata,
        cfg=resolved_cfg,
    )
    screened, train_matrix = _screen_candidate_pool(
        candidates=candidate_pool,
        X_train=np.asarray(bundle.X_train, dtype=float),
        y_train=np.asarray(bundle.y_train, dtype=float),
        feature_names=tuple(bundle.feature_names),
        candidate_limit=int(resolved_cfg.candidate_limit),
        cfg=resolved_cfg,
        graph_cache=graph_cache,
        consensus_prior_model=consensus_prior_model,
        interference_context=interference_context,
        periodic_context=periodic_context,
    )
    if not screened:
        raise RuntimeError("no valid symbolic candidates were screened for orthogonal basis discovery")

    group_payloads = _discover_group_candidates(
        screened=screened,
        train_matrix=train_matrix,
        y_train=np.asarray(bundle.y_train, dtype=float),
        raw_X=np.asarray(bundle.X_train, dtype=float),
        feature_names=tuple(bundle.feature_names),
        interference_context=interference_context,
        periodic_context=periodic_context,
        cfg=resolved_cfg,
        seed_genome=seed_genome,
    )
    if not group_payloads:
        raise RuntimeError("no orthogonal basis groups were generated from the screened candidate pool")

    evaluation_rows: list[dict[str, Any]] = []
    for group_payload in group_payloads:
        pool_indices = [int(value) for value in tuple(group_payload.get("pool_indices", ()))]
        selected_rows = tuple(group_payload.get("rows", ()) or ())
        if selected_rows:
            outer_basis_genome = [
                {
                    "name": str(row.name),
                    "expr": dict(row.expr),
                }
                for row in selected_rows
            ]
        else:
            outer_basis_genome = build_subset_genome(candidates=candidate_pool, subset_idx=pool_indices)
        orthogonality_metrics = dict(group_payload.get("orthogonality_metrics", {}))
        basis_rows = build_basis_term_rows(
            outer_basis_genome,
            feature_names=tuple(str(value) for value in bundle.feature_names),
            scope="global",
        )
        gate_feature_names = tuple(_configured_gate_feature_names(cfg=resolved_cfg, feature_names=bundle.feature_names))
        gate_feature_set = set(gate_feature_names)
        gate_basis_rows = [
            dict(row)
            for row in tuple(basis_rows)
            if bool(row.get("uses_piecewise_gate"))
            or bool(gate_feature_set & {str(name) for name in tuple(row.get("feature_names", ()))})
        ]
        residual_report = dict(group_payload.get("residual_complementarity_report", {}))
        semantic_report = dict(group_payload.get("semantic_dedup_report", {}))
        interference_report = dict(group_payload.get("interference_feature_report", {}))
        orthogonality_metrics = dict(group_payload.get("orthogonality_metrics", {}))
        orthogonality_metrics["semantic_unique_ratio"] = float(semantic_report.get("semantic_unique_ratio", 0.0))
        orthogonality_metrics["piecewise_gate_term_count"] = int(semantic_report.get("piecewise_gate_term_count", 0))
        orthogonality_metrics["residual_gain_mean"] = float(residual_report.get("mean_marginal_r2_gain", 0.0))
        orthogonality_metrics["residual_gain_min"] = float(residual_report.get("min_marginal_r2_gain", 0.0))
        assembler_result = _run_budgeted_symbolic_assembler(
            outer_basis_genome=outer_basis_genome,
            basis_rows=basis_rows,
            pool_indices=pool_indices,
            selected_rows=tuple(group_payload.get("rows", ())),
            train_matrix=train_matrix,
            target=np.asarray(bundle.y_train, dtype=float),
            raw_X=np.asarray(bundle.X_train, dtype=float),
            raw_feature_names=tuple(bundle.feature_names),
            cfg=resolved_cfg,
            graph_cache=graph_cache,
            orthogonality_metrics=orthogonality_metrics,
            residual_report=residual_report,
            semantic_report=semantic_report,
            interference_report=interference_report,
            screened_candidates=screened,
            interference_context=interference_context,
            periodic_context=periodic_context,
            realization_evidence_registry=realization_evidence_registry,
            gate_feature_names=gate_feature_names,
            data_metadata=data_metadata,
        )
        final_fit = dict(assembler_result.final_fit)
        pred_train = np.asarray(final_fit.get("pred_train"), dtype=float)
        if pred_train.ndim == 1:
            pred_train = pred_train.reshape(-1, 1)
        basis_semantics = build_basis_semantics_payload(
            basis_rows,
            source="orthogonal_basis_discovery",
            basis_scope="global",
            extra={
                "selection_mode": str(resolved_cfg.selection_mode),
                "relative_orthogonality": True,
                "selection_threshold": float(group_payload.get("threshold", resolved_cfg.max_pair_abs_corr)),
                "gate_feature_names": list(gate_feature_names),
                "semantic_unique_ratio": float(semantic_report.get("semantic_unique_ratio", 0.0)),
            },
        )
        basis_overlap_report = build_basis_overlap_report(
            basis_rows,
            source="orthogonal_basis_discovery",
            extra={
                "orthogonality_score": float(orthogonality_metrics.get("orthogonality_score", 0.0)),
                "pair_abs_corr_mean": float(orthogonality_metrics.get("pair_abs_corr_mean", 0.0)),
                "pair_abs_corr_max": float(orthogonality_metrics.get("pair_abs_corr_max", 0.0)),
                "semantic_unique_ratio": float(semantic_report.get("semantic_unique_ratio", 0.0)),
                "residual_gain_mean": float(residual_report.get("mean_marginal_r2_gain", 0.0)),
            },
        )
        assembler_budget = build_assembler_budget_payload(
            source="orthogonal_budgeted_symbolic_assembler",
            assembler_mode="budgeted_symbolic_regression",
            output_expression_count=1,
            selected_basis_count=int(len(pool_indices)),
            budget_axes={
                "basis_count": int(len(pool_indices)),
                "candidate_limit": int(resolved_cfg.candidate_limit),
                "assembler_max_terms": int(resolved_cfg.assembler_max_added_terms),
                "assembler_max_candidates_per_iter": int(resolved_cfg.assembler_max_candidates_per_iter),
                "assembler_max_expr_depth": int(resolved_cfg.assembler_max_expr_depth),
                "assembler_ridge_l2": float(resolved_cfg.assembler_ridge_l2),
            },
            budget_scale="small",
            uses_piecewise_gate=bool(gate_basis_rows),
            extra={
                "selection_mode": str(resolved_cfg.selection_mode),
                "gate_basis_count": int(len(gate_basis_rows)),
                "inner_iterations": int(len(tuple(assembler_result.inner_result.iterations))),
                "inner_terms": int(len(tuple(assembler_result.inner_result.genome))),
            },
        )
        expression_payload = dict(assembler_result.final_expression_payload)
        stage_head_protocols = _jsonable(dict(assembler_result.stage_head_protocols))
        assembler_stage_payload = dict(dict(assembler_result.stage_head_protocols).get("assembler", {}) or {})
        basis_context_payload = _jsonable(dict(assembler_result.basis_context))
        object_gradient_pool_payload = _jsonable(dict(assembler_result.object_gradient_pool))
        stage_head_summary = {
            "structure_head": assembler_stage_payload.get("structure_head"),
            "prediction_head": assembler_stage_payload.get("prediction_head"),
            "search_input_space": assembler_stage_payload.get("search_input_space"),
            "pool_expansion_unit": assembler_stage_payload.get("pool_expansion_unit"),
            "gradient_guidance_mode": assembler_stage_payload.get("gradient_guidance_mode"),
            "basis_binding_mode": assembler_stage_payload.get("basis_binding_mode"),
            "escape_policy": assembler_stage_payload.get("escape_policy"),
            "basis_source": dict(assembler_result.basis_context).get("basis_source"),
            "orchestration_mode": "basis_discovery_then_basis_conditioned_expression",
        }
        gate_indices = [
            int(index)
            for index, name in enumerate(tuple(str(value) for value in tuple(bundle.feature_names)))
            if str(name) in gate_feature_set
        ]
        surface_metadata = {
            "selected_basis": _jsonable(basis_rows),
            "basis_semantics": _jsonable(basis_semantics),
            "basis_overlap_report": _jsonable(basis_overlap_report),
            "residual_complementarity_report": _jsonable(residual_report),
            "semantic_dedup_report": _jsonable(semantic_report),
            "interference_feature_report": _jsonable(interference_report),
            "environment_invariance_audit": _jsonable(dict(assembler_result.environment_invariance_audit)),
            "periodic_equivalence_report": _jsonable(dict(assembler_result.periodic_equivalence_report)),
            "regional_correction_report": _jsonable(dict(assembler_result.regional_correction_report)),
            "mandatory_realization_closure_report": _jsonable(
                dict(assembler_result.mandatory_realization_closure_report)
            ),
            "same_source_over_realization_report": _jsonable(
                dict(assembler_result.same_source_over_realization_report)
            ),
            "assembler_budget": _jsonable(assembler_budget),
            "orthogonal_outer_basis_genome": _jsonable(tuple(dict(term) for term in tuple(outer_basis_genome))),
            "inner_symbolic_search": {
                "protocol": "budgeted_symbolic_assembler",
                "basis_feature_names": [str(value) for value in tuple(assembler_result.basis_feature_names)],
                "search_config": _jsonable(assembler_result.search_config),
                "basis_space_genome": _jsonable(tuple(dict(term) for term in tuple(assembler_result.basis_space_genome))),
                "assembled_genome": _jsonable(tuple(dict(term) for term in tuple(assembler_result.assembled_genome))),
                "base_metrics": _jsonable(dict(assembler_result.inner_result.base_metrics)),
                "final_metrics": _jsonable(dict(assembler_result.inner_result.final_metrics)),
                "score_trace": [float(value) for value in tuple(assembler_result.inner_result.score_trace)],
                "iterations": _jsonable([dict(item) for item in tuple(assembler_result.inner_result.iterations)]),
                "stage_head_spec": _jsonable(assembler_stage_payload),
                "basis_context": basis_context_payload,
                "object_gradient_pool": object_gradient_pool_payload,
                "mandatory_realization_closure_report": _jsonable(
                    dict(assembler_result.mandatory_realization_closure_report)
                ),
                "same_source_over_realization_report": _jsonable(
                    dict(assembler_result.same_source_over_realization_report)
                ),
            },
            "orthogonal_search_objective": _jsonable(assembler_result.outer_objective),
            "fold_report": _jsonable(assembler_result.fold_report),
            "structure_head": stage_head_summary.get("structure_head"),
            "prediction_head": stage_head_summary.get("prediction_head"),
            "search_input_space": stage_head_summary.get("search_input_space"),
            "pool_expansion_unit": stage_head_summary.get("pool_expansion_unit"),
            "gradient_guidance_mode": stage_head_summary.get("gradient_guidance_mode"),
            "basis_binding_mode": stage_head_summary.get("basis_binding_mode"),
            "escape_policy": stage_head_summary.get("escape_policy"),
            "stage_head_protocols": stage_head_protocols,
            "basis_context": basis_context_payload,
            "basis_object_gradient_pool": object_gradient_pool_payload,
            "gate_piecewise": {
                "gate_feature_names": [str(value) for value in gate_feature_names],
                "gate_indices": [int(value) for value in gate_indices],
                "gate_basis_terms": _jsonable(gate_basis_rows),
            },
            "symbolic": {
                "selected_basis": _jsonable(basis_rows),
                "basis_semantics": _jsonable(basis_semantics),
                "basis_overlap_report": _jsonable(basis_overlap_report),
                "residual_complementarity_report": _jsonable(residual_report),
                "semantic_dedup_report": _jsonable(semantic_report),
                "interference_feature_report": _jsonable(interference_report),
                "environment_invariance_audit": _jsonable(dict(assembler_result.environment_invariance_audit)),
                "periodic_equivalence_report": _jsonable(dict(assembler_result.periodic_equivalence_report)),
                "regional_correction_report": _jsonable(dict(assembler_result.regional_correction_report)),
                "mandatory_realization_closure_report": _jsonable(
                    dict(assembler_result.mandatory_realization_closure_report)
                ),
                "assembler_budget": _jsonable(assembler_budget),
                "orthogonal_outer_basis_genome": _jsonable(tuple(dict(term) for term in tuple(outer_basis_genome))),
                "inner_symbolic_search": {
                    "protocol": "budgeted_symbolic_assembler",
                    "basis_feature_names": [str(value) for value in tuple(assembler_result.basis_feature_names)],
                    "search_config": _jsonable(assembler_result.search_config),
                    "final_metrics": _jsonable(dict(assembler_result.inner_result.final_metrics)),
                    "stage_head_spec": _jsonable(assembler_stage_payload),
                    "basis_context": basis_context_payload,
                    "object_gradient_pool": object_gradient_pool_payload,
                    "mandatory_realization_closure_report": _jsonable(
                        dict(assembler_result.mandatory_realization_closure_report)
                    ),
                },
                "orthogonal_search_objective": _jsonable(assembler_result.outer_objective),
                "fold_report": _jsonable(assembler_result.fold_report),
                "structure_head": stage_head_summary.get("structure_head"),
                "prediction_head": stage_head_summary.get("prediction_head"),
                "search_input_space": stage_head_summary.get("search_input_space"),
                "pool_expansion_unit": stage_head_summary.get("pool_expansion_unit"),
                "gradient_guidance_mode": stage_head_summary.get("gradient_guidance_mode"),
                "basis_binding_mode": stage_head_summary.get("basis_binding_mode"),
                "escape_policy": stage_head_summary.get("escape_policy"),
                "stage_head_protocols": stage_head_protocols,
                "basis_context": basis_context_payload,
                "basis_object_gradient_pool": object_gradient_pool_payload,
            },
        }
        if symbolic_family_payload is not None:
            surface_metadata["symbolic_family"] = _jsonable(symbolic_family_payload)
            surface_metadata["symbolic"]["symbolic_family"] = _jsonable(symbolic_family_payload)
        symbolic_structure_surface = build_symbolic_structure_surface_payload(
            metadata=surface_metadata,
            final_expression=expression_payload,
            global_basis=basis_rows,
            local_basis_by_regime=None,
            gate_basis=gate_basis_rows,
            piecewise_enabled=False,
            basis_scope="global",
            basis_source="metadata.selected_basis",
            assembler_source="metadata.assembler_budget",
            composition_targets=("expression",),
            gate_feature_names=gate_feature_names,
            gate_indices=gate_indices,
        )
        evaluation_rows.append(
            {
                "pool_indices": pool_indices,
                "outer_basis_genome": tuple(dict(term) for term in tuple(outer_basis_genome)),
                "basis_space_genome": tuple(dict(term) for term in tuple(assembler_result.basis_space_genome)),
                "genome": tuple(dict(term) for term in tuple(assembler_result.assembled_genome)),
                "selected_l2": float(resolved_cfg.assembler_ridge_l2),
                "sort_key": _budgeted_assembler_sort_key(
                    inner_result=assembler_result.inner_result,
                    outer_objective=assembler_result.outer_objective,
                    orthogonality_metrics=orthogonality_metrics,
                    cfg=resolved_cfg,
                ),
                "orthogonality_metrics": dict(orthogonality_metrics),
                "basis_rows": list(basis_rows),
                "gate_basis_rows": list(gate_basis_rows),
                "basis_semantics": dict(basis_semantics),
                "basis_overlap_report": dict(basis_overlap_report),
                "residual_complementarity_report": dict(residual_report),
                "semantic_dedup_report": dict(semantic_report),
                "interference_feature_report": dict(interference_report),
                "assembler_budget": dict(assembler_budget),
                "expression_payload": dict(expression_payload),
                "final_fit": dict(final_fit),
                "fold_report": dict(assembler_result.fold_report),
                "inner_result": assembler_result.inner_result.to_dict(),
                "inner_final_metrics": dict(assembler_result.inner_result.final_metrics),
                "outer_objective": dict(assembler_result.outer_objective),
                "assembler_search_config": dict(assembler_result.search_config),
                "stage_head_protocols": dict(assembler_result.stage_head_protocols),
                "basis_context": dict(assembler_result.basis_context),
                "basis_object_gradient_pool": dict(assembler_result.object_gradient_pool),
                "environment_invariance_audit": dict(assembler_result.environment_invariance_audit),
                "periodic_equivalence_report": dict(assembler_result.periodic_equivalence_report),
                "regional_correction_report": dict(assembler_result.regional_correction_report),
                "mandatory_realization_closure_report": dict(
                    assembler_result.mandatory_realization_closure_report
                ),
                "same_source_over_realization_report": dict(
                    assembler_result.same_source_over_realization_report
                ),
                "stage_head_summary": dict(stage_head_summary),
                "symbolic_structure_surface": dict(symbolic_structure_surface),
                "group_payload": dict(group_payload),
            }
        )

    evaluation_rows.sort(key=lambda item: item["sort_key"])
    selected = evaluation_rows[0]
    pred_train = np.asarray(selected["final_fit"].get("pred_train"), dtype=float)
    if pred_train.ndim == 1:
        pred_train = pred_train.reshape(-1, 1)
    residual = np.asarray(yy, dtype=float) - np.asarray(pred_train, dtype=float)
    residual_std = np.std(residual, axis=0, ddof=1) + 1e-8
    search_summary = {
        "protocol": "orthogonal_structure_search_with_budgeted_symbolic_assembler",
        "candidate_pool_size": int(len(candidate_pool)),
        "screened_candidate_count": int(len(screened)),
        "screened_gate_candidate_count": int(sum(1 for row in tuple(screened) if bool(row.uses_piecewise_gate))),
        "generated_group_count": int(len(group_payloads)),
        "piecewise_gate_seed_count": int(len(gate_specs)),
        "consensus_prior_summary": _jsonable(dict(consensus_prior_model.get("summary", {}))),
        "selection_mode": str(resolved_cfg.selection_mode),
        "gate_candidate_screen_reserve": int(resolved_cfg.gate_candidate_screen_reserve),
        "periodic_candidate_screen_reserve": int(resolved_cfg.periodic_candidate_screen_reserve),
        "heterogeneous_exposure_candidate_screen_reserve": int(
            resolved_cfg.heterogeneous_exposure_candidate_screen_reserve
        ),
        "require_gate_candidate_in_group": bool(resolved_cfg.require_gate_candidate_in_group),
        "min_gate_basis_terms": int(_required_gate_basis_terms(resolved_cfg)),
        "regime_penetration_mode": str(resolved_cfg.regime_penetration_mode),
        "heterogeneous_exposure_mode": str(resolved_cfg.heterogeneous_exposure_mode),
        "same_source_over_realization_mode": str(resolved_cfg.same_source_over_realization_mode),
        "periodic_equivalence_disambiguation_mode": str(resolved_cfg.periodic_equivalence_disambiguation_mode),
        "phase_spectrum_audit_mode": str(resolved_cfg.phase_spectrum_audit_mode),
        "periodic_family_prior_mode": str(resolved_cfg.periodic_family_prior_mode),
        "regional_correction_promotion_mode": str(resolved_cfg.regional_correction_promotion_mode),
        "regional_correction_feature_scope": str(resolved_cfg.regional_correction_feature_scope),
        "regional_correction_topk": int(resolved_cfg.regional_correction_topk),
        "periodic_feature_names": [
            str(value) for value in tuple(periodic_context.get("periodic_feature_names", ()))
        ],
        "mechanistic_feature_groups": [
            [str(name) for name in tuple(group)]
            for group in tuple(_configured_mechanistic_feature_groups(cfg=resolved_cfg, feature_names=bundle.feature_names))
        ],
        "mechanistic_screen_bonus": float(resolved_cfg.mechanistic_screen_bonus),
        "mechanistic_group_bonus": float(resolved_cfg.mechanistic_group_bonus),
        "search_random_seed": int(resolved_cfg.random_seed),
        "greedy_choice_topk": int(resolved_cfg.greedy_choice_topk),
        "random_group_trials": int(resolved_cfg.random_group_trials),
        "outer_search_beam_width": int(resolved_cfg.outer_search_beam_width),
        "outer_search_branching_factor": int(resolved_cfg.outer_search_branching_factor),
        "outer_search_max_expansions": int(resolved_cfg.outer_search_max_expansions),
        "lock_seed_basis": bool(resolved_cfg.lock_seed_basis),
        "assembler_basis_binding_mode": str(resolved_cfg.assembler_basis_binding_mode),
        "assembler_escape_policy": str(resolved_cfg.assembler_escape_policy),
        "assembler_escape_feature_names": [str(value) for value in tuple(resolved_cfg.assembler_escape_feature_names)],
        "selected_l2": float(selected["selected_l2"]),
        "seed_basis_match_count": int(len(_match_seed_rows(screened=screened, seed_genome=seed_genome))),
        "selected_group": {
            "basis_count": int(len(tuple(selected["basis_rows"]))),
            "group_proposal_score": float(dict(selected["group_payload"]).get("group_score", 0.0)),
            "pair_abs_corr_mean": float(dict(selected["orthogonality_metrics"]).get("pair_abs_corr_mean", 0.0)),
            "orthogonality_score": float(dict(selected["orthogonality_metrics"]).get("orthogonality_score", 0.0)),
            "residual_gain_mean": float(dict(selected["residual_complementarity_report"]).get("mean_marginal_r2_gain", 0.0)),
            "semantic_unique_ratio": float(dict(selected["semantic_dedup_report"]).get("semantic_unique_ratio", 0.0)),
            "trivial_nonlinearity_penalty_mean": float(
                dict(selected["interference_feature_report"]).get("trivial_nonlinearity_penalty_mean", 0.0)
            ),
            "cross_explanatory_suspicious_pairs": int(
                dict(selected["interference_feature_report"]).get("suspicious_pair_count", 0)
            ),
            "periodic_equivalence_score": float(
                dict(selected["periodic_equivalence_report"]).get("overall_periodic_disambiguation_score", 0.0)
            ),
            "periodic_equivalence_penalty": float(
                dict(selected["periodic_equivalence_report"]).get("local_equivalence_penalty_mean", 0.0)
            ),
            "regional_correction_score": float(
                dict(selected["regional_correction_report"]).get("regional_correction_score", 0.0)
            ),
            "inner_fit_score": float(dict(selected["outer_objective"]).get("inner_fit_score", 0.0)),
            "outer_objective_score": float(dict(selected["outer_objective"]).get("outer_score", 0.0)),
            "same_source_realization_penalty": float(
                dict(selected["outer_objective"]).get("same_source_realization_penalty", 0.0)
            ),
            "environment_invariance_score": float(
                dict(selected["environment_invariance_audit"]).get("overall_invariance_score", 0.0)
            ),
            "mechanism_summary": _jsonable(dict(dict(selected["group_payload"]).get("mechanism_summary", {}) or {})),
            "mean_mechanistic_prior": float(
                dict(dict(selected["group_payload"]).get("screen_summary", {}) or {}).get("mean_mechanistic_prior", 0.0)
            ),
            "inner_final_metrics": _jsonable(dict(selected["inner_final_metrics"])),
        },
    }
    search_metadata = {
        "selected_basis": _jsonable(selected["basis_rows"]),
        "basis_semantics": _jsonable(selected["basis_semantics"]),
        "basis_overlap_report": _jsonable(selected["basis_overlap_report"]),
        "residual_complementarity_report": _jsonable(selected["residual_complementarity_report"]),
        "semantic_dedup_report": _jsonable(selected["semantic_dedup_report"]),
        "interference_feature_report": _jsonable(selected["interference_feature_report"]),
        "environment_invariance_audit": _jsonable(selected["environment_invariance_audit"]),
        "periodic_equivalence_report": _jsonable(selected["periodic_equivalence_report"]),
        "regional_correction_report": _jsonable(selected["regional_correction_report"]),
        "mandatory_realization_closure_report": _jsonable(selected["mandatory_realization_closure_report"]),
        "same_source_over_realization_report": _jsonable(selected["same_source_over_realization_report"]),
        "assembler_budget": _jsonable(selected["assembler_budget"]),
        "orthogonality_metrics": _jsonable(selected["orthogonality_metrics"]),
        "orthogonal_outer_basis_genome": _jsonable(selected["outer_basis_genome"]),
        "inner_symbolic_search": {
            "protocol": "budgeted_symbolic_assembler",
            "search_config": _jsonable(selected["assembler_search_config"]),
            "basis_space_genome": _jsonable(selected["basis_space_genome"]),
            "final_metrics": _jsonable(selected["inner_final_metrics"]),
            "search_result": _jsonable(selected["inner_result"]),
            "stage_head_spec": _jsonable(dict(dict(selected["stage_head_protocols"]).get("assembler", {}))),
            "basis_context": _jsonable(selected["basis_context"]),
            "object_gradient_pool": _jsonable(selected["basis_object_gradient_pool"]),
            "mandatory_realization_closure_report": _jsonable(selected["mandatory_realization_closure_report"]),
        },
        "orthogonal_search_objective": _jsonable(selected["outer_objective"]),
        "fold_report": _jsonable(selected["fold_report"]),
        "validation_summary": _jsonable(selected["fold_report"]),
        "structure_head": dict(selected["stage_head_summary"]).get("structure_head"),
        "prediction_head": dict(selected["stage_head_summary"]).get("prediction_head"),
        "search_input_space": dict(selected["stage_head_summary"]).get("search_input_space"),
        "pool_expansion_unit": dict(selected["stage_head_summary"]).get("pool_expansion_unit"),
        "gradient_guidance_mode": dict(selected["stage_head_summary"]).get("gradient_guidance_mode"),
        "basis_binding_mode": dict(selected["stage_head_summary"]).get("basis_binding_mode"),
        "escape_policy": dict(selected["stage_head_summary"]).get("escape_policy"),
        "stage_head_protocols": _jsonable(selected["stage_head_protocols"]),
        "basis_context": _jsonable(selected["basis_context"]),
        "basis_object_gradient_pool": _jsonable(selected["basis_object_gradient_pool"]),
        "mechanism_constraints": {
            "gate_candidate_screen_reserve": int(resolved_cfg.gate_candidate_screen_reserve),
            "periodic_candidate_screen_reserve": int(resolved_cfg.periodic_candidate_screen_reserve),
            "required_gate_basis_terms": int(_required_gate_basis_terms(resolved_cfg)),
            "periodic_equivalence_disambiguation_mode": str(
                resolved_cfg.periodic_equivalence_disambiguation_mode
            ),
            "periodic_family_prior_mode": str(resolved_cfg.periodic_family_prior_mode),
            "periodic_feature_names": [
                str(value) for value in tuple(periodic_context.get("periodic_feature_names", ()))
            ],
            "regional_correction_protocol": str(resolved_cfg.regional_correction_protocol),
            "regional_correction_promotion_mode": str(resolved_cfg.regional_correction_promotion_mode),
            "regional_correction_feature_scope": str(resolved_cfg.regional_correction_feature_scope),
            "regional_correction_topk": int(resolved_cfg.regional_correction_topk),
            "regional_correction_min_r2_gain": float(resolved_cfg.regional_correction_min_r2_gain),
            "mechanistic_feature_groups": [
                [str(name) for name in tuple(group)]
                for group in tuple(_configured_mechanistic_feature_groups(cfg=resolved_cfg, feature_names=bundle.feature_names))
            ],
            "mechanistic_screen_bonus": float(resolved_cfg.mechanistic_screen_bonus),
            "mechanistic_group_bonus": float(resolved_cfg.mechanistic_group_bonus),
            "assembler_basis_binding_mode": str(resolved_cfg.assembler_basis_binding_mode),
            "assembler_escape_policy": str(resolved_cfg.assembler_escape_policy),
            "assembler_escape_feature_names": [str(value) for value in tuple(resolved_cfg.assembler_escape_feature_names)],
        },
        "gate_piecewise": {
            "gate_feature_names": list(_configured_gate_feature_names(cfg=resolved_cfg, feature_names=bundle.feature_names)),
            "gate_indices": [
                int(index)
                for index, name in enumerate(tuple(str(value) for value in tuple(bundle.feature_names)))
                if str(name) in set(_configured_gate_feature_names(cfg=resolved_cfg, feature_names=bundle.feature_names))
            ],
            "gate_basis_terms": _jsonable(selected["gate_basis_rows"]),
        },
        "search": _jsonable(search_summary),
        "symbolic": {
            "selected_basis": _jsonable(selected["basis_rows"]),
            "basis_semantics": _jsonable(selected["basis_semantics"]),
            "basis_overlap_report": _jsonable(selected["basis_overlap_report"]),
            "residual_complementarity_report": _jsonable(selected["residual_complementarity_report"]),
            "semantic_dedup_report": _jsonable(selected["semantic_dedup_report"]),
            "interference_feature_report": _jsonable(selected["interference_feature_report"]),
            "environment_invariance_audit": _jsonable(selected["environment_invariance_audit"]),
            "periodic_equivalence_report": _jsonable(selected["periodic_equivalence_report"]),
            "regional_correction_report": _jsonable(selected["regional_correction_report"]),
            "mandatory_realization_closure_report": _jsonable(selected["mandatory_realization_closure_report"]),
            "assembler_budget": _jsonable(selected["assembler_budget"]),
            "orthogonality_metrics": _jsonable(selected["orthogonality_metrics"]),
            "orthogonal_outer_basis_genome": _jsonable(selected["outer_basis_genome"]),
            "inner_symbolic_search": {
                "protocol": "budgeted_symbolic_assembler",
                "search_config": _jsonable(selected["assembler_search_config"]),
                "basis_space_genome": _jsonable(selected["basis_space_genome"]),
                "final_metrics": _jsonable(selected["inner_final_metrics"]),
                "stage_head_spec": _jsonable(dict(dict(selected["stage_head_protocols"]).get("assembler", {}))),
                "basis_context": _jsonable(selected["basis_context"]),
                "object_gradient_pool": _jsonable(selected["basis_object_gradient_pool"]),
                "mandatory_realization_closure_report": _jsonable(selected["mandatory_realization_closure_report"]),
            },
            "orthogonal_search_objective": _jsonable(selected["outer_objective"]),
            "fold_report": _jsonable(selected["fold_report"]),
            "structure_head": dict(selected["stage_head_summary"]).get("structure_head"),
            "prediction_head": dict(selected["stage_head_summary"]).get("prediction_head"),
            "search_input_space": dict(selected["stage_head_summary"]).get("search_input_space"),
            "pool_expansion_unit": dict(selected["stage_head_summary"]).get("pool_expansion_unit"),
            "gradient_guidance_mode": dict(selected["stage_head_summary"]).get("gradient_guidance_mode"),
            "basis_binding_mode": dict(selected["stage_head_summary"]).get("basis_binding_mode"),
            "escape_policy": dict(selected["stage_head_summary"]).get("escape_policy"),
            "stage_head_protocols": _jsonable(selected["stage_head_protocols"]),
            "basis_context": _jsonable(selected["basis_context"]),
            "basis_object_gradient_pool": _jsonable(selected["basis_object_gradient_pool"]),
            "structure_engine": {
                "structure_mode": "orthogonal_basis_search",
                "search_driver": "orthogonal_basis_set_search",
                "screening_protocol": "target_corr+residual_gain+semantic_novelty+consensus_prior",
                "outer_search_protocol": "beam_basis_set_structure_search",
                "structure_head": dict(selected["stage_head_summary"]).get("structure_head"),
                "search_input_space": dict(selected["stage_head_summary"]).get("search_input_space"),
                "pool_expansion_unit": dict(selected["stage_head_summary"]).get("pool_expansion_unit"),
                "gradient_guidance_mode": dict(selected["stage_head_summary"]).get("gradient_guidance_mode"),
                "dynamic_pool_enabled": True,
                "metadata": {
                    "supports_piecewise_basis": bool(
                        _configured_gate_feature_names(cfg=resolved_cfg, feature_names=bundle.feature_names)
                    ),
                    "periodic_feature_names": [
                        str(value) for value in tuple(periodic_context.get("periodic_feature_names", ()))
                    ],
                    "screening_protocol": "target_corr+residual_gain+semantic_novelty+consensus_prior",
                    "outer_search_protocol": "beam_basis_set_structure_search",
                    "structure_head": dict(selected["stage_head_summary"]).get("structure_head"),
                    "search_input_space": dict(selected["stage_head_summary"]).get("search_input_space"),
                    "pool_expansion_unit": dict(selected["stage_head_summary"]).get("pool_expansion_unit"),
                    "gradient_guidance_mode": dict(selected["stage_head_summary"]).get("gradient_guidance_mode"),
                    "gate_candidate_screen_reserve": int(resolved_cfg.gate_candidate_screen_reserve),
                    "periodic_candidate_screen_reserve": int(resolved_cfg.periodic_candidate_screen_reserve),
                    "required_gate_basis_terms": int(_required_gate_basis_terms(resolved_cfg)),
                    "required_periodic_basis_terms": int(
                        _required_periodic_basis_terms(cfg=resolved_cfg, periodic_context=periodic_context)
                    ),
                    "require_gate_candidate_in_group": bool(resolved_cfg.require_gate_candidate_in_group),
                    "require_periodic_candidate_in_group": bool(resolved_cfg.require_periodic_candidate_in_group),
                    "outer_search_unit": str(resolved_cfg.outer_search_unit),
                    "representative_selection_rule": str(resolved_cfg.representative_selection_rule),
                    "mechanistic_feature_groups": [
                        [str(name) for name in tuple(group)]
                        for group in tuple(_configured_mechanistic_feature_groups(cfg=resolved_cfg, feature_names=bundle.feature_names))
                    ],
                    "mechanistic_screen_bonus": float(resolved_cfg.mechanistic_screen_bonus),
                    "mechanistic_group_bonus": float(resolved_cfg.mechanistic_group_bonus),
                    "assembler_basis_binding_mode": str(resolved_cfg.assembler_basis_binding_mode),
                    "assembler_escape_policy": str(resolved_cfg.assembler_escape_policy),
                    "assembler_escape_feature_names": [str(value) for value in tuple(resolved_cfg.assembler_escape_feature_names)],
                    "regional_correction_search_mode": str(resolved_cfg.regional_correction_search_mode),
                    "regional_local_search_beam_width": int(resolved_cfg.regional_local_search_beam_width),
                    "regional_local_search_branching_factor": int(
                        resolved_cfg.regional_local_search_branching_factor
                    ),
                    "regional_local_search_max_expansions": int(resolved_cfg.regional_local_search_max_expansions),
                },
            },
        },
        "symbolic_structure_surface": _jsonable(selected["symbolic_structure_surface"]),
    }
    basis_context_meta = dict(dict(selected.get("basis_context", {}) or {}).get("metadata", {}) or {})
    periodic_equivalence_payload = {
        "protocol": str(resolved_cfg.periodic_equivalence_protocol),
        "mode": str(resolved_cfg.periodic_equivalence_disambiguation_mode),
        "parent_protocol": str(resolved_cfg.equivalence_expression_protocol),
        "parent_mode_slot": "periodic_mode",
        "canonical_mode_name": "periodic_mode",
        "phase_spectrum_audit_mode": str(resolved_cfg.phase_spectrum_audit_mode),
        "periodic_family_prior_mode": str(resolved_cfg.periodic_family_prior_mode),
        "periodic_family_prior_weight": float(resolved_cfg.periodic_family_prior_weight),
        "periodic_candidate_screen_reserve": int(resolved_cfg.periodic_candidate_screen_reserve),
        "periodic_feature_names": [
            str(value) for value in tuple(periodic_context.get("periodic_feature_names", ()))
        ],
        "enabled_steps": [
            *(
                ["screen_periodic_family_prior"]
                if _periodic_family_prior_enabled(resolved_cfg)
                else []
            ),
            *(
                ["outer_periodic_disambiguation_penalty"]
                if _periodic_disambiguation_enabled(resolved_cfg)
                else []
            ),
            *(
                ["phase_spectrum_audit"]
                if _phase_spectrum_audit_enabled(resolved_cfg)
                else []
            ),
        ],
        "periodic_equivalence_report": _jsonable(selected["periodic_equivalence_report"]),
        "notes": (
            "Current implementation promotes periodic-family candidates during screening, penalizes "
            "local non-periodic surrogates on configured periodic features through a center-edge "
            "holdout audit, and records phase-spectrum style diagnostics for the selected basis set."
        ),
    }
    realization_prior_enabled = _realization_prior_injection_enabled(resolved_cfg)
    mandatory_realization_closure_enabled = _mandatory_realization_closure_enabled(resolved_cfg)
    same_source_over_realization_enabled = _same_source_over_realization_enabled(resolved_cfg)
    periodic_realization_competition_enabled = (
        _periodic_realization_competition_enabled(resolved_cfg)
        and bool(periodic_context.get("periodic_feature_names"))
    )
    chart_canonicalization_enabled = _chart_canonicalization_enabled(resolved_cfg)
    inner_chart_flip_enabled = _inner_chart_flip_compensation_enabled(resolved_cfg)
    regime_penetration_enabled = _regime_penetration_enabled(resolved_cfg)
    heterogeneous_exposure_enabled = _heterogeneous_exposure_enabled(resolved_cfg)
    causal_hierarchy_reuse_enabled = _causal_hierarchy_reuse_isolation_enabled(resolved_cfg)
    realization_prior_payload = {
        "protocol": str(resolved_cfg.realization_prior_injection_protocol),
        "mode": str(resolved_cfg.realization_prior_injection_mode),
        "parent_protocol": str(resolved_cfg.equivalence_expression_protocol),
        "parent_mode_slot": "realization_prior_injection_mode",
        "canonical_mode_name": "realization_prior_injection_mode",
        "realization_object_count": int(basis_context_meta.get("realization_object_count", 0) or 0),
        "realization_object_keys": [
            str(value) for value in tuple(basis_context_meta.get("realization_object_keys", ())) if str(value).strip()
        ],
        "notes": (
            "Current implementation collapses the outer basis into source objects for the inner stage, "
            "then re-injects evidence-backed realization heads such as exp(source), exp(-source), or square(source) "
            "as explicit basis-object competitors."
        ),
    }
    mandatory_realization_closure_payload = {
        "protocol": str(resolved_cfg.mandatory_realization_closure_protocol),
        "mode": str(resolved_cfg.mandatory_realization_closure_mode),
        "parent_protocol": str(resolved_cfg.equivalence_expression_protocol),
        "parent_mode_slot": "mandatory_realization_closure_mode",
        "canonical_mode_name": "mandatory_realization_closure_mode",
        "report": _jsonable(selected["mandatory_realization_closure_report"]),
        "notes": (
            "Current implementation does not force a realization head to win, but it does force every "
            "evidence-backed closure candidate to be explicitly scored against the inner search result."
        ),
    }
    periodic_realization_competition_payload = {
        "protocol": str(resolved_cfg.periodic_realization_competition_protocol),
        "mode": str(resolved_cfg.periodic_realization_competition_mode),
        "parent_protocol": str(resolved_cfg.equivalence_expression_protocol),
        "parent_mode_slot": "periodic_realization_competition_mode",
        "canonical_mode_name": "periodic_realization_competition_mode",
        "periodic_feature_names": [
            str(value) for value in tuple(periodic_context.get("periodic_feature_names", ())) if str(value).strip()
        ],
        "realization_object_count": int(basis_context_meta.get("realization_object_count", 0) or 0),
        "notes": (
            "Current implementation keeps the canonical periodic source object in the basis-conditioned stage and "
            "forces sin/cos realization heads to enter the same inner symbolic competition whenever periodic hints exist."
        ),
    }
    chart_canonicalization_payload = {
        "protocol": str(resolved_cfg.chart_canonicalization_protocol),
        "mode": str(resolved_cfg.chart_canonicalization_mode),
        "parent_protocol": str(resolved_cfg.equivalence_expression_protocol),
        "parent_mode_slot": "chart_canonicalization_mode",
        "canonical_mode_name": "chart_canonicalization_mode",
        "chart_orthodoxy_scoring_protocol": str(resolved_cfg.chart_orthodoxy_scoring_protocol),
        "chart_orthodoxy_scoring_mode": str(resolved_cfg.chart_orthodoxy_scoring_mode),
        "chart_object_count": int(basis_context_meta.get("chart_object_count", 0) or 0),
        "notes": (
            "Current implementation keeps basis seats at source-object level, then chooses a working chart "
            "for the locked basis object through canonical-identity bias plus numerical-stability guards."
        ),
    }
    same_source_over_realization_payload = {
        "protocol": str(resolved_cfg.same_source_over_realization_protocol),
        "mode": str(resolved_cfg.same_source_over_realization_mode),
        "parent_protocol": str(resolved_cfg.equivalence_expression_protocol),
        "parent_mode_slot": "same_source_over_realization_mode",
        "canonical_mode_name": "same_source_over_realization_mode",
        "same_source_realization_budget": int(resolved_cfg.same_source_realization_budget),
        "report": _jsonable(selected["same_source_over_realization_report"]),
        "notes": (
            "Current implementation limits same-source realization competitors in basis-object space and "
            "penalizes final inner assemblies that still let one source object occupy multiple basis seats."
        ),
    }
    inner_chart_flip_payload = {
        "protocol": str(resolved_cfg.inner_chart_flip_compensation_protocol),
        "mode": str(resolved_cfg.inner_chart_flip_compensation_mode),
        "parent_protocol": str(resolved_cfg.equivalence_expression_protocol),
        "parent_mode_slot": "inner_chart_flip_compensation_mode",
        "canonical_mode_name": "inner_chart_flip_compensation_mode",
        "chart_object_count": int(basis_context_meta.get("chart_object_count", 0) or 0),
        "notes": (
            "Current implementation re-opens reciprocal chart competitors for the same source object inside "
            "the basis-conditioned inner search, so a wrong outer working chart has a low-risk repair lane."
        ),
    }
    regional_correction_payload = {
        "protocol": str(resolved_cfg.regional_correction_protocol),
        "parent_protocol": str(resolved_cfg.interference_feature_protocol),
        "parent_mode_slot": "regional_correction_mode",
        "canonical_mode_name": "regional_correction_mode",
        "semantic_slot_name": "regional_residual_correction",
        "residual_regime_identification_mode": str(resolved_cfg.residual_regime_identification_mode),
        "regional_correction_basis_mode": str(resolved_cfg.regional_correction_basis_mode),
        "regional_correction_promotion_mode": str(resolved_cfg.regional_correction_promotion_mode),
        "regional_correction_feature_scope": str(resolved_cfg.regional_correction_feature_scope),
        "regional_correction_topk": int(resolved_cfg.regional_correction_topk),
        "regional_correction_min_r2_gain": float(resolved_cfg.regional_correction_min_r2_gain),
        "enabled_steps": [
            *(
                ["residual_regime_identification", "reopened_local_object_search"]
                if _regional_correction_enabled(resolved_cfg)
                else []
            ),
        ],
        "regional_correction_report": _jsonable(selected["regional_correction_report"]),
        "notes": (
            "Current implementation reopens a small residual-local gate search around the locked outer "
            "basis residual, merges those candidates with screened gate channels, and sends the best "
            "regional correction objects into the basis-conditioned inner symbolic search."
        ),
    }
    periodic_mode_enabled = (
        bool(periodic_context.get("periodic_feature_names"))
        or _periodic_family_prior_enabled(resolved_cfg)
        or _periodic_disambiguation_enabled(resolved_cfg)
        or _phase_spectrum_audit_enabled(resolved_cfg)
    )
    proxy_policy_mode = str(resolved_cfg.proxy_group_policy or "off").strip().lower()
    native_proxy_check_mode_enabled = _native_proxy_check_enabled(resolved_cfg)
    proxy_trunk_disqualification_mode_enabled = _proxy_trunk_disqualification_enabled(resolved_cfg)
    parasitic_rejection_mode_enabled = _parasitic_rejection_enabled(resolved_cfg)
    proxy_suppression_mode_enabled = (
        _cross_explanatory_rejection_enabled(resolved_cfg)
        or proxy_policy_mode not in {"", "off", "none", "disabled"}
    )
    trivial_nonlinearity_mode_enabled = _trivial_nonlinearity_penalty_enabled(resolved_cfg)
    regional_correction_mode_enabled = _regional_correction_enabled(resolved_cfg)
    environment_invariance_mode_enabled = (
        str(resolved_cfg.environment_invariance_audit_mode or "off").strip().lower()
        not in {"", "off", "none", "disabled"}
    )
    periodic_mode_payload = {
        "canonical_mode_name": "periodic_mode",
        "leaf_protocol_name": str(resolved_cfg.periodic_equivalence_protocol),
        "artifact_slot": "periodic_equivalence_disambiguation",
        "status": "enabled" if periodic_mode_enabled else "configured_off",
        "mode": str(resolved_cfg.periodic_equivalence_disambiguation_mode),
        "phase_spectrum_audit_mode": str(resolved_cfg.phase_spectrum_audit_mode),
        "periodic_family_prior_mode": str(resolved_cfg.periodic_family_prior_mode),
        "periodic_feature_names": [
            str(value) for value in tuple(periodic_context.get("periodic_feature_names", ()))
        ],
    }
    proxy_suppression_payload = {
        "canonical_mode_name": "proxy_suppression_mode",
        "artifact_slot": "interference_feature_handling",
        "status": "enabled" if proxy_suppression_mode_enabled else "configured_off",
        "cross_explanatory_rejection_mode": str(resolved_cfg.cross_explanatory_rejection_mode),
        "proxy_group_policy": str(resolved_cfg.proxy_group_policy),
        "report": _jsonable(selected["interference_feature_report"]),
    }
    native_proxy_check_payload = {
        "canonical_mode_name": "native_proxy_check_mode",
        "leaf_protocol_name": str(resolved_cfg.native_proxy_check_protocol),
        "artifact_slot": "interference_feature_handling",
        "status": "enabled" if native_proxy_check_mode_enabled else "configured_off",
        "mode": str(resolved_cfg.native_proxy_check_mode),
        "proxy_group_policy": str(resolved_cfg.proxy_group_policy),
        "notes": (
            "Current implementation raises native-trunk and identity-source representatives to the front "
            "when proxy-group conflicts are resolved during screen-level representative retention."
        ),
    }
    proxy_trunk_disqualification_payload = {
        "canonical_mode_name": "proxy_trunk_disqualification_mode",
        "leaf_protocol_name": str(resolved_cfg.proxy_trunk_disqualification_protocol),
        "artifact_slot": "interference_feature_handling",
        "status": "enabled" if proxy_trunk_disqualification_mode_enabled else "configured_off",
        "mode": str(resolved_cfg.proxy_trunk_disqualification_mode),
        "proxy_group_policy": str(resolved_cfg.proxy_group_policy),
        "notes": (
            "Current implementation upgrades proxy-group conflict handling from ranking preference to hard "
            "eligibility: when a native identity trunk exists in a proxy group, wrapped or branch variants "
            "cannot represent that group at screen admission."
        ),
    }
    trivial_nonlinearity_payload = {
        "canonical_mode_name": "trivial_nonlinearity_rejection_mode",
        "artifact_slot": "interference_feature_handling",
        "status": "enabled" if trivial_nonlinearity_mode_enabled else "configured_off",
        "trivial_nonlinearity_penalty_mode": str(resolved_cfg.trivial_nonlinearity_penalty_mode),
        "source_overlap_penalty_mode": str(resolved_cfg.source_overlap_penalty_mode),
        "report": _jsonable(selected["interference_feature_report"]),
    }
    parasitic_rejection_payload = {
        "canonical_mode_name": "parasitic_rejection_mode",
        "leaf_protocol_name": str(resolved_cfg.parasitic_rejection_protocol),
        "artifact_slot": "interference_feature_handling",
        "status": "enabled" if parasitic_rejection_mode_enabled else "configured_off",
        "mode": str(resolved_cfg.parasitic_rejection_mode),
        "notes": (
            "Current implementation blocks structural gate/regional branches from entering outer basis "
            "competition before their parent trunk source is present whenever a parent trunk candidate exists."
        ),
    }
    causal_hierarchy_reuse_payload = {
        "canonical_mode_name": "causal_hierarchy_reuse_isolation_mode",
        "leaf_protocol_name": str(resolved_cfg.causal_hierarchy_reuse_isolation_protocol),
        "artifact_slot": "interference_feature_handling",
        "status": "enabled" if causal_hierarchy_reuse_enabled else "configured_off",
        "mode": str(resolved_cfg.causal_hierarchy_reuse_isolation_mode),
        "max_feature_reuse": int(resolved_cfg.max_feature_reuse),
        "notes": (
            "Current implementation isolates correction-branch reuse from trunk reuse during outer basis assembly, "
            "so piecewise/gate branches do not consume the same source-feature quota as their parent trunk objects."
        ),
    }
    native_trunk_payload = {
        "protocol": str(resolved_cfg.native_trunk_boundary_protocol),
        "mode": str(resolved_cfg.native_trunk_channel_mode),
        "screen_reserve": int(resolved_cfg.native_trunk_candidate_screen_reserve),
        "require_in_group": bool(resolved_cfg.require_native_trunk_candidate_in_group),
        "min_basis_terms": int(resolved_cfg.min_native_trunk_basis_terms),
        "residual_gain_floor": float(resolved_cfg.native_trunk_residual_gain_floor),
        "interval_gain_floor": float(resolved_cfg.native_trunk_interval_gain_floor),
        "selected_native_trunk_count": int(
            dict(dict(selected.get("group_payload", {})).get("mechanism_summary", {}) or {}).get(
                "native_trunk_term_count",
                0,
            )
        ),
        "notes": (
            "Current implementation applies an outermost-peeling boundary rule to separate native trunk roots "
            "from challenger topology changes, requires native candidates to pass residual-novelty and interval-"
            "stability floors, and fills dedicated trunk seats before open challenger expansion."
        ),
    }
    selected_basis_rows_payload = tuple(selected.get("basis_rows", ()) or ())
    canonical_trunk_payload = {
        "protocol": str(resolved_cfg.canonical_trunk_lane_protocol),
        "mode": str(resolved_cfg.canonical_trunk_lane_mode),
        "screen_reserve": int(resolved_cfg.canonical_trunk_candidate_screen_reserve),
        "require_in_group": bool(resolved_cfg.require_canonical_trunk_candidate_in_group),
        "min_basis_terms": int(resolved_cfg.min_canonical_trunk_basis_terms),
        "tagged_canonical_trunk_count": int(
            sum(
                1
                for row in selected_basis_rows_payload
                if bool(dict(row).get("canonical_trunk_tagged")) or bool(dict(row).get("support_expansion_tagged"))
            )
        ),
        "selected_canonical_trunk_count": int(
            sum(
                1
                for row in selected_basis_rows_payload
                if bool(dict(row).get("canonical_trunk_candidate")) or bool(dict(row).get("support_expansion_candidate"))
            )
        ),
        "notes": (
            "Current implementation gives canonical multi-feature trunk charts a protected outer exposure lane, "
            "so same-support surrogate charts cannot occupy the trunk seat before a native canonical chart has been audited."
        ),
    }
    same_source_surrogate_payload = {
        "protocol": str(resolved_cfg.same_source_surrogate_lane_protocol),
        "mode": str(resolved_cfg.same_source_surrogate_lane_mode),
        "tagged_same_source_surrogate_count": int(
            sum(1 for row in selected_basis_rows_payload if bool(dict(row).get("same_source_surrogate_tagged")))
        ),
        "selected_same_source_surrogate_count": int(
            sum(1 for row in selected_basis_rows_payload if bool(dict(row).get("same_source_surrogate_candidate")))
        ),
        "notes": (
            "Current implementation keeps same-support nonlinear surrogates alive as challengers, but delays their outer "
            "basis admission until a canonical trunk chart from the same support pool has had a chance to enter."
        ),
    }
    regime_penetration_payload = {
        "canonical_mode_name": "regime_penetration_mode",
        "leaf_protocol_name": str(resolved_cfg.regime_penetration_protocol),
        "artifact_slot": "native_trunk_channel",
        "status": "enabled" if regime_penetration_enabled else "configured_off",
        "mode": str(resolved_cfg.regime_penetration_mode),
        "gain_floor": float(resolved_cfg.regime_penetration_gain_floor),
        "notes": (
            "Current implementation audits candidate source objects across feature-quantile regimes, recording "
            "cross-regime minimum gain and sign consistency before native/exposure seat allocation."
        ),
    }
    heterogeneous_exposure_payload = {
        "canonical_mode_name": "heterogeneous_exposure_mode",
        "leaf_protocol_name": str(resolved_cfg.heterogeneous_exposure_protocol),
        "artifact_slot": "native_trunk_channel",
        "status": "enabled" if heterogeneous_exposure_enabled else "configured_off",
        "mode": str(resolved_cfg.heterogeneous_exposure_mode),
        "screen_reserve": int(resolved_cfg.heterogeneous_exposure_candidate_screen_reserve),
        "min_score": float(resolved_cfg.heterogeneous_exposure_min_score),
        "notes": (
            "Current implementation opens a dedicated exposure lane that reserves and seeds regime-stable source "
            "objects before ordinary challenger ranking can suppress them."
        ),
    }
    regional_correction_mode_payload = {
        "canonical_mode_name": "regional_correction_mode",
        "semantic_slot_name": "regional_residual_correction",
        "leaf_protocol_name": str(resolved_cfg.regional_correction_protocol),
        "artifact_slot": "regional_correction_basis",
        "status": "enabled" if regional_correction_mode_enabled else "configured_off",
        "residual_regime_identification_mode": str(resolved_cfg.residual_regime_identification_mode),
        "regional_correction_basis_mode": str(resolved_cfg.regional_correction_basis_mode),
        "regional_correction_promotion_mode": str(resolved_cfg.regional_correction_promotion_mode),
        "regional_correction_feature_scope": str(resolved_cfg.regional_correction_feature_scope),
    }
    equivalence_expression_payload = {
        "protocol": str(resolved_cfg.equivalence_expression_protocol),
        "mode": str(resolved_cfg.equivalence_expression_mode),
        "class_scope": str(resolved_cfg.equivalence_class_scope),
        "equivalence_mode": "family+phase_equivalent",
        "implemented_submodes": [
            "semantic_family_equivalence",
            "phase_equivalent_truth_recovery",
            *(
                ["chart_canonicalization_mode"]
                if chart_canonicalization_enabled
                else []
            ),
            *(
                ["inner_chart_flip_compensation_mode"]
                if inner_chart_flip_enabled
                else []
            ),
            *(
                ["realization_prior_injection_mode"]
                if realization_prior_enabled
                else []
            ),
            *(
                ["mandatory_realization_closure_mode"]
                if mandatory_realization_closure_enabled
                else []
            ),
            *(
                ["same_source_over_realization_mode"]
                if same_source_over_realization_enabled
                else []
            ),
            *(
                ["periodic_realization_competition_mode"]
                if periodic_realization_competition_enabled
                else []
            ),
            *(
                ["periodic_mode"]
                if periodic_mode_enabled
                else []
            ),
        ],
        "child_modes": {
            "chart_canonicalization_mode": _jsonable(chart_canonicalization_payload),
            "inner_chart_flip_compensation_mode": _jsonable(inner_chart_flip_payload),
            "realization_prior_injection_mode": _jsonable(realization_prior_payload),
            "mandatory_realization_closure_mode": _jsonable(mandatory_realization_closure_payload),
            "same_source_over_realization_mode": _jsonable(same_source_over_realization_payload),
            "periodic_realization_competition_mode": _jsonable(periodic_realization_competition_payload),
            "periodic_mode": _jsonable(periodic_mode_payload),
        },
        "enabled_steps": [
            "candidate_screen",
            "semantic_dedup",
            "consensus",
            "truth_recovery",
            *(
                ["chart_canonicalization"]
                if chart_canonicalization_enabled
                else []
            ),
            *(
                ["inner_chart_flip_compensation"]
                if inner_chart_flip_enabled
                else []
            ),
            *(
                ["basis_object_realization_injection"]
                if realization_prior_enabled
                else []
            ),
            *(
                ["mandatory_realization_closure"]
                if mandatory_realization_closure_enabled
                else []
            ),
            *(
                ["same_source_over_realization_collapse"]
                if same_source_over_realization_enabled
                else []
            ),
            *(
                ["canonical_trunk_lane"]
                if _canonical_trunk_lane_enabled(resolved_cfg)
                else []
            ),
            *(
                ["same_source_surrogate_lane"]
                if _same_source_surrogate_lane_enabled(resolved_cfg)
                else []
            ),
            *(
                ["periodic_realization_competition"]
                if periodic_realization_competition_enabled
                else []
            ),
            *(
                ["periodic_disambiguation"]
                if periodic_mode_enabled
                else []
            ),
        ],
        "current_narrowness": [
            "Representative selection inside an equivalence class remains heuristic rather than a global canonicalization pass.",
            "Current local-equivalence disambiguation is implemented only as a periodic specialization on configured periodic features.",
            "Realization prior injection is currently driven by object-member evidence rather than a fully learned realization grammar.",
        ],
        "notes": (
            "Current implementation formalizes symbolic equivalence handling across semantic/family/phase-equivalent layers. "
            "PeriodicEquivalenceDisambiguationMechanism is retained as the implemented periodic child mode under this parent protocol, "
            "while RealizationPriorInjection, MandatoryRealizationClosure, and PeriodicRealizationCompetition now govern "
            "basis-conditioned realization-head competition."
        ),
    }
    interference_feature_payload = {
        "protocol": str(resolved_cfg.interference_feature_protocol),
        "mode": str(resolved_cfg.interference_feature_mode),
        "cross_explanatory_rejection_mode": str(resolved_cfg.cross_explanatory_rejection_mode),
        "trivial_nonlinearity_penalty_mode": str(resolved_cfg.trivial_nonlinearity_penalty_mode),
        "environment_invariance_audit_mode": str(resolved_cfg.environment_invariance_audit_mode),
        "proxy_group_policy": str(resolved_cfg.proxy_group_policy),
        "source_overlap_penalty_mode": str(resolved_cfg.source_overlap_penalty_mode),
        "implemented_submodes": [
            *(
                ["native_proxy_check_mode"]
                if native_proxy_check_mode_enabled
                else []
            ),
            *(
                ["proxy_trunk_disqualification_mode"]
                if proxy_trunk_disqualification_mode_enabled
                else []
            ),
            *(
                ["proxy_suppression_mode"]
                if proxy_suppression_mode_enabled
                else []
            ),
            *(
                ["trivial_nonlinearity_rejection_mode"]
                if trivial_nonlinearity_mode_enabled
                else []
            ),
            *(
                ["causal_hierarchy_reuse_isolation_mode"]
                if causal_hierarchy_reuse_enabled
                else []
            ),
            *(
                ["parasitic_rejection_mode"]
                if parasitic_rejection_mode_enabled
                else []
            ),
            *(
                ["regional_correction_mode"]
                if regional_correction_mode_enabled
                else []
            ),
            *(
                ["regime_penetration_mode"]
                if regime_penetration_enabled
                else []
            ),
            *(
                ["heterogeneous_exposure_mode"]
                if heterogeneous_exposure_enabled
                else []
            ),
            *(
                ["environment_invariance_audit_mode"]
                if environment_invariance_mode_enabled
                else []
            ),
        ],
        "child_modes": {
            "native_proxy_check_mode": _jsonable(native_proxy_check_payload),
            "proxy_trunk_disqualification_mode": _jsonable(proxy_trunk_disqualification_payload),
            "proxy_suppression_mode": _jsonable(proxy_suppression_payload),
            "trivial_nonlinearity_rejection_mode": _jsonable(trivial_nonlinearity_payload),
            "causal_hierarchy_reuse_isolation_mode": _jsonable(causal_hierarchy_reuse_payload),
            "parasitic_rejection_mode": _jsonable(parasitic_rejection_payload),
            "regional_correction_mode": _jsonable(regional_correction_mode_payload),
            "regime_penetration_mode": _jsonable(regime_penetration_payload),
            "heterogeneous_exposure_mode": _jsonable(heterogeneous_exposure_payload),
        },
        "enabled_steps": [
            "feature_overlap_penalty",
            "semantic_dedup",
            "mechanistic_bias",
            *(
                ["native_proxy_check"]
                if native_proxy_check_mode_enabled
                else []
            ),
            *(
                ["proxy_trunk_disqualification"]
                if proxy_trunk_disqualification_mode_enabled
                else []
            ),
            *(
                ["cross_explanatory_hard_rejection"]
                if _cross_explanatory_rejection_enabled(resolved_cfg)
                else []
            ),
            *(
                ["trivial_nonlinearity_penalty"]
                if trivial_nonlinearity_mode_enabled
                else []
            ),
            *(
                ["causal_hierarchy_reuse_isolation"]
                if causal_hierarchy_reuse_enabled
                else []
            ),
            *(
                ["parasitic_rejection"]
                if parasitic_rejection_mode_enabled
                else []
            ),
            *(
                ["regime_penetration"]
                if regime_penetration_enabled
                else []
            ),
            *(
                ["heterogeneous_exposure_lane"]
                if heterogeneous_exposure_enabled
                else []
            ),
            *(
                ["environment_invariance_audit"]
                if environment_invariance_mode_enabled
                else []
            ),
            *(
                ["regional_residual_correction"]
                if regional_correction_mode_enabled
                else []
            ),
        ],
        "interference_feature_report": _jsonable(selected["interference_feature_report"]),
        "environment_invariance_audit": _jsonable(selected["environment_invariance_audit"]),
        "current_narrowness": [
            "Proxy suppression still relies on proxy-group hints and pairwise explainability rather than full causal intervention.",
            "Regional correction is now reopened as a small residual-local object search, but the local generator is still gate/grid-based rather than a full regional symbolic structure searcher.",
        ],
        "notes": (
            "Current implementation applies proxy-aware cross-explanatory rejection during outer candidate assembly, "
            "feeds trivial proxy-overlap penalties into the outer objective, and records a switchable environment "
            "invariance audit on the selected basis set. RegionalCorrectionBasisProtocol is retained as the implemented "
            "regional correction child mode under this parent protocol, now with a reopened residual-local gate search "
            "instead of pure screened-pool promotion. CausalHierarchyReuseIsolation additionally prevents correction "
            "branches from stealing trunk feature reuse budget during outer basis assembly. ProxyTrunkDisqualification "
            "and ParasiticRejectionCriteria further upgrade proxy/group conflicts and branch-vs-trunk conflicts into "
            "hard eligibility rules rather than post-hoc score nudges."
        ),
    }
    search_metadata["equivalence_expression_protocol"] = str(resolved_cfg.equivalence_expression_protocol)
    search_metadata["equivalence_expression_mode"] = str(resolved_cfg.equivalence_expression_mode)
    search_metadata["equivalence_class_scope"] = str(resolved_cfg.equivalence_class_scope)
    search_metadata["chart_canonicalization_protocol"] = str(resolved_cfg.chart_canonicalization_protocol)
    search_metadata["chart_canonicalization_mode"] = str(resolved_cfg.chart_canonicalization_mode)
    search_metadata["chart_orthodoxy_scoring_protocol"] = str(resolved_cfg.chart_orthodoxy_scoring_protocol)
    search_metadata["chart_orthodoxy_scoring_mode"] = str(resolved_cfg.chart_orthodoxy_scoring_mode)
    search_metadata["inner_chart_flip_compensation_protocol"] = str(
        resolved_cfg.inner_chart_flip_compensation_protocol
    )
    search_metadata["inner_chart_flip_compensation_mode"] = str(
        resolved_cfg.inner_chart_flip_compensation_mode
    )
    search_metadata["realization_prior_injection_protocol"] = str(resolved_cfg.realization_prior_injection_protocol)
    search_metadata["realization_prior_injection_mode"] = str(resolved_cfg.realization_prior_injection_mode)
    search_metadata["mandatory_realization_closure_protocol"] = str(
        resolved_cfg.mandatory_realization_closure_protocol
    )
    search_metadata["mandatory_realization_closure_mode"] = str(resolved_cfg.mandatory_realization_closure_mode)
    search_metadata["same_source_over_realization_protocol"] = str(resolved_cfg.same_source_over_realization_protocol)
    search_metadata["same_source_over_realization_mode"] = str(resolved_cfg.same_source_over_realization_mode)
    search_metadata["same_source_realization_budget"] = int(resolved_cfg.same_source_realization_budget)
    search_metadata["canonical_trunk_lane_protocol"] = str(resolved_cfg.canonical_trunk_lane_protocol)
    search_metadata["canonical_trunk_lane_mode"] = str(resolved_cfg.canonical_trunk_lane_mode)
    search_metadata["canonical_trunk_candidate_screen_reserve"] = int(resolved_cfg.canonical_trunk_candidate_screen_reserve)
    search_metadata["require_canonical_trunk_candidate_in_group"] = bool(
        resolved_cfg.require_canonical_trunk_candidate_in_group
    )
    search_metadata["min_canonical_trunk_basis_terms"] = int(resolved_cfg.min_canonical_trunk_basis_terms)
    search_metadata["same_source_surrogate_lane_protocol"] = str(resolved_cfg.same_source_surrogate_lane_protocol)
    search_metadata["same_source_surrogate_lane_mode"] = str(resolved_cfg.same_source_surrogate_lane_mode)
    search_metadata["periodic_realization_competition_protocol"] = str(resolved_cfg.periodic_realization_competition_protocol)
    search_metadata["periodic_realization_competition_mode"] = str(resolved_cfg.periodic_realization_competition_mode)
    search_metadata["interference_feature_protocol"] = str(resolved_cfg.interference_feature_protocol)
    search_metadata["interference_feature_mode"] = str(resolved_cfg.interference_feature_mode)
    search_metadata["regime_penetration_protocol"] = str(resolved_cfg.regime_penetration_protocol)
    search_metadata["regime_penetration_mode"] = str(resolved_cfg.regime_penetration_mode)
    search_metadata["regime_penetration_gain_floor"] = float(resolved_cfg.regime_penetration_gain_floor)
    search_metadata["heterogeneous_exposure_protocol"] = str(resolved_cfg.heterogeneous_exposure_protocol)
    search_metadata["heterogeneous_exposure_mode"] = str(resolved_cfg.heterogeneous_exposure_mode)
    search_metadata["heterogeneous_exposure_candidate_screen_reserve"] = int(
        resolved_cfg.heterogeneous_exposure_candidate_screen_reserve
    )
    search_metadata["heterogeneous_exposure_min_score"] = float(resolved_cfg.heterogeneous_exposure_min_score)
    search_metadata["native_proxy_check_protocol"] = str(resolved_cfg.native_proxy_check_protocol)
    search_metadata["native_proxy_check_mode"] = str(resolved_cfg.native_proxy_check_mode)
    search_metadata["proxy_trunk_disqualification_protocol"] = str(
        resolved_cfg.proxy_trunk_disqualification_protocol
    )
    search_metadata["proxy_trunk_disqualification_mode"] = str(resolved_cfg.proxy_trunk_disqualification_mode)
    search_metadata["parasitic_rejection_protocol"] = str(resolved_cfg.parasitic_rejection_protocol)
    search_metadata["parasitic_rejection_mode"] = str(resolved_cfg.parasitic_rejection_mode)
    search_metadata["causal_hierarchy_reuse_isolation_protocol"] = str(
        resolved_cfg.causal_hierarchy_reuse_isolation_protocol
    )
    search_metadata["causal_hierarchy_reuse_isolation_mode"] = str(resolved_cfg.causal_hierarchy_reuse_isolation_mode)
    search_metadata["cross_explanatory_rejection_mode"] = str(resolved_cfg.cross_explanatory_rejection_mode)
    search_metadata["trivial_nonlinearity_penalty_mode"] = str(resolved_cfg.trivial_nonlinearity_penalty_mode)
    search_metadata["environment_invariance_audit_mode"] = str(resolved_cfg.environment_invariance_audit_mode)
    search_metadata["proxy_group_policy"] = str(resolved_cfg.proxy_group_policy)
    search_metadata["source_overlap_penalty_mode"] = str(resolved_cfg.source_overlap_penalty_mode)
    search_metadata["equivalence_expression_handling"] = _jsonable(equivalence_expression_payload)
    search_metadata["interference_feature_handling"] = _jsonable(interference_feature_payload)
    search_metadata["chart_canonicalization"] = _jsonable(chart_canonicalization_payload)
    search_metadata["inner_chart_flip_compensation"] = _jsonable(inner_chart_flip_payload)
    search_metadata["same_source_over_realization_collapse"] = _jsonable(same_source_over_realization_payload)
    search_metadata["native_proxy_check"] = _jsonable(native_proxy_check_payload)
    search_metadata["realization_prior_injection"] = _jsonable(realization_prior_payload)
    search_metadata["mandatory_realization_closure"] = _jsonable(mandatory_realization_closure_payload)
    search_metadata["proxy_trunk_disqualification"] = _jsonable(proxy_trunk_disqualification_payload)
    search_metadata["parasitic_rejection"] = _jsonable(parasitic_rejection_payload)
    search_metadata["periodic_realization_competition"] = _jsonable(periodic_realization_competition_payload)
    search_metadata["causal_hierarchy_reuse_isolation"] = _jsonable(causal_hierarchy_reuse_payload)
    search_metadata["regime_penetration"] = _jsonable(regime_penetration_payload)
    search_metadata["heterogeneous_exposure_lane"] = _jsonable(heterogeneous_exposure_payload)
    search_metadata["periodic_equivalence_protocol"] = str(resolved_cfg.periodic_equivalence_protocol)
    search_metadata["periodic_equivalence_disambiguation_mode"] = str(
        resolved_cfg.periodic_equivalence_disambiguation_mode
    )
    search_metadata["phase_spectrum_audit_mode"] = str(resolved_cfg.phase_spectrum_audit_mode)
    search_metadata["periodic_family_prior_mode"] = str(resolved_cfg.periodic_family_prior_mode)
    search_metadata["periodic_family_prior_weight"] = float(resolved_cfg.periodic_family_prior_weight)
    search_metadata["periodic_candidate_screen_reserve"] = int(resolved_cfg.periodic_candidate_screen_reserve)
    search_metadata["require_periodic_candidate_in_group"] = bool(resolved_cfg.require_periodic_candidate_in_group)
    search_metadata["min_periodic_basis_terms"] = int(resolved_cfg.min_periodic_basis_terms)
    search_metadata["regional_correction_protocol"] = str(resolved_cfg.regional_correction_protocol)
    search_metadata["residual_regime_identification_mode"] = str(
        resolved_cfg.residual_regime_identification_mode
    )
    search_metadata["regional_correction_basis_mode"] = str(resolved_cfg.regional_correction_basis_mode)
    search_metadata["regional_correction_promotion_mode"] = str(
        resolved_cfg.regional_correction_promotion_mode
    )
    search_metadata["regional_correction_feature_scope"] = str(resolved_cfg.regional_correction_feature_scope)
    search_metadata["regional_correction_topk"] = int(resolved_cfg.regional_correction_topk)
    search_metadata["regional_correction_min_r2_gain"] = float(resolved_cfg.regional_correction_min_r2_gain)
    search_metadata["regional_correction_search_mode"] = str(resolved_cfg.regional_correction_search_mode)
    search_metadata["regional_local_search_beam_width"] = int(resolved_cfg.regional_local_search_beam_width)
    search_metadata["regional_local_search_branching_factor"] = int(
        resolved_cfg.regional_local_search_branching_factor
    )
    search_metadata["regional_local_search_max_expansions"] = int(resolved_cfg.regional_local_search_max_expansions)
    search_metadata["native_trunk_boundary_protocol"] = str(resolved_cfg.native_trunk_boundary_protocol)
    search_metadata["native_trunk_channel_mode"] = str(resolved_cfg.native_trunk_channel_mode)
    search_metadata["native_trunk_candidate_screen_reserve"] = int(resolved_cfg.native_trunk_candidate_screen_reserve)
    search_metadata["require_native_trunk_candidate_in_group"] = bool(
        resolved_cfg.require_native_trunk_candidate_in_group
    )
    search_metadata["min_native_trunk_basis_terms"] = int(resolved_cfg.min_native_trunk_basis_terms)
    search_metadata["native_trunk_residual_gain_floor"] = float(resolved_cfg.native_trunk_residual_gain_floor)
    search_metadata["native_trunk_interval_gain_floor"] = float(resolved_cfg.native_trunk_interval_gain_floor)
    search_metadata["outer_search_unit"] = str(resolved_cfg.outer_search_unit)
    search_metadata["representative_selection_rule"] = str(resolved_cfg.representative_selection_rule)
    search_metadata["native_trunk_channel"] = _jsonable(native_trunk_payload)
    search_metadata["canonical_trunk_lane"] = _jsonable(canonical_trunk_payload)
    search_metadata["same_source_surrogate_lane"] = _jsonable(same_source_surrogate_payload)
    search_metadata["periodic_equivalence_disambiguation"] = _jsonable(periodic_equivalence_payload)
    search_metadata["regional_correction_basis"] = _jsonable(regional_correction_payload)
    search_metadata["symbolic"]["equivalence_expression_protocol"] = str(resolved_cfg.equivalence_expression_protocol)
    search_metadata["symbolic"]["equivalence_expression_mode"] = str(resolved_cfg.equivalence_expression_mode)
    search_metadata["symbolic"]["equivalence_class_scope"] = str(resolved_cfg.equivalence_class_scope)
    search_metadata["symbolic"]["chart_canonicalization_protocol"] = str(resolved_cfg.chart_canonicalization_protocol)
    search_metadata["symbolic"]["chart_canonicalization_mode"] = str(resolved_cfg.chart_canonicalization_mode)
    search_metadata["symbolic"]["chart_orthodoxy_scoring_protocol"] = str(
        resolved_cfg.chart_orthodoxy_scoring_protocol
    )
    search_metadata["symbolic"]["chart_orthodoxy_scoring_mode"] = str(resolved_cfg.chart_orthodoxy_scoring_mode)
    search_metadata["symbolic"]["inner_chart_flip_compensation_protocol"] = str(
        resolved_cfg.inner_chart_flip_compensation_protocol
    )
    search_metadata["symbolic"]["inner_chart_flip_compensation_mode"] = str(
        resolved_cfg.inner_chart_flip_compensation_mode
    )
    search_metadata["symbolic"]["realization_prior_injection_protocol"] = str(
        resolved_cfg.realization_prior_injection_protocol
    )
    search_metadata["symbolic"]["realization_prior_injection_mode"] = str(
        resolved_cfg.realization_prior_injection_mode
    )
    search_metadata["symbolic"]["mandatory_realization_closure_protocol"] = str(
        resolved_cfg.mandatory_realization_closure_protocol
    )
    search_metadata["symbolic"]["mandatory_realization_closure_mode"] = str(
        resolved_cfg.mandatory_realization_closure_mode
    )
    search_metadata["symbolic"]["same_source_over_realization_protocol"] = str(
        resolved_cfg.same_source_over_realization_protocol
    )
    search_metadata["symbolic"]["same_source_over_realization_mode"] = str(
        resolved_cfg.same_source_over_realization_mode
    )
    search_metadata["symbolic"]["same_source_realization_budget"] = int(resolved_cfg.same_source_realization_budget)
    search_metadata["symbolic"]["canonical_trunk_lane_protocol"] = str(resolved_cfg.canonical_trunk_lane_protocol)
    search_metadata["symbolic"]["canonical_trunk_lane_mode"] = str(resolved_cfg.canonical_trunk_lane_mode)
    search_metadata["symbolic"]["canonical_trunk_candidate_screen_reserve"] = int(
        resolved_cfg.canonical_trunk_candidate_screen_reserve
    )
    search_metadata["symbolic"]["require_canonical_trunk_candidate_in_group"] = bool(
        resolved_cfg.require_canonical_trunk_candidate_in_group
    )
    search_metadata["symbolic"]["min_canonical_trunk_basis_terms"] = int(
        resolved_cfg.min_canonical_trunk_basis_terms
    )
    search_metadata["symbolic"]["same_source_surrogate_lane_protocol"] = str(
        resolved_cfg.same_source_surrogate_lane_protocol
    )
    search_metadata["symbolic"]["same_source_surrogate_lane_mode"] = str(
        resolved_cfg.same_source_surrogate_lane_mode
    )
    search_metadata["symbolic"]["periodic_realization_competition_protocol"] = str(
        resolved_cfg.periodic_realization_competition_protocol
    )
    search_metadata["symbolic"]["periodic_realization_competition_mode"] = str(
        resolved_cfg.periodic_realization_competition_mode
    )
    search_metadata["symbolic"]["interference_feature_protocol"] = str(resolved_cfg.interference_feature_protocol)
    search_metadata["symbolic"]["interference_feature_mode"] = str(resolved_cfg.interference_feature_mode)
    search_metadata["symbolic"]["regime_penetration_protocol"] = str(resolved_cfg.regime_penetration_protocol)
    search_metadata["symbolic"]["regime_penetration_mode"] = str(resolved_cfg.regime_penetration_mode)
    search_metadata["symbolic"]["regime_penetration_gain_floor"] = float(resolved_cfg.regime_penetration_gain_floor)
    search_metadata["symbolic"]["heterogeneous_exposure_protocol"] = str(
        resolved_cfg.heterogeneous_exposure_protocol
    )
    search_metadata["symbolic"]["heterogeneous_exposure_mode"] = str(resolved_cfg.heterogeneous_exposure_mode)
    search_metadata["symbolic"]["heterogeneous_exposure_candidate_screen_reserve"] = int(
        resolved_cfg.heterogeneous_exposure_candidate_screen_reserve
    )
    search_metadata["symbolic"]["heterogeneous_exposure_min_score"] = float(
        resolved_cfg.heterogeneous_exposure_min_score
    )
    search_metadata["symbolic"]["native_proxy_check_protocol"] = str(resolved_cfg.native_proxy_check_protocol)
    search_metadata["symbolic"]["native_proxy_check_mode"] = str(resolved_cfg.native_proxy_check_mode)
    search_metadata["symbolic"]["proxy_trunk_disqualification_protocol"] = str(
        resolved_cfg.proxy_trunk_disqualification_protocol
    )
    search_metadata["symbolic"]["proxy_trunk_disqualification_mode"] = str(
        resolved_cfg.proxy_trunk_disqualification_mode
    )
    search_metadata["symbolic"]["parasitic_rejection_protocol"] = str(resolved_cfg.parasitic_rejection_protocol)
    search_metadata["symbolic"]["parasitic_rejection_mode"] = str(resolved_cfg.parasitic_rejection_mode)
    search_metadata["symbolic"]["causal_hierarchy_reuse_isolation_protocol"] = str(
        resolved_cfg.causal_hierarchy_reuse_isolation_protocol
    )
    search_metadata["symbolic"]["causal_hierarchy_reuse_isolation_mode"] = str(
        resolved_cfg.causal_hierarchy_reuse_isolation_mode
    )
    search_metadata["symbolic"]["cross_explanatory_rejection_mode"] = str(resolved_cfg.cross_explanatory_rejection_mode)
    search_metadata["symbolic"]["trivial_nonlinearity_penalty_mode"] = str(resolved_cfg.trivial_nonlinearity_penalty_mode)
    search_metadata["symbolic"]["environment_invariance_audit_mode"] = str(resolved_cfg.environment_invariance_audit_mode)
    search_metadata["symbolic"]["proxy_group_policy"] = str(resolved_cfg.proxy_group_policy)
    search_metadata["symbolic"]["source_overlap_penalty_mode"] = str(resolved_cfg.source_overlap_penalty_mode)
    search_metadata["symbolic"]["equivalence_expression_handling"] = _jsonable(equivalence_expression_payload)
    search_metadata["symbolic"]["interference_feature_handling"] = _jsonable(interference_feature_payload)
    search_metadata["symbolic"]["chart_canonicalization"] = _jsonable(chart_canonicalization_payload)
    search_metadata["symbolic"]["inner_chart_flip_compensation"] = _jsonable(inner_chart_flip_payload)
    search_metadata["symbolic"]["same_source_over_realization_collapse"] = _jsonable(
        same_source_over_realization_payload
    )
    search_metadata["symbolic"]["native_proxy_check"] = _jsonable(native_proxy_check_payload)
    search_metadata["symbolic"]["realization_prior_injection"] = _jsonable(realization_prior_payload)
    search_metadata["symbolic"]["mandatory_realization_closure"] = _jsonable(mandatory_realization_closure_payload)
    search_metadata["symbolic"]["proxy_trunk_disqualification"] = _jsonable(proxy_trunk_disqualification_payload)
    search_metadata["symbolic"]["parasitic_rejection"] = _jsonable(parasitic_rejection_payload)
    search_metadata["symbolic"]["periodic_realization_competition"] = _jsonable(periodic_realization_competition_payload)
    search_metadata["symbolic"]["causal_hierarchy_reuse_isolation"] = _jsonable(causal_hierarchy_reuse_payload)
    search_metadata["symbolic"]["regime_penetration"] = _jsonable(regime_penetration_payload)
    search_metadata["symbolic"]["heterogeneous_exposure_lane"] = _jsonable(heterogeneous_exposure_payload)
    search_metadata["symbolic"]["periodic_equivalence_protocol"] = str(resolved_cfg.periodic_equivalence_protocol)
    search_metadata["symbolic"]["periodic_equivalence_disambiguation_mode"] = str(
        resolved_cfg.periodic_equivalence_disambiguation_mode
    )
    search_metadata["symbolic"]["phase_spectrum_audit_mode"] = str(resolved_cfg.phase_spectrum_audit_mode)
    search_metadata["symbolic"]["periodic_family_prior_mode"] = str(resolved_cfg.periodic_family_prior_mode)
    search_metadata["symbolic"]["periodic_candidate_screen_reserve"] = int(
        resolved_cfg.periodic_candidate_screen_reserve
    )
    search_metadata["symbolic"]["require_periodic_candidate_in_group"] = bool(
        resolved_cfg.require_periodic_candidate_in_group
    )
    search_metadata["symbolic"]["min_periodic_basis_terms"] = int(resolved_cfg.min_periodic_basis_terms)
    search_metadata["symbolic"]["regional_correction_protocol"] = str(resolved_cfg.regional_correction_protocol)
    search_metadata["symbolic"]["residual_regime_identification_mode"] = str(
        resolved_cfg.residual_regime_identification_mode
    )
    search_metadata["symbolic"]["regional_correction_basis_mode"] = str(
        resolved_cfg.regional_correction_basis_mode
    )
    search_metadata["symbolic"]["regional_correction_promotion_mode"] = str(
        resolved_cfg.regional_correction_promotion_mode
    )
    search_metadata["symbolic"]["regional_correction_feature_scope"] = str(
        resolved_cfg.regional_correction_feature_scope
    )
    search_metadata["symbolic"]["regional_correction_topk"] = int(resolved_cfg.regional_correction_topk)
    search_metadata["symbolic"]["regional_correction_min_r2_gain"] = float(
        resolved_cfg.regional_correction_min_r2_gain
    )
    search_metadata["symbolic"]["regional_correction_search_mode"] = str(
        resolved_cfg.regional_correction_search_mode
    )
    search_metadata["symbolic"]["regional_local_search_beam_width"] = int(
        resolved_cfg.regional_local_search_beam_width
    )
    search_metadata["symbolic"]["regional_local_search_branching_factor"] = int(
        resolved_cfg.regional_local_search_branching_factor
    )
    search_metadata["symbolic"]["regional_local_search_max_expansions"] = int(
        resolved_cfg.regional_local_search_max_expansions
    )
    search_metadata["symbolic"]["native_trunk_boundary_protocol"] = str(
        resolved_cfg.native_trunk_boundary_protocol
    )
    search_metadata["symbolic"]["native_trunk_channel_mode"] = str(resolved_cfg.native_trunk_channel_mode)
    search_metadata["symbolic"]["native_trunk_candidate_screen_reserve"] = int(
        resolved_cfg.native_trunk_candidate_screen_reserve
    )
    search_metadata["symbolic"]["require_native_trunk_candidate_in_group"] = bool(
        resolved_cfg.require_native_trunk_candidate_in_group
    )
    search_metadata["symbolic"]["min_native_trunk_basis_terms"] = int(
        resolved_cfg.min_native_trunk_basis_terms
    )
    search_metadata["symbolic"]["native_trunk_residual_gain_floor"] = float(
        resolved_cfg.native_trunk_residual_gain_floor
    )
    search_metadata["symbolic"]["native_trunk_interval_gain_floor"] = float(
        resolved_cfg.native_trunk_interval_gain_floor
    )
    search_metadata["symbolic"]["outer_search_unit"] = str(resolved_cfg.outer_search_unit)
    search_metadata["symbolic"]["representative_selection_rule"] = str(
        resolved_cfg.representative_selection_rule
    )
    search_metadata["symbolic"]["native_trunk_channel"] = _jsonable(native_trunk_payload)
    search_metadata["symbolic"]["canonical_trunk_lane"] = _jsonable(canonical_trunk_payload)
    search_metadata["symbolic"]["same_source_surrogate_lane"] = _jsonable(same_source_surrogate_payload)
    search_metadata["symbolic"]["periodic_equivalence_disambiguation"] = _jsonable(periodic_equivalence_payload)
    search_metadata["symbolic"]["regional_correction_basis"] = _jsonable(regional_correction_payload)
    search_metadata["symbolic"]["structure_engine"]["metadata"]["equivalence_expression_protocol"] = str(
        resolved_cfg.equivalence_expression_protocol
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["equivalence_expression_mode"] = str(
        resolved_cfg.equivalence_expression_mode
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["equivalence_class_scope"] = str(
        resolved_cfg.equivalence_class_scope
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["chart_canonicalization_protocol"] = str(
        resolved_cfg.chart_canonicalization_protocol
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["chart_canonicalization_mode"] = str(
        resolved_cfg.chart_canonicalization_mode
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["inner_chart_flip_compensation_protocol"] = str(
        resolved_cfg.inner_chart_flip_compensation_protocol
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["inner_chart_flip_compensation_mode"] = str(
        resolved_cfg.inner_chart_flip_compensation_mode
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["mandatory_realization_closure_protocol"] = str(
        resolved_cfg.mandatory_realization_closure_protocol
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["mandatory_realization_closure_mode"] = str(
        resolved_cfg.mandatory_realization_closure_mode
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["interference_feature_protocol"] = str(
        resolved_cfg.interference_feature_protocol
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["interference_feature_mode"] = str(
        resolved_cfg.interference_feature_mode
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["native_proxy_check_protocol"] = str(
        resolved_cfg.native_proxy_check_protocol
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["native_proxy_check_mode"] = str(
        resolved_cfg.native_proxy_check_mode
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["proxy_trunk_disqualification_protocol"] = str(
        resolved_cfg.proxy_trunk_disqualification_protocol
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["proxy_trunk_disqualification_mode"] = str(
        resolved_cfg.proxy_trunk_disqualification_mode
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["parasitic_rejection_protocol"] = str(
        resolved_cfg.parasitic_rejection_protocol
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["parasitic_rejection_mode"] = str(
        resolved_cfg.parasitic_rejection_mode
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["cross_explanatory_rejection_mode"] = str(
        resolved_cfg.cross_explanatory_rejection_mode
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["trivial_nonlinearity_penalty_mode"] = str(
        resolved_cfg.trivial_nonlinearity_penalty_mode
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["environment_invariance_audit_mode"] = str(
        resolved_cfg.environment_invariance_audit_mode
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["proxy_group_policy"] = str(
        resolved_cfg.proxy_group_policy
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["source_overlap_penalty_mode"] = str(
        resolved_cfg.source_overlap_penalty_mode
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["periodic_equivalence_protocol"] = str(
        resolved_cfg.periodic_equivalence_protocol
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["periodic_equivalence_disambiguation_mode"] = str(
        resolved_cfg.periodic_equivalence_disambiguation_mode
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["phase_spectrum_audit_mode"] = str(
        resolved_cfg.phase_spectrum_audit_mode
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["periodic_family_prior_mode"] = str(
        resolved_cfg.periodic_family_prior_mode
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["periodic_candidate_screen_reserve"] = int(
        resolved_cfg.periodic_candidate_screen_reserve
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["require_periodic_candidate_in_group"] = bool(
        resolved_cfg.require_periodic_candidate_in_group
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["min_periodic_basis_terms"] = int(
        resolved_cfg.min_periodic_basis_terms
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["regional_correction_protocol"] = str(
        resolved_cfg.regional_correction_protocol
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["residual_regime_identification_mode"] = str(
        resolved_cfg.residual_regime_identification_mode
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["regional_correction_basis_mode"] = str(
        resolved_cfg.regional_correction_basis_mode
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["regional_correction_promotion_mode"] = str(
        resolved_cfg.regional_correction_promotion_mode
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["regional_correction_feature_scope"] = str(
        resolved_cfg.regional_correction_feature_scope
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["regional_correction_topk"] = int(
        resolved_cfg.regional_correction_topk
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["regional_correction_min_r2_gain"] = float(
        resolved_cfg.regional_correction_min_r2_gain
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["regional_correction_search_mode"] = str(
        resolved_cfg.regional_correction_search_mode
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["regional_local_search_beam_width"] = int(
        resolved_cfg.regional_local_search_beam_width
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["regional_local_search_branching_factor"] = int(
        resolved_cfg.regional_local_search_branching_factor
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["regional_local_search_max_expansions"] = int(
        resolved_cfg.regional_local_search_max_expansions
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["outer_search_unit"] = str(
        resolved_cfg.outer_search_unit
    )
    search_metadata["symbolic"]["structure_engine"]["metadata"]["representative_selection_rule"] = str(
        resolved_cfg.representative_selection_rule
    )
    if symbolic_family_payload is not None:
        search_metadata["symbolic_family"] = _jsonable(symbolic_family_payload)
        search_metadata["symbolic"]["symbolic_family"] = _jsonable(symbolic_family_payload)
    return OrthogonalBasisFitResult(
        genome=tuple(dict(term) for term in tuple(selected["genome"])),
        readout_weight=np.asarray(selected["final_fit"].get("weight"), dtype=float),
        readout_bias=np.asarray(selected["final_fit"].get("bias"), dtype=float),
        pred_train=np.asarray(pred_train, dtype=float),
        residual_std=np.asarray(residual_std, dtype=float),
        train_metrics=dict(selected["final_fit"].get("metrics_train", {})),
        metadata=search_metadata,
    )


__all__ = [
    "OrthogonalBasisFitResult",
    "OrthogonalBasisSearchConfig",
    "fit_orthogonal_basis_symbolic",
]
