from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from conditional.primitives import ConditionalPrimitiveSpec

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.symbolic.expression_graph_cache import ExpressionGraphCache
from core.symbolic.artifact_schema import build_symbolic_structure_surface_payload
from core.symbolic.structure_metadata import (
    build_assembler_budget_payload,
    build_basis_overlap_report,
    build_basis_semantics_payload,
    build_basis_term_rows,
    build_residual_complementarity_report,
    build_semantic_dedup_report,
)
from core.symbolic.trainer_family import build_unified_symbolic_family_spec
from core.symbolic.symbolic_dsl import expression_to_string
from core.symbolic.symbolic_structure_search import evaluate_genome_with_ridge
from examples.path_defaults import apply_env_defaults, default_work_ci_csv
from examples.work_ci_reader import WorkCiIntervalReader
from nowcasting_work_ci.mlblack_side.config import MlblackRuntimeConfig, build_output_root
from pipeline.feature_space import (
    CandidatePoolConfig,
    FeatureBundle,
    FeatureEngineeringConfig,
    batched_ridge_predict,
    build_full_candidate_pool,
    build_interval_subset_report,
    build_rolling_splits,
    build_subset_candidate_metadata,
    build_subset_genome,
    build_feature_bundle,
    design_matrix_for_genome,
    interval_metrics_batch,
    interval_objective_sort_key,
    symmetric_interval_batch,
)


def _as_2d_float(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("array must be 2D")
    return arr


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


def _parse_float_csv(raw: str, *, default: Sequence[float]) -> tuple[float, ...]:
    text = str(raw).strip()
    if not text:
        return tuple(float(v) for v in default)
    values: list[float] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(float(item))
    return tuple(values or [float(v) for v in default])


def _parse_str_csv(raw: str, *, default: Sequence[str]) -> tuple[str, ...]:
    text = str(raw).strip()
    if not text:
        return tuple(str(v).strip() for v in tuple(default) if str(v).strip())
    values = [str(part).strip() for part in text.split(",") if str(part).strip()]
    return tuple(values or [str(v).strip() for v in tuple(default) if str(v).strip()])


def _prepare_output_root(*, seed: int, stamp: str | None = None) -> Path:
    cfg = MlblackRuntimeConfig(
        output_prefix="nowcasting_orthogonal_basis_symbolic_work_ci_seed",
        graph_cache_namespace="work_ci_orthogonal_basis_symbolic",
    )
    return build_output_root(ROOT, seed=int(seed), stamp=stamp, cfg=cfg)


@dataclass(frozen=True)
class OrthogonalBasisSearchConfig:
    min_basis_count: int = 3
    max_basis_count: int = 6
    candidate_limit: int = 96
    seed_candidate_count: int = 18
    group_count: int = 12
    max_pair_abs_corr: float = 0.35
    max_feature_reuse: int = 2
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
    l2_grid: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1e-1)
    rolling_folds: int = 3
    rolling_val_ratio: float = 0.18
    min_train_ratio: float = 0.40
    interval_alpha: float = 0.20
    coverage_error_threshold: float = 0.08
    selection_mode: str = "interval_first"
    random_seed: int = 42
    max_semantic_repeats: int = 1
    max_piecewise_semantic_repeats: int = 2
    enable_piecewise_basis: bool = True
    gate_feature_names: tuple[str, ...] = tuple()
    gate_quantiles: tuple[float, ...] = (0.35, 0.50, 0.65)
    gate_families: tuple[str, ...] = ("gate_step", "piecewise_hinge", "piecewise")
    gate_slope: float = 8.0
    piecewise_left_mode: str = "identity"
    piecewise_right_mode: str = "relu"

    def normalized(self) -> "OrthogonalBasisSearchConfig":
        l2_grid = tuple(sorted(float(max(0.0, v)) for v in self.l2_grid))
        mode = str(self.selection_mode).strip().lower()
        if mode not in {"interval_first", "orthogonal_first", "rmse_first"}:
            raise ValueError("selection_mode must be interval_first | orthogonal_first | rmse_first")
        gate_quantiles = tuple(
            sorted(
                float(np.clip(value, 0.05, 0.95))
                for value in tuple(self.gate_quantiles)
                if np.isfinite(float(value))
            )
        )
        gate_feature_names = tuple(str(v).strip() for v in tuple(self.gate_feature_names) if str(v).strip())
        gate_families = tuple(
            family
            for family in tuple(str(v).strip().lower() for v in tuple(self.gate_families))
            if family in {"gate_step", "gate_soft", "piecewise_hinge", "piecewise"}
        )
        return OrthogonalBasisSearchConfig(
            min_basis_count=int(max(2, self.min_basis_count)),
            max_basis_count=int(max(max(2, self.min_basis_count), self.max_basis_count)),
            candidate_limit=int(max(8, self.candidate_limit)),
            seed_candidate_count=int(max(3, self.seed_candidate_count)),
            group_count=int(max(1, self.group_count)),
            max_pair_abs_corr=float(np.clip(self.max_pair_abs_corr, 0.05, 0.98)),
            max_feature_reuse=int(max(1, self.max_feature_reuse)),
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
            l2_grid=l2_grid or (1e-6, 1e-4, 1e-2, 1e-1),
            rolling_folds=int(max(1, self.rolling_folds)),
            rolling_val_ratio=float(np.clip(self.rolling_val_ratio, 0.05, 0.40)),
            min_train_ratio=float(np.clip(self.min_train_ratio, 0.10, 0.80)),
            interval_alpha=float(np.clip(self.interval_alpha, 1e-6, 0.99)),
            coverage_error_threshold=float(max(0.0, self.coverage_error_threshold)),
            selection_mode=mode,
            random_seed=int(self.random_seed),
            max_semantic_repeats=int(max(1, self.max_semantic_repeats)),
            max_piecewise_semantic_repeats=int(max(1, self.max_piecewise_semantic_repeats)),
            enable_piecewise_basis=bool(self.enable_piecewise_basis),
            gate_feature_names=gate_feature_names,
            gate_quantiles=gate_quantiles or (0.35, 0.50, 0.65),
            gate_families=gate_families or ("gate_step", "piecewise_hinge", "piecewise"),
            gate_slope=float(max(1.0, self.gate_slope)),
            piecewise_left_mode=str(self.piecewise_left_mode or "identity").strip().lower() or "identity",
            piecewise_right_mode=str(self.piecewise_right_mode or "relu").strip().lower() or "relu",
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
    activation_mean: float
    activation_std: float
    expression: str
    semantic_signature: str
    semantic_family: str
    uses_piecewise_gate: bool


def _candidate_expr_key(expr: Mapping[str, Any]) -> str:
    return json.dumps(dict(expr), sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _single_basis_row(
    *,
    name: str,
    expr: Mapping[str, Any],
    feature_names: Sequence[str],
    scope: str = "global",
) -> dict[str, Any]:
    rows = build_basis_term_rows(
        [{"name": str(name), "expr": dict(expr)}],
        feature_names=tuple(str(v) for v in tuple(feature_names)),
        scope=str(scope),
    )
    return dict(rows[0]) if rows else {"term_name": str(name), "expression": expression_to_string(dict(expr), precision=8)}


def _selected_basis_rows(selected_rows: Sequence[ScreenedCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "term_name": str(row.name),
            "expression": str(row.expression),
            "feature_names": [str(v) for v in tuple(row.features)],
            "feature_indices": [int(v) for v in tuple(row.features)],
            "feature_count": int(len(tuple(row.features))),
            "semantic_signature": str(row.semantic_signature),
            "semantic_family": str(row.semantic_family),
            "uses_piecewise_gate": bool(row.uses_piecewise_gate),
            "scope": "global",
        }
        for row in tuple(selected_rows)
    ]


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


def _selected_matrix(train_matrix: np.ndarray, selected_rows: Sequence[ScreenedCandidate]) -> np.ndarray:
    if not selected_rows:
        return np.zeros((int(np.asarray(train_matrix).shape[0]), 0), dtype=float)
    return np.asarray(train_matrix[:, [int(row.screen_index) for row in tuple(selected_rows)]], dtype=float)


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


def _configured_gate_feature_names(
    *,
    cfg: OrthogonalBasisSearchConfig,
    feature_names: Sequence[str],
) -> tuple[str, ...]:
    allowed = {str(name) for name in tuple(feature_names)}
    return tuple(name for name in tuple(cfg.gate_feature_names) if str(name) in allowed)


def _build_piecewise_gate_specs(
    *,
    feature_bundle: FeatureBundle,
    cfg: OrthogonalBasisSearchConfig,
) -> tuple[ConditionalPrimitiveSpec, ...]:
    if not bool(cfg.enable_piecewise_basis):
        return tuple()
    feature_names = tuple(str(v) for v in tuple(feature_bundle.feature_names))
    gate_features = _configured_gate_feature_names(cfg=cfg, feature_names=feature_names)
    if not gate_features:
        return tuple()
    x_train = np.asarray(feature_bundle.X_train, dtype=float)
    name_to_idx = {str(name): int(idx) for idx, name in enumerate(feature_names)}
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


def _augment_pool_config_with_gate_specs(
    *,
    pool_cfg: CandidatePoolConfig,
    extra_specs: Sequence[ConditionalPrimitiveSpec],
) -> CandidatePoolConfig:
    if not extra_specs:
        return pool_cfg
    existing = pool_cfg.conditional_config
    if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes, bytearray)):
        merged = tuple(spec for spec in existing if isinstance(spec, ConditionalPrimitiveSpec)) + tuple(extra_specs)
    elif existing is None:
        merged = tuple(extra_specs)
    else:
        merged = tuple(extra_specs)
    return replace(pool_cfg, conditional_config=merged)


def _screen_candidate_pool(
    *,
    candidates: Sequence[Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    candidate_limit: int,
    graph_cache: ExpressionGraphCache | None = None,
) -> tuple[list[ScreenedCandidate], np.ndarray]:
    xtr = np.asarray(X_train, dtype=float)
    ytr = np.asarray(y_train, dtype=float).reshape(-1)
    rows: list[tuple[ScreenedCandidate, np.ndarray]] = []
    seen_expr: set[str] = set()
    for pool_index, candidate in enumerate(tuple(candidates)):
        expr = dict(candidate.expr)
        expr_key = _candidate_expr_key(expr)
        if expr_key in seen_expr:
            continue
        basis_row = _single_basis_row(name=str(candidate.name), expr=expr, feature_names=feature_names)
        values = design_matrix_for_genome(
            [{"name": str(candidate.name), "expr": expr}],
            xtr,
            graph_cache=graph_cache,
            batch_key=f"screen::{pool_index}",
        ).reshape(-1)
        if values.shape[0] != xtr.shape[0]:
            continue
        if not np.all(np.isfinite(values)):
            continue
        std = float(np.std(values, ddof=0))
        if std <= 1e-10:
            continue
        target_corr = float(abs(_safe_corr(values, ytr)))
        screen_score = float(target_corr / (1.0 + 0.08 * float(candidate.complexity)))
        row = ScreenedCandidate(
            pool_index=int(pool_index),
            screen_index=-1,
            name=str(candidate.name),
            expr=expr,
            family=str(candidate.family),
            complexity=float(candidate.complexity),
            features=tuple(int(v) for v in tuple(candidate.features)),
            target_corr=float(target_corr),
            screen_score=float(screen_score),
            activation_mean=float(np.mean(values)),
            activation_std=float(std),
            expression=expression_to_string(expr, precision=8),
            semantic_signature=str(basis_row.get("semantic_signature", "")),
            semantic_family=str(basis_row.get("semantic_family", candidate.family)),
            uses_piecewise_gate=bool(basis_row.get("uses_piecewise_gate")),
        )
        rows.append((row, values))
        seen_expr.add(expr_key)

    rows.sort(key=lambda item: (-float(item[0].screen_score), float(item[0].complexity), str(item[0].name)))
    limited = rows[: int(candidate_limit)]

    screened: list[ScreenedCandidate] = []
    values_out: list[np.ndarray] = []
    for screen_index, (row, values) in enumerate(limited):
        screened.append(
            ScreenedCandidate(
                pool_index=int(row.pool_index),
                screen_index=int(screen_index),
                name=str(row.name),
                expr=dict(row.expr),
                family=str(row.family),
                complexity=float(row.complexity),
                features=tuple(int(v) for v in row.features),
                target_corr=float(row.target_corr),
                screen_score=float(row.screen_score),
                activation_mean=float(row.activation_mean),
                activation_std=float(row.activation_std),
                expression=str(row.expression),
                semantic_signature=str(row.semantic_signature),
                semantic_family=str(row.semantic_family),
                uses_piecewise_gate=bool(row.uses_piecewise_gate),
            )
        )
        values_out.append(np.asarray(values, dtype=float).reshape(-1))
    if not values_out:
        return [], np.zeros((int(np.asarray(X_train).shape[0]), 0), dtype=float)
    return screened, np.asarray(np.stack(values_out, axis=1), dtype=float)


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
    for i, left in enumerate(tuple(rows)):
        left_features = set(int(v) for v in left.features)
        for right in tuple(rows)[i + 1 :]:
            right_features = set(int(v) for v in right.features)
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
    condition = 1.0
    if singular.size > 0:
        floor = float(max(1e-12, np.min(singular)))
        condition = float(np.max(singular) / floor)
    rank = int(np.linalg.matrix_rank(std_matrix)) if std_matrix.size > 0 else 0
    pair_abs_corr_mean = float(np.mean(pair_values)) if pair_values.size > 0 else 0.0
    pair_abs_corr_max = float(np.max(pair_values)) if pair_values.size > 0 else 0.0
    feature_overlap_mean = float(_group_feature_overlap_mean(selected_rows))
    target_corr_mean = float(np.mean([float(row.target_corr) for row in selected_rows])) if selected_rows else 0.0
    target_corr_max = float(np.max([float(row.target_corr) for row in selected_rows])) if selected_rows else 0.0
    orthogonality_score = float(
        1.0
        / (
            1.0
            + pair_abs_corr_mean
            + 0.25 * pair_abs_corr_max
            + 0.10 * feature_overlap_mean
            + 0.02 * max(0.0, math.log1p(condition - 1.0))
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
    cfg: OrthogonalBasisSearchConfig,
    fallback_mode: str | None = None,
) -> dict[str, Any]:
    screen_positions = tuple(sorted(int(row.screen_index) for row in selected_rows))
    train_values = np.asarray(train_matrix[:, screen_positions], dtype=float)
    basis_rows = _selected_basis_rows(selected_rows)
    orthogonality = _orthogonality_metrics(selected_rows=selected_rows, train_values=train_values)
    residual_steps = _residual_complementarity_steps(
        selected_rows=selected_rows,
        train_matrix=train_matrix,
        target=target,
        l2_value=float(min(tuple(cfg.l2_grid)) if tuple(cfg.l2_grid) else 1e-6),
    )
    residual_report = build_residual_complementarity_report(
        residual_steps,
        source="orthogonal_basis_discovery",
        extra={"selection_threshold": float(threshold)},
    )
    semantic_report = build_semantic_dedup_report(
        basis_rows,
        source="orthogonal_basis_discovery",
        extra={"selection_threshold": float(threshold)},
    )
    orthogonality["semantic_unique_ratio"] = float(semantic_report.get("semantic_unique_ratio", 0.0))
    orthogonality["piecewise_gate_term_count"] = int(semantic_report.get("piecewise_gate_term_count", 0))
    orthogonality["residual_gain_mean"] = float(residual_report.get("mean_marginal_r2_gain", 0.0))
    orthogonality["residual_gain_min"] = float(residual_report.get("min_marginal_r2_gain", 0.0))
    group_score = float(
        np.mean([float(row.screen_score) for row in selected_rows])
        + 0.45 * float(orthogonality["orthogonality_score"])
        + float(cfg.residual_gain_weight) * float(residual_report.get("mean_marginal_r2_gain", 0.0))
        + 0.10 * float(semantic_report.get("semantic_unique_ratio", 0.0))
        - 0.20 * float(orthogonality["pair_abs_corr_mean"])
        - 0.05 * float(orthogonality["feature_overlap_mean"])
    )
    payload = {
        "threshold": float(threshold),
        "screen_positions": [int(v) for v in screen_positions],
        "pool_indices": [int(row.pool_index) for row in tuple(selected_rows)],
        "rows": [row for row in tuple(selected_rows)],
        "orthogonality_metrics": dict(orthogonality),
        "group_score": float(group_score),
        "residual_complementarity_report": _jsonable(residual_report),
        "semantic_dedup_report": _jsonable(semantic_report),
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
    cfg: OrthogonalBasisSearchConfig,
) -> float:
    if not selected_rows:
        return float(candidate.screen_score)
    selected_idx = [int(row.screen_index) for row in selected_rows]
    pair_corrs = [float(corr_matrix[int(candidate.screen_index), idx]) for idx in selected_idx]
    mean_pair = float(np.mean(pair_corrs)) if pair_corrs else 0.0
    overlap_count = float(sum(int(used_feature_counts.get(int(v), 0)) for v in candidate.features))
    new_feature_count = float(
        sum(1 for value in candidate.features if int(used_feature_counts.get(int(value), 0)) <= 0)
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
    semantic_repeat_penalty = float(cfg.semantic_dup_penalty) * float(
        signature_counts.get(str(candidate.semantic_signature), 0)
    )
    piecewise_requested = bool(cfg.enable_piecewise_basis and tuple(cfg.gate_feature_names))
    selected_piecewise = any(bool(row.uses_piecewise_gate) for row in tuple(selected_rows))
    piecewise_bonus = 0.0
    if bool(candidate.uses_piecewise_gate):
        piecewise_bonus = float(cfg.piecewise_gate_bonus)
        if piecewise_requested and not selected_piecewise:
            piecewise_bonus *= 2.5
    return float(
        cfg.target_score_weight * float(candidate.screen_score)
        - cfg.diversity_corr_weight * mean_pair
        - cfg.feature_overlap_penalty * overlap_count
        - cfg.complexity_penalty * float(candidate.complexity)
        - semantic_repeat_penalty
        + cfg.new_feature_bonus * new_feature_count
        + family_bonus
        + semantic_family_bonus
        + float(cfg.residual_corr_weight) * residual_abs_corr
        + float(cfg.residual_gain_weight) * max(0.0, marginal_r2_gain)
        + piecewise_bonus
    )


def _accept_candidate(
    *,
    candidate: ScreenedCandidate,
    selected_rows: Sequence[ScreenedCandidate],
    corr_matrix: np.ndarray,
    used_feature_counts: Counter[int],
    signature_counts: Counter[str],
    max_pair_abs_corr: float,
    max_feature_reuse: int,
    cfg: OrthogonalBasisSearchConfig,
) -> bool:
    if any(int(candidate.screen_index) == int(row.screen_index) for row in selected_rows):
        return False
    if selected_rows:
        pair_corrs = [float(corr_matrix[int(candidate.screen_index), int(row.screen_index)]) for row in selected_rows]
        if pair_corrs and float(max(pair_corrs)) > float(max_pair_abs_corr):
            return False
    for feature_index in candidate.features:
        if int(used_feature_counts.get(int(feature_index), 0)) >= int(max_feature_reuse):
            return False
    if int(signature_counts.get(str(candidate.semantic_signature), 0)) >= _semantic_repeat_limit(candidate, cfg):
        return False
    return True


def _discover_group_candidates(
    *,
    screened: Sequence[ScreenedCandidate],
    train_matrix: np.ndarray,
    y_train: np.ndarray,
    cfg: OrthogonalBasisSearchConfig,
) -> list[dict[str, Any]]:
    if not screened:
        return []
    corr_matrix = _pairwise_abs_corr(train_matrix)
    seed_limit = min(int(cfg.seed_candidate_count), int(len(screened)))
    thresholds = (
        float(cfg.max_pair_abs_corr),
        float(min(0.55, cfg.max_pair_abs_corr + 0.10)),
        float(min(0.75, cfg.max_pair_abs_corr + 0.25)),
    )
    seen_groups: set[tuple[int, ...]] = set()
    groups: list[dict[str, Any]] = []
    for seed in tuple(screened)[:seed_limit]:
        best_local: dict[str, Any] | None = None
        for threshold in thresholds:
            selected: list[ScreenedCandidate] = [seed]
            used_feature_counts: Counter[int] = Counter(int(v) for v in seed.features)
            signature_counts: Counter[str] = Counter([str(seed.semantic_signature)])
            while len(selected) < int(cfg.max_basis_count):
                current_fit = _ridge_projection(
                    _selected_matrix(train_matrix, selected),
                    np.asarray(y_train, dtype=float).reshape(-1),
                    l2_value=float(min(tuple(cfg.l2_grid)) if tuple(cfg.l2_grid) else 1e-6),
                )
                best_candidate: ScreenedCandidate | None = None
                best_score = float("-inf")
                for candidate in screened:
                    if not _accept_candidate(
                        candidate=candidate,
                        selected_rows=selected,
                        corr_matrix=corr_matrix,
                        used_feature_counts=used_feature_counts,
                        signature_counts=signature_counts,
                        max_pair_abs_corr=float(threshold),
                        max_feature_reuse=int(cfg.max_feature_reuse),
                        cfg=cfg,
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
                        cfg=cfg,
                    )
                    if score > best_score:
                        best_score = float(score)
                        best_candidate = candidate
                if best_candidate is None:
                    break
                selected.append(best_candidate)
                for value in best_candidate.features:
                    used_feature_counts[int(value)] += 1
                signature_counts[str(best_candidate.semantic_signature)] += 1
            if len(selected) < int(cfg.min_basis_count):
                continue
            payload = _group_summary_payload(
                selected_rows=selected,
                threshold=float(threshold),
                train_matrix=train_matrix,
                target=np.asarray(y_train, dtype=float).reshape(-1),
                cfg=cfg,
            )
            if best_local is None or float(payload["group_score"]) > float(best_local["group_score"]):
                best_local = payload
        if best_local is None:
            continue
        group_key = tuple(int(v) for v in best_local["pool_indices"])
        if group_key in seen_groups:
            continue
        seen_groups.add(group_key)
        groups.append(best_local)
    groups.sort(
        key=lambda item: (
            -float(item["group_score"]),
            float(dict(item["orthogonality_metrics"]).get("pair_abs_corr_mean", 1.0)),
            -int(len(tuple(item["screen_positions"]))),
        )
    )
    if groups:
        return groups[: int(cfg.group_count)]

    relaxed_selected: list[ScreenedCandidate] = []
    relaxed_feature_counts: Counter[int] = Counter()
    relaxed_signature_counts: Counter[str] = Counter()
    for candidate in screened:
        if not relaxed_selected:
            relaxed_selected.append(candidate)
            for value in candidate.features:
                relaxed_feature_counts[int(value)] += 1
            relaxed_signature_counts[str(candidate.semantic_signature)] += 1
            continue
        if len(relaxed_selected) >= int(cfg.max_basis_count):
            break
        if not _accept_candidate(
            candidate=candidate,
            selected_rows=relaxed_selected,
            corr_matrix=corr_matrix,
            used_feature_counts=relaxed_feature_counts,
            signature_counts=relaxed_signature_counts,
            max_pair_abs_corr=0.98,
            max_feature_reuse=int(cfg.max_feature_reuse) + 2,
            cfg=cfg,
        ):
            continue
        relaxed_selected.append(candidate)
        for value in candidate.features:
            relaxed_feature_counts[int(value)] += 1
        relaxed_signature_counts[str(candidate.semantic_signature)] += 1
    if len(relaxed_selected) >= int(cfg.min_basis_count):
        groups.append(
            _group_summary_payload(
                selected_rows=relaxed_selected,
                threshold=0.98,
                train_matrix=train_matrix,
                target=np.asarray(y_train, dtype=float).reshape(-1),
                cfg=cfg,
                fallback_mode="relaxed_threshold",
            )
        )
    return groups[: int(cfg.group_count)]


def _fold_rows_for_l2_grid(
    *,
    genome: Sequence[Mapping[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    l2_grid: Sequence[float],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    alpha: float,
    graph_cache: ExpressionGraphCache | None = None,
) -> dict[float, list[dict[str, Any]]]:
    rows_by_l2: dict[float, list[dict[str, Any]]] = {float(v): [] for v in l2_grid}
    for fold_index, (tr_idx, va_idx) in enumerate(tuple(splits)):
        xtr = np.asarray(X_train[tr_idx], dtype=float)
        ytr = np.asarray(y_train[tr_idx], dtype=float)
        xva = np.asarray(X_train[va_idx], dtype=float)
        yva = np.asarray(y_train[va_idx], dtype=float)
        genomes = [list(genome) for _ in l2_grid]
        pred_eval, pred_train = batched_ridge_predict(
            genomes=genomes,
            X_train=xtr,
            y_train=ytr,
            X_eval=xva,
            l2_values=[float(v) for v in l2_grid],
            graph_cache=graph_cache,
            batch_key_train=f"orthogonal_fold_{fold_index}_train",
            batch_key_eval=f"orthogonal_fold_{fold_index}_eval",
        )
        lower, upper, quantiles = symmetric_interval_batch(
            y_train=ytr,
            pred_train=pred_train,
            pred_eval=pred_eval,
            alpha=float(alpha),
        )
        metrics = interval_metrics_batch(
            y_true=yva,
            lower=lower,
            upper=upper,
            alpha=float(alpha),
        )
        for idx, l2_value in enumerate(tuple(float(v) for v in l2_grid)):
            rows_by_l2[float(l2_value)].append(
                {
                    "coverage_error": float(metrics["coverage_error"][idx]),
                    "picp": float(metrics["picp"][idx]),
                    "pinaw": float(metrics["pinaw"][idx]),
                    "interval_score": float(metrics["interval_score"][idx]),
                    "mean_width": float(metrics["mean_width"][idx]),
                    "rmse": float(_rmse(yva, pred_eval[idx])),
                    "branch_detail": {
                        "fold_index": int(fold_index),
                        "l2": float(l2_value),
                    },
                    "interval_info": {
                        "symmetric_residual_q": float(quantiles[idx]),
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


def _fit_group_on_test(
    *,
    genome: Sequence[Mapping[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    l2_value: float,
    alpha: float,
    graph_cache: ExpressionGraphCache | None = None,
) -> dict[str, Any]:
    fit = evaluate_genome_with_ridge(
        genome,
        X_train=np.asarray(X_train, dtype=float),
        y_train=np.asarray(y_train, dtype=float),
        X_eval=np.asarray(X_test, dtype=float),
        y_eval=np.asarray(y_test, dtype=float),
        l2=float(l2_value),
        graph_cache=graph_cache,
        train_batch_key="orthogonal_final_train",
        eval_batch_key="orthogonal_final_test",
    )
    pred_train = _as_2d_float(np.asarray(fit.get("pred_train"), dtype=float))
    pred_test = _as_2d_float(np.asarray(fit.get("pred_eval"), dtype=float))
    lower, upper, quantile = symmetric_interval_batch(
        y_train=np.asarray(y_train, dtype=float),
        pred_train=pred_train.reshape(1, pred_train.shape[0], pred_train.shape[1]),
        pred_eval=pred_test.reshape(1, pred_test.shape[0], pred_test.shape[1]),
        alpha=float(alpha),
    )
    interval_metrics = interval_metrics_batch(
        y_true=np.asarray(y_test, dtype=float),
        lower=lower,
        upper=upper,
        alpha=float(alpha),
    )
    return {
        "fit": fit,
        "test_interval": {
            "coverage_error": float(interval_metrics["coverage_error"][0]),
            "picp": float(interval_metrics["picp"][0]),
            "pinaw": float(interval_metrics["pinaw"][0]),
            "interval_score": float(interval_metrics["interval_score"][0]),
            "mean_width": float(interval_metrics["mean_width"][0]),
            "coverage_target": float(interval_metrics["coverage_target"][0]),
            "symmetric_residual_q": float(quantile[0]),
        },
        "lower": np.asarray(lower[0], dtype=float),
        "upper": np.asarray(upper[0], dtype=float),
    }


def _build_expression_payload(
    *,
    genome: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    weight: np.ndarray,
    bias: np.ndarray,
) -> dict[str, Any]:
    basis_rows = build_basis_term_rows(
        genome,
        feature_names=tuple(str(v) for v in feature_names),
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
                "feature_names": [str(v) for v in tuple(row.get("feature_names", ()))],
            }
        )
        expr_parts.append(f"({float(coef):.8g})*({expression})")
    expr_parts.append(f"({float(intercept):.8g})")
    return {
        "expression": " + ".join(expr_parts),
        "terms": terms,
        "intercept": float(intercept),
    }


def _evaluate_group(
    *,
    feature_bundle: FeatureBundle,
    candidates: Sequence[Any],
    group_payload: Mapping[str, Any],
    cfg: OrthogonalBasisSearchConfig,
    graph_cache: ExpressionGraphCache | None = None,
) -> dict[str, Any]:
    pool_indices = [int(v) for v in tuple(group_payload.get("pool_indices", ()))]
    genome = build_subset_genome(candidates=candidates, subset_idx=pool_indices)
    subset_candidates = build_subset_candidate_metadata(candidates=candidates, subset_idx=pool_indices)
    splits = build_rolling_splits(
        int(feature_bundle.X_train.shape[0]),
        folds=int(cfg.rolling_folds),
        val_ratio=float(cfg.rolling_val_ratio),
        min_train=max(64, int(round(float(cfg.min_train_ratio) * float(feature_bundle.X_train.shape[0])))),
    )
    rows_by_l2 = _fold_rows_for_l2_grid(
        genome=genome,
        X_train=np.asarray(feature_bundle.X_train, dtype=float),
        y_train=np.asarray(feature_bundle.y_train, dtype=float),
        l2_grid=tuple(float(v) for v in cfg.l2_grid),
        splits=splits,
        alpha=float(cfg.interval_alpha),
        graph_cache=graph_cache,
    )
    orthogonality_metrics = dict(group_payload.get("orthogonality_metrics", {}))
    validation_rows: list[dict[str, Any]] = []
    for l2_value, fold_rows in rows_by_l2.items():
        _, detail = build_interval_subset_report(
            subset_idx=pool_indices,
            subset_candidates=subset_candidates,
            fold_results=fold_rows,
            decode_meta={"tuned_l2": float(l2_value)},
            selection_coverage_error_threshold=float(cfg.coverage_error_threshold),
            jsonable_fn=_jsonable,
        )
        validation_rows.append(
            {
                "l2": float(l2_value),
                "detail": detail,
                "sort_key": _validation_sort_key(
                    summary_detail=detail,
                    orthogonality_metrics=orthogonality_metrics,
                    cfg=cfg,
                ),
            }
        )
    validation_rows.sort(key=lambda item: item["sort_key"])
    selected_validation = validation_rows[0]
    best_l2 = float(selected_validation["l2"])
    final_fit = _fit_group_on_test(
        genome=genome,
        X_train=np.asarray(feature_bundle.X_train, dtype=float),
        y_train=np.asarray(feature_bundle.y_train, dtype=float),
        X_test=np.asarray(feature_bundle.X_test, dtype=float),
        y_test=np.asarray(feature_bundle.y_test, dtype=float),
        l2_value=best_l2,
        alpha=float(cfg.interval_alpha),
        graph_cache=graph_cache,
    )
    basis_rows = build_basis_term_rows(
        genome,
        feature_names=tuple(str(v) for v in feature_bundle.feature_names),
        scope="global",
    )
    gate_feature_names = tuple(_configured_gate_feature_names(cfg=cfg, feature_names=feature_bundle.feature_names))
    gate_feature_set = set(gate_feature_names)
    gate_basis_rows = [
        dict(row)
        for row in tuple(basis_rows)
        if bool(row.get("uses_piecewise_gate"))
        or bool(gate_feature_set & {str(name) for name in tuple(row.get("feature_names", ()))})
    ]
    residual_complementarity_report = dict(group_payload.get("residual_complementarity_report", {}))
    if not residual_complementarity_report:
        residual_complementarity_report = build_residual_complementarity_report(
            _residual_complementarity_steps(
                selected_rows=tuple(group_payload.get("rows", ())),
                train_matrix=np.asarray(
                    design_matrix_for_genome(
                        genome,
                        np.asarray(feature_bundle.X_train, dtype=float),
                        graph_cache=graph_cache,
                        batch_key="orthogonal_group_recompute_train",
                    ),
                    dtype=float,
                ),
                target=np.asarray(feature_bundle.y_train, dtype=float),
                l2_value=float(best_l2),
            ),
            source="orthogonal_basis_discovery",
        )
    semantic_dedup_report = dict(group_payload.get("semantic_dedup_report", {}))
    if not semantic_dedup_report:
        semantic_dedup_report = build_semantic_dedup_report(
            basis_rows,
            source="orthogonal_basis_discovery",
        )
    orthogonality_metrics = dict(group_payload.get("orthogonality_metrics", {}))
    orthogonality_metrics["semantic_unique_ratio"] = float(semantic_dedup_report.get("semantic_unique_ratio", 0.0))
    orthogonality_metrics["piecewise_gate_term_count"] = int(semantic_dedup_report.get("piecewise_gate_term_count", 0))
    orthogonality_metrics["residual_gain_mean"] = float(
        residual_complementarity_report.get("mean_marginal_r2_gain", 0.0)
    )
    orthogonality_metrics["residual_gain_min"] = float(
        residual_complementarity_report.get("min_marginal_r2_gain", 0.0)
    )
    basis_semantics = build_basis_semantics_payload(
        basis_rows,
        source="orthogonal_basis_discovery",
        basis_scope="global",
        extra={
            "selection_mode": str(cfg.selection_mode),
            "relative_orthogonality": True,
            "selection_threshold": float(group_payload.get("threshold", cfg.max_pair_abs_corr)),
            "gate_feature_names": list(gate_feature_names),
            "semantic_unique_ratio": float(semantic_dedup_report.get("semantic_unique_ratio", 0.0)),
        },
    )
    basis_overlap_report = build_basis_overlap_report(
        basis_rows,
        source="orthogonal_basis_discovery",
        extra={
            "orthogonality_score": float(orthogonality_metrics.get("orthogonality_score", 0.0)),
            "pair_abs_corr_mean": float(orthogonality_metrics.get("pair_abs_corr_mean", 0.0)),
            "pair_abs_corr_max": float(orthogonality_metrics.get("pair_abs_corr_max", 0.0)),
            "semantic_unique_ratio": float(semantic_dedup_report.get("semantic_unique_ratio", 0.0)),
            "residual_gain_mean": float(residual_complementarity_report.get("mean_marginal_r2_gain", 0.0)),
        },
    )
    assembler_budget = build_assembler_budget_payload(
        source="orthogonal_budgeted_symbolic_regression",
        assembler_mode="budgeted_symbolic_regression",
        output_expression_count=1,
        selected_basis_count=int(len(pool_indices)),
        budget_axes={
            "basis_count": int(len(pool_indices)),
            "candidate_limit": int(cfg.candidate_limit),
            "selected_l2": float(best_l2),
            "rolling_folds": int(cfg.rolling_folds),
        },
        budget_scale="small",
        uses_piecewise_gate=bool(gate_basis_rows),
        extra={
            "selection_mode": str(cfg.selection_mode),
            "coverage_error_threshold": float(cfg.coverage_error_threshold),
            "gate_basis_count": int(len(gate_basis_rows)),
        },
    )
    expression_payload = _build_expression_payload(
        genome=genome,
        feature_names=feature_bundle.feature_names,
        weight=np.asarray(final_fit["fit"]["weight"], dtype=float),
        bias=np.asarray(final_fit["fit"]["bias"], dtype=float),
    )
    fit_metrics = dict(final_fit["fit"].get("metrics_eval", {}))
    symbolic_family = build_unified_symbolic_family_spec(
        trainer_key="symbolic",
        parameter_backend="ridge",
        task="point",
        trainer_state_enabled=True,
        supports_resume=True,
        supports_warm_start=True,
        supports_piecewise_basis=bool(gate_feature_names),
        metadata={
            "preset_kind": "orthogonal_basis_benchmark",
            "surface_status": "formal",
            "supports_piecewise_basis": bool(gate_feature_names),
            "discovery_mode": "correlation+residual+semantic_dedup",
        },
    )
    family_payload = dict(symbolic_family.description_dict())
    family_signature = symbolic_family.family_signature()
    validation_summary = dict(selected_validation["detail"])
    stability_metrics = {
        "fold_count": int(len(validation_rows[0]["detail"].get("fold_rmse", tuple())) if validation_rows else int(cfg.rolling_folds)),
        "fold_summary": _jsonable(validation_summary),
    }
    for key in (
        "rmse_mean",
        "rmse_std",
        "rmse_drift",
        "coverage_error_mean",
        "pinaw_mean",
        "interval_score_mean",
        "picp_mean",
        "mean_width_mean",
        "family_concentration",
        "feature_concentration",
    ):
        if key in validation_summary:
            stability_metrics[str(key)] = _jsonable(validation_summary.get(key))
    complexity_metrics = {
        "term_count": int(len(tuple(basis_rows))),
        "basis_count": int(len(tuple(basis_rows))),
        "gate_basis_count": int(len(gate_basis_rows)),
        "orthogonality_score": float(orthogonality_metrics.get("orthogonality_score", 0.0)),
        "pair_abs_corr_mean": float(orthogonality_metrics.get("pair_abs_corr_mean", 0.0)),
        "semantic_unique_ratio": float(semantic_dedup_report.get("semantic_unique_ratio", 0.0)),
        "mean_residual_gain": float(residual_complementarity_report.get("mean_marginal_r2_gain", 0.0)),
    }
    gate_indices = [
        int(index)
        for index, name in enumerate(tuple(str(v) for v in tuple(feature_bundle.feature_names)))
        if str(name) in gate_feature_set
    ]
    surface_metadata = {
        "symbolic_family": family_payload,
        "symbolic_family_signature": family_signature,
        "selected_basis": _jsonable(basis_rows),
        "basis_semantics": _jsonable(basis_semantics),
        "basis_overlap_report": _jsonable(basis_overlap_report),
        "residual_complementarity_report": _jsonable(residual_complementarity_report),
        "semantic_dedup_report": _jsonable(semantic_dedup_report),
        "assembler_budget": _jsonable(assembler_budget),
        "symbolic": {
            "selected_basis": _jsonable(basis_rows),
            "basis_semantics": _jsonable(basis_semantics),
            "basis_overlap_report": _jsonable(basis_overlap_report),
            "residual_complementarity_report": _jsonable(residual_complementarity_report),
            "semantic_dedup_report": _jsonable(semantic_dedup_report),
            "assembler_budget": _jsonable(assembler_budget),
        },
        "training_signature": {
            "symbolic_family_signature": family_signature,
            "metadata": {
                "symbolic_family": family_payload,
            },
        },
        "gate_piecewise": {
            "gate_feature_names": list(gate_feature_names),
            "gate_indices": [int(v) for v in gate_indices],
            "gate_basis_terms": _jsonable(gate_basis_rows),
        },
    }
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
    symbolic_artifact_schema = {
        "schema_key": "symbolic_artifact_v1",
        "head_semantics": {
            "task": "point",
            "outputs": ["mean"],
            "objective_family": "regression",
            "calibration_mode": "none",
        },
        "complexity_metrics": _jsonable(complexity_metrics),
        "stability_metrics": _jsonable(stability_metrics),
        **_jsonable(symbolic_structure_surface),
    }
    report = {
        "pool_indices": [int(v) for v in pool_indices],
        "basis_count": int(len(pool_indices)),
        "discovery_group": {
            "threshold": float(group_payload.get("threshold", cfg.max_pair_abs_corr)),
            "group_score": float(group_payload.get("group_score", 0.0)),
            "fallback_mode": None if group_payload.get("fallback_mode") is None else str(group_payload.get("fallback_mode")),
        },
        "subset_candidates": _jsonable(subset_candidates),
        "orthogonality_metrics": _jsonable(orthogonality_metrics),
        "validation_summary": _jsonable(validation_summary),
        "validation_l2_candidates": _jsonable(
            [{"l2": float(row["l2"]), "detail": row["detail"]} for row in validation_rows]
        ),
        "selected_l2": float(best_l2),
        "test_metrics": {
            "rmse": float(fit_metrics.get("rmse", float("inf"))),
            "mae": float(fit_metrics.get("mae", float("inf"))),
            "r2": float(fit_metrics.get("r2", float("nan"))),
        },
        "test_interval_metrics": _jsonable(final_fit["test_interval"]),
        "basis_rows": _jsonable(basis_rows),
        "basis_semantics": _jsonable(basis_semantics),
        "basis_overlap_report": _jsonable(basis_overlap_report),
        "residual_complementarity_report": _jsonable(residual_complementarity_report),
        "semantic_dedup_report": _jsonable(semantic_dedup_report),
        "assembly_budget_usage": _jsonable(assembler_budget),
        "final_expression": _jsonable(expression_payload),
        "symbolic_family": _jsonable(family_payload),
        "symbolic_family_signature": family_signature,
        "symbolic_structure_surface": _jsonable(symbolic_structure_surface),
        "symbolic_artifact_schema": _jsonable(symbolic_artifact_schema),
        "selection_summary": {
            "selection_mode": str(cfg.selection_mode),
            "selection_description": "orthogonal basis first, budgeted symbolic assembler second / 先找相对正交 basis，再做小预算符号组装",
        },
    }
    if group_payload.get("fallback_mode") is not None:
        report["fallback_mode"] = str(group_payload.get("fallback_mode"))
    report["selection_sort_key"] = _jsonable(
        _validation_sort_key(
            summary_detail=report["validation_summary"],
            orthogonality_metrics=orthogonality_metrics,
            cfg=cfg,
        )
    )
    return report


def run_orthogonal_symbolic_experiment(
    *,
    feature_bundle: FeatureBundle,
    cfg: OrthogonalBasisSearchConfig | None = None,
    candidates: Sequence[Any] | None = None,
    pool_cfg: CandidatePoolConfig | None = None,
    graph_cache: ExpressionGraphCache | None = None,
    experiment_name: str = "orthogonal_basis_symbolic",
    output_dir: str | Path | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_cfg = (cfg or OrthogonalBasisSearchConfig()).normalized()
    gate_specs = _build_piecewise_gate_specs(feature_bundle=feature_bundle, cfg=resolved_cfg)
    resolved_pool_cfg = _augment_pool_config_with_gate_specs(
        pool_cfg=pool_cfg if pool_cfg is not None else CandidatePoolConfig(),
        extra_specs=gate_specs,
    )
    candidate_pool = list(candidates) if candidates is not None else build_full_candidate_pool(
        feature_bundle,
        resolved_pool_cfg,
    )
    screened, train_matrix = _screen_candidate_pool(
        candidates=candidate_pool,
        X_train=np.asarray(feature_bundle.X_train, dtype=float),
        y_train=np.asarray(feature_bundle.y_train, dtype=float),
        feature_names=feature_bundle.feature_names,
        candidate_limit=int(resolved_cfg.candidate_limit),
        graph_cache=graph_cache,
    )
    if not screened:
        raise RuntimeError("no valid symbolic candidates were screened for orthogonal basis discovery")
    group_payloads = _discover_group_candidates(
        screened=screened,
        train_matrix=train_matrix,
        y_train=np.asarray(feature_bundle.y_train, dtype=float),
        cfg=resolved_cfg,
    )
    if not group_payloads:
        raise RuntimeError("no orthogonal basis groups were generated from the screened candidate pool")
    evaluated = [
        _evaluate_group(
            feature_bundle=feature_bundle,
            candidates=candidate_pool,
            group_payload=payload,
            cfg=resolved_cfg,
            graph_cache=graph_cache,
        )
        for payload in group_payloads
    ]
    evaluated.sort(
        key=lambda item: _validation_sort_key(
            summary_detail=dict(item["validation_summary"]),
            orthogonality_metrics=dict(item["orthogonality_metrics"]),
            cfg=resolved_cfg,
        )
    )
    best_group = evaluated[0]
    summary = {
        "generated_at": datetime.now().isoformat(),
        "experiment_name": str(experiment_name),
        "description": "orthogonal basis first, budgeted symbolic assembler second / 先找相对正交 basis，再做小预算符号组装",
        "config": _jsonable(asdict(resolved_cfg)),
        "data": {
            "n_train": int(np.asarray(feature_bundle.X_train).shape[0]),
            "n_test": int(np.asarray(feature_bundle.X_test).shape[0]),
            "feature_count": int(len(tuple(feature_bundle.feature_names))),
            "feature_names": [str(v) for v in tuple(feature_bundle.feature_names)],
            "raw_feature_count": int(feature_bundle.n_features_raw),
            "lag_added_features": [str(v) for v in tuple(feature_bundle.lag_added_features)],
            "lag_cross_added_features": [str(v) for v in tuple(feature_bundle.lag_cross_added_features)],
            "dropped_features": [str(v) for v in tuple(feature_bundle.dropped_features)],
        },
        "discovery": {
            "candidate_pool_size": int(len(candidate_pool)),
            "screened_candidate_count": int(len(screened)),
            "generated_group_count": int(len(group_payloads)),
            "piecewise_gate_seed_count": int(len(gate_specs)),
            "piecewise_gate_features": [str(v) for v in tuple(_configured_gate_feature_names(cfg=resolved_cfg, feature_names=feature_bundle.feature_names))],
            "screened_candidates": _jsonable(
                [
                    {
                        "screen_index": int(row.screen_index),
                        "pool_index": int(row.pool_index),
                        "name": str(row.name),
                        "family": str(row.family),
                        "complexity": float(row.complexity),
                        "target_corr": float(row.target_corr),
                        "screen_score": float(row.screen_score),
                        "features": [int(v) for v in tuple(row.features)],
                        "expression": str(row.expression),
                        "semantic_signature": str(row.semantic_signature),
                        "semantic_family": str(row.semantic_family),
                        "uses_piecewise_gate": bool(row.uses_piecewise_gate),
                    }
                    for row in screened
                ]
            ),
        },
        "best_group": _jsonable(best_group),
        "top_groups": _jsonable(evaluated[: int(resolved_cfg.group_count)]),
        "extra_metadata": {} if extra_metadata is None else _jsonable(dict(extra_metadata)),
    }
    if output_dir is not None:
        out_root = Path(output_dir).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        summary_path = out_root / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["summary_path"] = str(summary_path)
    return summary


def _build_feature_bundle_from_reader(
    *,
    csv_path: str,
    target_col: str,
    test_fold_col: str,
    lag_feature_enabled: bool,
    lag_orders_csv: str,
    lag_sources_csv: str,
    lag_cross_enabled: bool,
    lag_cross_quantiles_csv: str,
    drop_same_day_flow_speed_occ: bool,
    drop_feature_list_csv: str,
) -> tuple[FeatureBundle, dict[str, Any]]:
    reader = WorkCiIntervalReader(
        csv_path=str(csv_path),
        target_col=str(target_col),
        test_fold_col=str(test_fold_col),
    )
    bundle = reader.read()
    feature_bundle = build_feature_bundle(
        X_train=np.asarray(bundle.train.X_train, dtype=float),
        y_train=np.asarray(bundle.train.y_train, dtype=float),
        X_test=np.asarray(bundle.test.X_train, dtype=float),
        y_test=np.asarray(bundle.test.y_train, dtype=float),
        feature_names=tuple(str(v) for v in bundle.train.feature_names),
        cfg=FeatureEngineeringConfig(
            lag_feature_enabled=bool(lag_feature_enabled),
            lag_orders_csv=str(lag_orders_csv),
            lag_sources_csv=str(lag_sources_csv),
            lag_cross_enabled=bool(lag_cross_enabled),
            lag_cross_quantiles_csv=str(lag_cross_quantiles_csv),
            drop_same_day_flow_speed_occ=bool(drop_same_day_flow_speed_occ),
            drop_feature_list_csv=str(drop_feature_list_csv),
        ),
    )
    metadata = {
        "reader": "WorkCiIntervalReader",
        "csv_path": str(csv_path),
        "target_col": str(target_col),
        "test_fold_col": str(test_fold_col),
        "reader_metadata": _jsonable(bundle.metadata),
        "feature_engineering": {
            "lag_feature_enabled": bool(lag_feature_enabled),
            "lag_orders_csv": str(lag_orders_csv),
            "lag_sources_csv": str(lag_sources_csv),
            "lag_cross_enabled": bool(lag_cross_enabled),
            "lag_cross_quantiles_csv": str(lag_cross_quantiles_csv),
            "drop_same_day_flow_speed_occ": bool(drop_same_day_flow_speed_occ),
            "drop_feature_list_csv": str(drop_feature_list_csv),
        },
    }
    return feature_bundle, metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orthogonal-basis symbolic experiment for nowcasting work_ci.",
    )
    parser.add_argument("--csv-path", type=str, default=default_work_ci_csv())
    parser.add_argument("--target-col", type=str, default="ci")
    parser.add_argument("--test-fold-col", type=str, default="test_fold_10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stamp", type=str, default="")
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--candidate-limit", type=int, default=96)
    parser.add_argument("--seed-candidate-count", type=int, default=18)
    parser.add_argument("--group-count", type=int, default=12)
    parser.add_argument("--min-basis-count", type=int, default=3)
    parser.add_argument("--max-basis-count", type=int, default=6)
    parser.add_argument("--max-pair-abs-corr", type=float, default=0.35)
    parser.add_argument("--max-feature-reuse", type=int, default=2)
    parser.add_argument("--l2-grid", type=str, default="1e-6,1e-4,1e-2,1e-1")
    parser.add_argument("--rolling-folds", type=int, default=3)
    parser.add_argument("--rolling-val-ratio", type=float, default=0.18)
    parser.add_argument("--min-train-ratio", type=float, default=0.40)
    parser.add_argument("--interval-alpha", type=float, default=0.20)
    parser.add_argument("--coverage-error-threshold", type=float, default=0.08)
    parser.add_argument("--selection-mode", type=str, default="interval_first")
    parser.add_argument("--enable-piecewise-basis", type=int, default=1)
    parser.add_argument("--gate-feature-names", type=str, default="")
    parser.add_argument("--gate-quantiles", type=str, default="0.35,0.5,0.65")
    parser.add_argument("--gate-families", type=str, default="gate_step,piecewise_hinge,piecewise")
    parser.add_argument("--gate-slope", type=float, default=8.0)
    parser.add_argument("--lag-feature-enabled", type=int, default=1)
    parser.add_argument("--lag-orders", type=str, default="1,2,3")
    parser.add_argument("--lag-sources", type=str, default="ci,total_flow,avg_speed,avg_occ")
    parser.add_argument("--lag-cross-enabled", type=int, default=1)
    parser.add_argument("--lag-cross-quantiles", type=str, default="0.25,0.5,0.75")
    parser.add_argument("--drop-same-day-flow-speed-occ", type=int, default=1)
    parser.add_argument("--drop-feature-list", type=str, default="total_flow,avg_speed,avg_occ")
    parser.add_argument("--graph-cache-enabled", type=int, default=1)
    parser.add_argument("--graph-cache-backend", type=str, default="sqlite")
    return parser


def main(argv: list[str] | None = None) -> None:
    apply_env_defaults()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_root = (
        Path(args.output_root).resolve()
        if str(args.output_root).strip()
        else _prepare_output_root(seed=int(args.seed), stamp=(None if not str(args.stamp).strip() else str(args.stamp)))
    )
    output_root.mkdir(parents=True, exist_ok=True)

    graph_cache = ExpressionGraphCache(
        enabled=bool(int(args.graph_cache_enabled)),
        backend=str(args.graph_cache_backend),
        db_path=str(output_root / "expression_graph_cache.sqlite3"),
        namespace=f"orthogonal_basis::{args.test_fold_col}",
        persist_values=False,
    )
    feature_bundle, metadata = _build_feature_bundle_from_reader(
        csv_path=str(args.csv_path),
        target_col=str(args.target_col),
        test_fold_col=str(args.test_fold_col),
        lag_feature_enabled=bool(int(args.lag_feature_enabled)),
        lag_orders_csv=str(args.lag_orders),
        lag_sources_csv=str(args.lag_sources),
        lag_cross_enabled=bool(int(args.lag_cross_enabled)),
        lag_cross_quantiles_csv=str(args.lag_cross_quantiles),
        drop_same_day_flow_speed_occ=bool(int(args.drop_same_day_flow_speed_occ)),
        drop_feature_list_csv=str(args.drop_feature_list),
    )
    cfg = OrthogonalBasisSearchConfig(
        min_basis_count=int(args.min_basis_count),
        max_basis_count=int(args.max_basis_count),
        candidate_limit=int(args.candidate_limit),
        seed_candidate_count=int(args.seed_candidate_count),
        group_count=int(args.group_count),
        max_pair_abs_corr=float(args.max_pair_abs_corr),
        max_feature_reuse=int(args.max_feature_reuse),
        l2_grid=_parse_float_csv(str(args.l2_grid), default=(1e-6, 1e-4, 1e-2, 1e-1)),
        rolling_folds=int(args.rolling_folds),
        rolling_val_ratio=float(args.rolling_val_ratio),
        min_train_ratio=float(args.min_train_ratio),
        interval_alpha=float(args.interval_alpha),
        coverage_error_threshold=float(args.coverage_error_threshold),
        selection_mode=str(args.selection_mode),
        random_seed=int(args.seed),
        enable_piecewise_basis=bool(int(args.enable_piecewise_basis)),
        gate_feature_names=_parse_str_csv(str(args.gate_feature_names), default=tuple()),
        gate_quantiles=_parse_float_csv(str(args.gate_quantiles), default=(0.35, 0.5, 0.65)),
        gate_families=_parse_str_csv(str(args.gate_families), default=("gate_step", "piecewise_hinge", "piecewise")),
        gate_slope=float(args.gate_slope),
    )
    summary = run_orthogonal_symbolic_experiment(
        feature_bundle=feature_bundle,
        cfg=cfg,
        graph_cache=graph_cache,
        experiment_name="nowcasting_work_ci_orthogonal_basis_symbolic",
        output_dir=output_root,
        extra_metadata=metadata,
    )
    best_group = dict(summary["best_group"])
    test_metrics = dict(best_group.get("test_metrics", {}))
    test_interval = dict(best_group.get("test_interval_metrics", {}))
    orthogonality = dict(best_group.get("orthogonality_metrics", {}))
    residual = dict(best_group.get("residual_complementarity_report", {}))
    semantic = dict(best_group.get("semantic_dedup_report", {}))
    print("ORTHOGONAL BASIS SYMBOLIC RUN")
    print(f"summary={summary.get('summary_path', str(output_root / 'summary.json'))}")
    print(
        "rmse={rmse:.6f} mae={mae:.6f} r2={r2:.6f}".format(
            rmse=float(test_metrics.get("rmse", float("inf"))),
            mae=float(test_metrics.get("mae", float("inf"))),
            r2=float(test_metrics.get("r2", float("nan"))),
        )
    )
    print(
        "picp={picp:.6f} pinaw={pinaw:.6f} is={iscore:.6f} cov_err={cov:.6f}".format(
            picp=float(test_interval.get("picp", float("nan"))),
            pinaw=float(test_interval.get("pinaw", float("nan"))),
            iscore=float(test_interval.get("interval_score", float("nan"))),
            cov=float(test_interval.get("coverage_error", float("nan"))),
        )
    )
    print(
        "orthogonality_score={score:.6f} pair_corr_mean={mean_corr:.6f} pair_corr_max={max_corr:.6f}".format(
            score=float(orthogonality.get("orthogonality_score", float("nan"))),
            mean_corr=float(orthogonality.get("pair_abs_corr_mean", float("nan"))),
            max_corr=float(orthogonality.get("pair_abs_corr_max", float("nan"))),
        )
    )
    print(
        "residual_gain_mean={gain:.6f} semantic_unique_ratio={ratio:.6f} gate_basis_count={gate_count}".format(
            gain=float(residual.get("mean_marginal_r2_gain", float("nan"))),
            ratio=float(semantic.get("semantic_unique_ratio", float("nan"))),
            gate_count=int(dict(best_group.get("symbolic_structure_surface", {})).get("piecewise_gate_basis", {}).get("gate_basis_count", 0)),
        )
    )


__all__ = [
    "OrthogonalBasisSearchConfig",
    "build_arg_parser",
    "main",
    "run_orthogonal_symbolic_experiment",
]
