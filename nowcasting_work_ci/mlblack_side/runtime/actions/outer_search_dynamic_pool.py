from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from bias import collect_selected_expr_keys, should_expand_dynamic_pool
from core.symbolic.symbolic_structure_search import evaluate_genome_with_ridge
from model.interval_fit import _as_2d
from pipeline.feature_space import (
    CandidateTerm,
    _expand_candidate_pool_from_residual,
    _prune_candidate_pool,
)

from ..config import RuntimeCliConfig


def maybe_expand_candidate_pool(
    args: RuntimeCliConfig,
    prepared: Mapping[str, Any],
    *,
    epoch_idx: int,
    epoch_generations: Sequence[int],
    top_cache: Sequence[Mapping[str, Any]],
    row: Mapping[str, Any],
    subset_idx: Sequence[int],
    genome: Sequence[Mapping[str, Any]],
    candidates: Sequence[CandidateTerm],
    feature_names: Sequence[str],
) -> tuple[list[CandidateTerm], int, int]:
    dynamic_pool_policy = prepared["dynamic_pool_policy"]
    dynamic_top_cache_use = int(prepared["dynamic_top_cache_use"])
    dynamic_expand_max_new = int(prepared["dynamic_expand_max_new"])
    dynamic_focus_top_features = int(prepared["dynamic_focus_top_features"])
    dynamic_partner_topk = int(prepared["dynamic_partner_topk"])
    dynamic_max_pool_size = int(prepared["dynamic_max_pool_size"])
    dynamic_activation_cfg = dict(prepared["dynamic_activation_cfg"])
    conditional_config = prepared.get("conditional_config")

    selected_keys = collect_selected_expr_keys(
        top_cache,
        candidates,
        top_cache_use=int(dynamic_top_cache_use),
    )
    current_candidates = list(candidates)
    if not should_expand_dynamic_pool(
        dynamic_pool_policy,
        epoch_idx=epoch_idx,
        epoch_generations=epoch_generations,
        has_active_subset=bool(subset_idx),
    ):
        return current_candidates, 0, int(len(current_candidates))

    X_train = np.asarray(prepared["X_train"], dtype=float)
    y_train = np.asarray(prepared["y_train"], dtype=float)
    l2_ep = float(max(0.0, row.get("tuned_l2", args.ridge_l2)))
    fit_ep = evaluate_genome_with_ridge(
        genome,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_train,
        y_eval=y_train,
        l2=l2_ep,
    )
    pred_tr = _as_2d(np.asarray(fit_ep.get("pred_train"), dtype=float))
    res_tr = _as_2d(np.asarray(y_train - pred_tr, dtype=float))
    new_terms = _expand_candidate_pool_from_residual(
        X=X_train,
        y_residual=res_tr,
        feature_names=tuple(str(v) for v in feature_names),
        base_genome=genome,
        base_weight=_as_2d(np.asarray(fit_ep.get("weight"), dtype=float)),
        existing=current_candidates,
        max_new_terms=int(dynamic_expand_max_new),
        focus_top_features=int(dynamic_focus_top_features),
        partner_topk=int(dynamic_partner_topk),
        activation_config=dynamic_activation_cfg,
        conditional_config=conditional_config,
    )
    if new_terms:
        current_candidates = current_candidates + list(new_terms)
    current_candidates = _prune_candidate_pool(
        candidates=current_candidates,
        keep_expr_keys=selected_keys,
        feature_names=tuple(str(v) for v in feature_names),
        max_pool_size=int(dynamic_max_pool_size),
    )
    return current_candidates, int(len(new_terms)), int(len(current_candidates))


__all__ = ["maybe_expand_candidate_pool"]
