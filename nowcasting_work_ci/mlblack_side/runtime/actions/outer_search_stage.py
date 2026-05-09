from __future__ import annotations

from typing import Any, Mapping

from bias import build_epoch_generations

from ..config import RuntimeCliConfig
from ..contracts import RuntimeContextKey, SearchStageResult, ctx_require, ctx_set
from .outer_search_dynamic_pool import maybe_expand_candidate_pool
from .outer_search_problem import run_outer_epoch
from .outer_search_tracking import (
    BestSolutionTracker,
    build_best_decode_meta,
    build_dynamic_epoch_log,
    extract_epoch_leader,
    update_best_solution,
)


def run_outer_search(args: RuntimeCliConfig, prepared: Mapping[str, Any]) -> SearchStageResult:
    feature_names = tuple(str(v) for v in prepared["feature_names"])
    dynamic_pool_policy = prepared["dynamic_pool_policy"]
    dynamic_top_cache_use = int(prepared["dynamic_top_cache_use"])
    objective_policy = prepared["objective_policy"]
    candidates = list(prepared["candidates"])

    epoch_generations = build_epoch_generations(
        int(prepared["effective_generations"]),
        cfg=dynamic_pool_policy,
    )

    outer_sec = 0.0
    outer_meta: dict[str, Any] = {
        "strategy": str(args.outer_strategy),
        "max_generations": int(prepared["effective_generations"]),
    }
    resource_budget: dict[str, Any] = {}
    run: dict[str, Any] = {"status": "completed", "steps_executed": 0}
    top_cache: list[dict[str, Any]] = []
    problem = None
    best = BestSolutionTracker()
    dynamic_epoch_logs: list[dict[str, Any]] = []

    for ep, gens_this in enumerate(epoch_generations):
        epoch_run = run_outer_epoch(
            args,
            prepared,
            run_candidates=candidates,
            generations_this_epoch=int(gens_this),
            seed_this_epoch=int(args.seed + ep),
        )
        problem = epoch_run.problem
        outer_sec += float(epoch_run.duration_sec)
        run = dict(epoch_run.run)
        run["steps_executed"] = int(run.get("steps_executed", 0)) + int(sum(epoch_generations[:ep]))
        outer_meta = dict(epoch_run.outer_meta)
        resource_budget = dict(epoch_run.resource_budget)
        top_cache_ep = problem.cache_top(topn=max(50, dynamic_top_cache_use))
        if not top_cache_ep:
            continue
        top_cache = list(top_cache_ep)

        row0, idx0, genome0 = extract_epoch_leader(top_cache_ep, candidates)
        update_best_solution(
            best,
            row=row0,
            genome=genome0,
            objective_policy=objective_policy,
        )
        candidates, n_new, n_after_prune = maybe_expand_candidate_pool(
            args,
            prepared,
            epoch_idx=ep,
            epoch_generations=epoch_generations,
            top_cache=top_cache_ep,
            row=row0,
            subset_idx=idx0,
            genome=genome0,
            candidates=candidates,
            feature_names=feature_names,
        )
        dynamic_epoch_logs.append(
            build_dynamic_epoch_log(
                epoch_idx=ep,
                generations_this_epoch=int(gens_this),
                duration_sec=float(epoch_run.duration_sec),
                pool_size_before=int(len(problem.candidates)),
                pool_size_after=int(n_after_prune),
                new_terms_added=int(n_new),
                best_row=row0,
            )
        )

    if problem is None or best.row is None or best.genome is None:
        raise RuntimeError("outer search produced empty evaluation cache")

    best_subset_idx = [int(v) for v in best.row.get("subset_idx", [])]
    return SearchStageResult(
        problem=problem,
        outer_sec=float(outer_sec),
        outer_meta=dict(outer_meta),
        resource_budget=dict(resource_budget),
        run=dict(run),
        top_cache=tuple(dict(row) for row in top_cache),
        best_row=dict(best.row),
        best_genome=tuple(best.genome),
        best_k=int(best.k),
        best_decode_meta=build_best_decode_meta(best.row, float(args.ridge_l2)),
        best_subset_idx=tuple(int(v) for v in best_subset_idx),
        dynamic_epoch_logs=tuple(dict(row) for row in dynamic_epoch_logs),
        candidates=tuple(candidates),
    )



def run_outer_search_stage(context: dict[str, Any]) -> SearchStageResult:
    args = ctx_require(context, RuntimeContextKey.ARGS)
    prepared = ctx_require(context, RuntimeContextKey.PREPARED)
    search = run_outer_search(args, prepared)
    ctx_set(context, RuntimeContextKey.SEARCH, search)
    return search


__all__ = ["run_outer_search", "run_outer_search_stage"]
