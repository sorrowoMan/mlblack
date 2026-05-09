from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from bias import build_interval_row_objective_key
from model.interval_fit import _jsonable
from pipeline.feature_space import CandidateTerm


@dataclass
class BestSolutionTracker:
    row: dict[str, Any] | None = None
    genome: list[dict[str, Any]] | None = None
    k: int = 0


def extract_epoch_leader(
    top_cache: Sequence[Mapping[str, Any]],
    candidates: Sequence[CandidateTerm],
) -> tuple[dict[str, Any], list[int], list[dict[str, Any]]]:
    row0 = dict(top_cache[0])
    idx0 = [int(v) for v in row0.get("subset_idx", [])]
    genome0 = [{"name": candidates[i].name, "expr": dict(candidates[i].expr)} for i in idx0]
    return row0, idx0, genome0


def update_best_solution(
    tracker: BestSolutionTracker,
    *,
    row: Mapping[str, Any],
    genome: Sequence[Mapping[str, Any]],
    objective_policy: object,
) -> None:
    row_key = build_interval_row_objective_key(row, cfg=objective_policy)
    best_key = None if tracker.row is None else build_interval_row_objective_key(tracker.row, cfg=objective_policy)
    if best_key is None or row_key < best_key:
        tracker.row = dict(row)
        tracker.genome = [dict(item) for item in genome]
        tracker.k = int(row.get("subset_size", len(genome)))


def build_best_decode_meta(best_row: Mapping[str, Any], ridge_l2: float) -> dict[str, Any]:
    return {
        "objective_schema": ["coverage_error", "pinaw", "interval_score"],
        "coverage_error_mean": float(best_row.get("coverage_error_mean", float("inf"))),
        "pinaw_mean": float(best_row.get("pinaw_mean", float("inf"))),
        "interval_score_mean": float(best_row.get("interval_score_mean", float("inf"))),
        "picp_mean": float(best_row.get("picp_mean", float("nan"))),
        "mean_width_mean": float(best_row.get("mean_width_mean", float("nan"))),
        "rmse_mean": float(best_row.get("rmse_mean", float("inf"))),
        "rmse_std": float(best_row.get("rmse_std", float("inf"))),
        "obj_coverage_error": float(best_row.get("obj_coverage_error", float("inf"))),
        "obj_pinaw": float(best_row.get("obj_pinaw", float("inf"))),
        "obj_interval_score": float(best_row.get("obj_interval_score", float("inf"))),
        "decode_meta": _jsonable(best_row.get("decode_meta", {})),
        "tuned_l2": float(best_row.get("tuned_l2", max(0.0, ridge_l2))),
        "strict4_min_train_ratio": float(best_row.get("strict4_min_train_ratio", 0.08)),
    }


def build_dynamic_epoch_log(
    *,
    epoch_idx: int,
    generations_this_epoch: int,
    duration_sec: float,
    pool_size_before: int,
    pool_size_after: int,
    new_terms_added: int,
    best_row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch_idx + 1),
        "generations": int(generations_this_epoch),
        "duration_sec": float(duration_sec),
        "pool_size_before": int(pool_size_before),
        "pool_size_after": int(pool_size_after),
        "new_terms_added": int(new_terms_added),
        "best_obj_coverage_error": float(best_row.get("obj_coverage_error", float("inf"))),
        "best_obj_pinaw": float(best_row.get("obj_pinaw", float("inf"))),
        "best_obj_interval_score": float(best_row.get("obj_interval_score", float("inf"))),
        "best_subset_size": int(best_row.get("subset_size", 0)),
    }


__all__ = [
    "BestSolutionTracker",
    "build_best_decode_meta",
    "build_dynamic_epoch_log",
    "extract_epoch_leader",
    "update_best_solution",
]
