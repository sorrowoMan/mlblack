from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

JsonableFn = Callable[[Any], Any]


def build_interval_subset_report(
    *,
    subset_idx: Sequence[int],
    subset_candidates: Sequence[Mapping[str, Any]],
    fold_results: Sequence[Mapping[str, Any]],
    decode_meta: Mapping[str, Any],
    selection_coverage_error_threshold: float,
    objective_schema: Sequence[str] = ("coverage_error", "pinaw", "interval_score"),
    jsonable_fn: JsonableFn | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    cov_arr = np.asarray([float(row["coverage_error"]) for row in fold_results], dtype=float)
    pinaw_arr = np.asarray([float(row["pinaw"]) for row in fold_results], dtype=float)
    is_arr = np.asarray([float(row["interval_score"]) for row in fold_results], dtype=float)
    picp_arr = np.asarray([float(row["picp"]) for row in fold_results], dtype=float)
    width_arr = np.asarray([float(row["mean_width"]) for row in fold_results], dtype=float)
    rm_arr = np.asarray([float(row["rmse"]) for row in fold_results], dtype=float)
    fold_branch = [dict(row.get("branch_detail", {})) for row in fold_results]
    fold_interval_info = [dict(row.get("interval_info", {})) for row in fold_results]

    coverage_error_mean = float(np.mean(cov_arr))
    pinaw_mean = float(np.mean(pinaw_arr))
    interval_score_mean = float(np.mean(is_arr))
    picp_mean = float(np.mean(picp_arr))
    mean_width_mean = float(np.mean(width_arr))
    rmse_mean = float(np.mean(rm_arr))
    rmse_std = float(np.std(rm_arr))
    rmse_drift = float(np.mean(np.abs(np.diff(rm_arr)))) if rm_arr.size >= 2 else 0.0

    subset_ids = [int(v) for v in subset_idx]
    subset_names = [str(row.get("name", f"term_{i}")) for i, row in enumerate(subset_candidates)]
    subset_families = [str(row.get("family", "")) for row in subset_candidates]
    complexity = float(sum(float(row.get("complexity", 0.0)) for row in subset_candidates))

    fam_counts: dict[str, int] = {}
    feat_counts: dict[int, int] = {}
    for row in subset_candidates:
        fam = str(row.get("family", ""))
        fam_counts[fam] = int(fam_counts.get(fam, 0) + 1)
        for feat in row.get("features", ()):
            feat_counts[int(feat)] = int(feat_counts.get(int(feat), 0) + 1)
    fam_share = np.asarray([float(v) for v in fam_counts.values()], dtype=float)
    if fam_share.size > 0:
        fam_share = fam_share / float(np.sum(fam_share))
    feat_share = np.asarray([float(v) for v in feat_counts.values()], dtype=float)
    if feat_share.size > 0:
        feat_share = feat_share / float(np.sum(feat_share))
    fam_concentration = float(np.sum(fam_share**2)) if fam_share.size > 0 else 1.0
    feat_concentration = float(np.sum(feat_share**2)) if feat_share.size > 0 else 1.0

    meta = dict(decode_meta)
    complexity_scale = float(max(0.05, meta.get("complexity_scale", 1.0)))
    family_penalty_scale = float(max(0.05, meta.get("family_penalty_scale", 1.0)))
    feature_penalty_scale = float(max(0.05, meta.get("feature_penalty_scale", 1.0)))
    drift_weight = float(max(0.0, meta.get("drift_weight", 0.15)))
    tuned_l2 = float(max(0.0, meta.get("tuned_l2", 0.0)))
    strict4_ratio = float(np.clip(meta.get("strict4_min_train_ratio", 0.08), 0.01, 0.30))

    obj = np.asarray([coverage_error_mean, pinaw_mean, interval_score_mean], dtype=float)
    jsonify = jsonable_fn or (lambda x: x)
    detail = {
        "objective_schema": [str(v) for v in objective_schema],
        "selection_coverage_error_threshold": float(selection_coverage_error_threshold),
        "selection_meets_coverage_threshold": bool(
            coverage_error_mean <= float(selection_coverage_error_threshold)
        ),
        "subset_size": int(len(subset_ids)),
        "subset_idx": subset_ids,
        "subset_names": subset_names,
        "subset_families": subset_families,
        "fold_coverage_error": [float(v) for v in cov_arr.tolist()],
        "fold_pinaw": [float(v) for v in pinaw_arr.tolist()],
        "fold_interval_score": [float(v) for v in is_arr.tolist()],
        "fold_picp": [float(v) for v in picp_arr.tolist()],
        "fold_mean_width": [float(v) for v in width_arr.tolist()],
        "fold_rmse": [float(v) for v in rm_arr.tolist()],
        "fold_branch_detail": jsonify(fold_branch),
        "fold_interval_info": jsonify(fold_interval_info),
        "coverage_error_mean": float(coverage_error_mean),
        "pinaw_mean": float(pinaw_mean),
        "interval_score_mean": float(interval_score_mean),
        "picp_mean": float(picp_mean),
        "mean_width_mean": float(mean_width_mean),
        "rmse_mean": float(rmse_mean),
        "rmse_std": float(rmse_std),
        "rmse_drift": float(rmse_drift),
        "complexity_raw": float(complexity),
        "family_concentration": float(fam_concentration),
        "feature_concentration": float(feat_concentration),
        "tuned_l2": float(tuned_l2),
        "strict4_min_train_ratio": float(strict4_ratio),
        "complexity_scale": float(complexity_scale),
        "family_penalty_scale": float(family_penalty_scale),
        "feature_penalty_scale": float(feature_penalty_scale),
        "drift_weight": float(drift_weight),
        "decode_meta": jsonify(meta),
    }
    return obj, detail


__all__ = ["build_interval_subset_report"]
