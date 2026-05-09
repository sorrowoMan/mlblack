from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.common.contracts import ProcessedDataset
from core.trainers.symbolic_orthogonal_trainer import (
    SymbolicOrthogonalSurrogateTrainer,
    SymbolicOrthogonalTrainerConfig,
)
from examples.path_defaults import (
    apply_env_defaults,
    default_outputs_dir,
    default_work_ci_csv_no_flow_speed_occ_lag,
)
from examples.run_work_ci_orthogonal_probe import (
    _apply_imputer,
    _bucket_for_features,
    _df_to_markdown,
    _extract_basis_rows,
    _feature_bucket,
    _feature_set_key,
    _fit_imputer,
    _jsonable,
    _rolling_splits,
    _safe_get,
    _write_json,
)
from pipeline import ZScorePipeline
from project.scaffold import ScaffoldSpec, load_scaffold_spec, run_project_scaffold


DEFAULT_OLD_PARAMS_REPORT = (
    r"C:\Users\hp\Desktop\work\reports"
    r"\symbolic_interval_fixed_cv_rolling_eval_no_flow_speed_occ_lag_bestfeasible.json"
)


FALLBACK_OLD_INTERVAL_PARAMS: dict[str, Any] = {
    "version": "v2",
    "lower_quantile": 0.1253,
    "upper_quantile": 0.9653,
    "v2_continuous_ops": ["identity", "sin", "cos"],
    "v2_binary_ops": ["identity"],
    "v2_include_interactions": True,
    "v2_max_interactions": 17,
    "v2_topk_features": 4,
    "v2_include_hinge": True,
    "v2_hinge_quantiles": [0.25, 0.5, 0.75],
    "order_penalty": 2.683558562271297,
    "width_penalty": 0.0001358224322521336,
    "epochs": 220,
    "batch_size": 128,
    "lr": 0.007022247351848513,
    "weight_decay": 0.0001,
    "l1_readout": 0.0003113270330240271,
    "l1_params": 0.0,
    "device": "cpu",
    "conformal_calibration": True,
    "conformal_level": 0.7315,
    "stagewise_warmup_enabled": False,
    "gate_piecewise_enabled": False,
    "random_seed": 20288323,
}


def _build_feature_cols(df: pd.DataFrame, *, target_col: str, date_col: str) -> list[str]:
    fold_cols = [str(c) for c in df.columns if str(c).startswith("test_fold_")]
    drop = set(fold_cols)
    drop.add(str(target_col))
    if date_col in df.columns:
        drop.add(str(date_col))
    return [str(c) for c in df.columns if str(c) not in drop]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean(np.abs(yp - yt)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    denom = float(np.sum((yt - float(np.mean(yt))) ** 2))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - float(np.sum((yp - yt) ** 2)) / denom)


def _interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> np.ndarray:
    yt = np.asarray(y, dtype=float).reshape(-1)
    l = np.asarray(lo, dtype=float).reshape(-1)
    u = np.asarray(hi, dtype=float).reshape(-1)
    width = np.maximum(u - l, 0.0)
    under = (2.0 / float(alpha)) * (l - yt) * (yt < l)
    over = (2.0 / float(alpha)) * (yt - u) * (yt > u)
    return width + under + over


def _eval_interval_metrics(
    y_true: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    alpha: float,
    y_range: float,
) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    l = np.asarray(lo, dtype=float).reshape(-1)
    u = np.asarray(hi, dtype=float).reshape(-1)
    l2 = np.minimum(l, u)
    u2 = np.maximum(l, u)
    center = 0.5 * (l2 + u2)
    width = np.maximum(u2 - l2, 0.0)
    picp = float(np.mean((yt >= l2) & (yt <= u2)))
    return {
        "rmse": _rmse(yt, center),
        "mae": _mae(yt, center),
        "r2": _r2(yt, center),
        "picp": picp,
        "coverage_error": float(abs(picp - (1.0 - float(alpha)))),
        "mean_width": float(np.mean(width)),
        "pinaw": float(np.mean(width / max(1e-8, float(y_range)))),
        "interval_score": float(np.mean(_interval_score(yt, l2, u2, float(alpha)))),
    }


def _load_old_params(path: str, *, old_epochs: int | None = None) -> dict[str, Any]:
    p = Path(str(path)).expanduser()
    payload: Any = None
    if p.exists():
        payload = json.loads(p.read_text(encoding="utf-8-sig"))
    if isinstance(payload, Mapping) and isinstance(payload.get("params"), Mapping):
        params = dict(payload["params"])
    elif isinstance(payload, Mapping):
        params = dict(payload)
    else:
        params = dict(FALLBACK_OLD_INTERVAL_PARAMS)
    if old_epochs is not None and int(old_epochs) > 0:
        params["epochs"] = int(old_epochs)
    return params


def _save_artifact(artifact: Any, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        artifact.save(str(out_dir))
    except Exception as exc:  # pragma: no cover
        _write_json(out_dir / "artifact_save_error.json", {"error": str(exc), "artifact_type": type(artifact).__name__})


def _fit_old_symbolic_interval(
    *,
    base_spec: ScaffoldSpec,
    csv_path: Path,
    output_dir: Path,
    params: Mapping[str, Any],
    feature_recipe: str,
    split_name: str,
    alpha: float,
) -> tuple[dict[str, Any], dict[str, list[float]], float]:
    data_spec = replace(
        base_spec.data,
        csv_path=str(csv_path),
        split_mode="fold_flag",
        test_fold_col="test_fold_rolling",
        feature_recipe=str(feature_recipe),
    )
    train_spec = replace(
        base_spec.train,
        trainer_key="symbolic_torch_interval",
        trainer_params=dict(params),
        output_dir=str(output_dir),
        run_name=f"old_symbolic_interval_{split_name}",
    )
    t0 = time.perf_counter()
    result = run_project_scaffold(ScaffoldSpec(data=data_spec, train=train_spec))
    duration = float(time.perf_counter() - t0)

    X_test = np.asarray(result.processed.X_test, dtype=float)
    y_test = np.asarray(result.processed.y_test, dtype=float).reshape(-1)
    y_train = np.asarray(result.processed.y_train, dtype=float).reshape(-1)
    y_range = float(np.max(y_train) - np.min(y_train))
    lo, hi = result.artifact.predict_interval(X_test)
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    center = 0.5 * (lo + hi)
    metrics = _eval_interval_metrics(y_test, lo, hi, alpha=float(alpha), y_range=y_range)
    return (
        {
            "duration_sec": duration,
            "train_n": int(y_train.shape[0]),
            "test_n": int(y_test.shape[0]),
            "artifact_type": type(result.artifact).__name__,
            "interval_mode": "symbolic_torch_quantile_conformal",
            **metrics,
        },
        {
            "old_symbolic_lower": [float(v) for v in lo],
            "old_symbolic_upper": [float(v) for v in hi],
            "old_symbolic_center": [float(v) for v in center],
        },
        duration,
    )


def _fit_orthogonal_point_artifact(
    *,
    seed: int,
    feature_names: tuple[str, ...],
    X_train: np.ndarray,
    y_train: np.ndarray,
    enable_gates: bool,
) -> Any:
    feature_set = set(feature_names)
    periodic = tuple(name for name in ("dow_sin", "dow_cos", "doy_sin", "doy_cos") if name in feature_set)
    gate_names = tuple(
        name
        for name in ("ci_lag1", "ci_roll7_prev_mean", "aqi", "wind", "doy_sin")
        if enable_gates and name in feature_set
    )
    config = SymbolicOrthogonalTrainerConfig(
        artifact_id=f"work_ci_orthogonal_interval_center_seed{int(seed)}",
        candidate_limit=120,
        seed_candidate_count=24,
        group_count=12,
        min_basis_count=4,
        max_basis_count=7,
        max_pair_abs_corr=0.72,
        max_feature_reuse=2,
        max_semantic_repeats=1,
        selection_mode="interval_first",
        random_seed=int(seed),
        outer_search_beam_width=10,
        outer_search_branching_factor=3,
        outer_search_max_expansions=72,
        assembler_max_added_terms=4,
        assembler_topk_features=5,
        assembler_max_pair_terms=8,
        assembler_max_candidates_per_iter=96,
        assembler_candidate_keep_top=6,
        assembler_max_expr_depth=6,
        interval_alpha=0.30,
        coverage_error_threshold=0.10,
        enable_piecewise_basis=bool(enable_gates),
        gate_feature_names=gate_names,
        gate_candidate_screen_reserve=1 if enable_gates else 0,
        require_gate_candidate_in_group=bool(enable_gates),
        min_gate_basis_terms=1 if enable_gates else 0,
        periodic_feature_names=periodic,
        require_periodic_candidate_in_group=False,
        min_periodic_basis_terms=0,
        native_trunk_residual_gain_floor=0.015,
        native_trunk_interval_gain_floor=0.0,
        cross_explanatory_rejection_mode="off",
        source_overlap_penalty_mode="feature_overlap+proxy_overlap",
        proxy_group_policy="metadata_or_correlation_cluster",
        environment_invariance_audit_mode="off",
        residual_regime_identification_mode="off",
        regional_correction_basis_mode="off",
        regional_correction_promotion_mode="off",
    )
    train_ds = ProcessedDataset(
        X_train=np.asarray(X_train, dtype=float),
        y_train=np.asarray(y_train, dtype=float).reshape(-1, 1),
        feature_names=feature_names,
        target_names=("ci",),
        metadata={
            "scenario": "work_ci_orthogonal_interval_compare",
            "input_protocol": "processed_dataset",
            "feature_buckets": {name: _feature_bucket(name) for name in feature_names},
            "periodic_feature_names": periodic,
            "gate_feature_names": gate_names,
            "interval_wrapper": "symmetric_conformal_residual",
        },
    )
    trainer = SymbolicOrthogonalSurrogateTrainer(config=config, pipeline=ZScorePipeline())
    return trainer.fit(train_ds)


def _fit_orthogonal_interval(
    *,
    seed: int,
    feature_names: tuple[str, ...],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    conformal_level: float,
    enable_gates: bool,
) -> tuple[dict[str, Any], dict[str, list[float]], Any, float]:
    t0 = time.perf_counter()
    artifact = _fit_orthogonal_point_artifact(
        seed=int(seed),
        feature_names=feature_names,
        X_train=X_train,
        y_train=y_train,
        enable_gates=bool(enable_gates),
    )
    center_train = np.asarray(artifact.predict(np.asarray(X_train, dtype=float)), dtype=float).reshape(-1)
    center_test = np.asarray(artifact.predict(np.asarray(X_test, dtype=float)), dtype=float).reshape(-1)
    abs_resid = np.abs(np.asarray(y_train, dtype=float).reshape(-1) - center_train)
    margin = float(np.quantile(abs_resid, float(np.clip(conformal_level, 0.0, 1.0)))) if abs_resid.size else 0.0
    lo = center_test - margin
    hi = center_test + margin
    duration = float(time.perf_counter() - t0)
    y_range = float(np.max(y_train) - np.min(y_train))
    metrics = _eval_interval_metrics(y_test, lo, hi, alpha=float(alpha), y_range=y_range)
    return (
        {
            "duration_sec": duration,
            "train_n": int(len(y_train)),
            "test_n": int(len(y_test)),
            "artifact_type": type(artifact).__name__,
            "interval_mode": "orthogonal_center_symmetric_train_conformal",
            "conformal_level": float(conformal_level),
            "calibration_margin": float(margin),
            **metrics,
        },
        {
            "orthogonal_lower": [float(v) for v in lo],
            "orthogonal_upper": [float(v) for v in hi],
            "orthogonal_center": [float(v) for v in center_test],
        },
        artifact,
        duration,
    )


def _source_and_lane_reports(
    *,
    basis_df: pd.DataFrame,
    contrib_df: pd.DataFrame,
    split_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if basis_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    total_runs = int(len(set(zip(basis_df["split_name"], basis_df["seed"]))))
    weight_by_run_support = pd.DataFrame()
    if not contrib_df.empty:
        weight_by_run_support = (
            contrib_df.groupby(["split_name", "seed", "feature_set_key"], dropna=False)
            .agg(
                realized_term_count=("term_name", "count"),
                normalized_weight_sum=("normalized_weight", "sum"),
                max_normalized_weight=("normalized_weight", "max"),
            )
            .reset_index()
        )
    basis_aug = basis_df.copy()
    if not weight_by_run_support.empty:
        basis_aug = basis_aug.merge(
            weight_by_run_support,
            how="left",
            on=["split_name", "seed", "feature_set_key"],
        )
    else:
        basis_aug["realized_term_count"] = np.nan
        basis_aug["normalized_weight_sum"] = np.nan
        basis_aug["max_normalized_weight"] = np.nan
    basis_aug["run_key"] = basis_aug["split_name"].astype(str) + "::seed_" + basis_aug["seed"].astype(str)

    source_stability = (
        basis_aug.groupby(["source_object_key", "source_support_key", "feature_set_key", "feature_bucket"], dropna=False)
        .agg(
            selected_count=("object_key", "count"),
            run_support=("run_key", "nunique"),
            split_support=("split_name", "nunique"),
            seed_support=("seed", "nunique"),
            example_expression=("expression", "first"),
            example_features=("feature_names", "first"),
            object_kinds=("object_kind", lambda s: "|".join(sorted(set(str(v) for v in s if str(v))))),
            selection_channels=("selection_channel", lambda s: "|".join(sorted(set(str(v) for v in s if str(v))))),
            structural_channels=("structural_channel", lambda s: "|".join(sorted(set(str(v) for v in s if str(v))))),
            mean_normalized_weight=("normalized_weight_sum", "mean"),
            max_normalized_weight=("max_normalized_weight", "max"),
            mean_realized_term_count=("realized_term_count", "mean"),
        )
        .reset_index()
    )
    source_stability["support_rate"] = source_stability["run_support"].astype(float) / max(1, total_runs)
    source_stability = source_stability.sort_values(
        ["support_rate", "split_support", "mean_normalized_weight", "selected_count"],
        ascending=[False, False, False, False],
    )

    lane_group = basis_aug.copy()
    lane_group["lane_key"] = lane_group["selection_channel"].replace("", "unknown") + "::" + lane_group[
        "structural_channel"
    ].replace("", "unknown")
    lane_survival = (
        lane_group.groupby(["lane_key", "selection_channel", "structural_channel", "feature_bucket"], dropna=False)
        .agg(
            selected_count=("object_key", "count"),
            split_support=("split_name", "nunique"),
            example_features=("feature_names", "first"),
            example_expression=("expression", "first"),
            mean_normalized_weight=("normalized_weight_sum", "mean"),
        )
        .reset_index()
    )
    lane_survival["survival_rate"] = lane_survival["split_support"].astype(float) / max(1, int(split_count))
    lane_survival = lane_survival.sort_values(
        ["survival_rate", "selected_count", "mean_normalized_weight"],
        ascending=[False, False, False],
    )
    return source_stability, lane_survival


def _summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics_df.groupby("model", dropna=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            r2_mean=("r2", "mean"),
            picp_mean=("picp", "mean"),
            coverage_error_mean=("coverage_error", "mean"),
            mean_width_mean=("mean_width", "mean"),
            pinaw_mean=("pinaw", "mean"),
            interval_score_mean=("interval_score", "mean"),
            duration_sec_mean=("duration_sec", "mean"),
        )
        .reset_index()
        .sort_values(["interval_score_mean", "rmse_mean"], ascending=[True, True])
    )


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    apply_env_defaults()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else Path(default_outputs_dir()) / "work_ci_orthogonal_interval_compare" / timestamp
    )
    out_root.mkdir(parents=True, exist_ok=True)

    old_params = _load_old_params(args.old_params_json, old_epochs=args.old_epochs if args.old_epochs > 0 else None)
    conformal_level = (
        float(args.orthogonal_conformal_level)
        if float(args.orthogonal_conformal_level) >= 0.0
        else float(old_params.get("conformal_level", old_params.get("upper_quantile", 0.9) - old_params.get("lower_quantile", 0.1)))
    )
    conformal_level = float(np.clip(conformal_level, 0.0, 1.0))

    df = pd.read_csv(csv_path)
    if str(args.date_col) in df.columns:
        df[str(args.date_col)] = pd.to_datetime(df[str(args.date_col)])
        df = df.sort_values(str(args.date_col)).reset_index(drop=True)

    feature_names = tuple(_build_feature_cols(df, target_col=str(args.target_col), date_col=str(args.date_col)))
    X_all = df.loc[:, feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_all = pd.to_numeric(df[str(args.target_col)], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_all)
    if not np.all(valid):
        df = df.loc[valid].reset_index(drop=True)
        X_all = X_all[valid]
        y_all = y_all[valid]

    max_splits = None if int(args.rolling_splits) <= 0 else int(args.rolling_splits)
    splits = _rolling_splits(
        int(len(y_all)),
        min_train_size=int(args.min_train_size),
        test_size=int(args.test_size),
        step_size=int(args.step_size),
        max_splits=max_splits,
    )
    seeds = tuple(int(v) for v in str(args.seeds).replace(";", ",").split(",") if str(v).strip())
    base_spec = load_scaffold_spec(str(args.config))

    tmp_dir = out_root / "rolling_tmp_csv"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        out_root / "run_config.json",
        {
            "created_at": timestamp,
            "csv": str(csv_path),
            "config": str(Path(args.config).expanduser().resolve()),
            "old_params_json": str(Path(args.old_params_json).expanduser()),
            "old_params": old_params,
            "orthogonal_conformal_level": float(conformal_level),
            "alpha": float(args.alpha),
            "n_rows": int(len(y_all)),
            "feature_count": int(len(feature_names)),
            "feature_names": list(feature_names),
            "splits": [
                {
                    k: (int(v) if isinstance(v, (int, np.integer)) else str(v))
                    for k, v in split.items()
                    if k not in {"train_idx", "test_idx"}
                }
                for split in splits
            ],
            "seeds": list(seeds),
            "enable_gates": bool(args.enable_gates),
        },
    )

    metrics_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []

    for split in splits:
        split_name = str(split["split_name"])
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        split_dir = out_root / "splits" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        X_train_raw = X_all[train_idx]
        X_test_raw = X_all[test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        fill = _fit_imputer(X_train_raw)
        X_train = _apply_imputer(X_train_raw, fill)
        X_test = _apply_imputer(X_test_raw, fill)

        # The old pipeline is scaffold-based, so it consumes a fold-flag CSV.
        dfx = df.iloc[0 : int(test_idx[-1]) + 1].copy().reset_index(drop=True)
        dfx["test_fold_rolling"] = 0
        dfx.loc[int(test_idx[0]) : int(test_idx[-1]), "test_fold_rolling"] = 1
        tmp_csv = tmp_dir / f"{split_name}.csv"
        dfx.to_csv(tmp_csv, index=False)

        test_dates = {
            "train_start": str(df.iloc[int(train_idx[0])][str(args.date_col)]) if str(args.date_col) in df.columns else "",
            "train_end": str(df.iloc[int(train_idx[-1])][str(args.date_col)]) if str(args.date_col) in df.columns else "",
            "test_start": str(df.iloc[int(test_idx[0])][str(args.date_col)]) if str(args.date_col) in df.columns else "",
            "test_end": str(df.iloc[int(test_idx[-1])][str(args.date_col)]) if str(args.date_col) in df.columns else "",
        }

        old_row, old_preds, _ = _fit_old_symbolic_interval(
            base_spec=base_spec,
            csv_path=tmp_csv,
            output_dir=split_dir / "old_symbolic_interval",
            params=old_params,
            feature_recipe=str(args.feature_recipe),
            split_name=split_name,
            alpha=float(args.alpha),
        )
        metrics_rows.append(
            {
                "model": "old_symbolic_interval",
                "split_name": split_name,
                "split_id": int(split["split_id"]),
                "seed": int(old_params.get("random_seed", -1)),
                **test_dates,
                **old_row,
            }
        )

        for seed in seeds:
            seed_dir = split_dir / f"seed_{int(seed)}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            ortho_row, ortho_preds, artifact, _ = _fit_orthogonal_interval(
                seed=int(seed),
                feature_names=feature_names,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                alpha=float(args.alpha),
                conformal_level=float(conformal_level),
                enable_gates=bool(args.enable_gates),
            )
            _save_artifact(artifact, seed_dir / "orthogonal_center_artifact")
            rows, contribs, schema = _extract_basis_rows(artifact, split_name=split_name, seed=int(seed))
            basis_rows.extend(rows)
            contribution_rows.extend(contribs)
            _write_json(seed_dir / "orthogonal_schema_excerpt.json", schema)
            metrics_rows.append(
                {
                    "model": "orthogonal_interval",
                    "split_name": split_name,
                    "split_id": int(split["split_id"]),
                    "seed": int(seed),
                    **test_dates,
                    **ortho_row,
                }
            )
            pred_df = pd.DataFrame(
                {
                    "y_true": [float(v) for v in y_test],
                    **old_preds,
                    **ortho_preds,
                }
            )
            pred_df.to_csv(seed_dir / "interval_predictions.csv", index=False, encoding="utf-8-sig")

    metrics_df = pd.DataFrame(metrics_rows)
    basis_df = pd.DataFrame(basis_rows)
    contrib_df = pd.DataFrame(contribution_rows)
    source_stability, lane_survival = _source_and_lane_reports(
        basis_df=basis_df,
        contrib_df=contrib_df,
        split_count=int(len(splits)),
    )
    metric_summary = _summarize_metrics(metrics_df)

    metrics_df.to_csv(out_root / "rolling_metrics.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(out_root / "interval_metrics.csv", index=False, encoding="utf-8-sig")
    metric_summary.to_csv(out_root / "interval_metrics_summary.csv", index=False, encoding="utf-8-sig")
    basis_df.to_csv(out_root / "basis_terms_long.csv", index=False, encoding="utf-8-sig")
    contrib_df.to_csv(out_root / "realized_terms_long.csv", index=False, encoding="utf-8-sig")
    source_stability.to_csv(out_root / "source_stability.csv", index=False, encoding="utf-8-sig")
    lane_survival.to_csv(out_root / "lane_survival.csv", index=False, encoding="utf-8-sig")

    _write_json(out_root / "rolling_metrics.json", metrics_df.to_dict(orient="records"))
    _write_json(out_root / "interval_metrics.json", metrics_df.to_dict(orient="records"))
    _write_json(out_root / "interval_metrics_summary.json", metric_summary.to_dict(orient="records"))
    _write_json(out_root / "source_stability.json", source_stability.to_dict(orient="records"))
    _write_json(out_root / "lane_survival.json", lane_survival.to_dict(orient="records"))

    report_md = "\n".join(
        [
            "# work_ci symbolic interval comparison",
            "",
            f"- data: `{csv_path}`",
            f"- output: `{out_root}`",
            f"- rows: {len(y_all)}",
            f"- features: {len(feature_names)}",
            f"- rolling splits: {len(splits)}",
            f"- orthogonal seeds: {', '.join(str(v) for v in seeds)}",
            f"- alpha for interval score: {float(args.alpha):.4f}",
            f"- orthogonal interval mode: `orthogonal_center_symmetric_train_conformal`",
            f"- orthogonal conformal level: {float(conformal_level):.4f}",
            "",
            "## Interval Metrics Summary",
            "",
            _df_to_markdown(metric_summary),
            "",
            "## Rolling / Interval Metrics Long",
            "",
            _df_to_markdown(metrics_df.sort_values(["split_id", "model", "seed"]), max_rows=60),
            "",
            "## Orthogonal Source Stability",
            "",
            _df_to_markdown(source_stability.head(30) if not source_stability.empty else source_stability),
            "",
            "## Orthogonal Lane Survival",
            "",
            _df_to_markdown(lane_survival.head(30) if not lane_survival.empty else lane_survival),
            "",
            "Note: old symbolic interval uses the official symbolic_torch_interval quantile head. "
            "Orthogonal interval currently uses the orthogonal symbolic point artifact as center plus a symmetric "
            "train-residual conformal interval shell, so interval-head architecture is not identical yet.",
        ]
    )
    (out_root / "REPORT.md").write_text(report_md, encoding="utf-8")

    summary = {
        "out_root": str(out_root),
        "csv": str(csv_path),
        "split_count": int(len(splits)),
        "feature_count": int(len(feature_names)),
        "metric_summary": metric_summary.to_dict(orient="records"),
        "top_sources": source_stability.head(12).to_dict(orient="records") if not source_stability.empty else [],
        "top_lanes": lane_survival.head(12).to_dict(orient="records") if not lane_survival.empty else [],
    }
    _write_json(out_root / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare old work_ci symbolic interval against orthogonal interval.")
    parser.add_argument("--config", default=str(ROOT / "examples" / "configs" / "work_ci_symbolic_torch_interval.json"))
    parser.add_argument("--csv", default=default_work_ci_csv_no_flow_speed_occ_lag())
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--old-params-json", default=DEFAULT_OLD_PARAMS_REPORT)
    parser.add_argument("--old-epochs", type=int, default=0, help="Optional smoke override; 0 keeps formal params.")
    parser.add_argument("--feature-recipe", default="raw_all_numeric")
    parser.add_argument("--target-col", default="ci")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--rolling-splits", type=int, default=0, help="0 means all possible rolling splits.")
    parser.add_argument("--min-train-size", type=int, default=960)
    parser.add_argument("--test-size", type=int, default=120)
    parser.add_argument("--step-size", type=int, default=120)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--alpha", type=float, default=0.30)
    parser.add_argument("--orthogonal-conformal-level", type=float, default=-1.0)
    parser.add_argument("--enable-gates", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = run_compare(args)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
