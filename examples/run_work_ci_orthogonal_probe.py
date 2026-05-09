from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
from core.trainers.symbolic_stagewise_trainer import (
    SymbolicStagewiseSurrogateTrainer,
    SymbolicStagewiseTrainerConfig,
)
from examples.path_defaults import (
    apply_env_defaults,
    default_outputs_dir,
    default_work_ci_csv_no_flow_speed_occ_lag,
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in tuple(value)]
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


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


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae": _mae(y_true, y_pred),
        "r2": _r2(y_true, y_pred),
    }


def _summarize(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _build_feature_cols(df: pd.DataFrame, *, target_col: str, date_col: str) -> list[str]:
    fold_cols = [str(c) for c in df.columns if str(c).startswith("test_fold_")]
    drop = set(fold_cols)
    drop.add(str(target_col))
    if date_col in df.columns:
        drop.add(str(date_col))
    return [str(c) for c in df.columns if str(c) not in drop]


def _rolling_splits(
    n_rows: int,
    *,
    min_train_size: int,
    test_size: int,
    step_size: int,
    max_splits: int | None,
) -> list[dict[str, Any]]:
    splits: list[dict[str, Any]] = []
    train_end = int(min_train_size)
    split_id = 0
    while train_end + int(test_size) <= int(n_rows):
        splits.append(
            {
                "scope": "rolling",
                "split_id": int(split_id),
                "split_name": f"rolling_{split_id:02d}",
                "train_idx": np.arange(0, train_end, dtype=int),
                "test_idx": np.arange(train_end, train_end + int(test_size), dtype=int),
                "train_start": 0,
                "train_end": int(train_end),
                "test_start": int(train_end),
                "test_end": int(train_end + int(test_size)),
            }
        )
        split_id += 1
        if max_splits is not None and split_id >= int(max_splits):
            break
        train_end += int(step_size)
    if not splits:
        raise ValueError("No rolling splits were created; check min_train_size/test_size")
    return splits


def _fit_imputer(X_train: np.ndarray) -> np.ndarray:
    med = np.nanmedian(np.asarray(X_train, dtype=float), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return med.astype(float)


def _apply_imputer(X: np.ndarray, fill: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=float).copy()
    mask = ~np.isfinite(arr)
    if np.any(mask):
        arr[mask] = np.take(fill, np.where(mask)[1])
    return arr


def _fit_ridge_predict(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_mean = np.mean(X_train, axis=0)
    x_std = np.std(X_train, axis=0) + 1e-8
    Z = (X_train - x_mean) / x_std
    Zt = (X_test - x_mean) / x_std
    A = np.column_stack([np.ones(Z.shape[0]), Z])
    At = np.column_stack([np.ones(Zt.shape[0]), Zt])
    penalty = np.eye(A.shape[1], dtype=float) * float(alpha)
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(A.T @ A + penalty, A.T @ np.asarray(y_train, dtype=float).reshape(-1))
    return np.asarray(At @ coef, dtype=float).reshape(-1)


def _feature_bucket(name: str) -> str:
    key = str(name)
    if key.startswith(("ci_lag", "ci_roll", "ci_diff")):
        return "temporal_memory"
    if key in {"dow", "month", "dow_sin", "dow_cos", "doy_sin", "doy_cos"}:
        return "calendar_periodic"
    if key in {"weather_dummy", "wind", "aqi", "is_bad_weather", "is_aqi_high"}:
        return "weather_aqi"
    if key.startswith("is_holiday") or key in {"is_nonwork_weekend", "life_impact"}:
        return "holiday_event"
    return "other"


def _bucket_for_features(features: Sequence[str]) -> str:
    buckets = sorted({_feature_bucket(str(name)) for name in tuple(features) if str(name)})
    if not buckets:
        return "unknown"
    if len(buckets) == 1:
        return buckets[0]
    return "mixed:" + "+".join(buckets)


def _feature_set_key(features: Sequence[str]) -> str:
    clean = sorted(str(name) for name in tuple(features) if str(name))
    return "|".join(clean) if clean else "unknown"


def _safe_get(mapping: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _extract_term_contributions(schema: Mapping[str, Any], *, split_name: str, seed: int) -> list[dict[str, Any]]:
    raw = schema.get("term_contributions", {})
    rows: list[dict[str, Any]] = []
    if isinstance(raw, Mapping):
        iterator = raw.items()
    else:
        iterator = (("target", raw),)
    for target, terms in iterator:
        if not isinstance(terms, Sequence) or isinstance(terms, (str, bytes, bytearray)):
            continue
        for term in tuple(terms):
            if not isinstance(term, Mapping):
                continue
            features = tuple(str(v) for v in tuple(term.get("feature_names", ()) or ()) if str(v))
            rows.append(
                {
                    "split_name": split_name,
                    "seed": int(seed),
                    "target": str(target),
                    "term_name": str(term.get("term_name") or ""),
                    "expression_named": str(term.get("expression_named") or term.get("expression_raw") or ""),
                    "feature_names": "|".join(features),
                    "feature_set_key": _feature_set_key(features),
                    "feature_bucket": _bucket_for_features(features),
                    "coefficient": term.get("coefficient"),
                    "abs_coefficient": term.get("abs_coefficient"),
                    "normalized_weight": term.get("normalized_weight"),
                    "node_count": term.get("node_count"),
                    "depth": term.get("depth"),
                    "operator_cost": term.get("operator_cost"),
                    "interaction_order": term.get("interaction_order"),
                    "unary_op_count": term.get("unary_op_count"),
                    "binary_op_count": term.get("binary_op_count"),
                }
            )
    return rows


def _extract_basis_rows(
    artifact: Any,
    *,
    split_name: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Mapping[str, Any]]:
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    schema = dict(
        metadata.get("orthogonal_basis_artifact_schema")
        or metadata.get("symbolic_artifact_schema", {})
        or {}
    )
    symbolic = dict(
        metadata.get("orthogonal_basis_symbolic")
        or metadata.get("symbolic", {})
        or {}
    )
    if not schema:
        schema = {"basis_structure": {"basis_context": symbolic.get("basis_context", {})}}

    selected_basis = _safe_get(schema, ("basis_structure", "basis_context", "selected_basis"))
    if not isinstance(selected_basis, Sequence) or isinstance(selected_basis, (str, bytes, bytearray)):
        selected_basis = symbolic.get("basis_context", {}).get("selected_basis", ())
    if not isinstance(selected_basis, Sequence) or isinstance(selected_basis, (str, bytes, bytearray)):
        selected_basis = ()

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(tuple(selected_basis)):
        if not isinstance(item, Mapping):
            continue
        meta = dict(item.get("metadata", {}) or {})
        features = tuple(
            str(v)
            for v in tuple(item.get("source_features") or meta.get("feature_names") or ())
            if str(v)
        )
        source_key = str(meta.get("source_object_key") or meta.get("source_information_key") or item.get("object_key") or "")
        support_key = str(meta.get("source_support_key") or _feature_set_key(features))
        rows.append(
            {
                "split_name": split_name,
                "seed": int(seed),
                "basis_index": int(index),
                "object_key": str(item.get("object_key") or ""),
                "expression": str(item.get("expression") or meta.get("selected_evidence_expression") or ""),
                "family_ref": str(item.get("family_ref") or ""),
                "feature_names": "|".join(features),
                "feature_set_key": _feature_set_key(features),
                "feature_bucket": _bucket_for_features(features),
                "source_object_key": source_key,
                "source_support_key": support_key,
                "source_support_size": meta.get("source_support_size"),
                "object_kind": str(meta.get("object_kind") or ""),
                "object_role": str(meta.get("object_role") or ""),
                "selection_channel": str(meta.get("selection_channel") or ""),
                "structural_channel": str(meta.get("structural_channel") or ""),
                "chart_signature": str(meta.get("chart_signature") or ""),
                "realization_signature": str(meta.get("realization_signature") or meta.get("realization_head_signature") or ""),
                "required_realization_family": str(meta.get("required_realization_family") or ""),
                "global_uniform_candidate": bool(meta.get("global_uniform_candidate", False)),
                "modulated_branch_candidate": bool(meta.get("modulated_branch_candidate", False)),
                "uses_piecewise_gate": bool(meta.get("uses_piecewise_gate", False)),
                "rational_template_pinned": bool(meta.get("rational_template_pinned", False)),
                "canonical_trunk_tagged": bool(meta.get("canonical_trunk_tagged", False)),
                "support_expansion_tagged": bool(meta.get("support_expansion_tagged", False)),
                "same_source_surrogate_tagged": bool(meta.get("same_source_surrogate_tagged", False)),
            }
        )

    contribution_rows = _extract_term_contributions(schema, split_name=split_name, seed=seed)
    return rows, contribution_rows, schema


def _df_to_markdown(df: pd.DataFrame, *, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(int(max_rows)).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    headers = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in view.columns) + " |")
    return "\n".join(lines)


def _fit_stagewise(
    *,
    seed: int,
    feature_names: tuple[str, ...],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, Any, float]:
    trainer = SymbolicStagewiseSurrogateTrainer(
        config=SymbolicStagewiseTrainerConfig(
            artifact_id=f"work_ci_stagewise_seed{int(seed)}",
            force_linear_base="auto",
            keep_search_trace=False,
            auto_val_ratio=0.2,
            auto_min_val_samples=64,
            auto_random_seed=int(seed),
            search_max_added_terms=5,
            search_topk_features=min(6, int(len(feature_names))),
            search_max_pair_terms=6,
            search_max_candidates_per_iter=120,
            search_candidate_keep_top=6,
            search_enable_prune=True,
            search_prune_rmse_tolerance=1e-6,
            search_prune_max_removed_per_iter=1,
            search_path_memory_enabled=False,
            search_graph_cache_enabled=True,
            search_graph_cache_backend="memory",
            search_inner_opt_enabled=False,
        )
    )
    train_ds = ProcessedDataset(
        X_train=np.asarray(X_train, dtype=float),
        y_train=np.asarray(y_train, dtype=float).reshape(-1, 1),
        feature_names=feature_names,
        target_names=("ci",),
        metadata={"scenario": "work_ci_orthogonal_probe", "trainer_role": "stagewise_baseline"},
    )
    t0 = time.perf_counter()
    artifact = trainer.fit(train_ds)
    pred = np.asarray(artifact.predict(np.asarray(X_test, dtype=float)), dtype=float).reshape(-1)
    return pred, artifact, float(time.perf_counter() - t0)


def _fit_orthogonal(
    *,
    seed: int,
    feature_names: tuple[str, ...],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    enable_gates: bool,
) -> tuple[np.ndarray, Any, float]:
    periodic = tuple(
        name
        for name in ("dow_sin", "dow_cos", "doy_sin", "doy_cos")
        if name in set(feature_names)
    )
    gate_names = tuple(
        name
        for name in ("ci_lag1", "ci_roll7_prev_mean", "aqi", "wind", "doy_sin")
        if enable_gates and name in set(feature_names)
    )
    config = SymbolicOrthogonalTrainerConfig(
        artifact_id=f"work_ci_symbolic_orthogonal_seed{int(seed)}",
        candidate_limit=120,
        seed_candidate_count=24,
        group_count=12,
        min_basis_count=4,
        max_basis_count=7,
        max_pair_abs_corr=0.72,
        max_feature_reuse=2,
        max_semantic_repeats=1,
        selection_mode="rmse_first",
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
    metadata = {
        "scenario": "work_ci_orthogonal_probe",
        "input_protocol": "processed_dataset",
        "feature_buckets": {name: _feature_bucket(name) for name in feature_names},
        "periodic_feature_names": periodic,
        "gate_feature_names": gate_names,
        "probe_note": "No truth contract is available; reports audit stability and rolling generalization.",
    }
    train_ds = ProcessedDataset(
        X_train=np.asarray(X_train, dtype=float),
        y_train=np.asarray(y_train, dtype=float).reshape(-1, 1),
        feature_names=feature_names,
        target_names=("ci",),
        metadata=metadata,
    )
    trainer = SymbolicOrthogonalSurrogateTrainer(config=config)
    t0 = time.perf_counter()
    artifact = trainer.fit(train_ds)
    pred = np.asarray(artifact.predict(np.asarray(X_test, dtype=float)), dtype=float).reshape(-1)
    return pred, artifact, float(time.perf_counter() - t0)


def _save_artifact(artifact: Any, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        artifact.save(str(out_dir))
    except Exception as exc:  # pragma: no cover - report should survive artifact-save issues.
        _write_json(out_dir / "artifact_save_error.json", {"error": str(exc), "artifact_type": type(artifact).__name__})


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    apply_env_defaults()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir).expanduser().resolve() if args.out_dir else Path(default_outputs_dir()) / "work_ci_orthogonal_probe" / timestamp
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if str(args.date_col) in df.columns:
        df[str(args.date_col)] = pd.to_datetime(df[str(args.date_col)])
        df = df.sort_values(str(args.date_col)).reset_index(drop=True)
    feature_names = tuple(_build_feature_cols(df, target_col=str(args.target_col), date_col=str(args.date_col)))
    if str(args.target_col) not in df.columns:
        raise ValueError(f"Missing target column: {args.target_col}")
    if "ci_lag1" not in feature_names:
        raise ValueError("Expected ci_lag1 in strict-lag work_ci feature set")

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
    models = tuple(str(v).strip() for v in str(args.models).split(",") if str(v).strip())

    config_payload = {
        "created_at": timestamp,
        "csv": str(csv_path),
        "out_root": str(out_root),
        "n_rows": int(len(y_all)),
        "feature_count": int(len(feature_names)),
        "feature_names": list(feature_names),
        "target_col": str(args.target_col),
        "date_col": str(args.date_col),
        "splits": [
            {
                k: (int(v) if isinstance(v, (int, np.integer)) else str(v))
                for k, v in split.items()
                if k not in {"train_idx", "test_idx"}
            }
            for split in splits
        ],
        "seeds": list(seeds),
        "models": list(models),
        "enable_gates": bool(args.enable_gates),
    }
    _write_json(out_root / "run_config.json", config_payload)

    metric_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []

    lag1_index = feature_names.index("ci_lag1")

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

        date_train_start = str(df.iloc[int(train_idx[0])][str(args.date_col)]) if str(args.date_col) in df.columns else ""
        date_train_end = str(df.iloc[int(train_idx[-1])][str(args.date_col)]) if str(args.date_col) in df.columns else ""
        date_test_start = str(df.iloc[int(test_idx[0])][str(args.date_col)]) if str(args.date_col) in df.columns else ""
        date_test_end = str(df.iloc[int(test_idx[-1])][str(args.date_col)]) if str(args.date_col) in df.columns else ""

        for seed in seeds:
            seed_dir = split_dir / f"seed_{int(seed)}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            predictions: dict[str, list[float]] = {"y_true": [float(v) for v in y_test]}

            if "lag1" in models:
                t0 = time.perf_counter()
                pred = np.asarray(X_test[:, lag1_index], dtype=float).reshape(-1)
                duration = float(time.perf_counter() - t0)
                row = {
                    "model": "lag1",
                    "split_name": split_name,
                    "split_id": int(split["split_id"]),
                    "seed": int(seed),
                    "train_n": int(len(train_idx)),
                    "test_n": int(len(test_idx)),
                    "train_start": date_train_start,
                    "train_end": date_train_end,
                    "test_start": date_test_start,
                    "test_end": date_test_end,
                    "duration_sec": duration,
                    **_metric_row(y_test, pred),
                }
                metric_rows.append(row)
                predictions["lag1"] = [float(v) for v in pred]

            if "ridge" in models:
                t0 = time.perf_counter()
                pred = _fit_ridge_predict(X_train=X_train, y_train=y_train, X_test=X_test, alpha=float(args.ridge_alpha))
                duration = float(time.perf_counter() - t0)
                row = {
                    "model": "ridge",
                    "split_name": split_name,
                    "split_id": int(split["split_id"]),
                    "seed": int(seed),
                    "train_n": int(len(train_idx)),
                    "test_n": int(len(test_idx)),
                    "train_start": date_train_start,
                    "train_end": date_train_end,
                    "test_start": date_test_start,
                    "test_end": date_test_end,
                    "duration_sec": duration,
                    **_metric_row(y_test, pred),
                }
                metric_rows.append(row)
                predictions["ridge"] = [float(v) for v in pred]

            if "stagewise" in models:
                pred, artifact, duration = _fit_stagewise(
                    seed=seed,
                    feature_names=feature_names,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                )
                _save_artifact(artifact, seed_dir / "stagewise_artifact")
                metric_rows.append(
                    {
                        "model": "stagewise",
                        "split_name": split_name,
                        "split_id": int(split["split_id"]),
                        "seed": int(seed),
                        "train_n": int(len(train_idx)),
                        "test_n": int(len(test_idx)),
                        "train_start": date_train_start,
                        "train_end": date_train_end,
                        "test_start": date_test_start,
                        "test_end": date_test_end,
                        "duration_sec": duration,
                        **_metric_row(y_test, pred),
                    }
                )
                predictions["stagewise"] = [float(v) for v in pred]

            if "orthogonal" in models:
                pred, artifact, duration = _fit_orthogonal(
                    seed=seed,
                    feature_names=feature_names,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    enable_gates=bool(args.enable_gates),
                )
                artifact_dir = seed_dir / "orthogonal_artifact"
                _save_artifact(artifact, artifact_dir)
                rows, contribs, schema = _extract_basis_rows(artifact, split_name=split_name, seed=seed)
                basis_rows.extend(rows)
                contribution_rows.extend(contribs)
                _write_json(seed_dir / "orthogonal_schema_excerpt.json", schema)
                metric_rows.append(
                    {
                        "model": "orthogonal",
                        "split_name": split_name,
                        "split_id": int(split["split_id"]),
                        "seed": int(seed),
                        "train_n": int(len(train_idx)),
                        "test_n": int(len(test_idx)),
                        "train_start": date_train_start,
                        "train_end": date_train_end,
                        "test_start": date_test_start,
                        "test_end": date_test_end,
                        "duration_sec": duration,
                        **_metric_row(y_test, pred),
                    }
                )
                predictions["orthogonal"] = [float(v) for v in pred]

            pd.DataFrame(predictions).to_csv(seed_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    metrics_df = pd.DataFrame(metric_rows)
    basis_df = pd.DataFrame(basis_rows)
    contrib_df = pd.DataFrame(contribution_rows)
    metrics_df.to_csv(out_root / "rolling_metrics.csv", index=False, encoding="utf-8-sig")
    basis_df.to_csv(out_root / "basis_terms_long.csv", index=False, encoding="utf-8-sig")
    contrib_df.to_csv(out_root / "realized_terms_long.csv", index=False, encoding="utf-8-sig")

    if basis_df.empty:
        source_stability = pd.DataFrame()
        lane_survival = pd.DataFrame()
    else:
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
        lane_group["lane_key"] = lane_group["selection_channel"].replace("", "unknown") + "::" + lane_group["structural_channel"].replace("", "unknown")
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
        lane_survival["survival_rate"] = lane_survival["split_support"].astype(float) / max(1, int(len(splits)))
        lane_survival = lane_survival.sort_values(
            ["survival_rate", "selected_count", "mean_normalized_weight"],
            ascending=[False, False, False],
        )

    source_stability.to_csv(out_root / "source_stability.csv", index=False, encoding="utf-8-sig")
    lane_survival.to_csv(out_root / "lane_survival.csv", index=False, encoding="utf-8-sig")
    _write_json(out_root / "rolling_metrics.json", metrics_df.to_dict(orient="records"))
    _write_json(out_root / "source_stability.json", source_stability.to_dict(orient="records"))
    _write_json(out_root / "lane_survival.json", lane_survival.to_dict(orient="records"))

    metric_summary = (
        metrics_df.groupby("model", dropna=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            r2_mean=("r2", "mean"),
            duration_sec_mean=("duration_sec", "mean"),
        )
        .reset_index()
        .sort_values("rmse_mean", ascending=True)
    )
    metric_summary.to_csv(out_root / "rolling_metrics_summary.csv", index=False, encoding="utf-8-sig")

    report_md = "\n".join(
        [
            "# work_ci orthogonal symbolic probe",
            "",
            f"- data: `{csv_path}`",
            f"- output: `{out_root}`",
            f"- rows: {len(y_all)}",
            f"- features: {len(feature_names)}",
            f"- splits: {len(splits)}",
            f"- seeds: {', '.join(str(v) for v in seeds)}",
            f"- gates enabled: {bool(args.enable_gates)}",
            "",
            "## Rolling Metrics Summary",
            "",
            _df_to_markdown(metric_summary),
            "",
            "## Rolling Metrics Long",
            "",
            _df_to_markdown(metrics_df.sort_values(["split_id", "seed", "model"])),
            "",
            "## Source Stability",
            "",
            _df_to_markdown(source_stability.head(30) if not source_stability.empty else source_stability),
            "",
            "## Lane Survival",
            "",
            _df_to_markdown(lane_survival.head(30) if not lane_survival.empty else lane_survival),
            "",
            "Note: this real dataset has no known symbolic truth contract. Stability here means repeated exposure across rolling windows, not exact mechanism recovery.",
        ]
    )
    (out_root / "REPORT.md").write_text(report_md, encoding="utf-8")

    summary = {
        "out_root": str(out_root),
        "csv": str(csv_path),
        "n_rows": int(len(y_all)),
        "feature_count": int(len(feature_names)),
        "split_count": int(len(splits)),
        "seeds": list(seeds),
        "models": list(models),
        "metric_summary": metric_summary.to_dict(orient="records"),
        "top_sources": source_stability.head(12).to_dict(orient="records") if not source_stability.empty else [],
        "top_lanes": lane_survival.head(12).to_dict(orient="records") if not lane_survival.empty else [],
    }
    _write_json(out_root / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a work_ci orthogonal symbolic rolling probe.")
    parser.add_argument("--csv", default=default_work_ci_csv_no_flow_speed_occ_lag())
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--target-col", default="ci")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--rolling-splits", type=int, default=3, help="0 means all possible rolling splits.")
    parser.add_argument("--min-train-size", type=int, default=960)
    parser.add_argument("--test-size", type=int, default=120)
    parser.add_argument("--step-size", type=int, default=120)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--models", default="lag1,ridge,stagewise,orthogonal")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--enable-gates", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_probe(args)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
