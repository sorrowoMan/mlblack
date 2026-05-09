from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.path_defaults import apply_env_defaults, default_outputs_dir, default_reports_dir, default_work_ci_csv
from project.scaffold import ScaffoldSpec, load_scaffold_spec, run_project_scaffold


HARDCODED_BEST_PARAMS: dict[str, Any] = {
    "version": "v2",
    "lower_quantile": 0.1195,
    "upper_quantile": 0.8527,
    "v2_continuous_ops": ["identity", "sin", "cos"],
    "v2_binary_ops": ["identity"],
    "v2_include_interactions": True,
    "v2_max_interactions": 14,
    "v2_topk_features": 5,
    "v2_include_hinge": True,
    "v2_hinge_quantiles": [0.25, 0.5, 0.75],
    "order_penalty": 9.858985209852348,
    "width_penalty": 0.00013665897880253504,
    "epochs": 140,
    "batch_size": 128,
    "lr": 0.005292095244739332,
    "weight_decay": 0.0001,
    "l1_readout": 0.00011685592026743783,
    "l1_params": 0.0,
    "device": "cpu",
    "conformal_calibration": True,
    "conformal_level": 0.8018,
    "stagewise_warmup_enabled": False,
    "gate_piecewise_enabled": False,
    "random_seed": 20271720,
}


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    return str(v)


def _interval_score(y: np.ndarray, l: np.ndarray, u: np.ndarray, alpha: float) -> np.ndarray:
    width = u - l
    under = (2.0 / float(alpha)) * (l - y) * (y < l)
    over = (2.0 / float(alpha)) * (y - u) * (y > u)
    return width + under + over


def _eval_metrics(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, alpha: float, y_range: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    l = np.asarray(lo, dtype=float).reshape(-1)
    u = np.asarray(hi, dtype=float).reshape(-1)
    mid = 0.5 * (l + u)
    rmse = float(np.sqrt(np.mean((mid - y) ** 2)))
    picp = float(np.mean((y >= l) & (y <= u)))
    width = np.maximum(u - l, 0.0)
    mean_width = float(np.mean(width))
    return {
        "rmse": rmse,
        "picp": picp,
        "mean_width": mean_width,
        "pinaw": float(np.mean(width / max(1e-8, float(y_range)))),
        "interval_score": float(np.mean(_interval_score(y, l, u, alpha))),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = ["rmse", "picp", "mean_width", "pinaw", "interval_score"]
    out: dict[str, Any] = {"n": int(len(rows))}
    for k in keys:
        arr = np.asarray([float(r["metrics"][k]) for r in rows], dtype=float)
        out[k] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=0)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return out


def _load_params(args: argparse.Namespace) -> dict[str, Any]:
    if args.params_json:
        payload = json.loads(Path(args.params_json).read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict) and "params" in payload and isinstance(payload["params"], dict):
            return dict(payload["params"])
        if isinstance(payload, dict):
            return dict(payload)
        raise ValueError("params_json must be a dict or {'params': {...}}")

    if args.best_from_report:
        rp = Path(args.best_from_report)
        if rp.exists():
            payload = json.loads(rp.read_text(encoding="utf-8-sig"))
            p = payload.get("new_best_feasible", {}).get("params", None)
            if isinstance(p, dict):
                return dict(p)

    return dict(HARDCODED_BEST_PARAMS)


def _run_one(spec: ScaffoldSpec) -> dict[str, Any]:
    t0 = time.perf_counter()
    result = run_project_scaffold(spec)
    sec = float(time.perf_counter() - t0)

    X_test = np.asarray(result.processed.X_test, dtype=float)
    y_test = np.asarray(result.processed.y_test, dtype=float).reshape(-1)
    y_train = np.asarray(result.processed.y_train, dtype=float).reshape(-1)
    y_range = float(np.max(y_train) - np.min(y_train))
    lo, hi = result.artifact.predict_interval(X_test)
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)

    metrics = _eval_metrics(y_test, lo, hi, alpha=0.3, y_range=y_range)
    return {
        "duration_sec": sec,
        "n_train": int(y_train.shape[0]),
        "n_test": int(y_test.shape[0]),
        "metrics": metrics,
        "trainer_metrics": {k: dict(v) for k, v in result.metrics.items()},
        "artifact_type": type(result.artifact).__name__,
    }


def _build_rolling_splits(
    *,
    n_samples: int,
    min_train_size: int,
    test_size: int,
    step_size: int,
    mode: str,
    train_window_size: int,
) -> list[dict[str, int]]:
    if n_samples <= 0:
        return []
    if min_train_size <= 0 or test_size <= 0 or step_size <= 0:
        raise ValueError("min_train_size/test_size/step_size must be > 0")
    if min_train_size + test_size > n_samples:
        raise ValueError("min_train_size + test_size exceeds n_samples")
    key = str(mode).strip().lower()
    if key not in {"expanding", "sliding"}:
        raise ValueError("rolling_mode must be 'expanding' or 'sliding'")
    if key == "sliding" and train_window_size <= 0:
        raise ValueError("train_window_size must be > 0 for sliding")

    splits: list[dict[str, int]] = []
    train_end = int(min_train_size)
    sid = 0
    while train_end + int(test_size) <= int(n_samples):
        test_start = int(train_end)
        test_end = int(test_start + int(test_size))
        if key == "expanding":
            train_start = 0
        else:
            train_start = max(0, int(train_end) - int(train_window_size))
        if train_start >= train_end:
            break
        splits.append(
            {
                "split_id": int(sid),
                "train_start": int(train_start),
                "train_end": int(train_end),
                "test_start": int(test_start),
                "test_end": int(test_end),
            }
        )
        sid += 1
        train_end += int(step_size)
    return splits


def main() -> None:
    apply_env_defaults()

    parser = argparse.ArgumentParser(description="Fixed-parameter full-fold + rolling evaluation for work_ci symbolic interval.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "examples" / "configs" / "work_ci_symbolic_torch_interval.json"),
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=default_work_ci_csv(),
    )
    parser.add_argument("--feature-recipe", type=str, default="raw_all_numeric")
    parser.add_argument(
        "--best-from-report",
        type=str,
        default=str(Path(default_reports_dir()) / "compare_new_framework_parallel_best_of4_vs_old_work_fold10.json"),
    )
    parser.add_argument("--params-json", type=str, default="")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(Path(default_outputs_dir()) / "symbolic_interval" / "fixed_cv_rolling_eval"),
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=str(Path(default_reports_dir()) / "symbolic_interval_fixed_cv_rolling_eval.json"),
    )
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=10)
    parser.add_argument("--rolling-min-train-size", type=int, default=960)
    parser.add_argument("--rolling-test-size", type=int, default=120)
    parser.add_argument("--rolling-step-size", type=int, default=120)
    parser.add_argument("--rolling-mode", type=str, default="expanding")
    parser.add_argument("--rolling-train-window-size", type=int, default=720)
    args = parser.parse_args()

    params = _load_params(args)
    base_spec = load_scaffold_spec(args.config)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_json).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # -------- Full fold CV --------
    cv_rows: list[dict[str, Any]] = []
    for fold in range(int(args.fold_start), int(args.fold_end) + 1):
        fold_col = f"test_fold_{fold}"
        print(f"[CV] running {fold_col} ...")
        data_spec = replace(
            base_spec.data,
            csv_path=str(args.csv_path),
            split_mode="fold_flag",
            test_fold_col=str(fold_col),
            feature_recipe=str(args.feature_recipe),
        )
        train_spec = replace(
            base_spec.train,
            trainer_key="symbolic_torch_interval",
            trainer_params=dict(params),
            output_dir=str(output_root / f"cv_{fold_col}"),
            run_name=f"fixed_cv_{fold_col}",
        )
        spec = ScaffoldSpec(data=data_spec, train=train_spec)
        one = _run_one(spec)
        one["fold_col"] = str(fold_col)
        cv_rows.append(one)
        print(
            f"[CV] {fold_col} rmse={one['metrics']['rmse']:.4f} "
            f"picp={one['metrics']['picp']:.4f} width={one['metrics']['mean_width']:.4f}"
        )

    # -------- Rolling --------
    df = pd.read_csv(args.csv_path)
    date_col = str(base_spec.data.date_col or "date")
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found in {args.csv_path}")
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        raise ValueError(f"invalid date values in '{date_col}'")
    df = df.sort_values(date_col).reset_index(drop=True)

    splits = _build_rolling_splits(
        n_samples=int(df.shape[0]),
        min_train_size=int(args.rolling_min_train_size),
        test_size=int(args.rolling_test_size),
        step_size=int(args.rolling_step_size),
        mode=str(args.rolling_mode),
        train_window_size=int(args.rolling_train_window_size),
    )
    tmp_dir = output_root / "rolling_tmp_csv"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rolling_rows: list[dict[str, Any]] = []
    for sp in splits:
        sid = int(sp["split_id"])
        train_start = int(sp["train_start"])
        train_end = int(sp["train_end"])
        test_start = int(sp["test_start"])
        test_end = int(sp["test_end"])
        print(f"[ROLL] running split={sid} test=[{test_start},{test_end}) ...")

        # Strict time-order evaluation: keep only rows up to test_end.
        # This avoids training with future rows beyond the test window.
        window_start = 0 if str(args.rolling_mode).strip().lower() == "expanding" else train_start
        dfx = df.iloc[window_start:test_end].copy().reset_index(drop=True)
        dfx["test_fold_rolling"] = 0
        local_test_start = int(test_start - window_start)
        local_test_end = int(test_end - window_start)
        dfx.loc[local_test_start : local_test_end - 1, "test_fold_rolling"] = 1
        csv_tmp = tmp_dir / f"rolling_split_{sid:02d}.csv"
        dfx.to_csv(csv_tmp, index=False)

        data_spec = replace(
            base_spec.data,
            csv_path=str(csv_tmp),
            split_mode="fold_flag",
            test_fold_col="test_fold_rolling",
            feature_recipe=str(args.feature_recipe),
        )
        train_spec = replace(
            base_spec.train,
            trainer_key="symbolic_torch_interval",
            trainer_params=dict(params),
            output_dir=str(output_root / f"rolling_split_{sid:02d}"),
            run_name=f"fixed_rolling_{sid:02d}",
        )
        spec = ScaffoldSpec(data=data_spec, train=train_spec)
        one = _run_one(spec)
        one.update(
            {
                "split_id": sid,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "window_start": int(window_start),
                "window_end": int(test_end),
            }
        )
        rolling_rows.append(one)
        print(
            f"[ROLL] split={sid} rmse={one['metrics']['rmse']:.4f} "
            f"picp={one['metrics']['picp']:.4f} width={one['metrics']['mean_width']:.4f}"
        )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(Path(args.config).resolve()),
        "csv_path": str(Path(args.csv_path).resolve()),
        "feature_recipe": str(args.feature_recipe),
        "params": params,
        "cv": {
            "fold_range": [int(args.fold_start), int(args.fold_end)],
            "rows": cv_rows,
            "summary": _summarize(cv_rows),
        },
        "rolling": {
            "mode": str(args.rolling_mode),
            "min_train_size": int(args.rolling_min_train_size),
            "test_size": int(args.rolling_test_size),
            "step_size": int(args.rolling_step_size),
            "train_window_size": int(args.rolling_train_window_size),
            "n_splits": int(len(splits)),
            "rows": rolling_rows,
            "summary": _summarize(rolling_rows),
        },
    }
    report_path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE report={report_path}")


if __name__ == "__main__":
    main()
