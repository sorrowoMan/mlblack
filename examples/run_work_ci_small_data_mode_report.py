from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.path_defaults import default_work_ci_csv


def _parse_float_list(raw: str) -> tuple[float, ...]:
    txt = str(raw).strip()
    if not txt:
        return tuple()
    out: list[float] = []
    for p in txt.split(","):
        pp = p.strip()
        if not pp:
            continue
        out.append(float(pp))
    return tuple(out)


def _run_and_parse_summary(cmd: list[str], cwd: Path) -> tuple[dict[str, Any], str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        msg = (
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
        raise RuntimeError(msg)
    m = re.search(r"summary=(.+)", proc.stdout)
    if not m:
        raise RuntimeError(f"Cannot find summary path in output:\n{proc.stdout}")
    summary_path = Path(m.group(1).strip())
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return data, str(summary_path)


def _bootstrap_median_ci(values: np.ndarray, *, iters: int, seed: int) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return {"median": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = np.random.default_rng(int(seed))
    n = int(arr.size)
    meds = np.zeros((int(max(1, iters)),), dtype=float)
    for i in range(meds.shape[0]):
        idx = rng.integers(0, n, size=n, endpoint=False)
        meds[i] = float(np.median(arr[idx]))
    return {
        "median": float(np.median(arr)),
        "ci95_low": float(np.quantile(meds, 0.025)),
        "ci95_high": float(np.quantile(meds, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Small-data mode robust report (CV10 median + bootstrap + rolling median).")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=default_work_ci_csv(),
    )
    parser.add_argument("--target-col", type=str, default="ci")
    parser.add_argument("--kappa-candidates", type=str, default="768,1024")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)

    # Medium complexity settings for small-data mode.
    parser.add_argument("--min-leaf", type=int, default=64)
    parser.add_argument("--local-search-topk-features", type=int, default=8)
    parser.add_argument("--local-search-max-added-terms", type=int, default=8)
    parser.add_argument("--local-search-max-pair-terms", type=int, default=10)
    parser.add_argument("--local-search-max-candidates-per-iter", type=int, default=220)
    parser.add_argument("--local-search-candidate-keep-top", type=int, default=8)
    parser.add_argument("--local-search-unary-ops", type=str, default="square,sin,cos,tanh")
    parser.add_argument("--local-search-nested-unary-patterns", type=str, default="sin(square)")

    # Rolling config.
    parser.add_argument("--rolling-min-train-size", type=int, default=960)
    parser.add_argument("--rolling-test-size", type=int, default=120)
    parser.add_argument("--rolling-step-size", type=int, default=120)
    parser.add_argument("--rolling-split-mode", type=str, default="expanding", choices=["expanding", "sliding"])
    parser.add_argument("--rolling-train-window-size", type=int, default=720)
    args = parser.parse_args()

    kappas = tuple(float(v) for v in _parse_float_list(args.kappa_candidates))
    if not kappas:
        raise ValueError("kappa_candidates is empty")

    piecewise_script = ROOT / "examples" / "run_work_ci_fixed_holiday_piecewise_demo.py"
    rolling_script = ROOT / "examples" / "run_work_ci_fixed_holiday_rolling_eval.py"
    if not piecewise_script.exists():
        raise FileNotFoundError(piecewise_script)
    if not rolling_script.exists():
        raise FileNotFoundError(rolling_script)

    cv10_by_kappa: dict[str, Any] = {}
    rolling_by_kappa: dict[str, Any] = {}

    for kappa in kappas:
        kkey = str(int(kappa) if float(kappa).is_integer() else kappa)
        print(f"[CV10] kappa={kappa}")
        rows: list[dict[str, Any]] = []
        for i in range(1, 11):
            fold_col = f"test_fold_{i}"
            cmd = [
                sys.executable,
                str(piecewise_script),
                "--csv-path",
                str(args.csv_path),
                "--target-col",
                str(args.target_col),
                "--test-fold-col",
                str(fold_col),
                "--min-leaf",
                str(int(args.min_leaf)),
                "--blend-kappa",
                str(float(kappa)),
                "--local-search-topk-features",
                str(int(args.local_search_topk_features)),
                "--local-search-max-added-terms",
                str(int(args.local_search_max_added_terms)),
                "--local-search-max-pair-terms",
                str(int(args.local_search_max_pair_terms)),
                "--local-search-max-candidates-per-iter",
                str(int(args.local_search_max_candidates_per_iter)),
                "--local-search-candidate-keep-top",
                str(int(args.local_search_candidate_keep_top)),
                "--local-search-unary-ops",
                str(args.local_search_unary_ops),
                "--local-search-nested-unary-patterns",
                str(args.local_search_nested_unary_patterns),
            ]
            data, path = _run_and_parse_summary(cmd, ROOT)
            m = data["metrics"]
            rows.append(
                {
                    "fold_col": fold_col,
                    "summary_path": path,
                    "xgboost_rmse": float(m["xgboost_global"]["metrics_test"]["rmse"]),
                    "global_rmse": float(m["symbolic_stagewise_global"]["metrics_test"]["rmse"]),
                    "piece_rmse": float(m["symbolic_stagewise_fixed_piecewise"]["metrics_test"]["rmse"]),
                    "blend_rmse": float(m["symbolic_stagewise_fixed_piecewise_blended"]["metrics_test"]["rmse"]),
                }
            )

        xgb = np.asarray([r["xgboost_rmse"] for r in rows], dtype=float)
        glb = np.asarray([r["global_rmse"] for r in rows], dtype=float)
        pce = np.asarray([r["piece_rmse"] for r in rows], dtype=float)
        bld = np.asarray([r["blend_rmse"] for r in rows], dtype=float)
        cv10_by_kappa[kkey] = {
            "kappa": float(kappa),
            "folds": rows,
            "summary": {
                "xgboost_mean": float(np.mean(xgb)),
                "global_mean": float(np.mean(glb)),
                "piece_mean": float(np.mean(pce)),
                "blend_mean": float(np.mean(bld)),
                "xgboost_median": float(np.median(xgb)),
                "global_median": float(np.median(glb)),
                "piece_median": float(np.median(pce)),
                "blend_median": float(np.median(bld)),
                "wins_blend_vs_xgboost": int(np.sum(bld < xgb)),
                "wins_blend_vs_global": int(np.sum(bld < glb)),
            },
            "bootstrap_median_ci": {
                "xgboost": _bootstrap_median_ci(xgb, iters=int(args.bootstrap_iters), seed=int(args.bootstrap_seed) + 11),
                "global": _bootstrap_median_ci(glb, iters=int(args.bootstrap_iters), seed=int(args.bootstrap_seed) + 22),
                "piece": _bootstrap_median_ci(pce, iters=int(args.bootstrap_iters), seed=int(args.bootstrap_seed) + 33),
                "blend": _bootstrap_median_ci(bld, iters=int(args.bootstrap_iters), seed=int(args.bootstrap_seed) + 44),
            },
        }

        print(f"[Rolling] kappa={kappa}")
        roll_cmd = [
            sys.executable,
            str(rolling_script),
            "--csv-path",
            str(args.csv_path),
            "--target-col",
            str(args.target_col),
            "--split-mode",
            str(args.rolling_split_mode),
            "--min-train-size",
            str(int(args.rolling_min_train_size)),
            "--test-size",
            str(int(args.rolling_test_size)),
            "--step-size",
            str(int(args.rolling_step_size)),
            "--train-window-size",
            str(int(args.rolling_train_window_size)),
            "--min-leaf",
            str(int(args.min_leaf)),
            "--blend-kappa",
            str(float(kappa)),
            "--local-search-topk-features",
            str(int(args.local_search_topk_features)),
            "--local-search-max-added-terms",
            str(int(args.local_search_max_added_terms)),
            "--local-search-max-pair-terms",
            str(int(args.local_search_max_pair_terms)),
            "--local-search-max-candidates-per-iter",
            str(int(args.local_search_max_candidates_per_iter)),
            "--local-search-candidate-keep-top",
            str(int(args.local_search_candidate_keep_top)),
            "--local-search-unary-ops",
            str(args.local_search_unary_ops),
            "--local-search-nested-unary-patterns",
            str(args.local_search_nested_unary_patterns),
        ]
        roll_data, roll_path = _run_and_parse_summary(roll_cmd, ROOT)
        ag = roll_data["aggregate"]
        rolling_by_kappa[kkey] = {
            "kappa": float(kappa),
            "summary_path": roll_path,
            "aggregate": ag,
            "core_medians": {
                "xgboost_median": float(ag["xgboost_global_rmse"]["median"]),
                "global_median": float(ag["symbolic_stagewise_global_rmse"]["median"]),
                "piece_median": float(ag["symbolic_stagewise_fixed_piecewise_rmse"]["median"]),
                "blend_median": float(ag["symbolic_stagewise_fixed_piecewise_blended_rmse"]["median"]),
            },
        }

    # Decision by robust median first (rolling median priority), cv10 median as tie-breaker.
    def _rank_key(k: str) -> tuple[float, float]:
        r = rolling_by_kappa[k]["core_medians"]["blend_median"]
        c = cv10_by_kappa[k]["summary"]["blend_median"]
        return float(r), float(c)

    sorted_keys = sorted(rolling_by_kappa.keys(), key=_rank_key)
    best_key = sorted_keys[0]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "examples" / "out" / f"work_ci_small_data_mode_report_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "summary.json"

    report = {
        "timestamp": datetime.now().isoformat(),
        "mode": "small_data_mode",
        "strategy": {
            "shrinkage": "higher_blend_kappa",
            "complexity": "medium_local_search_budget",
            "decision_metric": "rolling_median_then_cv10_median",
        },
        "config": {
            "csv_path": str(args.csv_path),
            "target_col": str(args.target_col),
            "kappa_candidates": [float(v) for v in kappas],
            "bootstrap_iters": int(args.bootstrap_iters),
            "bootstrap_seed": int(args.bootstrap_seed),
            "min_leaf": int(args.min_leaf),
            "local_search_topk_features": int(args.local_search_topk_features),
            "local_search_max_added_terms": int(args.local_search_max_added_terms),
            "local_search_max_pair_terms": int(args.local_search_max_pair_terms),
            "local_search_max_candidates_per_iter": int(args.local_search_max_candidates_per_iter),
            "local_search_candidate_keep_top": int(args.local_search_candidate_keep_top),
            "local_search_unary_ops": str(args.local_search_unary_ops),
            "local_search_nested_unary_patterns": str(args.local_search_nested_unary_patterns),
            "rolling_min_train_size": int(args.rolling_min_train_size),
            "rolling_test_size": int(args.rolling_test_size),
            "rolling_step_size": int(args.rolling_step_size),
            "rolling_split_mode": str(args.rolling_split_mode),
            "rolling_train_window_size": int(args.rolling_train_window_size),
        },
        "cv10": cv10_by_kappa,
        "rolling": rolling_by_kappa,
        "decision": {
            "best_kappa": float(best_key),
            "ranking": [float(k) for k in sorted_keys],
            "best_rolling_blend_median": float(rolling_by_kappa[best_key]["core_medians"]["blend_median"]),
            "best_cv10_blend_median": float(cv10_by_kappa[best_key]["summary"]["blend_median"]),
        },
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("WORK_CI_SMALL_DATA_MODE_REPORT_DONE")
    print(f"output_root={out_root}")
    print(f"best_kappa={float(best_key):.0f}")
    print(
        "best metrics (median): "
        f"rolling_blend={report['decision']['best_rolling_blend_median']:.6f}, "
        f"cv10_blend={report['decision']['best_cv10_blend_median']:.6f}"
    )
    print(f"summary={out_path}")


if __name__ == "__main__":
    main()
