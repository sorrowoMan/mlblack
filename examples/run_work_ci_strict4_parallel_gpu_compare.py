from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.path_defaults import default_reports_dir, default_work_ci_csv


def _resolve_path(raw: str, *, base_dir: Path | None = None) -> str:
    txt = str(raw).strip()
    if not txt:
        return ""
    p = Path(txt).expanduser()
    if p.is_absolute():
        return str(p)
    if base_dir is not None:
        return str((base_dir / p).resolve())
    return str((ROOT / p).resolve())


def _summary_from_stdout(stdout: str) -> str:
    for line in reversed(str(stdout).splitlines()):
        m = re.match(r"^\s*summary\s*=\s*(.+?)\s*$", line)
        if m:
            return str(m.group(1)).strip()
    raise RuntimeError("Cannot locate summary=... in command stdout.")


def _build_cmd(
    *,
    config_root: Path,
    dataset_cfg: Mapping[str, Any],
    rolling_cfg: Mapping[str, Any],
    base_cfg: Mapping[str, Any],
    run_cfg: Mapping[str, Any],
    strict4_branch_hparams_json: str,
) -> list[str]:
    merged = dict(base_cfg)
    merged.update(dict(run_cfg.get("overrides", {})))

    csv_path = _resolve_path(str(dataset_cfg.get("csv_path", "")).strip(), base_dir=config_root)
    if not csv_path:
        csv_path = default_work_ci_csv()

    cmd: list[str] = [
        sys.executable,
        str(ROOT / "examples" / "run_work_ci_fixed_holiday_rolling_eval.py"),
        "--csv-path",
        str(csv_path),
        "--target-col",
        str(dataset_cfg.get("target_col", "ci")),
        "--date-col",
        str(dataset_cfg.get("date_col", "date")),
        "--split-mode",
        str(rolling_cfg.get("split_mode", "expanding")),
        "--min-train-size",
        str(int(rolling_cfg.get("min_train_size", 600))),
        "--test-size",
        str(int(rolling_cfg.get("test_size", 120))),
        "--step-size",
        str(int(rolling_cfg.get("step_size", 120))),
        "--train-window-size",
        str(int(rolling_cfg.get("train_window_size", 720))),
        "--regime-mode",
        str(merged.get("regime_mode", "strict4")),
        "--min-leaf",
        str(int(merged.get("min_leaf", 64))),
        "--blend-kappa",
        str(float(merged.get("blend_kappa", 512.0))),
        "--local-search-force-linear-base",
        str(merged.get("local_search_force_linear_base", "auto")),
        "--local-search-topk-features",
        str(int(merged.get("local_search_topk_features", 8))),
        "--local-search-max-added-terms",
        str(int(merged.get("local_search_max_added_terms", 12))),
        "--local-search-max-pair-terms",
        str(int(merged.get("local_search_max_pair_terms", 16))),
        "--local-search-max-candidates-per-iter",
        str(int(merged.get("local_search_max_candidates_per_iter", 500))),
        "--local-search-candidate-keep-top",
        str(int(merged.get("local_search_candidate_keep_top", 12))),
        "--local-search-ridge-l2",
        str(float(merged.get("local_search_ridge_l2", 1e-4))),
        "--local-search-unary-ops",
        str(merged.get("local_search_unary_ops", "square,sin,cos,tanh")),
        "--local-search-nested-unary-patterns",
        str(merged.get("local_search_nested_unary_patterns", "sin(square),cos(square)")),
        "--local-search-overfit-guard-val-ratio",
        str(float(merged.get("local_search_overfit_guard_val_ratio", 0.2))),
        "--local-search-overfit-guard-min-val-samples",
        str(int(merged.get("local_search_overfit_guard_min_val_samples", 64))),
        "--local-search-overfit-guard-min-val-rmse-gain",
        str(float(merged.get("local_search_overfit_guard_min_val_rmse_gain", 0.0))),
        "--local-search-overfit-guard-max-gap-increase",
        str(float(merged.get("local_search_overfit_guard_max_gap_increase", 0.05))),
        "--local-search-overfit-guard-patience",
        str(int(merged.get("local_search_overfit_guard_patience", 3))),
        "--local-search-interaction-budget-mode",
        str(merged.get("local_search_interaction_budget_mode", "fixed")),
        "--local-search-interaction-diag-threshold",
        str(float(merged.get("local_search_interaction_diag_threshold", 1.15))),
        "--local-search-interaction-diag-topk-features",
        str(int(merged.get("local_search_interaction_diag_topk_features", 8))),
        "--local-search-interaction-pair-budget-boost",
        str(float(merged.get("local_search_interaction_pair_budget_boost", 2.0))),
        "--local-search-interaction-grad-projection-budget-boost",
        str(float(merged.get("local_search_interaction_grad_projection_budget_boost", 1.5))),
        "--small-sample-guard-threshold",
        str(int(merged.get("small_sample_guard_threshold", 0))),
        "--blend-global-backbone-mode",
        str(merged.get("blend_global_backbone_mode", "symbolic_only")),
        "--blend-global-backbone-val-ratio",
        str(float(merged.get("blend_global_backbone_val_ratio", 0.2))),
        "--blend-global-backbone-min-val-samples",
        str(int(merged.get("blend_global_backbone_min_val_samples", 64))),
        "--blend-global-backbone-margin",
        str(float(merged.get("blend_global_backbone_margin", 0.0))),
        "--local-search-inner-opt-method",
        str(merged.get("local_search_inner_opt_method", "adam_lbfgs")),
        "--local-search-inner-opt-device",
        str(merged.get("local_search_inner_opt_device", "auto")),
        "--local-search-inner-opt-adam-steps",
        str(int(merged.get("local_search_inner_opt_adam_steps", 120))),
        "--local-search-inner-opt-adam-lr",
        str(float(merged.get("local_search_inner_opt_adam_lr", 5e-3))),
        "--local-search-inner-opt-lbfgs-steps",
        str(int(merged.get("local_search_inner_opt_lbfgs_steps", 60))),
        "--local-search-inner-opt-lbfgs-lr",
        str(float(merged.get("local_search_inner_opt_lbfgs_lr", 0.8))),
        "--local-search-inner-opt-l2",
        str(float(merged.get("local_search_inner_opt_l2", 0.0))),
        "--local-search-inner-opt-accept-rmse-tol",
        str(float(merged.get("local_search_inner_opt_accept_rmse_tol", 1e-6))),
        "--strict4-parallel-mode",
        str(run_cfg.get("strict4_parallel_mode", "serial")),
        "--strict4-max-workers",
        str(int(run_cfg.get("strict4_max_workers", 1))),
        "--strict4-gpu-strategy",
        str(run_cfg.get("strict4_gpu_strategy", "none")),
        "--strict4-gpu-devices",
        str(run_cfg.get("strict4_gpu_devices", "")),
    ]

    if bool(merged.get("disable_merge_rare_holiday_regimes", False)):
        cmd.append("--disable-merge-rare-holiday-regimes")
    if bool(merged.get("disable_confidence_blend", False)):
        cmd.append("--disable-confidence-blend")
    if bool(merged.get("local_search_overfit_guard_enabled", False)):
        cmd.append("--local-search-overfit-guard-enabled")
    if bool(merged.get("local_search_inner_opt_enabled", False)):
        cmd.append("--local-search-inner-opt-enabled")
    if bool(run_cfg.get("use_branch_hparams", False)):
        branch_json_path = _resolve_path(strict4_branch_hparams_json, base_dir=config_root)
        if not Path(branch_json_path).exists():
            branch_json_path = _resolve_path(strict4_branch_hparams_json, base_dir=ROOT)
        cmd.extend(["--strict4-branch-hparams-json", str(branch_json_path)])

    return cmd


def _extract_row(run_name: str, summary_path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    agg = dict(payload.get("aggregate", {}))
    wins = dict(agg.get("wins", {}))
    xgb = dict(agg.get("xgboost_global_rmse", {}))
    global_row = dict(agg.get("symbolic_stagewise_global_rmse", {}))
    piece = dict(agg.get("symbolic_stagewise_fixed_piecewise_rmse", {}))
    blend = dict(agg.get("symbolic_stagewise_fixed_piecewise_blended_rmse", {}))

    splits = list(payload.get("splits", []))
    runtime_set = []
    for item in splits:
        s = dict(item.get("summary", {}))
        runtime_set.append(dict(s.get("strict4_parallel_runtime", {})))
    runtime = runtime_set[0] if runtime_set else {}

    return {
        "run": str(run_name),
        "summary_path": str(summary_path),
        "xgb_mean": float(xgb.get("mean", float("nan"))),
        "global_mean": float(global_row.get("mean", float("nan"))),
        "piece_mean": float(piece.get("mean", float("nan"))),
        "blend_mean": float(blend.get("mean", float("nan"))),
        "blend_lt_xgb": f"{int(wins.get('blend_better_than_xgboost_count', 0))}/{int(wins.get('n_splits', 0))}",
        "blend_lt_global": f"{int(wins.get('blend_better_than_global_count', 0))}/{int(wins.get('n_splits', 0))}",
        "parallel_mode": str(runtime.get("effective_mode", "unknown")),
        "parallel_workers": int(runtime.get("effective_workers", 0)) if runtime.get("effective_workers") is not None else 0,
        "gpu_strategy": str(runtime.get("gpu_strategy", "none")),
        "gpu_devices": ",".join(str(x) for x in list(runtime.get("gpu_devices", []))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict4 parallel/GPU compare and emit report table.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "examples" / "configs" / "work_ci_strict4_parallel_gpu_compare_template.json"),
    )
    parser.add_argument("--name", type=str, default="work_ci_strict4_parallel_gpu_compare")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg_root = cfg_path.parent
    payload = json.loads(cfg_path.read_text(encoding="utf-8-sig"))

    dataset_cfg = dict(payload.get("dataset", {}))
    rolling_cfg = dict(payload.get("rolling", {}))
    base_cfg = dict(payload.get("base_args", {}))
    compare_runs = list(payload.get("compare_runs", []))
    strict4_branch_hparams_json = str(payload.get("strict4_branch_hparams_json", "")).strip()
    if not compare_runs:
        raise ValueError("compare_runs must not be empty")

    run_rows: list[dict[str, Any]] = []
    run_logs: list[dict[str, Any]] = []
    for item in compare_runs:
        run_cfg = dict(item)
        run_name = str(run_cfg.get("name", "unnamed"))
        cmd = _build_cmd(
            config_root=cfg_root,
            dataset_cfg=dataset_cfg,
            rolling_cfg=rolling_cfg,
            base_cfg=base_cfg,
            run_cfg=run_cfg,
            strict4_branch_hparams_json=strict4_branch_hparams_json,
        )
        print(f"[run] {run_name}")
        print("  " + " ".join(cmd))
        if bool(args.dry_run):
            continue

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"run '{run_name}' failed (exit={proc.returncode})\n"
                f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )
        summary_path = _summary_from_stdout(proc.stdout)
        summary_obj = json.loads(Path(summary_path).read_text(encoding="utf-8-sig"))
        row = _extract_row(run_name, summary_path, summary_obj)
        run_rows.append(row)
        run_logs.append(
            {
                "name": run_name,
                "command": cmd,
                "summary_path": summary_path,
                "stdout_tail": proc.stdout.splitlines()[-20:],
            }
        )
        print(
            f"  done: blend_mean={row['blend_mean']:.6f}, "
            f"global_mean={row['global_mean']:.6f}, xgb_mean={row['xgb_mean']:.6f}"
        )

    if bool(args.dry_run):
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path(default_reports_dir()).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    base = f"{args.name}_{ts}"
    csv_path = reports_dir / f"{base}.csv"
    md_path = reports_dir / f"{base}.md"
    json_path = reports_dir / f"{base}.json"

    csv_fields = [
        "run",
        "summary_path",
        "xgb_mean",
        "global_mean",
        "piece_mean",
        "blend_mean",
        "blend_lt_xgb",
        "blend_lt_global",
        "parallel_mode",
        "parallel_workers",
        "gpu_strategy",
        "gpu_devices",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for row in run_rows:
            w.writerow({k: row.get(k, "") for k in csv_fields})

    md_lines = [
        "# strict4 parallel/gpu compare",
        "",
        "| run | xgb_mean | global_mean | piece_mean | blend_mean | blend<xgb | blend<global | mode | workers | gpu_strategy | gpu_devices |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for row in run_rows:
        md_lines.append(
            f"| {row['run']} | {row['xgb_mean']:.6f} | {row['global_mean']:.6f} | "
            f"{row['piece_mean']:.6f} | {row['blend_mean']:.6f} | {row['blend_lt_xgb']} | "
            f"{row['blend_lt_global']} | {row['parallel_mode']} | {int(row['parallel_workers'])} | "
            f"{row['gpu_strategy']} | {row['gpu_devices']} |"
        )
    md_lines.append("")
    md_lines.append("## summaries")
    for row in run_rows:
        md_lines.append(f"- {row['run']}: `{row['summary_path']}`")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    out_payload = {
        "timestamp": datetime.now().isoformat(),
        "config": str(cfg_path),
        "rows": run_rows,
        "runs": run_logs,
        "csv": str(csv_path),
        "markdown": str(md_path),
    }
    json_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("STRICT4_PARALLEL_GPU_COMPARE_DONE")
    print(f"csv={csv_path}")
    print(f"markdown={md_path}")
    print(f"json={json_path}")


if __name__ == "__main__":
    main()
