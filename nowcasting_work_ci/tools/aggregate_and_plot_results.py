from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nowcasting_work_ci.mlblack_side.config import build_runs_root


def _safe_float(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def _safe_int(v: Any) -> int | None:
    try:
        return int(v)
    except Exception:
        return None


def _extract_row_from_summary(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    cfg = data.get("config", {}) if isinstance(data.get("config"), dict) else {}
    tc = data.get("test_compare", {}) if isinstance(data.get("test_compare"), dict) else {}
    itv = tc.get("interval_metrics", {}) if isinstance(tc.get("interval_metrics"), dict) else {}
    sym = itv.get("symbolic", {}) if isinstance(itv.get("symbolic"), dict) else {}
    xgb = itv.get("xgboost", {}) if isinstance(itv.get("xgboost"), dict) else {}
    ds = data.get("dataset", {}) if isinstance(data.get("dataset"), dict) else {}
    outer = data.get("outer_search", {}) if isinstance(data.get("outer_search"), dict) else {}

    alpha = _safe_float(cfg.get("interval_alpha"))
    target_cov = None if alpha is None else 1.0 - alpha
    picp = _safe_float(sym.get("picp"))
    pinaw = _safe_float(sym.get("pinaw"))
    iscore = _safe_float(sym.get("interval_score"))
    rmse = _safe_float(tc.get("symbolic_subset_rmse"))

    row = {
        "record_type": "summary",
        "source_file": str(path),
        "run_id": str(path.parent.name),
        "timestamp": str(data.get("timestamp", "")),
        "seed": _safe_int(cfg.get("seed")),
        "pop_size": _safe_int(cfg.get("pop_size")),
        "generations": _safe_int(cfg.get("generations")),
        "rolling_folds": _safe_int(cfg.get("rolling_folds")),
        "max_terms": _safe_int(cfg.get("max_terms")),
        "strict4_enabled": bool(cfg.get("strict4_branch_mode_enabled", False)),
        "interval_method": str(cfg.get("interval_method", "")),
        "interval_alpha": alpha,
        "interval_target_coverage": target_cov,
        "interval_calib_ratio": _safe_float(cfg.get("interval_calib_ratio")),
        "interval_quantile_l2": _safe_float(cfg.get("interval_quantile_l2")),
        "symbolic_rmse": rmse,
        "symbolic_picp": picp,
        "symbolic_pinaw": pinaw,
        "symbolic_is": iscore,
        "xgb_rmse": _safe_float(tc.get("xgboost_rmse")),
        "xgb_picp": _safe_float(xgb.get("picp")),
        "xgb_pinaw": _safe_float(xgb.get("pinaw")),
        "xgb_is": _safe_float(xgb.get("interval_score")),
        "picp_abs_gap_to_target": None
        if (picp is None or target_cov is None)
        else abs(picp - target_cov),
        "n_train": _safe_int(ds.get("n_train")),
        "n_test": _safe_int(ds.get("n_test")),
        "n_features": _safe_int(ds.get("n_features")),
        "outer_duration_sec": _safe_float(outer.get("duration_sec")),
    }
    return row


def _extract_rows_from_sweep_json(path: Path, data: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(data, list):
        iterable = data
    elif isinstance(data, dict) and isinstance(data.get("runs"), list):
        iterable = data.get("runs")
    else:
        return rows
    for i, r in enumerate(iterable):
        if not isinstance(r, dict):
            continue
        alpha = _safe_float(r.get("alpha"))
        target_cov = None if alpha is None else 1.0 - alpha
        picp = _safe_float(r.get("symbolic_picp"))
        row = {
            "record_type": "sweep",
            "source_file": str(path),
            "run_id": str(r.get("tag", f"{path.stem}#{i}")),
            "timestamp": "",
            "seed": _safe_int(r.get("seed")),
            "pop_size": None,
            "generations": None,
            "rolling_folds": None,
            "max_terms": None,
            "strict4_enabled": None,
            "interval_method": "",
            "interval_alpha": alpha,
            "interval_target_coverage": target_cov,
            "interval_calib_ratio": _safe_float(r.get("interval_calib_ratio")),
            "interval_quantile_l2": _safe_float(r.get("interval_quantile_l2")),
            "symbolic_rmse": _safe_float(r.get("symbolic_rmse")),
            "symbolic_picp": picp,
            "symbolic_pinaw": _safe_float(r.get("symbolic_pinaw")),
            "symbolic_is": _safe_float(r.get("symbolic_is")),
            "xgb_rmse": _safe_float(r.get("xgb_rmse")),
            "xgb_picp": _safe_float(r.get("xgb_picp")),
            "xgb_pinaw": _safe_float(r.get("xgb_pinaw")),
            "xgb_is": _safe_float(r.get("xgb_is")),
            "picp_abs_gap_to_target": None
            if (picp is None or target_cov is None)
            else abs(picp - target_cov),
            "n_train": None,
            "n_test": None,
            "n_features": None,
            "outer_duration_sec": None,
        }
        rows.append(row)
    return rows


def _experiment_key(row: dict[str, Any]) -> str:
    parts = [
        f"pop{row.get('pop_size')}",
        f"gen{row.get('generations')}",
        f"alpha{row.get('interval_alpha')}",
        f"calib{row.get('interval_calib_ratio')}",
        f"ql2{row.get('interval_quantile_l2')}",
        f"strict4{int(bool(row.get('strict4_enabled')))}",
        f"method{row.get('interval_method')}",
    ]
    return "|".join(parts)


def _write_csv(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def _write_markdown(path: Path, rows: list[dict[str, Any]], cols: list[str], limit: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    use = rows[: max(0, int(limit))]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in use:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")


def _make_plots(rows: list[dict[str, Any]], out_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    used = [r for r in rows if r.get("symbolic_pinaw") is not None and r.get("symbolic_picp") is not None]
    if not used:
        return []

    paths: list[str] = []

    # 1) PICP-PINAW scatter (core interval tradeoff)
    xs = [float(r["symbolic_pinaw"]) for r in used]
    ys = [float(r["symbolic_picp"]) for r in used]
    cs = [float(r["symbolic_is"]) if r.get("symbolic_is") is not None else 0.0 for r in used]
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", alpha=0.85, edgecolors="k", linewidths=0.3)
    ax.set_xlabel("PINAW (lower is better)")
    ax.set_ylabel("PICP (target = 1-alpha)")
    ax.set_title("Interval Tradeoff Map: PICP vs PINAW (color = Interval Score)")
    fig.colorbar(sc, ax=ax, label="Interval Score")
    p = out_dir / "picp_vs_pinaw.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))

    # 2) RMSE vs PINAW (show point-vs-interval compromise)
    used2 = [r for r in rows if r.get("symbolic_rmse") is not None and r.get("symbolic_pinaw") is not None]
    if used2:
        x2 = [float(r["symbolic_pinaw"]) for r in used2]
        y2 = [float(r["symbolic_rmse"]) for r in used2]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x2, y2, alpha=0.85, edgecolors="k", linewidths=0.3)
        ax.set_xlabel("PINAW (lower is better)")
        ax.set_ylabel("Symbolic RMSE")
        ax.set_title("Point-Interval Compromise: RMSE vs PINAW")
        p2 = out_dir / "rmse_vs_pinaw.png"
        fig.tight_layout()
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        paths.append(str(p2))

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate nowcasting interval experiment summaries and generate report tables/plots.")
    parser.add_argument("--out-root", type=str, default=str(build_runs_root(ROOT)))
    parser.add_argument("--report-dir", type=str, default="", help="Report output dir. Default: <out-root>/reports/<timestamp>")
    parser.add_argument("--include-sweep-json", type=int, default=1, help="Include top-level sweep/aggregate json files in out root.")
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    if not out_root.exists():
        raise FileNotFoundError(f"out root not found: {out_root}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir).resolve() if str(args.report_dir).strip() else (out_root / "reports" / stamp)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for p in sorted(out_root.glob("**/summary.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                rows.append(_extract_row_from_summary(p, data))
        except Exception:
            continue

    if bool(int(args.include_sweep_json)):
        for p in sorted(out_root.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.extend(_extract_rows_from_sweep_json(p, data))

    rows = [r for r in rows if r.get("symbolic_pinaw") is not None and r.get("symbolic_picp") is not None]
    rows.sort(key=lambda r: (float(r["symbolic_pinaw"]), float(r.get("symbolic_is") or 1e9)))

    for r in rows:
        r["experiment_key"] = _experiment_key(r)

    # Aggregate by experiment key
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r["experiment_key"])].append(r)

    agg_rows: list[dict[str, Any]] = []
    for k, g in groups.items():
        pinaw = [float(x["symbolic_pinaw"]) for x in g if x.get("symbolic_pinaw") is not None]
        picp = [float(x["symbolic_picp"]) for x in g if x.get("symbolic_picp") is not None]
        isv = [float(x["symbolic_is"]) for x in g if x.get("symbolic_is") is not None]
        rmse = [float(x["symbolic_rmse"]) for x in g if x.get("symbolic_rmse") is not None]
        target = [float(x["interval_target_coverage"]) for x in g if x.get("interval_target_coverage") is not None]
        row0 = g[0]
        agg_rows.append(
            {
                "experiment_key": k,
                "n_runs": len(g),
                "pop_size": row0.get("pop_size"),
                "generations": row0.get("generations"),
                "interval_alpha": row0.get("interval_alpha"),
                "interval_calib_ratio": row0.get("interval_calib_ratio"),
                "interval_quantile_l2": row0.get("interval_quantile_l2"),
                "strict4_enabled": row0.get("strict4_enabled"),
                "pinaw_mean": mean(pinaw) if pinaw else None,
                "pinaw_std": pstdev(pinaw) if len(pinaw) > 1 else 0.0,
                "picp_mean": mean(picp) if picp else None,
                "picp_std": pstdev(picp) if len(picp) > 1 else 0.0,
                "is_mean": mean(isv) if isv else None,
                "is_std": pstdev(isv) if len(isv) > 1 else 0.0,
                "rmse_mean": mean(rmse) if rmse else None,
                "target_coverage": mean(target) if target else None,
                "picp_gap_abs": None
                if not (picp and target)
                else abs(mean(picp) - mean(target)),
            }
        )
    agg_rows.sort(key=lambda r: (float(r["pinaw_mean"]) if r.get("pinaw_mean") is not None else 1e9, float(r.get("is_mean") or 1e9)))

    cols = [
        "record_type",
        "run_id",
        "seed",
        "pop_size",
        "generations",
        "strict4_enabled",
        "interval_alpha",
        "interval_target_coverage",
        "interval_calib_ratio",
        "interval_quantile_l2",
        "symbolic_picp",
        "symbolic_pinaw",
        "symbolic_is",
        "symbolic_rmse",
        "xgb_picp",
        "xgb_pinaw",
        "xgb_is",
        "xgb_rmse",
        "picp_abs_gap_to_target",
        "source_file",
        "experiment_key",
    ]
    agg_cols = [
        "experiment_key",
        "n_runs",
        "pop_size",
        "generations",
        "strict4_enabled",
        "interval_alpha",
        "interval_calib_ratio",
        "interval_quantile_l2",
        "target_coverage",
        "picp_mean",
        "picp_std",
        "picp_gap_abs",
        "pinaw_mean",
        "pinaw_std",
        "is_mean",
        "is_std",
        "rmse_mean",
    ]

    _write_csv(report_dir / "runs_flat.csv", rows, cols)
    _write_markdown(report_dir / "runs_flat.md", rows, cols, limit=300)
    _write_csv(report_dir / "experiment_aggregate.csv", agg_rows, agg_cols)
    _write_markdown(report_dir / "experiment_aggregate.md", agg_rows, agg_cols, limit=200)

    plot_paths = _make_plots(rows, report_dir)

    dashboard = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "out_root": str(out_root),
        "report_dir": str(report_dir),
        "n_run_records": len(rows),
        "n_experiment_groups": len(agg_rows),
        "best_by_pinaw": agg_rows[0] if agg_rows else {},
        "plot_files": plot_paths,
        "table_files": [
            str(report_dir / "runs_flat.csv"),
            str(report_dir / "runs_flat.md"),
            str(report_dir / "experiment_aggregate.csv"),
            str(report_dir / "experiment_aggregate.md"),
        ],
    }
    (report_dir / "dashboard.json").write_text(json.dumps(dashboard, ensure_ascii=False, indent=2), encoding="utf-8")

    print("NOWCASTING_REPORT_DONE")
    print(f"report_dir={report_dir}")
    print(f"n_run_records={len(rows)}")
    if agg_rows:
        b = agg_rows[0]
        print(
            "best_by_pinaw: "
            f"pinaw_mean={b.get('pinaw_mean')}, "
            f"picp_mean={b.get('picp_mean')}, "
            f"is_mean={b.get('is_mean')}, "
            f"exp={b.get('experiment_key')}"
        )


if __name__ == "__main__":
    main()
