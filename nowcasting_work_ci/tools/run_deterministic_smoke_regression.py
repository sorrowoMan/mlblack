# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RUN_ENTRY = PACKAGE_ROOT / "run.py"
DEFAULT_SMOKE_ARGS = [
    "--pop-size",
    "4",
    "--generations",
    "1",
    "--rolling-folds",
    "1",
    "--strict4-branch-mode",
    "--interval-alpha",
    "0.2",
    "--interval-method",
    "symmetric_residual",
    "--interval-calib-ratio",
    "0.05",
    "--selection-coverage-error-threshold",
    "0.06",
    "--seed",
    "42",
]


def _extract_summary_path(stdout: str) -> Path:
    for line in str(stdout).splitlines():
        if line.startswith("summary="):
            return Path(line.split("=", 1)[1].strip())
    raise RuntimeError("failed to parse summary path from run output")


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary_snapshot(summary: Mapping[str, Any]) -> dict[str, Any]:
    best_solution = dict(summary["best_solution"])
    test_compare = dict(summary["test_compare"])
    symbolic_interval = dict(test_compare["interval_metrics"]["symbolic"])
    xgb_interval = dict(test_compare["interval_metrics"]["xgboost"])

    return {
        "config": {
            "seed": int(summary["config"]["seed"]),
            "interval_method": str(summary["config"]["interval_method"]),
            "interval_alpha": float(summary["config"]["interval_alpha"]),
            "pop_size": int(summary["config"]["pop_size"]),
            "generations": int(summary["config"]["generations"]),
        },
        "best_solution": {
            "subset_idx": [int(v) for v in best_solution["subset_idx"]],
            "subset_names": list(best_solution["subset_names"]),
            "obj_coverage_error": float(best_solution["obj_coverage_error"]),
            "obj_pinaw": float(best_solution["obj_pinaw"]),
            "obj_interval_score": float(best_solution["obj_interval_score"]),
        },
        "test_compare": {
            "symbolic_subset_rmse": float(test_compare["symbolic_subset_rmse"]),
            "symbolic_subset_mae": float(test_compare["symbolic_subset_mae"]),
            "xgboost_rmse": float(test_compare["xgboost_rmse"]),
            "xgboost_mae": float(test_compare["xgboost_mae"]),
            "symbolic_interval": {
                "picp": float(symbolic_interval["picp"]),
                "pinaw": float(symbolic_interval["pinaw"]),
                "interval_score": float(symbolic_interval["interval_score"]),
                "mean_width": float(symbolic_interval["mean_width"]),
            },
            "xgboost_interval": {
                "picp": float(xgb_interval["picp"]),
                "pinaw": float(xgb_interval["pinaw"]),
                "interval_score": float(xgb_interval["interval_score"]),
                "mean_width": float(xgb_interval["mean_width"]),
            },
        },
    }


def diff_summary_snapshots(left: Any, right: Any, *, path: str = "") -> list[str]:
    current = str(path)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        diffs: list[str] = []
        keys = sorted(set(left.keys()) | set(right.keys()))
        for key in keys:
            next_path = f"{current}.{key}" if current else str(key)
            if key not in left:
                diffs.append(f"{next_path}: missing in left")
                continue
            if key not in right:
                diffs.append(f"{next_path}: missing in right")
                continue
            diffs.extend(diff_summary_snapshots(left[key], right[key], path=next_path))
        return diffs

    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{current}: length mismatch {len(left)} != {len(right)}"]
        diffs: list[str] = []
        for idx, (lv, rv) in enumerate(zip(left, right)):
            diffs.extend(diff_summary_snapshots(lv, rv, path=f"{current}[{idx}]"))
        return diffs

    if isinstance(left, float) or isinstance(right, float):
        lv = float(left)
        rv = float(right)
        if abs(lv - rv) > 1e-12:
            return [f"{current}: {lv} != {rv}"]
        return []

    if left != right:
        return [f"{current}: {left!r} != {right!r}"]
    return []


def run_smoke_once(*, python_exe: str, smoke_args: list[str]) -> tuple[Path, dict[str, Any], str]:
    command = [python_exe, str(RUN_ENTRY), *smoke_args]
    proc = subprocess.run(
        command,
        cwd=PACKAGE_ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "deterministic smoke run failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    summary_path = _extract_summary_path(proc.stdout)
    summary = _load_summary(summary_path)
    return summary_path, build_summary_snapshot(summary), proc.stdout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed nowcasting smoke command twice and compare stable summary fields.",
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for subprocess runs.")
    parser.add_argument("--runs", type=int, default=2, help="How many repeated runs to execute.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args appended after the default smoke command. Use '-- --seed 7' style.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    extra_args = list(args.extra_args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    smoke_args = [*DEFAULT_SMOKE_ARGS, *extra_args]

    print("DETERMINISTIC_SMOKE_REGRESSION_START")
    print(f"entry={RUN_ENTRY}")
    print(f"command={args.python} {' '.join([str(RUN_ENTRY), *smoke_args])}")

    runs: list[tuple[Path, dict[str, Any]]] = []
    for idx in range(int(max(2, args.runs))):
        summary_path, snapshot, stdout = run_smoke_once(
            python_exe=str(args.python),
            smoke_args=smoke_args,
        )
        runs.append((summary_path, snapshot))
        print(f"run[{idx}].summary={summary_path}")
        print(
            "run[{idx}].metrics="
            "coverage_error={coverage_error:.12f} pinaw={pinaw:.12f} is={interval_score:.12f} rmse={rmse:.12f}".format(
                idx=idx,
                coverage_error=float(snapshot["best_solution"]["obj_coverage_error"]),
                pinaw=float(snapshot["test_compare"]["symbolic_interval"]["pinaw"]),
                interval_score=float(snapshot["test_compare"]["symbolic_interval"]["interval_score"]),
                rmse=float(snapshot["test_compare"]["symbolic_subset_rmse"]),
            )
        )

    baseline_path, baseline_snapshot = runs[0]
    all_diffs: list[str] = []
    for idx, (summary_path, snapshot) in enumerate(runs[1:], start=1):
        diffs = diff_summary_snapshots(baseline_snapshot, snapshot)
        if diffs:
            all_diffs.append(f"run[0]={baseline_path} vs run[{idx}]={summary_path}")
            all_diffs.extend(diffs)

    if all_diffs:
        print("DETERMINISTIC_SMOKE_REGRESSION_FAIL")
        for line in all_diffs:
            print(line)
        return 1

    print("DETERMINISTIC_SMOKE_REGRESSION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
