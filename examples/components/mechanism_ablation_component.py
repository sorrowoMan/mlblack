from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MechanismPreset:
    name: str
    info_gain: int
    novelty: int
    curriculum: int
    dual_archive: int
    counterfactual: int


DEFAULT_PRESETS: tuple[MechanismPreset, ...] = (
    MechanismPreset("all_on", 1, 1, 1, 1, 1),
    MechanismPreset("all_off", 0, 0, 0, 0, 0),
    MechanismPreset("off_info_gain", 0, 1, 1, 1, 1),
    MechanismPreset("off_novelty", 1, 0, 1, 1, 1),
    MechanismPreset("off_curriculum", 1, 1, 0, 1, 1),
    MechanismPreset("off_dual_archive", 1, 1, 1, 0, 1),
    MechanismPreset("off_counterfactual", 1, 1, 1, 1, 0),
)


def _extract_summary_path(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("summary="):
            return line.split("=", 1)[1].strip()
    return ""


def _load_metrics(summary_path: str) -> dict[str, Any]:
    if not summary_path:
        return {}
    p = Path(summary_path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    tc = data.get("test_compare", {}) if isinstance(data.get("test_compare"), dict) else {}
    itv = tc.get("interval_metrics", {}) if isinstance(tc.get("interval_metrics"), dict) else {}
    sym = itv.get("symbolic", {}) if isinstance(itv.get("symbolic"), dict) else {}
    return {
        "symbolic_rmse": tc.get("symbolic_subset_rmse"),
        "symbolic_mae": tc.get("symbolic_subset_mae"),
        "symbolic_picp": sym.get("picp"),
        "symbolic_pinaw": sym.get("pinaw"),
        "symbolic_is": sym.get("interval_score"),
        "summary_path": str(p),
    }


def run_mechanism_ablation(
    *,
    target_script: Path,
    out_dir: Path,
    base_args: list[str],
    presets: list[MechanismPreset] | None = None,
    python_exe: str | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = python_exe or sys.executable
    use_presets = list(presets or DEFAULT_PRESETS)
    rows: list[dict[str, Any]] = []

    for preset in use_presets:
        cmd = [
            exe,
            str(target_script),
            *base_args,
            "--mechanism-info-gain-enabled",
            str(int(preset.info_gain)),
            "--mechanism-novelty-enabled",
            str(int(preset.novelty)),
            "--mechanism-curriculum-enabled",
            str(int(preset.curriculum)),
            "--mechanism-dual-archive-enabled",
            str(int(preset.dual_archive)),
            "--mechanism-counterfactual-enabled",
            str(int(preset.counterfactual)),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        summary_path = _extract_summary_path(proc.stdout)
        metrics = _load_metrics(summary_path)
        rows.append(
            {
                "preset": preset.name,
                "rc": int(proc.returncode),
                "info_gain": int(preset.info_gain),
                "novelty": int(preset.novelty),
                "curriculum": int(preset.curriculum),
                "dual_archive": int(preset.dual_archive),
                "counterfactual": int(preset.counterfactual),
                "symbolic_rmse": metrics.get("symbolic_rmse"),
                "symbolic_mae": metrics.get("symbolic_mae"),
                "symbolic_picp": metrics.get("symbolic_picp"),
                "symbolic_pinaw": metrics.get("symbolic_pinaw"),
                "symbolic_is": metrics.get("symbolic_is"),
                "summary_path": metrics.get("summary_path", summary_path),
            }
        )

    rows.sort(key=lambda r: (float(r.get("symbolic_pinaw") or 1e9), float(r.get("symbolic_is") or 1e9)))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"mechanism_ablation_{stamp}.json"
    csv_path = out_dir / f"mechanism_ablation_{stamp}.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "preset",
                "rc",
                "info_gain",
                "novelty",
                "curriculum",
                "dual_archive",
                "counterfactual",
                "symbolic_rmse",
                "symbolic_mae",
                "symbolic_picp",
                "symbolic_pinaw",
                "symbolic_is",
                "summary_path",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return json_path

