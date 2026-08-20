#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate every formal example Project with the shared strict Doctor."""

from __future__ import annotations

from pathlib import Path

from blackbase.project import format_doctor_report, run_common_project_doctor


ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = ROOT / "examples" / "cases"


def main() -> int:
    failed = False
    for project_root in sorted(path for path in PROJECTS_ROOT.iterdir() if path.is_dir()):
        if not all((project_root / name).exists() for name in ("project_config.py", "run_project.py", "cases")):
            continue
        report = run_common_project_doctor(project_root, strict=True)
        print(f"[{project_root.name}] {'OK' if report.ok else 'ERROR'}")
        if not report.ok:
            failed = True
            print(format_doctor_report(report))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
