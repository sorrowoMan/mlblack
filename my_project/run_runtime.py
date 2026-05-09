# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_project.build_runtime import build_runtime_components
from my_project.runtime.workflow import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MLBLACK scaffold template")
    parser.add_argument("--check", action="store_true", help="Check assembly only")
    parser.add_argument("--run-id", type=str, default="", help="Optional run id")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    components = build_runtime_components()
    if bool(args.check):
        print(
            "[check] mlblack scaffold ok | "
            "layout=config/problem/features/model/reporting/runtime"
        )
        return
    report_path = run_pipeline(components=components, run_id=args.run_id)
    print(f"summary={report_path}")


if __name__ == "__main__":
    main()
