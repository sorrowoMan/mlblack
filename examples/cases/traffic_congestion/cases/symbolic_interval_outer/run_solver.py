from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    from _bootstrap import ensure_case_importable  # noqa: E402
    ensure_case_importable(Path(__file__))
    from build_solver import build_solver  # noqa: E402
else:
    from ._bootstrap import ensure_case_importable  # noqa: E402
    ensure_case_importable(Path(__file__))
    from .build_solver import build_solver  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run traffic symbolic interval forecasting through the standard Case scaffold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args, unknown = build_parser().parse_known_args(list(argv) if argv is not None else None)
    forwarded = list(unknown or ())
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    bridge = build_solver(default_args=tuple(forwarded))
    if bool(args.check):
        print("[check] " + json.dumps(bridge.check(), ensure_ascii=False, sort_keys=True))
        return 0
    return int(bridge.run(()))


if __name__ == "__main__":
    raise SystemExit(main())
