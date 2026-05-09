from __future__ import annotations

import argparse
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="nsgablack-side scaffold entry for nowcasting_work_ci")
    p.add_argument("--check", action="store_true", help="Check scaffold wiring only.")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args, rest = parser.parse_known_args(list(argv) if argv is not None else None)
    if bool(args.check):
        print(
            "[check] nsgablack_side scaffold ok | "
            "layout=problem/pipeline/adapter/evaluation/plugins/build_solver/run_solver"
        )
        return
    from nowcasting_work_ci.run import main as run_main

    run_main(rest)


if __name__ == "__main__":
    main()
