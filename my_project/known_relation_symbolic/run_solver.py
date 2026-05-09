from __future__ import annotations

import argparse
from typing import Sequence

from my_project.known_relation_symbolic.build_solver import build_known_relation_symbolic_components


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Known-relation symbolic scaffold entrypoint.")
    parser.add_argument("--check", action="store_true", help="Validate scaffold wiring without running benchmarks.")
    parser.add_argument("--list-scenarios", action="store_true", help="Print registered known-relation scenarios.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    components = build_known_relation_symbolic_components()
    if args.check:
        print(
            "known_relation_symbolic scaffold ok | "
            f"scenarios={len(tuple(components['scenario_keys']))} | "
            f"outer_solver_backend={components['outer_solver_backend']}"
        )
        return
    if args.list_scenarios:
        for key in tuple(components["scenario_keys"]):
            print(str(key))
        return
    print("Use --check or --list-scenarios. Benchmark execution stays in formal runners.")


if __name__ == "__main__":
    main()
