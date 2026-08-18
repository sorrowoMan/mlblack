# -*- coding: utf-8 -*-
"""Canonical build entry for a migrated example case.

This case preserves an older standalone demo under original/. It exposes a
standard build_solver() surface so the shared Project substrate can discover
and run it as a Case.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import runpy

from mlblack.project.scaffold import print_case_check

_HERE = Path(__file__).resolve().parent


@dataclass
class MigratedExampleRunner:
    name: str
    original_path: Path
    resource_context: dict = field(default_factory=dict)

    def set_resource_context(self, context):
        self.resource_context = dict(context or {})
        return self

    def run(self) -> None:
        _run_original(self.original_path)


def _run_original(original_path: Path) -> None:
    if original_path.is_file():
        runpy.run_path(str(original_path), run_name="__main__")
        return
    for candidate in (
        original_path / "run_case.py",
        original_path / "server.py",
        original_path / "neural_graph_benchmark_matrix.py",
        original_path / "symbolic_orthogonal_nested_benchmark.py",
        original_path / "__main__.py",
    ):
        if candidate.exists():
            runpy.run_path(str(candidate), run_name="__main__")
            return
    raise FileNotFoundError(f"No runnable entrypoint found under {original_path}")


def build_solver(config=None, *, resource_context=None, component_overrides=None) -> MigratedExampleRunner:
    del config, component_overrides
    return MigratedExampleRunner(
        name="benchmarks",
        original_path=_HERE / "original" / "benchmarks",
        resource_context=dict(resource_context or {}),
    )


def build_project_trainer() -> MigratedExampleRunner:
    """Backward-compatible name for older docs; prefer build_solver()."""

    return build_solver()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the migrated benchmarks case.")
    parser.add_argument("--check", action="store_true", help="Build and validate only; do not run.")
    args = parser.parse_args(argv)
    runner = build_solver()
    if args.check:
        print_case_check(runner)
        return 0
    runner.run()
    return 0


if __name__ == "__main__":
    main()
