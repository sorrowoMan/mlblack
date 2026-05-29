# -*- coding: utf-8 -*-
"""Canonical build entry for a migrated example case.

This case preserves an older standalone demo under original/. It exposes a
standard build_solver() surface so the unified scaffold can discover and run it
without relying on legacy assembly/scaffold.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import runpy

_HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MigratedExampleRunner:
    name: str
    original_path: Path

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


def build_solver() -> MigratedExampleRunner:
    return MigratedExampleRunner(name="cross_framework", original_path=_HERE / "original" / "cross_framework")


def build_project_trainer() -> MigratedExampleRunner:
    """Backward-compatible name for older docs; prefer build_solver()."""

    return build_solver()


def main(argv=None) -> None:
    del argv
    build_solver().run()


if __name__ == "__main__":
    main()
