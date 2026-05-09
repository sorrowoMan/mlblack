from __future__ import annotations

from typing import Sequence

from my_project.runtime.workflow import main as run_workflow


def run(argv: Sequence[str] | None = None) -> None:
    run_workflow(list(argv) if argv is not None else None)
