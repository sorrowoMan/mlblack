from __future__ import annotations

"""Stable public runtime assembly forwarder."""

from typing import Sequence

from nowcasting_work_ci.mlblack_side.runtime import main as run_workflow


def run(argv: Sequence[str] | None = None) -> None:
    run_workflow(list(argv) if argv is not None else None)


__all__ = ["run"]
