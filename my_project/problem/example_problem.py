from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProblemContext:
    data_csv: str
    target_col: str
