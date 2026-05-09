from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class DoctorProblem:
    severity: str
    code: str
    message: str
    path: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "severity": str(self.severity),
            "code": str(self.code),
            "message": str(self.message),
            "path": self.path,
        }


@dataclass(frozen=True)
class DoctorRule:
    rule_id: str
    description: str
    run: Callable[[Path], list[DoctorProblem]]

