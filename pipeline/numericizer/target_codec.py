from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TargetCodec:
    kind: str = "numeric"
    classes: Sequence[Any] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def encode(self, y: Sequence[Any]) -> np.ndarray:
        if self.kind in {"numeric", "regression"}:
            return np.asarray(y, dtype=float).reshape(-1)
        classes = tuple(self.classes)
        index = {str(value): i for i, value in enumerate(classes)}
        return np.asarray([index[str(value)] for value in y], dtype=float)

    def decode(self, values: Sequence[float]) -> tuple[Any, ...]:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if self.kind in {"numeric", "regression"} or not self.classes:
            return tuple(float(v) for v in arr)
        classes = tuple(self.classes)
        return tuple(classes[int(np.clip(round(v), 0, len(classes) - 1))] for v in arr)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "classes": list(self.classes), "metadata": dict(self.metadata)}
