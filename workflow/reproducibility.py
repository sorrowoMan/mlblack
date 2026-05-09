from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReproducibilityConfig:
    seed: int
    deterministic_torch: bool = True
    torch_warn_only: bool = True


def apply_reproducibility(config: ReproducibilityConfig) -> dict[str, Any]:
    seed = int(config.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    info: dict[str, Any] = {
        "seed": int(seed),
        "python_random_seeded": True,
        "numpy_seeded": True,
        "torch_seeded": False,
        "torch_deterministic": False,
        "torch_status": "not_imported",
    }

    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        info["torch_status"] = f"unavailable:{type(exc).__name__}"
        return info

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    info["torch_seeded"] = True
    info["torch_status"] = "seeded"

    if bool(config.deterministic_torch):
        try:
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True, warn_only=bool(config.torch_warn_only))
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            info["torch_deterministic"] = True
        except Exception as exc:  # pragma: no cover - optional dependency
            info["torch_status"] = f"seeded_non_deterministic:{type(exc).__name__}"
    return info


__all__ = ["ReproducibilityConfig", "apply_reproducibility"]
