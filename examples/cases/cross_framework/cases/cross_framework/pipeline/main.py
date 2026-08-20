from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from nsgablack.representation import RepresentationPipeline
from nsgablack.representation.continuous import ClipRepair, UniformInitializer


def build_pipeline(
    problem,
    *,
    resource_context: Mapping[str, Any] | None = None,
) -> RepresentationPipeline:
    del problem, resource_context
    low = np.asarray([-2.0], dtype=float)
    high = np.asarray([-0.7], dtype=float)
    return RepresentationPipeline(
        initializer=UniformInitializer(low=low, high=high),
        repair=ClipRepair(low=low, high=high),
    )


__all__ = ["build_pipeline"]
