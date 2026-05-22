from __future__ import annotations

import numpy as np

from nsgablack.representation import RepresentationPipeline
from nsgablack.representation.continuous import ClipRepair, GaussianMutation, UniformInitializer


def build_representation_pipeline(problem, *, mutation_sigma: float) -> RepresentationPipeline:
    lows = np.asarray([problem.bounds[f"x{i}"][0] for i in range(problem.dimension)], dtype=float)
    highs = np.asarray([problem.bounds[f"x{i}"][1] for i in range(problem.dimension)], dtype=float)
    return RepresentationPipeline(
        initializer=UniformInitializer(low=lows, high=highs),
        mutator=GaussianMutation(sigma=float(mutation_sigma), low=lows, high=highs),
        repair=ClipRepair(low=lows, high=highs),
    )
