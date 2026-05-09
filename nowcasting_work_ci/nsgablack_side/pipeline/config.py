from __future__ import annotations

import numpy as np
from nsgablack.core.base import BlackBoxProblem
from nsgablack.representation import RepresentationPipeline
from nsgablack.representation.continuous import ClipRepair, ContextGaussianMutation, UniformInitializer


def bounds_arrays(problem: BlackBoxProblem) -> tuple[np.ndarray, np.ndarray]:
    b = getattr(problem, "bounds", None)
    d = int(getattr(problem, "dimension", 0))
    if isinstance(b, dict):
        keys = list(getattr(problem, "variables", []))
        if len(keys) != d or any(k not in b for k in keys):
            keys = list(b.keys())
        pairs = [b[k] for k in keys]
    else:
        pairs = list(b or [])
    if len(pairs) != d:
        low = np.full(d, -1.0, dtype=float)
        high = np.full(d, 1.0, dtype=float)
        return low, high
    low = np.asarray([float(p[0]) for p in pairs], dtype=float)
    high = np.asarray([float(p[1]) for p in pairs], dtype=float)
    lo = np.minimum(low, high)
    hi = np.maximum(low, high)
    return lo, hi


def _collect_seedable_components(pipeline: RepresentationPipeline) -> list[object]:
    visited: set[int] = set()
    ordered: list[object] = []

    def _visit(obj: object | None) -> None:
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)
        ordered.append(obj)
        for attr in ("initializer", "mutator", "repair", "crossover", "encoder", "inner"):
            _visit(getattr(obj, attr, None))
        initializers = getattr(obj, "initializers", None)
        if initializers:
            for item in list(initializers):
                if isinstance(item, tuple) and item:
                    _visit(item[0])

    _visit(pipeline)
    return ordered


def seed_pipeline_rngs(pipeline: RepresentationPipeline, seed: int | None) -> None:
    if seed is None:
        return
    components = _collect_seedable_components(pipeline)
    if not components:
        return
    seed_sequence = np.random.SeedSequence(int(seed))
    child_sequences = seed_sequence.spawn(len(components))
    for component, child_seq in zip(components, child_sequences):
        if hasattr(component, "_rng"):
            setattr(component, "_rng", np.random.default_rng(child_seq))


def build_pipeline(
    problem: BlackBoxProblem,
    *,
    base_sigma: float = 0.18,
    sigma_key: str = "mutation_sigma",
) -> RepresentationPipeline:
    low, high = bounds_arrays(problem)
    return RepresentationPipeline(
        initializer=UniformInitializer(low=low, high=high),
        mutator=ContextGaussianMutation(
            base_sigma=float(max(1e-6, base_sigma)),
            sigma_key=str(sigma_key),
            low=low,
            high=high,
        ),
        repair=ClipRepair(low=low, high=high),
    )
