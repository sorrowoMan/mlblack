from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from my_project.problem.example_problem import ProblemContext


@dataclass(frozen=True)
class FeatureBundle:
    X: np.ndarray
    y: np.ndarray


def build_features(problem: ProblemContext, *, add_bias: bool) -> FeatureBundle:
    _ = problem
    rng = np.random.default_rng(42)
    X = rng.normal(size=(128, 4))
    if add_bias:
        X = np.concatenate([np.ones((X.shape[0], 1), dtype=float), X], axis=1)
    y = 2.0 * X[:, -1] + rng.normal(scale=0.5, size=(X.shape[0],))
    return FeatureBundle(X=X.astype(float), y=y.astype(float))
