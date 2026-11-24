# -*- coding: utf-8 -*-
"""t-SNE demo: manual assembly of TSNE trainer with sklearn digits dataset."""

from __future__ import annotations

import numpy as np

from sklearn.datasets import load_digits

from mlblack.integrations.nsgablack_control import LearningSolver, build_learning_solver
from mlblack.integrations.nsgablack_optimization import build_optimization_adapter
from .pipeline.main import build_pipeline
from .problem.tsne_problem import TSNEProblem


def build_tsne_trainer(
    X: np.ndarray,
    *,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    exaggeration: float = 4.0,
    exaggeration_steps: int = 100,
    run_name: str = "tsne_digits",
    component_overrides=None,
    resource_context=None,
) -> LearningSolver:
    n_samples = X.shape[0]

    representation = build_pipeline(n_samples, output_dim=2, component_overrides=component_overrides)
    problem = TSNEProblem(
        X,
        perplexity=perplexity,
        exaggeration=exaggeration,
        exaggeration_steps=exaggeration_steps,
    )
    adapter = build_optimization_adapter(
        "gradient.sgd",
        learning_rate=learning_rate,
        min_learning_rate=1e-8,
        max_gradient_norm=None,
    )

    trainer = build_learning_solver(
        problem=problem,
        representation=representation,
        adapter=adapter,
        run_name=run_name,
        resource_context=resource_context,
    )
    return trainer


def build_solver(
    *,
    config=None,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    exaggeration: float = 4.0,
    exaggeration_steps: int = 100,
    resource_context=None,
    component_overrides=None,
):
    """Canonical unified scaffold entry; returns the assembled Trainer."""

    del config
    digits = load_digits()
    X = digits.data.astype(np.float64)
    return build_tsne_trainer(
        X,
        perplexity=perplexity,
        learning_rate=learning_rate,
        exaggeration=exaggeration,
        exaggeration_steps=exaggeration_steps,
        component_overrides=component_overrides,
        resource_context=resource_context,
    )
