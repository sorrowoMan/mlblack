# -*- coding: utf-8 -*-
"""t-SNE demo: manual assembly of TSNE trainer with sklearn digits dataset."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import argparse
import time

import numpy as np

from sklearn.datasets import load_digits

from mlblack.adapters.gradient_descent import GradientDescentAdapter, GradientDescentConfig
from mlblack.core.trainer import ComposableTrainer

from problem.tsne_problem import TSNEProblem
from representation.tsne_representation import TSNERepresentation


def build_tsne_trainer(
    X: np.ndarray,
    *,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    exaggeration: float = 4.0,
    exaggeration_steps: int = 100,
    run_name: str = "tsne_digits",
) -> ComposableTrainer:
    n_samples = X.shape[0]

    representation = TSNERepresentation(n_samples=n_samples, output_dim=2)
    problem = TSNEProblem(
        X,
        perplexity=perplexity,
        exaggeration=exaggeration,
        exaggeration_steps=exaggeration_steps,
    )
    adapter = GradientDescentAdapter(
        GradientDescentConfig(
            learning_rate=learning_rate,
            min_learning_rate=1e-8,
            max_grad_norm=None,
            require_gradient=True,
        )
    )

    trainer = ComposableTrainer(
        problem=problem,
        representation=representation,
        adapter=adapter,
        run_name=run_name,
    )
    return trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description="t-SNE with sklearn digits")
    parser.add_argument("--steps", type=int, default=500, help="Number of t-SNE iterations")
    parser.add_argument("--perplexity", type=float, default=30.0, help="Perplexity (default 30)")
    parser.add_argument("--lr", type=float, default=200.0, help="Learning rate")
    parser.add_argument("--exaggeration", type=float, default=4.0, help="Early exaggeration factor")
    parser.add_argument("--exaggeration-steps", type=int, default=100, help="Exaggeration phase steps")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--verbose", type=int, default=50, help="Report interval (steps)")
    args = parser.parse_args(argv)

    np.random.seed(args.seed)

    print("=== t-SNE: digits dataset ===")
    digits = load_digits()
    X = digits.data.astype(np.float64)
    y = digits.target

    print(f"  samples: {X.shape[0]}, features: {X.shape[1]}, classes: {len(np.unique(y))}")

    print("Building trainer ...")
    trainer = build_tsne_trainer(
        X,
        perplexity=args.perplexity,
        learning_rate=args.lr,
        exaggeration=args.exaggeration,
        exaggeration_steps=args.exaggeration_steps,
    )

    print(f"  steps={args.steps}, perplexity={args.perplexity}, lr={args.lr}")
    print("Training ...")
    t0 = time.perf_counter()

    result = trainer.fit(max_steps=args.steps)
    elapsed = time.perf_counter() - t0

    final_kl = result.best_feedback.loss if result.best_feedback else float("nan")
    if final_kl is None:
        final_kl = float(
            result.best_feedback.metrics.get("kl_divergence", float("nan"))
            if result.best_feedback
            else float("nan")
        )

    print(f"  elapsed: {elapsed:.2f}s")
    print(f"  final KL divergence: {final_kl:.6f}")
    print(f"  steps completed: {len(result.history)}")
    print("=== done ===")

    return result


if __name__ == "__main__":
    main()
