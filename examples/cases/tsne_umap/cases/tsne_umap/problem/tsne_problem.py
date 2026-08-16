# -*- coding: utf-8 -*-
"""t-SNE LearningProblem: KL divergence between high-dim and low-dim affinities."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.core.problem import LearningProblem
from mlblack.core.types import Feedback


_ATOL = 1e-12
_EPS = np.finfo(np.float64).eps


def _squared_distances(X: np.ndarray) -> np.ndarray:
    sum_sq = np.sum(X ** 2, axis=1, keepdims=True)
    D = sum_sq + sum_sq.T - 2 * (X @ X.T)
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)
    return D


def _binary_search_perplexity(
    D: np.ndarray,
    perplexity: float,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> np.ndarray:
    assert D.shape[0] == D.shape[1]
    n = D.shape[0]

    target_entropy = np.log2(perplexity)

    D_sorted = np.sort(D, axis=1)
    k = min(max(1, int(perplexity)), n - 1)
    sigma = np.sqrt(D_sorted[:, k]) * 0.5 + 1e-3
    sigma = np.maximum(sigma, 1e-6)

    lower = np.zeros(n, dtype=np.float64)
    upper = np.full(n, np.inf, dtype=np.float64)

    for _ in range(max_iter):
        beta = 1.0 / (2.0 * np.maximum(sigma, _EPS) ** 2)
        P = np.exp(-D * beta[:, None])
        np.fill_diagonal(P, 0.0)
        row_sum = P.sum(axis=1)
        row_sum = np.maximum(row_sum, _EPS)
        P = P / row_sum[:, None]

        entropy = -np.sum(P * np.log2(np.maximum(P, _EPS)), axis=1)

        mask_high = entropy > target_entropy
        upper[mask_high] = sigma[mask_high]
        lower[~mask_high] = sigma[~mask_high]

        sigma[mask_high & np.isinf(upper)] = sigma[mask_high & np.isinf(upper)] * 2.0
        sigma[~mask_high & (upper == np.inf)] = sigma[~mask_high & (upper == np.inf)] * 2.0

        mask_bisect = ~np.isinf(upper) & (upper > lower)
        sigma[mask_bisect] = (lower[mask_bisect] + upper[mask_bisect]) / 2.0

        if np.all(np.abs(entropy - target_entropy) < tol):
            break

    return sigma


def _compute_high_affinities(X: np.ndarray, perplexity: float) -> np.ndarray:
    n = X.shape[0]
    D = _squared_distances(X)

    sigma = _binary_search_perplexity(D, perplexity)

    P = np.exp(-D / (2.0 * np.maximum(sigma[:, None] ** 2, _EPS)))
    np.fill_diagonal(P, 0.0)
    row_sum = P.sum(axis=1)
    row_sum = np.maximum(row_sum, _EPS)
    P = P / row_sum[:, None]

    P = (P + P.T) / (2.0 * n)
    P = np.maximum(P, _EPS)
    return P


def _compute_low_affinities(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    D = _squared_distances(Y)
    Q_num = 1.0 / (1.0 + D)
    np.fill_diagonal(Q_num, 0.0)
    Z = Q_num.sum()
    Z = max(Z, _EPS)
    Q = Q_num / Z
    return Q, Q_num


def _tsne_gradient(P: np.ndarray, Q: np.ndarray, Y: np.ndarray) -> np.ndarray:
    n = Y.shape[0]
    PQ_diff = P - Q
    Y_diff = Y[:, None, :] - Y[None, :, :]
    D_sq = np.sum(Y_diff ** 2, axis=2)
    inv_one_plus_D = 1.0 / (1.0 + np.maximum(D_sq, 0.0))
    np.fill_diagonal(inv_one_plus_D, 0.0)

    grad = 4.0 * np.sum(
        PQ_diff[:, :, np.newaxis] * Y_diff * inv_one_plus_D[:, :, np.newaxis],
        axis=1,
    )
    return grad


class TSNEProblem(LearningProblem):
    """t-SNE dimensionality reduction as optimization.

    Computes KL(P || Q) where P are high-dim Gaussian affinities
    and Q are low-dim Student-t (1-dof) affinities.
    """

    context_requires = ()
    context_provides = ("feedback.objectives", "feedback.gradients", "feedback.metrics")
    context_mutates = ()
    context_cache = ()
    context_notes = "Computes t-SNE KL divergence and gradient from high-dim affinities."

    def __init__(
        self,
        X: np.ndarray,
        *,
        perplexity: float = 30.0,
        exaggeration: float = 4.0,
        exaggeration_steps: int = 100,
        name: str = "tsne",
    ):
        self._X = np.asarray(X, dtype=np.float64)
        n = self._X.shape[0]
        if n < 3:
            raise ValueError("t-SNE requires at least 3 samples")
        self.perplexity = float(perplexity)
        self.exaggeration = float(exaggeration)
        self.exaggeration_steps = max(0, int(exaggeration_steps))

        self._P = _compute_high_affinities(self._X, self.perplexity)
        self._P_early = self._P * self.exaggeration
        self._step_count = 0

        self.name = str(name)

    def evaluate(self, model: Any, state: Any, context: Mapping[str, Any] | None = None) -> Feedback:
        Y = np.asarray(model, dtype=np.float64)
        if Y.ndim != 2 or Y.shape[1] < 2:
            Y = Y.reshape(-1, max(2, Y.shape[-1]))

        n = Y.shape[0]

        Q, Q_num = _compute_low_affinities(Y)

        step = self._step_count
        if context is not None:
            step = int(context.get("step", step))
        P_active = self._P_early if step < self.exaggeration_steps else self._P

        kl = float(np.sum(P_active * np.log(np.maximum(P_active, _EPS) / np.maximum(Q, _EPS))))

        grad = _tsne_gradient(P_active, Q, Y)

        return Feedback(
            objectives=np.array([kl]),
            gradients=grad.ravel(),
            loss=kl,
            constraints=np.zeros(0, dtype=float),
            metrics={"kl_divergence": kl, "n_samples": n, "perplexity": self.perplexity},
        )

    def inc_step(self) -> None:
        self._step_count += 1

    def reset_step(self) -> None:
        self._step_count = 0

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "n_samples": int(self._X.shape[0]),
            "n_features": int(self._X.shape[1]),
            "perplexity": self.perplexity,
            "exaggeration": self.exaggeration,
            "exaggeration_steps": self.exaggeration_steps,
            "objective": "minimize KL(P || Q)",
        }
