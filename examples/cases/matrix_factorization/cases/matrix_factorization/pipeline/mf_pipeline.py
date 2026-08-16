# -*- coding: utf-8 -*-
"""Data pipeline for matrix factorization: synthetic rating generation."""

from __future__ import annotations

import numpy as np

from mlblack.pipeline.data_views import NumericDataView


def generate_synthetic_ratings(
    n_users=100,
    n_items=200,
    k=5,
    sparsity=0.80,
    noise=0.1,
    seed=42,
    return_true=False,
):
    """Generate a low-rank rating matrix with missing entries and noise.

    Returns (R_observed, mask) or (R, R_observed, mask, true_U, true_V, R_true).
    """
    rng = np.random.default_rng(seed)
    true_U = rng.normal(0.0, 1.0, size=(n_users, k)) * 0.5
    true_V = rng.normal(0.0, 1.0, size=(n_items, k)) * 0.5

    R_true = true_U @ true_V.T
    R = R_true + rng.normal(0.0, noise, size=R_true.shape)
    R = np.clip(R, 1.0, 5.0)

    mask = rng.random(size=R.shape) > sparsity
    if mask.sum() < 10:
        idx = rng.choice(n_users * n_items, size=10, replace=False)
        mask_flat = mask.ravel()
        mask_flat[idx] = True
        mask = mask_flat.reshape(R.shape)

    R_observed = np.where(mask, R, 0.0)

    if return_true:
        return R, R_observed, mask, true_U, true_V, R_true
    return R_observed, mask


def build_rating_data_view(R, mask, feature_names=None, target_name="rating"):
    """Wrap a dense rating matrix into a NumericDataView (for catalog compat)."""
    n_users, n_items = R.shape
    observed_ratings = R[mask]
    user_indices, item_indices = np.where(mask)

    X = np.column_stack([
        user_indices.astype(float),
        item_indices.astype(float),
    ])
    y = observed_ratings

    if feature_names is None:
        feature_names = ["user_idx", "item_idx"]

    return NumericDataView(
        X_train=X,
        y_train=y,
        feature_names=list(feature_names),
        target_name=target_name,
    )
