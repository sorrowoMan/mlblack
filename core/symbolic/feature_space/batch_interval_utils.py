from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from core.symbolic.expression_graph_cache import ExpressionGraphCache
from core.symbolic.symbolic_dsl import evaluate_genome_numpy
from core.symbolic.symbolic_structure_search import evaluate_genome_with_ridge


def as_2d(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("array must be 2D")
    return arr


def design_matrix_for_genome(
    genome: Sequence[Mapping[str, Any]],
    X: np.ndarray,
    *,
    graph_cache: ExpressionGraphCache | None = None,
    batch_key: str | None = None,
) -> np.ndarray:
    x = np.asarray(X, dtype=float)
    if x.ndim != 2:
        raise ValueError("X must be 2D")
    if len(genome) <= 0:
        return np.zeros((int(x.shape[0]), 0), dtype=float)
    if graph_cache is None:
        phi = evaluate_genome_numpy(genome, x)
        return np.asarray(phi, dtype=float)
    cols: list[np.ndarray] = []
    for term in genome:
        expr = term.get("expr", term)
        z = graph_cache.evaluate_expression(
            expr,
            x,
            param_values=None,
            eps=1e-6,
            batch_key=batch_key,
        )
        cols.append(np.asarray(z, dtype=float).reshape(-1, 1))
    if not cols:
        return np.zeros((int(x.shape[0]), 0), dtype=float)
    return np.concatenate(cols, axis=1)


def batched_ridge_predict(
    *,
    genomes: Sequence[Sequence[Mapping[str, Any]]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    l2_values: Sequence[float],
    graph_cache: ExpressionGraphCache | None = None,
    batch_key_train: str | None = None,
    batch_key_eval: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    xtr = np.asarray(X_train, dtype=float)
    ytr = as_2d(np.asarray(y_train, dtype=float))
    xev = np.asarray(X_eval, dtype=float)
    batch_size = int(len(genomes))
    if batch_size <= 0:
        return (
            np.zeros((0, int(xev.shape[0]), int(ytr.shape[1])), dtype=float),
            np.zeros((0, int(xtr.shape[0]), int(ytr.shape[1])), dtype=float),
        )

    groups: dict[int, list[int]] = {}
    for idx, genome in enumerate(genomes):
        groups.setdefault(int(len(genome)), []).append(int(idx))

    pred_eval = np.zeros((batch_size, int(xev.shape[0]), int(ytr.shape[1])), dtype=float)
    pred_train = np.zeros((batch_size, int(xtr.shape[0]), int(ytr.shape[1])), dtype=float)

    try:
        import torch
    except Exception:
        for idx, genome in enumerate(genomes):
            fit = evaluate_genome_with_ridge(
                genome,
                X_train=xtr,
                y_train=ytr,
                X_eval=xev,
                y_eval=None,
                l2=float(max(0.0, l2_values[idx])),
            )
            pred_eval[idx] = as_2d(np.asarray(fit.get("pred_eval"), dtype=float))
            pred_train[idx] = as_2d(np.asarray(fit.get("pred_train"), dtype=float))
        return pred_eval, pred_train

    ytr_t = torch.as_tensor(ytr, dtype=torch.float64)

    for term_count, idxs in groups.items():
        if int(term_count) <= 0:
            intercept = np.mean(ytr, axis=0, keepdims=True)
            for idx in idxs:
                pred_train[idx] = np.repeat(intercept, repeats=int(xtr.shape[0]), axis=0)
                pred_eval[idx] = np.repeat(intercept, repeats=int(xev.shape[0]), axis=0)
            continue

        phis_tr = []
        phis_ev = []
        reg_vals = []
        for idx in idxs:
            genome = genomes[int(idx)]
            phis_tr.append(
                design_matrix_for_genome(
                    genome,
                    xtr,
                    graph_cache=graph_cache,
                    batch_key=batch_key_train,
                )
            )
            phis_ev.append(
                design_matrix_for_genome(
                    genome,
                    xev,
                    graph_cache=graph_cache,
                    batch_key=batch_key_eval,
                )
            )
            reg_vals.append(float(max(0.0, l2_values[int(idx)])))

        A_tr = np.asarray(np.stack(phis_tr, axis=0), dtype=float)
        A_ev = np.asarray(np.stack(phis_ev, axis=0), dtype=float)
        batch_group = int(A_tr.shape[0])
        ones_tr = np.ones((batch_group, int(A_tr.shape[1]), 1), dtype=float)
        ones_ev = np.ones((batch_group, int(A_ev.shape[1]), 1), dtype=float)
        Atr = np.concatenate([A_tr, ones_tr], axis=2)
        Aev = np.concatenate([A_ev, ones_ev], axis=2)

        Atr_t = torch.as_tensor(Atr, dtype=torch.float64)
        Aev_t = torch.as_tensor(Aev, dtype=torch.float64)
        yb_t = ytr_t.unsqueeze(0).expand(batch_group, -1, -1)

        At = Atr_t.transpose(1, 2)
        lhs = torch.bmm(At, Atr_t)
        rhs = torch.bmm(At, yb_t)

        reg = torch.eye(int(term_count + 1), dtype=torch.float64).unsqueeze(0).repeat(batch_group, 1, 1)
        reg[:, -1, -1] = 0.0
        lam = torch.as_tensor(np.asarray(reg_vals, dtype=float), dtype=torch.float64).reshape(batch_group, 1, 1)
        lhs = lhs + lam * reg

        try:
            weights = torch.linalg.solve(lhs, rhs)
        except Exception:
            weights = torch.matmul(torch.linalg.pinv(lhs), rhs)

        pred_tr_g = torch.bmm(Atr_t, weights).cpu().numpy()
        pred_ev_g = torch.bmm(Aev_t, weights).cpu().numpy()
        for loc, idx in enumerate(idxs):
            pred_train[int(idx)] = np.asarray(pred_tr_g[int(loc)], dtype=float)
            pred_eval[int(idx)] = np.asarray(pred_ev_g[int(loc)], dtype=float)

    return pred_eval, pred_train


def symmetric_interval_batch(
    *,
    y_train: np.ndarray,
    pred_train: np.ndarray,
    pred_eval: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yt = as_2d(np.asarray(y_train, dtype=float)).reshape(1, -1, 1)
    pt = np.asarray(pred_train, dtype=float)
    pe = np.asarray(pred_eval, dtype=float)
    a = float(np.clip(alpha, 1e-6, 0.99))
    if pt.ndim != 3 or pe.ndim != 3:
        raise ValueError("pred_train and pred_eval must be rank-3 batch tensors")
    q = np.quantile(np.abs(pt - yt).reshape(int(pt.shape[0]), -1), 1.0 - a, axis=1)
    lower = pe - q[:, None, None]
    upper = pe + q[:, None, None]
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float), np.asarray(q, dtype=float)


def interval_metrics_batch(
    *,
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    y = as_2d(np.asarray(y_true, dtype=float)).reshape(1, -1, 1)
    lo = np.asarray(lower, dtype=float)
    up = np.asarray(upper, dtype=float)
    a = float(np.clip(alpha, 1e-6, 0.99))
    inside = np.logical_and(y >= lo, y <= up)
    picp = np.mean(inside, axis=(1, 2))
    width = np.asarray(up - lo, dtype=float)
    y_range = float(max(1e-8, float(np.max(y) - np.min(y))))
    pinaw = np.mean(width, axis=(1, 2)) / y_range
    below = np.asarray(lo - y, dtype=float)
    above = np.asarray(y - up, dtype=float)
    interval_score = np.mean(
        width + (2.0 / a) * np.maximum(0.0, below) + (2.0 / a) * np.maximum(0.0, above),
        axis=(1, 2),
    )
    coverage_target = float(1.0 - a)
    coverage_error = np.abs(picp - coverage_target)
    return {
        "coverage_error": np.asarray(coverage_error, dtype=float),
        "picp": np.asarray(picp, dtype=float),
        "pinaw": np.asarray(pinaw, dtype=float),
        "interval_score": np.asarray(interval_score, dtype=float),
        "mean_width": np.asarray(np.mean(width, axis=(1, 2)), dtype=float),
        "coverage_target": np.full((int(lo.shape[0]),), coverage_target, dtype=float),
    }


__all__ = [
    "as_2d",
    "design_matrix_for_genome",
    "batched_ridge_predict",
    "symmetric_interval_batch",
    "interval_metrics_batch",
]
