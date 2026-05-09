from __future__ import annotations

from typing import Sequence

import numpy as np

from .contracts import DecodedBatchEvaluationFn, DecodedDecision, DecodedEvaluationFn, DecisionDecodeFn


class DecisionEvaluationBridge:
    """Decode decisions then delegate scoring to MLBLACK-side evaluators."""

    def __init__(
        self,
        *,
        decode_fn: DecisionDecodeFn,
        evaluate_decoded_fn: DecodedEvaluationFn,
        evaluate_decoded_batch_fn: DecodedBatchEvaluationFn | None = None,
        objective_dim: int = 3,
        fallback_objectives: Sequence[float] | None = None,
    ) -> None:
        self._decode_fn = decode_fn
        self._evaluate_decoded_fn = evaluate_decoded_fn
        self._evaluate_decoded_batch_fn = evaluate_decoded_batch_fn
        self._objective_dim = int(max(1, objective_dim))
        if fallback_objectives is None:
            fallback_objectives = [1e6] + [1e3 for _ in range(max(0, self._objective_dim - 1))]
        fb = np.asarray(list(fallback_objectives), dtype=float).reshape(-1)
        if fb.size < self._objective_dim:
            fb = np.pad(fb, (0, int(self._objective_dim - fb.size)), constant_values=1e3)
        self._fallback = np.asarray(fb[: self._objective_dim], dtype=float)

    def _decode_one(self, x: np.ndarray) -> DecodedDecision:
        subset_idx, _k, meta = self._decode_fn(np.asarray(x, dtype=float))
        return DecodedDecision(
            subset_idx=tuple(sorted(int(v) for v in subset_idx)),
            meta=dict(meta),
        )

    def evaluate_population(self, population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pop = np.asarray(population, dtype=float)
        if pop.ndim == 1:
            pop = pop.reshape(1, -1)
        n = int(pop.shape[0])
        out_obj = np.zeros((n, self._objective_dim), dtype=float)
        out_vio = np.zeros((n,), dtype=float)
        if n <= 0:
            return out_obj, out_vio

        decoded_valid: list[DecodedDecision] = []
        valid_indices: list[int] = []
        for i in range(n):
            try:
                decoded = self._decode_one(pop[i])
                decoded_valid.append(decoded)
                valid_indices.append(int(i))
            except Exception:
                out_obj[i] = self._fallback

        if decoded_valid:
            try:
                if self._evaluate_decoded_batch_fn is not None:
                    batch_obj = np.asarray(self._evaluate_decoded_batch_fn(decoded_valid), dtype=float)
                    if batch_obj.ndim == 1:
                        batch_obj = batch_obj.reshape(1, -1)
                    if int(batch_obj.shape[0]) != int(len(decoded_valid)):
                        raise ValueError("batch evaluator returned mismatched candidate count")
                    for loc, i in enumerate(valid_indices):
                        row = np.asarray(batch_obj[loc], dtype=float).reshape(-1)
                        if row.size < self._objective_dim:
                            row = np.pad(row, (0, int(self._objective_dim - row.size)), constant_values=1e3)
                        out_obj[i] = row[: self._objective_dim]
                else:
                    for dec, i in zip(decoded_valid, valid_indices):
                        row = np.asarray(self._evaluate_decoded_fn(dec), dtype=float).reshape(-1)
                        if row.size < self._objective_dim:
                            row = np.pad(row, (0, int(self._objective_dim - row.size)), constant_values=1e3)
                        out_obj[i] = row[: self._objective_dim]
            except Exception:
                for i in valid_indices:
                    out_obj[i] = self._fallback

        return out_obj, out_vio

    def evaluate_one(self, x: np.ndarray) -> np.ndarray:
        try:
            decoded = self._decode_one(np.asarray(x, dtype=float))
            row = np.asarray(self._evaluate_decoded_fn(decoded), dtype=float).reshape(-1)
            if row.size < self._objective_dim:
                row = np.pad(row, (0, int(self._objective_dim - row.size)), constant_values=1e3)
            return np.asarray(row[: self._objective_dim], dtype=float)
        except Exception:
            return np.asarray(self._fallback, dtype=float)


__all__ = ["DecisionEvaluationBridge"]
