from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from core.symbolic.symbolic_gradient import evaluate_gradient_numpy


def _as_2d_float(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError("array must be 1D or 2D")
    return out


@dataclass(frozen=True)
class GradientSignal:
    overall_mismatch: float
    feature_mismatch: np.ndarray
    feature_priority: np.ndarray
    feature_priority_multiscale: np.ndarray
    feature_stability: np.ndarray
    feature_gap_signed_mean: np.ndarray
    feature_gap_abs_mean: np.ndarray
    feature_valid_fraction: np.ndarray
    cross_feature_priority: np.ndarray
    gap_by_feature: tuple[np.ndarray, ...]
    signal_signature: str


class GradientParser:
    """Build gradient-level diagnostics for symbolic structure updates."""

    @staticmethod
    def _local_slope_1d(x_col: np.ndarray, y_mat: np.ndarray) -> np.ndarray:
        x = np.asarray(x_col, dtype=float).reshape(-1)
        y = _as_2d_float(y_mat)
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y row mismatch for local slope")

        n = int(x.shape[0])
        m = int(y.shape[1])
        out = np.full((n, m), np.nan, dtype=float)
        if n < 3:
            return out

        order = np.argsort(x)
        inv = np.empty_like(order)
        inv[order] = np.arange(n)

        xs = x[order]
        ys = y[order]

        dx = xs[2:] - xs[:-2]
        dy = ys[2:, :] - ys[:-2, :]

        middle = np.full((n - 2, m), np.nan, dtype=float)
        valid = np.abs(dx) > 1e-12
        if np.any(valid):
            middle[valid, :] = dy[valid, :] / dx[valid, None]

        sorted_out = np.full((n, m), np.nan, dtype=float)
        sorted_out[1:-1, :] = middle
        sorted_out[0, :] = sorted_out[1, :]
        sorted_out[-1, :] = sorted_out[-2, :]

        return sorted_out[inv, :]

    @classmethod
    def _local_slope_binned_median(
        cls,
        x_col: np.ndarray,
        y_mat: np.ndarray,
        *,
        bins: int,
        min_bin_samples: int,
    ) -> np.ndarray:
        x = np.asarray(x_col, dtype=float).reshape(-1)
        y = _as_2d_float(y_mat)
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y row mismatch for local slope")

        n = int(x.shape[0])
        m = int(y.shape[1])
        if n < 3:
            return np.full((n, m), np.nan, dtype=float)

        min_count = max(3, int(min_bin_samples))
        max_bins = max(3, int(bins))
        if n < 2 * min_count:
            return cls._local_slope_1d(x, y)

        order = np.argsort(x)
        inv = np.empty_like(order)
        inv[order] = np.arange(n)

        xs = x[order]
        ys = y[order]

        n_bins = max(3, min(max_bins, n // min_count))
        chunks = [chunk for chunk in np.array_split(np.arange(n, dtype=int), n_bins) if int(chunk.size) > 0]

        valid_chunks: list[np.ndarray] = []
        bx: list[float] = []
        by: list[np.ndarray] = []
        for chunk in chunks:
            if int(chunk.size) < min_count:
                continue
            valid_chunks.append(chunk)
            bx.append(float(np.median(xs[chunk])))
            by.append(np.median(ys[chunk, :], axis=0))

        if len(valid_chunks) < 3:
            return cls._local_slope_1d(x, y)

        b = int(len(valid_chunks))
        bx_arr = np.asarray(bx, dtype=float)
        by_mat = np.asarray(np.vstack(by), dtype=float)

        slopes = np.full((b, m), np.nan, dtype=float)
        for k in range(1, b - 1):
            den = float(bx_arr[k + 1] - bx_arr[k - 1])
            if abs(den) <= 1e-12:
                continue
            slopes[k, :] = (by_mat[k + 1, :] - by_mat[k - 1, :]) / den

        slopes[0, :] = slopes[1, :]
        slopes[-1, :] = slopes[-2, :]

        sorted_out = np.full((n, m), np.nan, dtype=float)
        for chunk, slope in zip(valid_chunks, slopes):
            sorted_out[chunk, :] = slope

        finite_rows = np.where(np.all(np.isfinite(sorted_out), axis=1))[0]
        if finite_rows.size == 0:
            return cls._local_slope_1d(x, y)

        bad_rows = np.where(~np.all(np.isfinite(sorted_out), axis=1))[0]
        for r in bad_rows:
            nearest = int(finite_rows[np.argmin(np.abs(finite_rows - r))])
            sorted_out[r, :] = sorted_out[nearest, :]

        return sorted_out[inv, :]

    @classmethod
    def _local_slope(
        cls,
        x_col: np.ndarray,
        y_mat: np.ndarray,
        *,
        mode: str,
        bins: int,
        min_bin_samples: int,
    ) -> np.ndarray:
        key = str(mode).strip().lower()
        if key in {"central_diff", "diff", "default"}:
            return cls._local_slope_1d(x_col, y_mat)
        if key in {"binned_median", "robust", "bin_median"}:
            return cls._local_slope_binned_median(
                x_col,
                y_mat,
                bins=int(bins),
                min_bin_samples=int(min_bin_samples),
            )
        raise ValueError(f"Unsupported slope mode: {mode}")

    @staticmethod
    def model_partial_derivative(
        genome: Sequence[Mapping[str, Any]],
        weight: np.ndarray,
        X: np.ndarray,
        *,
        feature_index: int,
        graph_cache: Any | None = None,
        batch_key: str | None = None,
    ) -> np.ndarray:
        terms = list(genome)
        x = _as_2d_float(np.asarray(X, dtype=float))
        w = _as_2d_float(np.asarray(weight, dtype=float))

        n = int(x.shape[0])
        m = int(w.shape[1])
        if w.shape[0] != len(terms):
            raise ValueError("weight term dimension mismatch")

        out = np.zeros((n, m), dtype=float)
        for i in range(len(terms)):
            try:
                g = evaluate_gradient_numpy(
                    terms[i]["expr"],
                    x,
                    feature_index=int(feature_index),
                    graph_cache=graph_cache,
                    batch_key=batch_key,
                ).reshape(-1)
            except Exception:
                continue

            if g.shape[0] != n:
                continue
            if not np.all(np.isfinite(g)):
                continue

            out += g.reshape(-1, 1) * w[i, :].reshape(1, -1)

        return out

    @staticmethod
    def _nanmean_or_zero(arr: np.ndarray) -> float:
        a = np.asarray(arr, dtype=float)
        mask = np.isfinite(a)
        if not bool(np.any(mask)):
            return 0.0
        return float(np.mean(a[mask]))

    @classmethod
    def build_signal(
        cls,
        *,
        genome: Sequence[Mapping[str, Any]],
        weight: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        slope_mode: str = "central_diff",
        slope_bins: int = 24,
        slope_min_bin_samples: int = 12,
        graph_cache: Any | None = None,
        batch_key: str | None = None,
    ) -> GradientSignal:
        x = _as_2d_float(np.asarray(X, dtype=float))
        yt = _as_2d_float(np.asarray(y, dtype=float))

        if x.shape[0] != yt.shape[0]:
            raise ValueError("X and y row mismatch")

        n, d = x.shape
        m = int(yt.shape[1])
        if n < 3:
            zeros_d = np.zeros((d,), dtype=float)
            return GradientSignal(
                overall_mismatch=0.0,
                feature_mismatch=zeros_d.copy(),
                feature_priority=(np.ones((d,), dtype=float) / max(1, d)),
                feature_priority_multiscale=(np.ones((d,), dtype=float) / max(1, d)),
                feature_stability=np.ones((d,), dtype=float),
                feature_gap_signed_mean=zeros_d.copy(),
                feature_gap_abs_mean=zeros_d.copy(),
                feature_valid_fraction=zeros_d.copy(),
                cross_feature_priority=np.zeros((d, d), dtype=float),
                gap_by_feature=tuple(np.zeros((n, m), dtype=float) for _ in range(d)),
                signal_signature="",
            )

        mismatch = np.zeros((d,), dtype=float)
        mismatch_aux = np.zeros((d,), dtype=float)
        feature_stability = np.ones((d,), dtype=float)
        gap_signed_mean = np.zeros((d,), dtype=float)
        gap_abs_mean = np.zeros((d,), dtype=float)
        valid_fraction = np.zeros((d,), dtype=float)
        gap_list: list[np.ndarray] = []
        row_gap_mean_list: list[np.ndarray] = []

        aux_mode = "binned_median" if str(slope_mode).strip().lower() in {"central_diff", "diff", "default"} else "central_diff"

        for j in range(d):
            slope_true = cls._local_slope(
                x[:, j],
                yt,
                mode=str(slope_mode),
                bins=int(slope_bins),
                min_bin_samples=int(slope_min_bin_samples),
            )
            slope_pred = cls.model_partial_derivative(
                genome,
                weight,
                x,
                feature_index=j,
                graph_cache=graph_cache,
                batch_key=batch_key,
            )
            slope_true_aux = cls._local_slope(
                x[:, j],
                yt,
                mode=str(aux_mode),
                bins=max(8, int(slope_bins // 2)),
                min_bin_samples=max(6, int(slope_min_bin_samples // 2)),
            )

            gap = slope_true - slope_pred
            gap_aux = slope_true_aux - slope_pred
            valid = np.isfinite(slope_true) & np.isfinite(slope_pred)
            valid_aux = np.isfinite(slope_true_aux) & np.isfinite(slope_pred)

            gap_store = np.full((n, m), np.nan, dtype=float)
            gap_store[valid] = gap[valid]
            gap_list.append(gap_store)

            row_gap_mean = np.full((n,), np.nan, dtype=float)
            if bool(np.any(valid)):
                valid_count_row = np.sum(valid, axis=1)
                has_valid_row = valid_count_row > 0
                if bool(np.any(has_valid_row)):
                    row_num = np.sum(np.where(valid, gap, 0.0), axis=1)
                    row_gap_mean[has_valid_row] = row_num[has_valid_row] / valid_count_row[has_valid_row]
            row_gap_mean_list.append(np.asarray(row_gap_mean, dtype=float))

            valid_fraction[j] = float(np.mean(valid.astype(float)))
            gap_signed_mean[j] = cls._nanmean_or_zero(gap_store)
            gap_abs_mean[j] = cls._nanmean_or_zero(np.abs(gap_store))

            if bool(np.any(valid)):
                scale = float(np.std(slope_true[valid])) + 1e-8
                err = (gap[valid] / scale) ** 2
                mismatch[j] = float(np.sqrt(np.mean(err)))
            else:
                mismatch[j] = 0.0

            if bool(np.any(valid_aux)):
                scale_aux = float(np.std(slope_true_aux[valid_aux])) + 1e-8
                err_aux = (gap_aux[valid_aux] / scale_aux) ** 2
                mismatch_aux[j] = float(np.sqrt(np.mean(err_aux)))
            else:
                mismatch_aux[j] = mismatch[j]

            den = abs(float(mismatch[j])) + abs(float(mismatch_aux[j])) + 1e-12
            st = 1.0 - abs(float(mismatch[j]) - float(mismatch_aux[j])) / den
            feature_stability[j] = float(np.clip(st, 0.0, 1.0))

        mismatch_ms = 0.5 * np.maximum(mismatch, 0.0) + 0.5 * np.maximum(mismatch_aux, 0.0)
        total = float(np.sum(mismatch_ms))
        if total <= 1e-12:
            priority = np.ones((d,), dtype=float) / max(1, d)
        else:
            # Downweight unstable directions to improve candidate precision.
            p_raw = mismatch_ms * (0.20 + 0.80 * np.clip(feature_stability, 0.0, 1.0))
            p_sum = float(np.sum(np.maximum(p_raw, 0.0)))
            if p_sum <= 1e-12:
                priority = np.ones((d,), dtype=float) / max(1, d)
            else:
                priority = np.maximum(p_raw, 0.0) / p_sum

        pms_sum = float(np.sum(np.maximum(mismatch_ms, 0.0)))
        if pms_sum <= 1e-12:
            priority_ms = np.ones((d,), dtype=float) / max(1, d)
        else:
            priority_ms = np.maximum(mismatch_ms, 0.0) / pms_sum

        # Cross-feature interaction prior: how much u_i can be explained by x_j (i != j).
        cross = np.zeros((d, d), dtype=float)
        for i in range(d):
            ui = np.asarray(row_gap_mean_list[i], dtype=float).reshape(-1)
            for j in range(d):
                if i == j:
                    continue
                xj = np.asarray(x[:, j], dtype=float).reshape(-1)
                valid_ij = np.isfinite(ui) & np.isfinite(xj)
                if not bool(np.any(valid_ij)):
                    continue
                uv = ui[valid_ij]
                xv = xj[valid_ij]
                if uv.size < 5:
                    continue
                uc = uv - float(np.mean(uv))
                xc = xv - float(np.mean(xv))
                den = float(np.sqrt(np.dot(uc, uc) * np.dot(xc, xc))) + 1e-12
                if den <= 1e-12:
                    continue
                corr = float(np.dot(uc, xc) / den)
                cross[i, j] = float(abs(corr) * priority[i] * (0.25 + 0.75 * priority_ms[j]))

        overall = float(np.sqrt(np.mean(mismatch**2))) if d > 0 else 0.0
        sig_payload = {
            "overall": round(float(overall), 8),
            "priority": np.asarray(priority, dtype=float).round(6).tolist(),
            "stability": np.asarray(feature_stability, dtype=float).round(6).tolist(),
            "mismatch": np.asarray(mismatch, dtype=float).round(6).tolist(),
        }
        signature = hashlib.md5(json.dumps(sig_payload, sort_keys=True).encode("utf-8")).hexdigest()

        return GradientSignal(
            overall_mismatch=overall,
            feature_mismatch=np.asarray(mismatch, dtype=float),
            feature_priority=np.asarray(priority, dtype=float),
            feature_priority_multiscale=np.asarray(priority_ms, dtype=float),
            feature_stability=np.asarray(feature_stability, dtype=float),
            feature_gap_signed_mean=np.asarray(gap_signed_mean, dtype=float),
            feature_gap_abs_mean=np.asarray(gap_abs_mean, dtype=float),
            feature_valid_fraction=np.asarray(valid_fraction, dtype=float),
            cross_feature_priority=np.asarray(cross, dtype=float),
            gap_by_feature=tuple(np.asarray(g, dtype=float) for g in gap_list),
            signal_signature=str(signature),
        )

    @classmethod
    def gradient_mismatch(
        cls,
        *,
        genome: Sequence[Mapping[str, Any]],
        weight: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        slope_mode: str = "central_diff",
        slope_bins: int = 24,
        slope_min_bin_samples: int = 12,
        graph_cache: Any | None = None,
        batch_key: str | None = None,
    ) -> float:
        signal = cls.build_signal(
            genome=genome,
            weight=weight,
            X=X,
            y=y,
            slope_mode=str(slope_mode),
            slope_bins=int(slope_bins),
            slope_min_bin_samples=int(slope_min_bin_samples),
            graph_cache=graph_cache,
            batch_key=batch_key,
        )
        return float(signal.overall_mismatch)


__all__ = [
    "GradientSignal",
    "GradientParser",
]
