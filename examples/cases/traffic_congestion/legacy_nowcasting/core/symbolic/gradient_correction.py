from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from core.symbolic.gradient_parser import GradientSignal
from core.symbolic.symbolic_gradient import evaluate_gradient_numpy


def _as_2d_float(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError("array must be 1D or 2D")
    return out


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("corr shape mismatch")

    valid = np.isfinite(x) & np.isfinite(y)
    if not bool(np.any(valid)):
        return 0.0

    xv = x[valid]
    yv = y[valid]
    xc = xv - float(np.mean(xv))
    yc = yv - float(np.mean(yv))
    denom = float(np.sqrt(np.dot(xc, xc) * np.dot(yc, yc))) + 1e-12
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(xc, yc) / denom)


@dataclass(frozen=True)
class GradientCorrectionConfig:
    focus_topk_features: int = 3
    min_priority: float = 1e-4
    interaction_bonus_scale: float = 0.15
    stability_weight_floor: float = 0.25


class GradientCorrection:
    """Convert gradient signal into direction-aware candidate guidance."""

    def __init__(
        self,
        signal: GradientSignal,
        *,
        config: GradientCorrectionConfig | None = None,
    ) -> None:
        self.signal = signal
        self.config = config or GradientCorrectionConfig()
        self._active_features = self._build_active_feature_index()

    def _build_active_feature_index(self) -> tuple[int, ...]:
        p = np.asarray(self.signal.feature_priority, dtype=float).reshape(-1)
        if p.size == 0:
            return tuple()

        ranked = list(np.argsort(-p))
        topk = max(1, min(int(self.config.focus_topk_features), int(p.size)))

        keep: list[int] = []
        for j in ranked[:topk]:
            if float(p[j]) >= float(self.config.min_priority):
                keep.append(int(j))

        if not keep:
            keep.append(int(ranked[0]))

        return tuple(keep)

    @property
    def active_features(self) -> tuple[int, ...]:
        return self._active_features

    def score_candidate(
        self,
        *,
        expr: Mapping[str, Any],
        X: np.ndarray,
        coeff_vector: np.ndarray,
        feature_indices: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        x = _as_2d_float(np.asarray(X, dtype=float))
        coeff = np.asarray(coeff_vector, dtype=float).reshape(-1)
        if coeff.size == 0:
            return {
                "grad_alignment": 0.0,
                "used_features": [],
                "per_feature": [],
            }

        if feature_indices is None:
            cand_features = tuple(self._active_features)
        else:
            cand_features = tuple(int(i) for i in feature_indices)

        candidate_set = set(cand_features)
        used: list[int] = []
        parts: list[dict[str, Any]] = []
        weighted_sum = 0.0
        weight_sum = 0.0

        for j in self._active_features:
            if j not in candidate_set:
                continue
            if j < 0 or j >= len(self.signal.gap_by_feature):
                continue

            gap = np.asarray(self.signal.gap_by_feature[j], dtype=float)
            if gap.shape[0] != x.shape[0]:
                continue

            try:
                dterm = evaluate_gradient_numpy(expr, x, feature_index=int(j)).reshape(-1)
            except Exception:
                continue

            if dterm.shape[0] != x.shape[0] or not np.all(np.isfinite(dterm)):
                continue

            contrib = dterm.reshape(-1, 1) * coeff.reshape(1, -1)
            if gap.shape[1] != contrib.shape[1]:
                min_m = min(int(gap.shape[1]), int(contrib.shape[1]))
                gap = gap[:, :min_m]
                contrib = contrib[:, :min_m]

            corr = _safe_corr(contrib, gap)
            base_w = float(self.signal.feature_priority[j])
            st = 1.0
            try:
                st = float(np.asarray(self.signal.feature_stability, dtype=float).reshape(-1)[j])
            except Exception:
                st = 1.0
            w = float(base_w * max(float(self.config.stability_weight_floor), st))

            weighted_sum += w * corr
            weight_sum += w
            used.append(int(j))
            parts.append(
                {
                    "feature_index": int(j),
                    "priority": float(w),
                    "corr": float(corr),
                    "signed_gap_mean": float(self.signal.feature_gap_signed_mean[j]),
                    "abs_gap_mean": float(self.signal.feature_gap_abs_mean[j]),
                }
            )

        align = 0.0 if weight_sum <= 1e-12 else float(weighted_sum / weight_sum)

        # Pairwise interaction prior from GradientParser cross-feature matrix.
        inter_bonus = 0.0
        if feature_indices is not None and len(tuple(feature_indices)) >= 2:
            try:
                cross = np.asarray(self.signal.cross_feature_priority, dtype=float)
                feat = [int(i) for i in feature_indices]
                bonus_vals: list[float] = []
                for a in feat:
                    for b in feat:
                        if a == b:
                            continue
                        if 0 <= a < cross.shape[0] and 0 <= b < cross.shape[1]:
                            bonus_vals.append(float(cross[a, b]))
                if bonus_vals:
                    inter_bonus = float(np.mean(np.asarray(bonus_vals, dtype=float)))
            except Exception:
                inter_bonus = 0.0

        align_total = float(align + float(self.config.interaction_bonus_scale) * float(inter_bonus))
        return {
            "grad_alignment": float(align_total),
            "grad_alignment_base": float(align),
            "grad_interaction_bonus": float(inter_bonus),
            "used_features": [int(i) for i in used],
            "per_feature": parts,
        }


__all__ = [
    "GradientCorrectionConfig",
    "GradientCorrection",
]
