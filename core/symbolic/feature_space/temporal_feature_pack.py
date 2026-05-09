from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np


_LAG_NAME_RE = re.compile(r"^(?P<source>[A-Za-z0-9_]+)_lag(?P<lag>\d+)$")


@dataclass(frozen=True)
class TemporalFeaturePackConfig:
    enabled: bool = True
    rolling_enabled: bool = True
    momentum_enabled: bool = True
    cross_enabled: bool = True
    ratio_enabled: bool = True
    cross_quantiles: tuple[float, ...] = (0.5,)
    safe_ratio_eps: float = 1e-3


@dataclass(frozen=True)
class TemporalFeaturePackResult:
    X_train: np.ndarray
    X_test: np.ndarray
    feature_names: tuple[str, ...]
    added_features: tuple[str, ...]
    rolling_added: tuple[str, ...]
    momentum_added: tuple[str, ...]
    cross_added: tuple[str, ...]
    ratio_added: tuple[str, ...]


def _lag_map(feature_names: Sequence[str]) -> dict[str, dict[int, int]]:
    out: dict[str, dict[int, int]] = {}
    for idx, name in enumerate(feature_names):
        m = _LAG_NAME_RE.match(str(name))
        if not m:
            continue
        src = str(m.group("source"))
        lag = int(m.group("lag"))
        out.setdefault(src, {})[lag] = int(idx)
    return out


def _safe_name_quantile(q: float) -> str:
    return f"q{int(round(float(q) * 100.0)):02d}"


def apply_temporal_feature_pack(
    *,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: Sequence[str],
    config: TemporalFeaturePackConfig,
) -> TemporalFeaturePackResult:
    xtr = np.asarray(X_train, dtype=float)
    xte = np.asarray(X_test, dtype=float)
    names = [str(v) for v in feature_names]
    if not bool(config.enabled):
        return TemporalFeaturePackResult(
            X_train=xtr,
            X_test=xte,
            feature_names=tuple(names),
            added_features=(),
            rolling_added=(),
            momentum_added=(),
            cross_added=(),
            ratio_added=(),
        )

    lag_map = _lag_map(names)
    tr_cols: list[np.ndarray] = [xtr]
    te_cols: list[np.ndarray] = [xte]
    ext_names: list[str] = list(names)
    added: list[str] = []
    rolling_added: list[str] = []
    momentum_added: list[str] = []
    cross_added: list[str] = []
    ratio_added: list[str] = []
    existing = set(ext_names)

    def _append(train_col: np.ndarray, test_col: np.ndarray, name: str, category: str) -> None:
        nm = str(name)
        if nm in existing:
            return
        tr_cols.append(np.asarray(train_col, dtype=float).reshape(-1, 1))
        te_cols.append(np.asarray(test_col, dtype=float).reshape(-1, 1))
        ext_names.append(nm)
        existing.add(nm)
        added.append(nm)
        if category == "rolling":
            rolling_added.append(nm)
        elif category == "momentum":
            momentum_added.append(nm)
        elif category == "cross":
            cross_added.append(nm)
        elif category == "ratio":
            ratio_added.append(nm)

    for src in ("ci", "total_flow", "avg_speed", "avg_occ"):
        src_lags = lag_map.get(str(src), {})
        lag1 = src_lags.get(1)
        lag2 = src_lags.get(2)
        lag3 = src_lags.get(3)
        available = [(lag, idx) for lag, idx in sorted(src_lags.items()) if lag in (1, 2, 3)]
        if bool(config.rolling_enabled) and len(available) >= 2:
            tr_mat = np.column_stack([xtr[:, idx] for _, idx in available])
            te_mat = np.column_stack([xte[:, idx] for _, idx in available])
            _append(np.mean(tr_mat, axis=1), np.mean(te_mat, axis=1), f"{src}_lagmean_{len(available)}", "rolling")
            _append(np.std(tr_mat, axis=1), np.std(te_mat, axis=1), f"{src}_lagstd_{len(available)}", "rolling")
        if bool(config.momentum_enabled) and lag1 is not None and lag2 is not None:
            tr_delta12 = np.asarray(xtr[:, lag1] - xtr[:, lag2], dtype=float)
            te_delta12 = np.asarray(xte[:, lag1] - xte[:, lag2], dtype=float)
            _append(tr_delta12, te_delta12, f"{src}_lagdelta_12", "momentum")
        if bool(config.momentum_enabled) and lag1 is not None and lag3 is not None:
            tr_delta13 = np.asarray(xtr[:, lag1] - xtr[:, lag3], dtype=float)
            te_delta13 = np.asarray(xte[:, lag1] - xte[:, lag3], dtype=float)
            _append(tr_delta13, te_delta13, f"{src}_lagdelta_13", "momentum")
            _append(0.5 * tr_delta13, 0.5 * te_delta13, f"{src}_lagslope_13", "momentum")

    ci_lag1 = lag_map.get("ci", {}).get(1)
    if bool(config.cross_enabled) and ci_lag1 is not None:
        ci_tr = np.asarray(xtr[:, ci_lag1], dtype=float).reshape(-1)
        ci_te = np.asarray(xte[:, ci_lag1], dtype=float).reshape(-1)
        for target_src in ("avg_speed", "avg_occ", "total_flow"):
            tgt_idx = lag_map.get(str(target_src), {}).get(1)
            if tgt_idx is None:
                continue
            tgt_tr = np.asarray(xtr[:, tgt_idx], dtype=float).reshape(-1)
            tgt_te = np.asarray(xte[:, tgt_idx], dtype=float).reshape(-1)
            _append(ci_tr * tgt_tr, ci_te * tgt_te, f"ci_lag1_x_{target_src}_lag1", "cross")
            for q in tuple(float(np.clip(v, 0.01, 0.99)) for v in config.cross_quantiles):
                c = float(np.quantile(ci_tr, q))
                tag = _safe_name_quantile(q)
                _append(
                    np.maximum(0.0, ci_tr - c) * tgt_tr,
                    np.maximum(0.0, ci_te - c) * tgt_te,
                    f"hinge_pos_ci_lag1_{tag}_x_{target_src}_lag1",
                    "cross",
                )
                _append(
                    np.maximum(0.0, c - ci_tr) * tgt_tr,
                    np.maximum(0.0, c - ci_te) * tgt_te,
                    f"hinge_neg_ci_lag1_{tag}_x_{target_src}_lag1",
                    "cross",
                )

    if bool(config.ratio_enabled):
        speed_idx = lag_map.get("avg_speed", {}).get(1)
        occ_idx = lag_map.get("avg_occ", {}).get(1)
        flow_idx = lag_map.get("total_flow", {}).get(1)
        eps = float(max(1e-8, config.safe_ratio_eps))
        if speed_idx is not None and occ_idx is not None:
            denom_tr = np.abs(np.asarray(xtr[:, speed_idx], dtype=float)) + eps
            denom_te = np.abs(np.asarray(xte[:, speed_idx], dtype=float)) + eps
            _append(
                np.asarray(xtr[:, occ_idx], dtype=float) / denom_tr,
                np.asarray(xte[:, occ_idx], dtype=float) / denom_te,
                "avg_occ_over_speed_lag1",
                "ratio",
            )
        if speed_idx is not None and flow_idx is not None:
            denom_tr = np.abs(np.asarray(xtr[:, speed_idx], dtype=float)) + eps
            denom_te = np.abs(np.asarray(xte[:, speed_idx], dtype=float)) + eps
            _append(
                np.asarray(xtr[:, flow_idx], dtype=float) / denom_tr,
                np.asarray(xte[:, flow_idx], dtype=float) / denom_te,
                "total_flow_over_speed_lag1",
                "ratio",
            )

    return TemporalFeaturePackResult(
        X_train=np.concatenate(tr_cols, axis=1),
        X_test=np.concatenate(te_cols, axis=1),
        feature_names=tuple(str(v) for v in ext_names),
        added_features=tuple(str(v) for v in added),
        rolling_added=tuple(str(v) for v in rolling_added),
        momentum_added=tuple(str(v) for v in momentum_added),
        cross_added=tuple(str(v) for v in cross_added),
        ratio_added=tuple(str(v) for v in ratio_added),
    )


__all__ = [
    "TemporalFeaturePackConfig",
    "TemporalFeaturePackResult",
    "apply_temporal_feature_pack",
]
