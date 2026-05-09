from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np


_LAG_NAME_RE = re.compile(r"^(?P<source>[A-Za-z0-9_]+)_lag(?P<lag>\d+)$")


@dataclass(frozen=True)
class RegimeFeaturePackConfig:
    enabled: bool = True
    volatility_enabled: bool = True
    shock_enabled: bool = True
    ci_regime_enabled: bool = True
    shock_quantiles: tuple[float, ...] = (0.8, 0.9)
    ci_regime_quantiles: tuple[float, ...] = (0.33, 0.66)
    eps: float = 1e-6


@dataclass(frozen=True)
class RegimeFeaturePackResult:
    X_train: np.ndarray
    X_test: np.ndarray
    feature_names: tuple[str, ...]
    added_features: tuple[str, ...]
    volatility_added: tuple[str, ...]
    shock_added: tuple[str, ...]
    ci_regime_added: tuple[str, ...]


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


def _q_tag(q: float) -> str:
    return f"q{int(round(float(q) * 100.0)):02d}"


def apply_regime_feature_pack(
    *,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: Sequence[str],
    config: RegimeFeaturePackConfig,
) -> RegimeFeaturePackResult:
    xtr = np.asarray(X_train, dtype=float)
    xte = np.asarray(X_test, dtype=float)
    names = [str(v) for v in feature_names]
    if not bool(config.enabled):
        return RegimeFeaturePackResult(
            X_train=xtr,
            X_test=xte,
            feature_names=tuple(names),
            added_features=(),
            volatility_added=(),
            shock_added=(),
            ci_regime_added=(),
        )

    lag_map = _lag_map(names)
    tr_cols: list[np.ndarray] = [xtr]
    te_cols: list[np.ndarray] = [xte]
    ext_names: list[str] = list(names)
    added: list[str] = []
    volatility_added: list[str] = []
    shock_added: list[str] = []
    ci_regime_added: list[str] = []
    existing = set(ext_names)
    eps = float(max(1e-10, config.eps))

    def _append(train_col: np.ndarray, test_col: np.ndarray, name: str, category: str) -> None:
        nm = str(name)
        if nm in existing:
            return
        tr_cols.append(np.asarray(train_col, dtype=float).reshape(-1, 1))
        te_cols.append(np.asarray(test_col, dtype=float).reshape(-1, 1))
        ext_names.append(nm)
        existing.add(nm)
        added.append(nm)
        if category == "volatility":
            volatility_added.append(nm)
        elif category == "shock":
            shock_added.append(nm)
        elif category == "ci_regime":
            ci_regime_added.append(nm)

    for src in ("ci", "total_flow", "avg_speed", "avg_occ"):
        src_lags = lag_map.get(src, {})
        i1 = src_lags.get(1)
        i2 = src_lags.get(2)
        i3 = src_lags.get(3)
        if i1 is None or i2 is None:
            continue
        d12_tr = np.asarray(xtr[:, i1] - xtr[:, i2], dtype=float)
        d12_te = np.asarray(xte[:, i1] - xte[:, i2], dtype=float)
        if bool(config.volatility_enabled):
            _append(np.abs(d12_tr), np.abs(d12_te), f"{src}_lagabsdelta_12", "volatility")
            if i3 is not None:
                d23_tr = np.asarray(xtr[:, i2] - xtr[:, i3], dtype=float)
                d23_te = np.asarray(xte[:, i2] - xte[:, i3], dtype=float)
                _append(np.abs(d23_tr), np.abs(d23_te), f"{src}_lagabsdelta_23", "volatility")
                _append(np.abs(d12_tr - d23_tr), np.abs(d12_te - d23_te), f"{src}_lagaccel_abs", "volatility")
        if bool(config.shock_enabled):
            scale = float(np.std(d12_tr) + eps)
            z_tr = np.abs(d12_tr) / scale
            z_te = np.abs(d12_te) / scale
            _append(z_tr, z_te, f"{src}_lagshock_z", "shock")
            for q in tuple(float(np.clip(v, 0.5, 0.999)) for v in config.shock_quantiles):
                c = float(np.quantile(z_tr, q))
                tag = _q_tag(q)
                _append((z_tr > c).astype(float), (z_te > c).astype(float), f"{src}_lagshock_flag_{tag}", "shock")

    ci_lag1 = lag_map.get("ci", {}).get(1)
    if bool(config.ci_regime_enabled) and ci_lag1 is not None:
        ci_tr = np.asarray(xtr[:, ci_lag1], dtype=float)
        ci_te = np.asarray(xte[:, ci_lag1], dtype=float)
        qvals = [float(np.quantile(ci_tr, float(np.clip(q, 0.01, 0.99)))) for q in config.ci_regime_quantiles]
        if qvals:
            qvals = sorted(qvals)
            _append((ci_tr <= qvals[0]).astype(float), (ci_te <= qvals[0]).astype(float), "ci_regime_low", "ci_regime")
            _append((ci_tr > qvals[-1]).astype(float), (ci_te > qvals[-1]).astype(float), "ci_regime_high", "ci_regime")
            if len(qvals) >= 2:
                _append(
                    ((ci_tr > qvals[0]) & (ci_tr <= qvals[-1])).astype(float),
                    ((ci_te > qvals[0]) & (ci_te <= qvals[-1])).astype(float),
                    "ci_regime_mid",
                    "ci_regime",
                )

    return RegimeFeaturePackResult(
        X_train=np.concatenate(tr_cols, axis=1),
        X_test=np.concatenate(te_cols, axis=1),
        feature_names=tuple(str(v) for v in ext_names),
        added_features=tuple(str(v) for v in added),
        volatility_added=tuple(str(v) for v in volatility_added),
        shock_added=tuple(str(v) for v in shock_added),
        ci_regime_added=tuple(str(v) for v in ci_regime_added),
    )


__all__ = [
    "RegimeFeaturePackConfig",
    "RegimeFeaturePackResult",
    "apply_regime_feature_pack",
]
