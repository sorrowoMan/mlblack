from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.symbolic.feature_space.lag_utils import make_lag_from_history, parse_float_list_csv, parse_int_list_csv


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    lag_feature_enabled: bool = True
    lag_orders_csv: str = "1,2,3"
    lag_sources_csv: str = "ci,total_flow,avg_speed,avg_occ"
    lag_cross_enabled: bool = True
    lag_cross_quantiles_csv: str = "0.25,0.5,0.75"
    drop_same_day_flow_speed_occ: bool = True
    drop_feature_list_csv: str = "total_flow,avg_speed,avg_occ"


@dataclass(frozen=True)
class FeatureBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    n_features_raw: int
    feature_names_raw: tuple[str, ...]
    lag_added_features: tuple[str, ...]
    lag_cross_added_features: tuple[str, ...]
    dropped_features: tuple[str, ...]


def build_feature_bundle(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    cfg: FeatureEngineeringConfig,
) -> FeatureBundle:
    x_train = np.asarray(X_train, dtype=float)
    x_test = np.asarray(X_test, dtype=float)
    y_tr = np.asarray(y_train, dtype=float).reshape(-1, 1)
    y_te = np.asarray(y_test, dtype=float).reshape(-1, 1)
    names = tuple(str(v) for v in feature_names)
    n_features_raw = int(x_train.shape[1])
    names_raw = tuple(str(v) for v in names)

    lag_added_features: list[str] = []
    lag_cross_added_features: list[str] = []
    dropped_features: list[str] = []

    lag_enabled = bool(cfg.lag_feature_enabled)
    lag_orders = parse_int_list_csv(str(cfg.lag_orders_csv), default=(1, 2, 3))
    lag_sources = [s.strip() for s in str(cfg.lag_sources_csv).split(",") if s.strip()]
    lag_source_set = {s for s in set(lag_sources) if s in {"ci", "total_flow", "avg_speed", "avg_occ"}}

    if lag_enabled and lag_orders and lag_source_set:
        name_to_idx_raw = {str(nm): int(i) for i, nm in enumerate(names)}
        tr_cols: list[np.ndarray] = [np.asarray(x_train, dtype=float)]
        te_cols: list[np.ndarray] = [np.asarray(x_test, dtype=float)]
        ext_names: list[str] = list(names)

        def _append_lags(src_name: str, tr_src: np.ndarray, te_src: np.ndarray) -> None:
            nonlocal tr_cols, te_cols, ext_names, lag_added_features
            for lag in lag_orders:
                tr_l, te_l = make_lag_from_history(tr_src, te_src, int(lag))
                name = f"{src_name}_lag{int(lag)}"
                tr_cols.append(tr_l.reshape(-1, 1))
                te_cols.append(te_l.reshape(-1, 1))
                ext_names.append(name)
                lag_added_features.append(name)

        if "ci" in lag_source_set:
            _append_lags("ci", y_tr.reshape(-1), y_te.reshape(-1))
        for src in ("total_flow", "avg_speed", "avg_occ"):
            if src not in lag_source_set:
                continue
            idx = name_to_idx_raw.get(src)
            if idx is None:
                continue
            _append_lags(src, x_train[:, int(idx)].reshape(-1), x_test[:, int(idx)].reshape(-1))

        x_train = np.concatenate(tr_cols, axis=1)
        x_test = np.concatenate(te_cols, axis=1)
        names = tuple(str(v) for v in ext_names)

    lag_cross_enabled = bool(cfg.lag_cross_enabled)
    lag_cross_q = [
        float(np.clip(v, 0.01, 0.99))
        for v in parse_float_list_csv(str(cfg.lag_cross_quantiles_csv), default=(0.25, 0.5, 0.75))
    ]
    if lag_cross_enabled:
        name_to_idx = {str(nm): int(i) for i, nm in enumerate(names)}
        i_ci = name_to_idx.get("ci_lag1")
        i_sp = name_to_idx.get("avg_speed_lag1")
        if i_ci is not None and i_sp is not None:
            ci_tr = np.asarray(x_train[:, int(i_ci)], dtype=float).reshape(-1)
            ci_te = np.asarray(x_test[:, int(i_ci)], dtype=float).reshape(-1)
            sp_tr = np.asarray(x_train[:, int(i_sp)], dtype=float).reshape(-1)
            sp_te = np.asarray(x_test[:, int(i_sp)], dtype=float).reshape(-1)
            tr_cols = [np.asarray(x_train, dtype=float)]
            te_cols = [np.asarray(x_test, dtype=float)]
            ext_names = list(names)
            for qv in lag_cross_q:
                c = float(np.quantile(ci_tr, float(qv)))
                hz_tr = np.maximum(0.0, ci_tr - c) * sp_tr
                hz_te = np.maximum(0.0, ci_te - c) * sp_te
                name = f"hinge_ci_lag1_q{int(round(qv * 100.0)):02d}_x_avg_speed_lag1"
                tr_cols.append(hz_tr.reshape(-1, 1))
                te_cols.append(hz_te.reshape(-1, 1))
                ext_names.append(name)
                lag_cross_added_features.append(name)
            x_train = np.concatenate(tr_cols, axis=1)
            x_test = np.concatenate(te_cols, axis=1)
            names = tuple(str(v) for v in ext_names)

    if bool(cfg.drop_same_day_flow_speed_occ):
        drop_set = {s.strip() for s in str(cfg.drop_feature_list_csv).split(",") if s.strip()}
        if drop_set:
            keep_idx = [i for i, nm in enumerate(names) if str(nm) not in drop_set]
            keep_set = set(keep_idx)
            dropped_features = [str(names[i]) for i in range(len(names)) if i not in keep_set]
            if not keep_idx:
                raise ValueError("all features were dropped; adjust drop_feature_list")
            x_train = np.asarray(x_train[:, keep_idx], dtype=float)
            x_test = np.asarray(x_test[:, keep_idx], dtype=float)
            names = tuple(str(names[i]) for i in keep_idx)

    return FeatureBundle(
        X_train=np.asarray(x_train, dtype=float),
        y_train=np.asarray(y_tr, dtype=float),
        X_test=np.asarray(x_test, dtype=float),
        y_test=np.asarray(y_te, dtype=float),
        feature_names=tuple(str(v) for v in names),
        n_features_raw=int(n_features_raw),
        feature_names_raw=tuple(str(v) for v in names_raw),
        lag_added_features=tuple(str(v) for v in lag_added_features),
        lag_cross_added_features=tuple(str(v) for v in lag_cross_added_features),
        dropped_features=tuple(str(v) for v in dropped_features),
    )


__all__ = [
    "FeatureEngineeringConfig",
    "FeatureBundle",
    "build_feature_bundle",
]
