from __future__ import annotations

from typing import Sequence

import numpy as np


def parse_int_list_csv(text: str, *, default: Sequence[int]) -> list[int]:
    vals: list[int] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            vals.append(int(item))
        except Exception:
            continue
    out = [int(v) for v in vals if int(v) > 0]
    if not out:
        out = [int(v) for v in default if int(v) > 0]
    return sorted(set(out))


def parse_float_list_csv(text: str, *, default: Sequence[float]) -> list[float]:
    vals: list[float] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            vals.append(float(item))
        except Exception:
            continue
    out = [float(v) for v in vals if np.isfinite(v)]
    if not out:
        out = [float(v) for v in default if np.isfinite(v)]
    return sorted(set(out))


def make_lag_from_history(train_series: np.ndarray, test_series: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    tr = np.asarray(train_series, dtype=float).reshape(-1)
    te = np.asarray(test_series, dtype=float).reshape(-1)
    ntr = int(tr.size)
    nte = int(te.size)
    l = int(max(1, lag))
    full = np.concatenate([tr, te], axis=0)
    out = np.empty_like(full, dtype=float)
    out[:l] = np.nan
    out[l:] = full[:-l]
    tr_lag = np.asarray(out[:ntr], dtype=float)
    te_lag = np.asarray(out[ntr : ntr + nte], dtype=float)
    fill = float(tr[0]) if ntr > 0 else 0.0
    tr_lag = np.where(np.isfinite(tr_lag), tr_lag, fill)
    te_lag = np.where(np.isfinite(te_lag), te_lag, fill)
    return tr_lag, te_lag


__all__ = [
    "parse_int_list_csv",
    "parse_float_list_csv",
    "make_lag_from_history",
]
