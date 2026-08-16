from __future__ import annotations

from ..runtime.legacy_imports import *

def _parse_int_list_csv(text: str, *, default: Sequence[int]) -> list[int]:
    vals: list[int] = []
    for s in str(text).split(","):
        ss = s.strip()
        if not ss:
            continue
        try:
            vals.append(int(ss))
        except Exception:
            continue
    out = [int(v) for v in vals if int(v) > 0]
    if not out:
        out = [int(v) for v in default if int(v) > 0]
    out = sorted(set(out))
    return out

def _parse_float_list_csv(text: str, *, default: Sequence[float]) -> list[float]:
    vals: list[float] = []
    for s in str(text).split(","):
        ss = s.strip()
        if not ss:
            continue
        try:
            vals.append(float(ss))
        except Exception:
            continue
    out = [float(v) for v in vals if np.isfinite(v)]
    if not out:
        out = [float(v) for v in default if np.isfinite(v)]
    out = sorted(set(out))
    return out

def _make_lag_from_history(train_series: np.ndarray, test_series: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
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
    # deterministic fill for earliest rows only
    fill = float(tr[0]) if ntr > 0 else 0.0
    tr_lag = np.where(np.isfinite(tr_lag), tr_lag, fill)
    te_lag = np.where(np.isfinite(te_lag), te_lag, fill)
    return tr_lag, te_lag

def _rolling_splits(n: int, *, folds: int, val_ratio: float, min_train: int) -> list[tuple[np.ndarray, np.ndarray]]:
    nn = int(n)
    ff = int(max(1, folds))
    val_size = max(64, int(round(float(val_ratio) * nn)))
    val_size = min(val_size, max(64, nn // 3))

    start_min = max(int(min_train), val_size + 64)
    start_max = nn - val_size
    if start_max <= start_min:
        split = int(round(nn * 0.75))
        split = max(64, min(split, nn - 64))
        return [(np.arange(0, split, dtype=int), np.arange(split, nn, dtype=int))]

    anchors = np.linspace(start_min, start_max, num=ff, dtype=int)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for s in anchors:
        start = int(s)
        end = min(nn, start + val_size)
        if end - start < 32:
            continue
        tr = np.arange(0, start, dtype=int)
        va = np.arange(start, end, dtype=int)
        if tr.size >= 64 and va.size >= 32:
            out.append((tr, va))
    if not out:
        split = int(round(nn * 0.75))
        split = max(64, min(split, nn - 64))
        out.append((np.arange(0, split, dtype=int), np.arange(split, nn, dtype=int)))
    return out

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    denom = float(np.sqrt(np.dot(xc, xc) * np.dot(yc, yc))) + 1e-12
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(xc, yc) / denom)

__all__ = ['_parse_int_list_csv', '_parse_float_list_csv', '_make_lag_from_history', '_rolling_splits', '_safe_corr']
