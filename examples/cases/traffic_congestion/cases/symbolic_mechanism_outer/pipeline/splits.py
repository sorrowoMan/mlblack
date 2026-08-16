from __future__ import annotations

from ..runtime.legacy_imports import *

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

__all__ = ['_rolling_splits', '_safe_corr']
