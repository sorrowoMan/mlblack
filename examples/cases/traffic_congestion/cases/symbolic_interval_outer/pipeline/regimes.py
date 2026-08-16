from __future__ import annotations

from ..runtime.legacy_imports import *

STRICT4_REGIME_ORDER: tuple[tuple[int, int, int, int], ...] = (
    (1, 1, 0, 0),  # holiday_near
    (1, 0, 1, 0),  # holiday_mid
    (0, 0, 0, 1),  # weekend
    (0, 0, 0, 0),  # regular
)

STRICT4_HOLIDAY_KEYS: tuple[tuple[int, int, int, int], ...] = (
    (1, 1, 0, 0),
    (1, 0, 1, 0),
)

def _normalize_fixed4_key(raw_key: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(raw_key) >= 4:
        return tuple(int(v > 0) for v in raw_key[:4])  # type: ignore[return-value]
    padded = list(int(v > 0) for v in raw_key)
    while len(padded) < 4:
        padded.append(0)
    return tuple(padded[:4])  # type: ignore[return-value]

def _map_to_strict4_regime(raw_key: tuple[int, ...]) -> tuple[int, int, int, int]:
    k = _normalize_fixed4_key(raw_key)
    if k in STRICT4_REGIME_ORDER:
        return k
    if k[0] > 0 and k[1] > 0:
        return (1, 1, 0, 0)
    if k[0] > 0 and k[2] > 0:
        return (1, 0, 1, 0)
    if k[3] > 0:
        return (0, 0, 0, 1)
    if k[0] == 0 and k[1] == 0 and k[2] == 0:
        return (0, 0, 0, 0)
    return (0, 0, 0, 0)

def _strict4_keys_from_X(X: np.ndarray, gate_idx: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], ...]:
    x = np.asarray(X, dtype=float)
    i0, i1, i2, i3 = gate_idx
    out: list[tuple[int, int, int, int]] = []
    for r in range(int(x.shape[0])):
        raw = (
            int(x[r, i0] > 0.5),
            int(x[r, i1] > 0.5),
            int(x[r, i2] > 0.5),
            int(x[r, i3] > 0.5),
        )
        out.append(_map_to_strict4_regime(raw))
    return tuple(out)

__all__ = ['STRICT4_REGIME_ORDER', 'STRICT4_HOLIDAY_KEYS', '_normalize_fixed4_key', '_map_to_strict4_regime', '_strict4_keys_from_X']
