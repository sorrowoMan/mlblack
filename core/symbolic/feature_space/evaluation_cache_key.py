from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


DEFAULT_META_SIGNATURE_KEYS: tuple[str, ...] = (
    "tuned_l2",
    "complexity_scale",
    "family_penalty_scale",
    "feature_penalty_scale",
    "drift_weight",
    "strict4_min_train_ratio",
    "prior_corr_w",
    "family_bias_scale",
    "threshold_q",
    "interaction_cap",
    "k",
)


def normalize_subset_key(subset_idx: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(int(v) for v in subset_idx))


def build_meta_signature(
    meta: Mapping[str, Any],
    *,
    keys: Sequence[str] = DEFAULT_META_SIGNATURE_KEYS,
    float_precision: int = 7,
) -> str:
    out: dict[str, Any] = {}
    for key in keys:
        value = meta.get(key)
        if isinstance(value, float):
            out[str(key)] = round(float(value), int(float_precision))
        else:
            out[str(key)] = value
    return json.dumps(out, sort_keys=True, separators=(",", ":"))


def build_subset_meta_cache_key(
    subset_idx: Sequence[int],
    meta: Mapping[str, Any],
    *,
    keys: Sequence[str] = DEFAULT_META_SIGNATURE_KEYS,
    float_precision: int = 7,
) -> tuple[tuple[int, ...], str]:
    return (
        normalize_subset_key(subset_idx),
        build_meta_signature(meta, keys=keys, float_precision=float_precision),
    )


__all__ = [
    "DEFAULT_META_SIGNATURE_KEYS",
    "normalize_subset_key",
    "build_meta_signature",
    "build_subset_meta_cache_key",
]
