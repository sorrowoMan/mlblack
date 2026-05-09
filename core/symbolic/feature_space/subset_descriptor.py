from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_subset_genome(
    *,
    candidates: Sequence[Any],
    subset_idx: Sequence[int],
) -> list[dict[str, Any]]:
    subset_ids = [int(v) for v in subset_idx]
    return [
        {
            "name": str(candidates[idx].name),
            "expr": dict(candidates[idx].expr),
        }
        for idx in subset_ids
    ]


def build_subset_candidate_metadata(
    *,
    candidates: Sequence[Any],
    subset_idx: Sequence[int],
) -> list[dict[str, Any]]:
    subset_ids = [int(v) for v in subset_idx]
    return [
        {
            "name": str(candidates[idx].name),
            "family": str(candidates[idx].family),
            "complexity": float(candidates[idx].complexity),
            "features": [int(v) for v in candidates[idx].features],
        }
        for idx in subset_ids
    ]


def build_subset_descriptor(
    *,
    candidates: Sequence[Any],
    subset_idx: Sequence[int],
) -> dict[str, Any]:
    subset_ids = [int(v) for v in subset_idx]
    return {
        "subset_idx": subset_ids,
        "genome": build_subset_genome(candidates=candidates, subset_idx=subset_ids),
        "subset_candidates": build_subset_candidate_metadata(candidates=candidates, subset_idx=subset_ids),
    }


__all__ = [
    "build_subset_genome",
    "build_subset_candidate_metadata",
    "build_subset_descriptor",
]
