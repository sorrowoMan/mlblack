from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from core.common.contracts import Sample, SampleDataset


class ViewBuildError(ValueError):
    pass


def _sample_has_target(sample: Sample, target_key: str) -> bool:
    if target_key in sample.labels:
        return True
    if target_key in sample.cells:
        return True
    for cell in sample.cells.values():
        if target_key in cell.labels:
            return True
    return False


def _infer_target_names(data: SampleDataset, target_key: str) -> tuple[str, ...] | None:
    if not data.samples:
        return (target_key,)

    first = data.samples[0]
    if target_key in first.labels:
        value = first.labels[target_key]
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            return None
        if arr.size <= 1:
            return (target_key,)
        return tuple(f"{target_key}[{i}]" for i in range(int(arr.size)))

    if target_key in first.cells:
        return None

    for cell in first.cells.values():
        if target_key in cell.labels:
            return None

    return (target_key,)


def _default_feature_keys(
    data: SampleDataset,
    *,
    target_key: str,
    exclude_target_from_features: bool,
) -> tuple[str, ...]:
    if data.feature_cell_keys is not None:
        keys = [str(k) for k in data.feature_cell_keys]
    elif data.samples:
        keys = sorted(str(k) for k in data.samples[0].cells.keys())
    else:
        keys = []

    if exclude_target_from_features:
        keys = [k for k in keys if k != str(target_key)]

    if not keys:
        raise ViewBuildError("No feature_cell_keys available for target view")
    return tuple(keys)


def build_target_view(
    data: SampleDataset,
    target_key: str,
    *,
    feature_cell_keys: Sequence[str] | None = None,
    target_names: Sequence[str] | None = None,
    description: str | None = None,
    strict: bool = True,
    exclude_target_from_features: bool = True,
) -> SampleDataset:
    """Create one target-specific view_j dataset from a multi-target SampleDataset."""

    key = str(target_key)
    if not key:
        raise ViewBuildError("target_key must not be empty")

    if strict:
        missing = [s.sample_id for s in data.samples if not _sample_has_target(s, key)]
        if missing:
            head = ", ".join(missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            raise ViewBuildError(f"Target '{key}' missing in {len(missing)} samples: {head}{suffix}")

    if feature_cell_keys is None:
        fkeys = _default_feature_keys(data, target_key=key, exclude_target_from_features=exclude_target_from_features)
    else:
        fkeys = tuple(str(k) for k in feature_cell_keys)
        if not fkeys:
            raise ViewBuildError("feature_cell_keys must not be empty")
        if exclude_target_from_features:
            fkeys = tuple(k for k in fkeys if k != key)
            if not fkeys:
                raise ViewBuildError("feature_cell_keys became empty after excluding target key")

    if target_names is None:
        tnames = _infer_target_names(data, key)
    else:
        tnames = tuple(str(n) for n in target_names)

    desc = description if description is not None else data.description
    return SampleDataset(
        samples=data.samples,
        target_key=key,
        feature_cell_keys=fkeys,
        target_names=tnames,
        description=desc,
    )


def _default_target_keys(data: SampleDataset) -> tuple[str, ...]:
    if data.samples:
        label_keys = sorted(str(k) for k in data.samples[0].labels.keys())
        if label_keys:
            return tuple(label_keys)
    if data.target_names:
        return tuple(str(k) for k in data.target_names)
    return (str(data.target_key),)


def build_target_views(
    data: SampleDataset,
    *,
    target_keys: Sequence[str] | None = None,
    feature_key_map: Mapping[str, Sequence[str]] | None = None,
    strict: bool = True,
    exclude_target_from_features: bool = True,
) -> Dict[str, SampleDataset]:
    """Create {target_key: view_j} for multi-target training pipelines."""

    keys = tuple(str(k) for k in (target_keys if target_keys is not None else _default_target_keys(data)))
    if not keys:
        raise ViewBuildError("No target_keys specified or inferable")

    views: Dict[str, SampleDataset] = {}
    fmap = dict(feature_key_map or {})

    for key in keys:
        views[key] = build_target_view(
            data,
            key,
            feature_cell_keys=fmap.get(key),
            strict=strict,
            exclude_target_from_features=exclude_target_from_features,
        )

    return views
