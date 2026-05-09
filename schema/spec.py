from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


DType = str


@dataclass(frozen=True)
class FeatureSpec:
    """Schema for one feature column/cell."""

    key: str
    dtype: DType
    encoder: str
    required: bool = True
    modality: str | None = None
    vocab: Sequence[Any] | None = None
    ordered: bool = False
    item_dtype: DType | None = None
    unknown: str = "error"  # error | allow
    constraints: Mapping[str, Any] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetSpec:
    """Schema for one training target."""

    key: str = "target"
    dtype: DType = "numeric"
    required: bool = True
    vocab: Sequence[Any] | None = None
    ordered: bool = False
    constraints: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSchema:
    """Dataset-level schema for parser/validator.

    Notes:
    - `targets` is the canonical multi-target definition.
    - `target` property keeps single-target compatibility by returning the first target.
    """

    features: Sequence[FeatureSpec]
    targets: Sequence[TargetSpec] = field(default_factory=lambda: (TargetSpec(),))
    id_key: str | None = None
    strict: bool = True
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("DatasetSchema.features must not be empty")
        if not self.targets:
            raise ValueError("DatasetSchema.targets must not be empty")

        feature_keys = [str(f.key) for f in self.features]
        if len(set(feature_keys)) != len(feature_keys):
            raise ValueError("DatasetSchema.features contains duplicated keys")

        target_keys = [str(t.key) for t in self.targets]
        if len(set(target_keys)) != len(target_keys):
            raise ValueError("DatasetSchema.targets contains duplicated keys")

        overlap = sorted(set(feature_keys).intersection(target_keys))
        if overlap:
            raise ValueError(f"Feature/target key collision is not allowed: {overlap}")

    @property
    def target(self) -> TargetSpec:
        """Single-target compatibility accessor (first target)."""
        return self.targets[0]

    @property
    def target_keys(self) -> tuple[str, ...]:
        return tuple(str(t.key) for t in self.targets)
