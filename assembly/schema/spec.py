from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

DType = str


@dataclass(frozen=True)
class FeatureSpec:
    """Schema for one raw feature before numericization."""

    key: str
    dtype: DType = "numeric"
    encoder: str = "auto"
    required: bool = True
    modality: str | None = None
    vocab: Sequence[Any] | None = None
    ordered: bool = False
    unknown: str = "error"
    constraints: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: str | Mapping[str, Any] | "FeatureSpec") -> "FeatureSpec":
        if isinstance(value, FeatureSpec):
            return value
        if isinstance(value, str):
            return cls(key=value)
        payload = dict(value)
        return cls(
            key=str(payload.get("key", payload.get("name", ""))),
            dtype=str(payload.get("dtype", "numeric")),
            encoder=str(payload.get("encoder", "auto")),
            required=bool(payload.get("required", True)),
            modality=None if payload.get("modality") is None else str(payload.get("modality")),
            vocab=None if payload.get("vocab") is None else tuple(payload.get("vocab", ())),
            ordered=bool(payload.get("ordered", False)),
            unknown=str(payload.get("unknown", "error")),
            constraints=dict(payload.get("constraints", {}) or {}),
            metadata=dict(payload.get("metadata", payload.get("meta", {})) or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "dtype": self.dtype,
            "encoder": self.encoder,
            "required": bool(self.required),
            "modality": self.modality,
            "vocab": None if self.vocab is None else list(self.vocab),
            "ordered": bool(self.ordered),
            "unknown": self.unknown,
            "constraints": dict(self.constraints),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TargetSpec:
    """Schema for one target column."""

    key: str = "target"
    dtype: DType = "numeric"
    required: bool = True
    vocab: Sequence[Any] | None = None
    ordered: bool = False
    constraints: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: str | Mapping[str, Any] | "TargetSpec") -> "TargetSpec":
        if isinstance(value, TargetSpec):
            return value
        if isinstance(value, str):
            return cls(key=value)
        payload = dict(value)
        return cls(
            key=str(payload.get("key", payload.get("name", "target"))),
            dtype=str(payload.get("dtype", "numeric")),
            required=bool(payload.get("required", True)),
            vocab=None if payload.get("vocab") is None else tuple(payload.get("vocab", ())),
            ordered=bool(payload.get("ordered", False)),
            constraints=dict(payload.get("constraints", {}) or {}),
            metadata=dict(payload.get("metadata", payload.get("meta", {})) or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "dtype": self.dtype,
            "required": bool(self.required),
            "vocab": None if self.vocab is None else list(self.vocab),
            "ordered": bool(self.ordered),
            "constraints": dict(self.constraints),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DatasetSchema:
    """Stable raw-data schema used by numericizer and scaffold config."""

    features: Sequence[FeatureSpec | Mapping[str, Any] | str]
    targets: Sequence[TargetSpec | Mapping[str, Any] | str] = field(default_factory=lambda: (TargetSpec(),))
    id_key: str | None = None
    strict: bool = True
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        features = tuple(FeatureSpec.from_value(item) for item in self.features)
        targets = tuple(TargetSpec.from_value(item) for item in self.targets)
        if not features:
            raise ValueError("DatasetSchema.features must not be empty")
        if not targets:
            raise ValueError("DatasetSchema.targets must not be empty")
        feature_keys = [item.key for item in features]
        target_keys = [item.key for item in targets]
        if len(set(feature_keys)) != len(feature_keys):
            raise ValueError("DatasetSchema.features contains duplicated keys")
        if len(set(target_keys)) != len(target_keys):
            raise ValueError("DatasetSchema.targets contains duplicated keys")
        overlap = sorted(set(feature_keys).intersection(target_keys))
        if overlap:
            raise ValueError(f"Feature/target key collision is not allowed: {overlap}")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "targets", targets)

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | "DatasetSchema") -> "DatasetSchema":
        if isinstance(value, DatasetSchema):
            return value
        payload = dict(value)
        target_payload = payload.get("targets")
        if target_payload is None and payload.get("target") is not None:
            target_payload = (payload.get("target"),)
        return cls(
            features=tuple(FeatureSpec.from_value(item) for item in payload.get("features", ())),
            targets=tuple(TargetSpec.from_value(item) for item in (target_payload or (TargetSpec(),))),
            id_key=None if payload.get("id_key") is None else str(payload.get("id_key")),
            strict=bool(payload.get("strict", True)),
            description=str(payload.get("description", "")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    @property
    def target(self) -> TargetSpec:
        return tuple(self.targets)[0]

    @property
    def feature_keys(self) -> tuple[str, ...]:
        return tuple(item.key for item in self.features)

    @property
    def target_keys(self) -> tuple[str, ...]:
        return tuple(item.key for item in self.targets)

    def as_dict(self) -> dict[str, Any]:
        return {
            "features": [FeatureSpec.from_value(item).as_dict() for item in self.features],
            "targets": [TargetSpec.from_value(item).as_dict() for item in self.targets],
            "id_key": self.id_key,
            "strict": bool(self.strict),
            "description": self.description,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ScaffoldConfig:
    """Top-level JSON-compatible scaffold contract."""

    name: str
    schema: DatasetSchema | Mapping[str, Any]
    inner_training: Mapping[str, Any]
    version: str = "0.1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | "ScaffoldConfig") -> "ScaffoldConfig":
        if isinstance(value, ScaffoldConfig):
            return value
        payload = dict(value)
        if payload.get("flow") not in (None, {}, (), [], ""):
            raise ValueError("ScaffoldConfig no longer accepts flow; use inner_training for the mlblack inner trainer surface.")
        return cls(
            name=str(payload.get("name", "mlblack_project")),
            schema=DatasetSchema.from_value(payload.get("schema", {})),
            inner_training=dict(payload.get("inner_training", {}) or {}),
            version=str(payload.get("version", "0.1")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def schema_spec(self) -> DatasetSchema:
        return DatasetSchema.from_value(self.schema)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "schema": self.schema_spec().as_dict(),
            "inner_training": dict(self.inner_training),
            "metadata": dict(self.metadata),
        }

