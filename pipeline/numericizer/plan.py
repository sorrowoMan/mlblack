from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from mlblack.assembly.schema import DatasetSchema, FeatureSpec, TargetSpec


@dataclass(frozen=True)
class NumericFeatureColumn:
    name: str
    source_key: str
    kind: str = "numeric"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "source_key": self.source_key, "kind": self.kind, "metadata": dict(self.metadata)}


@dataclass(frozen=True)
class NumericizationPlan:
    schema: DatasetSchema
    columns: Sequence[NumericFeatureColumn]
    target: TargetSpec
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_schema(cls, schema: DatasetSchema | Mapping[str, Any]) -> "NumericizationPlan":
        spec = DatasetSchema.from_value(schema)
        columns: list[NumericFeatureColumn] = []
        for feature in spec.features:
            f = FeatureSpec.from_value(feature)
            encoder = f.encoder if f.encoder != "auto" else _default_encoder(f.dtype)
            if encoder in {"numeric", "identity", "float"}:
                columns.append(NumericFeatureColumn(name=f.key, source_key=f.key, kind="numeric"))
            elif encoder in {"boolean", "bool"}:
                columns.append(NumericFeatureColumn(name=f.key, source_key=f.key, kind="boolean"))
            elif encoder in {"onehot", "one_hot", "categorical"}:
                vocab = tuple(f.vocab or ())
                if not vocab:
                    columns.append(NumericFeatureColumn(name=f.key, source_key=f.key, kind="categorical_pending"))
                else:
                    for value in vocab:
                        columns.append(NumericFeatureColumn(name=f"{f.key}={value}", source_key=f.key, kind="onehot", metadata={"value": value}))
            else:
                raise ValueError(f"unsupported encoder for feature {f.key}: {encoder}")
        return cls(schema=spec, columns=tuple(columns), target=spec.target)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.columns)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema.as_dict(),
            "columns": [item.as_dict() for item in self.columns],
            "target": self.target.as_dict(),
            "metadata": dict(self.metadata),
        }


def _default_encoder(dtype: str) -> str:
    normalized = str(dtype).lower()
    if normalized in {"category", "categorical", "str", "string"}:
        return "onehot"
    if normalized in {"bool", "boolean"}:
        return "boolean"
    return "numeric"

