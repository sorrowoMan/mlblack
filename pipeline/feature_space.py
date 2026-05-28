from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from mlblack.pipeline.data_views import NumericDataView
from mlblack.pipeline.conditional import PrimitiveFeatureComposer
from mlblack.pipeline.base import DataPipelineComponent


@dataclass(frozen=True)
class FeatureSpace:
    names: Sequence[str]
    source_names: Sequence[str] = tuple()
    groups: Mapping[str, Sequence[str]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_data(cls, data: NumericDataView, *, group: str = "all") -> "FeatureSpace":
        names = data.effective_feature_names
        return cls(names=names, source_names=names, groups={group: names}, metadata=dict(data.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "source_names": list(self.source_names),
            "groups": {str(k): list(v) for k, v in self.groups.items()},
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FeatureSpaceComponent(DataPipelineComponent):
    name = "feature_space"
    context_requires = ("data.numeric_view",)
    context_optional = ("trainer.context",)
    context_provides = ("pipeline.feature_space", "data.feature_names")
    context_mutates = ("pipeline.component_state",)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Reads NumericDataView feature metadata and provides pipeline.feature_space."
    group: str = "all"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def fit(self, data: NumericDataView, context: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        _ = context
        return FeatureSpace.from_data(data, group=self.group).as_dict()

    def transform(
        self,
        data: NumericDataView,
        state: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> NumericDataView:
        _ = context
        return NumericDataView(
            X_train=data.X_train,
            y_train=data.y_train,
            X_valid=data.X_valid,
            y_valid=data.y_valid,
            feature_names=data.feature_names,
            target_name=data.target_name,
            metadata={**dict(data.metadata), "feature_space": dict(state or {})},
        )

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "group": self.group, "metadata": dict(self.metadata)}


@dataclass(frozen=True)
class ConditionalPrimitiveFeatureComponent(DataPipelineComponent):
    """Append deterministic conditional primitive features to NumericDataView."""

    name = "conditional_primitives"
    context_requires = ("data.numeric_view",)
    context_optional = ("trainer.context",)
    context_provides = ("pipeline.conditional_features", "data.numeric_view")
    context_mutates = ("pipeline.component_state",)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Adds deterministic conditional primitive columns to NumericDataView."
    primitives: Sequence[Any] = tuple()
    include_original: bool = True

    def transform(
        self,
        data: NumericDataView,
        state: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> NumericDataView:
        _ = state
        _ = context
        composer = PrimitiveFeatureComposer(primitives=self.primitives, include_original=self.include_original)
        X_train = composer.transform(data.X_train)
        X_valid = None if data.X_valid is None else composer.transform(data.X_valid)
        base_names = tuple(data.effective_feature_names)
        primitive_names: list[str] = []
        for item in composer.primitive_objects():
            part = item.transform(data.X_train)
            width = 1 if part.ndim == 1 else int(part.shape[1])
            primitive_names.extend([str(item.describe()["name"])] if width == 1 else [f"{item.describe()['name']}[{idx}]" for idx in range(width)])
        feature_names = base_names + tuple(primitive_names) if self.include_original else tuple(primitive_names)
        return NumericDataView(
            X_train=X_train,
            y_train=data.y_train,
            X_valid=X_valid,
            y_valid=data.y_valid,
            feature_names=feature_names,
            target_name=data.target_name,
            metadata={**dict(data.metadata), "conditional_primitives": composer.describe()},
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "include_original": bool(self.include_original),
            "primitives": [item.describe() for item in PrimitiveFeatureComposer(self.primitives).primitive_objects()],
        }

