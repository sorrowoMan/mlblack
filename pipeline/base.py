from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from mlblack.core.contracts import ComponentContract, ContractMixin
from mlblack.pipeline.data import NumericDataView


@dataclass(frozen=True)
class PipelineFitState:
    component_states: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "component_states": {str(k): dict(v) for k, v in self.component_states.items()},
            "metadata": dict(self.metadata),
        }


class DataPipelineComponent(ContractMixin):
    """Data pipeline component.

    Pipeline components prepare data. They do not optimize parameters and do
    not inspect trainer state.
    """

    name = "pipeline_component"
    context_requires = ("data.numeric_view",)
    context_optional = ("trainer.context",)
    context_provides = ("data.numeric_view",)
    context_mutates = ("pipeline.component_state",)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Reads a NumericDataView, returns a transformed NumericDataView, and may store lightweight component fit state."
    contract = ComponentContract(
        name=name,
        requires=("data.numeric_view",),
        optional=("trainer.context",),
        provides=("data.numeric_view",),
        mutates=("pipeline.component_state",),
        supports_batch=True,
        metadata={"layer": "pipeline"},
    )

    def fit(self, data: NumericDataView, context: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        _ = data
        _ = context
        return {}

    def transform(
        self,
        data: NumericDataView,
        state: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> NumericDataView:
        _ = state
        _ = context
        return data

    def fit_transform(
        self,
        data: NumericDataView,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[NumericDataView, Mapping[str, Any]]:
        state = self.fit(data, context)
        return self.transform(data, state, context), state

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


class DataPipeline:
    """Ordered data preparation chain for a trainer assembly."""

    def __init__(self, components: Sequence[DataPipelineComponent] | None = None, *, name: str = "data_pipeline") -> None:
        self.name = str(name)
        self.components = tuple(components or ())
        self.fit_state = PipelineFitState()

    def fit_transform(
        self,
        data: NumericDataView,
        context: Mapping[str, Any] | None = None,
    ) -> NumericDataView:
        current = data
        states: dict[str, Mapping[str, Any]] = {}
        for component in self.components:
            current, state = component.fit_transform(current, context)
            states[str(component.name)] = dict(state)
        self.fit_state = PipelineFitState(component_states=states)
        return current

    def transform(
        self,
        data: NumericDataView,
        context: Mapping[str, Any] | None = None,
    ) -> NumericDataView:
        current = data
        for component in self.components:
            state = self.fit_state.component_states.get(str(component.name), {})
            current = component.transform(current, state, context)
        return current

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "components": [component.describe() for component in self.components],
            "fit_state": self.fit_state.as_dict(),
        }


