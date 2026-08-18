"""mlblack semantic adapter for the shared pipeline slot kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from blackbase.kernel import (
    PipelineKernelBuild as SharedPipelineKernelBuild,
    PipelineSlotSpec as SharedPipelineSlotSpec,
    PipelineSpec as SharedPipelineSpec,
    build_pipeline_kernel as build_shared_pipeline_kernel,
)

from .base import DataPipeline, DataPipelineComponent
from .data_views import NumericDataView


PipelineSlotSpec = SharedPipelineSlotSpec
PipelineSpec = SharedPipelineSpec


class _SharedSlotComponent(DataPipelineComponent):
    name = "slot_kernel_transform"
    context_requires = ("data.numeric_view",)
    context_optional = ("pipeline.slot_context",)
    context_provides = ("data.numeric_view",)
    context_mutates = ()
    context_cache = ()
    context_notes = "Delegates transform slot execution to the shared blackbase pipeline kernel."

    def __init__(self, *, kernel: SharedPipelineKernelBuild, transform_slot: str = "transform") -> None:
        self._kernel = kernel
        self._slot = str(transform_slot or "transform")

    def transform(
        self,
        data: NumericDataView,
        state: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> NumericDataView:
        _ = state
        if context is None:
            slot_context: MutableMapping[str, Any] | None = None
        else:
            slot_context = dict(context)
        out = self._kernel.run_slot(self._slot, data, slot_context)
        return out if isinstance(out, NumericDataView) else data


@dataclass
class MLPipelineKernelBuild:
    shared: SharedPipelineKernelBuild
    data_pipeline: DataPipeline

    def run_slot(self, slot: str, value: Any, context: Optional[MutableMapping[str, Any]] = None) -> Any:
        return self.shared.run_slot(slot, value, context)

    @property
    def representation_pipeline(self):
        return self.shared.representation_pipeline


def build_pipeline_kernel(
    spec: PipelineSpec | Mapping[str, Any] | None,
    *,
    operator_registry: Mapping[str, Any],
    strict: bool = True,
    transform_slot: str = "transform",
) -> MLPipelineKernelBuild:
    shared_spec = PipelineSpec.from_value(spec)
    shared = build_shared_pipeline_kernel(
        shared_spec,
        operator_registry=operator_registry,
        strict=bool(strict),
    )
    component = _SharedSlotComponent(kernel=shared, transform_slot=transform_slot)
    data_pipeline = DataPipeline(components=(component,), name=f"{shared_spec.key}_slot_kernel")
    return MLPipelineKernelBuild(shared=shared, data_pipeline=data_pipeline)
