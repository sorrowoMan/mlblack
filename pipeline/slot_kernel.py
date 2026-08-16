"""mlblack semantic adapter for the shared pipeline slot kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from blackbase.kernel import (
    PipelineKernelBuild as SharedPipelineKernelBuild,
    PipelineSlotSpec as SharedPipelineSlotSpec,
    PipelineSpec as SharedPipelineSpec,
    build_pipeline_kernel as build_shared_pipeline_kernel,
)

from .base import DataPipeline, DataPipelineComponent
from .data_views import NumericDataView


@dataclass(frozen=True)
class PipelineSlotSpec:
    slot: str
    operators: Sequence[str] = ()
    mode: str = "serial"
    method: str | None = None
    routes: Mapping[str, str] = field(default_factory=dict)
    selector_key: str = "route"
    default_operator: str | None = None
    strict: bool | None = None
    merge: str | None = None

    def as_shared(self) -> SharedPipelineSlotSpec:
        return SharedPipelineSlotSpec(
            slot=self.slot,
            operators=tuple(self.operators),
            mode=self.mode,
            method=self.method,
            routes=dict(self.routes),
            selector_key=self.selector_key,
            default_operator=self.default_operator,
            strict=self.strict,
            merge=self.merge,
        )


@dataclass(frozen=True)
class PipelineSpec:
    key: str = "ml_pipeline"
    slots: Sequence[PipelineSlotSpec] = field(default_factory=tuple)
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | "PipelineSpec" | None) -> "PipelineSpec":
        if isinstance(value, PipelineSpec):
            return value
        payload = dict(value or {})
        raw_slots = payload.get("slots", ())
        slots = []
        for item in raw_slots:
            if isinstance(item, PipelineSlotSpec):
                slots.append(item)
                continue
            spec = dict(item or {})
            slots.append(
                PipelineSlotSpec(
                    slot=str(spec.get("slot", spec.get("name", ""))),
                    operators=tuple(str(name) for name in spec.get("operators", ()) if str(name).strip()),
                    mode=str(spec.get("mode", "serial") or "serial"),
                    method=str(spec.get("method")) if spec.get("method") not in (None, "") else None,
                    routes={str(k): str(v) for k, v in dict(spec.get("routes", {}) or {}).items()},
                    selector_key=str(spec.get("selector_key", "route") or "route"),
                    default_operator=(
                        str(spec.get("default_operator")).strip()
                        if spec.get("default_operator") not in (None, "")
                        else None
                    ),
                    strict=spec.get("strict"),
                    merge=str(spec.get("merge")) if spec.get("merge") not in (None, "") else None,
                )
            )
        return cls(
            key=str(payload.get("key", "ml_pipeline") or "ml_pipeline"),
            slots=tuple(slots),
            params=dict(payload.get("params", {}) or {}),
        )

    def as_shared(self) -> SharedPipelineSpec:
        return SharedPipelineSpec(
            key=self.key,
            slots=tuple(slot.as_shared() for slot in self.slots),
            params=dict(self.params),
        )


class _SharedSlotComponent(DataPipelineComponent):
    name = "slot_kernel_transform"
    context_requires = ("data.numeric_view",)
    context_optional = ("pipeline.slot_context",)
    context_provides = ("data.numeric_view",)
    context_mutates = ()
    context_cache = ()
    context_notes = "Delegates transform slot execution to shared nsgablack pipeline slot kernel."

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
    ml_spec = PipelineSpec.from_value(spec)
    shared = build_shared_pipeline_kernel(
        ml_spec.as_shared(),
        operator_registry=operator_registry,
        strict=bool(strict),
    )
    component = _SharedSlotComponent(kernel=shared, transform_slot=transform_slot)
    data_pipeline = DataPipeline(components=(component,), name=f"{ml_spec.key}_slot_kernel")
    return MLPipelineKernelBuild(shared=shared, data_pipeline=data_pipeline)
