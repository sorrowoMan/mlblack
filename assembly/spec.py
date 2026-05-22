from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: str | Mapping[str, Any] | "ComponentSpec") -> "ComponentSpec":
        if isinstance(value, ComponentSpec):
            return value
        if isinstance(value, str):
            return cls(name=value)
        payload = dict(value)
        return cls(
            name=str(payload.get("name", payload.get("kind", ""))),
            params=dict(payload.get("params", payload.get("config", {})) or {}),
            enabled=bool(payload.get("enabled", True)),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": dict(self.params),
            "enabled": bool(self.enabled),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CapabilitySpec(ComponentSpec):
    pass


@dataclass(frozen=True)
class BiasSpec(ComponentSpec):
    pass


@dataclass(frozen=True)
class PipelineSpec:
    components: Sequence[ComponentSpec | Mapping[str, Any] | str] = tuple()
    name: str = "data_pipeline"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | "PipelineSpec" | None) -> "PipelineSpec":
        if isinstance(value, PipelineSpec):
            return value
        payload = dict(value or {})
        return cls(
            name=str(payload.get("name", "data_pipeline")),
            components=tuple(ComponentSpec.from_value(item) for item in payload.get("components", ())),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def component_specs(self) -> tuple[ComponentSpec, ...]:
        return tuple(ComponentSpec.from_value(item) for item in self.components)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "components": [item.as_dict() for item in self.component_specs()],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TrainerAssemblySpec:
    preset: str
    params: Mapping[str, Any] = field(default_factory=dict)
    run_name: str = ""
    compute_backend: str | Mapping[str, Any] | None = None
    resource_context: Mapping[str, Any] = field(default_factory=dict)
    capabilities: Sequence[CapabilitySpec | Mapping[str, Any] | str] = tuple()
    biases: Sequence[BiasSpec | Mapping[str, Any] | str] = tuple()
    component_overrides: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | "TrainerAssemblySpec") -> "TrainerAssemblySpec":
        if isinstance(value, TrainerAssemblySpec):
            return value
        payload = dict(value)
        unsupported = tuple(
            key
            for key in ("runtime", "orchestration", "resource_request", "workflow")
            if key in payload and payload.get(key) not in (None, {}, (), [], "")
        )
        if unsupported:
            raise ValueError(
                "TrainerAssemblySpec no longer accepts orchestration/runtime/resource fields; "
                f"move {unsupported} to nsgablack."
            )
        return cls(
            preset=str(payload.get("preset", payload.get("name", ""))),
            params=dict(payload.get("params", {}) or {}),
            run_name=str(payload.get("run_name", "")),
            compute_backend=payload.get("compute_backend", payload.get("backend")),
            resource_context=dict(payload.get("resource_context", {}) or {}),
            capabilities=tuple(CapabilitySpec.from_value(item) for item in payload.get("capabilities", ())),
            biases=tuple(BiasSpec.from_value(item) for item in payload.get("biases", ())),
            component_overrides=dict(payload.get("component_overrides", {}) or {}),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def capability_specs(self) -> tuple[CapabilitySpec, ...]:
        return tuple(CapabilitySpec.from_value(item) for item in self.capabilities)

    def bias_specs(self) -> tuple[BiasSpec, ...]:
        return tuple(BiasSpec.from_value(item) for item in self.biases)

    def effective_params(self) -> dict[str, Any]:
        params = dict(self.params)
        preset_params = dict(self.component_overrides.get("preset_params", {}) or {})
        trainer_params = dict(self.component_overrides.get("trainer", {}) or {})
        params.update(preset_params)
        params.update(trainer_params)
        if self.run_name:
            params.setdefault("run_name", self.run_name)
        if self.compute_backend not in (None, "", {}, (), []):
            params.setdefault("compute_backend", self.compute_backend)
        return params

    def as_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "params": dict(self.params),
            "run_name": self.run_name,
            "compute_backend": self.compute_backend,
            "resource_context": dict(self.resource_context),
            "capabilities": [item.as_dict() for item in self.capability_specs()],
            "biases": [item.as_dict() for item in self.bias_specs()],
            "component_overrides": dict(self.component_overrides),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class InnerTrainingAssemblySpec:
    trainer: TrainerAssemblySpec | Mapping[str, Any]
    pipeline: PipelineSpec | Mapping[str, Any] | None = None
    name: str = "inner_training"
    resource_context: Mapping[str, Any] = field(default_factory=dict)
    capabilities: Sequence[CapabilitySpec | Mapping[str, Any] | str] = tuple()
    report: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | "InnerTrainingAssemblySpec") -> "InnerTrainingAssemblySpec":
        if isinstance(value, InnerTrainingAssemblySpec):
            return value
        payload = dict(value)
        if payload.get("workflow") not in (None, {}, (), [], ""):
            raise ValueError("InnerTrainingAssemblySpec no longer accepts workflow; use nsgablack orchestration.")
        return cls(
            name=str(payload.get("name", "inner_training")),
            trainer=TrainerAssemblySpec.from_value(payload.get("trainer", {})),
            pipeline=PipelineSpec.from_value(payload.get("pipeline")),
            resource_context=dict(payload.get("resource_context", {}) or {}),
            capabilities=tuple(CapabilitySpec.from_value(item) for item in payload.get("capabilities", ())),
            report=dict(payload.get("report", {}) or {}),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def trainer_spec(self) -> TrainerAssemblySpec:
        return TrainerAssemblySpec.from_value(self.trainer)

    def pipeline_spec(self) -> PipelineSpec:
        return PipelineSpec.from_value(self.pipeline)

    def capability_specs(self) -> tuple[CapabilitySpec, ...]:
        return tuple(CapabilitySpec.from_value(item) for item in self.capabilities)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trainer": self.trainer_spec().as_dict(),
            "pipeline": self.pipeline_spec().as_dict(),
            "resource_context": dict(self.resource_context),
            "capabilities": [item.as_dict() for item in self.capability_specs()],
            "report": dict(self.report),
            "metadata": dict(self.metadata),
        }


