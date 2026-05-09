from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


SYMBOLIC_STRUCTURE_HEADS: tuple[str, ...] = ("basis_set", "expression")
SYMBOLIC_PREDICTION_HEADS: tuple[str, ...] = ("none", "point", "interval")
SYMBOLIC_SEARCH_INPUT_SPACES: tuple[str, ...] = ("raw_feature_space", "basis_object_space")
SYMBOLIC_POOL_EXPANSION_UNITS: tuple[str, ...] = ("raw_feature", "basis_object")
SYMBOLIC_GRADIENT_GUIDANCE_MODES: tuple[str, ...] = (
    "off",
    "raw_feature_gradient",
    "basis_object_gradient",
    "hybrid",
)
SYMBOLIC_BASIS_BINDING_MODES: tuple[str, ...] = ("off", "bound", "defining")
SYMBOLIC_ESCAPE_POLICIES: tuple[str, ...] = ("forbid", "budgeted_escape", "fallback_to_generic")


def _normalize_name(value: str | None, default: str) -> str:
    text = str(value or "").strip().lower()
    return text or str(default)


def _normalize_mode(value: str | None, *, allowed: Sequence[str], default: str, label: str) -> str:
    mode = _normalize_name(value, default)
    if mode not in tuple(allowed):
        raise ValueError(f"{label} must be one of {tuple(allowed)}, got '{value}'")
    return mode


def _normalize_fields(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return tuple()
    return tuple(str(v).strip() for v in tuple(values) if str(v).strip())


@dataclass(frozen=True)
class BasisObjectRef:
    object_key: str
    expression: str | None = None
    family_ref: str | None = None
    source_features: tuple[str, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_key", str(self.object_key).strip())
        object.__setattr__(self, "expression", None if self.expression is None else str(self.expression))
        object.__setattr__(self, "family_ref", None if self.family_ref is None else str(self.family_ref))
        object.__setattr__(self, "source_features", _normalize_fields(self.source_features))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not str(self.object_key):
            raise ValueError("basis object_key must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "object_key": str(self.object_key),
            "expression": None if self.expression is None else str(self.expression),
            "family_ref": None if self.family_ref is None else str(self.family_ref),
            "source_features": [str(v) for v in self.source_features],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SymbolicBasisContext:
    basis_source: str = "basis_stage"
    binding_mode: str = "bound"
    equivalence_mode: str = "family-level"
    selected_basis: tuple[BasisObjectRef, ...] = tuple()
    locked_basis_keys: tuple[str, ...] = tuple()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "basis_source", str(self.basis_source or "basis_stage").strip() or "basis_stage")
        object.__setattr__(
            self,
            "binding_mode",
            _normalize_mode(
                self.binding_mode,
                allowed=SYMBOLIC_BASIS_BINDING_MODES,
                default="bound",
                label="symbolic basis binding mode",
            ),
        )
        object.__setattr__(
            self,
            "equivalence_mode",
            str(self.equivalence_mode or "family-level").strip() or "family-level",
        )
        object.__setattr__(
            self,
            "selected_basis",
            tuple(
                value if isinstance(value, BasisObjectRef) else BasisObjectRef(**dict(value))
                for value in tuple(self.selected_basis)
            ),
        )
        object.__setattr__(self, "locked_basis_keys", _normalize_fields(self.locked_basis_keys))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "basis_source": str(self.basis_source),
            "binding_mode": str(self.binding_mode),
            "equivalence_mode": str(self.equivalence_mode),
            "selected_basis": [value.as_dict() for value in self.selected_basis],
            "locked_basis_keys": [str(v) for v in self.locked_basis_keys],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SymbolicStageHeadSpec:
    structure_head: str = "expression"
    prediction_head: str = "none"
    search_input_space: str = "raw_feature_space"
    pool_expansion_unit: str = "raw_feature"
    gradient_guidance_mode: str = "off"
    basis_binding_mode: str = "off"
    escape_policy: str = "fallback_to_generic"
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        structure_head = _normalize_mode(
            self.structure_head,
            allowed=SYMBOLIC_STRUCTURE_HEADS,
            default="expression",
            label="symbolic structure_head",
        )
        prediction_head = _normalize_mode(
            self.prediction_head,
            allowed=SYMBOLIC_PREDICTION_HEADS,
            default="none",
            label="symbolic prediction_head",
        )
        search_input_space = _normalize_mode(
            self.search_input_space,
            allowed=SYMBOLIC_SEARCH_INPUT_SPACES,
            default="raw_feature_space",
            label="symbolic search_input_space",
        )
        pool_expansion_unit = _normalize_mode(
            self.pool_expansion_unit,
            allowed=SYMBOLIC_POOL_EXPANSION_UNITS,
            default="raw_feature",
            label="symbolic pool_expansion_unit",
        )
        gradient_guidance_mode = _normalize_mode(
            self.gradient_guidance_mode,
            allowed=SYMBOLIC_GRADIENT_GUIDANCE_MODES,
            default="off",
            label="symbolic gradient_guidance_mode",
        )
        basis_binding_mode = _normalize_mode(
            self.basis_binding_mode,
            allowed=SYMBOLIC_BASIS_BINDING_MODES,
            default="off",
            label="symbolic basis_binding_mode",
        )
        escape_policy = _normalize_mode(
            self.escape_policy,
            allowed=SYMBOLIC_ESCAPE_POLICIES,
            default="fallback_to_generic",
            label="symbolic escape_policy",
        )

        if structure_head == "basis_set":
            if prediction_head != "none":
                raise ValueError("basis_set structure_head must use prediction_head='none'")
            if search_input_space != "raw_feature_space":
                raise ValueError("basis_set structure_head must search in raw_feature_space")
            if pool_expansion_unit != "raw_feature":
                raise ValueError("basis_set structure_head must expand raw_feature units")
            if basis_binding_mode != "off":
                raise ValueError("basis_set structure_head must use basis_binding_mode='off'")

        if search_input_space == "basis_object_space":
            if structure_head != "expression":
                raise ValueError("basis_object_space is currently only valid for expression stages")
            if pool_expansion_unit != "basis_object":
                raise ValueError("basis_object_space must use pool_expansion_unit='basis_object'")
            if basis_binding_mode == "off":
                raise ValueError("basis_object_space requires basis_binding_mode='bound' or 'defining'")
            if gradient_guidance_mode == "raw_feature_gradient":
                raise ValueError("basis_object_space cannot use raw_feature_gradient without hybrid mode")

        if search_input_space == "raw_feature_space":
            if pool_expansion_unit != "raw_feature":
                raise ValueError("raw_feature_space must use pool_expansion_unit='raw_feature'")
            if basis_binding_mode != "off":
                raise ValueError("raw_feature_space stages must use basis_binding_mode='off'")
            if gradient_guidance_mode == "basis_object_gradient":
                raise ValueError("raw_feature_space cannot use basis_object_gradient")

        object.__setattr__(self, "structure_head", structure_head)
        object.__setattr__(self, "prediction_head", prediction_head)
        object.__setattr__(self, "search_input_space", search_input_space)
        object.__setattr__(self, "pool_expansion_unit", pool_expansion_unit)
        object.__setattr__(self, "gradient_guidance_mode", gradient_guidance_mode)
        object.__setattr__(self, "basis_binding_mode", basis_binding_mode)
        object.__setattr__(self, "escape_policy", escape_policy)
        object.__setattr__(self, "notes", None if self.notes is None else str(self.notes))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "structure_head": str(self.structure_head),
            "prediction_head": str(self.prediction_head),
            "search_input_space": str(self.search_input_space),
            "pool_expansion_unit": str(self.pool_expansion_unit),
            "gradient_guidance_mode": str(self.gradient_guidance_mode),
            "basis_binding_mode": str(self.basis_binding_mode),
            "escape_policy": str(self.escape_policy),
            "notes": None if self.notes is None else str(self.notes),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ObjectGradientSignal:
    object_key: str
    gradient_score: float
    abs_gradient_score: float | None = None
    residual_gain: float | None = None
    stability: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_key", str(self.object_key).strip())
        object.__setattr__(self, "gradient_score", float(self.gradient_score))
        object.__setattr__(
            self,
            "abs_gradient_score",
            abs(float(self.gradient_score)) if self.abs_gradient_score is None else float(self.abs_gradient_score),
        )
        object.__setattr__(self, "residual_gain", None if self.residual_gain is None else float(self.residual_gain))
        object.__setattr__(self, "stability", None if self.stability is None else float(self.stability))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not str(self.object_key):
            raise ValueError("object gradient signal requires a non-empty object_key")

    def as_dict(self) -> dict[str, Any]:
        return {
            "object_key": str(self.object_key),
            "gradient_score": float(self.gradient_score),
            "abs_gradient_score": float(self.abs_gradient_score) if self.abs_gradient_score is not None else None,
            "residual_gain": float(self.residual_gain) if self.residual_gain is not None else None,
            "stability": float(self.stability) if self.stability is not None else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PoolExpansionCandidate:
    candidate_key: str
    source_object_keys: tuple[str, ...] = tuple()
    expression: str | None = None
    priority: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_key", str(self.candidate_key).strip())
        object.__setattr__(self, "source_object_keys", _normalize_fields(self.source_object_keys))
        object.__setattr__(self, "expression", None if self.expression is None else str(self.expression))
        object.__setattr__(self, "priority", None if self.priority is None else float(self.priority))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not str(self.candidate_key):
            raise ValueError("pool expansion candidate requires a non-empty candidate_key")

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": str(self.candidate_key),
            "source_object_keys": [str(v) for v in self.source_object_keys],
            "expression": None if self.expression is None else str(self.expression),
            "priority": float(self.priority) if self.priority is not None else None,
            "metadata": dict(self.metadata),
        }


class ObjectLevelGradientPoolExpander(Protocol):
    """Interface draft for object-level pool expansion in basis-conditioned symbolic stages."""

    name: str

    def rank_objects(
        self,
        stage_spec: SymbolicStageHeadSpec,
        basis_context: SymbolicBasisContext,
        gradient_signals: Sequence[ObjectGradientSignal],
    ) -> Sequence[ObjectGradientSignal]:
        ...

    def expand_pool(
        self,
        stage_spec: SymbolicStageHeadSpec,
        basis_context: SymbolicBasisContext,
        gradient_signals: Sequence[ObjectGradientSignal],
        *,
        max_new_candidates: int | None = None,
    ) -> Sequence[PoolExpansionCandidate]:
        ...


def is_basis_conditioned_stage(stage_spec: SymbolicStageHeadSpec) -> bool:
    return bool(
        str(stage_spec.search_input_space) == "basis_object_space"
        and str(stage_spec.pool_expansion_unit) == "basis_object"
        and str(stage_spec.structure_head) == "expression"
    )


def build_basis_discovery_stage_spec(
    *,
    gradient_guidance_mode: str = "raw_feature_gradient",
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SymbolicStageHeadSpec:
    return SymbolicStageHeadSpec(
        structure_head="basis_set",
        prediction_head="none",
        search_input_space="raw_feature_space",
        pool_expansion_unit="raw_feature",
        gradient_guidance_mode=gradient_guidance_mode,
        basis_binding_mode="off",
        escape_policy="fallback_to_generic",
        notes=notes,
        metadata={} if metadata is None else dict(metadata),
    )


def build_generic_expression_stage_spec(
    *,
    prediction_head: str = "point",
    gradient_guidance_mode: str = "raw_feature_gradient",
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SymbolicStageHeadSpec:
    return SymbolicStageHeadSpec(
        structure_head="expression",
        prediction_head=prediction_head,
        search_input_space="raw_feature_space",
        pool_expansion_unit="raw_feature",
        gradient_guidance_mode=gradient_guidance_mode,
        basis_binding_mode="off",
        escape_policy="fallback_to_generic",
        notes=notes,
        metadata={} if metadata is None else dict(metadata),
    )


def build_basis_conditioned_expression_stage_spec(
    *,
    prediction_head: str = "point",
    basis_binding_mode: str = "defining",
    gradient_guidance_mode: str = "basis_object_gradient",
    escape_policy: str = "forbid",
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SymbolicStageHeadSpec:
    return SymbolicStageHeadSpec(
        structure_head="expression",
        prediction_head=prediction_head,
        search_input_space="basis_object_space",
        pool_expansion_unit="basis_object",
        gradient_guidance_mode=gradient_guidance_mode,
        basis_binding_mode=basis_binding_mode,
        escape_policy=escape_policy,
        notes=notes,
        metadata={} if metadata is None else dict(metadata),
    )


__all__ = [
    "BasisObjectRef",
    "ObjectGradientSignal",
    "ObjectLevelGradientPoolExpander",
    "PoolExpansionCandidate",
    "SYMBOLIC_BASIS_BINDING_MODES",
    "SYMBOLIC_ESCAPE_POLICIES",
    "SYMBOLIC_GRADIENT_GUIDANCE_MODES",
    "SYMBOLIC_POOL_EXPANSION_UNITS",
    "SYMBOLIC_PREDICTION_HEADS",
    "SYMBOLIC_SEARCH_INPUT_SPACES",
    "SYMBOLIC_STRUCTURE_HEADS",
    "SymbolicBasisContext",
    "SymbolicStageHeadSpec",
    "build_basis_conditioned_expression_stage_spec",
    "build_basis_discovery_stage_spec",
    "build_generic_expression_stage_spec",
    "is_basis_conditioned_stage",
]
