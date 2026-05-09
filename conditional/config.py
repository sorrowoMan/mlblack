from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from conditional.composer.spec import (
    RoutePlusPrimitivesSpec,
    RouteThenFormulaSpec,
    SharedBackboneResidualSpec,
)
from conditional.primitives import (
    BinaryGate,
    ConditionalPrimitiveSpec,
    HingePrimitive,
    OneHotGate,
    PiecewisePrimitive,
    SoftGatePrimitive,
    StepPrimitive,
)
from conditional.router import RouterPolicyAdapter, adapt_router_policy


def _normalize_feature_names(value: Sequence[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        text = str(value).strip()
        return tuple() if not text else (text,)
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        name = str(item).strip()
        if not name or name in seen:
            continue
        out.append(name)
        seen.add(name)
    return tuple(out)


@dataclass(frozen=True)
class FeatureRoleConfig:
    router_features: tuple[str, ...] = ()
    binary_gate_features: tuple[str, ...] = ()
    onehot_gate_features: tuple[str, ...] = ()
    threshold_features: tuple[str, ...] = ()
    smooth_features: tuple[str, ...] = ()
    excluded_features: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "router_features", _normalize_feature_names(self.router_features))
        object.__setattr__(self, "binary_gate_features", _normalize_feature_names(self.binary_gate_features))
        object.__setattr__(self, "onehot_gate_features", _normalize_feature_names(self.onehot_gate_features))
        object.__setattr__(self, "threshold_features", _normalize_feature_names(self.threshold_features))
        object.__setattr__(self, "smooth_features", _normalize_feature_names(self.smooth_features))
        object.__setattr__(self, "excluded_features", _normalize_feature_names(self.excluded_features))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def roles_for(self, feature_name: str) -> tuple[str, ...]:
        name = str(feature_name)
        roles: list[str] = []
        if name in self.router_features:
            roles.append("router")
        if name in self.binary_gate_features:
            roles.append("binary_gate")
        if name in self.onehot_gate_features:
            roles.append("onehot_gate")
        if name in self.threshold_features:
            roles.append("threshold")
        if name in self.smooth_features:
            roles.append("smooth")
        if name in self.excluded_features:
            roles.append("excluded")
        return tuple(roles)

    def role_index(self) -> dict[str, tuple[str, ...]]:
        names = set(self.router_features)
        names.update(self.binary_gate_features)
        names.update(self.onehot_gate_features)
        names.update(self.threshold_features)
        names.update(self.smooth_features)
        names.update(self.excluded_features)
        return {str(name): self.roles_for(str(name)) for name in sorted(names)}


def build_feature_role_config(
    feature_names: Sequence[str],
    *,
    router_features: Sequence[str] | str | None = None,
    binary_gate_features: Sequence[str] | str | None = None,
    onehot_gate_features: Sequence[str] | str | None = None,
    threshold_features: Sequence[str] | str | None = None,
    smooth_features: Sequence[str] | str | None = None,
    excluded_features: Sequence[str] | str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FeatureRoleConfig:
    feature_names_norm = _normalize_feature_names(feature_names)
    router = _normalize_feature_names(router_features)
    binary_gate = _normalize_feature_names(binary_gate_features)
    onehot_gate = _normalize_feature_names(onehot_gate_features)
    threshold = _normalize_feature_names(threshold_features)
    excluded = _normalize_feature_names(excluded_features)
    if smooth_features is None:
        forbidden = set(router) | set(onehot_gate) | set(excluded)
        smooth = tuple(name for name in feature_names_norm if name not in forbidden)
    else:
        smooth = _normalize_feature_names(smooth_features)
    return FeatureRoleConfig(
        router_features=router,
        binary_gate_features=binary_gate,
        onehot_gate_features=onehot_gate,
        threshold_features=threshold,
        smooth_features=smooth,
        excluded_features=excluded,
        metadata=dict(metadata or {"feature_names": feature_names_norm}),
    )


@dataclass(frozen=True)
class BinaryGateBinding:
    feature_name: str
    threshold: float = 0.5
    positive_value: float = 1.0
    negative_value: float = 0.0

    def to_primitive(self) -> ConditionalPrimitiveSpec:
        return BinaryGate(
            feature_name=str(self.feature_name),
            threshold=float(self.threshold),
            positive_value=float(self.positive_value),
            negative_value=float(self.negative_value),
        ).to_spec()


@dataclass(frozen=True)
class OneHotGateBinding:
    feature_name: str
    categories: tuple[str, ...] = ()

    def to_primitive(self) -> ConditionalPrimitiveSpec:
        return OneHotGate(
            feature_name=str(self.feature_name),
            categories=tuple(str(v) for v in self.categories),
        ).to_spec()


@dataclass(frozen=True)
class ThresholdPrimitiveBinding:
    feature_name: str
    primitive_family: str = "hinge"
    cuts: tuple[float, ...] = (0.0,)
    direction: str = "positive"
    multiplier_feature: str | None = None
    slope: float = 4.0
    left_mode: str = "identity"
    right_mode: str = "identity"

    def to_primitives(self) -> tuple[ConditionalPrimitiveSpec, ...]:
        family = str(self.primitive_family).strip().lower()
        cuts = tuple(float(v) for v in self.cuts) if self.cuts else (0.0,)
        out: list[ConditionalPrimitiveSpec] = []
        for cut in cuts:
            if family == "hinge":
                out.append(
                    HingePrimitive(
                        feature_name=str(self.feature_name),
                        cut=float(cut),
                        direction=str(self.direction),
                        multiplier_feature=None if self.multiplier_feature is None else str(self.multiplier_feature),
                    ).to_spec()
                )
            elif family == "step":
                out.append(
                    StepPrimitive(
                        feature_name=str(self.feature_name),
                        cut=float(cut),
                        slope=float(self.slope),
                        multiplier_feature=None if self.multiplier_feature is None else str(self.multiplier_feature),
                    ).to_spec()
                )
            elif family == "soft_gate":
                out.append(
                    SoftGatePrimitive(
                        feature_name=str(self.feature_name),
                        cut=float(cut),
                        slope=float(self.slope),
                        multiplier_feature=None if self.multiplier_feature is None else str(self.multiplier_feature),
                    ).to_spec()
                )
            elif family == "piecewise":
                out.append(
                    PiecewisePrimitive(
                        feature_name=str(self.feature_name),
                        cut=float(cut),
                        left_mode=str(self.left_mode),
                        right_mode=str(self.right_mode),
                        multiplier_feature=None if self.multiplier_feature is None else str(self.multiplier_feature),
                    ).to_spec()
                )
            else:
                raise ValueError(f"unknown threshold primitive family: {self.primitive_family}")
        return tuple(out)


@dataclass(frozen=True)
class AutoThresholdBindingSpec:
    primitive_family: str = "hinge"
    quantiles: tuple[float, ...] = (0.5,)
    directions: tuple[str, ...] = ("positive",)
    multiplier_feature: str | None = None
    slope: float = 4.0
    left_mode: str = "identity"
    right_mode: str = "identity"
    min_unique_values: int = 6
    min_cut_separation_ratio: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "quantiles",
            tuple(
                float(v)
                for v in self.quantiles
                if np.isfinite(float(v)) and 0.0 < float(v) < 1.0
            )
            or (0.5,),
        )
        object.__setattr__(
            self,
            "directions",
            tuple(str(v).strip().lower() for v in self.directions if str(v).strip()) or ("positive",),
        )
        object.__setattr__(self, "min_unique_values", int(max(2, self.min_unique_values)))
        object.__setattr__(self, "min_cut_separation_ratio", float(max(0.0, self.min_cut_separation_ratio)))
        object.__setattr__(self, "slope", float(max(1.0, self.slope)))


def _resolve_auto_threshold_spec(
    value: AutoThresholdBindingSpec | Mapping[str, Any] | None,
) -> AutoThresholdBindingSpec:
    if value is None:
        return AutoThresholdBindingSpec()
    if isinstance(value, AutoThresholdBindingSpec):
        return value
    return AutoThresholdBindingSpec(**dict(value))


def _derive_threshold_cuts(
    values: np.ndarray,
    *,
    quantiles: Sequence[float],
    min_unique_values: int,
    min_cut_separation_ratio: float,
) -> tuple[float, ...]:
    z = np.asarray(values, dtype=float).reshape(-1)
    z = z[np.isfinite(z)]
    if z.size < int(max(4, min_unique_values)):
        return tuple()
    unique = np.unique(z)
    if unique.size < int(min_unique_values):
        return tuple()
    spread = float(np.quantile(z, 0.9) - np.quantile(z, 0.1))
    if not np.isfinite(spread) or spread <= 1e-12:
        spread = float(np.max(z) - np.min(z))
    if not np.isfinite(spread) or spread <= 1e-12:
        return tuple()
    min_gap = float(max(1e-8, spread * float(min_cut_separation_ratio)))
    cuts: list[float] = []
    for q in quantiles:
        cut = float(np.quantile(z, float(q)))
        if not np.isfinite(cut):
            continue
        if any(abs(cut - prev) < min_gap for prev in cuts):
            continue
        cuts.append(cut)
    return tuple(cuts)


def build_auto_threshold_bindings(
    X: np.ndarray,
    feature_names: Sequence[str],
    *,
    threshold_features: Sequence[str] | str | None = None,
    default_spec: AutoThresholdBindingSpec | Mapping[str, Any] | None = None,
    per_feature_specs: Mapping[str, AutoThresholdBindingSpec | Mapping[str, Any]] | None = None,
) -> tuple[ThresholdPrimitiveBinding, ...]:
    x = np.asarray(X, dtype=float)
    feature_names_norm = _normalize_feature_names(feature_names)
    name_to_idx = {str(name): int(idx) for idx, name in enumerate(feature_names_norm)}
    target_features = (
        _normalize_feature_names(threshold_features)
        if threshold_features is not None
        else feature_names_norm
    )
    default_auto_spec = _resolve_auto_threshold_spec(default_spec)
    override_specs = {str(k): _resolve_auto_threshold_spec(v) for k, v in dict(per_feature_specs or {}).items()}
    out: list[ThresholdPrimitiveBinding] = []
    seen: set[ThresholdPrimitiveBinding] = set()

    for feature_name in target_features:
        idx = name_to_idx.get(str(feature_name))
        if idx is None or idx < 0 or idx >= x.shape[1]:
            continue
        spec = override_specs.get(str(feature_name), default_auto_spec)
        cuts = _derive_threshold_cuts(
            x[:, idx],
            quantiles=spec.quantiles,
            min_unique_values=int(spec.min_unique_values),
            min_cut_separation_ratio=float(spec.min_cut_separation_ratio),
        )
        if not cuts:
            continue
        multiplier_feature = str(spec.multiplier_feature) if spec.multiplier_feature is not None else None
        if multiplier_feature is not None and multiplier_feature not in name_to_idx:
            multiplier_feature = None
        family = str(spec.primitive_family).strip().lower()
        if family == "hinge":
            directions = tuple(str(v) for v in spec.directions)
            for direction in directions:
                binding = ThresholdPrimitiveBinding(
                    feature_name=str(feature_name),
                    primitive_family="hinge",
                    cuts=tuple(float(v) for v in cuts),
                    direction=str(direction),
                    multiplier_feature=multiplier_feature,
                    slope=float(spec.slope),
                    left_mode=str(spec.left_mode),
                    right_mode=str(spec.right_mode),
                )
                if binding not in seen:
                    seen.add(binding)
                    out.append(binding)
            continue
        binding = ThresholdPrimitiveBinding(
            feature_name=str(feature_name),
            primitive_family=family,
            cuts=tuple(float(v) for v in cuts),
            direction="positive",
            multiplier_feature=multiplier_feature,
            slope=float(spec.slope),
            left_mode=str(spec.left_mode),
            right_mode=str(spec.right_mode),
        )
        if binding not in seen:
            seen.add(binding)
            out.append(binding)
    return tuple(out)


@dataclass(frozen=True)
class ConditionalComposerConfig:
    mode: str = "auto"
    share_candidate_pool: bool = True
    branch_formula_name: str = "symbolic_formula"
    backbone_name: str = "shared_backbone"
    residual_name: str = "regime_residual"
    residual_target: str = "residual"

    def resolve_mode(self, *, has_router: bool, has_primitives: bool) -> str:
        mode = str(self.mode).strip().lower()
        if mode != "auto":
            return mode
        if has_router and has_primitives:
            return "route_plus_primitives"
        if has_router:
            return "route_then_formula"
        return "formula_only_primitives"


@dataclass(frozen=True)
class ConditionalConfig:
    enabled: bool = True
    feature_roles: FeatureRoleConfig = field(default_factory=FeatureRoleConfig)
    router_policy: RouterPolicyAdapter | object | None = None
    binary_gates: tuple[BinaryGateBinding, ...] = ()
    onehot_gates: tuple[OneHotGateBinding, ...] = ()
    threshold_primitives: tuple[ThresholdPrimitiveBinding, ...] = ()
    composer: ConditionalComposerConfig = field(default_factory=ConditionalComposerConfig)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "binary_gates", tuple(self.binary_gates))
        object.__setattr__(self, "onehot_gates", tuple(self.onehot_gates))
        object.__setattr__(self, "threshold_primitives", tuple(self.threshold_primitives))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def adapted_router_policy(self) -> RouterPolicyAdapter | None:
        if self.router_policy is None:
            return None
        adapted = adapt_router_policy(self.router_policy)
        return adapted if adapted.gate_names else None

    def primitive_specs(self) -> tuple[ConditionalPrimitiveSpec, ...]:
        out: list[ConditionalPrimitiveSpec] = []
        for binding in self.binary_gates:
            out.append(binding.to_primitive())
        for binding in self.onehot_gates:
            out.append(binding.to_primitive())
        for binding in self.threshold_primitives:
            out.extend(binding.to_primitives())
        return tuple(out)

    def composer_spec(self) -> RouteThenFormulaSpec | SharedBackboneResidualSpec | RoutePlusPrimitivesSpec | None:
        if not bool(self.enabled):
            return None
        router_policy = self.adapted_router_policy
        primitive_specs = self.primitive_specs()
        has_router = router_policy is not None
        has_primitives = bool(primitive_specs)
        mode = self.composer.resolve_mode(has_router=has_router, has_primitives=has_primitives)

        if mode == "formula_only_primitives":
            return None
        if router_policy is None:
            raise ValueError("conditional composer mode requires a router_policy with gate_names")
        if mode == "route_then_formula":
            return RouteThenFormulaSpec(
                router_policy=router_policy,
                branch_formula_name=str(self.composer.branch_formula_name),
                share_candidate_pool=bool(self.composer.share_candidate_pool),
            )
        if mode == "shared_backbone_regime_residual":
            return SharedBackboneResidualSpec(
                router_policy=router_policy,
                backbone_name=str(self.composer.backbone_name),
                residual_name=str(self.composer.residual_name),
                residual_target=str(self.composer.residual_target),
                share_candidate_pool=bool(self.composer.share_candidate_pool),
            )
        if mode == "route_plus_primitives":
            return RoutePlusPrimitivesSpec(
                router_policy=router_policy,
                primitives=tuple(primitive_specs),
                branch_formula_name=str(self.composer.branch_formula_name),
            )
        raise ValueError(f"unknown conditional composer mode: {self.composer.mode}")


__all__ = [
    "AutoThresholdBindingSpec",
    "BinaryGateBinding",
    "build_auto_threshold_bindings",
    "ConditionalComposerConfig",
    "ConditionalConfig",
    "FeatureRoleConfig",
    "OneHotGateBinding",
    "ThresholdPrimitiveBinding",
    "build_feature_role_config",
]
