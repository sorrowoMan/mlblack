from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.bias.base import OptimizationBias
from mlblack.core.contracts import ComponentContract
from mlblack.core.types import Feedback, UnknownState


class NoopBias(OptimizationBias):
    name = "noop_bias"
    context_requires = ()
    context_optional = ('trainer.context',)
    context_provides = ('bias.noop',)
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'provides bias.noop.'
    contract = ComponentContract(
        name=name,
        optional=("trainer.context",),
        provides=("bias.noop",),
        mutates=tuple(),
        supports_batch=True,
        metadata={"bias": "noop"},
    )


@dataclass(frozen=True)
class ObjectiveWeightBias(OptimizationBias):
    """Softly reweight objective dimensions before adapter.update()."""

    weights: Sequence[float]
    name = "objective_weight_bias"
    context_requires = ('feedback.objectives',)
    context_optional = ()
    context_provides = ('bias.objective_weights',)
    context_mutates = ('feedback.objectives',)
    context_cache = ()
    requires_metrics = ('objective',)
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: feedback.objectives; provides bias.objective_weights; mutates feedback.objectives.'
    contract = ComponentContract(
        name=name,
        requires=("feedback.objectives",),
        provides=("bias.objective_weights",),
        mutates=("feedback.objectives",),
        supports_batch=True,
        metadata={"bias": "objective_weight"},
    )

    def adjust_feedback(
        self,
        trainer: Any,
        states: Sequence[UnknownState],
        feedback: Sequence[Feedback],
        context: Mapping[str, Any],
    ) -> tuple[Feedback, ...]:
        _ = trainer
        _ = states
        _ = context
        weights = np.asarray(tuple(self.weights), dtype=float).reshape(-1)
        out: list[Feedback] = []
        for fb in feedback:
            objectives = np.asarray(fb.objectives, dtype=float).reshape(-1)
            if weights.shape[0] != objectives.shape[0]:
                raise ValueError("ObjectiveWeightBias weights must match objective dimension")
            metrics = {**dict(fb.metrics), "bias.objective_weight_applied": True}
            out.append(replace(fb, objectives=objectives * weights, metrics=metrics))
        return tuple(out)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "weights": [float(v) for v in self.weights]}


@dataclass(frozen=True)
class StateL2Bias(OptimizationBias):
    """Add a soft L2 penalty on UnknownState values to objective[0]."""

    weight: float = 1e-4
    exclude_bias: bool = True
    name = "state_l2_bias"
    context_requires = ('candidate.unknown_state', 'feedback.objectives')
    context_optional = ()
    context_provides = ('bias.state_l2_penalty',)
    context_mutates = ('feedback.objectives', 'feedback.metrics')
    context_cache = ()
    requires_metrics = ('objective',)
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: candidate.unknown_state, feedback.objectives; provides bias.state_l2_penalty; mutates feedback.objectives, feedback.metrics.'
    contract = ComponentContract(
        name=name,
        requires=("candidate.unknown_state", "feedback.objectives"),
        provides=("bias.state_l2_penalty",),
        mutates=("feedback.objectives", "feedback.metrics"),
        supports_batch=True,
        metadata={"bias": "state_l2"},
    )

    def adjust_feedback(
        self,
        trainer: Any,
        states: Sequence[UnknownState],
        feedback: Sequence[Feedback],
        context: Mapping[str, Any],
    ) -> tuple[Feedback, ...]:
        _ = trainer
        _ = context
        out: list[Feedback] = []
        for state, fb in zip(states, feedback):
            values = state.as_array()
            penalized = values[1:] if bool(self.exclude_bias) and values.shape[0] > 1 else values
            penalty = float(self.weight) * float(np.sum(penalized ** 2))
            objectives = np.asarray(fb.objectives, dtype=float).reshape(-1).copy()
            if objectives.size == 0:
                objectives = np.asarray([penalty], dtype=float)
            else:
                objectives[0] = float(objectives[0]) + penalty
            metrics = {**dict(fb.metrics), "bias.state_l2_penalty": penalty}
            out.append(replace(fb, objectives=objectives, metrics=metrics))
        return tuple(out)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "weight": float(self.weight), "exclude_bias": bool(self.exclude_bias)}


@dataclass(frozen=True)
class L2ScaleBias(OptimizationBias):
    """Expose a dynamic L2 scale to problems/adapters through context."""

    base: float = 0.0
    step_scale: float = 0.0
    max_value: float | None = None
    name = "l2_scale_bias"
    context_requires = ()
    context_optional = ('trainer.context', 'trainer.step')
    context_provides = ('bias.l2_scale',)
    context_mutates = ('trainer.context',)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'provides bias.l2_scale; mutates trainer.context.'
    contract = ComponentContract(
        name=name,
        optional=("trainer.context", "trainer.step"),
        provides=("bias.l2_scale",),
        mutates=("trainer.context",),
        supports_batch=True,
        metadata={"bias": "l2_scale"},
    )

    def project_context(self, trainer: Any, context: Mapping[str, Any]) -> Mapping[str, Any]:
        ctx = dict(context)
        step = int(getattr(trainer, "step_index", ctx.get("step", 0)) or 0)
        value = float(self.base) + (float(self.step_scale) * float(step))
        if self.max_value is not None:
            value = min(value, float(self.max_value))
        ctx["bias.l2_scale"] = float(value)
        return ctx

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base": float(self.base),
            "step_scale": float(self.step_scale),
            "max_value": self.max_value,
        }


@dataclass(frozen=True)
class ObjectivePolicyBias(OptimizationBias):
    """Context-aware objective reweighting policy.

    This is a soft policy layer: it changes feedback exposed to the adapter, not
    the underlying problem definition.
    """

    weights: Sequence[float] = (1.0,)
    metric_thresholds: Mapping[str, float] = None
    threshold_multiplier: float = 1.0
    name = "objective_policy_bias"
    context_requires = ('feedback.objectives',)
    context_optional = ('feedback.metrics', 'trainer.context')
    context_provides = ('bias.objective_policy',)
    context_mutates = ('feedback.objectives', 'feedback.metrics')
    context_cache = ()
    requires_metrics = ('objective',)
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: feedback.objectives; provides bias.objective_policy; mutates feedback.objectives, feedback.metrics.'
    contract = ComponentContract(
        name=name,
        requires=("feedback.objectives",),
        optional=("feedback.metrics", "trainer.context"),
        provides=("bias.objective_policy",),
        mutates=("feedback.objectives", "feedback.metrics"),
        supports_batch=True,
        metadata={"bias": "objective_policy"},
    )

    def adjust_feedback(
        self,
        trainer: Any,
        states: Sequence[UnknownState],
        feedback: Sequence[Feedback],
        context: Mapping[str, Any],
    ) -> tuple[Feedback, ...]:
        _ = trainer
        _ = states
        _ = context
        base_weights = np.asarray(tuple(self.weights), dtype=float).reshape(-1)
        thresholds = dict(self.metric_thresholds or {})
        out: list[Feedback] = []
        for fb in feedback:
            objectives = np.asarray(fb.objectives, dtype=float).reshape(-1)
            weights = _resize_weights(base_weights, objectives.shape[0])
            triggered = []
            for metric, threshold in thresholds.items():
                value = fb.metrics.get(metric)
                if value is not None and float(value) >= float(threshold):
                    weights = weights * float(self.threshold_multiplier)
                    triggered.append(metric)
            metrics = {
                **dict(fb.metrics),
                "bias.objective_policy_applied": True,
                "bias.objective_policy_triggered": tuple(triggered),
            }
            out.append(replace(fb, objectives=objectives * weights, metrics=metrics))
        return tuple(out)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weights": [float(v) for v in self.weights],
            "metric_thresholds": dict(self.metric_thresholds or {}),
            "threshold_multiplier": float(self.threshold_multiplier),
        }


@dataclass(frozen=True)
class BranchPolicyBias(OptimizationBias):
    """Expose branch preferences for conditional/piecewise representations."""

    preferred_branch: int | None = None
    branch_weights: Sequence[float] = tuple()
    context_key: str = "bias.branch"
    name = "branch_policy_bias"
    context_requires = ()
    context_optional = ('trainer.context', 'candidate.branch')
    context_provides = ('bias.branch',)
    context_mutates = ('trainer.context',)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'provides bias.branch; mutates trainer.context.'
    contract = ComponentContract(
        name=name,
        optional=("trainer.context", "candidate.branch"),
        provides=("bias.branch",),
        mutates=("trainer.context",),
        supports_batch=True,
        metadata={"bias": "branch_policy"},
    )

    def project_context(self, trainer: Any, context: Mapping[str, Any]) -> Mapping[str, Any]:
        _ = trainer
        ctx = dict(context)
        if self.preferred_branch is not None:
            ctx[f"{self.context_key}.preferred"] = int(self.preferred_branch)
        if self.branch_weights:
            ctx[f"{self.context_key}.weights"] = tuple(float(v) for v in self.branch_weights)
        return ctx

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "preferred_branch": self.preferred_branch,
            "branch_weights": [float(v) for v in self.branch_weights],
            "context_key": self.context_key,
        }


@dataclass(frozen=True)
class DynamicPoolBias(OptimizationBias):
    """Project a context-dependent candidate/model pool hint."""

    pool_name: str = "default"
    members: Sequence[str] = tuple()
    signal_key: str = "signal.pool"
    fallback_members: Sequence[str] = tuple()
    name = "dynamic_pool_bias"
    context_requires = ()
    context_optional = ('trainer.context', 'signal.pool')
    context_provides = ('bias.dynamic_pool',)
    context_mutates = ('trainer.context',)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'provides bias.dynamic_pool; mutates trainer.context.'
    contract = ComponentContract(
        name=name,
        optional=("trainer.context", "signal.pool"),
        provides=("bias.dynamic_pool",),
        mutates=("trainer.context",),
        supports_batch=True,
        metadata={"bias": "dynamic_pool"},
    )

    def project_context(self, trainer: Any, context: Mapping[str, Any]) -> Mapping[str, Any]:
        _ = trainer
        ctx = dict(context)
        active = tuple(str(v) for v in self.members)
        if self.signal_key and ctx.get(self.signal_key) == "fallback":
            active = tuple(str(v) for v in self.fallback_members)
        ctx["bias.dynamic_pool.name"] = self.pool_name
        ctx["bias.dynamic_pool.members"] = active
        return ctx

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pool_name": self.pool_name,
            "members": list(self.members),
            "signal_key": self.signal_key,
            "fallback_members": list(self.fallback_members),
        }


def _resize_weights(weights: np.ndarray, size: int) -> np.ndarray:
    if weights.shape[0] == size:
        return weights.copy()
    if weights.shape[0] == 1:
        return np.full(size, float(weights[0]), dtype=float)
    fixed = np.ones(size, dtype=float)
    fixed[: min(size, weights.shape[0])] = weights[: min(size, weights.shape[0])]
    return fixed

