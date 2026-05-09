from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class SymbolicOuterSearchCandidate:
    """Generic symbolic structure candidate exposed to an outer solver."""

    scenario_key: str
    basis_objects: tuple[Any, ...] = field(default_factory=tuple)
    chart_variants: tuple[Any, ...] = field(default_factory=tuple)
    realization_heads: tuple[Any, ...] = field(default_factory=tuple)
    branch_specs: tuple[Any, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SymbolicOuterEvaluationResult:
    """Generic evaluation payload returned from ML-side symbolic candidate scoring."""

    objectives: tuple[float, ...]
    violations: tuple[float, ...] = field(default_factory=tuple)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifact_refs: Mapping[str, Any] = field(default_factory=dict)
    audit: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SymbolicOuterEvaluationProxyConfig:
    n_total: int = 2400
    train_ratio: float = 0.8
    noise_std: float = 0.025
    seed: int = 42


class SymbolicOuterEvaluationProxyProtocol:
    """Protocol base for evaluate_individual/evaluate_population outer-solver bridges."""

    def evaluate_individual(self, candidate: SymbolicOuterSearchCandidate) -> SymbolicOuterEvaluationResult:
        raise NotImplementedError

    def evaluate_population(
        self,
        candidates: Sequence[SymbolicOuterSearchCandidate],
    ) -> tuple[SymbolicOuterEvaluationResult, ...]:
        return tuple(self.evaluate_individual(candidate) for candidate in tuple(candidates))


class SymbolicScenarioEvaluationProxy(SymbolicOuterEvaluationProxyProtocol):
    """Reusable scenario-bundle proxy shell for nsgablack outer search integration."""

    def __init__(
        self,
        *,
        bundle_builder: Callable[[str, int, float, float, int], tuple[Any, Any, dict[str, Any]]],
        config: SymbolicOuterEvaluationProxyConfig | None = None,
    ) -> None:
        self._bundle_builder = bundle_builder
        self.config = config or SymbolicOuterEvaluationProxyConfig()

    def build_bundle(self, scenario_key: str) -> tuple[Any, Any, dict[str, Any]]:
        return self._bundle_builder(
            str(scenario_key),
            int(self.config.n_total),
            float(self.config.train_ratio),
            float(self.config.noise_std),
            int(self.config.seed),
        )

    def evaluate_individual(self, candidate: SymbolicOuterSearchCandidate) -> SymbolicOuterEvaluationResult:
        definition, _bundle, truth_payload = self.build_bundle(candidate.scenario_key)
        audit = {
            "status": "bundle_built_only",
            "scenario": str(getattr(definition, "key", "")),
            "feature_names": tuple(getattr(definition, "feature_names", tuple())),
            "truth_formula": dict(truth_payload).get("formula"),
            "metadata": dict(candidate.metadata or {}),
        }
        return SymbolicOuterEvaluationResult(
            objectives=tuple(),
            violations=tuple(),
            metrics={},
            artifact_refs={},
            audit=audit,
        )


__all__ = [
    "SymbolicOuterEvaluationProxyConfig",
    "SymbolicOuterEvaluationProxyProtocol",
    "SymbolicOuterEvaluationResult",
    "SymbolicOuterSearchCandidate",
    "SymbolicScenarioEvaluationProxy",
]
