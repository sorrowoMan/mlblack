from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


SYMBOLIC_STRUCTURE_CONTRACT_VERSION = 1
SYMBOLIC_REGIME_DISCOVERY_MODES: tuple[str, ...] = (
    "global_only",
    "piecewise_gate",
    "soft_gate",
    "hybrid",
)
SYMBOLIC_BASIS_SCOPES: tuple[str, ...] = (
    "global",
    "local",
    "gate_conditioned",
    "global+local",
)
SYMBOLIC_ASSEMBLER_MODES: tuple[str, ...] = (
    "budgeted_symbolic_regression",
    "piecewise_budgeted_symbolic_regression",
)


def _normalize_name(value: str | None, default: str) -> str:
    text = str(value or "").strip().lower()
    return text or str(default)


def _normalize_fields(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return tuple()
    return tuple(str(v).strip() for v in tuple(values) if str(v).strip())


def _normalize_mode(value: str | None, *, allowed: Sequence[str], default: str, label: str) -> str:
    mode = _normalize_name(value, default)
    if mode not in tuple(allowed):
        raise ValueError(f"{label} must be one of {tuple(allowed)}, got '{value}'")
    return mode


@dataclass(frozen=True)
class SymbolicRegimeDiscoveryContract:
    contract_key: str = "regime_discovery"
    regime_mode: str = "global_only"
    consume: tuple[str, ...] = tuple()
    produce: tuple[str, ...] = tuple()
    mutate: tuple[str, ...] = tuple()
    checkpoint: tuple[str, ...] = tuple()
    replay: tuple[str, ...] = tuple()
    checkpointable: bool = False
    replayable: bool = False
    affects_family_signature: bool = True
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract_key", _normalize_name(self.contract_key, "regime_discovery"))
        object.__setattr__(
            self,
            "regime_mode",
            _normalize_mode(
                self.regime_mode,
                allowed=SYMBOLIC_REGIME_DISCOVERY_MODES,
                default="global_only",
                label="symbolic regime discovery mode",
            ),
        )
        object.__setattr__(self, "consume", _normalize_fields(self.consume))
        object.__setattr__(self, "produce", _normalize_fields(self.produce))
        object.__setattr__(self, "mutate", _normalize_fields(self.mutate))
        object.__setattr__(self, "checkpoint", _normalize_fields(self.checkpoint))
        object.__setattr__(self, "replay", _normalize_fields(self.replay))
        object.__setattr__(self, "checkpointable", bool(self.checkpointable))
        object.__setattr__(self, "replayable", bool(self.replayable))
        object.__setattr__(self, "affects_family_signature", bool(self.affects_family_signature))
        object.__setattr__(self, "notes", None if self.notes is None else str(self.notes))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_version": int(SYMBOLIC_STRUCTURE_CONTRACT_VERSION),
            "contract_type": "SymbolicRegimeDiscoveryContract",
            "contract_key": str(self.contract_key),
            "regime_mode": str(self.regime_mode),
            "consume": [str(v) for v in self.consume],
            "produce": [str(v) for v in self.produce],
            "mutate": [str(v) for v in self.mutate],
            "checkpoint": [str(v) for v in self.checkpoint],
            "replay": [str(v) for v in self.replay],
            "checkpointable": bool(self.checkpointable),
            "replayable": bool(self.replayable),
            "affects_family_signature": bool(self.affects_family_signature),
            "notes": None if self.notes is None else str(self.notes),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SymbolicBasisDiscoveryContract:
    contract_key: str = "basis_discovery"
    basis_scope: str = "global"
    consume: tuple[str, ...] = tuple()
    produce: tuple[str, ...] = tuple()
    mutate: tuple[str, ...] = tuple()
    checkpoint: tuple[str, ...] = tuple()
    replay: tuple[str, ...] = tuple()
    checkpointable: bool = False
    replayable: bool = False
    affects_family_signature: bool = True
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract_key", _normalize_name(self.contract_key, "basis_discovery"))
        object.__setattr__(
            self,
            "basis_scope",
            _normalize_mode(
                self.basis_scope,
                allowed=SYMBOLIC_BASIS_SCOPES,
                default="global",
                label="symbolic basis scope",
            ),
        )
        object.__setattr__(self, "consume", _normalize_fields(self.consume))
        object.__setattr__(self, "produce", _normalize_fields(self.produce))
        object.__setattr__(self, "mutate", _normalize_fields(self.mutate))
        object.__setattr__(self, "checkpoint", _normalize_fields(self.checkpoint))
        object.__setattr__(self, "replay", _normalize_fields(self.replay))
        object.__setattr__(self, "checkpointable", bool(self.checkpointable))
        object.__setattr__(self, "replayable", bool(self.replayable))
        object.__setattr__(self, "affects_family_signature", bool(self.affects_family_signature))
        object.__setattr__(self, "notes", None if self.notes is None else str(self.notes))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_version": int(SYMBOLIC_STRUCTURE_CONTRACT_VERSION),
            "contract_type": "SymbolicBasisDiscoveryContract",
            "contract_key": str(self.contract_key),
            "basis_scope": str(self.basis_scope),
            "consume": [str(v) for v in self.consume],
            "produce": [str(v) for v in self.produce],
            "mutate": [str(v) for v in self.mutate],
            "checkpoint": [str(v) for v in self.checkpoint],
            "replay": [str(v) for v in self.replay],
            "checkpointable": bool(self.checkpointable),
            "replayable": bool(self.replayable),
            "affects_family_signature": bool(self.affects_family_signature),
            "notes": None if self.notes is None else str(self.notes),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BudgetedSymbolicAssemblerContract:
    contract_key: str = "budgeted_symbolic_assembler"
    assembler_mode: str = "budgeted_symbolic_regression"
    consume: tuple[str, ...] = tuple()
    produce: tuple[str, ...] = tuple()
    mutate: tuple[str, ...] = tuple()
    checkpoint: tuple[str, ...] = tuple()
    replay: tuple[str, ...] = tuple()
    checkpointable: bool = False
    replayable: bool = False
    affects_family_signature: bool = True
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "contract_key",
            _normalize_name(self.contract_key, "budgeted_symbolic_assembler"),
        )
        object.__setattr__(
            self,
            "assembler_mode",
            _normalize_mode(
                self.assembler_mode,
                allowed=SYMBOLIC_ASSEMBLER_MODES,
                default="budgeted_symbolic_regression",
                label="budgeted symbolic assembler mode",
            ),
        )
        object.__setattr__(self, "consume", _normalize_fields(self.consume))
        object.__setattr__(self, "produce", _normalize_fields(self.produce))
        object.__setattr__(self, "mutate", _normalize_fields(self.mutate))
        object.__setattr__(self, "checkpoint", _normalize_fields(self.checkpoint))
        object.__setattr__(self, "replay", _normalize_fields(self.replay))
        object.__setattr__(self, "checkpointable", bool(self.checkpointable))
        object.__setattr__(self, "replayable", bool(self.replayable))
        object.__setattr__(self, "affects_family_signature", bool(self.affects_family_signature))
        object.__setattr__(self, "notes", None if self.notes is None else str(self.notes))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_version": int(SYMBOLIC_STRUCTURE_CONTRACT_VERSION),
            "contract_type": "BudgetedSymbolicAssemblerContract",
            "contract_key": str(self.contract_key),
            "assembler_mode": str(self.assembler_mode),
            "consume": [str(v) for v in self.consume],
            "produce": [str(v) for v in self.produce],
            "mutate": [str(v) for v in self.mutate],
            "checkpoint": [str(v) for v in self.checkpoint],
            "replay": [str(v) for v in self.replay],
            "checkpointable": bool(self.checkpointable),
            "replayable": bool(self.replayable),
            "affects_family_signature": bool(self.affects_family_signature),
            "notes": None if self.notes is None else str(self.notes),
            "metadata": dict(self.metadata),
        }


def build_symbolic_regime_discovery_contract(
    *,
    task: str = "point",
    supports_piecewise: bool = False,
) -> SymbolicRegimeDiscoveryContract:
    task_name = _normalize_name(task, "point")
    regime_mode = "piecewise_gate" if supports_piecewise else "global_only"
    return SymbolicRegimeDiscoveryContract(
        regime_mode=regime_mode,
        consume=(
            "candidate_pool",
            "gate_feature_candidates",
            "residual_ref",
            "search_trace.iterations",
            "structure_config.gate_threshold",
            "structure_config.gate_min_leaf",
        ),
        produce=(
            "regime_partition",
            "gate_basis",
            "regime_assignments",
            "regime_manifest",
        )
        if supports_piecewise
        else (
            "global_regime",
            "regime_manifest",
        ),
        mutate=(
            "search_state.regime_partition",
            "search_state.gate_basis",
            "search_trace.iterations",
        ),
        checkpoint=(
            "search_state.regime_partition",
            "search_state.gate_basis",
            "regime_manifest",
        ),
        replay=(
            "structure_config.gate_threshold",
            "structure_config.gate_min_leaf",
            "regime_manifest",
        ),
        checkpointable=True,
        replayable=True,
        affects_family_signature=True,
        notes=(
            "Discovers whether symbolic structure stays global or must branch into gate-controlled local regimes "
            "before basis discovery and final assembly."
        ),
        metadata={
            "task": str(task_name),
            "supports_piecewise": bool(supports_piecewise),
            "activation_outputs": ["gate_basis", "regime_partition"],
        },
    )


def build_symbolic_basis_discovery_contract(
    *,
    supports_piecewise: bool = False,
) -> SymbolicBasisDiscoveryContract:
    basis_scope = "global+local" if supports_piecewise else "global"
    return SymbolicBasisDiscoveryContract(
        basis_scope=basis_scope,
        consume=(
            "primitive_registry",
            "candidate_pool",
            "regime_partition",
            "residual_ref",
            "gradient_signal.feature_priority",
            "search_trace.iterations",
        ),
        produce=(
            "basis_candidates",
            "basis_scores",
            "selected_basis",
            "basis_overlap_report",
            "basis_semantics",
            "residual_complementarity_report",
            "semantic_dedup_report",
            "piecewise_gate_basis",
        ),
        mutate=(
            "candidate_pool",
            "search_state.selected_basis",
            "search_state.basis_overlap",
            "search_state.residual_complementarity",
            "search_state.semantic_dedup",
            "search_trace.iterations",
        ),
        checkpoint=(
            "search_state.selected_basis",
            "basis_scores",
            "basis_overlap_report",
            "basis_semantics",
            "residual_complementarity_report",
            "semantic_dedup_report",
        ),
        replay=(
            "primitive_registry",
            "candidate_pool",
            "basis_overlap_report",
            "basis_semantics",
            "residual_complementarity_report",
            "semantic_dedup_report",
        ),
        checkpointable=True,
        replayable=True,
        affects_family_signature=True,
        notes=(
            "Searches for reusable symbolic basis terms under correlation control, residual complementarity, semantic "
            "de-duplication, and cross-fold stability objectives before the final small-budget symbolic composition stage."
        ),
        metadata={
            "orthogonality_objectives": (
                "low_pairwise_correlation",
                "residual_complementarity",
                "semantic_deduplication",
                "fold_stability",
                "piecewise_gate_readiness",
            ),
            "supports_piecewise": bool(supports_piecewise),
            "reports": (
                "basis_overlap_report",
                "residual_complementarity_report",
                "semantic_dedup_report",
            ),
        },
    )


def build_budgeted_symbolic_assembler_contract(
    *,
    supports_piecewise: bool = False,
) -> BudgetedSymbolicAssemblerContract:
    assembler_mode = (
        "piecewise_budgeted_symbolic_regression"
        if supports_piecewise
        else "budgeted_symbolic_regression"
    )
    return BudgetedSymbolicAssemblerContract(
        assembler_mode=assembler_mode,
        consume=(
            "selected_basis",
            "basis_scores",
            "regime_partition",
            "assembler_budget",
            "parameter_backend",
            "task_head",
        ),
        produce=(
            "assembled_expression",
            "assembly_score",
            "assembly_trace",
            "assembly_budget_usage",
            "piecewise_gate_basis",
        ),
        mutate=(
            "search_state.assembled_expression",
            "search_state.assembly_budget_usage",
            "search_trace.iterations",
        ),
        checkpoint=(
            "search_state.assembled_expression",
            "assembly_trace",
            "assembly_budget_usage",
        ),
        replay=(
            "selected_basis",
            "assembler_budget",
            "assembly_trace",
        ),
        checkpointable=True,
        replayable=True,
        affects_family_signature=True,
        notes=(
            "Runs a deliberately small-budget second-stage symbolic regression over discovered basis terms, keeping "
            "assembly complexity bounded after the heavier regime/basis search stages, including optional gate-conditioned "
            "local assembly when piecewise basis terms are active."
        ),
        metadata={
            "budget_axes": (
                "beam_width",
                "max_terms",
                "max_depth",
                "max_interaction_order",
                "max_piecewise_branches",
            ),
            "supports_piecewise": bool(supports_piecewise),
            "budget_scale": "small",
        },
    )


__all__ = [
    "BudgetedSymbolicAssemblerContract",
    "build_budgeted_symbolic_assembler_contract",
    "build_symbolic_basis_discovery_contract",
    "build_symbolic_regime_discovery_contract",
    "SYMBOLIC_ASSEMBLER_MODES",
    "SYMBOLIC_BASIS_SCOPES",
    "SYMBOLIC_REGIME_DISCOVERY_MODES",
    "SYMBOLIC_STRUCTURE_CONTRACT_VERSION",
    "SymbolicBasisDiscoveryContract",
    "SymbolicRegimeDiscoveryContract",
]
