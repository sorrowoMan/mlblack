from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


SYMBOLIC_SEARCH_BINDING_LEVELS: tuple[str, ...] = ("optional", "bound", "defining")
SYMBOLIC_SEARCH_MECHANISM_KINDS: tuple[str, ...] = (
    "search_policy",
    "structure_expansion",
    "search_guidance",
    "parameter_refinement",
    "search_cache",
    "search_memory",
)


def _normalize_name(value: str | None, default: str) -> str:
    text = str(value or "").strip().lower()
    return text or str(default)


def _normalize_binding_level(value: str | None) -> str:
    level = _normalize_name(value, "optional")
    if level not in SYMBOLIC_SEARCH_BINDING_LEVELS:
        raise ValueError(
            f"symbolic search binding_level must be one of {SYMBOLIC_SEARCH_BINDING_LEVELS}, got '{value}'"
        )
    return level


def _normalize_kind(value: str | None, default: str) -> str:
    kind = _normalize_name(value, default)
    if kind not in SYMBOLIC_SEARCH_MECHANISM_KINDS:
        raise ValueError(
            f"symbolic search mechanism kind must be one of {SYMBOLIC_SEARCH_MECHANISM_KINDS}, got '{value}'"
        )
    return kind


def _normalize_fields(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return tuple()
    return tuple(str(v).strip() for v in tuple(values) if str(v).strip())


@dataclass(frozen=True)
class SymbolicSearchMechanismContract:
    mechanism_key: str
    mechanism_kind: str
    binding_level: str = "optional"
    consume: tuple[str, ...] = tuple()
    produce: tuple[str, ...] = tuple()
    mutate: tuple[str, ...] = tuple()
    checkpoint: tuple[str, ...] = tuple()
    replay: tuple[str, ...] = tuple()
    checkpointable: bool = False
    replayable: bool = False
    affects_family_signature: bool = False
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mechanism_key", _normalize_name(self.mechanism_key, "search_mechanism"))
        object.__setattr__(self, "mechanism_kind", _normalize_kind(self.mechanism_kind, "search_policy"))
        object.__setattr__(self, "binding_level", _normalize_binding_level(self.binding_level))
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
            "contract_version": 1,
            "mechanism_key": str(self.mechanism_key),
            "mechanism_kind": str(self.mechanism_kind),
            "binding_level": str(self.binding_level),
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


def coerce_symbolic_search_mechanism_contract(
    value: SymbolicSearchMechanismContract | Mapping[str, Any],
) -> SymbolicSearchMechanismContract:
    if isinstance(value, SymbolicSearchMechanismContract):
        return value
    raw = dict(value)
    return SymbolicSearchMechanismContract(
        mechanism_key=str(raw.get("mechanism_key", "search_mechanism")),
        mechanism_kind=str(raw.get("mechanism_kind", "search_policy")),
        binding_level=str(raw.get("binding_level", "optional")),
        consume=tuple(raw.get("consume", tuple())),
        produce=tuple(raw.get("produce", tuple())),
        mutate=tuple(raw.get("mutate", tuple())),
        checkpoint=tuple(raw.get("checkpoint", tuple())),
        replay=tuple(raw.get("replay", tuple())),
        checkpointable=bool(raw.get("checkpointable", False)),
        replayable=bool(raw.get("replayable", False)),
        affects_family_signature=bool(raw.get("affects_family_signature", False)),
        notes=raw.get("notes"),
        metadata=dict(raw.get("metadata", {})),
    )


def serialize_symbolic_search_mechanism_contracts(
    values: Sequence[SymbolicSearchMechanismContract | Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if values is None:
        return []
    return [coerce_symbolic_search_mechanism_contract(value).as_dict() for value in tuple(values)]


def build_symbolic_search_mechanism_contracts() -> tuple[SymbolicSearchMechanismContract, ...]:
    common_source = ("core/symbolic/symbolic_structure_search.py",)
    return (
        SymbolicSearchMechanismContract(
            mechanism_key="beam_selection",
            mechanism_kind="search_policy",
            binding_level="bound",
            consume=(
                "candidate_pool",
                "candidate_scores",
                "search_trace.iterations",
                "structure_config.auto_nested_beam_width",
            ),
            produce=("beam_candidates", "beam_rankings"),
            mutate=("search_trace.iterations", "search_state.beam_frontier"),
            checkpoint=("search_state.beam_frontier", "search_trace.iterations", "score_trace"),
            replay=("search_state.beam_frontier", "search_trace.iterations", "structure_config.auto_nested_beam_width"),
            checkpointable=True,
            replayable=True,
            affects_family_signature=True,
            notes="Controls bounded candidate ranking and frontier retention during nested symbolic expansion.",
            metadata={"source_modules": common_source, "default_enabled": True},
        ),
        SymbolicSearchMechanismContract(
            mechanism_key="candidate_budget_policy",
            mechanism_kind="search_policy",
            binding_level="bound",
            consume=(
                "candidate_pool",
                "activation_plan.family_budget",
                "structure_config.max_new_terms",
                "structure_config.interaction_budget_mode",
            ),
            produce=("budget_policy", "budget_filtered_candidates"),
            mutate=("candidate_pool", "search_state.candidate_budget"),
            checkpoint=("search_state.candidate_budget", "search_trace.iterations"),
            replay=(
                "activation_plan.family_budget",
                "structure_config.max_new_terms",
                "structure_config.interaction_budget_mode",
            ),
            checkpointable=True,
            replayable=True,
            affects_family_signature=True,
            notes="Defines how symbolic family budgets and candidate caps are enforced before scoring and acceptance.",
            metadata={
                "source_modules": (
                    "core/symbolic/symbolic_structure_search.py",
                    "core/symbolic/feature_space/candidate_pool.py",
                ),
                "default_enabled": True,
            },
        ),
        SymbolicSearchMechanismContract(
            mechanism_key="nested_expression_expansion",
            mechanism_kind="structure_expansion",
            binding_level="optional",
            consume=(
                "candidate_pool.base_terms",
                "structure_config.nested_mode",
                "structure_config.auto_nested_allowed_ops",
                "structure_config.auto_nested_beam_width",
            ),
            produce=("nested_candidate_specs", "nested_candidates"),
            mutate=("candidate_pool", "search_state.nested_expansion"),
            checkpoint=("candidate_pool", "search_state.nested_expansion"),
            replay=(
                "candidate_pool.base_terms",
                "structure_config.nested_mode",
                "structure_config.auto_nested_allowed_ops",
            ),
            checkpointable=True,
            replayable=True,
            affects_family_signature=True,
            notes="Expands unary-nested symbolic candidates without redefining the overall symbolic family backbone.",
            metadata={"source_modules": common_source, "default_enabled": True},
        ),
        SymbolicSearchMechanismContract(
            mechanism_key="gradient_projection_guidance",
            mechanism_kind="search_guidance",
            binding_level="optional",
            consume=(
                "residual_ref",
                "gradient_signal.signal_signature",
                "gradient_signal.feature_priority",
                "structure_config.grad_projection_partner_orders",
            ),
            produce=("gradient_summary", "projected_candidates", "gradient_guidance_scores"),
            mutate=("search_trace.iterations", "search_state.gradient_guidance"),
            checkpoint=("gradient_summary", "search_state.gradient_guidance", "search_trace.iterations"),
            replay=(
                "gradient_signal.signal_signature",
                "structure_config.grad_projection_partner_orders",
                "structure_config.grad_projection_topk_focus",
            ),
            checkpointable=True,
            replayable=True,
            affects_family_signature=True,
            notes="Consumes gradient-derived state signals to guide higher-order symbolic candidate generation and reranking.",
            metadata={
                "source_modules": (
                    "core/symbolic/symbolic_structure_search.py",
                    "core/symbolic/gradient_parser.py",
                    "core/symbolic/gradient_correction.py",
                ),
                "default_enabled": True,
            },
        ),
        SymbolicSearchMechanismContract(
            mechanism_key="inner_optimizer",
            mechanism_kind="parameter_refinement",
            binding_level="bound",
            consume=("candidate_genome", "train_data_ref", "parameter_backend", "sample_weight_ref"),
            produce=("readout_weight", "readout_bias", "candidate_metrics", "inner_opt_info"),
            mutate=("search_state.parameter_fit", "search_trace.iterations"),
            checkpoint=("readout_weight", "readout_bias", "inner_opt_info"),
            replay=("candidate_genome", "parameter_backend", "readout_weight", "readout_bias"),
            checkpointable=True,
            replayable=True,
            affects_family_signature=True,
            notes="Fits or refines candidate readout parameters while symbolic structure search iterates.",
            metadata={
                "source_modules": (
                    "core/symbolic/symbolic_structure_search.py",
                    "core/symbolic/feature_space/branch_evaluator.py",
                ),
                "default_enabled": True,
            },
        ),
        SymbolicSearchMechanismContract(
            mechanism_key="expression_graph_cache",
            mechanism_kind="search_cache",
            binding_level="optional",
            consume=("expression_key", "feature_batch_ref", "structure_config.graph_cache_backend"),
            produce=("cached_expression_values", "cached_derivatives", "graph_cache_stats"),
            mutate=("graph_cache_store",),
            checkpoint=("graph_cache_store", "graph_cache_namespace", "graph_cache_db_path"),
            replay=("graph_cache_namespace", "graph_cache_db_path", "structure_config.graph_cache_backend"),
            checkpointable=True,
            replayable=True,
            affects_family_signature=False,
            notes="Acceleration-only cache for symbolic value and derivative evaluation; useful for replay fidelity, not family identity.",
            metadata={
                "source_modules": (
                    "core/symbolic/symbolic_structure_search.py",
                    "core/symbolic/expression_graph_cache.py",
                ),
                "default_enabled": True,
            },
        ),
        SymbolicSearchMechanismContract(
            mechanism_key="path_memory",
            mechanism_kind="search_memory",
            binding_level="optional",
            consume=("expr_key", "path_memory_store", "structure_config.path_memory_namespace"),
            produce=("path_prior", "path_memory_status", "tabu_signal"),
            mutate=("path_memory_store", "search_state.path_memory"),
            checkpoint=("path_memory_store", "path_memory_namespace", "expr_stats", "edge_stats"),
            replay=("path_memory_namespace", "path_memory_db_path", "expr_stats", "edge_stats"),
            checkpointable=True,
            replayable=True,
            affects_family_signature=False,
            notes="Cross-task search memory that influences ranking and tabu behavior through persistent symbolic path priors.",
            metadata={
                "source_modules": (
                    "core/symbolic/symbolic_structure_search.py",
                    "core/symbolic/path_memory.py",
                ),
                "default_enabled": True,
            },
        ),
    )


__all__ = [
    "SYMBOLIC_SEARCH_BINDING_LEVELS",
    "SYMBOLIC_SEARCH_MECHANISM_KINDS",
    "SymbolicSearchMechanismContract",
    "build_symbolic_search_mechanism_contracts",
    "coerce_symbolic_search_mechanism_contract",
    "serialize_symbolic_search_mechanism_contracts",
]
