from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from bias import BaseTrainingBias, FitContext, NoOpBias
from core.artifacts.symbolic_artifact import SymbolicSurrogateArtifact
from core.common.base_trainer import BaseSurrogateTrainer
from core.common.contracts import ProcessedDataset, SampleDataset
from core.common.trainer_shared import prepare_training_data
from core.execution import ExecutionResourceRequest
from core.symbolic.orthogonal_basis_search import (
    OrthogonalBasisSearchConfig,
    fit_orthogonal_basis_symbolic,
)
from core.symbolic.trainer_family import SymbolicStructureEngineSpec
from core.symbolic.trainer_state_io import (
    clone_symbolic_payload_cpu,
    load_symbolic_trainer_state_file,
    save_symbolic_trainer_state_file,
)
from core.symbolic.expression_graph_cache import ExpressionGraphCache
from core.trainers.symbolic_stagewise_trainer import SymbolicStagewiseTrainerConfig
from numericizer import BaseNumericizer, DefaultNumericizer, ModalityEncoder, TargetCodec
from pipeline import BasePipeline, IdentityPipeline
from training import (
    FitResult,
    TrainTask,
    TrainerState,
    TrainingInit,
    TrainingLineage,
    attach_signature_to_artifact,
    build_task_signature,
    coerce_trainer_capabilities,
    coerce_training_signature,
    require_training_setup,
)


@dataclass(frozen=True)
class SymbolicOrthogonalTrainerConfig:
    artifact_id: str = "symbolic_orthogonal_surrogate_v1"
    candidate_limit: int = 96
    seed_candidate_count: int = 18
    group_count: int = 12
    min_basis_count: int = 3
    max_basis_count: int = 6
    max_pair_abs_corr: float = 0.35
    max_feature_reuse: int = 2
    max_semantic_repeats: int = 1
    max_piecewise_semantic_repeats: int = 2
    target_score_weight: float = 1.0
    diversity_corr_weight: float = 0.80
    feature_overlap_penalty: float = 0.20
    complexity_penalty: float = 0.03
    new_feature_bonus: float = 0.05
    family_diversity_bonus: float = 0.03
    semantic_family_bonus: float = 0.05
    residual_corr_weight: float = 0.55
    residual_gain_weight: float = 0.85
    semantic_dup_penalty: float = 0.30
    piecewise_gate_bonus: float = 0.14
    native_structure_group_bonus: float = 0.0
    native_structure_representative_bonus: float = 0.0
    native_structure_screen_bonus: float = 0.0
    native_trunk_boundary_protocol: str = "OutermostPeelingBoundaryLock"
    native_trunk_channel_mode: str = "outermost_peeling"
    native_trunk_candidate_screen_reserve: int = 2
    require_native_trunk_candidate_in_group: bool = True
    min_native_trunk_basis_terms: int = 1
    native_trunk_residual_gain_floor: float = 0.05
    native_trunk_interval_gain_floor: float = 0.005
    screen_target_corr_weight: float = 1.0
    screen_residual_gain_weight: float = 0.65
    screen_semantic_novelty_weight: float = 0.20
    screen_consensus_prior_weight: float = 0.40
    screen_complexity_penalty: float = 0.08
    gate_candidate_screen_reserve: int = 0
    require_gate_candidate_in_group: bool = False
    min_gate_basis_terms: int = 0
    require_periodic_candidate_in_group: bool = False
    min_periodic_basis_terms: int = 0
    mechanistic_feature_groups: Sequence[Sequence[str]] = tuple()
    mechanistic_screen_bonus: float = 0.0
    mechanistic_group_bonus: float = 0.0
    l2_grid: Sequence[float] = (1e-6, 1e-4, 1e-2, 1e-1)
    rolling_folds: int = 3
    rolling_val_ratio: float = 0.18
    min_train_ratio: float = 0.40
    interval_alpha: float = 0.20
    coverage_error_threshold: float = 0.08
    outer_search_beam_width: int = 12
    outer_search_branching_factor: int = 3
    outer_search_max_expansions: int = 96
    selection_mode: str = "interval_first"
    random_seed: int = 42
    greedy_choice_topk: int = 1
    random_group_trials: int = 0
    outer_search_unit: str = "mechanism_object"
    representative_selection_rule: str = "balanced"
    lock_seed_basis: bool = False
    enable_piecewise_basis: bool = True
    gate_feature_names: Sequence[str] = tuple()
    periodic_feature_names: Sequence[str] = tuple()
    gate_quantiles: Sequence[float] = (0.35, 0.50, 0.65)
    gate_families: Sequence[str] = ("gate_step", "piecewise_hinge", "piecewise")
    gate_slope: float = 8.0
    piecewise_left_mode: str = "identity"
    piecewise_right_mode: str = "relu"
    assembler_max_added_terms: int = 4
    assembler_topk_features: int = 4
    assembler_max_pair_terms: int = 8
    assembler_max_candidates_per_iter: int = 96
    assembler_candidate_keep_top: int = 6
    assembler_max_expr_depth: int = 6
    assembler_ridge_l2: float = 1e-4
    assembler_path_memory_enabled: bool = False
    assembler_graph_cache_enabled: bool = False
    assembler_hinge_quantiles: Sequence[float] = (0.25, 0.50, 0.75)
    assembler_basis_binding_mode: str = "defining"
    assembler_escape_policy: str = "forbid"
    assembler_escape_feature_names: Sequence[str] = tuple()
    equivalence_expression_protocol: str = "EquivalenceExpressionHandlingProtocol"
    equivalence_expression_mode: str = "family+phase_equivalent+semantic"
    equivalence_class_scope: str = "candidate_screen+consensus+truth_recovery"
    chart_canonicalization_protocol: str = "ChartCanonicalizationPriority"
    chart_canonicalization_mode: str = "canonical_identity_with_stability_guard"
    chart_orthodoxy_scoring_protocol: str = "ChartOrthodoxyScoring"
    chart_orthodoxy_scoring_mode: str = "safe_wrapper_penalty+pole_safety+cross_interval_stability"
    support_expansion_protection_protocol: str = "SupportExpansionProtection"
    support_expansion_protection_mode: str = "full_support_native_template+seat_guard"
    support_expansion_candidate_screen_reserve: int = 1
    require_support_expansion_candidate_in_group: bool = True
    min_support_expansion_basis_terms: int = 1
    canonical_trunk_lane_protocol: str = "CanonicalTrunkLane"
    canonical_trunk_lane_mode: str = "support_pool_exposure+seat_guard"
    canonical_trunk_candidate_screen_reserve: int = 1
    require_canonical_trunk_candidate_in_group: bool = True
    min_canonical_trunk_basis_terms: int = 1
    same_source_surrogate_lane_protocol: str = "SameSourceSurrogateLane"
    same_source_surrogate_lane_mode: str = "support_pool_open_lane"
    rational_template_pinning_protocol: str = "RationalTemplatePinning"
    rational_template_pinning_mode: str = "mechanistic_pair_canonical_ratio_injection"
    global_first_preemption_protocol: str = "GlobalFirstPreemption"
    global_first_preemption_mode: str = "plain_support_parent_first"
    inner_chart_flip_compensation_protocol: str = "InnerChartFlipCompensation"
    inner_chart_flip_compensation_mode: str = "same_source_reciprocal_competition"
    realization_prior_injection_protocol: str = "RealizationPriorInjection"
    realization_prior_injection_mode: str = "object_member_evidence"
    mandatory_realization_closure_protocol: str = "MandatoryRealizationClosure"
    mandatory_realization_closure_mode: str = "explicit_evidence_competition"
    same_source_over_realization_protocol: str = "SameSourceOverRealizationCollapse"
    same_source_over_realization_mode: str = "inner_basis_object_budget"
    same_source_realization_budget: int = 1
    periodic_realization_competition_protocol: str = "PeriodicRealizationCompetition"
    periodic_realization_competition_mode: str = "sin_cos_basis_competition"
    interference_feature_protocol: str = "InterferenceFeatureHandlingProtocol"
    interference_feature_mode: str = "feature_overlap+semantic_dedup+mechanistic_bias"
    regime_penetration_protocol: str = "RegimePenetrationScore"
    regime_penetration_mode: str = "feature_quantile_penetration"
    regime_penetration_gain_floor: float = 0.01
    heterogeneous_exposure_protocol: str = "HeterogeneousExposureLane"
    heterogeneous_exposure_mode: str = "screen_reserve+seed_lane"
    heterogeneous_exposure_candidate_screen_reserve: int = 1
    heterogeneous_exposure_min_score: float = 0.20
    native_proxy_check_protocol: str = "NativeProxyCheck"
    native_proxy_check_mode: str = "proxy_group_native_election"
    proxy_trunk_disqualification_protocol: str = "ProxyTrunkDisqualification"
    proxy_trunk_disqualification_mode: str = "native_identity_only_when_available"
    parasitic_rejection_protocol: str = "ParasiticRejectionCriteria"
    parasitic_rejection_mode: str = "parent_trunk_required_for_branch_entry"
    cross_explanatory_rejection_mode: str = "off"
    trivial_nonlinearity_penalty_mode: str = "heuristic_semantic_overlap"
    environment_invariance_audit_mode: str = "off"
    periodic_equivalence_protocol: str = "PeriodicEquivalenceDisambiguationMechanism"
    periodic_equivalence_disambiguation_mode: str = "off"
    phase_spectrum_audit_mode: str = "off"
    periodic_family_prior_mode: str = "off"
    periodic_family_prior_weight: float = 0.30
    periodic_candidate_screen_reserve: int = 0
    regional_correction_protocol: str = "RegionalCorrectionBasisProtocol"
    residual_regime_identification_mode: str = "off"
    regional_correction_basis_mode: str = "off"
    regional_correction_promotion_mode: str = "off"
    regional_correction_feature_scope: str = "gate_only"
    regional_correction_topk: int = 0
    regional_correction_min_r2_gain: float = 0.0
    regional_correction_search_mode: str = "reopened_local_object_search"
    regional_local_search_beam_width: int = 6
    regional_local_search_branching_factor: int = 2
    regional_local_search_max_expansions: int = 24
    proxy_group_policy: str = "hint_if_available"
    source_overlap_penalty_mode: str = "feature_overlap_penalty"
    search_graph_cache_enabled: bool = True
    search_graph_cache_backend: str = "memory"
    search_graph_cache_db_path: str = ""
    search_graph_cache_namespace: str = "symbolic_orthogonal"
    search_graph_cache_persist_values: bool = False
    search_graph_cache_max_value_entries: int = 20000
    search_graph_cache_max_derivative_entries: int = 50000
    ood_z_threshold: float = 4.0
    epsilon: float = 1e-6

    def search_config(self) -> OrthogonalBasisSearchConfig:
        return OrthogonalBasisSearchConfig(
            candidate_limit=int(self.candidate_limit),
            seed_candidate_count=int(self.seed_candidate_count),
            group_count=int(self.group_count),
            min_basis_count=int(self.min_basis_count),
            max_basis_count=int(self.max_basis_count),
            max_pair_abs_corr=float(self.max_pair_abs_corr),
            max_feature_reuse=int(self.max_feature_reuse),
            max_semantic_repeats=int(self.max_semantic_repeats),
            max_piecewise_semantic_repeats=int(self.max_piecewise_semantic_repeats),
            target_score_weight=float(self.target_score_weight),
            diversity_corr_weight=float(self.diversity_corr_weight),
            feature_overlap_penalty=float(self.feature_overlap_penalty),
            complexity_penalty=float(self.complexity_penalty),
            new_feature_bonus=float(self.new_feature_bonus),
            family_diversity_bonus=float(self.family_diversity_bonus),
            semantic_family_bonus=float(self.semantic_family_bonus),
            residual_corr_weight=float(self.residual_corr_weight),
            residual_gain_weight=float(self.residual_gain_weight),
            semantic_dup_penalty=float(self.semantic_dup_penalty),
            piecewise_gate_bonus=float(self.piecewise_gate_bonus),
            native_structure_group_bonus=float(self.native_structure_group_bonus),
            native_structure_representative_bonus=float(self.native_structure_representative_bonus),
            native_structure_screen_bonus=float(self.native_structure_screen_bonus),
            native_trunk_boundary_protocol=str(self.native_trunk_boundary_protocol),
            native_trunk_channel_mode=str(self.native_trunk_channel_mode),
            native_trunk_candidate_screen_reserve=int(self.native_trunk_candidate_screen_reserve),
            require_native_trunk_candidate_in_group=bool(self.require_native_trunk_candidate_in_group),
            min_native_trunk_basis_terms=int(self.min_native_trunk_basis_terms),
            native_trunk_residual_gain_floor=float(self.native_trunk_residual_gain_floor),
            native_trunk_interval_gain_floor=float(self.native_trunk_interval_gain_floor),
            screen_target_corr_weight=float(self.screen_target_corr_weight),
            screen_residual_gain_weight=float(self.screen_residual_gain_weight),
            screen_semantic_novelty_weight=float(self.screen_semantic_novelty_weight),
            screen_consensus_prior_weight=float(self.screen_consensus_prior_weight),
            screen_complexity_penalty=float(self.screen_complexity_penalty),
            gate_candidate_screen_reserve=int(self.gate_candidate_screen_reserve),
            require_gate_candidate_in_group=bool(self.require_gate_candidate_in_group),
            min_gate_basis_terms=int(self.min_gate_basis_terms),
            require_periodic_candidate_in_group=bool(self.require_periodic_candidate_in_group),
            min_periodic_basis_terms=int(self.min_periodic_basis_terms),
            mechanistic_feature_groups=tuple(
                tuple(str(name) for name in tuple(group))
                for group in tuple(self.mechanistic_feature_groups)
            ),
            mechanistic_screen_bonus=float(self.mechanistic_screen_bonus),
            mechanistic_group_bonus=float(self.mechanistic_group_bonus),
            l2_grid=tuple(float(value) for value in self.l2_grid),
            rolling_folds=int(self.rolling_folds),
            rolling_val_ratio=float(self.rolling_val_ratio),
            min_train_ratio=float(self.min_train_ratio),
            interval_alpha=float(self.interval_alpha),
            coverage_error_threshold=float(self.coverage_error_threshold),
            outer_search_beam_width=int(self.outer_search_beam_width),
            outer_search_branching_factor=int(self.outer_search_branching_factor),
            outer_search_max_expansions=int(self.outer_search_max_expansions),
            selection_mode=str(self.selection_mode),
            random_seed=int(self.random_seed),
            greedy_choice_topk=int(self.greedy_choice_topk),
            random_group_trials=int(self.random_group_trials),
            outer_search_unit=str(self.outer_search_unit),
            representative_selection_rule=str(self.representative_selection_rule),
            lock_seed_basis=bool(self.lock_seed_basis),
            enable_piecewise_basis=bool(self.enable_piecewise_basis),
            gate_feature_names=tuple(str(value) for value in self.gate_feature_names),
            periodic_feature_names=tuple(str(value) for value in self.periodic_feature_names),
            gate_quantiles=tuple(float(value) for value in self.gate_quantiles),
            gate_families=tuple(str(value) for value in self.gate_families),
            gate_slope=float(self.gate_slope),
            piecewise_left_mode=str(self.piecewise_left_mode),
            piecewise_right_mode=str(self.piecewise_right_mode),
            assembler_max_added_terms=int(self.assembler_max_added_terms),
            assembler_topk_features=int(self.assembler_topk_features),
            assembler_max_pair_terms=int(self.assembler_max_pair_terms),
            assembler_max_candidates_per_iter=int(self.assembler_max_candidates_per_iter),
            assembler_candidate_keep_top=int(self.assembler_candidate_keep_top),
            assembler_max_expr_depth=int(self.assembler_max_expr_depth),
            assembler_ridge_l2=float(self.assembler_ridge_l2),
            assembler_path_memory_enabled=bool(self.assembler_path_memory_enabled),
            assembler_graph_cache_enabled=bool(self.assembler_graph_cache_enabled),
            assembler_hinge_quantiles=tuple(float(value) for value in self.assembler_hinge_quantiles),
            assembler_basis_binding_mode=str(self.assembler_basis_binding_mode),
            assembler_escape_policy=str(self.assembler_escape_policy),
            assembler_escape_feature_names=tuple(str(value) for value in self.assembler_escape_feature_names),
            equivalence_expression_protocol=str(self.equivalence_expression_protocol),
            equivalence_expression_mode=str(self.equivalence_expression_mode),
            equivalence_class_scope=str(self.equivalence_class_scope),
            chart_canonicalization_protocol=str(self.chart_canonicalization_protocol),
            chart_canonicalization_mode=str(self.chart_canonicalization_mode),
            chart_orthodoxy_scoring_protocol=str(self.chart_orthodoxy_scoring_protocol),
            chart_orthodoxy_scoring_mode=str(self.chart_orthodoxy_scoring_mode),
            support_expansion_protection_protocol=str(self.support_expansion_protection_protocol),
            support_expansion_protection_mode=str(self.support_expansion_protection_mode),
            support_expansion_candidate_screen_reserve=int(self.support_expansion_candidate_screen_reserve),
            require_support_expansion_candidate_in_group=bool(self.require_support_expansion_candidate_in_group),
            min_support_expansion_basis_terms=int(self.min_support_expansion_basis_terms),
            canonical_trunk_lane_protocol=str(self.canonical_trunk_lane_protocol),
            canonical_trunk_lane_mode=str(self.canonical_trunk_lane_mode),
            canonical_trunk_candidate_screen_reserve=int(self.canonical_trunk_candidate_screen_reserve),
            require_canonical_trunk_candidate_in_group=bool(self.require_canonical_trunk_candidate_in_group),
            min_canonical_trunk_basis_terms=int(self.min_canonical_trunk_basis_terms),
            same_source_surrogate_lane_protocol=str(self.same_source_surrogate_lane_protocol),
            same_source_surrogate_lane_mode=str(self.same_source_surrogate_lane_mode),
            rational_template_pinning_protocol=str(self.rational_template_pinning_protocol),
            rational_template_pinning_mode=str(self.rational_template_pinning_mode),
            global_first_preemption_protocol=str(self.global_first_preemption_protocol),
            global_first_preemption_mode=str(self.global_first_preemption_mode),
            inner_chart_flip_compensation_protocol=str(self.inner_chart_flip_compensation_protocol),
            inner_chart_flip_compensation_mode=str(self.inner_chart_flip_compensation_mode),
            realization_prior_injection_protocol=str(self.realization_prior_injection_protocol),
            realization_prior_injection_mode=str(self.realization_prior_injection_mode),
            mandatory_realization_closure_protocol=str(self.mandatory_realization_closure_protocol),
            mandatory_realization_closure_mode=str(self.mandatory_realization_closure_mode),
            same_source_over_realization_protocol=str(self.same_source_over_realization_protocol),
            same_source_over_realization_mode=str(self.same_source_over_realization_mode),
            same_source_realization_budget=int(self.same_source_realization_budget),
            periodic_realization_competition_protocol=str(self.periodic_realization_competition_protocol),
            periodic_realization_competition_mode=str(self.periodic_realization_competition_mode),
            interference_feature_protocol=str(self.interference_feature_protocol),
            interference_feature_mode=str(self.interference_feature_mode),
            regime_penetration_protocol=str(self.regime_penetration_protocol),
            regime_penetration_mode=str(self.regime_penetration_mode),
            regime_penetration_gain_floor=float(self.regime_penetration_gain_floor),
            heterogeneous_exposure_protocol=str(self.heterogeneous_exposure_protocol),
            heterogeneous_exposure_mode=str(self.heterogeneous_exposure_mode),
            heterogeneous_exposure_candidate_screen_reserve=int(self.heterogeneous_exposure_candidate_screen_reserve),
            heterogeneous_exposure_min_score=float(self.heterogeneous_exposure_min_score),
            native_proxy_check_protocol=str(self.native_proxy_check_protocol),
            native_proxy_check_mode=str(self.native_proxy_check_mode),
            proxy_trunk_disqualification_protocol=str(self.proxy_trunk_disqualification_protocol),
            proxy_trunk_disqualification_mode=str(self.proxy_trunk_disqualification_mode),
            parasitic_rejection_protocol=str(self.parasitic_rejection_protocol),
            parasitic_rejection_mode=str(self.parasitic_rejection_mode),
            cross_explanatory_rejection_mode=str(self.cross_explanatory_rejection_mode),
            trivial_nonlinearity_penalty_mode=str(self.trivial_nonlinearity_penalty_mode),
            environment_invariance_audit_mode=str(self.environment_invariance_audit_mode),
            periodic_equivalence_protocol=str(self.periodic_equivalence_protocol),
            periodic_equivalence_disambiguation_mode=str(self.periodic_equivalence_disambiguation_mode),
            phase_spectrum_audit_mode=str(self.phase_spectrum_audit_mode),
            periodic_family_prior_mode=str(self.periodic_family_prior_mode),
            periodic_family_prior_weight=float(self.periodic_family_prior_weight),
            periodic_candidate_screen_reserve=int(self.periodic_candidate_screen_reserve),
            regional_correction_protocol=str(self.regional_correction_protocol),
            residual_regime_identification_mode=str(self.residual_regime_identification_mode),
            regional_correction_basis_mode=str(self.regional_correction_basis_mode),
            regional_correction_promotion_mode=str(self.regional_correction_promotion_mode),
            regional_correction_feature_scope=str(self.regional_correction_feature_scope),
            regional_correction_topk=int(self.regional_correction_topk),
            regional_correction_min_r2_gain=float(self.regional_correction_min_r2_gain),
            regional_correction_search_mode=str(self.regional_correction_search_mode),
            regional_local_search_beam_width=int(self.regional_local_search_beam_width),
            regional_local_search_branching_factor=int(self.regional_local_search_branching_factor),
            regional_local_search_max_expansions=int(self.regional_local_search_max_expansions),
            proxy_group_policy=str(self.proxy_group_policy),
            source_overlap_penalty_mode=str(self.source_overlap_penalty_mode),
        )


class SymbolicOrthogonalSurrogateTrainer(BaseSurrogateTrainer):
    name = "symbolic_orthogonal"

    def __init__(
        self,
        config: SymbolicOrthogonalTrainerConfig | None = None,
        *,
        pipeline: BasePipeline | None = None,
        biases: Sequence[BaseTrainingBias] | None = None,
        numericizer: BaseNumericizer | None = None,
        modality_encoders: Mapping[str, ModalityEncoder] | None = None,
        target_codecs: Mapping[str, TargetCodec] | None = None,
        target_codec: str | None = None,
        categorical_unknown: str = "error",
    ) -> None:
        self.config = config or SymbolicOrthogonalTrainerConfig()
        self.pipeline = pipeline or IdentityPipeline()
        self.biases = list(biases) if biases else [NoOpBias()]

        if numericizer is not None and (
            modality_encoders is not None or target_codecs is not None or target_codec is not None
        ):
            raise ValueError("Provide either numericizer or encoder/codec options, not both")

        if numericizer is not None:
            self.numericizer = numericizer
        else:
            self.numericizer = DefaultNumericizer(
                modality_encoders=modality_encoders,
                target_codecs=target_codecs,
                target_codec=target_codec,
                categorical_unknown=categorical_unknown,
            )

    @staticmethod
    def _clone_payload_cpu(value: Any) -> Any:
        return clone_symbolic_payload_cpu(value)

    @staticmethod
    def _clone_state_cpu(state: Mapping[str, Any]) -> dict[str, Any]:
        return dict(SymbolicOrthogonalSurrogateTrainer._clone_payload_cpu(dict(state)))

    @staticmethod
    def _copy_genome(genome: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
        return tuple(dict(term) for term in tuple(genome))

    @classmethod
    def _load_state_payload(cls, path: str | Path) -> dict[str, Any]:
        return load_symbolic_trainer_state_file(path)

    @classmethod
    def save_trainer_state(cls, path: str | Path, state: TrainerState) -> str:
        return save_symbolic_trainer_state_file(
            path,
            trainer_name=str(getattr(state, "trainer_name", cls.name)),
            payload=dict(getattr(state, "payload", {})),
            metadata=dict(getattr(state, "metadata", {})),
        )

    @classmethod
    def load_trainer_state(cls, path: str | Path) -> TrainerState:
        resume_path = Path(path).resolve()
        payload = cls._load_state_payload(resume_path)
        signature = coerce_training_signature(payload.get("training_signature"))
        return TrainerState(
            trainer_name=str(payload.get("trainer_name", cls.name)),
            payload=dict(payload),
            schema_signature=signature.schema_signature,
            feature_signature=signature.feature_signature,
            target_signature=signature.target_signature,
            objective_signature=signature.objective_signature,
            pipeline_signature=signature.pipeline_signature,
            numericizer_signature=signature.numericizer_signature,
            regime_signature=signature.regime_signature,
            symbolic_family_signature=signature.symbolic_family_signature,
            metadata={
                "resume_source": str(resume_path),
                "search_completed": bool(payload.get("search_completed", True)),
                "training_signature": signature.as_dict(),
            },
        )

    @staticmethod
    def _seed_payload_from_artifact(artifact: Any) -> dict[str, Any] | None:
        if not isinstance(artifact, SymbolicSurrogateArtifact):
            return None
        metadata = dict(getattr(artifact, "metadata", {}) or {})
        search = dict(metadata.get("search", {}) or {})
        raw_outer_basis = metadata.get("orthogonal_outer_basis_genome")
        if not (
            isinstance(raw_outer_basis, Sequence)
            and not isinstance(raw_outer_basis, (str, bytes, bytearray))
        ):
            raw_outer_basis = dict(metadata.get("symbolic", {}) or {}).get("orthogonal_outer_basis_genome")
        seed_genome = (
            tuple(
                dict(term)
                for term in tuple(raw_outer_basis)
                if isinstance(term, Mapping) and isinstance(dict(term).get("expr"), Mapping)
            )
            if isinstance(raw_outer_basis, Sequence) and not isinstance(raw_outer_basis, (str, bytes, bytearray))
            else tuple()
        )
        return {
            "schema_version": 1,
            "trainer_name": "symbolic_orthogonal",
            "search_completed": True,
            "genome": (
                SymbolicOrthogonalSurrogateTrainer._copy_genome(seed_genome)
                if seed_genome
                else SymbolicOrthogonalSurrogateTrainer._copy_genome(artifact.genome)
            ),
            "assembled_genome": SymbolicOrthogonalSurrogateTrainer._copy_genome(artifact.genome),
            "parameter_values": dict(getattr(artifact, "parameter_values", {}) or {}),
            "readout_weight": np.asarray(getattr(artifact, "readout_weight"), dtype=float),
            "readout_bias": np.asarray(getattr(artifact, "readout_bias"), dtype=float),
            "residual_std": np.asarray(getattr(artifact, "residual_std"), dtype=float),
            "feature_names": tuple(str(value) for value in getattr(artifact, "feature_names", ()) or ()),
            "target_names": tuple(str(value) for value in getattr(artifact, "target_names", ()) or ()),
            "search_summary": dict(search),
            "consensus_prior_rows": tuple(
                dict(row)
                for row in tuple(
                    metadata.get("consensus_prior_rows")
                    or dict(metadata.get("symbolic", {}) or {}).get("consensus_prior_rows")
                    or ()
                )
                if isinstance(row, Mapping)
            ),
            "seed_protocol": "outer_basis_genome" if seed_genome else "artifact_genome",
            "training_signature": metadata.get("training_signature"),
        }

    def capabilities(self) -> dict[str, object]:
        return {
            "supports_fresh": True,
            "supports_resume": True,
            "supports_warm_start": True,
            "supports_incremental": True,
            "supports_recalibration": False,
            "name": self.name,
            "model_family": "symbolic_orthogonal",
            "backend": "numpy",
            "nonlinear": True,
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "sample_weight": False,
                "target_codec": True,
                "orthogonal_basis_discovery": True,
                "piecewise_gate_basis": True,
                "residual_complementarity_report": True,
                "semantic_dedup_report": True,
                "locked_core_refinement": True,
                "trainer_state": True,
                "graph_cache_persistent_backend": "memory|sqlite|lmdb",
            },
            "artifacts": {
                "type": "SymbolicSurrogateArtifact",
                "uncertainty": "residual_std",
                "ood_validity": True,
            },
            "runtime": {
                "resume_from_trainer_state": True,
                "resume_semantics": "seeded_basis_restart",
                "warm_start_semantics": "reuse_parent_outer_basis_as_seed",
                "incremental_semantics": "reuse_parent_outer_basis_as_seed",
            },
        }

    def execution_resource_request(self) -> ExecutionResourceRequest:
        return ExecutionResourceRequest(
            threads=1,
            backend="serial",
            label=str(self.name),
        )

    def _resolve_structure_engine(self) -> SymbolicStructureEngineSpec:
        family_spec = getattr(self, "symbolic_family_spec", None)
        structure_engine = getattr(family_spec, "structure_engine", None)
        if structure_engine is not None and hasattr(structure_engine, "as_dict"):
            return structure_engine
        return SymbolicStructureEngineSpec(
            structure_mode="orthogonal_basis_search",
            search_driver="orthogonal_basis",
            dynamic_pool_enabled=True,
            metadata={"supports_piecewise_basis": bool(tuple(self.config.gate_feature_names))},
        )

    @staticmethod
    def _merge_structure_engine_payload(
        base_structure_engine: Mapping[str, Any] | None,
        runtime_structure_engine: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        base_payload = dict(base_structure_engine or {})
        runtime_payload = dict(runtime_structure_engine or {})
        merged_metadata = dict(base_payload.get("metadata", {}) or {})
        merged_metadata.update(dict(runtime_payload.get("metadata", {}) or {}))
        merged = dict(base_payload)
        merged.update(runtime_payload)
        if merged_metadata:
            merged["metadata"] = merged_metadata
        return merged

    def _build_graph_cache(self) -> ExpressionGraphCache | None:
        if not bool(self.config.search_graph_cache_enabled):
            return None
        return ExpressionGraphCache(
            enabled=True,
            max_value_entries=int(self.config.search_graph_cache_max_value_entries),
            max_derivative_entries=int(self.config.search_graph_cache_max_derivative_entries),
            backend=str(self.config.search_graph_cache_backend),
            db_path=str(self.config.search_graph_cache_db_path),
            namespace=str(self.config.search_graph_cache_namespace),
            persist_values=bool(self.config.search_graph_cache_persist_values),
        )

    def _fit_internal(
        self,
        data: ProcessedDataset | SampleDataset,
        *,
        init: TrainingInit | None = None,
        training_signature: Mapping[str, Any] | None = None,
    ) -> tuple[SymbolicSurrogateArtifact, TrainerState | None]:
        init_eff = init or TrainingInit()
        mode = str(init_eff.mode).strip().lower() or "fresh"
        training_signature_payload = {} if training_signature is None else dict(training_signature)
        training_signature_meta = dict(training_signature_payload.get("metadata", {}) or {})
        parent_payload: dict[str, Any] | None = None
        parent_seed_source: str | None = None
        parent_seed_kind: str | None = None

        if init_eff.parent_state is not None:
            parent_payload = self._clone_state_cpu(getattr(init_eff.parent_state, "payload", {}))
            parent_seed_source = str(
                dict(getattr(init_eff.parent_state, "metadata", {}) or {}).get("resume_source")
                or getattr(init_eff.parent_state, "trainer_name", type(init_eff.parent_state).__name__)
            )
            parent_seed_kind = "trainer_state"
        elif init_eff.parent_artifact is not None:
            parent_payload = self._seed_payload_from_artifact(init_eff.parent_artifact)
            parent_seed_source = str(
                getattr(init_eff.parent_artifact, "artifact_id", type(init_eff.parent_artifact).__name__)
            )
            parent_seed_kind = "artifact"

        prepared = prepare_training_data(
            data=data,
            numericizer=self.numericizer,
            pipeline=self.pipeline,
            biases=self.biases,
            fit_context_cls=FitContext,
        )
        normalized = prepared.normalized
        context = prepared.context
        X_train = prepared.X
        y_train = prepared.Y
        n, d, m = int(prepared.n), int(prepared.d), int(prepared.m)
        feature_names = prepared.feature_names
        target_names = prepared.target_names

        if context.sample_weight is not None:
            warning_msg = (
                "symbolic_orthogonal does not implement weighted basis discovery yet; "
                "sample_weight from the bias layer will be ignored for this fit"
            )
            warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)
            metadata = getattr(context, "metadata", None)
            if isinstance(metadata, dict):
                existing = metadata.get("warnings")
                if isinstance(existing, list):
                    existing.append(str(warning_msg))
                else:
                    metadata["warnings"] = [str(warning_msg)]
                metadata["sample_weight_ignored"] = True
            context.sample_weight = None

        seed_genome_override: tuple[dict[str, Any], ...] | None = None
        consensus_prior_rows: tuple[dict[str, Any], ...] | None = None
        if isinstance(parent_payload, Mapping):
            raw_seed = parent_payload.get("genome")
            if isinstance(raw_seed, Sequence):
                seed_genome_override = tuple(dict(term) for term in tuple(raw_seed) if isinstance(term, Mapping))
                if not seed_genome_override:
                    seed_genome_override = None
            raw_consensus_prior = parent_payload.get("consensus_prior_rows")
            if isinstance(raw_consensus_prior, Sequence) and not isinstance(raw_consensus_prior, (str, bytes, bytearray)):
                consensus_prior_rows = tuple(
                    dict(row) for row in tuple(raw_consensus_prior) if isinstance(row, Mapping)
                )
                if not consensus_prior_rows:
                    consensus_prior_rows = None

        family_payload = dict(getattr(self, "symbolic_family_metadata", {}))
        if not family_payload:
            family_spec = getattr(self, "symbolic_family_spec", None)
            if family_spec is not None and hasattr(family_spec, "description_dict"):
                family_payload = dict(family_spec.description_dict())

        graph_cache = self._build_graph_cache()
        try:
            orthogonal_result = fit_orthogonal_basis_symbolic(
                X=np.asarray(X_train, dtype=float),
                y=np.asarray(y_train, dtype=float),
                feature_names=tuple(feature_names),
                cfg=self.config.search_config(),
                graph_cache=graph_cache,
                seed_genome=seed_genome_override,
                consensus_prior_rows=consensus_prior_rows,
                symbolic_family_payload=family_payload or None,
                data_metadata=dict(normalized.metadata or {}),
            )
        finally:
            if graph_cache is not None:
                graph_cache.close()

        generic_metadata = {
            "trainer": "SymbolicOrthogonalSurrogateTrainer",
            "n_train": int(n),
            "feature_dim": int(d),
            "target_dim": int(m),
            "pipeline": str(getattr(self.pipeline, "name", "identity")),
            "biases": [str(getattr(bias, "name", type(bias).__name__)) for bias in self.biases],
            "fit_context": dict(context.metadata),
            "data_protocol": str((normalized.metadata or {}).get("input_protocol", "processed_dataset")),
            "data_metadata": dict(normalized.metadata or {}),
            "numericizer": str(getattr(self.numericizer, "name", type(self.numericizer).__name__)),
            "training_init": {
                "mode": str(mode),
                "parent_source": parent_seed_source,
                "parent_kind": parent_seed_kind,
            },
            "resume": {
                "enabled": bool(mode in {"resume", "warm_start", "incremental"} and seed_genome_override is not None),
                "mode": str(mode),
                "seed_source": parent_seed_source,
                "seed_kind": parent_seed_kind,
                "seed_terms": 0 if seed_genome_override is None else int(len(seed_genome_override)),
            },
            "consensus_prior_rows": [] if consensus_prior_rows is None else [dict(row) for row in tuple(consensus_prior_rows)],
            "search_driver": "orthogonal_basis_set_search",
            "structure_mode": "orthogonal_basis_search",
            "train_metrics": dict(orthogonal_result.train_metrics),
        }
        fit_context_metadata = dict(context.metadata or {})
        lane_context = dict(fit_context_metadata.get("heterogeneous_multi_lane_context", {}) or {})
        if not lane_context:
            lane_context = {
                "protocol": fit_context_metadata.get("heterogeneous_multi_lane_protocol"),
                "lane_id": fit_context_metadata.get("lane_id"),
                "lane_family": fit_context_metadata.get("lane_family"),
                "lane_label": fit_context_metadata.get("lane_label"),
                "lane_description": fit_context_metadata.get("lane_description"),
                "lane_weight": fit_context_metadata.get("lane_weight"),
                "screening_protocol": fit_context_metadata.get("screening_protocol"),
                "challenger_objective_protocol": fit_context_metadata.get(
                    "challenger_objective_protocol"
                ),
                "pool_expansion_bias_protocol": fit_context_metadata.get(
                    "pool_expansion_bias_protocol"
                ),
                "lane_spec": fit_context_metadata.get("lane_spec"),
            }
            lane_context = {
                str(key): value
                for key, value in lane_context.items()
                if value is not None and (not isinstance(value, str) or str(value).strip())
            }
        if lane_context:
            generic_metadata["heterogeneous_multi_lane_context"] = dict(lane_context)
            generic_metadata["lane_id"] = lane_context.get("lane_id")
            generic_metadata["lane_family"] = lane_context.get("lane_family")
            generic_metadata["challenger_objective_protocol"] = lane_context.get(
                "challenger_objective_protocol"
            )
            generic_metadata["pool_expansion_bias_protocol"] = lane_context.get(
                "pool_expansion_bias_protocol"
            )
        generic_metadata.update(dict(orthogonal_result.metadata))
        symbolic_block = dict(generic_metadata.get("symbolic", {}) or {})
        symbolic_block["structure_engine"] = self._merge_structure_engine_payload(
            self._resolve_structure_engine().as_dict(),
            symbolic_block.get("structure_engine"),
        )
        symbolic_block["selected_basis"] = generic_metadata.get("selected_basis")
        symbolic_block["basis_semantics"] = generic_metadata.get("basis_semantics")
        symbolic_block["basis_overlap_report"] = generic_metadata.get("basis_overlap_report")
        symbolic_block["residual_complementarity_report"] = generic_metadata.get("residual_complementarity_report")
        symbolic_block["semantic_dedup_report"] = generic_metadata.get("semantic_dedup_report")
        symbolic_block["assembler_budget"] = generic_metadata.get("assembler_budget")
        symbolic_block["inner_symbolic_search"] = generic_metadata.get("inner_symbolic_search")
        symbolic_block["orthogonal_search_objective"] = generic_metadata.get("orthogonal_search_objective")
        symbolic_block["orthogonal_outer_basis_genome"] = generic_metadata.get("orthogonal_outer_basis_genome")
        symbolic_block["fold_report"] = generic_metadata.get("fold_report")
        symbolic_block["consensus_prior_rows"] = generic_metadata.get("consensus_prior_rows")
        symbolic_block["structure_head"] = generic_metadata.get("structure_head")
        symbolic_block["prediction_head"] = generic_metadata.get("prediction_head")
        symbolic_block["search_input_space"] = generic_metadata.get("search_input_space")
        symbolic_block["pool_expansion_unit"] = generic_metadata.get("pool_expansion_unit")
        symbolic_block["gradient_guidance_mode"] = generic_metadata.get("gradient_guidance_mode")
        symbolic_block["basis_binding_mode"] = generic_metadata.get("basis_binding_mode")
        symbolic_block["escape_policy"] = generic_metadata.get("escape_policy")
        symbolic_block["equivalence_expression_protocol"] = generic_metadata.get("equivalence_expression_protocol")
        symbolic_block["equivalence_expression_mode"] = generic_metadata.get("equivalence_expression_mode")
        symbolic_block["equivalence_class_scope"] = generic_metadata.get("equivalence_class_scope")
        symbolic_block["interference_feature_protocol"] = generic_metadata.get("interference_feature_protocol")
        symbolic_block["interference_feature_mode"] = generic_metadata.get("interference_feature_mode")
        symbolic_block["cross_explanatory_rejection_mode"] = generic_metadata.get("cross_explanatory_rejection_mode")
        symbolic_block["trivial_nonlinearity_penalty_mode"] = generic_metadata.get("trivial_nonlinearity_penalty_mode")
        symbolic_block["environment_invariance_audit_mode"] = generic_metadata.get("environment_invariance_audit_mode")
        symbolic_block["periodic_equivalence_protocol"] = generic_metadata.get("periodic_equivalence_protocol")
        symbolic_block["periodic_equivalence_disambiguation_mode"] = generic_metadata.get(
            "periodic_equivalence_disambiguation_mode"
        )
        symbolic_block["phase_spectrum_audit_mode"] = generic_metadata.get("phase_spectrum_audit_mode")
        symbolic_block["periodic_family_prior_mode"] = generic_metadata.get("periodic_family_prior_mode")
        symbolic_block["periodic_candidate_screen_reserve"] = generic_metadata.get(
            "periodic_candidate_screen_reserve"
        )
        symbolic_block["regional_correction_protocol"] = generic_metadata.get("regional_correction_protocol")
        symbolic_block["residual_regime_identification_mode"] = generic_metadata.get(
            "residual_regime_identification_mode"
        )
        symbolic_block["regional_correction_basis_mode"] = generic_metadata.get(
            "regional_correction_basis_mode"
        )
        symbolic_block["regional_correction_promotion_mode"] = generic_metadata.get(
            "regional_correction_promotion_mode"
        )
        symbolic_block["regional_correction_feature_scope"] = generic_metadata.get(
            "regional_correction_feature_scope"
        )
        symbolic_block["regional_correction_topk"] = generic_metadata.get("regional_correction_topk")
        symbolic_block["regional_correction_min_r2_gain"] = generic_metadata.get(
            "regional_correction_min_r2_gain"
        )
        symbolic_block["proxy_group_policy"] = generic_metadata.get("proxy_group_policy")
        symbolic_block["source_overlap_penalty_mode"] = generic_metadata.get("source_overlap_penalty_mode")
        symbolic_block["equivalence_expression_handling"] = generic_metadata.get("equivalence_expression_handling")
        symbolic_block["interference_feature_handling"] = generic_metadata.get("interference_feature_handling")
        symbolic_block["periodic_equivalence_disambiguation"] = generic_metadata.get(
            "periodic_equivalence_disambiguation"
        )
        symbolic_block["regional_correction_basis"] = generic_metadata.get("regional_correction_basis")
        symbolic_block["stage_head_protocols"] = generic_metadata.get("stage_head_protocols")
        symbolic_block["basis_context"] = generic_metadata.get("basis_context")
        symbolic_block["basis_object_gradient_pool"] = generic_metadata.get("basis_object_gradient_pool")
        symbolic_block["heterogeneous_multi_lane_context"] = generic_metadata.get(
            "heterogeneous_multi_lane_context"
        )
        generic_metadata["symbolic"] = symbolic_block
        if family_payload:
            generic_metadata["symbolic_family"] = dict(family_payload)

        artifact = SymbolicSurrogateArtifact(
            artifact_id=str(self.config.artifact_id),
            genome=tuple(orthogonal_result.genome),
            parameter_values={},
            readout_weight=np.asarray(orthogonal_result.readout_weight, dtype=float),
            readout_bias=np.asarray(orthogonal_result.readout_bias, dtype=float),
            x_mean=np.mean(X_train, axis=0),
            x_std=np.std(X_train, axis=0) + 1e-8,
            residual_std=np.asarray(orthogonal_result.residual_std, dtype=float),
            feature_names=feature_names,
            target_names=target_names,
            pipeline_name=str(getattr(self.pipeline, "name", "identity")),
            pipeline_state=dict(getattr(self.pipeline, "state_dict")()),
            ood_z_threshold=float(self.config.ood_z_threshold),
            epsilon=float(self.config.epsilon),
            metadata=generic_metadata,
        )
        if training_signature is None:
            return artifact, None

        signature_obj = coerce_training_signature(training_signature)
        state_seed_genome = generic_metadata.get("orthogonal_outer_basis_genome")
        if not (
            isinstance(state_seed_genome, Sequence)
            and not isinstance(state_seed_genome, (str, bytes, bytearray))
        ):
            state_seed_genome = orthogonal_result.genome
        trainer_state_payload = {
            "schema_version": 1,
            "trainer_name": str(self.name),
            "search_completed": True,
            "genome": self._copy_genome(state_seed_genome),
            "assembled_genome": self._copy_genome(orthogonal_result.genome),
            "parameter_values": {},
            "readout_weight": np.asarray(orthogonal_result.readout_weight, dtype=float),
            "readout_bias": np.asarray(orthogonal_result.readout_bias, dtype=float),
            "residual_std": np.asarray(orthogonal_result.residual_std, dtype=float),
            "feature_names": tuple(str(value) for value in feature_names),
            "target_names": tuple(str(value) for value in target_names),
            "train_metrics": dict(orthogonal_result.train_metrics),
            "selected_basis": generic_metadata.get("selected_basis"),
            "basis_semantics": generic_metadata.get("basis_semantics"),
            "basis_overlap_report": generic_metadata.get("basis_overlap_report"),
            "residual_complementarity_report": generic_metadata.get("residual_complementarity_report"),
            "semantic_dedup_report": generic_metadata.get("semantic_dedup_report"),
            "assembler_budget": generic_metadata.get("assembler_budget"),
            "fold_report": generic_metadata.get("fold_report"),
            "inner_symbolic_search": generic_metadata.get("inner_symbolic_search"),
            "orthogonal_search_objective": generic_metadata.get("orthogonal_search_objective"),
            "orthogonal_outer_basis_genome": generic_metadata.get("orthogonal_outer_basis_genome"),
            "consensus_prior_rows": generic_metadata.get("consensus_prior_rows"),
            "structure_head": generic_metadata.get("structure_head"),
            "prediction_head": generic_metadata.get("prediction_head"),
            "search_input_space": generic_metadata.get("search_input_space"),
            "pool_expansion_unit": generic_metadata.get("pool_expansion_unit"),
            "gradient_guidance_mode": generic_metadata.get("gradient_guidance_mode"),
            "basis_binding_mode": generic_metadata.get("basis_binding_mode"),
            "escape_policy": generic_metadata.get("escape_policy"),
            "equivalence_expression_protocol": generic_metadata.get("equivalence_expression_protocol"),
            "equivalence_expression_mode": generic_metadata.get("equivalence_expression_mode"),
            "equivalence_class_scope": generic_metadata.get("equivalence_class_scope"),
            "interference_feature_protocol": generic_metadata.get("interference_feature_protocol"),
            "interference_feature_mode": generic_metadata.get("interference_feature_mode"),
            "cross_explanatory_rejection_mode": generic_metadata.get("cross_explanatory_rejection_mode"),
            "trivial_nonlinearity_penalty_mode": generic_metadata.get("trivial_nonlinearity_penalty_mode"),
            "environment_invariance_audit_mode": generic_metadata.get("environment_invariance_audit_mode"),
            "periodic_equivalence_protocol": generic_metadata.get("periodic_equivalence_protocol"),
            "periodic_equivalence_disambiguation_mode": generic_metadata.get(
                "periodic_equivalence_disambiguation_mode"
            ),
            "phase_spectrum_audit_mode": generic_metadata.get("phase_spectrum_audit_mode"),
            "periodic_family_prior_mode": generic_metadata.get("periodic_family_prior_mode"),
            "periodic_candidate_screen_reserve": generic_metadata.get("periodic_candidate_screen_reserve"),
            "regional_correction_protocol": generic_metadata.get("regional_correction_protocol"),
            "residual_regime_identification_mode": generic_metadata.get(
                "residual_regime_identification_mode"
            ),
            "regional_correction_basis_mode": generic_metadata.get("regional_correction_basis_mode"),
            "regional_correction_promotion_mode": generic_metadata.get(
                "regional_correction_promotion_mode"
            ),
            "regional_correction_feature_scope": generic_metadata.get("regional_correction_feature_scope"),
            "regional_correction_topk": generic_metadata.get("regional_correction_topk"),
            "regional_correction_min_r2_gain": generic_metadata.get(
                "regional_correction_min_r2_gain"
            ),
            "proxy_group_policy": generic_metadata.get("proxy_group_policy"),
            "source_overlap_penalty_mode": generic_metadata.get("source_overlap_penalty_mode"),
            "equivalence_expression_handling": generic_metadata.get("equivalence_expression_handling"),
            "interference_feature_handling": generic_metadata.get("interference_feature_handling"),
            "periodic_equivalence_disambiguation": generic_metadata.get(
                "periodic_equivalence_disambiguation"
            ),
            "regional_correction_basis": generic_metadata.get("regional_correction_basis"),
            "stage_head_protocols": generic_metadata.get("stage_head_protocols"),
            "basis_context": generic_metadata.get("basis_context"),
            "basis_object_gradient_pool": generic_metadata.get("basis_object_gradient_pool"),
            "heterogeneous_multi_lane_context": generic_metadata.get(
                "heterogeneous_multi_lane_context"
            ),
            "lane_id": generic_metadata.get("lane_id"),
            "lane_family": generic_metadata.get("lane_family"),
            "challenger_objective_protocol": generic_metadata.get(
                "challenger_objective_protocol"
            ),
            "pool_expansion_bias_protocol": generic_metadata.get(
                "pool_expansion_bias_protocol"
            ),
            "seed_protocol": "outer_basis_genome",
            "search_summary": generic_metadata.get("search"),
            "training_signature": signature_obj.as_dict(),
        }
        trainer_state = TrainerState(
            trainer_name=str(self.name),
            payload=trainer_state_payload,
            schema_signature=signature_obj.schema_signature,
            feature_signature=signature_obj.feature_signature,
            target_signature=signature_obj.target_signature,
            objective_signature=signature_obj.objective_signature,
            pipeline_signature=signature_obj.pipeline_signature,
            numericizer_signature=signature_obj.numericizer_signature,
            regime_signature=signature_obj.regime_signature,
            symbolic_family_signature=signature_obj.symbolic_family_signature,
            metadata={
                "resume_source": parent_seed_source if mode == "resume" else None,
                "seed_source": parent_seed_source,
                "seed_kind": parent_seed_kind,
                "search_completed": True,
                "training_signature": signature_obj.as_dict(),
            },
        )
        return artifact, trainer_state

    def fit(self, data: ProcessedDataset | SampleDataset) -> SymbolicSurrogateArtifact:
        artifact, _ = self._fit_internal(data)
        return artifact

    def fit_task(
        self,
        task: TrainTask,
        init: TrainingInit | None = None,
    ) -> FitResult:
        init_eff = init or TrainingInit()
        caps = coerce_trainer_capabilities(self.capabilities())
        task_signature = build_task_signature(task, trainer=self)
        verdict = require_training_setup(
            caps,
            init_eff,
            trainer_name=str(getattr(self, "name", type(self).__name__)),
            current_signature=task_signature,
        )
        artifact, trainer_state = self._fit_internal(
            task.data,
            init=init_eff,
            training_signature=task_signature.as_dict(),
        )
        attach_signature_to_artifact(artifact, task_signature)
        lineage = TrainingLineage(
            mode=str(init_eff.mode),
            trainer_name=str(getattr(self, "name", type(self).__name__)),
            parent_artifact_id=(
                None
                if init_eff.parent_artifact is None
                else str(getattr(init_eff.parent_artifact, "artifact_id", type(init_eff.parent_artifact).__name__))
            ),
            parent_state_trainer=(
                None
                if init_eff.parent_state is None
                else str(getattr(init_eff.parent_state, "trainer_name", type(init_eff.parent_state).__name__))
            ),
            metadata={
                "task_id": str(task.task_id),
                "task_metadata": dict(task.metadata),
                "task_signature": task_signature.as_dict(),
            },
        )
        return FitResult(
            artifact=artifact,
            trainer_state=trainer_state,
            report={
                "training_mode": str(init_eff.mode),
                "trainer_capabilities": caps.as_dict(),
                "task_signature": task_signature.as_dict(),
                "compatibility": verdict.metadata,
                "compatibility_warnings": list(verdict.warnings),
            },
            lineage=lineage,
        )


__all__ = [
    "SymbolicOrthogonalSurrogateTrainer",
    "SymbolicOrthogonalTrainerConfig",
]
