from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from bias import BaseTrainingBias, FitContext, NoOpBias
from numericizer import BaseNumericizer, DefaultNumericizer, ModalityEncoder, TargetCodec
from pipeline import BasePipeline, IdentityPipeline

from core.common.base_trainer import BaseSurrogateTrainer
from core.common.contracts import ProcessedDataset, SampleDataset
from core.artifacts.symbolic_artifact import SymbolicSurrogateArtifact
from core.execution import ExecutionResourceRequest
from core.symbolic.gradient_parser import GradientParser
from core.symbolic.structure_metadata import (
    build_assembler_budget_payload,
    build_basis_overlap_report,
    build_basis_semantics_payload,
    build_basis_term_rows,
)
from core.common.trainer_shared import prepare_training_data, resolve_torch_device, set_torch_seed
from core.symbolic.trainer_state_io import (
    clone_symbolic_payload_cpu,
    load_symbolic_trainer_state_file,
    save_symbolic_trainer_state_file,
)
from core.symbolic.trainer_family import SymbolicStructureEngineSpec, coerce_symbolic_structure_engine_spec
from core.symbolic.symbolic_dsl import evaluate_genome_numpy, normalize_genome
from core.symbolic.symbolic_structure_search import (
    StructureSearchConfig,
    StructureSearchResult,
    evaluate_genome_with_ridge,
    regression_metrics,
    residual_guided_structure_search,
)
from training import (
    FitResult,
    InnerRuntimeDispatcher,
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
class SymbolicStagewiseStrategyConfig:
    force_linear_base: str | bool
    keep_search_trace: bool


@dataclass(frozen=True)
class SymbolicStagewiseAutoModeConfig:
    val_ratio: float
    min_val_samples: int
    random_seed: int
    term_penalty: float
    depth_penalty: float
    grad_penalty: float


@dataclass(frozen=True)
class SymbolicStagewiseSearchCoreConfig:
    max_added_terms: int
    topk_features: int
    max_pair_terms: int
    max_candidates_per_iter: int
    candidate_keep_top: int
    max_arity: int
    max_expr_depth: int
    min_actual_rmse_gain: float
    ridge_l2: float
    min_score: float
    min_projected_gain: float
    score_complexity_penalty: float
    score_corr_bonus: float


@dataclass(frozen=True)
class SymbolicStagewiseSearchOverfitConfig:
    enabled: bool
    val_ratio: float
    min_val_samples: int
    random_seed: int
    min_val_rmse_gain: float
    max_gap_increase: float
    patience: int
    snapshot_min_improve: float
    tabu_rounds: int
    replace_topk: int
    replace_drop_topk: int


@dataclass(frozen=True)
class SymbolicStagewiseSearchGradientConfig:
    guidance_bonus: float
    focus_topk: int
    min_priority: float
    slope_mode: str
    slope_bins: int
    slope_min_bin_samples: int
    adv_check: bool
    adv_trials: int
    adv_noise_std: float
    adv_min_stability: float
    adv_random_seed: int
    enable_residual_projection: bool
    projection_topk_focus: int
    projection_partner_pool: int
    projection_topk_partners: int
    projection_topk_unary: int
    projection_focus_include_transforms: bool
    projection_focus_topk_transforms: int
    projection_partner_orders: Sequence[int]
    projection_enable_pair_dictionary: bool
    projection_min_abs_corr: float
    projection_max_generated: int
    interaction_grad_projection_budget_boost: float


@dataclass(frozen=True)
class SymbolicStagewiseSearchFamilyConfig:
    include_hinge: bool
    hinge_quantiles: Sequence[float]
    unary_ops: Sequence[str]
    nested_mode: str
    nested_unary_patterns: Sequence[str]
    auto_nested_allowed_ops: Sequence[str]
    auto_nested_min_depth: int
    auto_nested_max_depth: int
    auto_nested_beam_width: int
    auto_nested_max_patterns_per_feature: int
    interaction_budget_mode: str
    interaction_diag_threshold: float
    interaction_diag_topk_features: int
    interaction_pair_budget_boost: float


@dataclass(frozen=True)
class SymbolicStagewiseSearchPruneConfig:
    enabled: bool
    rmse_tolerance: float
    max_removed_per_iter: int


@dataclass(frozen=True)
class SymbolicStagewiseSearchPathMemoryConfig:
    enabled: bool
    db_path: str
    namespace: str
    prior_bonus: float
    tabu_penalty: float
    min_outcomes: int
    hard_tabu: bool
    hard_tabu_accept_rate: float


@dataclass(frozen=True)
class SymbolicStagewiseSearchGraphCacheConfig:
    enabled: bool
    max_value_entries: int
    max_derivative_entries: int
    backend: str
    db_path: str
    namespace: str
    persist_values: bool


@dataclass(frozen=True)
class SymbolicStagewiseSearchBeamConfig:
    enabled: bool
    width: int
    bundle_size: int
    branches_per_beam: int
    jitter: float
    early_stop_rounds: int


@dataclass(frozen=True)
class SymbolicStagewiseSearchJointBundleConfig:
    enabled: bool
    max_terms: int
    preselect_topk: int
    max_combos: int
    l1_alpha: float
    l1_iters: int


@dataclass(frozen=True)
class SymbolicStagewiseSearchInnerOptConfig:
    enabled: bool
    method: str
    device: str
    random_seed: int
    adam_steps: int
    adam_lr: float
    adam_weight_decay: float
    lbfgs_steps: int
    lbfgs_lr: float
    l2: float
    accept_rmse_tol: float


@dataclass(frozen=True)
class SymbolicStagewiseArtifactRuntimeConfig:
    ood_z_threshold: float
    epsilon: float


@dataclass(frozen=True)
class SymbolicStagewiseTrainerConfig:
    artifact_id: str = "symbolic_stagewise_surrogate_v1"
    structure_engine: SymbolicStructureEngineSpec | Mapping[str, Any] | None = None

    # strategy mode: on | off | auto (bool true/false also accepted)
    force_linear_base: str | bool = "on"
    keep_search_trace: bool = True

    # auto mode controls
    auto_val_ratio: float = 0.2
    auto_min_val_samples: int = 64
    auto_random_seed: int = 42
    auto_term_penalty: float = 1e-3
    auto_depth_penalty: float = 2e-3
    auto_grad_penalty: float = 5e-2

    # search controls
    search_max_added_terms: int = 10
    search_topk_features: int = 8
    search_max_pair_terms: int = 16
    search_max_candidates_per_iter: int = 500
    search_candidate_keep_top: int = 12
    search_max_arity: int = 3
    search_max_expr_depth: int = 8
    search_min_actual_rmse_gain: float = 0.0
    search_overfit_guard_enabled: bool = False
    search_overfit_guard_val_ratio: float = 0.2
    search_overfit_guard_min_val_samples: int = 64
    search_overfit_guard_random_seed: int = 42
    search_overfit_guard_min_val_rmse_gain: float = 0.0
    search_overfit_guard_max_gap_increase: float = 0.05
    search_overfit_guard_patience: int = 3
    search_overfit_guard_snapshot_min_improve: float = 0.0
    search_overfit_guard_tabu_rounds: int = 2
    search_overfit_guard_replace_topk: int = 3
    search_overfit_guard_replace_drop_topk: int = 3

    # score controls
    search_ridge_l2: float = 1e-4
    search_min_score: float = 1e-6
    search_min_projected_gain: float = 1e-7
    search_score_complexity_penalty: float = 7e-4
    search_score_corr_bonus: float = 0.04
    search_grad_guidance_bonus: float = 0.08
    search_grad_focus_topk: int = 3
    search_grad_min_priority: float = 1e-4
    search_grad_slope_mode: str = "central_diff"
    search_grad_slope_bins: int = 24
    search_grad_slope_min_bin_samples: int = 12

    search_grad_adv_check: bool = False
    search_grad_adv_trials: int = 3
    search_grad_adv_noise_std: float = 0.02
    search_grad_adv_min_stability: float = 0.0
    search_grad_adv_random_seed: int = 42

    # function family
    search_include_hinge: bool = True
    search_hinge_quantiles: Sequence[float] = (0.25, 0.5, 0.75)
    search_unary_ops: Sequence[str] = ("square", "sin", "cos", "tanh")
    search_nested_mode: str = "auto"  # manual | auto | hybrid
    search_nested_unary_patterns: Sequence[str] = ("sin(square)", "cos(square)")
    search_auto_nested_allowed_ops: Sequence[str] = ("square", "sin", "cos", "tanh")
    search_auto_nested_min_depth: int = 2
    search_auto_nested_max_depth: int = 3
    search_auto_nested_beam_width: int = 8
    search_auto_nested_max_patterns_per_feature: int = 16
    search_interaction_budget_mode: str = "fixed"  # fixed | interaction_first
    search_interaction_diag_threshold: float = 1.15
    search_interaction_diag_topk_features: int = 6
    search_interaction_pair_budget_boost: float = 2.0
    search_interaction_grad_projection_budget_boost: float = 1.5
    search_enable_grad_residual_projection: bool = True
    search_grad_projection_topk_focus: int = 3
    search_grad_projection_partner_pool: int = 8
    search_grad_projection_topk_partners: int = 3
    search_grad_projection_topk_unary: int = 2
    search_grad_projection_focus_include_transforms: bool = True
    search_grad_projection_focus_topk_transforms: int = 2
    # k means x_i multiplied by k projected partner transforms (total arity = k + 1)
    search_grad_projection_partner_orders: Sequence[int] = (1, 2)
    search_grad_projection_enable_pair_dictionary: bool = True
    search_grad_projection_min_abs_corr: float = 0.05
    search_grad_projection_max_generated: int = 120

    # structural pruning
    search_enable_prune: bool = True
    search_prune_rmse_tolerance: float = 1e-8
    search_prune_max_removed_per_iter: int = 1

    # persistent path memory
    search_path_memory_enabled: bool = True
    search_path_memory_db_path: str = ""
    search_path_memory_namespace: str = "global"
    search_path_memory_prior_bonus: float = 0.03
    search_path_memory_tabu_penalty: float = 0.06
    search_path_memory_min_outcomes: int = 3
    search_path_memory_hard_tabu: bool = False
    search_path_memory_hard_tabu_accept_rate: float = 0.1

    # reusable compute-graph cache
    search_graph_cache_enabled: bool = True
    search_graph_cache_max_value_entries: int = 20000
    search_graph_cache_max_derivative_entries: int = 50000
    search_graph_cache_backend: str = "memory"  # memory | sqlite | lmdb
    search_graph_cache_db_path: str = ""
    search_graph_cache_namespace: str = "global"
    search_graph_cache_persist_values: bool = False

    # online beam+bundle search wrapper (trainer-level orchestration)
    search_online_beam_enabled: bool = True
    search_online_beam_width: int = 4
    search_online_bundle_size: int = 2
    search_online_branches_per_beam: int = 2
    search_online_beam_jitter: float = 0.08
    search_online_early_stop_rounds: int = 2

    # inner-loop joint bundle (approximate L0/L1 subset on shortlist)
    search_joint_bundle_enabled: bool = False
    search_joint_bundle_max_terms: int = 3
    search_joint_bundle_preselect_topk: int = 8
    search_joint_bundle_max_combos: int = 48
    search_joint_bundle_l1_alpha: float = 1e-3
    search_joint_bundle_l1_iters: int = 20

    # fixed-structure inner loop (readout-only)
    search_inner_opt_enabled: bool = False
    search_inner_opt_method: str = "adam_lbfgs"  # adam_lbfgs | adam | lbfgs
    search_inner_opt_device: str = "auto"  # auto | cpu | cuda | cuda:<index>
    search_inner_opt_random_seed: int = 42
    search_inner_opt_adam_steps: int = 120
    search_inner_opt_adam_lr: float = 5e-3
    search_inner_opt_adam_weight_decay: float = 0.0
    search_inner_opt_lbfgs_steps: int = 60
    search_inner_opt_lbfgs_lr: float = 0.8
    search_inner_opt_l2: float = 0.0
    search_inner_opt_accept_rmse_tol: float = 1e-6

    # artifact/runtime
    ood_z_threshold: float = 4.0
    epsilon: float = 1e-6

    def strategy_config(self) -> SymbolicStagewiseStrategyConfig:
        return SymbolicStagewiseStrategyConfig(
            force_linear_base=self.force_linear_base,
            keep_search_trace=bool(self.keep_search_trace),
        )

    def auto_mode_config(self) -> SymbolicStagewiseAutoModeConfig:
        return SymbolicStagewiseAutoModeConfig(
            val_ratio=float(self.auto_val_ratio),
            min_val_samples=int(self.auto_min_val_samples),
            random_seed=int(self.auto_random_seed),
            term_penalty=float(self.auto_term_penalty),
            depth_penalty=float(self.auto_depth_penalty),
            grad_penalty=float(self.auto_grad_penalty),
        )

    def search_core_config(self) -> SymbolicStagewiseSearchCoreConfig:
        return SymbolicStagewiseSearchCoreConfig(
            max_added_terms=int(self.search_max_added_terms),
            topk_features=int(self.search_topk_features),
            max_pair_terms=int(self.search_max_pair_terms),
            max_candidates_per_iter=int(self.search_max_candidates_per_iter),
            candidate_keep_top=int(self.search_candidate_keep_top),
            max_arity=int(self.search_max_arity),
            max_expr_depth=int(self.search_max_expr_depth),
            min_actual_rmse_gain=float(self.search_min_actual_rmse_gain),
            ridge_l2=float(self.search_ridge_l2),
            min_score=float(self.search_min_score),
            min_projected_gain=float(self.search_min_projected_gain),
            score_complexity_penalty=float(self.search_score_complexity_penalty),
            score_corr_bonus=float(self.search_score_corr_bonus),
        )

    def search_overfit_config(self) -> SymbolicStagewiseSearchOverfitConfig:
        return SymbolicStagewiseSearchOverfitConfig(
            enabled=bool(self.search_overfit_guard_enabled),
            val_ratio=float(self.search_overfit_guard_val_ratio),
            min_val_samples=int(self.search_overfit_guard_min_val_samples),
            random_seed=int(self.search_overfit_guard_random_seed),
            min_val_rmse_gain=float(self.search_overfit_guard_min_val_rmse_gain),
            max_gap_increase=float(self.search_overfit_guard_max_gap_increase),
            patience=int(self.search_overfit_guard_patience),
            snapshot_min_improve=float(self.search_overfit_guard_snapshot_min_improve),
            tabu_rounds=int(self.search_overfit_guard_tabu_rounds),
            replace_topk=int(self.search_overfit_guard_replace_topk),
            replace_drop_topk=int(self.search_overfit_guard_replace_drop_topk),
        )

    def search_gradient_config(self) -> SymbolicStagewiseSearchGradientConfig:
        return SymbolicStagewiseSearchGradientConfig(
            guidance_bonus=float(self.search_grad_guidance_bonus),
            focus_topk=int(self.search_grad_focus_topk),
            min_priority=float(self.search_grad_min_priority),
            slope_mode=str(self.search_grad_slope_mode),
            slope_bins=int(self.search_grad_slope_bins),
            slope_min_bin_samples=int(self.search_grad_slope_min_bin_samples),
            adv_check=bool(self.search_grad_adv_check),
            adv_trials=int(self.search_grad_adv_trials),
            adv_noise_std=float(self.search_grad_adv_noise_std),
            adv_min_stability=float(self.search_grad_adv_min_stability),
            adv_random_seed=int(self.search_grad_adv_random_seed),
            enable_residual_projection=bool(self.search_enable_grad_residual_projection),
            projection_topk_focus=int(self.search_grad_projection_topk_focus),
            projection_partner_pool=int(self.search_grad_projection_partner_pool),
            projection_topk_partners=int(self.search_grad_projection_topk_partners),
            projection_topk_unary=int(self.search_grad_projection_topk_unary),
            projection_focus_include_transforms=bool(self.search_grad_projection_focus_include_transforms),
            projection_focus_topk_transforms=int(self.search_grad_projection_focus_topk_transforms),
            projection_partner_orders=tuple(int(v) for v in self.search_grad_projection_partner_orders),
            projection_enable_pair_dictionary=bool(self.search_grad_projection_enable_pair_dictionary),
            projection_min_abs_corr=float(self.search_grad_projection_min_abs_corr),
            projection_max_generated=int(self.search_grad_projection_max_generated),
            interaction_grad_projection_budget_boost=float(self.search_interaction_grad_projection_budget_boost),
        )

    def search_family_config(self) -> SymbolicStagewiseSearchFamilyConfig:
        return SymbolicStagewiseSearchFamilyConfig(
            include_hinge=bool(self.search_include_hinge),
            hinge_quantiles=tuple(float(v) for v in self.search_hinge_quantiles),
            unary_ops=tuple(str(v) for v in self.search_unary_ops),
            nested_mode=str(self.search_nested_mode),
            nested_unary_patterns=tuple(str(v) for v in self.search_nested_unary_patterns),
            auto_nested_allowed_ops=tuple(str(v) for v in self.search_auto_nested_allowed_ops),
            auto_nested_min_depth=int(self.search_auto_nested_min_depth),
            auto_nested_max_depth=int(self.search_auto_nested_max_depth),
            auto_nested_beam_width=int(self.search_auto_nested_beam_width),
            auto_nested_max_patterns_per_feature=int(self.search_auto_nested_max_patterns_per_feature),
            interaction_budget_mode=str(self.search_interaction_budget_mode),
            interaction_diag_threshold=float(self.search_interaction_diag_threshold),
            interaction_diag_topk_features=int(self.search_interaction_diag_topk_features),
            interaction_pair_budget_boost=float(self.search_interaction_pair_budget_boost),
        )

    def search_prune_config(self) -> SymbolicStagewiseSearchPruneConfig:
        return SymbolicStagewiseSearchPruneConfig(
            enabled=bool(self.search_enable_prune),
            rmse_tolerance=float(self.search_prune_rmse_tolerance),
            max_removed_per_iter=int(self.search_prune_max_removed_per_iter),
        )

    def search_path_memory_config(self) -> SymbolicStagewiseSearchPathMemoryConfig:
        return SymbolicStagewiseSearchPathMemoryConfig(
            enabled=bool(self.search_path_memory_enabled),
            db_path=str(self.search_path_memory_db_path),
            namespace=str(self.search_path_memory_namespace),
            prior_bonus=float(self.search_path_memory_prior_bonus),
            tabu_penalty=float(self.search_path_memory_tabu_penalty),
            min_outcomes=int(self.search_path_memory_min_outcomes),
            hard_tabu=bool(self.search_path_memory_hard_tabu),
            hard_tabu_accept_rate=float(self.search_path_memory_hard_tabu_accept_rate),
        )

    def search_graph_cache_config(self) -> SymbolicStagewiseSearchGraphCacheConfig:
        return SymbolicStagewiseSearchGraphCacheConfig(
            enabled=bool(self.search_graph_cache_enabled),
            max_value_entries=int(self.search_graph_cache_max_value_entries),
            max_derivative_entries=int(self.search_graph_cache_max_derivative_entries),
            backend=str(self.search_graph_cache_backend),
            db_path=str(self.search_graph_cache_db_path),
            namespace=str(self.search_graph_cache_namespace),
            persist_values=bool(self.search_graph_cache_persist_values),
        )

    def search_online_beam_config(self) -> SymbolicStagewiseSearchBeamConfig:
        return SymbolicStagewiseSearchBeamConfig(
            enabled=bool(self.search_online_beam_enabled),
            width=int(self.search_online_beam_width),
            bundle_size=int(self.search_online_bundle_size),
            branches_per_beam=int(self.search_online_branches_per_beam),
            jitter=float(self.search_online_beam_jitter),
            early_stop_rounds=int(self.search_online_early_stop_rounds),
        )

    def search_joint_bundle_config(self) -> SymbolicStagewiseSearchJointBundleConfig:
        return SymbolicStagewiseSearchJointBundleConfig(
            enabled=bool(self.search_joint_bundle_enabled),
            max_terms=int(self.search_joint_bundle_max_terms),
            preselect_topk=int(self.search_joint_bundle_preselect_topk),
            max_combos=int(self.search_joint_bundle_max_combos),
            l1_alpha=float(self.search_joint_bundle_l1_alpha),
            l1_iters=int(self.search_joint_bundle_l1_iters),
        )

    def search_inner_opt_config(self) -> SymbolicStagewiseSearchInnerOptConfig:
        return SymbolicStagewiseSearchInnerOptConfig(
            enabled=bool(self.search_inner_opt_enabled),
            method=str(self.search_inner_opt_method),
            device=str(self.search_inner_opt_device),
            random_seed=int(self.search_inner_opt_random_seed),
            adam_steps=int(self.search_inner_opt_adam_steps),
            adam_lr=float(self.search_inner_opt_adam_lr),
            adam_weight_decay=float(self.search_inner_opt_adam_weight_decay),
            lbfgs_steps=int(self.search_inner_opt_lbfgs_steps),
            lbfgs_lr=float(self.search_inner_opt_lbfgs_lr),
            l2=float(self.search_inner_opt_l2),
            accept_rmse_tol=float(self.search_inner_opt_accept_rmse_tol),
        )

    def artifact_runtime_config(self) -> SymbolicStagewiseArtifactRuntimeConfig:
        return SymbolicStagewiseArtifactRuntimeConfig(
            ood_z_threshold=float(self.ood_z_threshold),
            epsilon=float(self.epsilon),
        )

    def grouped_view(self) -> dict[str, Any]:
        return {
            "strategy": asdict(self.strategy_config()),
            "auto_mode": asdict(self.auto_mode_config()),
            "search_core": asdict(self.search_core_config()),
            "search_overfit": asdict(self.search_overfit_config()),
            "search_gradient": asdict(self.search_gradient_config()),
            "search_family": asdict(self.search_family_config()),
            "search_prune": asdict(self.search_prune_config()),
            "search_path_memory": asdict(self.search_path_memory_config()),
            "search_graph_cache": asdict(self.search_graph_cache_config()),
            "search_online_beam": asdict(self.search_online_beam_config()),
            "search_joint_bundle": asdict(self.search_joint_bundle_config()),
            "search_inner_opt": asdict(self.search_inner_opt_config()),
            "artifact_runtime": asdict(self.artifact_runtime_config()),
        }



@dataclass(frozen=True)
class BeamState:
    genome: tuple[dict[str, Any], ...]
    rmse: float
    score: float
    search_res: StructureSearchResult | None = None
    source: str = "seed"


class SymbolicStagewiseSurrogateTrainer(BaseSurrogateTrainer):
    """Linear-floor + residual-increment symbolic search trainer.

    Strategy modes:
    - on: always start with linear base terms x0..xd-1
    - off: start from empty base (bias only + nonlinear increments)
    - auto: run on/off branch selection on a validation split, then refit selected mode on full train split
    """

    name = "symbolic_stagewise"

    def __init__(
        self,
        config: SymbolicStagewiseTrainerConfig | None = None,
        *,
        pipeline: BasePipeline | None = None,
        biases: Sequence[BaseTrainingBias] | None = None,
        numericizer: BaseNumericizer | None = None,
        modality_encoders: Mapping[str, ModalityEncoder] | None = None,
        target_codecs: Mapping[str, TargetCodec] | None = None,
        target_codec: str | None = None,
        categorical_unknown: str = "error",
    ) -> None:
        self.config = config or SymbolicStagewiseTrainerConfig()
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
        return dict(SymbolicStagewiseSurrogateTrainer._clone_payload_cpu(dict(state)))

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
                "epoch_done": int(payload.get("epoch_done", 0)),
                "training_signature": signature.as_dict(),
            },
        )

    def _resolve_structure_engine(self) -> SymbolicStructureEngineSpec:
        default = SymbolicStructureEngineSpec(
            structure_mode="stagewise_search",
            search_driver="nsgablack",
            dynamic_pool_enabled=True,
        )
        raw = getattr(self.config, "structure_engine", None)
        if raw is None:
            family_spec = getattr(self, "symbolic_family_spec", None)
            raw = getattr(family_spec, "structure_engine", None) if family_spec is not None else None
        return coerce_symbolic_structure_engine_spec(raw, default=default)

    @staticmethod
    def _seed_payload_from_artifact(artifact: Any) -> dict[str, Any] | None:
        if not isinstance(artifact, SymbolicSurrogateArtifact):
            return None
        metadata = dict(getattr(artifact, "metadata", {}) or {})
        strategy = dict(metadata.get("strategy", {}) or {})
        search_trace = dict(metadata.get("search_trace", {}) or {})
        return {
            "schema_version": 1,
            "trainer_name": "symbolic_stagewise",
            "search_completed": True,
            "epoch_done": 0,
            "selected_mode": str(strategy.get("force_linear_base_selected", "unknown")),
            "requested_mode": str(strategy.get("force_linear_base_requested", "unknown")),
            "genome": SymbolicStagewiseSurrogateTrainer._copy_genome(artifact.genome),
            "parameter_values": dict(getattr(artifact, "parameter_values", {}) or {}),
            "readout_weight": np.asarray(getattr(artifact, "readout_weight"), dtype=float),
            "readout_bias": np.asarray(getattr(artifact, "readout_bias"), dtype=float),
            "residual_std": np.asarray(getattr(artifact, "residual_std"), dtype=float),
            "feature_names": tuple(str(v) for v in getattr(artifact, "feature_names", ()) or ()),
            "target_names": tuple(str(v) for v in getattr(artifact, "target_names", ()) or ()),
            "base_metrics": dict(strategy.get("base_metrics", {}) or {}),
            "final_metrics": dict(strategy.get("final_metrics", {}) or {}),
            "artifact_train_metrics": dict(strategy.get("artifact_train_metrics", {}) or {}),
            "inner_opt": dict(strategy.get("inner_opt", {}) or {}),
            "search_config_grouped": dict(strategy.get("search_config_grouped", {}) or {}),
            "search_iterations": [dict(item) for item in search_trace.get("iterations", [])],
            "search_score_trace": [float(v) for v in search_trace.get("score_trace", [])],
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
            "model_family": "symbolic_stagewise",
            "backend": "numpy",
            "nonlinear": True,
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "sample_weight": False,
                "target_codec": True,
                "linear_floor_mode": "on|off|auto",
                "residual_increment_search": True,
                "path_memory": True,
                "add_and_drop_terms": True,
                "expression_export": True,
                "gradient_directional_guidance": True,
                "gradient_residual_projection": True,
                "max_arity_depth_controls": True,
                "overfit_guard_rollback_replace": True,
                "fixed_structure_inner_opt": "adam_lbfgs|adam|lbfgs",
                "graph_cache_persistent_backend": "memory|sqlite|lmdb",
                "trainer_state": True,
            },
            "artifacts": {
                "type": "SymbolicSurrogateArtifact",
                "uncertainty": "residual_std",
                "ood_validity": True,
            },
            "runtime": {
                "resume_from_trainer_state": True,
                "resume_semantics": "seeded_structure_restart",
                "warm_start_semantics": "reuse_parent_genome_as_seed",
                "incremental_semantics": "reuse_parent_genome_as_seed",
            },
        }

    def execution_resource_request(self) -> ExecutionResourceRequest:
        device_tokens: tuple[str, ...] = tuple()
        if bool(self.config.search_inner_opt_enabled):
            device_tokens = self.resolve_execution_device_tokens(self.config.search_inner_opt_device)
        return ExecutionResourceRequest(
            threads=1,
            backend="serial",
            label=str(self.name),
            device_tokens=device_tokens,
            metadata={
                "backend_family": "numpy_symbolic_stagewise",
                "inner_opt_enabled": bool(self.config.search_inner_opt_enabled),
                "requested_device": str(self.config.search_inner_opt_device),
            },
        )

    def _build_search_config(self) -> StructureSearchConfig:
        core = self.config.search_core_config()
        overfit = self.config.search_overfit_config()
        gradient = self.config.search_gradient_config()
        family = self.config.search_family_config()
        prune = self.config.search_prune_config()
        path_memory = self.config.search_path_memory_config()
        graph_cache = self.config.search_graph_cache_config()
        joint_bundle = self.config.search_joint_bundle_config()
        return StructureSearchConfig(
            max_added_terms=int(max(0, core.max_added_terms)),
            topk_features=int(max(1, core.topk_features)),
            max_pair_terms=int(max(0, core.max_pair_terms)),
            max_candidates_per_iter=int(max(1, core.max_candidates_per_iter)),
            candidate_keep_top=int(max(1, core.candidate_keep_top)),
            max_arity=int(max(1, core.max_arity)),
            max_expr_depth=int(max(1, core.max_expr_depth)),
            ridge_l2=float(max(0.0, core.ridge_l2)),
            min_score=float(core.min_score),
            min_projected_gain=float(core.min_projected_gain),
            min_actual_rmse_gain=float(core.min_actual_rmse_gain),
            overfit_guard_enabled=bool(overfit.enabled),
            overfit_guard_val_ratio=float(np.clip(float(overfit.val_ratio), 0.0, 0.9)),
            overfit_guard_min_val_samples=int(max(1, overfit.min_val_samples)),
            overfit_guard_random_seed=int(overfit.random_seed),
            overfit_guard_min_val_rmse_gain=float(overfit.min_val_rmse_gain),
            overfit_guard_max_gap_increase=float(overfit.max_gap_increase),
            overfit_guard_patience=int(max(0, overfit.patience)),
            overfit_guard_snapshot_min_improve=float(max(0.0, overfit.snapshot_min_improve)),
            overfit_guard_tabu_rounds=int(max(0, overfit.tabu_rounds)),
            overfit_guard_replace_topk=int(max(0, overfit.replace_topk)),
            overfit_guard_replace_drop_topk=int(max(0, overfit.replace_drop_topk)),
            score_complexity_penalty=float(core.score_complexity_penalty),
            score_corr_bonus=float(core.score_corr_bonus),
            score_grad_guidance_bonus=float(gradient.guidance_bonus),
            grad_focus_topk=int(max(1, gradient.focus_topk)),
            grad_min_priority=float(max(0.0, gradient.min_priority)),
            grad_slope_mode=str(gradient.slope_mode),
            grad_slope_bins=int(max(3, gradient.slope_bins)),
            grad_slope_min_bin_samples=int(max(3, gradient.slope_min_bin_samples)),
            grad_adv_check=bool(gradient.adv_check),
            grad_adv_trials=int(max(0, gradient.adv_trials)),
            grad_adv_noise_std=float(max(0.0, gradient.adv_noise_std)),
            grad_adv_min_stability=float(max(0.0, gradient.adv_min_stability)),
            grad_adv_random_seed=int(gradient.adv_random_seed),
            include_hinge=bool(family.include_hinge),
            hinge_quantiles=tuple(float(v) for v in family.hinge_quantiles),
            unary_ops=tuple(str(v) for v in family.unary_ops),
            nested_mode=str(family.nested_mode),
            nested_unary_patterns=tuple(str(v) for v in family.nested_unary_patterns),
            auto_nested_allowed_ops=tuple(str(v) for v in family.auto_nested_allowed_ops),
            auto_nested_min_depth=int(max(1, family.auto_nested_min_depth)),
            auto_nested_max_depth=int(max(1, family.auto_nested_max_depth)),
            auto_nested_beam_width=int(max(1, family.auto_nested_beam_width)),
            auto_nested_max_patterns_per_feature=int(max(0, family.auto_nested_max_patterns_per_feature)),
            interaction_budget_mode=str(family.interaction_budget_mode),
            interaction_diag_threshold=float(max(0.0, family.interaction_diag_threshold)),
            interaction_diag_topk_features=int(max(2, family.interaction_diag_topk_features)),
            interaction_pair_budget_boost=float(max(1.0, family.interaction_pair_budget_boost)),
            interaction_grad_projection_budget_boost=float(max(1.0, gradient.interaction_grad_projection_budget_boost)),
            enable_grad_residual_projection=bool(gradient.enable_residual_projection),
            grad_projection_topk_focus=int(max(1, gradient.projection_topk_focus)),
            grad_projection_partner_pool=int(max(2, gradient.projection_partner_pool)),
            grad_projection_topk_partners=int(max(1, gradient.projection_topk_partners)),
            grad_projection_topk_unary=int(max(1, gradient.projection_topk_unary)),
            grad_projection_focus_include_transforms=bool(gradient.projection_focus_include_transforms),
            grad_projection_focus_topk_transforms=int(max(1, gradient.projection_focus_topk_transforms)),
            grad_projection_partner_orders=tuple(int(max(1, v)) for v in gradient.projection_partner_orders),
            grad_projection_enable_pair_dictionary=bool(gradient.projection_enable_pair_dictionary),
            grad_projection_min_abs_corr=float(max(0.0, gradient.projection_min_abs_corr)),
            grad_projection_max_generated=int(max(0, gradient.projection_max_generated)),
            enable_prune=bool(prune.enabled),
            prune_rmse_tolerance=float(max(0.0, prune.rmse_tolerance)),
            prune_max_removed_per_iter=int(max(0, prune.max_removed_per_iter)),
            path_memory_enabled=bool(path_memory.enabled),
            path_memory_db_path=str(path_memory.db_path),
            path_memory_namespace=str(path_memory.namespace),
            path_memory_prior_bonus=float(max(0.0, path_memory.prior_bonus)),
            path_memory_tabu_penalty=float(max(0.0, path_memory.tabu_penalty)),
            path_memory_min_outcomes=int(max(0, path_memory.min_outcomes)),
            path_memory_hard_tabu=bool(path_memory.hard_tabu),
            path_memory_hard_tabu_accept_rate=float(np.clip(float(path_memory.hard_tabu_accept_rate), 0.0, 1.0)),
            graph_cache_enabled=bool(graph_cache.enabled),
            graph_cache_max_value_entries=int(max(1, graph_cache.max_value_entries)),
            graph_cache_max_derivative_entries=int(max(1, graph_cache.max_derivative_entries)),
            graph_cache_backend=str(graph_cache.backend),
            graph_cache_db_path=str(graph_cache.db_path),
            graph_cache_namespace=str(graph_cache.namespace),
            graph_cache_persist_values=bool(graph_cache.persist_values),
            joint_bundle_enabled=bool(joint_bundle.enabled),
            joint_bundle_max_terms=int(max(2, joint_bundle.max_terms)),
            joint_bundle_preselect_topk=int(max(2, joint_bundle.preselect_topk)),
            joint_bundle_max_combos=int(max(1, joint_bundle.max_combos)),
            joint_bundle_l1_alpha=float(max(0.0, joint_bundle.l1_alpha)),
            joint_bundle_l1_iters=int(max(1, joint_bundle.l1_iters)),
        )

    @staticmethod
    def _linear_seed_genome(input_dim: int) -> tuple[dict[str, Any], ...]:
        out: list[dict[str, Any]] = []
        for i in range(int(input_dim)):
            out.append(
                {
                    "name": f"x{i}",
                    "expr": {
                        "type": "feature",
                        "index": int(i),
                    },
                }
            )
        return tuple(out)

    @staticmethod
    def _normalize_linear_mode(value: str | bool) -> str:
        if isinstance(value, bool):
            return "on" if value else "off"

        key = str(value).strip().lower()
        if key in {"on", "true", "yes", "1", "linear", "base"}:
            return "on"
        if key in {"off", "false", "no", "0", "none"}:
            return "off"
        if key == "auto":
            return "auto"
        raise ValueError("force_linear_base must be one of: on|off|auto (or bool true/false)")

    @classmethod
    def _seed_genome_by_mode(cls, mode: str, input_dim: int) -> tuple[dict[str, Any], ...]:
        if str(mode) == "on":
            return cls._linear_seed_genome(input_dim)
        if str(mode) == "off":
            return tuple()
        raise ValueError(f"Unsupported mode: {mode}")

    @staticmethod
    def _expr_depth(expr: Mapping[str, Any]) -> int:
        t = str(expr.get("type", ""))
        if t in {"feature", "const", "param"}:
            return 1
        if t == "unary":
            return 1 + SymbolicStagewiseSurrogateTrainer._expr_depth(expr["arg"])
        if t == "binary":
            return 1 + max(
                SymbolicStagewiseSurrogateTrainer._expr_depth(expr["left"]),
                SymbolicStagewiseSurrogateTrainer._expr_depth(expr["right"]),
            )
        return 1

    @classmethod
    def _genome_complexity(cls, genome: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        terms = list(genome)
        if not terms:
            return {
                "n_terms": 0.0,
                "max_depth": 0.0,
                "mean_depth": 0.0,
            }

        depths = [float(cls._expr_depth(term["expr"])) for term in terms]
        return {
            "n_terms": float(len(terms)),
            "max_depth": float(max(depths)),
            "mean_depth": float(np.mean(depths)),
        }

    @staticmethod
    def _local_slope_1d(x_col: np.ndarray, y_mat: np.ndarray) -> np.ndarray:
        return GradientParser._local_slope_1d(x_col, y_mat)

    @staticmethod
    def _model_partial_derivative(
        genome: Sequence[Mapping[str, Any]],
        weight: np.ndarray,
        X: np.ndarray,
        *,
        feature_index: int,
    ) -> np.ndarray:
        return GradientParser.model_partial_derivative(
            genome=genome,
            weight=weight,
            X=X,
            feature_index=feature_index,
        )

    def _gradient_mismatch_metric(
        self,
        *,
        genome: Sequence[Mapping[str, Any]],
        weight: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        gradient = self.config.search_gradient_config()
        return GradientParser.gradient_mismatch(
            genome=genome,
            weight=weight,
            X=X_val,
            y=y_val,
            slope_mode=str(gradient.slope_mode),
            slope_bins=int(max(3, gradient.slope_bins)),
            slope_min_bin_samples=int(max(3, gradient.slope_min_bin_samples)),
        )

    @staticmethod
    def _extend_inner_runtime_context(
        base_context: Mapping[str, Any] | None,
        *,
        suffix: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged = {} if base_context is None else dict(base_context)
        base_run_id = str(merged.get("run_id", merged.get("task_id", "symbolic_stagewise_search")))
        suffix_text = str(suffix).strip()
        merged["run_id"] = f"{base_run_id}:{suffix_text}" if suffix_text else base_run_id
        merged.setdefault("runtime_key", "symbolic_structure_search")
        if extra:
            for key, value in dict(extra).items():
                if value is not None:
                    merged[str(key)] = value
        return merged

    def _fit_search_once(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        feature_names: Sequence[str],
        mode: str,
        search_cfg: StructureSearchConfig,
        seed_genome_override: Sequence[Mapping[str, Any]] | None = None,
        inner_runtime_dispatcher: InnerRuntimeDispatcher | None = None,
        inner_runtime_context: Mapping[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if seed_genome_override is None:
            seed_genome = self._seed_genome_by_mode(mode, int(X.shape[1]))
        else:
            seed_genome = tuple(dict(term) for term in tuple(seed_genome_override))

        search_res = residual_guided_structure_search(
            X,
            Y,
            feature_names=feature_names,
            seed_genome=seed_genome,
            config=search_cfg,
            inner_runtime_dispatcher=inner_runtime_dispatcher,
            inner_runtime_context=self._extend_inner_runtime_context(
                inner_runtime_context,
                suffix=f"fit_once:{mode}",
                extra={
                    "search_variant": "fit_once",
                    "regime_mode": str(mode),
                },
            ),
        )

        ridge_eval = evaluate_genome_with_ridge(
            search_res.genome,
            X_train=X,
            y_train=Y,
            l2=float(search_cfg.ridge_l2),
        )

        return search_res, ridge_eval

    def _fit_search_with_online_beam(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        feature_names: Sequence[str],
        mode: str,
        search_cfg: StructureSearchConfig,
        seed_genome_override: Sequence[Mapping[str, Any]] | None = None,
        inner_runtime_dispatcher: InnerRuntimeDispatcher | None = None,
        inner_runtime_context: Mapping[str, Any] | None = None,
    ) -> tuple[StructureSearchResult, dict[str, Any]]:
        beam_cfg = self.config.search_online_beam_config()
        if not bool(beam_cfg.enabled):
            return self._fit_search_once(
                X,
                Y,
                feature_names=feature_names,
                mode=mode,
                search_cfg=search_cfg,
                seed_genome_override=seed_genome_override,
                inner_runtime_dispatcher=inner_runtime_dispatcher,
                inner_runtime_context=inner_runtime_context,
            )

        beam_width = int(max(1, beam_cfg.width))
        bundle_size = int(max(1, beam_cfg.bundle_size))
        branches = int(max(1, beam_cfg.branches_per_beam))
        rounds = int(max(0, search_cfg.max_added_terms))
        if rounds <= 0:
            return self._fit_search_once(
                X,
                Y,
                feature_names=feature_names,
                mode=mode,
                search_cfg=search_cfg,
                seed_genome_override=seed_genome_override,
                inner_runtime_dispatcher=inner_runtime_dispatcher,
                inner_runtime_context=inner_runtime_context,
            )
        jitter = float(max(0.0, beam_cfg.jitter))
        early_stop_rounds = int(max(1, beam_cfg.early_stop_rounds))
        base_penalty = float(max(0.0, search_cfg.score_complexity_penalty))

        if seed_genome_override is None:
            seed_genome = tuple(self._seed_genome_by_mode(mode, int(X.shape[1])))
        else:
            seed_genome = tuple(dict(term) for term in tuple(seed_genome_override))
        seed_eval = evaluate_genome_with_ridge(
            seed_genome,
            X_train=X,
            y_train=Y,
            l2=float(search_cfg.ridge_l2),
        )
        seed_rmse = float(dict(seed_eval.get("metrics_train", {})).get("rmse", float("inf")))
        beam_states: list[BeamState] = [
            BeamState(
                genome=tuple(seed_genome),
                rmse=float(seed_rmse),
                score=self._beam_state_score(
                    rmse=seed_rmse,
                    n_terms=len(seed_genome),
                    complexity_penalty=base_penalty,
                ),
                search_res=None,
                source="seed",
            )
        ]
        history: list[dict[str, Any]] = []
        best_score = float(beam_states[0].score)
        stale_rounds = 0

        for r in range(rounds):
            expanded: list[BeamState] = []
            for b_idx, state in enumerate(beam_states):
                expanded.append(state)
                for k in range(branches):
                    factor = 1.0 + jitter * ((float(k) / max(1.0, float(branches - 1))) - 0.5) * 2.0
                    branch_penalty = float(max(0.0, base_penalty * factor))
                    branch_cfg = replace(
                        search_cfg,
                        max_added_terms=int(bundle_size),
                        score_complexity_penalty=float(branch_penalty),
                        path_memory_namespace=(
                            f"{search_cfg.path_memory_namespace}/beam_r{r}_b{b_idx}_k{k}"
                        ),
                    )
                    try:
                        res = residual_guided_structure_search(
                            X,
                            Y,
                            feature_names=feature_names,
                            seed_genome=tuple(state["genome"]),
                            config=branch_cfg,
                            inner_runtime_dispatcher=inner_runtime_dispatcher,
                            inner_runtime_context=self._extend_inner_runtime_context(
                                inner_runtime_context,
                                suffix=f"beam:r{r + 1}:b{b_idx}:k{k}",
                                extra={
                                    "search_variant": "online_beam_branch",
                                    "regime_mode": str(mode),
                                    "beam_round": int(r + 1),
                                    "beam_index": int(b_idx),
                                    "branch_index": int(k),
                                },
                            ),
                        )
                    except Exception:
                        continue

                    g = tuple(res.genome)
                    rmse = float(dict(res.final_metrics).get("rmse", float("inf")))
                    score = self._beam_state_score(
                        rmse=rmse,
                        n_terms=len(g),
                        complexity_penalty=branch_penalty,
                    )
                    expanded.append(
                        BeamState(
                            genome=g,
                            rmse=rmse,
                            score=score,
                            search_res=res,
                            source=f"expand:r{r + 1}:b{b_idx}:k{k}",
                        )
                    )

            dedup: dict[tuple[str, ...], BeamState] = {}
            for st in expanded:
                key = self._genome_key(st.genome)
                old = dedup.get(key)
                if old is None or float(st.score) < float(old.score):
                    dedup[key] = st
            ranked = sorted(dedup.values(), key=lambda s: float(s.score))
            beam_states = ranked[:beam_width]
            if not beam_states:
                break

            round_best = float(beam_states[0].score)
            round_best_state = beam_states[0]
            history.append(
                {
                    "round": int(r + 1),
                    "beam_size": int(len(beam_states)),
                    "best_score": float(round_best),
                    "best_rmse": float(round_best_state.rmse),
                    "best_n_terms": int(len(round_best_state.genome)),
                    "best_source": str(round_best_state.source),
                }
            )
            if round_best + 1e-12 < best_score:
                best_score = round_best
                stale_rounds = 0
            else:
                stale_rounds += 1
                if stale_rounds >= early_stop_rounds:
                    break

        if not beam_states:
            return self._fit_search_once(
                X,
                Y,
                feature_names=feature_names,
                mode=mode,
                search_cfg=search_cfg,
                seed_genome_override=seed_genome_override,
                inner_runtime_dispatcher=inner_runtime_dispatcher,
                inner_runtime_context=inner_runtime_context,
            )

        best_state = min(beam_states, key=lambda s: float(s.score))
        best_res = best_state.search_res
        if best_res is None:
            best_res = residual_guided_structure_search(
                X,
                Y,
                feature_names=feature_names,
                seed_genome=tuple(best_state.genome),
                config=replace(search_cfg, max_added_terms=0),
                inner_runtime_dispatcher=inner_runtime_dispatcher,
                inner_runtime_context=self._extend_inner_runtime_context(
                    inner_runtime_context,
                    suffix="beam:finalize",
                    extra={
                        "search_variant": "online_beam_finalize",
                        "regime_mode": str(mode),
                    },
                ),
            )
        ridge_eval = evaluate_genome_with_ridge(
            best_res.genome,
            X_train=X,
            y_train=Y,
            l2=float(search_cfg.ridge_l2),
        )

        merged_iters = list(best_res.iterations) + [
            {
                "beam_online_summary": {
                    "enabled": True,
                    "beam_width": int(beam_width),
                    "bundle_size": int(bundle_size),
                    "branches_per_beam": int(branches),
                    "rounds_planned": int(rounds),
                    "rounds_executed": int(len(history)),
                    "history": [dict(item) for item in history],
                }
            }
        ]
        best_res = StructureSearchResult(
            genome=tuple(best_res.genome),
            base_metrics=dict(best_res.base_metrics),
            final_metrics=dict(best_res.final_metrics),
            iterations=tuple(merged_iters),
            weight=np.asarray(best_res.weight, dtype=float),
            bias=np.asarray(best_res.bias, dtype=float),
            score_trace=tuple(float(v) for v in best_res.score_trace),
        )
        return best_res, ridge_eval

    def _auto_select_mode(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        feature_names: Sequence[str],
        search_cfg: StructureSearchConfig,
        inner_runtime_dispatcher: InnerRuntimeDispatcher | None = None,
        inner_runtime_context: Mapping[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        auto_cfg = self.config.auto_mode_config()
        n = int(X.shape[0])
        val_ratio = float(np.clip(float(auto_cfg.val_ratio), 0.05, 0.45))
        n_val = int(round(float(n) * val_ratio))
        n_val = max(1, min(n_val, n - 1))

        if n < int(max(2, auto_cfg.min_val_samples)):
            return "on", {
                "mode": "auto",
                "fallback": "on",
                "reason": "insufficient_samples_for_auto_split",
                "n_train_total": int(n),
                "auto_min_val_samples": int(auto_cfg.min_val_samples),
            }

        rng = np.random.default_rng(int(auto_cfg.random_seed))
        idx = np.arange(n, dtype=int)
        rng.shuffle(idx)
        val_idx = idx[:n_val]
        fit_idx = idx[n_val:]

        X_fit = np.asarray(X[fit_idx], dtype=float)
        Y_fit = np.asarray(Y[fit_idx], dtype=float)
        X_val = np.asarray(X[val_idx], dtype=float)
        Y_val = np.asarray(Y[val_idx], dtype=float)

        records: list[dict[str, Any]] = []

        for branch_mode in ("on", "off"):
            row: dict[str, Any] = {
                "mode": str(branch_mode),
                "status": "failed",
                "score": float("inf"),
                "val_rmse": float("inf"),
                "grad_mismatch": float("inf"),
                "complexity": {},
                "error": "",
            }
            try:
                search_res, _ = self._fit_search_once(
                    X_fit,
                    Y_fit,
                    feature_names=feature_names,
                    mode=str(branch_mode),
                    search_cfg=search_cfg,
                    inner_runtime_dispatcher=inner_runtime_dispatcher,
                    inner_runtime_context=self._extend_inner_runtime_context(
                        inner_runtime_context,
                        suffix=f"auto_probe:{branch_mode}",
                        extra={
                            "search_variant": "auto_mode_probe",
                            "regime_mode": str(branch_mode),
                        },
                    ),
                )
                val_eval = evaluate_genome_with_ridge(
                    search_res.genome,
                    X_train=X_fit,
                    y_train=Y_fit,
                    X_eval=X_val,
                    y_eval=Y_val,
                    l2=float(search_cfg.ridge_l2),
                )
                m_val = dict(val_eval.get("metrics_eval", {}))
                val_rmse = float(m_val.get("rmse", float("inf")))

                comp = self._genome_complexity(search_res.genome)
                grad_mismatch = self._gradient_mismatch_metric(
                    genome=search_res.genome,
                    weight=np.asarray(val_eval["weight"], dtype=float),
                    X_val=X_val,
                    y_val=Y_val,
                )

                term_pen = float(auto_cfg.term_penalty) * float(comp["n_terms"])
                depth_pen = float(auto_cfg.depth_penalty) * float(comp["max_depth"])
                grad_pen = float(auto_cfg.grad_penalty) * float(grad_mismatch)

                score = val_rmse + term_pen + depth_pen + grad_pen

                row.update(
                    {
                        "status": "ok",
                        "score": float(score),
                        "val_rmse": float(val_rmse),
                        "grad_mismatch": float(grad_mismatch),
                        "score_parts": {
                            "val_rmse": float(val_rmse),
                            "term_penalty": float(term_pen),
                            "depth_penalty": float(depth_pen),
                            "grad_penalty": float(grad_pen),
                        },
                        "complexity": {
                            "n_terms": int(comp["n_terms"]),
                            "max_depth": float(comp["max_depth"]),
                            "mean_depth": float(comp["mean_depth"]),
                        },
                        "fit_metrics": dict(search_res.final_metrics),
                    }
                )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"

            records.append(row)

        ok_records = [r for r in records if r.get("status") == "ok"]
        if not ok_records:
            raise RuntimeError("auto mode failed for both branches")

        best = min(ok_records, key=lambda r: float(r.get("score", float("inf"))))
        selected = str(best["mode"])

        log = {
            "mode": "auto",
            "split": {
                "n_train_total": int(n),
                "n_fit": int(X_fit.shape[0]),
                "n_val": int(X_val.shape[0]),
                "val_ratio": float(val_ratio),
                "seed": int(auto_cfg.random_seed),
            },
            "penalty": {
                "term": float(auto_cfg.term_penalty),
                "depth": float(auto_cfg.depth_penalty),
                "grad": float(auto_cfg.grad_penalty),
            },
            "candidates": records,
            "selected": selected,
        }
        return selected, log

    @staticmethod
    def _design_matrix(genome: Sequence[Mapping[str, Any]], X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        if x.ndim != 2:
            raise ValueError("X must be 2D")
        terms = list(genome)
        if not terms:
            return np.zeros((int(x.shape[0]), 0), dtype=float)
        normalized = normalize_genome(terms, input_dim=int(x.shape[1]))
        if not normalized:
            return np.zeros((int(x.shape[0]), 0), dtype=float)
        return np.asarray(evaluate_genome_numpy(normalized, x), dtype=float)

    @staticmethod
    def _genome_key(genome: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
        out: list[str] = []
        for term in genome:
            out.append(f"{term.get('name', '')}|{term.get('expr', {})}")
        return tuple(out)

    @staticmethod
    def _beam_state_score(*, rmse: float, n_terms: int, complexity_penalty: float) -> float:
        return float(rmse) + float(complexity_penalty) * float(max(0, int(n_terms)))

    def _run_fixed_structure_inner_opt(
        self,
        *,
        genome: Sequence[Mapping[str, Any]],
        X: np.ndarray,
        Y: np.ndarray,
        base_weight: np.ndarray,
        base_bias: np.ndarray,
        base_pred: np.ndarray,
    ) -> dict[str, Any]:
        y = np.asarray(Y, dtype=float)
        w0 = np.asarray(base_weight, dtype=float)
        b0 = np.asarray(base_bias, dtype=float).reshape(-1)
        pred0 = np.asarray(base_pred, dtype=float)

        metrics_before = regression_metrics(y, pred0)
        inner_opt_cfg = self.config.search_inner_opt_config()
        info: dict[str, Any] = {
            "enabled_requested": bool(inner_opt_cfg.enabled),
            "applied": False,
            "status": "disabled",
            "method": str(inner_opt_cfg.method),
            "device": None,
            "train_rmse_before": float(metrics_before.get("rmse", float("inf"))),
            "train_rmse_after": float(metrics_before.get("rmse", float("inf"))),
            "train_rmse_gain": 0.0,
            "objective_before": None,
            "objective_after": None,
            "adam_steps": 0,
            "lbfgs_steps": 0,
        }

        result = {
            "weight": np.asarray(w0, dtype=float),
            "bias": np.asarray(b0, dtype=float),
            "pred_train": np.asarray(pred0, dtype=float),
            "metrics_before": dict(metrics_before),
            "metrics_after": dict(metrics_before),
            "info": info,
        }

        if not bool(inner_opt_cfg.enabled):
            return result

        try:
            import torch
        except Exception as exc:
            info["status"] = "skipped_no_torch"
            info["error"] = f"{type(exc).__name__}: {exc}"
            return result

        method = str(inner_opt_cfg.method or "adam_lbfgs").strip().lower()
        if method not in {"adam_lbfgs", "adam", "lbfgs"}:
            info["status"] = "invalid_method"
            info["error"] = f"unsupported method: {method}"
            return result

        adam_steps = int(max(0, inner_opt_cfg.adam_steps))
        lbfgs_steps = int(max(0, inner_opt_cfg.lbfgs_steps))
        do_adam = method in {"adam_lbfgs", "adam"} and adam_steps > 0
        do_lbfgs = method in {"adam_lbfgs", "lbfgs"} and lbfgs_steps > 0
        info["adam_steps"] = int(adam_steps if do_adam else 0)
        info["lbfgs_steps"] = int(lbfgs_steps if do_lbfgs else 0)
        if not do_adam and not do_lbfgs:
            info["status"] = "skipped_no_steps"
            return result

        try:
            device = resolve_torch_device(torch, str(inner_opt_cfg.device))
            set_torch_seed(torch, int(inner_opt_cfg.random_seed))
            info["device"] = str(device)

            phi = self._design_matrix(genome, X)
            phi_t = torch.as_tensor(phi, dtype=torch.float32, device=device)
            y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
            w_t = torch.nn.Parameter(torch.as_tensor(w0, dtype=torch.float32, device=device).clone())
            b_t = torch.nn.Parameter(torch.as_tensor(b0, dtype=torch.float32, device=device).clone())

            l2 = float(max(0.0, inner_opt_cfg.l2))

            def objective() -> Any:
                pred_t = phi_t @ w_t + b_t.reshape(1, -1)
                loss_t = torch.mean((pred_t - y_t) ** 2)
                if l2 > 0.0:
                    loss_t = loss_t + float(l2) * torch.mean(w_t * w_t)
                return loss_t

            objective_before = float(objective().detach().cpu())
            objective_after = float(objective_before)

            if do_adam:
                opt_adam = torch.optim.Adam(
                    [w_t, b_t],
                    lr=float(max(1e-8, inner_opt_cfg.adam_lr)),
                    weight_decay=float(max(0.0, inner_opt_cfg.adam_weight_decay)),
                )
                for _ in range(adam_steps):
                    opt_adam.zero_grad(set_to_none=True)
                    loss_t = objective()
                    loss_t.backward()
                    opt_adam.step()
                objective_after = float(objective().detach().cpu())

            if do_lbfgs:
                opt_lbfgs = torch.optim.LBFGS(
                    [w_t, b_t],
                    lr=float(max(1e-8, inner_opt_cfg.lbfgs_lr)),
                    max_iter=int(lbfgs_steps),
                    history_size=20,
                    line_search_fn="strong_wolfe",
                )

                def closure() -> Any:
                    opt_lbfgs.zero_grad(set_to_none=True)
                    loss_t = objective()
                    loss_t.backward()
                    return loss_t

                opt_lbfgs.step(closure)
                objective_after = float(objective().detach().cpu())

            pred_after = np.asarray((phi_t @ w_t + b_t.reshape(1, -1)).detach().cpu().numpy(), dtype=float)
            metrics_after = regression_metrics(y, pred_after)
            rmse_before = float(metrics_before.get("rmse", float("inf")))
            rmse_after = float(metrics_after.get("rmse", float("inf")))
            tol = float(max(0.0, inner_opt_cfg.accept_rmse_tol))
            accepted = bool(rmse_after <= rmse_before + tol)

            info.update(
                {
                    "status": "applied" if accepted else "rejected_rmse_guard",
                    "applied": bool(accepted),
                    "train_rmse_after": float(rmse_after),
                    "train_rmse_gain": float(rmse_before - rmse_after),
                    "objective_before": float(objective_before),
                    "objective_after": float(objective_after),
                }
            )

            if not accepted:
                return result

            result["weight"] = np.asarray(w_t.detach().cpu().numpy(), dtype=float)
            result["bias"] = np.asarray(b_t.detach().cpu().numpy(), dtype=float).reshape(-1)
            result["pred_train"] = np.asarray(pred_after, dtype=float)
            result["metrics_after"] = dict(metrics_after)
            return result
        except Exception as exc:
            info["status"] = "failed"
            info["error"] = f"{type(exc).__name__}: {exc}"
            return result

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
        inner_runtime_hooks = tuple(init_eff.inner_runtime_hooks)
        inner_runtime_dispatcher = (
            InnerRuntimeDispatcher.from_hooks(inner_runtime_hooks) if inner_runtime_hooks else None
        )
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
            parent_seed_source = str(getattr(init_eff.parent_artifact, "artifact_id", type(init_eff.parent_artifact).__name__))
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
        Xb = prepared.X
        Yb = prepared.Y

        if context.sample_weight is not None:
            warning_msg = (
                "symbolic_stagewise does not implement a weighted residual-guided search path yet; "
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
                metadata["sample_weight_size"] = int(np.asarray(context.sample_weight).reshape(-1).shape[0])
            context.sample_weight = None

        n, d, m = int(prepared.n), int(prepared.d), int(prepared.m)
        feature_names = prepared.feature_names
        inner_runtime_context = {
            "task_id": str(training_signature_meta.get("task_id", "train_task")),
            "run_id": str(training_signature_meta.get("task_id", "train_task")),
            "trainer_name": str(getattr(self, "name", type(self).__name__)),
            "training_mode": str(mode),
            "runtime_key": "symbolic_structure_search",
            "search_driver": "nsgablack",
            "structure_mode": "stagewise_search",
            "artifact_id": str(self.config.artifact_id),
        }
        target_names = prepared.target_names

        strategy_cfg = self.config.strategy_config()
        inner_opt_cfg = self.config.search_inner_opt_config()
        beam_cfg = self.config.search_online_beam_config()
        grouped_cfg = self.config.grouped_view()
        search_cfg = self._build_search_config()

        requested_mode = self._normalize_linear_mode(strategy_cfg.force_linear_base)
        decision_log: dict[str, Any] | None = None
        seed_genome_override: tuple[dict[str, Any], ...] | None = None
        if isinstance(parent_payload, Mapping):
            raw_seed = parent_payload.get("genome")
            if isinstance(raw_seed, Sequence):
                seed_genome_override = tuple(dict(term) for term in tuple(raw_seed) if isinstance(term, Mapping))
                if len(seed_genome_override) == 0:
                    seed_genome_override = None

        if seed_genome_override is not None:
            selected_mode = (
                str(parent_payload.get("selected_mode", "seeded"))
                if isinstance(parent_payload, Mapping)
                else "seeded"
            )
            decision_log = {
                "mode": "seeded_restart",
                "requested_mode": str(requested_mode),
                "selected_mode": str(selected_mode),
                "seed_source": parent_seed_source,
                "seed_kind": parent_seed_kind,
                "seed_terms": int(len(seed_genome_override)),
            }
        elif requested_mode == "auto":
            selected_mode, decision_log = self._auto_select_mode(
                Xb,
                Yb,
                feature_names=feature_names,
                search_cfg=search_cfg,
                inner_runtime_dispatcher=inner_runtime_dispatcher,
                inner_runtime_context=inner_runtime_context,
            )
        else:
            selected_mode = requested_mode

        search_res, ridge_eval = self._fit_search_with_online_beam(
            Xb,
            Yb,
            feature_names=feature_names,
            mode=selected_mode,
            search_cfg=search_cfg,
            seed_genome_override=seed_genome_override,
            inner_runtime_dispatcher=inner_runtime_dispatcher,
            inner_runtime_context=inner_runtime_context,
        )

        inner_opt_result = self._run_fixed_structure_inner_opt(
            genome=search_res.genome,
            X=Xb,
            Y=Yb,
            base_weight=np.asarray(ridge_eval["weight"], dtype=float),
            base_bias=np.asarray(ridge_eval["bias"], dtype=float),
            base_pred=np.asarray(ridge_eval["pred_train"], dtype=float),
        )
        readout_weight = np.asarray(inner_opt_result["weight"], dtype=float)
        readout_bias = np.asarray(inner_opt_result["bias"], dtype=float)
        pred_train = np.asarray(inner_opt_result["pred_train"], dtype=float)
        artifact_train_metrics = dict(inner_opt_result["metrics_after"])

        residual = Yb - pred_train
        residual_std = np.std(residual, axis=0, ddof=1) + 1e-8

        metadata = {
            "trainer": "SymbolicStagewiseSurrogateTrainer",
            "n_train": int(n),
            "feature_dim": int(d),
            "target_dim": int(m),
            "pipeline": str(getattr(self.pipeline, "name", "identity")),
            "biases": [str(getattr(b, "name", type(b).__name__)) for b in self.biases],
            "fit_context": dict(context.metadata),
            "data_protocol": str((normalized.metadata or {}).get("input_protocol", "processed_dataset")),
            "data_metadata": dict(normalized.metadata or {}),
            "numericizer": str(getattr(self.numericizer, "name", type(self.numericizer).__name__)),
            "strategy": {
                "name": "linear_floor_residual_increment",
                "force_linear_base_requested": str(requested_mode),
                "force_linear_base_selected": str(selected_mode),
                "search_config": {
                    "max_added_terms": int(search_cfg.max_added_terms),
                    "topk_features": int(search_cfg.topk_features),
                    "max_pair_terms": int(search_cfg.max_pair_terms),
                    "max_candidates_per_iter": int(search_cfg.max_candidates_per_iter),
                    "candidate_keep_top": int(search_cfg.candidate_keep_top),
                    "max_arity": int(search_cfg.max_arity),
                    "max_expr_depth": int(search_cfg.max_expr_depth),
                    "ridge_l2": float(search_cfg.ridge_l2),
                    "min_score": float(search_cfg.min_score),
                    "min_projected_gain": float(search_cfg.min_projected_gain),
                    "min_actual_rmse_gain": float(search_cfg.min_actual_rmse_gain),
                    "score_complexity_penalty": float(search_cfg.score_complexity_penalty),
                    "score_corr_bonus": float(search_cfg.score_corr_bonus),
                    "score_grad_guidance_bonus": float(search_cfg.score_grad_guidance_bonus),
                    "grad_focus_topk": int(search_cfg.grad_focus_topk),
                    "grad_min_priority": float(search_cfg.grad_min_priority),
                    "grad_slope_mode": str(search_cfg.grad_slope_mode),
                    "grad_slope_bins": int(search_cfg.grad_slope_bins),
                    "grad_slope_min_bin_samples": int(search_cfg.grad_slope_min_bin_samples),
                    "grad_adv_check": bool(search_cfg.grad_adv_check),
                    "grad_adv_trials": int(search_cfg.grad_adv_trials),
                    "grad_adv_noise_std": float(search_cfg.grad_adv_noise_std),
                    "grad_adv_min_stability": float(search_cfg.grad_adv_min_stability),
                    "grad_adv_random_seed": int(search_cfg.grad_adv_random_seed),
                    "include_hinge": bool(search_cfg.include_hinge),
                    "hinge_quantiles": [float(v) for v in search_cfg.hinge_quantiles],
                    "unary_ops": [str(v) for v in search_cfg.unary_ops],
                    "nested_unary_patterns": [str(v) for v in search_cfg.nested_unary_patterns],
                    "enable_grad_residual_projection": bool(search_cfg.enable_grad_residual_projection),
                    "grad_projection_topk_focus": int(search_cfg.grad_projection_topk_focus),
                    "grad_projection_partner_pool": int(search_cfg.grad_projection_partner_pool),
                    "grad_projection_topk_partners": int(search_cfg.grad_projection_topk_partners),
                    "grad_projection_topk_unary": int(search_cfg.grad_projection_topk_unary),
                    "grad_projection_enable_pair_dictionary": bool(search_cfg.grad_projection_enable_pair_dictionary),
                    "grad_projection_min_abs_corr": float(search_cfg.grad_projection_min_abs_corr),
                    "grad_projection_max_generated": int(search_cfg.grad_projection_max_generated),
                    "enable_prune": bool(search_cfg.enable_prune),
                    "prune_rmse_tolerance": float(search_cfg.prune_rmse_tolerance),
                    "prune_max_removed_per_iter": int(search_cfg.prune_max_removed_per_iter),
                    "path_memory_enabled": bool(search_cfg.path_memory_enabled),
                    "path_memory_db_path": str(search_cfg.path_memory_db_path),
                    "path_memory_namespace": str(search_cfg.path_memory_namespace),
                    "path_memory_prior_bonus": float(search_cfg.path_memory_prior_bonus),
                    "path_memory_tabu_penalty": float(search_cfg.path_memory_tabu_penalty),
                    "path_memory_min_outcomes": int(search_cfg.path_memory_min_outcomes),
                    "path_memory_hard_tabu": bool(search_cfg.path_memory_hard_tabu),
                    "path_memory_hard_tabu_accept_rate": float(search_cfg.path_memory_hard_tabu_accept_rate),
                    "graph_cache_enabled": bool(search_cfg.graph_cache_enabled),
                    "graph_cache_max_value_entries": int(search_cfg.graph_cache_max_value_entries),
                    "graph_cache_max_derivative_entries": int(search_cfg.graph_cache_max_derivative_entries),
                    "graph_cache_backend": str(search_cfg.graph_cache_backend),
                    "graph_cache_db_path": str(search_cfg.graph_cache_db_path),
                    "graph_cache_namespace": str(search_cfg.graph_cache_namespace),
                    "graph_cache_persist_values": bool(search_cfg.graph_cache_persist_values),
                    "joint_bundle_enabled": bool(search_cfg.joint_bundle_enabled),
                    "joint_bundle_max_terms": int(search_cfg.joint_bundle_max_terms),
                    "joint_bundle_preselect_topk": int(search_cfg.joint_bundle_preselect_topk),
                    "joint_bundle_max_combos": int(search_cfg.joint_bundle_max_combos),
                    "joint_bundle_l1_alpha": float(search_cfg.joint_bundle_l1_alpha),
                    "joint_bundle_l1_iters": int(search_cfg.joint_bundle_l1_iters),
                    "inner_opt_enabled": bool(inner_opt_cfg.enabled),
                    "inner_opt_method": str(inner_opt_cfg.method),
                    "inner_opt_device": str(inner_opt_cfg.device),
                    "inner_opt_random_seed": int(inner_opt_cfg.random_seed),
                    "inner_opt_adam_steps": int(inner_opt_cfg.adam_steps),
                    "inner_opt_adam_lr": float(inner_opt_cfg.adam_lr),
                    "inner_opt_adam_weight_decay": float(inner_opt_cfg.adam_weight_decay),
                    "inner_opt_lbfgs_steps": int(inner_opt_cfg.lbfgs_steps),
                    "inner_opt_lbfgs_lr": float(inner_opt_cfg.lbfgs_lr),
                    "inner_opt_l2": float(inner_opt_cfg.l2),
                    "inner_opt_accept_rmse_tol": float(inner_opt_cfg.accept_rmse_tol),
                    "online_beam_enabled": bool(beam_cfg.enabled),
                    "online_beam_width": int(beam_cfg.width),
                    "online_bundle_size": int(beam_cfg.bundle_size),
                    "online_branches_per_beam": int(beam_cfg.branches_per_beam),
                    "online_beam_jitter": float(beam_cfg.jitter),
                    "online_early_stop_rounds": int(beam_cfg.early_stop_rounds),
                },
                "search_config_grouped": dict(grouped_cfg),
                "base_metrics": dict(search_res.base_metrics),
                "final_metrics": dict(search_res.final_metrics),
                "artifact_train_metrics": dict(artifact_train_metrics),
                "inner_opt": dict(inner_opt_result["info"]),
                "iterations": int(len(search_res.iterations)),
                "terms": int(len(search_res.genome)),
            },
        }

        if decision_log is not None:
            metadata["strategy"]["auto_decision"] = dict(decision_log)

        if bool(strategy_cfg.keep_search_trace):
            metadata["search_trace"] = {
                "score_trace": [float(v) for v in search_res.score_trace],
                "iterations": [dict(item) for item in search_res.iterations],
            }
        metadata["resume"] = {
            "enabled": bool(mode in {"resume", "warm_start", "incremental"} and seed_genome_override is not None),
            "mode": str(mode),
            "seed_source": parent_seed_source,
            "seed_kind": parent_seed_kind,
            "seed_terms": 0 if seed_genome_override is None else int(len(seed_genome_override)),
        }
        metadata["training_init"] = {
            "mode": str(mode),
            "parent_source": parent_seed_source,
            "parent_kind": parent_seed_kind,
        }
        selected_basis_rows = build_basis_term_rows(
            search_res.genome,
            feature_names=feature_names,
            scope="global",
        )
        basis_semantics = build_basis_semantics_payload(
            selected_basis_rows,
            source="symbolic_stagewise.final_genome",
            basis_scope="global",
            extra={
                "selected_mode": str(selected_mode),
                "search_status": "completed",
            },
        )
        basis_overlap_report = build_basis_overlap_report(
            selected_basis_rows,
            source="symbolic_stagewise.final_genome",
            extra={"selected_mode": str(selected_mode)},
        )
        assembler_budget = build_assembler_budget_payload(
            source="symbolic_stagewise.search_config",
            assembler_mode="budgeted_symbolic_regression",
            output_expression_count=int(max(1, m)),
            selected_basis_count=int(len(selected_basis_rows)),
            budget_axes={
                "beam_width": getattr(search_cfg, "auto_nested_beam_width", None),
                "max_terms": getattr(search_cfg, "max_added_terms", None),
                "max_depth": getattr(search_cfg, "max_expr_depth", None),
                "max_interaction_order": getattr(search_cfg, "max_arity", None),
                "max_piecewise_branches": 1,
            },
            extra={
                "candidate_keep_top": int(search_cfg.candidate_keep_top),
                "terms_after_search": int(len(search_res.genome)),
            },
        )
        metadata["selected_basis"] = list(selected_basis_rows)
        metadata["basis_semantics"] = dict(basis_semantics)
        metadata["basis_overlap_report"] = dict(basis_overlap_report)
        metadata["assembler_budget"] = dict(assembler_budget)
        metadata["symbolic"] = {
            "structure_engine": self._resolve_structure_engine().as_dict(),
            "selected_basis": list(selected_basis_rows),
            "basis_semantics": dict(basis_semantics),
            "basis_overlap_report": dict(basis_overlap_report),
            "assembler_budget": dict(assembler_budget),
        }
        family_spec = getattr(self, "symbolic_family_spec", None)
        if family_spec is not None and hasattr(family_spec, "as_dict"):
            metadata["symbolic_family"] = family_spec.as_dict()

        artifact = SymbolicSurrogateArtifact(
            artifact_id=str(self.config.artifact_id),
            genome=tuple(search_res.genome),
            parameter_values={},
            readout_weight=np.asarray(readout_weight, dtype=float),
            readout_bias=np.asarray(readout_bias, dtype=float),
            x_mean=np.mean(Xb, axis=0),
            x_std=np.std(Xb, axis=0) + 1e-8,
            residual_std=np.asarray(residual_std, dtype=float),
            feature_names=feature_names,
            target_names=target_names,
            pipeline_name=str(getattr(self.pipeline, "name", "identity")),
            pipeline_state=dict(getattr(self.pipeline, "state_dict")()),
            ood_z_threshold=float(self.config.ood_z_threshold),
            epsilon=float(self.config.epsilon),
            metadata=metadata,
        )
        if training_signature is None:
            return artifact, None

        signature_obj = coerce_training_signature(training_signature)
        trainer_state_payload = {
            "schema_version": 1,
            "trainer_name": str(self.name),
            "search_completed": True,
            "epoch_done": 0,
            "selected_mode": str(selected_mode),
            "requested_mode": str(requested_mode),
            "seed_source": parent_seed_source,
            "seed_kind": parent_seed_kind,
            "genome": self._copy_genome(search_res.genome),
            "parameter_values": {},
            "readout_weight": np.asarray(readout_weight, dtype=float),
            "readout_bias": np.asarray(readout_bias, dtype=float),
            "residual_std": np.asarray(residual_std, dtype=float),
            "feature_names": tuple(str(v) for v in feature_names),
            "target_names": tuple(str(v) for v in target_names),
            "base_metrics": dict(search_res.base_metrics),
            "final_metrics": dict(search_res.final_metrics),
            "artifact_train_metrics": dict(artifact_train_metrics),
            "inner_opt": dict(inner_opt_result["info"]),
            "search_config_grouped": dict(grouped_cfg),
            "search_iterations": [dict(item) for item in search_res.iterations],
            "search_score_trace": [float(v) for v in search_res.score_trace],
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
                "epoch_done": 0,
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





