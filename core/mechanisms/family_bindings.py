from __future__ import annotations

from typing import Sequence

from .protocols import (
    AggregationProtocol,
    MechanismProtocolBase,
    SampleWeightingProtocol,
    SamplingProtocol,
    StateSignalViewProtocol,
)


def _build_tree_bagging_bindings(
    *,
    family: str,
    sampling_notes: str,
    state_notes: str,
) -> tuple[MechanismProtocolBase, ...]:
    return (
        SamplingProtocol(
            binding_level="bound",
            required_fields=("data_ref",),
            optional_fields=("feature_subspace_ref", "sample_index_ref"),
            notes=sampling_notes,
            metadata={"family": family, "selection_axes": ("row", "feature")},
        ),
        SampleWeightingProtocol(
            binding_level="optional",
            required_fields=("data_ref",),
            optional_fields=("sample_weight_ref", "difficulty_score_ref"),
            notes=f"{family} can consume sample weights without making weighting part of the family definition.",
            metadata={"family": family},
        ),
        StateSignalViewProtocol(
            binding_level="optional",
            optional_fields=("prediction_ref", "uncertainty_ref", "oob_score_ref", "per_sample_loss_ref"),
            provides_fields=("prediction_ref", "uncertainty_ref"),
            signal_names=("prediction", "uncertainty", "oob_score", "loss"),
            notes=state_notes,
            metadata={"family": family},
        ),
        AggregationProtocol(
            binding_level="defining",
            required_fields=("local_output_ref",),
            provides_fields=("aggregated_output_ref",),
            aggregation_mode="mean_vote",
            notes=f"{family} identity is defined by aggregating multiple tree outputs into one ensemble prediction.",
            metadata={"family": family},
        ),
    )


def build_linear_family_mechanism_bindings() -> tuple[MechanismProtocolBase, ...]:
    return (
        SamplingProtocol(
            binding_level="optional",
            required_fields=("data_ref",),
            optional_fields=("sample_index_ref", "batch_index_ref"),
            notes="Linear family can consume sampled data views but does not require them to define the family.",
            metadata={"family": "linear", "selection_axes": ("row", "feature")},
        ),
        SampleWeightingProtocol(
            binding_level="optional",
            required_fields=("data_ref",),
            optional_fields=("difficulty_score_ref", "sample_weight_ref"),
            notes="Weighted least squares or curriculum weighting is an enhancement, not the definition of linear fitting.",
            metadata={"family": "linear"},
        ),
        StateSignalViewProtocol(
            binding_level="optional",
            optional_fields=("prediction_ref", "per_sample_loss_ref", "residual_ref"),
            provides_fields=("prediction_ref", "residual_ref"),
            signal_names=("prediction", "loss", "residual"),
            notes="Linear family may expose prediction or residual views for downstream dynamic policies.",
            metadata={"family": "linear"},
        ),
        AggregationProtocol(
            binding_level="optional",
            required_fields=("local_output_ref",),
            provides_fields=("aggregated_output_ref",),
            aggregation_mode="mean",
            notes="Linear family usually has a single model output, so aggregation is optional ensemble behavior.",
            metadata={"family": "linear"},
        ),
    )


def build_tree_boosting_family_mechanism_bindings() -> tuple[MechanismProtocolBase, ...]:
    return (
        SamplingProtocol(
            binding_level="bound",
            required_fields=("data_ref",),
            optional_fields=("sample_weight_ref", "feature_subspace_ref"),
            notes="Tree boosting commonly binds row/feature subsampling into the trainer preset.",
            metadata={"family": "tree_boosting", "selection_axes": ("row", "feature")},
        ),
        SampleWeightingProtocol(
            binding_level="optional",
            required_fields=("data_ref",),
            optional_fields=("sample_weight_ref", "difficulty_score_ref"),
            notes="External sample weighting is supported but does not define boosting identity.",
            metadata={"family": "tree_boosting"},
        ),
        StateSignalViewProtocol(
            binding_level="defining",
            required_fields=("prediction_ref",),
            optional_fields=("per_sample_loss_ref", "gradient_ref", "hessian_ref"),
            provides_fields=("prediction_ref", "gradient_ref", "hessian_ref"),
            signal_names=("prediction", "loss", "gradient", "hessian"),
            notes="Boosting rounds fundamentally depend on current prediction state and residual-like signals.",
            metadata={"family": "tree_boosting"},
        ),
        AggregationProtocol(
            binding_level="defining",
            required_fields=("local_output_ref",),
            provides_fields=("aggregated_output_ref",),
            aggregation_mode="additive",
            notes="Boosting identity is defined by additive aggregation of weak learners.",
            metadata={"family": "tree_boosting"},
        ),
    )


def build_random_forest_mechanism_bindings() -> tuple[MechanismProtocolBase, ...]:
    return _build_tree_bagging_bindings(
        family="random_forest",
        sampling_notes="Bootstrap row sampling and feature subsampling are typical bound mechanisms of random forest presets.",
        state_notes="State signals are useful for active policies but not required for the family definition.",
    )


def build_extra_trees_mechanism_bindings() -> tuple[MechanismProtocolBase, ...]:
    return _build_tree_bagging_bindings(
        family="extra_trees",
        sampling_notes="Extra trees binds random split selection and optional row sampling into the ensemble preset.",
        state_notes="State signals are helpful for diagnostics or active routing but are not required to define extra trees.",
    )


def build_bagging_mechanism_bindings() -> tuple[MechanismProtocolBase, ...]:
    return _build_tree_bagging_bindings(
        family="bagging",
        sampling_notes="Bagging binds bootstrap or subset sampling directly into the ensemble preset.",
        state_notes="Bagging may expose prediction or OOB views, but those runtime signals are not defining.",
    )


def build_neural_family_mechanism_bindings() -> tuple[MechanismProtocolBase, ...]:
    return (
        SamplingProtocol(
            binding_level="optional",
            required_fields=("data_ref",),
            optional_fields=("batch_index_ref", "sample_index_ref"),
            notes="Mini-batch or subset sampling strengthens training efficiency but does not define neural family identity.",
            metadata={"family": "neural"},
        ),
        SampleWeightingProtocol(
            binding_level="optional",
            required_fields=("data_ref",),
            optional_fields=("sample_weight_ref", "difficulty_score_ref", "uncertainty_ref"),
            notes="Curriculum learning, hard example mining, and focal-style weighting are neural enhancements.",
            metadata={"family": "neural"},
        ),
        StateSignalViewProtocol(
            binding_level="optional",
            optional_fields=("prediction_ref", "per_sample_loss_ref", "uncertainty_ref", "gradient_norm_ref"),
            provides_fields=("prediction_ref", "per_sample_loss_ref", "gradient_norm_ref"),
            signal_names=("prediction", "loss", "uncertainty", "gradient_norm"),
            notes="Advanced neural policies often consume runtime state signals, but the family can run without them.",
            metadata={"family": "neural"},
        ),
        AggregationProtocol(
            binding_level="optional",
            required_fields=("local_output_ref",),
            provides_fields=("aggregated_output_ref",),
            aggregation_mode="mixture_mean",
            notes="Aggregation becomes relevant for ensembles, mixtures, or residual stacking, not plain neural backbones.",
            metadata={"family": "neural"},
        ),
    )


def build_symbolic_family_mechanism_bindings() -> tuple[MechanismProtocolBase, ...]:
    return (
        SamplingProtocol(
            binding_level="optional",
            required_fields=("data_ref",),
            optional_fields=("sample_index_ref", "candidate_subset_ref", "difficulty_score_ref"),
            notes="Symbolic family can consume sampled views, but sample-level sampling does not define symbolic identity by itself.",
            metadata={"family": "symbolic"},
        ),
        SampleWeightingProtocol(
            binding_level="optional",
            required_fields=("data_ref",),
            optional_fields=("sample_weight_ref", "difficulty_score_ref", "residual_ref"),
            notes="Residual-aware sample weighting is a symbolic enhancement, not the definition of symbolic search.",
            metadata={"family": "symbolic"},
        ),
        StateSignalViewProtocol(
            binding_level="bound",
            optional_fields=("prediction_ref", "per_sample_loss_ref", "residual_ref", "gradient_norm_ref"),
            provides_fields=("prediction_ref", "residual_ref", "gradient_norm_ref"),
            signal_names=("prediction", "loss", "residual", "gradient_norm"),
            notes="Many symbolic search refinements bind to residual or gradient-style runtime views.",
            metadata={"family": "symbolic"},
        ),
        AggregationProtocol(
            binding_level="optional",
            required_fields=("local_output_ref",),
            optional_fields=("global_output_ref", "regime_output_ref"),
            provides_fields=("aggregated_output_ref",),
            aggregation_mode="shared_backbone_plus_residual",
            notes="Aggregation becomes important for shared-backbone, regime-residual, or piecewise symbolic compositions.",
            metadata={"family": "symbolic"},
        ),
    )


def build_adaboost_mechanism_bindings() -> tuple[MechanismProtocolBase, ...]:
    return (
        SamplingProtocol(
            binding_level="optional",
            required_fields=("data_ref",),
            optional_fields=("sample_index_ref",),
            notes="AdaBoost may sample, but its core identity does not rely on plain subsampling.",
            metadata={"family": "adaboost"},
        ),
        SampleWeightingProtocol(
            binding_level="defining",
            required_fields=("data_ref", "prediction_ref"),
            optional_fields=("per_sample_loss_ref",),
            provides_fields=("sample_weight_ref",),
            notes="Sample reweighting is one of the defining mechanisms of AdaBoost.",
            metadata={"family": "adaboost"},
        ),
        StateSignalViewProtocol(
            binding_level="bound",
            required_fields=("prediction_ref",),
            optional_fields=("per_sample_loss_ref", "misclassified_mask_ref"),
            provides_fields=("prediction_ref", "per_sample_loss_ref"),
            signal_names=("prediction", "loss", "misclassified_mask"),
            notes="AdaBoost requires current weak-learner error views to update sample weights.",
            metadata={"family": "adaboost"},
        ),
        AggregationProtocol(
            binding_level="defining",
            required_fields=("local_output_ref",),
            provides_fields=("aggregated_output_ref",),
            aggregation_mode="weighted_additive",
            notes="AdaBoost identity is defined by weighted additive aggregation of weak learners.",
            metadata={"family": "adaboost"},
        ),
    )


def serialize_family_bindings(
    values: Sequence[MechanismProtocolBase],
) -> list[dict[str, object]]:
    return [value.as_dict() for value in tuple(values)]


__all__ = [
    "build_adaboost_mechanism_bindings",
    "build_bagging_mechanism_bindings",
    "build_extra_trees_mechanism_bindings",
    "build_linear_family_mechanism_bindings",
    "build_neural_family_mechanism_bindings",
    "build_random_forest_mechanism_bindings",
    "build_symbolic_family_mechanism_bindings",
    "build_tree_boosting_family_mechanism_bindings",
    "serialize_family_bindings",
]
