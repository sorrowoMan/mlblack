from __future__ import annotations

from typing import Any

from core.orthogonal_source import OrthogonalSourceConfig, OrthogonalSourceLayer, OrthogonalSourceResult
from my_project.orthogonal_source_image_classification.config import ImageClassificationConfig
from my_project.orthogonal_source_image_classification.problem import ImageClassificationDataset
from my_project.orthogonal_source_image_classification.pipeline.representation_search import RepresentationFormulaSearchResult


def build_orthogonal_image_sources(
    dataset: ImageClassificationDataset,
    cfg: ImageClassificationConfig,
    representation: RepresentationFormulaSearchResult,
) -> OrthogonalSourceResult:
    layer = OrthogonalSourceLayer(
        OrthogonalSourceConfig(
            max_sources=int(cfg.max_sources),
            candidate_keep_top=int(cfg.candidate_keep_top),
            max_pair_abs_corr=float(cfg.max_pair_abs_corr),
            target_task="classification",
            min_abs_target_corr=0.015,
            include_raw=True,
            include_unary=False,
            include_pairwise=False,
            include_triple_ratio=False,
            include_hinge=False,
            include_exp_ratio=False,
        )
    )
    metadata: dict[str, Any] = dict(dataset.metadata)
    metadata["target_task"] = "classification"
    metadata["raw_observation_space"] = "flattened_8x8_pixels"
    metadata["orthogonal_input_space"] = "searched_objectified_image_representation"
    metadata["orthogonal_input_protocol"] = "searched_symbolic_representation_formula_pool"
    metadata["orthogonal_governance_mode"] = "identity_source_selection_no_secondary_formula_expansion"
    metadata["representation_formula_search"] = dict(representation.report)
    return layer.fit_transform(
        X_train=representation.selected_train,
        y_train=dataset.y_train,
        X_test=representation.selected_test,
        feature_names=representation.selected_feature_names,
        metadata=metadata,
    )


__all__ = ["build_orthogonal_image_sources"]
