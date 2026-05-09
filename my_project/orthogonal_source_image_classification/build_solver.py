from __future__ import annotations

from my_project.orthogonal_source_image_classification.config import ImageClassificationConfig
from my_project.orthogonal_source_image_classification.orchestration import run_suite


def build_orthogonal_source_image_classification_components(
    cfg: ImageClassificationConfig | None = None,
) -> dict[str, object]:
    config = cfg or ImageClassificationConfig()
    return {
        "config": config,
        "runner": run_suite,
        "protocol": "raw_pixels -> searchable_symbolic_phi_pool -> selected_representation_objects -> orthogonal_sources -> classification_head",
        "representation_component": "pipeline.representation_search.search_image_representation_formulas",
        "orthogonal_component": "core.orthogonal_source.OrthogonalSourceLayer(target_task=classification)",
        "problem_surface": "sklearn_digits_symbolic_representation_formula_search",
        "outer_solver_hook": "nsgablack can replace the candidate-pool selection policy with a full representation-program outer solver",
    }


__all__ = ["ImageClassificationConfig", "build_orthogonal_source_image_classification_components"]
