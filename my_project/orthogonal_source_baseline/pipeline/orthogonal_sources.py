from __future__ import annotations

from core.orthogonal_source import OrthogonalSourceConfig, OrthogonalSourceLayer, OrthogonalSourceResult

from my_project.orthogonal_source_baseline.config import OrthogonalSourceBaselineConfig
from my_project.orthogonal_source_baseline.problem.dataset import ScenarioDataset


def build_orthogonal_sources(
    dataset: ScenarioDataset,
    cfg: OrthogonalSourceBaselineConfig,
) -> OrthogonalSourceResult:
    layer = OrthogonalSourceLayer(
        OrthogonalSourceConfig(
            max_sources=int(cfg.max_sources),
            max_pair_abs_corr=float(cfg.max_pair_abs_corr),
        )
    )
    return layer.fit_transform(
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_test=dataset.X_test,
        feature_names=dataset.feature_names,
        metadata=dataset.metadata,
    )


__all__ = ["build_orthogonal_sources"]
