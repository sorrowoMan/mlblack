from __future__ import annotations

from core.orthogonal_source import OrthogonalSourceConfig, OrthogonalSourceLayer, OrthogonalSourceResult

from my_project.etf_quant_interval_proxy.config import EtfQuantIntervalConfig
from my_project.etf_quant_interval_proxy.problem import EtfPanelDataset


def build_orthogonal_etf_sources(dataset: EtfPanelDataset, cfg: EtfQuantIntervalConfig) -> OrthogonalSourceResult:
    layer = OrthogonalSourceLayer(
        OrthogonalSourceConfig(
            max_sources=int(cfg.max_sources),
            candidate_keep_top=int(cfg.candidate_keep_top),
            max_pair_abs_corr=float(cfg.max_pair_abs_corr),
            min_abs_target_corr=float(cfg.min_abs_target_corr),
            include_raw=True,
            include_unary=True,
            include_pairwise=True,
            include_triple_ratio=False,
            include_hinge=True,
            include_exp_ratio=False,
            target_task="regression",
        )
    )
    return layer.fit_transform(
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_test=dataset.X_test,
        feature_names=dataset.feature_names,
        metadata=dict(dataset.metadata),
    )


__all__ = ["build_orthogonal_etf_sources"]
