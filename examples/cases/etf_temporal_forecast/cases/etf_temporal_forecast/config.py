"""Component registry aggregation for etf_temporal_forecast case."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

if __package__ in {None, ""}:
    import sys

    _BASE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_BASE))
    sys.path.insert(0, str(_BASE / "problem"))
    sys.path.insert(0, str(_BASE / "pipeline"))
    sys.path.insert(0, str(_BASE / "plugins"))

    from etf_temporal_problem import EtfTemporalProblem  # type: ignore
    from etf_feature_construction import EtfFeatureBuilder  # type: ignore
    from etf_report_plugin import EtfReportPlugin  # type: ignore
    from etf_observability_plugin import EtfObservabilityPlugin  # type: ignore
else:
    # Standard mlblack integration entry
    from mlblack.integrations.etf_temporal_forecast import (
        EtfTemporalForecastConfig,
        WalkForwardSpec,
        run_etf_temporal_forecast_multi_seed,
    )

    from .pipeline import EtfFeatureBuilder
    from .plugins import EtfObservabilityPlugin, EtfReportPlugin
    from .problem import EtfTemporalProblem

if __package__ in {None, ""}:
    # Re-export integration symbols lazily for top-level script compatibility.
    from mlblack.integrations.etf_temporal_forecast import (  # type: ignore
        EtfTemporalForecastConfig,
        WalkForwardSpec,
        run_etf_temporal_forecast_multi_seed,
    )


@dataclass(frozen=True)
class EtfCaseComponentSpec:
    """Case-level component descriptor for ETF temporal forecast."""

    dataset_url: str = str(
        Path(__file__).resolve().parents[1]
        / "cache"
        / "multi_etf_returns_momodel_kaggle.parquet"
    )
    dataset_label: str = "multi_etf_returns_momodel_kaggle"
    models: tuple[str, ...] = ("ridge", "hist_gradient_boosting")
    target_horizon: int = 1
    transaction_cost: float = 0.0005
    max_folds: int = 2
    seed: int = 42


# Plugin registry (for future capability registration)
PLUGINS_REGISTRY = {
    "report": EtfReportPlugin,
    "observability": EtfObservabilityPlugin,
}

# Problem/Pipeline registry (for future component discovery)
PROBLEM_REGISTRY = {
    "etf_temporal": EtfTemporalProblem,
}

PIPELINE_REGISTRY = {
    "etf_feature": EtfFeatureBuilder,
}

__all__ = [
    "EtfCaseComponentSpec",
    "EtfTemporalForecastConfig",
    "WalkForwardSpec",
    "run_etf_temporal_forecast_multi_seed",
    "PLUGINS_REGISTRY",
    "PROBLEM_REGISTRY",
    "PIPELINE_REGISTRY",
]
