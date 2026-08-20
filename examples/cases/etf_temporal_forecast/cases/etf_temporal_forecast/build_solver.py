from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from nsgablack.core import BudgetController

from mlblack.integrations import build_learning_solver, build_optimization_adapter
from mlblack.integrations.etf_temporal_forecast import (
    DEFAULT_DATASET_URL,
    WalkForwardSpec,
)

try:
    from .pipeline import EtfTemporalRepresentation, FeatureBuildSpec, build_pipeline
    from .plugins import EtfObservabilityPlugin, EtfReportPlugin
    from .problem import EtfTemporalForecastSpec, EtfTemporalProblem
except ImportError:  # direct canonical CLI execution
    from pipeline import (  # type: ignore
        EtfTemporalRepresentation,
        FeatureBuildSpec,
        build_pipeline,
    )
    from plugins import EtfObservabilityPlugin, EtfReportPlugin  # type: ignore
    from problem import EtfTemporalForecastSpec, EtfTemporalProblem  # type: ignore


def build_solver(
    config=None,
    *,
    dataset_url: str = str(DEFAULT_DATASET_URL),
    dataset_label: str = "multi_etf_returns_momodel_kaggle",
    models: Sequence[str] = ("ridge", "hist_gradient_boosting"),
    seeds: Sequence[int] = (42,),
    suite_id: str = "etf_temporal_forecast",
    output_dir: str | Path = "runs/etf_temporal_forecast",
    walkforward: WalkForwardSpec | Mapping[str, Any] | None = None,
    target_horizon: int = 1,
    transaction_cost: float = 0.0005,
    resource_context: Mapping[str, Any] | None = None,
    component_overrides: Mapping[str, Any] | None = None,
):
    """Build one canonical LearningSolver for the ETF evaluation procedure."""

    payload = dict(config or {}) if isinstance(config, Mapping) else {}
    overrides = dict(component_overrides or {})
    dataset_url = str(payload.get("dataset_url", dataset_url))
    dataset_label = str(payload.get("dataset_label", dataset_label))
    models = tuple(str(item) for item in payload.get("models", models))
    seeds = tuple(int(item) for item in payload.get("seeds", seeds))
    suite_id = str(payload.get("suite_id", suite_id))
    output_dir = Path(payload.get("output_dir", output_dir))
    target_horizon = int(payload.get("target_horizon", target_horizon))
    transaction_cost = float(payload.get("transaction_cost", transaction_cost))

    feature_builder = overrides.pop("feature_builder", None)
    if feature_builder is None:
        feature_builder = build_pipeline(
            spec=FeatureBuildSpec(target_horizon=target_horizon),
            resource_context=resource_context,
            component_overrides=overrides.pop("pipeline", None),
        )
    candidate = overrides.pop("candidate", None)
    representation = overrides.pop("representation", None) or EtfTemporalRepresentation(
        candidate
    )
    problem = overrides.pop("problem", None) or EtfTemporalProblem(
        EtfTemporalForecastSpec(
            dataset_url=dataset_url,
            dataset_label=dataset_label,
            models=models,
            target_horizon=target_horizon,
            transaction_cost=transaction_cost,
        ),
        walkforward=walkforward,
        seeds=seeds,
        suite_id=suite_id,
        output_dir=str(output_dir),
        feature_builder=feature_builder,
    )
    adapter = overrides.pop("adapter", None) or build_optimization_adapter(
        "evaluation.fixed"
    )
    plugin_values = overrides.pop(
        "plugins",
        (
            EtfReportPlugin(output_dir=output_dir, run_id=suite_id),
            EtfObservabilityPlugin(output_dir=output_dir, run_id=suite_id),
        ),
    )
    if overrides:
        raise ValueError(f"unsupported ETF component overrides: {sorted(overrides)}")

    solver = build_learning_solver(
        problem=problem,
        representation=representation,
        adapter=adapter,
        run_name=suite_id,
        resource_context=resource_context,
    )
    for plugin in tuple(plugin_values or ()):
        solver.add_capability(plugin)
    solver.register_controller(
        BudgetController(max_generations=1, name=f"{suite_id}.one_evaluation")
    )
    return solver


__all__ = ["build_solver"]
