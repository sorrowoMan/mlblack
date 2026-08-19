from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (  # type: ignore
        EtfCaseComponentSpec,
        EtfFeatureBuilder,
        EtfObservabilityPlugin,
        EtfReportPlugin,
        EtfTemporalProblem,
    )
    from pipeline.main import build_pipeline  # type: ignore
else:
    # Import from case-local components (standard scaffold layer)
    from .config import EtfCaseComponentSpec, EtfFeatureBuilder, EtfObservabilityPlugin, EtfReportPlugin, EtfTemporalProblem
    from .pipeline.main import build_pipeline

# Still use integration entry for execution (backward compatible)
from mlblack.integrations.etf_temporal_forecast import (
    DEFAULT_DATASET_URL,
    EtfTemporalForecastConfig,
    WalkForwardSpec,
    run_etf_temporal_forecast_multi_seed,
)


@dataclass
class EtfTemporalForecastRunner:
    """
    Canonical runner: orchestrates problem, pipeline, and plugins.
    
    New scaffold: case-local components expose cleaner interfaces.
    Backward compatible: still uses integration entry for execution.
    """

    problem: EtfTemporalProblem
    feature_builder: EtfFeatureBuilder
    cfg: EtfTemporalForecastConfig
    walkforward: WalkForwardSpec
    plugins: tuple[object, ...] = ()
    seeds: tuple[int, ...] = (42,)
    suite_id: str = "etf_temporal_forecast"
    output_dir: Path = Path("runs/etf_temporal_forecast")
    resource_context: Mapping[str, object] | None = None

    def set_resource_context(self, context):
        self.resource_context = dict(context or {})
        return self

    def run(self):
        """Execute via problem.evaluate() which delegates to integration entry."""
        result = self.problem.evaluate(
            seeds=self.seeds,
            context={"output_dir": str(self.output_dir), "max_folds": self.walkforward.max_folds},
        )

        for plugin in self.plugins:
            on_fit_end = getattr(plugin, "on_fit_end", None)
            if callable(on_fit_end):
                on_fit_end(self, {"output_dir": str(self.output_dir), "suite_id": self.suite_id}, result)

        return result


def build_solver(
    *,
    config=None,
    dataset_url: str = str(DEFAULT_DATASET_URL),
    dataset_label: str = "multi_etf_returns_momodel_kaggle",
    models: Sequence[str] = ("ridge", "hist_gradient_boosting"),
    seeds: Sequence[int] = (42,),
    suite_id: str = "etf_temporal_forecast",
    output_dir: str | Path = "runs/etf_temporal_forecast",
    walkforward: WalkForwardSpec | Mapping[str, Any] | None = None,
    resource_context: Mapping[str, object] | None = None,
    component_overrides: Mapping[str, object] | None = None,
):
    """Canonical Case assembly entry: orchestrates standard scaffold components."""

    del config

    # Assemble case-local components
    spec = EtfCaseComponentSpec(
        dataset_url=str(dataset_url),
        dataset_label=str(dataset_label),
        models=tuple(str(m) for m in models),
    )

    problem = EtfTemporalProblem(spec)
    feature_builder = build_pipeline(
        resource_context=resource_context,
        component_overrides=component_overrides,
    )
    plugins = (
        EtfReportPlugin(output_dir=output_dir, run_id=str(suite_id)),
        EtfObservabilityPlugin(output_dir=output_dir, run_id=str(suite_id)),
    )

    # Create runner with standard scaffold structure
    runner = EtfTemporalForecastRunner(
        problem=problem,
        feature_builder=feature_builder,
        cfg=EtfTemporalForecastConfig(
            dataset_url=str(dataset_url),
            dataset_label=str(dataset_label),
            models=tuple(str(item) for item in models),
        ),
        walkforward=WalkForwardSpec(**dict(walkforward)) if isinstance(walkforward, Mapping) else (walkforward or WalkForwardSpec()),
        plugins=plugins,
        seeds=tuple(int(seed) for seed in seeds),
        suite_id=str(suite_id),
        output_dir=Path(output_dir),
    )
    if resource_context is not None:
        runner.set_resource_context(resource_context)
    return runner
