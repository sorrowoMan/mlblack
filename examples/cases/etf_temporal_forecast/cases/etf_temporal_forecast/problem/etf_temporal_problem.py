"""ETF Temporal Forecast Problem: encapsulates dataset, evaluation, and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any, Mapping

from mlblack.integrations.etf_temporal_forecast import (
    run_etf_temporal_forecast_multi_seed,
    EtfTemporalForecastConfig,
    WalkForwardSpec,
)


@dataclass(frozen=True)
class EtfTemporalForecastSpec:
    """Problem specification: ETF temporal forecasting via walk-forward evaluation."""

    dataset_url: str
    dataset_label: str = "multi_etf_returns_momodel_kaggle"
    models: tuple[str, ...] = ("ridge", "hist_gradient_boosting")
    target_horizon: int = 1
    transaction_cost: float = 0.0005


class EtfTemporalProblem:
    """
    ETF temporal forecast problem.

    Encapsulates:
    - Dataset loading and panel construction (delegated to integration)
    - Walk-forward evaluation via multi-seed runs
    - Metric aggregation (RMSE, rank IC, Sharpe proxy, drawdown, turnover)

    This class is the unique stable interface that consumes data and produces feedback.
    """

    def __init__(self, spec: EtfTemporalForecastSpec | Mapping[str, Any]):
        if isinstance(spec, Mapping):
            spec = EtfTemporalForecastSpec(**spec)
        self.spec = spec
        self.name = "etf_temporal_forecast"

    def evaluate(
        self,
        candidate: Mapping[str, Any] | None = None,
        seeds: tuple[int, ...] = (42,),
        context: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Evaluate candidate (model configuration or lane weights).

        Args:
            candidate: Model spec or lane weights dict (lane_bundle).
                      If None, uses default models from spec.
            seeds: Random seeds for reproducibility.
            context: Runtime context (folds, output_dir, etc.).
            **kwargs: Additional override parameters.

        Returns:
            Feedback dict with objectives, constraints, and full summary.
        """
        # Construct configuration
        cfg = EtfTemporalForecastConfig(
            dataset_url=str(self.spec.dataset_url),
            dataset_label=str(self.spec.dataset_label),
            models=self.spec.models,
            target_horizon=int(self.spec.target_horizon),
            transaction_cost=float(self.spec.transaction_cost),
        )

        # Construct walk-forward spec (can be overridden via context)
        wf_overrides: dict[str, Any] = {}
        if context is not None:
            for key in ("max_folds", "test_size", "min_train_size"):
                if key in context:
                    wf_overrides[key] = context[key]
        wf_spec = replace(WalkForwardSpec(), **wf_overrides) if wf_overrides else WalkForwardSpec()

        # Run evaluation (delegate to integration entry)
        result = run_etf_temporal_forecast_multi_seed(
            cfg=cfg,
            walkforward=wf_spec,
            seeds=seeds,
            suite_id=self.name,
            output_dir=context.get("output_dir", "runs/etf_temporal_forecast")
            if context
            else "runs/etf_temporal_forecast",
            potential_params_override=candidate,  # lane weights, top-k, blend mode, etc.
        )

        # Extract aggregate metrics
        agg = result.summary.get("aggregate", {})

        # Construct feedback in unified format
        return {
            "objectives": {
                # Lower is better
                "composite_test_rmse": float(agg.get("composite_test_rmse_mean", 1e9)),
                # Higher is better (negate for minimization if needed)
                "composite_rank_ic": float(agg.get("composite_rank_ic_mean", 0.0)),
                "composite_hit_rate": float(agg.get("composite_hit_rate_mean", 0.0)),
                "composite_net_sharpe_proxy": float(
                    agg.get("composite_net_sharpe_proxy_mean", 0.0)
                ),
            },
            "constraints": {
                # Lower is better (hard constraints)
                "composite_max_drawdown_abs": float(
                    agg.get("composite_max_drawdown_abs_mean", 1e9)
                ),
                "composite_turnover_proxy": float(
                    agg.get("composite_turnover_proxy_mean", 1e9)
                ),
            },
            "metric_stability": {
                "composite_rank_ic_std": float(agg.get("composite_rank_ic_std", 0.0)),
            },
            "summary": result.summary,
            "output_dir": str(result.output_dir),
        }
