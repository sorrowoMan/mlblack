"""ETF Temporal Forecast Problem: encapsulates dataset, evaluation, and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from mlblack.core import Feedback, LearningProblem, UnknownState

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


@dataclass(frozen=True)
class EtfTemporalCandidate:
    """One explicit forecasting procedure evaluated by the fixed Adapter."""

    potential_params_override: Mapping[str, Any] | None = None


class EtfTemporalProblem(LearningProblem):
    """
    ETF temporal forecast problem.

    Encapsulates:
    - Dataset loading and panel construction (delegated to integration)
    - Walk-forward evaluation via multi-seed runs
    - Metric aggregation (RMSE, rank IC, Sharpe proxy, drawdown, turnover)

    This class is the unique stable interface that consumes data and produces feedback.
    """

    objective_count = 5

    def __init__(
        self,
        spec: EtfTemporalForecastSpec | Mapping[str, Any],
        *,
        walkforward: WalkForwardSpec | Mapping[str, Any] | None = None,
        seeds: tuple[int, ...] = (42,),
        suite_id: str = "etf_temporal_forecast",
        output_dir: str = "runs/etf_temporal_forecast",
        feature_builder: Any | None = None,
    ):
        if isinstance(spec, Mapping):
            spec = EtfTemporalForecastSpec(**spec)
        self.spec = spec
        self.name = "etf_temporal_forecast"
        self.walkforward = (
            WalkForwardSpec(**dict(walkforward))
            if isinstance(walkforward, Mapping)
            else (walkforward or WalkForwardSpec())
        )
        self.seeds = tuple(int(seed) for seed in seeds)
        self.suite_id = str(suite_id)
        self.output_dir = str(output_dir)
        self.feature_builder = feature_builder
        self.last_result = None

    def evaluate(
        self,
        model: EtfTemporalCandidate,
        state: UnknownState,
        context: Mapping[str, Any],
    ) -> Feedback:
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
        del state
        if not isinstance(model, EtfTemporalCandidate):
            raise TypeError("ETF representation must decode EtfTemporalCandidate")
        cfg = EtfTemporalForecastConfig(
            dataset_url=str(self.spec.dataset_url),
            dataset_label=str(self.spec.dataset_label),
            models=self.spec.models,
            target_horizon=int(self.spec.target_horizon),
            transaction_cost=float(self.spec.transaction_cost),
        )

        result = run_etf_temporal_forecast_multi_seed(
            cfg=cfg,
            walkforward=self.walkforward,
            seeds=self.seeds,
            suite_id=self.suite_id,
            output_dir=self.output_dir,
            potential_params_override=model.potential_params_override,
            resource_context=context.get("resource_context"),
            panel_builder=self.feature_builder,
        )
        self.last_result = result

        # Extract aggregate metrics
        agg = result.summary.get("aggregate", {})

        rmse = float(agg.get("composite_test_rmse_mean", 1e9))
        rank_ic = float(agg.get("composite_rank_ic_mean", 0.0))
        sharpe = float(agg.get("composite_net_sharpe_proxy_mean", 0.0))
        drawdown = float(agg.get("composite_max_drawdown_abs_mean", 1e9))
        turnover = float(agg.get("composite_turnover_proxy_mean", 1e9))
        return Feedback(
            objectives=np.asarray([rmse, -rank_ic, -sharpe, drawdown, turnover]),
            constraints=np.zeros(0, dtype=float),
            loss=rmse,
            metrics={
                "etf.rmse": rmse,
                "etf.rank_ic": rank_ic,
                "etf.rank_ic_std": float(agg.get("composite_rank_ic_std", 0.0)),
                "etf.hit_rate": float(agg.get("composite_hit_rate_mean", 0.0)),
                "etf.sharpe_proxy": sharpe,
                "etf.max_drawdown_abs": drawdown,
                "etf.turnover_proxy": turnover,
                "etf.fold_count": int(result.summary.get("fold_count", 0)),
            },
            signals={
                "primary_objective": "etf.rmse",
                "output_dir": str(result.output_dir),
                "procedure": "walk_forward_multi_seed",
            },
        )

    def build_model_artifact(
        self,
        model: EtfTemporalCandidate,
        context: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        del context
        return {
            "kind": "etf_temporal_forecast_procedure",
            "candidate": {
                "potential_params_override": model.potential_params_override,
            },
            "summary": None if self.last_result is None else dict(self.last_result.summary),
            "output_dir": None if self.last_result is None else str(self.last_result.output_dir),
        }


__all__ = ["EtfTemporalCandidate", "EtfTemporalForecastSpec", "EtfTemporalProblem"]
