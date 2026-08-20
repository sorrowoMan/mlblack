from __future__ import annotations

import importlib
from pathlib import Path

from mlblack.integrations import LearningSolver
from mlblack.integrations.etf_temporal_forecast import EtfTemporalForecastResult
from nsgablack.adapters import FixedCandidateAdapter


def test_etf_case_executes_as_one_learning_solver_evaluation(monkeypatch, tmp_path) -> None:
    problem_module = importlib.import_module(
        "examples.cases.etf_temporal_forecast.cases.etf_temporal_forecast."
        "problem.etf_temporal_problem"
    )
    builder_module = importlib.import_module(
        "examples.cases.etf_temporal_forecast.cases.etf_temporal_forecast.build_solver"
    )
    calls: list[dict] = []

    def fake_run(**kwargs):
        calls.append(dict(kwargs))
        return EtfTemporalForecastResult(
            summary={
                "suite_id": "etf_test",
                "case": "etf_temporal_forecast",
                "fold_count": 2,
                "dataset": {"rows": 10, "assets": 2},
                "aggregate": {
                    "composite_test_rmse_mean": 0.2,
                    "composite_rank_ic_mean": 0.3,
                    "composite_rank_ic_std": 0.01,
                    "composite_hit_rate_mean": 0.6,
                    "composite_net_sharpe_proxy_mean": 1.2,
                    "composite_max_drawdown_abs_mean": 0.1,
                    "composite_turnover_proxy_mean": 0.4,
                },
            },
            output_dir=Path(tmp_path),
        )

    monkeypatch.setattr(
        problem_module,
        "run_etf_temporal_forecast_multi_seed",
        fake_run,
    )
    solver = builder_module.build_solver(
        dataset_url=str(tmp_path / "returns.parquet"),
        suite_id="etf_test",
        output_dir=tmp_path,
        resource_context={
            "threads": 2,
            "namespace": "tests.etf",
            "grant": {"threads": 2, "workers": 1},
        },
        component_overrides={"plugins": ()},
    )

    result = solver.fit()

    assert isinstance(solver, LearningSolver)
    assert isinstance(solver.adapter, FixedCandidateAdapter)
    assert result.report["optimization_runtime"]["steps_executed"] == 1
    assert result.best_feedback.objectives.tolist() == [0.2, -0.3, -1.2, 0.1, 0.4]
    assert len(calls) == 1
    assert calls[0]["resource_context"]["grant"]["threads"] == 2
    assert calls[0]["panel_builder"] is solver.semantic_problem.feature_builder


def test_ml_core_has_no_nsgablack_imports() -> None:
    core_root = Path(__file__).resolve().parents[1] / "core"
    offenders = []
    for path in core_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "from nsgablack" in text or "import nsgablack" in text:
            offenders.append(path.relative_to(core_root).as_posix())
    assert offenders == []
