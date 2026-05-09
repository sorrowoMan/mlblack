from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from config import FlowAssemblySpec, NumericizerSpec, TrainerAssemblySpec
from core.common.contracts import ProcessedDataset
from core.execution import ExecutionBudgetError, ExecutionResourceOffer
from project.scaffold import _build_sqlalchemy_url, build_scaffold_spec, run_project_scaffold
from workflow import (
    ModelSpec,
    SemanticTrainFlowSpec,
    TrainDataBundle,
    run_semantic_portfolio_flow,
)


class TestSemanticPortfolioRuntime(unittest.TestCase):
    def test_semantic_portfolio_picks_lower_rmse_model(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(140, 3))
        y = (2.2 * X[:, 0] - 0.3 * X[:, 1] + 0.05 * rng.normal(size=140)).reshape(-1, 1)

        train = ProcessedDataset(
            X_train=X[:100],
            y_train=y[:100],
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
        )
        test = ProcessedDataset(
            X_train=X[100:],
            y_train=y[100:],
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
        )
        bundle = TrainDataBundle(train=train, test=test)

        spec = SemanticTrainFlowSpec(
            assembly=FlowAssemblySpec(
                trainer=TrainerAssemblySpec(trainer_key="ridge", trainer_params={"l2": 0.0}),
                numericizer=NumericizerSpec(key="default", params={}),
            ),
            eval_splits=("train", "test"),
            save_artifact=False,
            save_report=False,
            run_name="portfolio_runtime_smoke",
        )

        portfolio = run_semantic_portfolio_flow(
            bundle,
            spec=spec,
            model_specs=(
                ModelSpec(model_id="good", feature_names=("x0", "x1"), strict=True),
                ModelSpec(model_id="bad", feature_names=("x2",), strict=True),
            ),
        )

        self.assertEqual(int(portfolio.summary.get("portfolio_size", 0)), 2)
        self.assertEqual(str(portfolio.summary.get("best_model_id")), "good")
        self.assertIn("good", portfolio.runs)
        self.assertIn("bad", portfolio.runs)

        rmse_good = float(portfolio.runs["good"].metrics["test"]["rmse"])
        rmse_bad = float(portfolio.runs["bad"].metrics["test"]["rmse"])
        self.assertLess(rmse_good, rmse_bad)

    def test_semantic_portfolio_thread_mode_runs(self) -> None:
        rng = np.random.default_rng(9)
        X = rng.normal(size=(120, 3))
        y = (1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.03 * rng.normal(size=120)).reshape(-1, 1)

        train = ProcessedDataset(
            X_train=X[:90],
            y_train=y[:90],
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
        )
        test = ProcessedDataset(
            X_train=X[90:],
            y_train=y[90:],
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
        )
        bundle = TrainDataBundle(train=train, test=test)

        spec = SemanticTrainFlowSpec(
            assembly=FlowAssemblySpec(
                trainer=TrainerAssemblySpec(trainer_key="ridge", trainer_params={"l2": 0.0}),
                numericizer=NumericizerSpec(key="default", params={}),
            ),
            eval_splits=("train", "test"),
            save_artifact=False,
            save_report=False,
            run_name="portfolio_thread_smoke",
            portfolio_parallel_mode="thread",
            portfolio_max_workers=2,
        )

        portfolio = run_semantic_portfolio_flow(
            bundle,
            spec=spec,
            model_specs=(
                ModelSpec(model_id="good", feature_names=("x0", "x1"), strict=True),
                ModelSpec(model_id="bad", feature_names=("x2",), strict=True),
            ),
        )

        runtime = dict(portfolio.summary.get("runtime", {}))
        self.assertEqual(str(runtime.get("parallel_mode")), "thread")
        self.assertEqual(int(runtime.get("max_workers", 0)), 2)
        self.assertEqual(str(portfolio.summary.get("best_model_id")), "good")

    def test_semantic_portfolio_execution_spec_overrides_legacy_runtime_flags(self) -> None:
        rng = np.random.default_rng(10)
        X = rng.normal(size=(120, 3))
        y = (1.1 * X[:, 0] - 0.7 * X[:, 1] + 0.03 * rng.normal(size=120)).reshape(-1, 1)

        train = ProcessedDataset(
            X_train=X[:90],
            y_train=y[:90],
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
        )
        test = ProcessedDataset(
            X_train=X[90:],
            y_train=y[90:],
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
        )
        bundle = TrainDataBundle(train=train, test=test)

        spec = SemanticTrainFlowSpec(
            assembly=FlowAssemblySpec(
                trainer=TrainerAssemblySpec(trainer_key="ridge", trainer_params={"l2": 0.0}),
                numericizer=NumericizerSpec(key="default", params={}),
            ),
            execution={
                "backend": "thread",
                "max_workers": 2,
                "fail_fast": False,
                "gpu_strategy": "none",
                "gpu_devices": [],
                "default_device": None,
            },
            eval_splits=("train", "test"),
            save_artifact=False,
            save_report=False,
            run_name="portfolio_execution_spec_smoke",
            portfolio_parallel_mode="serial",
            portfolio_max_workers=1,
        )

        portfolio = run_semantic_portfolio_flow(
            bundle,
            spec=spec,
            model_specs=(
                ModelSpec(model_id="good", feature_names=("x0", "x1"), strict=True),
                ModelSpec(model_id="bad", feature_names=("x2",), strict=True),
            ),
        )

        runtime = dict(portfolio.summary.get("runtime", {}))
        declared_execution = dict(runtime.get("declared_execution", {}))
        self.assertEqual(str(runtime.get("parallel_mode")), "thread")
        self.assertEqual(int(runtime.get("max_workers", 0)), 2)
        self.assertEqual(str(declared_execution.get("backend")), "thread")
        self.assertFalse(bool(declared_execution.get("fail_fast")))

    def test_semantic_portfolio_process_mode_runs(self) -> None:
        rng = np.random.default_rng(11)
        X = rng.normal(size=(120, 3))
        y = (1.0 * X[:, 0] + 0.5 * X[:, 1] + 0.05 * rng.normal(size=120)).reshape(-1, 1)

        train = ProcessedDataset(
            X_train=X[:90],
            y_train=y[:90],
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
        )
        test = ProcessedDataset(
            X_train=X[90:],
            y_train=y[90:],
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
        )
        bundle = TrainDataBundle(train=train, test=test)

        spec = SemanticTrainFlowSpec(
            assembly=FlowAssemblySpec(
                trainer=TrainerAssemblySpec(trainer_key="ridge", trainer_params={"l2": 0.0}),
                numericizer=NumericizerSpec(key="default", params={}),
            ),
            eval_splits=("train", "test"),
            save_artifact=False,
            save_report=False,
            run_name="portfolio_process_smoke",
            portfolio_parallel_mode="process",
            portfolio_max_workers=2,
        )

        portfolio = run_semantic_portfolio_flow(
            bundle,
            spec=spec,
            model_specs=(
                ModelSpec(model_id="good", feature_names=("x0", "x1"), strict=True),
                ModelSpec(model_id="bad", feature_names=("x2",), strict=True),
            ),
        )

        runtime = dict(portfolio.summary.get("runtime", {}))
        self.assertEqual(str(runtime.get("parallel_mode")), "process")
        self.assertEqual(int(runtime.get("max_workers", 0)), 2)
        self.assertEqual(str(portfolio.summary.get("best_model_id")), "good")

    def test_semantic_portfolio_round_robin_assigns_devices(self) -> None:
        rng = np.random.default_rng(21)
        X = rng.normal(size=(100, 3))
        y = (1.3 * X[:, 0] - 0.2 * X[:, 2] + 0.05 * rng.normal(size=100)).reshape(-1, 1)

        train = ProcessedDataset(
            X_train=X[:80],
            y_train=y[:80],
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
        )
        test = ProcessedDataset(
            X_train=X[80:],
            y_train=y[80:],
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
        )
        bundle = TrainDataBundle(train=train, test=test)

        spec = SemanticTrainFlowSpec(
            assembly=FlowAssemblySpec(
                trainer=TrainerAssemblySpec(
                    trainer_key="symbolic_stagewise",
                    trainer_params={
                        "artifact_id": "portfolio_gpu_schedule_case",
                        "search_max_added_terms": 0,
                        "keep_search_trace": False,
                    },
                ),
                numericizer=NumericizerSpec(key="default", params={}),
            ),
            eval_splits=("train", "test"),
            save_artifact=False,
            save_report=False,
            run_name="portfolio_gpu_schedule_smoke",
            portfolio_parallel_mode="serial",
            portfolio_gpu_strategy="round_robin",
            portfolio_gpu_devices=(0, 1),
        )

        with patch("core.orchestration.workflow.discover_execution_devices", return_value=("cuda:0", "cuda:1")), patch(
            "core.orchestration.workflow.detect_local_execution_offer",
            return_value=ExecutionResourceOffer(threads=4, cuda_devices=("cuda:0", "cuda:1")),
        ):
            portfolio = run_semantic_portfolio_flow(
                bundle,
                spec=spec,
                model_specs=(
                    ModelSpec(model_id="m0", feature_names=("x0",), strict=True),
                    ModelSpec(model_id="m1", feature_names=("x0", "x2"), strict=True),
                ),
            )

        rows = {str(row["model_id"]): dict(row) for row in portfolio.summary.get("models", [])}
        self.assertEqual(str(rows["m0"].get("assigned_device")), "cuda:0")
        self.assertEqual(str(rows["m1"].get("assigned_device")), "cuda:1")

        sem0 = dict(portfolio.runs["m0"].report.get("semantic_assembly", {}))
        sem1 = dict(portfolio.runs["m1"].report.get("semantic_assembly", {}))
        tr0 = dict(sem0.get("trainer", {}))
        tr1 = dict(sem1.get("trainer", {}))
        p0 = dict(tr0.get("trainer_params", {}))
        p1 = dict(tr1.get("trainer_params", {}))
        self.assertEqual(str(p0.get("search_inner_opt_device")), "cuda:0")
        self.assertEqual(str(p1.get("search_inner_opt_device")), "cuda:1")

    def test_semantic_portfolio_hard_checks_phase_threads_against_offer(self) -> None:
        rng = np.random.default_rng(31)
        X = rng.normal(size=(120, 3))
        y = (1.4 * X[:, 0] - 0.4 * X[:, 1] + 0.02 * rng.normal(size=120)).reshape(-1, 1)

        train = ProcessedDataset(
            X_train=X[:90],
            y_train=y[:90],
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
        )
        test = ProcessedDataset(
            X_train=X[90:],
            y_train=y[90:],
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
        )
        bundle = TrainDataBundle(train=train, test=test)

        spec = SemanticTrainFlowSpec(
            assembly=FlowAssemblySpec(
                trainer=TrainerAssemblySpec(trainer_key="ridge", trainer_params={"l2": 0.0}),
                numericizer=NumericizerSpec(key="default", params={}),
            ),
            eval_splits=("train", "test"),
            save_artifact=False,
            save_report=False,
            run_name="portfolio_budget_threads",
            portfolio_parallel_mode="thread",
            portfolio_max_workers=3,
        )

        with patch(
            "core.orchestration.workflow.detect_local_execution_offer",
            return_value=ExecutionResourceOffer(threads=2),
        ):
            with self.assertRaises(ExecutionBudgetError):
                run_semantic_portfolio_flow(
                    bundle,
                    spec=spec,
                    model_specs=(
                        ModelSpec(model_id="m0", feature_names=("x0",), strict=True),
                        ModelSpec(model_id="m1", feature_names=("x1",), strict=True),
                        ModelSpec(model_id="m2", feature_names=("x2",), strict=True),
                    ),
                )

    def test_semantic_portfolio_aggregates_trainer_thread_requests(self) -> None:
        rng = np.random.default_rng(33)
        X = rng.normal(size=(120, 3))
        y = (0.9 * X[:, 0] + 0.2 * X[:, 1] + 0.03 * rng.normal(size=120)).reshape(-1, 1)

        train = ProcessedDataset(
            X_train=X[:90],
            y_train=y[:90],
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
        )
        test = ProcessedDataset(
            X_train=X[90:],
            y_train=y[90:],
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
        )
        bundle = TrainDataBundle(train=train, test=test)

        spec = SemanticTrainFlowSpec(
            assembly=FlowAssemblySpec(
                trainer=TrainerAssemblySpec(
                    trainer_key="xgboost",
                    trainer_params={"n_estimators": 8, "n_jobs": 2, "verbosity": 0},
                ),
                numericizer=NumericizerSpec(key="default", params={}),
            ),
            eval_splits=("train", "test"),
            save_artifact=False,
            save_report=False,
            run_name="portfolio_budget_trainer_threads",
            portfolio_parallel_mode="thread",
            portfolio_max_workers=2,
        )

        with patch(
            "core.orchestration.workflow.detect_local_execution_offer",
            return_value=ExecutionResourceOffer(threads=3),
        ):
            with self.assertRaises(ExecutionBudgetError):
                run_semantic_portfolio_flow(
                    bundle,
                    spec=spec,
                    model_specs=(
                        ModelSpec(model_id="m0", feature_names=("x0",), strict=True),
                        ModelSpec(model_id="m1", feature_names=("x1",), strict=True),
                    ),
                )

    def test_semantic_portfolio_hard_checks_gpu_device_oversubscription(self) -> None:
        rng = np.random.default_rng(32)
        X = rng.normal(size=(100, 3))
        y = (1.0 * X[:, 0] - 0.2 * X[:, 2] + 0.03 * rng.normal(size=100)).reshape(-1, 1)

        train = ProcessedDataset(
            X_train=X[:80],
            y_train=y[:80],
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
        )
        test = ProcessedDataset(
            X_train=X[80:],
            y_train=y[80:],
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
        )
        bundle = TrainDataBundle(train=train, test=test)

        spec = SemanticTrainFlowSpec(
            assembly=FlowAssemblySpec(
                trainer=TrainerAssemblySpec(
                    trainer_key="mlp_torch",
                    trainer_params={
                        "artifact_id": "portfolio_budget_gpu_case",
                        "device": "cuda:0",
                    },
                ),
                numericizer=NumericizerSpec(key="default", params={}),
            ),
            eval_splits=("train", "test"),
            save_artifact=False,
            save_report=False,
            run_name="portfolio_budget_gpu",
            portfolio_parallel_mode="thread",
            portfolio_max_workers=2,
            portfolio_gpu_strategy="fixed",
            portfolio_gpu_devices=(0,),
        )

        with patch(
            "core.orchestration.workflow.detect_local_execution_offer",
            return_value=ExecutionResourceOffer(threads=4, cuda_devices=("cuda:0",)),
        ):
            with self.assertRaises(ExecutionBudgetError):
                run_semantic_portfolio_flow(
                    bundle,
                    spec=spec,
                    model_specs=(
                        ModelSpec(model_id="m0", feature_names=("x0",), strict=True),
                        ModelSpec(model_id="m1", feature_names=("x0", "x2"), strict=True),
                    ),
                )

    def test_semantic_portfolio_aggregates_model_spec_execution_resources(self) -> None:
        rng = np.random.default_rng(34)
        X = rng.normal(size=(120, 3))
        y = (0.7 * X[:, 0] - 0.1 * X[:, 2] + 0.02 * rng.normal(size=120)).reshape(-1, 1)

        train = ProcessedDataset(
            X_train=X[:90],
            y_train=y[:90],
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
        )
        test = ProcessedDataset(
            X_train=X[90:],
            y_train=y[90:],
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
        )
        bundle = TrainDataBundle(train=train, test=test)

        spec = SemanticTrainFlowSpec(
            assembly=FlowAssemblySpec(
                trainer=TrainerAssemblySpec(trainer_key="ridge", trainer_params={"l2": 0.0}),
                numericizer=NumericizerSpec(key="default", params={}),
            ),
            eval_splits=("train", "test"),
            save_artifact=False,
            save_report=False,
            run_name="portfolio_budget_model_metadata",
            portfolio_parallel_mode="thread",
            portfolio_max_workers=2,
        )

        extra_resources = {
            "problem_execution_resources": {
                "components": [
                    {
                        "threads": 2,
                        "backend": "thread",
                        "label": "problem_eval:branch",
                        "metadata": {"rolling_folds": 6, "regime_branch_parallel_workers": 2},
                    }
                ]
            }
        }

        with patch(
            "core.orchestration.workflow.detect_local_execution_offer",
            return_value=ExecutionResourceOffer(threads=5),
        ):
            with self.assertRaises(ExecutionBudgetError):
                run_semantic_portfolio_flow(
                    bundle,
                    spec=spec,
                    model_specs=(
                        ModelSpec(model_id="m0", feature_names=("x0",), strict=True, metadata=extra_resources),
                        ModelSpec(model_id="m1", feature_names=("x1",), strict=True, metadata=extra_resources),
                    ),
                )


class TestScaffoldPortfolioRuntime(unittest.TestCase):
    def test_scaffold_model_specs_runs_portfolio_and_returns_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            n = 120
            rng = np.random.default_rng(123)
            x0 = rng.normal(size=n)
            x1 = rng.normal(size=n)
            y = 1.7 * x0 + 0.1 * rng.normal(size=n)

            df = pd.DataFrame(
                {
                    "target": y,
                    "x0": x0,
                    "x1": x1,
                }
            )
            csv_path = root / "data.csv"
            df.to_csv(csv_path, index=False)

            payload = {
                "data": {
                    "csv_path": str(csv_path),
                    "target_col": "target",
                    "date_col": None,
                    "feature_recipe": "raw_all_numeric",
                    "split_mode": "ratio",
                    "test_ratio": 0.25,
                    "random_seed": 3,
                },
                "train": {
                    "trainer_key": "ridge",
                    "trainer_params": {"l2": 0.0},
                    "output_dir": str(root / "runs"),
                    "run_name": "scaffold_portfolio_smoke",
                    "model_specs": [
                        {"model_id": "good", "feature_names": ["x0"], "target_names": ["target"], "strict": True},
                        {"model_id": "bad", "feature_names": ["x1"], "target_names": ["target"], "strict": True},
                    ],
                },
            }

            spec = build_scaffold_spec(payload)
            result = run_project_scaffold(spec)

            self.assertIn("portfolio", result.report)
            portfolio = dict(result.report["portfolio"])
            self.assertEqual(str(portfolio.get("best_model_id")), "good")
            self.assertEqual(str(result.report.get("model_spec", {}).get("model_id")), "good")

            portfolio_report = root / "runs" / "portfolio_report.json"
            self.assertTrue(portfolio_report.exists())

    def test_build_sqlalchemy_url_from_db_fields(self) -> None:
        pg_url, pg_label = _build_sqlalchemy_url(
            {
                "host": "127.0.0.1",
                "database": "traffic",
                "user": "demo",
                "password": "p@ss",
            },
            source_name="pg_demo",
            default_driver="postgresql+psycopg2",
            default_port=5432,
        )
        self.assertTrue(pg_url.startswith("postgresql+psycopg2://demo:p%40ss@127.0.0.1:5432/traffic"))
        self.assertEqual(pg_label, "postgresql+psycopg2://127.0.0.1:5432/traffic")

        my_url, my_label = _build_sqlalchemy_url(
            {
                "host": "localhost",
                "port": 3307,
                "database": "mall",
                "user": "root",
                "password": "123456",
            },
            source_name="mysql_demo",
            default_driver="mysql+pymysql",
            default_port=3306,
        )
        self.assertTrue(my_url.startswith("mysql+pymysql://root:123456@localhost:3307/mall"))
        self.assertEqual(my_label, "mysql+pymysql://localhost:3307/mall")


if __name__ == "__main__":
    unittest.main()
