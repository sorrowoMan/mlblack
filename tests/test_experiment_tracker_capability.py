from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
import gc
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from config import CapabilitySpec, FlowAssemblySpec, NumericizerSpec, TrainerAssemblySpec
from core.common.contracts import ProcessedDataset
from core.flow_experiment_tracker import (
    ExperimentTrackerCapability,
    list_experiment_artifact_catalog,
    list_experiment_run_catalog,
    show_experiment_artifact_catalog_entry,
    show_experiment_run_catalog_entry,
)
from core.state.context_keys import RUN_STAGE
from core.symbolic import SymbolicSearchMechanismContract, build_symbolic_search_mechanism_contracts
from experiment.contracts import RUN_SURFACE_CONTRACT_VERSION
from training import TrainTask, TrainingCompatibilityError, TrainingInit
from workflow import ModelSpec, SemanticTrainFlowSpec, TrainDataBundle, run_semantic_train_flow


class TestExperimentTrackerCapability(unittest.TestCase):
    def _bundle(self) -> TrainDataBundle:
        rng = np.random.default_rng(19)
        X = rng.normal(size=(120, 4))
        y = (1.3 * X[:, 0] - 0.4 * X[:, 1] + 0.2 * X[:, 2]).reshape(-1, 1)
        return TrainDataBundle(
            train=ProcessedDataset(
                X_train=X[:90],
                y_train=y[:90],
                feature_names=("x0", "x1", "x2", "x3"),
                target_names=("y",),
            ),
            test=ProcessedDataset(
                X_train=X[90:],
                y_train=y[90:],
                feature_names=("x0", "x1", "x2", "x3"),
                target_names=("y",),
            ),
        )

    def test_tracker_writes_runs_events_metrics(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="mlblack_exp_tracker_"))
        try:
            db_path = tmp_dir / "experiments.sqlite3"
            spec = SemanticTrainFlowSpec(
                assembly=FlowAssemblySpec(
                    trainer=TrainerAssemblySpec(
                        trainer_key="ridge",
                        trainer_params={"l2": 0.0},
                    ),
                    numericizer=NumericizerSpec(key="default", params={}),
                    capabilities=(
                        CapabilitySpec(
                            key="experiment_tracker",
                            params={
                                "db_path": str(db_path),
                                "namespace": "ut",
                                "io_mode": "batched",
                                "commit_interval": 0,
                            },
                        ),
                    ),
                ),
                model_spec=ModelSpec(
                    model_id="m_demo",
                    feature_names=("x0", "x1"),
                    target_names=("y",),
                    strict=True,
                ),
                eval_splits=("train", "test"),
                save_artifact=False,
                save_report=False,
                capability_strict=True,
                run_name="exp_tracker_smoke",
            )

            result = run_semantic_train_flow(self._bundle(), spec=spec)

            tracker = dict(result.report.get("experiment_tracker", {}))
            run_id = str(tracker.get("run_id", ""))
            self.assertTrue(run_id)
            self.assertTrue(db_path.exists())
            self.assertEqual(str(tracker.get("status")), "finished")

            with sqlite3.connect(str(db_path)) as conn:
                runs_row = conn.execute(
                    "SELECT status, run_name FROM experiment_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                self.assertIsNotNone(runs_row)
                assert runs_row is not None
                self.assertEqual(str(runs_row[0]), "finished")
                self.assertEqual(str(runs_row[1]), "exp_tracker_smoke")

                events_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM experiment_events WHERE run_id = ?",
                        (run_id,),
                    ).fetchone()[0]
                )
                self.assertGreaterEqual(events_count, 8)

                metrics_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM experiment_metrics WHERE run_id = ?",
                        (run_id,),
                    ).fetchone()[0]
                )
                self.assertGreaterEqual(metrics_count, 6)
                trace_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM experiment_training_trace WHERE run_id = ?",
                        (run_id,),
                    ).fetchone()[0]
                )
                # ridge path does not emit search_trace
                self.assertEqual(trace_count, 0)
        finally:
            gc.collect()
            time.sleep(0.1)
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_tracker_legacy_mode_compatibility(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="mlblack_exp_tracker_legacy_"))
        try:
            db_path = tmp_dir / "experiments.sqlite3"
            cap = ExperimentTrackerCapability(
                db_path=str(db_path),
                namespace="ut_legacy",
                io_mode="legacy",
            )
            ctx: dict[str, object] = {
                "run_name": "legacy_smoke",
                "context_refs": {RUN_STAGE: "fit"},
                "snapshot_count": 1,
                "metrics": {
                    "train": {"rmse": 1.0, "mae": 0.5, "r2": 0.8},
                    "test": {"rmse": 1.2, "mae": 0.6, "r2": 0.7},
                },
            }

            cap.on_flow_start(ctx)
            cap.on_post_eval(ctx)
            cap.on_flow_finish(ctx)

            tracker_raw = ctx.get("experiment_tracker", {})
            tracker = tracker_raw if isinstance(tracker_raw, dict) else {}
            run_id = str(tracker.get("run_id", ""))
            self.assertTrue(run_id)

            with sqlite3.connect(str(db_path)) as conn:
                runs_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM experiment_runs WHERE run_id = ?",
                        (run_id,),
                    ).fetchone()[0]
                )
                metrics_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM experiment_metrics WHERE run_id = ?",
                        (run_id,),
                    ).fetchone()[0]
                )
                self.assertEqual(runs_count, 1)
                self.assertGreaterEqual(metrics_count, 6)
        finally:
            gc.collect()
            time.sleep(0.1)
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_tracker_writes_training_trace_rows_from_search_trace(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="mlblack_exp_tracker_trace_"))
        try:
            db_path = tmp_dir / "experiments.sqlite3"
            cap = ExperimentTrackerCapability(
                db_path=str(db_path),
                namespace="ut_trace",
            )
            ctx: dict[str, object] = {
                "run_name": "trace_manual_smoke",
                "context_refs": {RUN_STAGE: "fit"},
                "snapshot_count": 3,
            }

            cap.on_flow_start(ctx)
            artifact = SimpleNamespace(
                artifact_id="artifact_demo",
                metadata={
                    "search_trace": {
                        "iterations": [
                            {
                                "iteration": 1,
                                "selected": {
                                    "operation": "add",
                                    "name": "x0*x1",
                                    "family": "mul",
                                    "expr": "mul(x0,x1)",
                                },
                                "metrics_before": {"rmse": 1.2},
                                "metrics_after": {"rmse": 1.0},
                                "metrics_before_val": {"rmse": 1.3},
                                "metrics_after_val": {"rmse": 1.1},
                                "gradient_summary": {"overall_mismatch": 0.42},
                                "readout": {
                                    "before": {"weight_l2": 0.9},
                                    "after": {"weight_l2": 1.2},
                                },
                                "n_terms_before": 2,
                                "n_terms_after": 3,
                            }
                        ]
                    }
                },
            )
            ctx["artifact"] = artifact
            cap.on_post_fit(ctx)
            cap.on_flow_finish(ctx)

            with sqlite3.connect(str(db_path)) as conn:
                row = conn.execute(
                    """
                    SELECT operation, selected_name, selected_family, rmse_before, rmse_after,
                           grad_overall_mismatch, weight_l2_before, weight_l2_after
                    FROM experiment_training_trace
                    ORDER BY id DESC LIMIT 1
                    """
                ).fetchone()
                self.assertIsNotNone(row)
                assert row is not None
                self.assertEqual(str(row[0]), "add")
                self.assertEqual(str(row[1]), "x0*x1")
                self.assertEqual(str(row[2]), "mul")
                self.assertAlmostEqual(float(row[3]), 1.2, places=6)
                self.assertAlmostEqual(float(row[4]), 1.0, places=6)
                self.assertAlmostEqual(float(row[5]), 0.42, places=6)
                self.assertAlmostEqual(float(row[6]), 0.9, places=6)
                self.assertAlmostEqual(float(row[7]), 1.2, places=6)
        finally:
            gc.collect()
            time.sleep(0.1)
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_tracker_materializes_run_and_artifact_catalog_views(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="mlblack_exp_tracker_catalog_"))
        try:
            db_path = tmp_dir / "experiments.sqlite3"
            cap = ExperimentTrackerCapability(
                db_path=str(db_path),
                namespace="ut_catalog",
            )
            ctx: dict[str, object] = {
                "run_name": "catalog_materialize_smoke",
                "context_refs": {RUN_STAGE: "report"},
                "snapshot_count": 2,
                "output_dir": str(tmp_dir / "run_out"),
                "trainer": SimpleNamespace(name="symbolic_torch_interval"),
                "report": {
                    "run_name": "catalog_materialize_smoke",
                    "trainer_name": "symbolic_torch_interval",
                    "training": {
                        "requested_init": {"mode": "warm_start"},
                        "task_signature": {
                            "symbolic_family_signature": "sig_demo",
                            "metadata": {
                                "symbolic_family": {
                                    "search_mechanism_contracts": [
                                        {"mechanism_key": "beam_selection"},
                                        {"mechanism_key": "inner_optimizer"},
                                    ],
                                    "search_family_signature_contracts": [
                                        {"mechanism_key": "beam_selection", "consume": ["gradient_signal"]}
                                    ],
                                }
                            },
                        },
                        "compatibility": {
                            "symbolic_family_signature_drift": {
                                "parent_artifact": {
                                    "changed_mechanisms": [
                                        {
                                            "mechanism_key": "beam_selection",
                                            "change_type": "modified",
                                            "changed_fields": ["consume"],
                                        }
                                    ]
                                }
                            }
                        },
                    },
                    "artifact": {
                    "artifact_id": "artifact_interval_demo",
                    "symbolic_artifact_schema": {
                        "head_semantics": {"task": "interval"},
                        "complexity_metrics": {"term_count": 4},
                        "regime_structure": {
                            "mode": "piecewise",
                            "local_regime_count": 2,
                        },
                        "basis_structure": {
                            "basis_scope": "global+local",
                            "basis_count": 7,
                            "orthogonality_status": {
                                "status": "reported",
                                "orthogonality_score": 0.74,
                                "pair_abs_corr_mean": 0.18,
                            },
                            "residual_complementarity": {
                                "status": "reported",
                                "recorded": {
                                    "mean_marginal_r2_gain": 0.22,
                                },
                            },
                            "semantic_deduplication": {
                                "status": "reported",
                                "recorded": {
                                    "semantic_unique_ratio": 0.86,
                                },
                            },
                        },
                        "assembler_structure": {
                            "assembler_mode": "piecewise_budgeted_symbolic_regression",
                            "output_expression_count": 6,
                        },
                        "piecewise_gate_basis": {
                            "status": "enabled",
                            "gate_basis_count": 2,
                        },
                        "truth_contract_recovery": {
                            "status": "reported",
                            "exact_basis_hit_score": 0.75,
                            "exact_term_recovery_score": 0.5,
                            "truth_basis_matches": [
                                {"truth_term": "safe_ratio(voltage,resistance)", "term_recovered": True}
                            ],
                        },
                        "orthogonal_search_objective": {
                            "status": "reported",
                            "protocol": "orthogonal_structure_search_with_budgeted_symbolic_assembler",
                            "inner_fit_score": 0.84,
                            "outer_score": 1.42,
                        },
                        "stability_metrics": {
                            "fold_count": 3,
                            "fold_summary": {
                                    "rmse_mean": 0.33,
                                    "rmse_std": 0.02,
                                    "coverage_error_mean": 0.04,
                                },
                                "rmse_mean": 0.33,
                                "rmse_std": 0.02,
                                "coverage_error_mean": 0.04,
                                "pinaw_mean": 0.2,
                                "interval_score_mean": 0.5,
                                "picp_mean": 0.9,
                                "mean_width_mean": 1.2,
                                "family_concentration": 0.7,
                                "feature_concentration": 0.6,
                            },
                        },
                    },
                },
            }

            cap.on_flow_start(ctx)
            cap.on_flow_finish(ctx)

            run_rows = list_experiment_run_catalog(
                db_path,
                trainer_name="symbolic_torch_interval",
                has_fold_summary=True,
                max_rmse_std=0.03,
                max_coverage_error_mean=0.05,
            )
            self.assertEqual(len(run_rows), 1)
            run_row = run_rows[0]
            self.assertEqual(str(run_row.get("run_name")), "catalog_materialize_smoke")
            self.assertEqual(float(run_row.get("rmse_std")), 0.02)
            self.assertEqual(dict(run_row.get("fold_summary_json", {})).get("coverage_error_mean"), 0.04)
            self.assertIn("beam_selection", tuple(run_row.get("search_family_signature_keys_json", ())))
            surface_record = dict(run_row.get("surface_record_json", {}))
            assembly_record = dict(run_row.get("assembly_record_json", {}))
            run_record = dict(run_row.get("run_record_json", {}))
            self.assertEqual(str(surface_record.get("contract_version")), RUN_SURFACE_CONTRACT_VERSION)
            self.assertEqual(str(surface_record.get("framework")), "mlblack")
            self.assertEqual(str(surface_record.get("surface_kind")), "flow")
            self.assertEqual(str(surface_record.get("driver_ref")), "trainer:symbolic_torch_interval")
            self.assertTrue(str(surface_record.get("surface_signature") or "").strip())
            self.assertEqual(str(assembly_record.get("family_ref")), "family:symbolic")
            self.assertEqual(str(assembly_record.get("preset_ref")), "preset:symbolic_torch_interval")
            self.assertEqual(str(assembly_record.get("head_ref")), "head:interval")
            self.assertTrue(str(assembly_record.get("assembly_signature") or "").strip())
            self.assertEqual(str(run_record.get("framework")), "mlblack")
            self.assertEqual(str(run_record.get("driver_ref")), "trainer:symbolic_torch_interval")
            self.assertIn("artifact_interval_demo", tuple(run_record.get("artifact_ids", ())))
            self.assertTrue(str(run_record.get("subject_signature") or "").strip())
            self.assertTrue(str(run_record.get("param_signature") or "").strip())
            self.assertEqual(str(run_row.get("surface_key")), "flow:catalog_materialize_smoke")
            self.assertEqual(str(run_row.get("surface_kind")), "flow")
            self.assertEqual(str(run_row.get("family_ref")), "family:symbolic")
            self.assertEqual(str(run_row.get("driver_ref")), "trainer:symbolic_torch_interval")
            self.assertEqual(str(run_row.get("assembly_signature")), str(assembly_record.get("assembly_signature")))
            self.assertEqual(str(run_row.get("surface_signature")), str(surface_record.get("surface_signature")))
            self.assertEqual(str(run_row.get("regime_mode")), "piecewise")
            self.assertEqual(str(run_row.get("basis_scope")), "global+local")
            self.assertEqual(str(run_row.get("assembler_mode")), "piecewise_budgeted_symbolic_regression")
            self.assertEqual(str(run_row.get("piecewise_gate_status")), "enabled")
            self.assertEqual(str(run_row.get("orthogonality_status")), "reported")
            self.assertEqual(str(run_row.get("residual_complementarity_status")), "reported")
            self.assertEqual(str(run_row.get("semantic_dedup_status")), "reported")
            self.assertAlmostEqual(float(run_row.get("orthogonality_score")), 0.74, places=6)
            self.assertAlmostEqual(float(run_row.get("pair_abs_corr_mean")), 0.18, places=6)
            self.assertAlmostEqual(float(run_row.get("residual_gain_mean")), 0.22, places=6)
            self.assertAlmostEqual(float(run_row.get("semantic_unique_ratio")), 0.86, places=6)
            self.assertEqual(int(run_row.get("gate_basis_count")), 2)
            self.assertEqual(int(run_row.get("selected_regime_count")), 2)
            self.assertEqual(int(run_row.get("basis_count")), 7)
            self.assertEqual(int(run_row.get("output_expression_count")), 6)
            self.assertAlmostEqual(float(run_row.get("exact_basis_hit_score")), 0.75, places=6)
            self.assertAlmostEqual(float(run_row.get("exact_term_recovery_score")), 0.5, places=6)
            self.assertAlmostEqual(float(run_row.get("outer_objective_score")), 1.42, places=6)
            self.assertAlmostEqual(float(run_row.get("inner_fit_score")), 0.84, places=6)
            self.assertEqual(dict(run_row.get("regime_structure_json", {})).get("mode"), "piecewise")
            self.assertEqual(dict(run_row.get("basis_structure_json", {})).get("basis_scope"), "global+local")
            self.assertEqual(
                dict(run_row.get("assembler_structure_json", {})).get("assembler_mode"),
                "piecewise_budgeted_symbolic_regression",
            )
            self.assertEqual(dict(run_row.get("piecewise_gate_basis_json", {})).get("status"), "enabled")
            self.assertEqual(dict(run_row.get("truth_contract_recovery_json", {})).get("status"), "reported")
            self.assertEqual(dict(run_row.get("orthogonal_search_objective_json", {})).get("status"), "reported")
            self.assertEqual(
                dict(run_row.get("artifact_catalog_json", {})).get("basis_structure", {}).get("basis_scope"),
                "global+local",
            )

            filtered_rows = list_experiment_run_catalog(
                db_path,
                surface_key="flow:catalog_materialize_smoke",
                family_ref="family:symbolic",
                assembly_signature=str(assembly_record.get("assembly_signature")),
                regime_mode="piecewise",
                basis_scope="global+local",
                assembler_mode="piecewise_budgeted_symbolic_regression",
                piecewise_gate_status="enabled",
                orthogonality_status="reported",
                residual_complementarity_status="reported",
                semantic_dedup_status="reported",
            )
            self.assertEqual(len(filtered_rows), 1)
            self.assertEqual(str(filtered_rows[0].get("run_id")), str(run_row.get("run_id")))

            filtered_recovery_rows = list_experiment_run_catalog(
                db_path,
                min_exact_basis_hit_score=0.7,
                min_exact_term_recovery_score=0.4,
                min_outer_objective_score=1.0,
            )
            self.assertEqual(len(filtered_recovery_rows), 1)

            shown_run = show_experiment_run_catalog_entry(db_path, run_id=str(run_row.get("run_id")))
            self.assertIsNotNone(shown_run)
            assert shown_run is not None
            self.assertEqual(str(shown_run.get("artifact_id")), "artifact_interval_demo")

            artifact_rows = list_experiment_artifact_catalog(
                db_path,
                trainer_name="symbolic_torch_interval",
                head_task="interval",
                has_fold_summary=True,
                max_rmse_std=0.03,
            )
            self.assertEqual(len(artifact_rows), 1)
            artifact_row = artifact_rows[0]
            self.assertEqual(str(artifact_row.get("artifact_id")), "artifact_interval_demo")
            self.assertEqual(float(artifact_row.get("coverage_error_mean")), 0.04)
            self.assertEqual(str(artifact_row.get("regime_mode")), "piecewise")
            self.assertEqual(str(artifact_row.get("basis_scope")), "global+local")
            self.assertEqual(str(artifact_row.get("assembler_mode")), "piecewise_budgeted_symbolic_regression")
            self.assertEqual(str(artifact_row.get("piecewise_gate_status")), "enabled")
            self.assertEqual(str(artifact_row.get("orthogonality_status")), "reported")
            self.assertEqual(str(artifact_row.get("residual_complementarity_status")), "reported")
            self.assertEqual(str(artifact_row.get("semantic_dedup_status")), "reported")
            self.assertAlmostEqual(float(artifact_row.get("orthogonality_score")), 0.74, places=6)
            self.assertAlmostEqual(float(artifact_row.get("pair_abs_corr_mean")), 0.18, places=6)
            self.assertAlmostEqual(float(artifact_row.get("residual_gain_mean")), 0.22, places=6)
            self.assertAlmostEqual(float(artifact_row.get("semantic_unique_ratio")), 0.86, places=6)
            self.assertEqual(int(artifact_row.get("gate_basis_count")), 2)
            self.assertEqual(int(artifact_row.get("selected_regime_count")), 2)
            self.assertEqual(int(artifact_row.get("basis_count")), 7)
            self.assertEqual(int(artifact_row.get("output_expression_count")), 6)
            self.assertAlmostEqual(float(artifact_row.get("exact_basis_hit_score")), 0.75, places=6)
            self.assertAlmostEqual(float(artifact_row.get("exact_term_recovery_score")), 0.5, places=6)
            self.assertAlmostEqual(float(artifact_row.get("outer_objective_score")), 1.42, places=6)
            self.assertAlmostEqual(float(artifact_row.get("inner_fit_score")), 0.84, places=6)

            filtered_artifact_rows = list_experiment_artifact_catalog(
                db_path,
                trainer_name="symbolic_torch_interval",
                head_task="interval",
                regime_mode="piecewise",
                basis_scope="global+local",
                assembler_mode="piecewise_budgeted_symbolic_regression",
                piecewise_gate_status="enabled",
                orthogonality_status="reported",
                residual_complementarity_status="reported",
                semantic_dedup_status="reported",
            )
            self.assertEqual(len(filtered_artifact_rows), 1)
            self.assertEqual(str(filtered_artifact_rows[0].get("artifact_id")), "artifact_interval_demo")

            filtered_recovery_artifacts = list_experiment_artifact_catalog(
                db_path,
                min_exact_basis_hit_score=0.7,
                min_exact_term_recovery_score=0.4,
                min_outer_objective_score=1.0,
            )
            self.assertEqual(len(filtered_recovery_artifacts), 1)

            shown_artifact = show_experiment_artifact_catalog_entry(
                db_path,
                run_id=str(run_row.get("run_id")),
                artifact_id="artifact_interval_demo",
            )
            self.assertIsNotNone(shown_artifact)
            assert shown_artifact is not None
            self.assertEqual(dict(shown_artifact.get("head_semantics_json", {})).get("task"), "interval")
            artifact_record = dict(shown_artifact.get("artifact_record_json", {}))
            self.assertEqual(str(artifact_record.get("contract_version")), RUN_SURFACE_CONTRACT_VERSION)
            self.assertEqual(str(artifact_record.get("artifact_id")), "artifact_interval_demo")
            self.assertEqual(str(artifact_record.get("artifact_role")), "primary_model_artifact")
            self.assertEqual(str(artifact_record.get("producer_ref")), "preset:symbolic_torch_interval")
            self.assertEqual(dict(shown_artifact.get("regime_structure_json", {})).get("mode"), "piecewise")
            self.assertEqual(dict(shown_artifact.get("basis_structure_json", {})).get("basis_count"), 7)
            self.assertEqual(
                dict(shown_artifact.get("assembler_structure_json", {})).get("output_expression_count"),
                6,
            )
            self.assertEqual(dict(shown_artifact.get("piecewise_gate_basis_json", {})).get("status"), "enabled")
            self.assertEqual(dict(shown_artifact.get("truth_contract_recovery_json", {})).get("status"), "reported")
            self.assertEqual(dict(shown_artifact.get("orthogonal_search_objective_json", {})).get("status"), "reported")
        finally:
            gc.collect()
            time.sleep(0.1)
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_tracker_records_failed_symbolic_contract_drift_runs(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="mlblack_exp_tracker_failed_catalog_"))
        try:
            db_path = tmp_dir / "experiments.sqlite3"
            rng = np.random.default_rng(91)
            X = rng.normal(size=(36, 4))
            y = (1.1 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * np.sin(X[:, 2])).reshape(-1, 1)
            data = ProcessedDataset(
                X_train=X,
                y_train=y,
                feature_names=("x0", "x1", "x2", "x3"),
                target_names=("y",),
            )

            parent_spec = SemanticTrainFlowSpec(
                assembly=FlowAssemblySpec(
                    trainer=TrainerAssemblySpec(
                        trainer_key="symbolic",
                        trainer_params={
                            "parameter_backend": "torch",
                            "task": "point",
                            "structure_engine": {
                                "structure_mode": "seed_library",
                                "search_driver": "local_seed_builder",
                                "dynamic_pool_enabled": False,
                            },
                            "device": "cpu",
                            "epochs": 2,
                            "batch_size": 8,
                            "val_ratio": 0.2,
                            "early_stop_patience": 8,
                            "random_seed": 42,
                        },
                    ),
                    numericizer=NumericizerSpec(key="default", params={}),
                ),
                save_artifact=False,
                save_report=False,
                run_name="parent_symbolic_catalog_failure",
            )
            parent_result = run_semantic_train_flow(TrainDataBundle(train=data), spec=parent_spec)

            rows = tuple(build_symbolic_search_mechanism_contracts())
            changed_beam = tuple(
                SymbolicSearchMechanismContract(
                    **(
                        {
                            "mechanism_key": contract.mechanism_key,
                            "mechanism_kind": contract.mechanism_kind,
                            "consume": tuple(contract.consume) + ("catalog_failure_signal",),
                            "produce": tuple(contract.produce),
                            "mutate": tuple(contract.mutate),
                            "checkpoint": tuple(contract.checkpoint),
                            "replay": tuple(contract.replay),
                            "checkpointable": bool(contract.checkpointable),
                            "replayable": bool(contract.replayable),
                            "affects_family_signature": bool(contract.affects_family_signature),
                        }
                        if contract.mechanism_key == "beam_selection"
                        else {
                            "mechanism_key": contract.mechanism_key,
                            "mechanism_kind": contract.mechanism_kind,
                            "consume": tuple(contract.consume),
                            "produce": tuple(contract.produce),
                            "mutate": tuple(contract.mutate),
                            "checkpoint": tuple(contract.checkpoint),
                            "replay": tuple(contract.replay),
                            "checkpointable": bool(contract.checkpointable),
                            "replayable": bool(contract.replayable),
                            "affects_family_signature": bool(contract.affects_family_signature),
                        }
                    )
                )
                for contract in rows
            )

            failing_spec = SemanticTrainFlowSpec(
                assembly=FlowAssemblySpec(
                    trainer=TrainerAssemblySpec(
                        trainer_key="symbolic",
                        trainer_params={
                            "parameter_backend": "torch",
                            "task": "point",
                            "structure_engine": {
                                "structure_mode": "seed_library",
                                "search_driver": "local_seed_builder",
                                "dynamic_pool_enabled": False,
                            },
                            "device": "cpu",
                            "epochs": 2,
                            "batch_size": 8,
                            "val_ratio": 0.2,
                            "early_stop_patience": 8,
                            "random_seed": 42,
                        },
                    ),
                    numericizer=NumericizerSpec(key="default", params={}),
                    capabilities=(
                        CapabilitySpec(
                            key="experiment_tracker",
                            params={
                                "db_path": str(db_path),
                                "namespace": "ut_failed",
                                "io_mode": "batched",
                                "commit_interval": 0,
                            },
                        ),
                    ),
                ),
                training_init=TrainingInit(mode="warm_start", parent_artifact=parent_result.artifact),
                save_artifact=False,
                save_report=False,
                capability_strict=True,
                run_name="symbolic_failed_catalog_failure",
            )

            with patch("core.symbolic.trainer_family.build_symbolic_search_mechanism_contracts", return_value=changed_beam):
                with self.assertRaises(TrainingCompatibilityError):
                    run_semantic_train_flow(TrainDataBundle(train=data), spec=failing_spec)

            failed_rows = list_experiment_run_catalog(db_path, status="failed", trainer_name="symbolic_torch")
            self.assertEqual(len(failed_rows), 1)
            failed_row = failed_rows[0]
            self.assertEqual(str(failed_row.get("status")), "failed")
            drift = dict(failed_row.get("compatibility_drift_json", {}))
            self.assertIn("parent_artifact", drift)
            changed = tuple(dict(drift["parent_artifact"]).get("changed_mechanisms", ()))
            self.assertTrue(any(str(dict(item).get("mechanism_key")) == "beam_selection" for item in changed))
        finally:
            gc.collect()
            time.sleep(0.1)
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
