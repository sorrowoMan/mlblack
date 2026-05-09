from __future__ import annotations

import unittest
from dataclasses import dataclass, field

import numpy as np

from config import TrainerAssemblySpec
from core.common.contracts import ProcessedDataset
from workflow import FlowCapability, ModelSpec, TrainFlowSpec, run_train_flow


@dataclass
class TraceCapability(FlowCapability):
    events: list[str] = field(default_factory=list)

    def on_flow_start(self, context):
        self.events.append("on_flow_start")

    def on_data_ready(self, context):
        self.events.append("on_data_ready")

    def on_pre_fit(self, context):
        self.events.append("on_pre_fit")

    def on_post_fit(self, context):
        self.events.append("on_post_fit")

    def on_pre_eval(self, context):
        self.events.append("on_pre_eval")

    def on_post_eval(self, context):
        self.events.append("on_post_eval")

    def on_pre_persist(self, context):
        self.events.append("on_pre_persist")

    def on_post_persist(self, context):
        self.events.append("on_post_persist")

    def on_flow_finish(self, context):
        self.events.append("on_flow_finish")


@dataclass
class RuntimeAliasCapability(FlowCapability):
    events: list[tuple[str, str]] = field(default_factory=list)

    def on_experiment_start(self, context):
        self.events.append(("runtime", "start"))

    def on_experiment_finish(self, result, context):
        self.events.append(("runtime", "finish"))

    def on_experiment_error(self, error, context):
        self.events.append(("runtime", f"error:{type(error).__name__}:{context.get('failed_stage')}"))


class TestFlowCapabilities(unittest.TestCase):
    def test_capability_lifecycle_hooks_are_called(self) -> None:
        rng = np.random.default_rng(23)
        X = rng.normal(size=(48, 4))
        y = (0.9 * X[:, 0] - 0.6 * X[:, 1] + 0.1).reshape(-1, 1)

        data = ProcessedDataset(
            X_train=X,
            y_train=y,
            feature_names=("x0", "x1", "x2", "x3"),
            target_names=("y",),
        )

        trace = TraceCapability(name="trace_cap")
        spec = TrainFlowSpec(
            assembly=TrainerAssemblySpec(
                trainer_key="ridge",
                trainer_params={"l2": 0.0},
            ),
            eval_splits=("train",),
            save_artifact=False,
            save_report=False,
            capabilities=(trace,),
            capability_strict=True,
            run_name="flow_capability_test",
        )

        result = run_train_flow(data, spec=spec)
        self.assertIn("train", result.metrics)

        expected = [
            "on_flow_start",
            "on_data_ready",
            "on_pre_fit",
            "on_post_fit",
            "on_pre_eval",
            "on_post_eval",
            "on_pre_persist",
            "on_post_persist",
            "on_flow_finish",
        ]
        self.assertEqual(trace.events, expected)

        cap_report = result.report.get("capabilities", {})
        self.assertEqual(int(cap_report.get("count", -1)), 1)
        items = list(cap_report.get("items", []))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].get("name"), "trace_cap")
        self.assertIn("on_post_eval", dict(items[0].get("profile", {}).get("events", {})))

    def test_runtime_style_hooks_are_called_via_flow_lifecycle_aliases(self) -> None:
        rng = np.random.default_rng(31)
        X = rng.normal(size=(24, 3))
        y = (1.2 * X[:, 0] + 0.4 * X[:, 1]).reshape(-1, 1)

        data = ProcessedDataset(
            X_train=X,
            y_train=y,
            feature_names=("x0", "x1", "x2"),
            target_names=("y",),
        )

        capability = RuntimeAliasCapability(name="runtime_alias")
        spec = TrainFlowSpec(
            assembly=TrainerAssemblySpec(
                trainer_key="ridge",
                trainer_params={"l2": 0.0},
            ),
            eval_splits=("train",),
            save_artifact=False,
            save_report=False,
            capabilities=(capability,),
            capability_strict=True,
            run_name="flow_runtime_alias_test",
        )

        result = run_train_flow(data, spec=spec)

        self.assertEqual(
            capability.events,
            [
                ("runtime", "start"),
                ("runtime", "finish"),
            ],
        )
        lifecycle_rows = list(result.report.get("lifecycle_events", []))
        self.assertTrue(any("on_flow_start" in tuple(row.get("dispatch_names", ())) for row in lifecycle_rows))
        self.assertTrue(any("on_experiment_start" in tuple(row.get("dispatch_names", ())) for row in lifecycle_rows))
        control_plane_contract = dict(result.report.get("control_plane_contract", {}))
        self.assertIn("lifecycle_events", control_plane_contract)
        self.assertIn("inner_runtime_events", control_plane_contract)
        inner_rows = list(control_plane_contract.get("inner_runtime_events", []))
        self.assertTrue(any(str(row.get("runtime_key")) == "symbolic_structure_search" for row in inner_rows))
        self.assertTrue(any("source_modules" in row for row in inner_rows))

    def test_flow_error_dispatches_runtime_error_alias(self) -> None:
        rng = np.random.default_rng(37)
        X = rng.normal(size=(20, 2))
        y = (0.5 * X[:, 0]).reshape(-1, 1)
        data = ProcessedDataset(
            X_train=X,
            y_train=y,
            feature_names=("x0", "x1"),
            target_names=("y",),
        )

        capability = RuntimeAliasCapability(name="runtime_error_alias")
        broken_spec = TrainFlowSpec(
            assembly=TrainerAssemblySpec(
                trainer_key="ridge",
                trainer_params={"l2": 0.0},
            ),
            eval_splits=("train",),
            save_artifact=False,
            save_report=False,
            capabilities=(capability,),
            capability_strict=True,
            run_name="flow_runtime_error_alias_test",
            model_spec=ModelSpec(feature_indices=(999,), strict=True),
        )

        with self.assertRaises(ValueError):
            run_train_flow(data, spec=broken_spec)

        self.assertEqual(capability.events[0], ("runtime", "start"))
        self.assertEqual(capability.events[1], ("runtime", "error:ValueError:data_ready"))


if __name__ == "__main__":
    unittest.main()
