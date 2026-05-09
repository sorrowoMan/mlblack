from __future__ import annotations

import unittest
from unittest.mock import patch

from core.execution import (
    ExecutionBudgetError,
    ExecutionRuntime,
    ExecutionRuntimeError,
    ExecutionResourceGrant,
    ExecutionResourceOffer,
    ExecutionResourceRequest,
    ExecutionTask,
    assert_phase_resource_budget,
    build_execution_usage_report,
    clamp_worker_count,
    constrain_execution_offer_to_grant,
    describe_registered_execution_backends,
    describe_registered_execution_device_kinds,
    issue_execution_resource_grant,
    list_registered_execution_device_kinds,
    list_registered_execution_backends,
    nested_total_execution_request,
    normalize_execution_device_token,
    resolve_torch_execution_device,
)


def _double(value: int) -> int:
    return int(value) * 2


def _explode(value: int) -> int:
    raise ValueError(f"boom:{int(value)}")


class TestExecutionRuntime(unittest.TestCase):
    def test_global_backend_registry_exposes_builtin_backends(self) -> None:
        keys = {str(spec.key) for spec in list_registered_execution_backends()}
        self.assertTrue({"serial", "thread", "process"}.issubset(keys))
        rows = {str(row["key"]): dict(row) for row in describe_registered_execution_backends()}
        self.assertIn("sync", tuple(rows["serial"]["aliases"]))
        self.assertEqual(tuple(rows["process"]["supported_device_kinds"]), ("cpu",))

    def test_device_registry_normalizes_and_resolves_cuda_tokens(self) -> None:
        class _FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_count() -> int:
                return 2

        class _FakeTorch:
            cuda = _FakeCuda()

            @staticmethod
            def device(token: str) -> str:
                return f"device<{token}>"

        self.assertEqual(str(normalize_execution_device_token("GPU:1")), "cuda:1")
        self.assertEqual(str(normalize_execution_device_token(0)), "cuda:0")
        self.assertEqual(str(normalize_execution_device_token("cpu:0")), "cpu")
        self.assertEqual(str(normalize_execution_device_token("mps:0")), "mps")
        self.assertEqual(
            str(resolve_torch_execution_device(_FakeTorch(), "gpu:1")),
            "device<cuda:1>",
        )

    def test_device_registry_exposes_kind_catalog_and_mps_resolution(self) -> None:
        class _FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def device_count() -> int:
                return 0

        class _FakeMps:
            @staticmethod
            def is_available() -> bool:
                return True

        class _FakeBackends:
            mps = _FakeMps()

        class _FakeTorch:
            cuda = _FakeCuda()
            backends = _FakeBackends()

            @staticmethod
            def device(token: str) -> str:
                return f"device<{token}>"

        kinds = set(list_registered_execution_device_kinds())
        self.assertTrue({"cpu", "cuda", "mps"}.issubset(kinds))

        rows = {str(row["kind"]): dict(row) for row in describe_registered_execution_device_kinds(torch_module=_FakeTorch())}
        self.assertTrue(bool(rows["cpu"]["available"]))
        self.assertTrue(bool(rows["mps"]["available"]))
        self.assertIn("mps", tuple(rows["mps"]["discovered_tokens"]))
        self.assertEqual(str(resolve_torch_execution_device(_FakeTorch(), "mps")), "device<mps>")
        self.assertEqual(str(resolve_torch_execution_device(_FakeTorch(), "auto")), "device<mps>")

    def test_worker_count_is_clamped_to_offered_threads(self) -> None:
        offer = ExecutionResourceOffer(threads=3, cuda_devices=(), mps_devices=())
        self.assertEqual(int(clamp_worker_count(None, n_tasks=10, offer=offer)), 3)
        self.assertEqual(int(clamp_worker_count(8, n_tasks=10, offer=offer)), 3)
        self.assertEqual(int(clamp_worker_count(2, n_tasks=10, offer=offer)), 2)

    def test_nested_total_request_counts_outer_and_inner_threads(self) -> None:
        total = nested_total_execution_request(
            ExecutionResourceRequest(threads=2, backend="thread", label="outer"),
            ExecutionResourceRequest(threads=1, backend="thread", label="inner"),
            fanout=2,
            label="outer_inner",
        )
        self.assertEqual(int(total.threads), 4)
        self.assertEqual(str(total.label), "outer_inner")

    def test_resource_grant_can_cap_offer_and_emit_usage_report(self) -> None:
        request = ExecutionResourceRequest(threads=4, backend="thread", label="inner_problem")
        grant = issue_execution_resource_grant(
            request,
            phase="mlblack_inner_problem",
            label="inner_problem",
            max_threads=2,
            metadata={"wrapper": "nowcasting_outer_bridge"},
        )
        offer = constrain_execution_offer_to_grant(
            ExecutionResourceOffer(threads=8, cuda_devices=(), mps_devices=()),
            grant,
        )
        usage = build_execution_usage_report(
            grant,
            label="branch_evaluation.regime_fold",
            peak_threads=2,
            used_threads=2,
            backend="thread",
        )

        self.assertIsInstance(grant, ExecutionResourceGrant)
        self.assertEqual(str(grant.phase), "mlblack_inner_problem")
        self.assertEqual(int(grant.threads), 2)
        self.assertEqual(int(offer.threads), 2)
        self.assertEqual(int(usage.granted_threads), 2)
        self.assertEqual(int(usage.peak_threads), 2)

    def test_phase_budget_rejects_gpu_oversubscription(self) -> None:
        offer = ExecutionResourceOffer(threads=4, cuda_devices=("cuda:0", "cuda:1"))
        with self.assertRaises(ExecutionBudgetError):
            assert_phase_resource_budget(
                "portfolio",
                (
                    ExecutionResourceRequest(threads=1, backend="thread", label="m0", device_tokens=("cuda:0",)),
                    ExecutionResourceRequest(threads=1, backend="thread", label="m1", device_tokens=("cuda:0",)),
                ),
                offer=offer,
            )

    def test_serial_backend_runs_tasks_in_order(self) -> None:
        runtime = ExecutionRuntime()
        surface = runtime.describe_surface()
        self.assertTrue(any(str(row.get("key")) == "serial" for row in surface["backends"]))
        self.assertTrue(any(str(row.get("kind")) == "cpu" for row in surface["device_kinds"]))
        batch = runtime.map(
            (
                ExecutionTask(task_id="a", fn=_double, args=(2,)),
                ExecutionTask(task_id="b", fn=_double, args=(3,)),
            ),
            backend="serial",
        )

        self.assertEqual(str(batch.backend), "serial")
        self.assertEqual(int(batch.submitted), 2)
        self.assertEqual(int(batch.succeeded), 2)
        self.assertEqual([record.value for record in batch.records], [4, 6])

    def test_thread_backend_captures_soft_failures(self) -> None:
        runtime = ExecutionRuntime()
        with patch("core.execution.runtime.detect_local_execution_offer", return_value=ExecutionResourceOffer(threads=2)):
            batch = runtime.map(
                (
                    ExecutionTask(task_id="ok", fn=_double, args=(5,)),
                    ExecutionTask(task_id="bad", fn=_explode, args=(7,)),
                ),
                backend="thread",
                max_workers=9,
                fail_fast=False,
            )

        self.assertEqual(str(batch.backend), "thread")
        self.assertEqual(int(batch.failed), 1)
        self.assertEqual(int(batch.metadata.get("effective_max_workers", 0)), 2)
        records = {str(record.task_id): record for record in batch.records}
        self.assertTrue(bool(records["ok"].ok))
        self.assertFalse(bool(records["bad"].ok))
        self.assertIn("boom:7", str(records["bad"].error))

    def test_process_backend_runs_picklable_tasks(self) -> None:
        runtime = ExecutionRuntime()
        batch = runtime.map(
            (
                ExecutionTask(task_id="p0", fn=_double, args=(4,)),
                ExecutionTask(task_id="p1", fn=_double, args=(6,)),
            ),
            backend="process",
            max_workers=2,
        )

        self.assertEqual(str(batch.backend), "process")
        self.assertEqual([record.value for record in batch.records], [8, 12])

    def test_fail_fast_raises_runtime_error(self) -> None:
        runtime = ExecutionRuntime()
        with self.assertRaises(ExecutionRuntimeError):
            runtime.map(
                (
                    ExecutionTask(task_id="bad", fn=_explode, args=(1,)),
                    ExecutionTask(task_id="ok", fn=_double, args=(2,)),
                ),
                backend="serial",
                fail_fast=True,
            )


if __name__ == "__main__":
    unittest.main()
