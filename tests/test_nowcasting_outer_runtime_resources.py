from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from core.execution import ExecutionBudgetError, ExecutionResourceOffer, ExecutionResourceRequest
from nowcasting_work_ci.mlblack_side.runtime.actions.outer_search_problem import (
    _assert_nowcasting_outer_epoch_budget,
    run_outer_epoch,
)
from nowcasting_work_ci.mlblack_side.runtime.config import RuntimeCliConfig


class _FakeProblem:
    def __init__(self, threads: int) -> None:
        self._threads = int(threads)
        self.evaluate_population_batch = lambda *args, **kwargs: (args, kwargs)
        self.bound_grant = None

    def execution_resource_requests(self):
        return (
            ExecutionResourceRequest(
                threads=int(self._threads),
                backend="thread",
                label="problem_eval",
                metadata={"rolling_folds": 3},
            ),
        )

    def set_execution_resource_grant(self, grant):
        self.bound_grant = grant
        return grant


class _FakeSolver:
    def run(self):
        return {"status": "completed", "steps_executed": 2}


def _prepared_payload() -> dict[str, object]:
    return {
        "X_train": np.zeros((12, 4), dtype=float),
        "y_train": np.zeros((12, 1), dtype=float),
        "branch_policy": object(),
        "objective_policy": object(),
        "branch_resolution": object(),
        "effective_strict4_workers": 4,
        "graph_cache": object(),
        "effective_pop_size": 16,
        "effective_vns_batch_size": 16,
        "batched_eval_enabled": True,
        "strict4_enabled": True,
    }


class TestNowcastingOuterRuntimeResources(unittest.TestCase):
    def test_nowcasting_outer_epoch_budget_hard_checks_problem_requests(self) -> None:
        args = RuntimeCliConfig()
        prepared = _prepared_payload()
        problem = _FakeProblem(threads=4)
        outer_meta = {
            "strategy": "portfolio",
            "portfolio_phases": (
                {"name": "nsga2", "steps": 2},
                {"name": "moead", "steps": 1},
                {"name": "vns", "steps": 1},
            ),
        }

        with patch(
            "nowcasting_work_ci.mlblack_side.runtime.actions.outer_search_problem.detect_local_execution_offer",
            return_value=ExecutionResourceOffer(threads=3),
        ):
            with self.assertRaises(ExecutionBudgetError):
                _assert_nowcasting_outer_epoch_budget(
                    args,
                    prepared,
                    problem_local=problem,  # type: ignore[arg-type]
                    outer_meta=outer_meta,
                    generations_this_epoch=5,
                    seed_this_epoch=77,
                )

    def test_run_outer_epoch_carries_resource_budget_in_result(self) -> None:
        args = RuntimeCliConfig()
        prepared = _prepared_payload()
        fake_problem = _FakeProblem(threads=3)

        with patch(
            "nowcasting_work_ci.mlblack_side.runtime.actions.outer_search_problem.build_problem",
            return_value=fake_problem,
        ), patch(
            "nowcasting_work_ci.mlblack_side.runtime.actions.outer_search_problem.build_nowcasting_solver",
            return_value=(
                _FakeSolver(),
                {
                    "strategy": "portfolio",
                    "portfolio_phases": (
                        {"name": "nsga2", "steps": 2},
                        {"name": "moead", "steps": 1},
                        {"name": "vns", "steps": 1},
                    ),
                },
                None,
            ),
        ), patch(
            "nowcasting_work_ci.mlblack_side.runtime.actions.outer_search_problem.detect_local_execution_offer",
            return_value=ExecutionResourceOffer(threads=6),
        ):
            result = run_outer_epoch(
                args,
                prepared,
                run_candidates=(),
                generations_this_epoch=4,
                seed_this_epoch=99,
            )

        self.assertEqual(str(result.run.get("status")), "completed")
        self.assertEqual(str(result.resource_budget.get("phase")), "nowcasting_outer_search")
        phase_plan = list(result.resource_budget.get("phase_plan", []))
        self.assertEqual([str(row.get("name")) for row in phase_plan], ["nsga2", "moead", "vns"])
        self.assertEqual([int(row.get("steps", 0)) for row in phase_plan], [2, 1, 1])
        phase_budgets = list(result.resource_budget.get("phase_budgets", []))
        self.assertEqual(len(phase_budgets), 3)
        self.assertEqual(str(result.resource_budget.get("dominant_phase")), "nsga2")
        total_request = dict(result.resource_budget.get("total_request", {}))
        self.assertEqual(int(total_request.get("threads", 0)), 4)
        problem_grant = dict(result.resource_budget.get("problem_grant", {}))
        self.assertEqual(str(problem_grant.get("phase")), "mlblack_inner_problem")
        self.assertEqual(int(problem_grant.get("threads", 0)), 3)
        requests = list(result.resource_budget.get("requests", []))
        self.assertEqual(len(requests), 2)
        self.assertEqual(str(requests[0].get("label")), "outer_phase:nsga2:0")
        self.assertEqual(str(requests[1].get("label")), "problem_eval")
        self.assertEqual(str(requests[0].get("backend")), "serial")
        self.assertEqual(int(dict(requests[0].get("metadata", {})).get("phase_steps", 0)), 2)
        self.assertIsNotNone(fake_problem.bound_grant)


if __name__ == "__main__":
    unittest.main()
