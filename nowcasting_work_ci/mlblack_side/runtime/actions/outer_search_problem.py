from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from core.execution import (
    ExecutionResourceGrant,
    ExecutionResourceRequest,
    assert_phase_resource_budget,
    coerce_execution_resource_request,
    detect_local_execution_offer,
    issue_execution_resource_grant,
    sum_execution_resource_requests,
)
from nowcasting_work_ci.build_solver import NowcastingSolverBuildConfig, build_nowcasting_solver
from nowcasting_work_ci.mlblack_side.problem.config import ProblemConfig, build_problem
from nowcasting_work_ci.mlblack_side.problem.problem_model import SymbolicSubsetSelectionProblem
from pipeline.feature_space import CandidateTerm

from ..config import RuntimeCliConfig


@dataclass(frozen=True)
class OuterEpochRun:
    problem: SymbolicSubsetSelectionProblem
    outer_meta: dict[str, Any]
    resource_budget: dict[str, Any]
    run: dict[str, Any]
    duration_sec: float


def build_problem_config(
    args: RuntimeCliConfig,
    *,
    branch_policy: object,
    objective_policy: object,
) -> ProblemConfig:
    return ProblemConfig(
        max_terms=int(max(2, args.max_terms)),
        ridge_l2=float(max(0.0, args.ridge_l2)),
        rolling_folds=int(max(1, args.rolling_folds)),
        rolling_val_ratio=float(np.clip(args.rolling_val_ratio, 0.05, 0.45)),
        min_train_ratio=0.4,
        random_seed=int(args.seed),
        inner_opt_enabled=bool(int(args.inner_opt_enabled)),
        inner_opt_adam_steps=int(max(0, args.inner_opt_adam_steps)),
        inner_opt_adam_lr=float(max(1e-8, args.inner_opt_adam_lr)),
        inner_opt_lbfgs_steps=int(max(0, args.inner_opt_lbfgs_steps)),
        inner_opt_lbfgs_lr=float(max(1e-8, args.inner_opt_lbfgs_lr)),
        inner_opt_accept_rmse_tol=float(max(0.0, args.inner_opt_accept_rmse_tol)),
        inner_opt_accept_rel_tol=float(max(0.0, args.inner_opt_accept_rel_tol)),
        inner_opt_guard_patience=int(max(1, args.inner_opt_guard_patience)),
        inner_opt_guard_check_interval=int(max(1, args.inner_opt_guard_check_interval)),
        inner_opt_alt_freeze_readout=bool(int(args.inner_opt_alt_freeze_readout)),
        inner_opt_grad_clip_norm=float(max(0.0, args.inner_opt_grad_clip_norm)),
        inner_opt_residual_clip_q=float(np.clip(args.inner_opt_residual_clip_q, 0.70, 0.999)),
        interval_alpha=float(np.clip(args.interval_alpha, 1e-6, 0.99)),
        interval_method=str(args.interval_method),
        interval_calib_ratio=float(np.clip(args.interval_calib_ratio, 0.05, 0.4)),
        interval_quantile_l2=float(max(0.0, args.interval_quantile_l2)),
        branch_policy=branch_policy,
        objective_policy=objective_policy,
    )


def run_outer_epoch(
    args: RuntimeCliConfig,
    prepared: Mapping[str, Any],
    *,
    run_candidates: Sequence[CandidateTerm],
    generations_this_epoch: int,
    seed_this_epoch: int,
) -> OuterEpochRun:
    X_train = np.asarray(prepared["X_train"], dtype=float)
    y_train = np.asarray(prepared["y_train"], dtype=float)
    problem_local = build_problem(
        X_fit=X_train,
        y_fit=y_train,
        candidates=run_candidates,
        cfg=build_problem_config(
            args,
            branch_policy=prepared["branch_policy"],
            objective_policy=prepared["objective_policy"],
        ),
        branch_resolution=prepared["branch_resolution"],
        strict4_workers=int(prepared["effective_strict4_workers"]),
        graph_cache=prepared["graph_cache"],
    )
    solver_cfg = _build_outer_solver_cfg(
        args,
        prepared,
        generations_this_epoch=generations_this_epoch,
        seed_this_epoch=seed_this_epoch,
    )
    solver_local, outer_meta_local, _adapter_local = build_nowcasting_solver(
        problem=problem_local,
        cfg=solver_cfg,
        eval_batch_proxy_fn=problem_local.evaluate_population_batch if bool(prepared["batched_eval_enabled"]) else None,
    )
    resource_budget = _assert_nowcasting_outer_epoch_budget(
        args,
        prepared,
        problem_local=problem_local,
        outer_meta=outer_meta_local,
        generations_this_epoch=generations_this_epoch,
        seed_this_epoch=seed_this_epoch,
    )
    _bind_problem_execution_grant(problem_local, resource_budget)
    t0_local = time.perf_counter()
    run_local = solver_local.run()
    outer_sec_local = float(time.perf_counter() - t0_local)
    return OuterEpochRun(
        problem=problem_local,
        outer_meta=dict(outer_meta_local),
        resource_budget=dict(resource_budget),
        run=dict(run_local),
        duration_sec=outer_sec_local,
    )


def _problem_resource_requests(problem_local: SymbolicSubsetSelectionProblem) -> tuple[ExecutionResourceRequest, ...]:
    many_getter = getattr(problem_local, "execution_resource_requests", None)
    if callable(many_getter):
        return tuple(coerce_execution_resource_request(item) for item in tuple(many_getter()))
    one_getter = getattr(problem_local, "execution_resource_request", None)
    if callable(one_getter):
        return (coerce_execution_resource_request(one_getter()),)
    return tuple()


def _problem_total_request(problem_local: SymbolicSubsetSelectionProblem) -> ExecutionResourceRequest:
    components = _problem_resource_requests(problem_local)
    if not components:
        return ExecutionResourceRequest(
            threads=1,
            backend="serial",
            label="symbolic_subset_selection_problem",
        )
    return sum_execution_resource_requests(components, label="symbolic_subset_selection_problem")


def _build_problem_execution_grant(
    problem_local: SymbolicSubsetSelectionProblem,
    *,
    outer_strategy: str,
    dominant_phase: str,
) -> ExecutionResourceGrant:
    total_request = _problem_total_request(problem_local)
    return issue_execution_resource_grant(
        total_request,
        phase="mlblack_inner_problem",
        label="mlblack_inner_problem",
        metadata={
            "owner": "mlblack",
            "wrapper": "nowcasting_outer_bridge",
            "outer_strategy": str(outer_strategy),
            "dominant_phase": str(dominant_phase),
        },
    )


def _bind_problem_execution_grant(
    problem_local: SymbolicSubsetSelectionProblem,
    resource_budget: Mapping[str, Any],
) -> ExecutionResourceGrant | None:
    setter = getattr(problem_local, "set_execution_resource_grant", None)
    grant_payload = resource_budget.get("problem_grant")
    if not callable(setter) or not isinstance(grant_payload, Mapping):
        return None
    return setter(grant_payload)


def _build_outer_solver_cfg(
    args: RuntimeCliConfig,
    prepared: Mapping[str, Any],
    *,
    generations_this_epoch: int,
    seed_this_epoch: int,
) -> NowcastingSolverBuildConfig:
    return NowcastingSolverBuildConfig(
        strategy=str(args.outer_strategy),
        pop_size=int(prepared["effective_pop_size"]),
        generations=int(max(1, generations_this_epoch)),
        portfolio_phases_csv=str(args.portfolio_phases),
        portfolio_weights_csv=str(args.portfolio_phase_weights),
        moead_neighborhood_size=int(max(2, args.moead_neighborhood_size)),
        moead_delta=float(args.moead_delta),
        moead_nr=int(max(1, args.moead_nr)),
        vns_k_max=int(max(1, args.vns_k_max)),
        vns_batch_size=int(prepared["effective_vns_batch_size"]),
        batched_eval_enabled=bool(prepared["batched_eval_enabled"]),
        seed=int(seed_this_epoch),
    )


def _resolve_outer_phase_plan(
    args: RuntimeCliConfig,
    outer_meta: Mapping[str, Any],
    *,
    generations_this_epoch: int,
) -> tuple[dict[str, Any], ...]:
    phase_rows = outer_meta.get("portfolio_phases")
    if isinstance(phase_rows, Sequence) and not isinstance(phase_rows, (str, bytes, bytearray)):
        out: list[dict[str, Any]] = []
        for idx, row in enumerate(tuple(phase_rows)):
            if not isinstance(row, Mapping):
                continue
            name = str(row.get("name", f"phase_{idx}")).strip().lower() or f"phase_{idx}"
            steps = int(max(1, int(row.get("steps", 1) or 1)))
            out.append(
                {
                    "phase_index": int(idx),
                    "name": str(name),
                    "steps": int(steps),
                }
            )
        if out:
            return tuple(out)

    return (
        {
            "phase_index": 0,
            "name": str(args.outer_strategy).strip().lower() or "outer_search",
            "steps": int(max(1, generations_this_epoch)),
        },
    )


def _build_outer_phase_request(
    args: RuntimeCliConfig,
    prepared: Mapping[str, Any],
    *,
    phase_name: str,
    phase_steps: int,
    phase_index: int,
    generations_this_epoch: int,
    seed_this_epoch: int,
) -> ExecutionResourceRequest:
    metadata = {
        "outer_strategy": str(args.outer_strategy),
        "phase_name": str(phase_name),
        "phase_index": int(phase_index),
        "phase_steps": int(max(1, phase_steps)),
        "generations_this_epoch": int(max(1, generations_this_epoch)),
        "effective_pop_size": int(prepared["effective_pop_size"]),
        "effective_vns_batch_size": int(prepared["effective_vns_batch_size"]),
        "batched_eval_enabled": bool(prepared["batched_eval_enabled"]),
        "strict4_enabled": bool(prepared["strict4_enabled"]),
        "effective_strict4_workers": int(prepared["effective_strict4_workers"]),
        "seed_this_epoch": int(seed_this_epoch),
    }
    phase_key = str(phase_name).strip().lower()
    if phase_key == "nsga2":
        metadata.update(
            {
                "adapter_family": "nsga2",
                "population_size": int(prepared["effective_pop_size"]),
                "offspring_size": int(prepared["effective_pop_size"]),
            }
        )
    elif phase_key == "moead":
        metadata.update(
            {
                "adapter_family": "moead",
                "population_size": int(prepared["effective_pop_size"]),
                "neighborhood_size": int(max(2, args.moead_neighborhood_size)),
                "delta": float(args.moead_delta),
                "nr": int(max(1, args.moead_nr)),
            }
        )
    elif phase_key == "vns":
        metadata.update(
            {
                "adapter_family": "vns",
                "batch_size": int(prepared["effective_vns_batch_size"]),
                "k_max": int(max(1, args.vns_k_max)),
            }
        )
    else:
        metadata["adapter_family"] = str(phase_key or "unknown")

    return ExecutionResourceRequest(
        threads=1,
        backend="serial",
        label=f"outer_phase:{phase_key}:{int(phase_index)}",
        metadata=metadata,
    )


def _assert_nowcasting_outer_epoch_budget(
    args: RuntimeCliConfig,
    prepared: Mapping[str, Any],
    *,
    problem_local: SymbolicSubsetSelectionProblem,
    outer_meta: Mapping[str, Any],
    generations_this_epoch: int,
    seed_this_epoch: int,
) -> dict[str, Any]:
    offer = detect_local_execution_offer()
    problem_components = tuple(_problem_resource_requests(problem_local))
    phase_plan = _resolve_outer_phase_plan(
        args,
        outer_meta,
        generations_this_epoch=generations_this_epoch,
    )
    phase_budgets: list[dict[str, Any]] = []
    phase_requests: list[dict[str, Any]] = []

    for phase in phase_plan:
        control_request = _build_outer_phase_request(
            args,
            prepared,
            phase_name=str(phase["name"]),
            phase_steps=int(phase["steps"]),
            phase_index=int(phase["phase_index"]),
            generations_this_epoch=generations_this_epoch,
            seed_this_epoch=seed_this_epoch,
        )
        requests = (control_request, *problem_components)
        budget = assert_phase_resource_budget(
            f"nowcasting_outer_search:{str(phase['name'])}",
            requests,
            offer=offer,
        )
        budget["phase_name"] = str(phase["name"])
        budget["phase_index"] = int(phase["phase_index"])
        budget["phase_steps"] = int(phase["steps"])
        phase_budgets.append(dict(budget))
        phase_requests.append(control_request.as_dict())

    dominant_budget = max(
        phase_budgets,
        key=lambda row: (
            int(dict(row.get("total_request", {})).get("threads", 0)),
            int(dict(row.get("total_request", {})).get("gpus", 0)),
        ),
    )
    problem_grant = _build_problem_execution_grant(
        problem_local,
        outer_strategy=str(args.outer_strategy),
        dominant_phase=str(dominant_budget.get("phase_name", "")),
    )
    return {
        "phase": "nowcasting_outer_search",
        "offer": offer.as_dict(),
        "outer_strategy": str(args.outer_strategy),
        "generations_this_epoch": int(max(1, generations_this_epoch)),
        "effective_pop_size": int(prepared["effective_pop_size"]),
        "effective_vns_batch_size": int(prepared["effective_vns_batch_size"]),
        "strict4_enabled": bool(prepared["strict4_enabled"]),
        "effective_strict4_workers": int(prepared["effective_strict4_workers"]),
        "phase_plan": [dict(item) for item in phase_plan],
        "outer_phase_requests": list(phase_requests),
        "phase_budgets": list(phase_budgets),
        "dominant_phase": str(dominant_budget.get("phase_name", "")),
        "requests": list(dominant_budget.get("requests", [])),
        "total_request": dict(dominant_budget.get("total_request", {})),
        "problem_grant": problem_grant.as_dict(),
    }


__all__ = [
    "OuterEpochRun",
    "_assert_nowcasting_outer_epoch_budget",
    "_build_outer_phase_request",
    "_build_outer_solver_cfg",
    "_resolve_outer_phase_plan",
    "build_problem_config",
    "run_outer_epoch",
]
