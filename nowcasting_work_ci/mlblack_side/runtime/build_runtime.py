from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from pipeline.feature_space import augment_candidate_pool_with_conditional_config

from .assembly import (
    assemble_runtime_context,
    reg_branch,
    reg_budget,
    reg_conditional,
    reg_data,
    reg_dynamic_pool,
    reg_feature_space,
    reg_graph_cache,
    reg_objective,
)
from .config import RuntimeCliConfig


def build_runtime(args: RuntimeCliConfig) -> dict[str, Any]:
    data_runtime = reg_data(args)
    dynamic_runtime = reg_dynamic_pool(args)
    feature_runtime = reg_feature_space(
        args,
        train=data_runtime["train"],
        test=data_runtime["test"],
        dynamic_runtime=dynamic_runtime,
    )
    branch_runtime = reg_branch(
        args,
        feature_names=tuple(str(v) for v in feature_runtime["feature_bundle"].feature_names),
    )
    conditional_runtime = reg_conditional(
        args,
        feature_names=tuple(str(v) for v in feature_runtime["feature_bundle"].feature_names),
        branch_runtime=branch_runtime,
        X_train=np.asarray(feature_runtime["feature_bundle"].X_train, dtype=float),
    )
    augmented_candidates = augment_candidate_pool_with_conditional_config(
        feature_runtime["candidates"],
        X=np.asarray(feature_runtime["feature_bundle"].X_train, dtype=float),
        y=np.asarray(feature_runtime["feature_bundle"].y_train, dtype=float),
        feature_names=feature_runtime["feature_bundle"].feature_names,
        conditional_config=conditional_runtime["config"],
    )
    feature_runtime["candidates"] = list(augmented_candidates)
    feature_runtime["feature_space"] = replace(
        feature_runtime["feature_space"],
        candidates=tuple(augmented_candidates),
    )
    budget_runtime = reg_budget(args, branch_runtime=branch_runtime)
    objective_policy = reg_objective(args)
    graph_cache_runtime = reg_graph_cache(args)
    return assemble_runtime_context(
        args,
        data_runtime=data_runtime,
        feature_runtime=feature_runtime,
        branch_runtime=branch_runtime,
        conditional_runtime=conditional_runtime,
        budget_runtime=budget_runtime,
        dynamic_runtime=dynamic_runtime,
        objective_policy=objective_policy,
        graph_cache_runtime=graph_cache_runtime,
    )


__all__ = ["build_runtime"]
