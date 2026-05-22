from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from mlblack.integrations.nsgablack_symbolic import (
    BasisConditionedSymbolicTaskConfig,
    BasisConditionedSymbolicTaskProblem,
    OrthogonalBasisOuterProblem,
    OrthogonalBasisOuterProblemConfig,
    OrthogonalBasisSetArtifact,
)
from mlblack.pipeline.data import NumericDataView

try:
    from ..config import SymbolicOrthogonalNestedCaseConfig
except ImportError:  # direct script mode: case root is on sys.path
    from config import SymbolicOrthogonalNestedCaseConfig


def build_stage1_problem(
    cfg: SymbolicOrthogonalNestedCaseConfig,
    data: NumericDataView,
    *,
    output_dir: Path,
    resource_context: Mapping[str, Any] | None = None,
) -> OrthogonalBasisOuterProblem:
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_cfg = OrthogonalBasisOuterProblemConfig(
        basis_size=int(cfg.stage1_basis_size),
        pool_max_terms=int(cfg.stage1_pool_max_terms),
        inner_steps=int(cfg.stage1_inner_steps),
        inner_population_size=int(cfg.stage1_inner_population_size),
        random_seed=int(cfg.seed),
        enable_path_memory=bool(cfg.enable_path_memory),
        path_memory_db_path=str(output_dir / "symbolic_path_memory.sqlite3") if cfg.enable_path_memory else "",
        path_memory_namespace="symbolic_orthogonal_nested_stage1",
        enable_graph_cache=bool(cfg.enable_graph_cache),
        graph_cache_backend="sqlite" if cfg.enable_graph_cache else "memory",
        graph_cache_db_path=str(output_dir / "symbolic_graph_cache.sqlite3"),
    )
    return OrthogonalBasisOuterProblem(
        data,
        config=problem_cfg,
        resource_context=dict(resource_context or cfg.resource_context),
    )


def build_stage2_problem(
    cfg: SymbolicOrthogonalNestedCaseConfig,
    data: NumericDataView,
    *,
    basis_artifact: OrthogonalBasisSetArtifact,
    output_dir: Path,
    resource_context: Mapping[str, Any] | None = None,
) -> BasisConditionedSymbolicTaskProblem:
    output_dir.mkdir(parents=True, exist_ok=True)
    task_cfg = BasisConditionedSymbolicTaskConfig(
        task_kind=str(cfg.stage2_task_kind),
        head_kind=str(cfg.stage2_head_kind),
        task_terms=int(cfg.stage2_task_terms),
        pool_max_terms=int(cfg.stage2_pool_max_terms),
        inner_steps=int(cfg.stage2_inner_steps),
        inner_population_size=int(cfg.stage2_inner_population_size),
        learning_rate=float(cfg.stage2_learning_rate),
        random_seed=int(cfg.seed) + 101,
        enable_path_memory=bool(cfg.enable_path_memory),
        path_memory_db_path=str(output_dir / "symbolic_path_memory.sqlite3") if cfg.enable_path_memory else "",
        path_memory_namespace="symbolic_orthogonal_nested_stage2",
        enable_graph_cache=bool(cfg.enable_graph_cache),
        graph_cache_backend="sqlite" if cfg.enable_graph_cache else "memory",
        graph_cache_db_path=str(output_dir / "symbolic_graph_cache.sqlite3"),
        function_pool_config={
            "max_terms": int(cfg.stage2_pool_max_terms),
            "recursive_depth": 2,
            "pair_top_k": 16,
        },
    )
    return BasisConditionedSymbolicTaskProblem(
        data,
        basis_artifact=basis_artifact,
        config=task_cfg,
        resource_context=dict(resource_context or cfg.resource_context),
    )
