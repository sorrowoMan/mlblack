"""ML-owned symbolic provider for nsgablack evaluation requests.

The provider accepts a structural request and never imports nsgablack internals.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blackbase.resources import coerce_resource_context

@dataclass(init=False)
class MlblackSymbolicConsensusBackendConfig:
    """Validated configuration for the ML-owned symbolic consensus backend."""

    benchmark_key: str
    n_total: int
    train_ratio: float
    noise_std: float
    dataset_seed: int
    output_root: str | None
    db_path: str | None
    namespace: str
    tag_prefix: str
    consensus_cycles: int
    unlocked_runs_per_cycle: int
    locked_runs_per_cycle: int
    vanilla_runs: int
    locked_runs: int
    search_seed_base: int
    locked_search_seed_base: int
    core_equivalence_mode: str
    core_max_terms: int
    orth_candidate_limit: int
    inner_steps: int
    inner_population_size: int
    stage2_inner_steps: int
    stage2_inner_population_size: int
    cache_results: bool
    save_artifact: bool
    save_report: bool
    fallback_objective: float

    def __init__(
        self,
        *,
        benchmark_key: str = "arrhenius_gate_like",
        n_total: int = 240,
        train_ratio: float = 0.8,
        noise_std: float = 0.025,
        dataset_seed: int = 42,
        output_root: str | None = None,
        db_path: str | None = None,
        namespace: str = "nsgablack_mlblack_symbolic_consensus",
        tag_prefix: str = "nsgablack",
        consensus_cycles: int = 1,
        unlocked_runs_per_cycle: int = 3,
        locked_runs_per_cycle: int = 2,
        vanilla_runs: int = 3,
        locked_runs: int = 2,
        search_seed_base: int = 100,
        locked_search_seed_base: int = 900,
        core_equivalence_mode: str = "family",
        core_max_terms: int = 4,
        orth_candidate_limit: int = 80,
        inner_steps: int = 2,
        inner_population_size: int = 4,
        stage2_inner_steps: int = 3,
        stage2_inner_population_size: int = 4,
        cache_results: bool = True,
        save_artifact: bool = True,
        save_report: bool = True,
        fallback_objective: float = 1.0e6,
    ) -> None:
        self.benchmark_key = str(benchmark_key)
        self.n_total = max(24, int(n_total))
        self.train_ratio = float(train_ratio)
        self.noise_std = max(0.0, float(noise_std))
        self.dataset_seed = int(dataset_seed)
        self.output_root = None if output_root is None else str(output_root)
        self.db_path = None if db_path is None else str(db_path)
        self.namespace = str(namespace)
        self.tag_prefix = str(tag_prefix)
        self.consensus_cycles = max(1, int(consensus_cycles))
        self.unlocked_runs_per_cycle = max(1, int(unlocked_runs_per_cycle or vanilla_runs or 1))
        self.locked_runs_per_cycle = max(0, int(locked_runs_per_cycle))
        self.vanilla_runs = max(1, int(vanilla_runs))
        self.locked_runs = max(0, int(locked_runs))
        self.search_seed_base = int(search_seed_base)
        self.locked_search_seed_base = int(locked_search_seed_base)
        self.core_equivalence_mode = str(core_equivalence_mode)
        self.core_max_terms = max(2, int(core_max_terms))
        self.orth_candidate_limit = max(8, int(orth_candidate_limit))
        self.inner_steps = max(1, int(inner_steps))
        self.inner_population_size = max(2, int(inner_population_size))
        self.stage2_inner_steps = max(1, int(stage2_inner_steps))
        self.stage2_inner_population_size = max(2, int(stage2_inner_population_size))
        self.cache_results = bool(cache_results)
        self.save_artifact = bool(save_artifact)
        self.save_report = bool(save_report)
        self.fallback_objective = float(fallback_objective)


class MlblackSymbolicConsensusBackend:
    """Evaluate an nsgablack candidate through maintained mlblack semantics."""

    protocol = "nsgablack_mlblack_symbolic_bridge_v3"

    def __init__(self, *, config: MlblackSymbolicConsensusBackendConfig | None = None) -> None:
        self.cfg = config or MlblackSymbolicConsensusBackendConfig()
        self._import_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._imported = False
        self._ml: dict[str, Any] = {}
        self._cache: dict[str, dict[str, Any]] = {}

    def _ensure_mlblack_imports(self) -> None:
        with self._import_lock:
            if self._imported:
                return
            from mlblack.integrations.nsgablack_symbolic import (
                BasisConditionedSymbolicTaskConfig,
                OrthogonalBasisOuterProblemConfig,
                build_symbolic_benchmark_data,
                build_symbolic_orthogonal_suite,
            )

            self._ml.update(
                {
                    "BasisConditionedSymbolicTaskConfig": BasisConditionedSymbolicTaskConfig,
                    "OrthogonalBasisOuterProblemConfig": OrthogonalBasisOuterProblemConfig,
                    "build_symbolic_benchmark_data": build_symbolic_benchmark_data,
                    "build_symbolic_orthogonal_suite": build_symbolic_orthogonal_suite,
                }
            )
            self._imported = True

    def solve(self, request: Any) -> Mapping[str, Any]:
        self._ensure_mlblack_imports()
        plan = self._resolve_plan(request)
        signature = self._signature(request, plan)
        if bool(self.cfg.cache_results) and not bool(plan["force_recompute"]):
            with self._cache_lock:
                cached = self._cache.get(signature)
            if cached is not None:
                return dict(cached)

        result = self._solve_current(request, plan=plan, signature=signature)
        if bool(self.cfg.cache_results):
            with self._cache_lock:
                self._cache[signature] = dict(result)
        return result

    def _solve_current(
        self,
        request: Any,
        *,
        plan: Mapping[str, Any],
        signature: str,
    ) -> dict[str, Any]:
        build_data = self._ml["build_symbolic_benchmark_data"]
        build_suite = self._ml["build_symbolic_orthogonal_suite"]
        Stage1Config = self._ml["OrthogonalBasisOuterProblemConfig"]
        Stage2Config = self._ml["BasisConditionedSymbolicTaskConfig"]

        resource_context = self._child_resource_context(request.eval_context)
        data = build_data(
            str(plan["benchmark_key"]),
            n_total=int(plan["n_total"]),
            train_ratio=float(plan["train_ratio"]),
            noise_std=float(plan["noise_std"]),
            seed=int(plan["dataset_seed"]),
        )
        stage1_config = Stage1Config(
            basis_size=int(plan["basis_size"]),
            pool_max_terms=int(plan["pool_max_terms"]),
            inner_steps=int(plan["inner_steps"]),
            inner_population_size=int(plan["inner_population_size"]),
            random_seed=int(plan["dataset_seed"]),
            enable_path_memory=False,
            enable_graph_cache=True,
            graph_cache_backend="memory",
            metadata={
                "benchmark_key": str(plan["benchmark_key"]),
                "truth_contracts": list(data.metadata.get("truth_contracts", ())),
                "bridge_protocol": self.protocol,
            },
        )
        stage2_config = Stage2Config(
            task_kind="regression",
            head_kind="point",
            task_terms=int(plan["basis_size"]),
            pool_max_terms=int(plan["pool_max_terms"]),
            inner_steps=int(plan["stage2_inner_steps"]),
            inner_population_size=int(plan["stage2_inner_population_size"]),
            inner_adapter="random_search",
            random_seed=int(plan["dataset_seed"]) + 101,
            enable_path_memory=False,
            enable_graph_cache=True,
            graph_cache_backend="memory",
            metadata={
                "benchmark_key": str(plan["benchmark_key"]),
                "truth_contracts": list(data.metadata.get("truth_contracts", ())),
                "bridge_protocol": self.protocol,
            },
        )
        suite = build_suite(
            data,
            stage1_config=stage1_config,
            stage2_config=stage2_config,
            resource_context=resource_context,
        )

        rng = np.random.default_rng(int(plan["search_seed_base"]))
        candidate = np.asarray(request.candidate, dtype=float).reshape(-1)
        cycle_reports: list[dict[str, Any]] = []
        stage_reports: list[dict[str, Any]] = []
        best_task_record = None
        best_basis_artifact = None
        best_task_artifact = None

        for cycle_index in range(int(plan["consensus_cycles"])):
            stage1_records = []
            for trial in range(int(plan["stage1_trials"])):
                stage_candidate = self._problem_candidate(
                    suite.stage1_problem,
                    source=candidate,
                    rng=rng,
                    trial=cycle_index * int(plan["stage1_trials"]) + trial,
                )
                stage1_records.append(suite.stage1_problem.evaluate_detailed(stage_candidate))
            basis_artifact = suite.stage1_problem.build_artifact()
            stage2_problem = suite.build_stage2_from_artifact(basis_artifact)
            stage2_records = []
            for trial in range(int(plan["stage2_trials"])):
                task_candidate = self._problem_candidate(
                    stage2_problem,
                    source=candidate[::-1],
                    rng=rng,
                    trial=cycle_index * int(plan["stage2_trials"]) + trial,
                )
                record = stage2_problem.evaluate_detailed(task_candidate)
                stage2_records.append(record)
                if best_task_record is None or self._record_score(record) < self._record_score(best_task_record):
                    best_task_record = record
                    best_basis_artifact = basis_artifact
                    best_task_artifact = stage2_problem.build_artifact(record)

            stage1_best = min(stage1_records, key=self._record_score)
            stage2_best = min(stage2_records, key=self._record_score)
            cycle_report = {
                "cycle_index": int(cycle_index),
                "cycle_key": f"cycle_{cycle_index:02d}",
                "unlocked_run_count": int(len(stage1_records)),
                "locked_run_count": int(len(stage2_records)),
                "core_basis_count": int(len(basis_artifact.selected_indices)),
                "locked_seed_terms": int(len(basis_artifact.selected_indices)),
                "unlocked_best_run": stage1_best.as_dict(),
                "locked_best_run": stage2_best.as_dict(),
                "resource_context": dict(resource_context),
            }
            cycle_reports.append(cycle_report)
            stage_reports.extend(
                [
                    {"cycle_index": int(cycle_index), "stage": "orthogonal_basis", "best": stage1_best.as_dict()},
                    {"cycle_index": int(cycle_index), "stage": "basis_conditioned_task", "best": stage2_best.as_dict()},
                ]
            )

        if best_task_record is None or best_basis_artifact is None or best_task_artifact is None:
            raise RuntimeError("mlblack symbolic bridge produced no task result")

        truth = dict(best_task_artifact.metadata.get("truth_contract_recovery", {}) or {})
        if not truth:
            truth = dict(best_basis_artifact.metadata.get("truth_contract_recovery", {}) or {})
        exact_score = float(truth.get("exact_term_recovery_score", 0.0) or 0.0)
        family_score = float(truth.get("family_recovery_score", 0.0) or 0.0)
        metrics = dict(best_task_record.metrics)
        best_rmse = float(metrics.get("valid.rmse", metrics.get("train.rmse", self.cfg.fallback_objective)))
        objective_vector = np.asarray(
            [1.0 - exact_score, 1.0 - family_score, best_rmse, self._record_score(best_task_record)],
            dtype=float,
        )

        output_root = Path(str(plan["output_root"])).expanduser().resolve()
        result_dir = output_root / str(plan["benchmark_key"]) / signature
        result_dir.mkdir(parents=True, exist_ok=True)
        core_selection = {
            "artifact_id": str(best_basis_artifact.artifact_id),
            "selected_indices": list(best_basis_artifact.selected_indices),
            "selected_terms": [dict(row) for row in best_basis_artifact.selected_terms],
            "truth_contract_recovery": truth,
        }
        comparison = {
            "protocol": self.protocol,
            "basis_candidate_score": self._record_score(suite.stage1_problem.best_record),
            "task_candidate_score": self._record_score(best_task_record),
            "best_test_rmse": best_rmse,
            "exact_term_recovery_score": exact_score,
            "family_level_term_recovery_score": family_score,
        }
        basis_description = best_basis_artifact.describe(include_record=True)
        task_description = best_task_artifact.describe()
        basis_artifact_id = str(
            basis_description.get("artifact_id")
            or getattr(best_basis_artifact, "artifact_id", "")
        )
        task_artifact_id = str(
            task_description.get("artifact_id")
            or getattr(best_task_artifact, "artifact_id", "")
        )
        summary_payload = {
            "protocol": self.protocol,
            "signature": signature,
            "plan": dict(plan),
            "resource_context": dict(resource_context),
            "cycle_reports": cycle_reports,
            "stage_reports": stage_reports,
            "core_selection": core_selection,
            "comparison": comparison,
            "basis_artifact": basis_description,
            "task_artifact": task_description,
        }
        summary_path = self._write_json(result_dir / "summary.json", summary_payload)
        comparison_path = self._write_json(result_dir / "comparison.json", comparison)
        core_selection_path = self._write_json(result_dir / "core_selection.json", core_selection)
        total_inner_runs = sum(
            int(row["unlocked_run_count"]) + int(row["locked_run_count"])
            for row in cycle_reports
        )
        result = {
            "status": "ok",
            "protocol": self.protocol,
            "benchmark_key": str(plan["benchmark_key"]),
            "best_phase": "basis_conditioned_task",
            "best_test_rmse": float(best_rmse),
            "best_exact_term_recovery_score": float(exact_score),
            "best_family_level_term_recovery_score": float(family_score),
            "best_outer_objective_score": float(np.sum(objective_vector)),
            "core_basis_count": int(len(best_basis_artifact.selected_indices)),
            "locked_seed_terms": int(len(best_basis_artifact.selected_indices)),
            "total_inner_runs": int(total_inner_runs),
            "consensus_cycles": int(plan["consensus_cycles"]),
            "cycle_reports": cycle_reports,
            "stage_reports": stage_reports,
            "summary_path": str(summary_path),
            "comparison_path": str(comparison_path),
            "core_selection_path": str(core_selection_path),
            "resource_context": dict(resource_context),
            "objectives": objective_vector.tolist(),
            "objective": float(np.sum(objective_vector)),
            "violation": 0.0,
            "metrics": {
                "mlblack.best_test_rmse": float(best_rmse),
                "mlblack.best_exact_term_recovery_score": float(exact_score),
                "mlblack.best_family_level_term_recovery_score": float(family_score),
                "mlblack.total_inner_runs": int(total_inner_runs),
            },
            "artifact_refs": {
                "summary": str(summary_path),
                "comparison": str(comparison_path),
                "core_selection": str(core_selection_path),
            },
            "payload": {
                "protocol": self.protocol,
                "basis_artifact_id": basis_artifact_id,
                "task_artifact_id": task_artifact_id,
                "truth_contract_recovery": truth,
            },
        }
        return _jsonable(result)

    def _resolve_plan(self, request: Any) -> dict[str, Any]:
        inner = dict(request.inner_problem) if isinstance(request.inner_problem, Mapping) else {}
        payload = dict(request.payload or {})
        overrides = dict(inner.get("trainer_params_overrides", {}) or {})
        candidate = np.asarray(request.candidate, dtype=float).reshape(-1)
        candidate_pool = int(round(abs(candidate[0]))) if candidate.size else int(self.cfg.orth_candidate_limit)
        candidate_basis = int(round(abs(candidate[3]))) if candidate.size > 3 else int(self.cfg.core_max_terms)
        output_root = (
            inner.get("output_root")
            or self.cfg.output_root
            or (Path.cwd() / "runs" / "mlblack_symbolic_bridge")
        )
        return {
            "benchmark_key": str(inner.get("benchmark_key") or self.cfg.benchmark_key),
            "n_total": max(24, int(inner.get("n_total", self.cfg.n_total))),
            "train_ratio": float(inner.get("train_ratio", self.cfg.train_ratio)),
            "noise_std": max(0.0, float(inner.get("noise_std", self.cfg.noise_std))),
            "dataset_seed": int(inner.get("dataset_seed", self.cfg.dataset_seed)),
            "output_root": str(Path(output_root).expanduser().resolve()),
            "basis_size": int(np.clip(candidate_basis, 2, 6)),
            "pool_max_terms": int(np.clip(candidate_pool, 8, 128)),
            "inner_steps": max(1, int(overrides.get("inner_steps", self.cfg.inner_steps))),
            "inner_population_size": max(
                2,
                int(overrides.get("inner_population_size", self.cfg.inner_population_size)),
            ),
            "stage2_inner_steps": max(
                1,
                int(overrides.get("stage2_inner_steps", self.cfg.stage2_inner_steps)),
            ),
            "stage2_inner_population_size": max(
                2,
                int(
                    overrides.get(
                        "stage2_inner_population_size",
                        self.cfg.stage2_inner_population_size,
                    )
                ),
            ),
            "consensus_cycles": max(1, int(inner.get("consensus_cycles", self.cfg.consensus_cycles))),
            "stage1_trials": max(
                1,
                int(
                    inner.get(
                        "unlocked_runs_per_cycle",
                        inner.get("vanilla_runs", self.cfg.unlocked_runs_per_cycle),
                    )
                ),
            ),
            "stage2_trials": max(
                1,
                int(
                    inner.get(
                        "locked_runs_per_cycle",
                        inner.get("locked_runs", self.cfg.locked_runs_per_cycle),
                    )
                    or 1
                ),
            ),
            "search_seed_base": int(inner.get("search_seed_base", self.cfg.search_seed_base)),
            "force_recompute": bool(inner.get("force_recompute", payload.get("force_recompute", False))),
        }

    def _child_resource_context(self, eval_context: Mapping[str, Any]) -> dict[str, Any]:
        raw: Any = {}
        for key in ("resource_context", "resource.context", "resource"):
            value = eval_context.get(key)
            if isinstance(value, Mapping):
                raw = value
                break
        parent = coerce_resource_context(raw)
        namespace = str(parent.namespace or self.cfg.namespace).strip()
        suffix = "mlblack_inner"
        if not parent.namespace and namespace:
            parent = coerce_resource_context({**parent.as_dict(), "namespace": namespace})
        return parent.derive_child(
            scope="training",
            namespace_suffix=suffix,
            metadata={"bridge_protocol": self.protocol},
        ).as_dict()

    @staticmethod
    def _problem_candidate(problem: Any, *, source: np.ndarray, rng: Any, trial: int) -> np.ndarray:
        dimension = int(problem.dimension)
        upper = np.asarray(
            [float(problem.bounds[f"x{i}"][1]) for i in range(dimension)],
            dtype=float,
        )
        if int(trial) == 0 and source.size:
            values = np.resize(np.abs(source), dimension)
            return np.mod(np.round(values), upper + 1.0)
        return np.asarray(
            [rng.integers(0, max(1, int(round(limit)) + 1)) for limit in upper],
            dtype=float,
        )

    @staticmethod
    def _record_score(record: Any) -> float:
        if record is None:
            return float("inf")
        report = dict(getattr(record, "report", {}) or {})
        candidate_score = dict(report.get("candidate_score", {}) or {})
        if candidate_score.get("score") is not None:
            return float(candidate_score["score"])
        objectives = np.asarray(getattr(record, "objectives", ()), dtype=float).reshape(-1)
        constraints = np.asarray(getattr(record, "constraints", ()), dtype=float).reshape(-1)
        return float(np.sum(objectives) + np.sum(np.maximum(constraints, 0.0)))

    def _signature(self, request: Any, plan: Mapping[str, Any]) -> str:
        payload = {
            "candidate": np.asarray(request.candidate, dtype=float).reshape(-1).tolist(),
            "plan": dict(plan),
            "resource": self._child_resource_context(request.eval_context),
            "protocol": self.protocol,
        }
        raw = json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:20]

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return _jsonable(as_dict())
    describe = getattr(value, "describe", None)
    if callable(describe):
        return _jsonable(describe())
    return repr(value)


__all__ = [
    "MlblackSymbolicConsensusBackend",
    "MlblackSymbolicConsensusBackendConfig",
]
