from __future__ import annotations

import inspect
from typing import Any, Mapping, MutableMapping, Sequence
from uuid import uuid4

import numpy as np

from nsgablack.plugins.base import Plugin, PluginManager

from .adapter import OptimizerAdapter
from .backend_session import ComputeBackendSession, ComputeBackendSpec
from .capability import Capability
from .problem import LearningProblem
from .representation import ModelRepresentation
from .resources import ResourceContext, coerce_resource_context
from .stores import InMemoryContextStore, InMemorySnapshotStore
from .state import build_trainer_state
from .types import Feedback, PopulationSnapshot, TrainerResult, UnknownState


class BlankTrainer:
    """Control-plane base class for ML-as-optimization.

    This is the mlblack equivalent of nsgablack's blank solver:
    it owns lifecycle, context/snapshot access, representation flow and
    evaluation entry points. It does not own an optimization algorithm.
    """

    def __init__(
        self,
        *,
        problem: LearningProblem | None = None,
        representation: ModelRepresentation | None = None,
        run_name: str = "trainer_run",
        constraint_penalty: float = 1e6,
        resource_context: Mapping[str, Any] | ResourceContext | None = None,
        compute_backend: str | Mapping[str, Any] | ComputeBackendSpec | ComputeBackendSession | None = None,
    ) -> None:
        self.problem = problem
        self.representation_pipeline = representation
        self.run_name = str(run_name)
        self.constraint_penalty = float(constraint_penalty)
        self.resource_context = coerce_resource_context(resource_context)
        self.compute_backend_session = (
            compute_backend
            if isinstance(compute_backend, ComputeBackendSession)
            else ComputeBackendSession(
                ComputeBackendSpec.from_value(
                    compute_backend,
                    resource_context=self.resource_context.as_dict(),
                )
            )
        )

        self.plugin_manager = PluginManager()
        self._capability_adapters: list[_CapabilityPluginAdapter] = []  # backward compat
        self.biases: list[Any] = []
        self.context_store: MutableMapping[str, Any] = InMemoryContextStore()
        self.snapshot_store: MutableMapping[str, Any] = InMemorySnapshotStore()
        self.history: list[dict[str, Any]] = []

        self.step_index = 0
        self.population: tuple[UnknownState, ...] = tuple()
        self.feedback: tuple[Feedback, ...] = tuple()
        self.best_state: UnknownState | None = None
        self.best_model: Any | None = None
        self.best_feedback: Feedback | None = None
        self.best_score: float | None = None

        self._l0_pool: Any = None  # PoolScheduler, created when threads > 1

    @property
    def representation(self) -> ModelRepresentation | None:
        """Compatibility alias for representation_pipeline."""

        return self.representation_pipeline

    @representation.setter
    def representation(self, value: ModelRepresentation | None) -> None:
        self.representation_pipeline = value

    @property
    def context(self) -> MutableMapping[str, Any]:
        """Compatibility alias for the lightweight context store."""

        return self.context_store

    def set_problem(self, problem: LearningProblem) -> "BlankTrainer":
        self.problem = problem
        return self

    def set_representation_pipeline(self, representation: ModelRepresentation) -> "BlankTrainer":
        self.representation_pipeline = representation
        return self

    def set_representation(self, representation: ModelRepresentation) -> "BlankTrainer":
        return self.set_representation_pipeline(representation)

    def add_plugin(self, plugin: Plugin) -> "BlankTrainer":
        """Register a nsgablack Plugin (unified capability layer)."""
        self.plugin_manager.register(plugin)
        plugin.attach(self)
        return self

    def add_capability(self, capability: Capability) -> "BlankTrainer":
        """Backward-compat: wrap a legacy Capability as a Plugin adapter."""
        adapter = _CapabilityPluginAdapter(capability)
        self._capability_adapters.append(adapter)
        return self.add_plugin(adapter)

    def add_bias(self, bias: Any) -> "BlankTrainer":
        self.biases.append(bias)
        return self

    def set_context_store(self, store: MutableMapping[str, Any]) -> "BlankTrainer":
        self.context_store = store
        return self

    def set_snapshot_store(self, store: MutableMapping[str, Any]) -> "BlankTrainer":
        self.snapshot_store = store
        return self

    def set_resource_context(self, context: Mapping[str, Any] | ResourceContext | None) -> "BlankTrainer":
        self.resource_context = coerce_resource_context(context)
        return self

    def get_resource_context(self) -> ResourceContext:
        return self.resource_context

    def get_population(self) -> tuple[UnknownState, ...]:
        return tuple(self.population)

    def set_population(self, population: Sequence[UnknownState]) -> "BlankTrainer":
        self.population = tuple(population)
        return self

    def get_feedback(self) -> tuple[Feedback, ...]:
        return tuple(self.feedback)

    def set_feedback(self, feedback: Sequence[Feedback]) -> "BlankTrainer":
        self.feedback = tuple(feedback)
        return self

    def build_context(self, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        ctx = dict(self.context_store)
        resource_context = self.resource_context.as_dict()
        ctx.update(
            {
                "run_name": self.run_name,
                "step": int(self.step_index),
                "best_score": self.best_score,
                "best_state": None if self.best_state is None else self.best_state.as_array().copy(),
                "population_size": int(len(self.population)),
                "resource_context": resource_context,
                **self.resource_context.context_items(prefix="resource"),
                **self.compute_backend_session.context_items(),
            }
        )
        if extra:
            ctx.update(dict(extra))
        for bias in self.biases:
            project = getattr(bias, "project_context", None)
            if callable(project):
                ctx = dict(project(self, ctx))
        return ctx

    def get_context_projection(self) -> Mapping[str, Any]:
        return {
            "run_name": self.run_name,
            "step": int(self.step_index),
            "population_size": int(len(self.population)),
            "best_score": self.best_score,
            "best_state": None if self.best_state is None else self.best_state.as_array().tolist(),
            "last_population_snapshot": self.context_store.get("last_population_snapshot"),
            "resource_context": self.resource_context.as_dict(),
            "compute_backend": self.compute_backend_session.as_dict(),
        }

    def set_compute_backend(
        self,
        backend: str | Mapping[str, Any] | ComputeBackendSpec | ComputeBackendSession | None,
    ) -> "BlankTrainer":
        self.compute_backend_session = (
            backend
            if isinstance(backend, ComputeBackendSession)
            else ComputeBackendSession(
                ComputeBackendSpec.from_value(
                    backend,
                    resource_context=self.resource_context.as_dict(),
                )
            )
        )
        return self

    def require_compute_backend(self, requirements: Sequence[str], *, consumer: str = "") -> Any:
        return self.compute_backend_session.ensure(tuple(str(item) for item in requirements), consumer=consumer)

    def init_candidate(self, context: Mapping[str, Any] | None = None) -> UnknownState:
        if self.representation_pipeline is None:
            raise ValueError("Trainer requires representation_pipeline before init_candidate()")
        return self.representation_pipeline.init(dict(context or {}))

    def init_population(self, n: int, context: Mapping[str, Any] | None = None) -> tuple[UnknownState, ...]:
        if self.representation_pipeline is None:
            raise ValueError("Trainer requires representation_pipeline before init_population()")
        return self.representation_pipeline.init_batch(int(n), dict(context or {}))

    def repair_candidate(self, candidate: UnknownState, context: Mapping[str, Any] | None = None) -> UnknownState:
        if self.representation_pipeline is None:
            raise ValueError("Trainer requires representation_pipeline before repair_candidate()")
        repaired = self.representation_pipeline.repair(candidate, dict(context or {}))
        arr = np.asarray(repaired.values, dtype=float).reshape(-1)
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
            repaired = repaired.with_values(arr, repaired=True)
        return repaired

    def repair_population(
        self,
        population: Sequence[UnknownState],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[UnknownState, ...]:
        ctx = dict(context or {})
        return tuple(self.repair_candidate(candidate, ctx) for candidate in tuple(population))

    def encode_candidate(self, model: Any, context: Mapping[str, Any] | None = None) -> UnknownState:
        if self.representation_pipeline is None:
            raise ValueError("Trainer requires representation_pipeline before encode_candidate()")
        return self.representation_pipeline.encode(model, dict(context or {}))

    def decode_candidate(self, candidate: UnknownState, context: Mapping[str, Any] | None = None) -> Any:
        if self.representation_pipeline is None:
            raise ValueError("Trainer requires representation_pipeline before decode_candidate()")
        return self.representation_pipeline.decode(candidate, dict(context or {}))

    def decode_population(
        self,
        population: Sequence[UnknownState],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        if self.representation_pipeline is None:
            raise ValueError("Trainer requires representation_pipeline before decode_population()")
        return self.representation_pipeline.decode_batch(tuple(population), dict(context or {}))

    def evaluate_individual(
        self,
        candidate: UnknownState,
        context: Mapping[str, Any] | None = None,
    ) -> Feedback:
        if self.problem is None:
            raise ValueError("Trainer requires problem before evaluate_individual()")
        ctx = dict(context or {})
        self.plugin_manager.on_evaluate_start(candidate, ctx)
        repaired = self.repair_candidate(candidate, ctx)
        model = self.decode_candidate(repaired, ctx)
        feedback = self.problem.evaluate(model, repaired, ctx)
        self.plugin_manager.on_evaluate_end(candidate, feedback, ctx)
        return feedback

    def evaluate_population(
        self,
        population: Sequence[UnknownState],
        context: Mapping[str, Any] | None = None,
    ) -> list[Feedback]:
        ctx = dict(context or {})
        pool = self._l0_pool
        if pool is not None and len(population) >= 4:
            with pool.as_executor(pool.available()) as ex:
                results = list(ex.map(self.evaluate_individual, tuple(population), [ctx] * len(population)))
            return results
        return [self.evaluate_individual(candidate, ctx) for candidate in tuple(population)]

    def adjust_feedback_with_biases(
        self,
        population: Sequence[UnknownState],
        feedback: Sequence[Feedback],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[Feedback, ...]:
        adjusted = tuple(feedback)
        ctx = dict(context or {})
        for bias in self.biases:
            adjust = getattr(bias, "adjust_feedback", None)
            if callable(adjust):
                adjusted = tuple(adjust(self, tuple(population), adjusted, ctx))
        return adjusted

    def write_population_snapshot(
        self,
        population: Sequence[UnknownState],
        feedback: Sequence[Feedback],
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        key = f"population:{self.run_name}:{self.step_index}:{uuid4().hex}"
        snapshot = PopulationSnapshot(
            states=tuple(population),
            feedback=tuple(feedback),
            step=int(self.step_index),
            metadata=dict(metadata or {}),
        )
        self.snapshot_store[key] = snapshot
        self.context_store["last_population_snapshot"] = key
        return key

    def write_snapshot(
        self,
        payload: Any,
        *,
        key: str | None = None,
        context_key: str | None = None,
    ) -> str:
        snapshot_key = str(key or f"snapshot:{self.run_name}:{self.step_index}:{uuid4().hex}")
        self.snapshot_store[snapshot_key] = payload
        if context_key is not None:
            self.context_store[str(context_key)] = snapshot_key
        return snapshot_key

    def read_snapshot(self, snapshot_key: str) -> Any:
        return self.snapshot_store[str(snapshot_key)]

    def get_state(self) -> Mapping[str, Any]:
        return {
            "run_name": self.run_name,
            "step_index": int(self.step_index),
            "best_score": self.best_score,
            "best_state": None if self.best_state is None else self.best_state.as_array().tolist(),
            "history": tuple(dict(row) for row in self.history),
            "context": dict(self.context_store),
            "resource_context": self.resource_context.as_dict(),
            "compute_backend": self.compute_backend_session.as_dict(),
            "biases": [bias.describe() if hasattr(bias, "describe") else {"name": type(bias).__name__} for bias in self.biases],
        }

    def set_state(self, state: Mapping[str, Any]) -> "BlankTrainer":
        self.step_index = int(state.get("step_index", self.step_index))
        best_state = state.get("best_state")
        self.best_state = None if best_state is None else UnknownState(values=np.asarray(best_state, dtype=float))
        best_score = state.get("best_score")
        self.best_score = None if best_score is None else float(best_score)
        self.history = [dict(row) for row in state.get("history", tuple())]
        context = state.get("context")
        if isinstance(context, Mapping):
            self.context_store.clear()
            self.context_store.update(dict(context))
        resource_context = state.get("resource_context")
        if isinstance(resource_context, Mapping):
            self.resource_context = coerce_resource_context(resource_context)
        compute_backend = state.get("compute_backend")
        if isinstance(compute_backend, Mapping):
            self.compute_backend_session = ComputeBackendSession(ComputeBackendSpec.from_value(compute_backend))
        if self.best_state is not None and self.representation_pipeline is not None:
            self.best_model = self.decode_candidate(self.best_state, self.build_context())
        return self

    def build_trainer_state(self, *, metadata: Mapping[str, Any] | None = None) -> Any:
        return build_trainer_state(self, metadata=metadata)

    def update_best(
        self,
        candidate: UnknownState,
        feedback: Feedback,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        ctx = dict(context or {})
        score = feedback.scalar_score(constraint_penalty=self.constraint_penalty)
        if self.best_score is None or score < self.best_score:
            self.best_score = float(score)
            self.best_state = candidate
            self.best_feedback = feedback
            self.best_model = self.decode_candidate(candidate, ctx)

    def setup(self) -> None:
        return None

    def step(self, context: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        _ = context
        raise NotImplementedError(f"{type(self).__name__}.step(...) is not implemented")

    def teardown(self) -> None:
        return None

    def fit(self, max_steps: int = 100) -> TrainerResult:
        try:
            self.setup()
            start_ctx = self.build_context()
            self.plugin_manager.on_solver_init(self)
            start_step = int(self.step_index) + 1 if self.history else int(self.step_index)
            for offset in range(int(max_steps)):
                self.step_index = int(start_step + offset)
                self.step()
        except BaseException as exc:
            error_ctx = self.build_context({"error": type(exc).__name__})
            self.plugin_manager.on_error(exc, error_ctx)
            raise
        finally:
            self.teardown()

        report = self.build_report()
        end_ctx = self.build_context()
        self.plugin_manager.on_solver_finish({"report": report, "context": end_ctx})

        return TrainerResult(
            best_state=self.best_state,
            best_model=self.best_model,
            best_feedback=self.best_feedback,
            history=tuple(self.history),
            report=report,
        )

    run = fit

    def evaluate(
        self,
        candidate: UnknownState | np.ndarray,
        *,
        resource_context: Mapping[str, Any] | ResourceContext | None = None,
        max_steps: int | None = None,
        **kwargs,
    ) -> TrainerResult:
        """nsgablack inner_runtime_evaluator interface.

        Accepts a candidate state, optionally overrides resource context,
        runs fit, and returns the result. Compatible with nsgablack nested
        solver evaluation — no bridge layer needed.
        """
        if resource_context is not None:
            from mlblack.core.resources import coerce_resource_context
            self.resource_context = coerce_resource_context(resource_context)
        return self.fit(max_steps=int(max_steps or 1))

    def _history_row(
        self,
        *,
        step: int,
        population: Sequence[UnknownState],
        feedback: Sequence[Feedback],
        snapshot_key: str | None = None,
    ) -> dict[str, Any]:
        scores = [fb.scalar_score(constraint_penalty=self.constraint_penalty) for fb in feedback]
        best_idx = int(np.argmin(scores)) if scores else -1
        return {
            "step": int(step),
            "num_candidates": int(len(population)),
            "best_batch_index": best_idx,
            "best_batch_score": None if best_idx < 0 else float(scores[best_idx]),
            "global_best_score": self.best_score,
            "snapshot_key": snapshot_key,
            "metrics": {} if best_idx < 0 else dict(feedback[best_idx].metrics),
        }

    def build_report(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "status": "finished",
            "steps": int(len(self.history)),
            "best_score": self.best_score,
            "best_state": None if self.best_state is None else self.best_state.as_array().tolist(),
            "best_metrics": {} if self.best_feedback is None else dict(self.best_feedback.metrics),
            "representation": None
            if self.representation_pipeline is None
            else dict(self.representation_pipeline.describe()),
            "problem": None if self.problem is None else dict(self.problem.describe()),
            "resources": self.resource_context.as_dict(),
            "compute_backend": self.compute_backend_session.as_dict(),
            "biases": [bias.describe() if hasattr(bias, "describe") else {"name": type(bias).__name__} for bias in self.biases],
            "contracts": self.build_contract_report(),
            "state_signature": self.build_trainer_state().signature,
        }

    def build_contract_report(self) -> dict[str, Any]:
        contracts: dict[str, Any] = {}
        if self.representation_pipeline is not None and hasattr(self.representation_pipeline, "get_contract"):
            contracts["representation"] = self.representation_pipeline.get_contract().describe()
        if self.problem is not None and hasattr(self.problem, "get_contract"):
            contracts["problem"] = self.problem.get_contract().describe()
        plugin_contracts = []
        for plugin in self.plugin_manager.plugins:
            if hasattr(plugin, "get_context_contract"):
                plugin_contracts.append(plugin.get_context_contract())
        if plugin_contracts:
            contracts["plugins"] = plugin_contracts
        # Backward compat: legacy capabilities wrapped as plugins
        if self._capability_adapters:
            contracts["capabilities"] = [
                adapter._capability.get_contract().describe()
                for adapter in self._capability_adapters
                if hasattr(adapter._capability, "get_contract")
            ]
        if self.biases:
            contracts["biases"] = [
                bias.get_contract().describe()
                for bias in self.biases
                if hasattr(bias, "get_contract")
            ]
        return contracts

    # Compatibility aliases for the first prototype and common ML terminology.
    init_state = init_candidate
    repair_state = repair_candidate
    decode_state = decode_candidate
    encode_state = encode_candidate
    evaluate_state = evaluate_individual
    evaluate_states = evaluate_population


class ComposableTrainer(BlankTrainer):
    """Trainer with an optimizer adapter mounted as strategy plane."""

    def __init__(
        self,
        *,
        problem: LearningProblem | None = None,
        representation: ModelRepresentation | None = None,
        adapter: OptimizerAdapter | None = None,
        run_name: str = "trainer_run",
        constraint_penalty: float = 1e6,
        resource_context: Mapping[str, Any] | ResourceContext | None = None,
        compute_backend: str | Mapping[str, Any] | ComputeBackendSpec | ComputeBackendSession | None = None,
    ) -> None:
        super().__init__(
            problem=problem,
            representation=representation,
            run_name=run_name,
            constraint_penalty=constraint_penalty,
            resource_context=resource_context,
            compute_backend=compute_backend,
        )
        self.adapter = adapter

    def set_adapter(self, adapter: OptimizerAdapter) -> "ComposableTrainer":
        self.adapter = adapter
        return self

    def setup(self) -> None:
        if self.adapter is None:
            raise ValueError("ComposableTrainer requires adapter before fit()")
        if self.representation_pipeline is None:
            raise ValueError("ComposableTrainer requires representation_pipeline before fit()")
        if self.problem is None:
            raise ValueError("ComposableTrainer requires problem before fit()")
        requirements = _collect_backend_requirements(self.representation_pipeline, self.problem, self.adapter)
        if requirements:
            self.require_compute_backend(requirements, consumer="trainer.setup")
        _setup_component(self.representation_pipeline, self, self.build_context())
        self.adapter.setup(self)

        # L0: create shared thread pool from resource grant
        threads = int(self.resource_context.threads or 1)
        if threads > 1:
            from mlblack.core.resources.compute.pool import PoolScheduler
            self._l0_pool = PoolScheduler(threads)

    def step(self, context: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        if self.adapter is None:
            raise ValueError("ComposableTrainer requires adapter before step()")

        ctx = self.build_context(context)
        self.plugin_manager.on_generation_start(self.step_index)

        proposed = self.adapter.coerce_states(self.adapter.propose(self, ctx))
        population = self.repair_population(proposed, ctx)
        feedback = self.adjust_feedback_with_biases(population, tuple(self.evaluate_population(population, ctx)), ctx)

        if len(population) != len(feedback):
            raise ValueError("evaluate_population() must return one Feedback per candidate")

        self.population = tuple(population)
        self.feedback = tuple(feedback)

        for candidate, fb in zip(self.population, self.feedback):
            self.update_best(candidate, fb, ctx)

        self.adapter.update(self, self.population, self.feedback, ctx)
        snapshot_key = self.write_population_snapshot(
            self.population,
            self.feedback,
            metadata={"adapter": getattr(self.adapter, "name", type(self.adapter).__name__)},
        )
        row = self._history_row(
            step=self.step_index,
            population=self.population,
            feedback=self.feedback,
            snapshot_key=snapshot_key,
        )
        self.history.append(row)

        self.plugin_manager.on_generation_end(self.step_index)

        return row

    def teardown(self) -> None:
        if self.adapter is not None:
            self.adapter.teardown(self)

    def build_report(self) -> dict[str, Any]:
        report = super().build_report()
        report["adapter"] = None if self.adapter is None else {
            "name": getattr(self.adapter, "name", type(self.adapter).__name__),
            "state": dict(self.adapter.get_state()),
        }
        if self.adapter is not None and hasattr(self.adapter, "get_contract"):
            report.setdefault("contracts", {})["adapter"] = self.adapter.get_contract().describe()
        return report

    def get_state(self) -> Mapping[str, Any]:
        state = dict(super().get_state())
        state["adapter"] = None if self.adapter is None else {
            "name": getattr(self.adapter, "name", type(self.adapter).__name__),
            "state": dict(self.adapter.get_state()),
        }
        return state

    def set_state(self, state: Mapping[str, Any]) -> "ComposableTrainer":
        super().set_state(state)
        adapter_state = state.get("adapter")
        if self.adapter is not None and isinstance(adapter_state, Mapping):
            raw_state = adapter_state.get("state", adapter_state)
            if isinstance(raw_state, Mapping):
                self.adapter.set_state(raw_state)
        return self


Trainer = ComposableTrainer


def _collect_backend_requirements(*components: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for component in components:
        for item in tuple(getattr(component, "backend_requires", ()) or ()):
            key = str(item)
            if key and key not in seen:
                seen.add(key)
                out.append(key)
    return tuple(out)


def _setup_component(component: Any, trainer: Any, context: Mapping[str, Any]) -> None:
    setup = getattr(component, "setup", None)
    if not callable(setup):
        return
    signature = inspect.signature(setup)
    positional = [
        param
        for param in signature.parameters.values()
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 2:
        setup(trainer, context)
    else:
        setup(context)


class _CapabilityPluginAdapter(Plugin):
    """Bridges a legacy mlblack Capability into the unified nsgablack Plugin system.

    Translates Plugin lifecycle hooks → Capability hook calls, preserving
    backward compatibility for existing Capability implementations.
    """

    def __init__(self, capability: Capability, name: str | None = None):
        super().__init__(name=name or getattr(capability, "name", "capability_adapter"))
        self._capability = capability

    # -- Plugin hooks → Capability hooks --

    def on_solver_init(self, solver):
        ctx = getattr(solver, "context_store", {}) or {}
        self._capability.on_fit_start(solver, ctx)

    def on_generation_start(self, generation: int):
        ctx = getattr(self.solver, "context_store", {}) or {}
        self._capability.on_step_start(self.solver, ctx)

    def on_generation_end(self, generation: int):
        ctx = getattr(self.solver, "context_store", {}) or {}
        row = {}
        if self.solver and hasattr(self.solver, "history") and self.solver.history:
            row = self.solver.history[-1]
        self._capability.on_step_end(self.solver, ctx, row)

    def on_evaluate_start(self, candidate, context=None):
        self._capability.on_evaluate_start(self.solver, candidate, context or {})

    def on_evaluate_end(self, candidate, feedback, context=None):
        self._capability.on_evaluate_end(self.solver, candidate, feedback, context or {})

    def on_solver_finish(self, result):
        ctx = getattr(self.solver, "context_store", {}) or {}
        report = result.get("report", {}) if isinstance(result, dict) else {}
        self._capability.on_fit_end(self.solver, ctx, report)

    def on_error(self, error: BaseException, context=None):
        self._capability.on_error(self.solver, error, context or {})

