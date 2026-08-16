from __future__ import annotations

import inspect
import time
from typing import Any, Mapping, MutableMapping, Sequence
from uuid import uuid4

import numpy as np

from blackbase.adapters.mlblack.plugin import CapabilityPluginAdapter
from blackbase.context import (
    GENERIC_SNAPSHOT_SCHEMA,
    unwrap_snapshot_payload,
    wrap_snapshot_payload,
)
from blackbase.plugin import Plugin, PluginManager
from blackbase.resources import DataRef

from .adapter import OptimizerAdapter
from .artifact_provider import ArtifactProvider, CaseRuntimeArtifactProvider
from .backend_session import ComputeBackendSession, ComputeBackendSpec
from .capability import Capability
from .problem import LearningProblem
from .representation import ModelRepresentation, unknown_state_fingerprint
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
        parallel_evaluation: bool = False,
        artifact_provider: ArtifactProvider | None = None,
    ) -> None:
        self.problem = problem
        self.representation_pipeline = representation
        self.run_name = str(run_name)
        self.constraint_penalty = float(constraint_penalty)
        self._resource_context_explicit = resource_context is not None
        self.resource_context = coerce_resource_context(resource_context)
        self.case_runtime: Any | None = None
        self.artifact_provider: ArtifactProvider = (
            artifact_provider or CaseRuntimeArtifactProvider()
        )
        self._compute_backend_request = (
            compute_backend.spec if isinstance(compute_backend, ComputeBackendSession) else compute_backend
        )
        self.compute_backend_session = (
            compute_backend
            if isinstance(compute_backend, ComputeBackendSession) and not self._resource_context_explicit
            else self._build_compute_backend_session()
        )

        self.plugin_manager = PluginManager()
        self._capability_adapters: list[CapabilityPluginAdapter] = []  # backward compat
        self.biases: list[Any] = []
        self.context_store: MutableMapping[str, Any] = InMemoryContextStore()
        self.snapshot_store: MutableMapping[str, Any] = InMemorySnapshotStore()
        self.history: list[dict[str, Any]] = []
        self.parallel_evaluation = bool(parallel_evaluation)
        self._completion_policy: Any = None

        self.step_index = 0
        self.population: tuple[UnknownState, ...] = tuple()
        self.feedback: tuple[Feedback, ...] = tuple()
        self.last_evaluated_population: tuple[UnknownState, ...] = tuple()
        self.last_evaluated_feedback: tuple[Feedback, ...] = tuple()
        self._evaluation_seed_population: tuple[UnknownState, ...] = tuple()
        self.best_state: UnknownState | None = None
        self.best_model: Any | None = None
        self.best_model_ref: DataRef | None = None
        self.result_artifact_refs: dict[str, DataRef] = {}
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
        adapter = CapabilityPluginAdapter(capability)
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
        previous_session = self.compute_backend_session
        self._resource_context_explicit = context is not None
        self.resource_context = coerce_resource_context(context)
        self.compute_backend_session = self._build_compute_backend_session()
        previous_session.close()
        self._refresh_l0_pool_after_resource_change()
        return self

    def get_resource_context(self) -> ResourceContext:
        return self.resource_context

    def set_case_runtime(self, runtime: Any) -> "BlankTrainer":
        """Accept the shared Case runtime without owning orchestration."""

        self.case_runtime = runtime
        return self

    def set_artifact_provider(self, provider: ArtifactProvider) -> "BlankTrainer":
        if not callable(getattr(provider, "publish_best_model", None)):
            raise TypeError("artifact provider must implement publish_best_model(trainer)")
        self.artifact_provider = provider
        return self

    def checkpoint_case_runtime(self) -> None:
        checkpoint = getattr(self.case_runtime, "checkpoint", None)
        if callable(checkpoint):
            checkpoint()

    def get_population(self) -> tuple[UnknownState, ...]:
        return tuple(self.population)

    def set_population(self, population: Sequence[UnknownState]) -> "BlankTrainer":
        self.population = tuple(population)
        return self

    def get_feedback(self) -> tuple[Feedback, ...]:
        return tuple(self.feedback)

    def set_best_model_ref(self, ref: DataRef | Mapping[str, Any] | None) -> "BlankTrainer":
        """Attach the published artifact used by the transport result codec."""

        self.best_model_ref = (
            None
            if ref is None
            else ref
            if isinstance(ref, DataRef)
            else DataRef.from_dict(ref)
        )
        if self.best_model_ref is None:
            self.result_artifact_refs.pop("best_model", None)
        else:
            self.result_artifact_refs["best_model"] = self.best_model_ref
        return self

    def register_result_artifact(
        self,
        name: str,
        ref: DataRef | Mapping[str, Any],
    ) -> "BlankTrainer":
        item = ref if isinstance(ref, DataRef) else DataRef.from_dict(ref)
        self.result_artifact_refs[str(name)] = item
        return self

    def publish_best_model_artifact(self) -> DataRef | None:
        """Publish the selected model before constructing a transport result."""

        if self.best_model is None or self.best_model_ref is not None:
            return self.best_model_ref
        self.checkpoint_case_runtime()
        ref = self.artifact_provider.publish_best_model(self)
        if ref is not None:
            self.set_best_model_ref(ref)
        self.checkpoint_case_runtime()
        return self.best_model_ref

    def set_feedback(self, feedback: Sequence[Feedback]) -> "BlankTrainer":
        self.feedback = tuple(feedback)
        return self

    def set_parallel_evaluation(self, enabled: bool = True) -> "BlankTrainer":
        """Explicitly opt into thread-parallel candidate evaluation.

        The problem, representation and enabled evaluation plugins must each
        declare ``thread_safe_evaluation = True`` (or ``thread_safe = True``).
        A resource grant alone never implies component thread safety.
        """
        self.parallel_evaluation = bool(enabled)
        self._refresh_l0_pool_after_resource_change()
        return self

    def set_completion_policy(self, policy: Any) -> "BlankTrainer":
        """Install a cooperative policy checked between Trainer steps."""
        if policy is not None and not callable(getattr(policy, "is_complete", None)):
            raise TypeError("completion policy must provide is_complete(...) support")
        self._completion_policy = policy
        return self

    def build_context(self, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        ctx = self.context_store.snapshot()
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
        plugin_context = self.plugin_manager.on_context_build(ctx)
        if isinstance(plugin_context, dict):
            ctx = plugin_context
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
            "parallel_evaluation": bool(self.parallel_evaluation),
        }

    def set_compute_backend(
        self,
        backend: str | Mapping[str, Any] | ComputeBackendSpec | ComputeBackendSession | None,
    ) -> "BlankTrainer":
        previous_session = self.compute_backend_session
        self._compute_backend_request = backend.spec if isinstance(backend, ComputeBackendSession) else backend
        self.compute_backend_session = (
            backend
            if isinstance(backend, ComputeBackendSession) and not self._resource_context_explicit
            else self._build_compute_backend_session()
        )
        if previous_session is not self.compute_backend_session:
            previous_session.close()
        return self

    def _build_compute_backend_session(self) -> ComputeBackendSession:
        spec = ComputeBackendSpec.from_value(
            self._compute_backend_request,
            resource_context=self.resource_context.as_dict(),
        )
        if self._resource_context_explicit:
            granted_backend = str(self.resource_context.compute_backend or "auto").strip().lower()
            spec = ComputeBackendSpec(
                name=spec.name if granted_backend in {"", "auto", "cpu", "cuda", "gpu", "tpu"} else granted_backend,
                device=str(self.resource_context.device),
                device_policy=spec.device_policy,
                metadata={
                    **dict(spec.metadata),
                    "resource_namespace": str(self.resource_context.namespace),
                },
            )
        return ComputeBackendSession(spec)

    def _refresh_l0_pool_after_resource_change(self) -> None:
        pool = self._l0_pool
        if pool is not None:
            closer = getattr(pool, "shutdown", None)
            if callable(closer):
                closer(wait=True)
        self._l0_pool = None
        threads = int(self.resource_context.threads or 1)
        if self.parallel_evaluation and threads > 1:
            from mlblack.core.resources.compute.pool import PoolScheduler

            self._l0_pool = PoolScheduler(threads)

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
        self.checkpoint_case_runtime()
        if self.problem is None:
            raise ValueError("Trainer requires problem before evaluate_individual()")
        ctx = dict(context or {})
        self.plugin_manager.on_evaluate_start(candidate, ctx)
        repaired = self.repair_candidate(candidate, ctx)
        model = self.decode_candidate(repaired, ctx)
        feedback = self.problem.evaluate(model, repaired, ctx)
        self.plugin_manager.on_evaluate_end(candidate, feedback, ctx)
        self.checkpoint_case_runtime()
        return feedback

    def evaluate_population(
        self,
        population: Sequence[UnknownState],
        context: Mapping[str, Any] | None = None,
    ) -> list[Feedback]:
        self.checkpoint_case_runtime()
        ctx = dict(context or {})
        pool = self._l0_pool
        if self.parallel_evaluation and pool is not None and len(population) >= 4:
            unsafe = self._unsafe_parallel_evaluation_components()
            if unsafe:
                raise RuntimeError(
                    "parallel evaluation requires explicit thread-safety declarations: "
                    + ", ".join(unsafe)
                )
            with pool.as_executor(pool.available()) as ex:
                results = list(ex.map(self.evaluate_individual, tuple(population), [ctx] * len(population)))
            self.checkpoint_case_runtime()
            return results
        results = [self.evaluate_individual(candidate, ctx) for candidate in tuple(population)]
        self.checkpoint_case_runtime()
        return results

    def _unsafe_parallel_evaluation_components(self) -> tuple[str, ...]:
        components: list[tuple[str, Any]] = [
            ("problem", self.problem),
            ("representation", self.representation_pipeline),
        ]
        components.extend(
            (f"plugin.{plugin.name}", plugin)
            for plugin in self.plugin_manager.plugins
            if bool(getattr(plugin, "enabled", True))
        )
        unsafe: list[str] = []
        for name, component in components:
            if component is None:
                continue
            declared_safe = (
                getattr(component, "thread_safe_evaluation", None) is True
                or getattr(component, "thread_safe", None) is True
            )
            if not declared_safe:
                unsafe.append(name)
        return tuple(unsafe)

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
        *,
        evaluated_population: Sequence[UnknownState] | None = None,
        evaluated_feedback: Sequence[Feedback] | None = None,
    ) -> str:
        population_tuple = tuple(population)
        feedback_tuple = tuple(feedback)
        evaluated_population_tuple = tuple(
            population_tuple if evaluated_population is None else evaluated_population
        )
        evaluated_feedback_tuple = tuple(
            feedback_tuple if evaluated_feedback is None else evaluated_feedback
        )
        objectives = np.array([f.objectives for f in feedback_tuple]) if feedback_tuple else None
        constraints = np.array([f.constraints for f in feedback_tuple]) if feedback_tuple else None
        evaluated_objectives = (
            np.array([f.objectives for f in evaluated_feedback_tuple])
            if evaluated_feedback_tuple
            else None
        )
        evaluated_constraints = (
            np.array([f.constraints for f in evaluated_feedback_tuple])
            if evaluated_feedback_tuple
            else None
        )
        feedback_aligned = len(population_tuple) == len(feedback_tuple)
        snapshot_metadata = {
            **dict(metadata or {}),
            "feedback_aligned": bool(feedback_aligned),
        }

        snapshot_data = {
            "candidates": population_tuple,
            "objectives": objectives,
            "constraints": constraints,
            "evaluated_candidates": evaluated_population_tuple,
            "evaluated_objectives": evaluated_objectives,
            "evaluated_constraints": evaluated_constraints,
            "generation": int(self.step_index),
            "metadata": snapshot_metadata,
        }

        snapshot_key = f"population:{self.run_name}:{self.step_index}:{uuid4().hex}"
        self.checkpoint_case_runtime()
        handle = self.snapshot_store.write(
            snapshot_data,
            key=snapshot_key,
            meta=snapshot_metadata,
            schema="mlblack_population_snapshot_v2",
        )
        self.context_store.set("last_population_snapshot", handle.key)
        return handle.key

    def write_snapshot(
        self,
        payload: Any,
        *,
        key: str | None = None,
        context_key: str | None = None,
    ) -> str:
        snapshot_key = str(key or f"snapshot:{self.run_name}:{self.step_index}:{uuid4().hex}")
        self.checkpoint_case_runtime()
        self.snapshot_store.write(
            wrap_snapshot_payload(payload),
            key=snapshot_key,
            schema=GENERIC_SNAPSHOT_SCHEMA,
        )
        if context_key is not None:
            self.context_store.set(str(context_key), snapshot_key)
        return snapshot_key

    def read_snapshot(self, snapshot_key: str) -> Any:
        record = self.snapshot_store.read(str(snapshot_key))
        if record is None:
            return None
        payload = unwrap_snapshot_payload(record)
        # Backward compatibility for snapshots written as {snapshot_key: payload}.
        if isinstance(payload, Mapping) and set(payload) == {str(snapshot_key)}:
            return payload[str(snapshot_key)]
        return payload

    def get_state(self) -> Mapping[str, Any]:
        return {
            "state_version": 2,
            "run_name": self.run_name,
            "step_index": int(self.step_index),
            "best_score": self.best_score,
            "best_state": None if self.best_state is None else self.best_state.as_dict(),
            "best_model_ref": (
                None if self.best_model_ref is None else self.best_model_ref.as_dict()
            ),
            "result_artifact_refs": {
                name: ref.as_dict() for name, ref in self.result_artifact_refs.items()
            },
            "best_feedback": _feedback_to_state(self.best_feedback),
            "population": tuple(candidate.as_dict() for candidate in self.population),
            "feedback": tuple(_feedback_to_state(item) for item in self.feedback),
            "last_evaluated_population": tuple(
                candidate.as_dict() for candidate in self.last_evaluated_population
            ),
            "last_evaluated_feedback": tuple(
                _feedback_to_state(item) for item in self.last_evaluated_feedback
            ),
            "history": tuple(dict(row) for row in self.history),
            "context": dict(self.context_store),
            "resource_context": self.resource_context.as_dict(),
            "compute_backend": self.compute_backend_session.as_dict(),
            "parallel_evaluation": bool(self.parallel_evaluation),
            "biases": [bias.describe() if hasattr(bias, "describe") else {"name": type(bias).__name__} for bias in self.biases],
        }

    def set_state(self, state: Mapping[str, Any]) -> "BlankTrainer":
        self.run_name = str(state.get("run_name", self.run_name))
        self.parallel_evaluation = bool(
            state.get("parallel_evaluation", self.parallel_evaluation)
        )
        self.step_index = int(state.get("step_index", self.step_index))
        best_state = state.get("best_state")
        self.best_state = None if best_state is None else _unknown_state_from_state(best_state)
        self.best_feedback = _feedback_from_state(state.get("best_feedback"))
        raw_best_model_ref = state.get("best_model_ref")
        self.best_model_ref = (
            DataRef.from_dict(raw_best_model_ref)
            if isinstance(raw_best_model_ref, Mapping)
            else None
        )
        self.result_artifact_refs = {
            str(name): DataRef.from_dict(ref)
            for name, ref in dict(state.get("result_artifact_refs", {}) or {}).items()
            if isinstance(ref, Mapping)
        }
        if self.best_model_ref is not None:
            self.result_artifact_refs.setdefault("best_model", self.best_model_ref)
        best_score = state.get("best_score")
        self.best_score = None if best_score is None else float(best_score)
        self.population = tuple(
            _unknown_state_from_state(item) for item in state.get("population", tuple())
        )
        self.feedback = tuple(
            item
            for item in (_feedback_from_state(raw) for raw in state.get("feedback", tuple()))
            if item is not None
        )
        self.last_evaluated_population = tuple(
            _unknown_state_from_state(item)
            for item in state.get("last_evaluated_population", tuple())
        )
        self.last_evaluated_feedback = tuple(
            item
            for item in (
                _feedback_from_state(raw)
                for raw in state.get("last_evaluated_feedback", tuple())
            )
            if item is not None
        )
        self.history = [dict(row) for row in state.get("history", tuple())]
        context = state.get("context")
        if isinstance(context, Mapping):
            self.context_store.clear()
            self.context_store.update(dict(context))
        resource_context = state.get("resource_context")
        if isinstance(resource_context, Mapping):
            self.set_resource_context(resource_context)
        compute_backend = state.get("compute_backend")
        if isinstance(compute_backend, Mapping):
            self.set_compute_backend(compute_backend)
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
            # A newly selected model invalidates any artifact published for the
            # previous best.  A plugin/provider may publish and attach a fresh
            # ref before the result envelope is serialized.
            self.set_best_model_ref(None)

    def setup(self) -> None:
        return None

    def step(self, context: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        _ = context
        raise NotImplementedError(f"{type(self).__name__}.step(...) is not implemented")

    def teardown(self) -> None:
        if self._l0_pool is not None:
            self._l0_pool.shutdown(wait=True)
            self._l0_pool = None

    def fit(self, max_steps: int = 100) -> TrainerResult:
        run_started = time.monotonic()
        try:
            self.checkpoint_case_runtime()
            self.setup()
            self.checkpoint_case_runtime()
            start_ctx = self.build_context()
            self.plugin_manager.on_solver_init(self)
            start_step = int(self.step_index) + 1 if self.history else int(self.step_index)
            for offset in range(int(max_steps)):
                self.checkpoint_case_runtime()
                if self._completion_policy is not None and self._completion_policy.is_complete(
                    step=int(offset),
                    elapsed=float(time.monotonic() - run_started),
                    ctx={"trainer": self, "run_name": self.run_name},
                ):
                    break
                self.step_index = int(start_step + offset)
                self.step()
                self.checkpoint_case_runtime()
        except BaseException as exc:
            error_ctx = self.build_context({"error": type(exc).__name__})
            self.plugin_manager.on_error(exc, error_ctx)
            raise
        finally:
            self.teardown()

        report = self.build_report()
        self.checkpoint_case_runtime()
        end_ctx = self.build_context()
        self.plugin_manager.on_solver_finish({"report": report, "context": end_ctx})
        # Plugins/providers get the finish hook first.  If none attached a
        # durable ref, the configured ML artifact provider publishes through
        # the shared Project authority.  The Trainer never invents a URI.
        self.publish_best_model_artifact()

        return TrainerResult(
            best_state=self.best_state,
            best_model=self.best_model,
            best_objectives=None
            if self.best_feedback is None
            else np.asarray(self.best_feedback.objectives, dtype=float),
            best_feedback=self.best_feedback,
            history=tuple(self.history),
            report=report,
            best_model_ref=self.best_model_ref,
            artifact_refs=self.result_artifact_refs,
        )

    def run(self, max_steps: int = 100) -> TrainerResult:
        """Dynamically dispatch to ``fit`` so subclass overrides are honored."""

        return self.fit(max_steps=max_steps)

    def evaluate(
        self,
        candidate: UnknownState | np.ndarray,
        *,
        resource_context: Mapping[str, Any] | ResourceContext | None = None,
        max_steps: int | None = None,
        **kwargs,
    ) -> TrainerResult:
        """Evaluate a candidate as the initial state of this Trainer run.

        Cross-framework callers should use the formal adapter in
        ``mlblack.integrations.nsgablack_trainer_evaluator``; this method
        deliberately returns a native ``TrainerResult``.
        """
        if resource_context is not None:
            self.set_resource_context(resource_context)
        state = candidate if isinstance(candidate, UnknownState) else UnknownState(values=candidate)
        self.population = (state,)
        self._evaluation_seed_population = self.population
        adapter = getattr(self, "adapter", None)
        set_population = getattr(adapter, "set_population", None)
        if callable(set_population):
            set_population(self.population)
        self.write_snapshot(
            state.to_protocol_payload(),
            key=f"input_candidate:{self.run_name}:{uuid4().hex}",
            context_key="input_candidate_snapshot",
        )
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
            "parallel_evaluation": {
                "enabled": bool(self.parallel_evaluation),
                "thread_safe": not bool(self._unsafe_parallel_evaluation_components()),
            },
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
        parallel_evaluation: bool = False,
    ) -> None:
        super().__init__(
            problem=problem,
            representation=representation,
            run_name=run_name,
            constraint_penalty=constraint_penalty,
            resource_context=resource_context,
            compute_backend=compute_backend,
            parallel_evaluation=parallel_evaluation,
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
        if self._evaluation_seed_population:
            self.population = tuple(self._evaluation_seed_population)
            self.adapter.set_population(self.population)

        # L0: create shared thread pool from resource grant
        if self._l0_pool is not None:
            self._l0_pool.shutdown(wait=True)
            self._l0_pool = None
        threads = int(self.resource_context.threads or 1)
        if self.parallel_evaluation and threads > 1:
            from mlblack.core.resources.compute.pool import PoolScheduler
            self._l0_pool = PoolScheduler(threads)

    def step(self, context: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        if self.adapter is None:
            raise ValueError("ComposableTrainer requires adapter before step()")

        propose_context = self.build_context(context)
        self.plugin_manager.on_generation_start(self.step_index)

        proposed = self.adapter.coerce_states(self.adapter.propose(self, propose_context))
        population = self.repair_population(proposed, propose_context)
        feedback = self.adjust_feedback_with_biases(
            population,
            tuple(self.evaluate_population(population, propose_context)),
            propose_context,
        )

        if len(population) != len(feedback):
            raise ValueError("evaluate_population() must return one Feedback per candidate")

        evaluated_population = tuple(population)
        evaluated_feedback = tuple(feedback)
        self.last_evaluated_population = evaluated_population
        self.last_evaluated_feedback = evaluated_feedback

        for candidate, fb in zip(evaluated_population, evaluated_feedback):
            self.update_best(candidate, fb, propose_context)

        update_context = self.build_context(context)
        self.adapter.update(self, evaluated_population, evaluated_feedback, update_context)
        authoritative_population = self._resolve_adapter_population(evaluated_population)
        aligned_feedback = _align_feedback(
            authoritative_population,
            evaluated_population,
            evaluated_feedback,
            representation=self.representation_pipeline,
        )
        self.population = authoritative_population
        self.feedback = aligned_feedback
        snapshot_key = self.write_population_snapshot(
            self.population,
            self.feedback,
            metadata={"adapter": getattr(self.adapter, "name", type(self.adapter).__name__)},
            evaluated_population=evaluated_population,
            evaluated_feedback=evaluated_feedback,
        )
        row = self._history_row(
            step=self.step_index,
            population=evaluated_population,
            feedback=evaluated_feedback,
            snapshot_key=snapshot_key,
        )
        self.history.append(row)

        self.plugin_manager.on_generation_end(self.step_index)

        return row

    def teardown(self) -> None:
        try:
            if self.adapter is not None:
                self.adapter.teardown(self)
        finally:
            if self._l0_pool is not None:
                self._l0_pool.shutdown(wait=True)
                self._l0_pool = None

    def _resolve_adapter_population(
        self,
        fallback: Sequence[UnknownState],
    ) -> tuple[UnknownState, ...]:
        if self.adapter is None:
            return tuple(fallback)
        getter = getattr(self.adapter, "get_population", None)
        if not callable(getter):
            return tuple(fallback)
        resolved = getter()
        if resolved is None:
            return tuple(fallback)
        return self.adapter.coerce_states(resolved)

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
        if self.adapter is not None and self.population:
            self.adapter.set_population(self.population)
        return self


Trainer = ComposableTrainer


def _unknown_state_from_state(value: Any) -> UnknownState:
    if isinstance(value, UnknownState):
        return value
    if isinstance(value, Mapping):
        return UnknownState.from_protocol_payload(value)
    return UnknownState(values=np.asarray(value, dtype=float))


def _feedback_to_state(value: Feedback | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return {
        "objectives": value.objectives.tolist(),
        "constraints": value.constraints.tolist(),
        "gradients": None if value.gradients is None else value.gradients.tolist(),
        "loss": value.loss,
        "metrics": dict(value.metrics),
        "residuals": None if value.residuals is None else value.residuals.tolist(),
        "signals": dict(value.signals),
        "info": dict(value.info),
    }


def _feedback_from_state(value: Any) -> Feedback | None:
    if value is None:
        return None
    if isinstance(value, Feedback):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("feedback state must be a mapping")
    gradients = value.get("gradients")
    residuals = value.get("residuals")
    return Feedback(
        objectives=np.asarray(value.get("objectives", ()), dtype=float),
        constraints=np.asarray(value.get("constraints", ()), dtype=float),
        gradients=None if gradients is None else np.asarray(gradients, dtype=float),
        loss=None if value.get("loss") is None else float(value.get("loss")),
        metrics=dict(value.get("metrics", {}) or {}),
        residuals=None if residuals is None else np.asarray(residuals, dtype=float),
        signals=dict(value.get("signals", {}) or {}),
        info=dict(value.get("info", {}) or {}),
    )


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


def _align_feedback(
    authoritative: Sequence[UnknownState],
    evaluated: Sequence[UnknownState],
    feedback: Sequence[Feedback],
    *,
    representation: ModelRepresentation | None = None,
) -> tuple[Feedback, ...]:
    """Return feedback only when every authoritative state was evaluated."""
    evaluated_states = tuple(evaluated)
    evaluated_feedback = tuple(feedback)
    if len(evaluated_states) != len(evaluated_feedback):
        return tuple()
    aligned: list[Feedback] = []
    used: set[int] = set()
    for state in tuple(authoritative):
        match = None
        for idx, candidate in enumerate(evaluated_states):
            if idx in used:
                continue
            equivalent = getattr(representation, "equivalent", None)
            if callable(equivalent):
                is_equivalent = bool(equivalent(state, candidate))
            else:
                is_equivalent = (
                    unknown_state_fingerprint(state)
                    == unknown_state_fingerprint(candidate)
                )
            if is_equivalent:
                match = idx
                break
        if match is None:
            return tuple()
        used.add(match)
        aligned.append(evaluated_feedback[match])
    return tuple(aligned)
