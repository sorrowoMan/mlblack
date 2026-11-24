"""ML semantic facade over the canonical NSGABlack control plane.

This module is deliberately an integration surface: MLBlack keeps ownership
of data/model/problem/backend semantics while NSGABlack owns the optimization
lifecycle, incumbent, snapshots, budgets, cancellation, and Adapter execution.
"""

from __future__ import annotations

import copy
import inspect
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from blackbase.runtime import FinalizationHooks
from blackbase.representation import RepresentationCapabilities
from blackbase.preferences import preserve_feedback_evidence
from blackbase.resources import DataRef, ResourceContext, coerce_resource_context
from blackbase.types import (
    CandidateBatch,
    Feedback,
    TrainerResult,
    UnknownState,
    decode_shared_value,
    encode_shared_value,
)
from blackbase.context.context_keys import KEY_INDIVIDUAL_ID
from blackbase.evaluation import StateReleaseRequest, StateReleaseResult
from mlblack.core.artifact_provider import (
    ArtifactFinalizationSession,
    ArtifactProvider,
    CaseRuntimeArtifactProvider,
)
from mlblack.core.backend_session import ComputeBackendSession, ComputeBackendSpec
from mlblack.core.capability import Capability
from mlblack.core.state import build_trainer_state

from nsgablack.core.base import BlackBoxProblem
from nsgablack.core.composable_solver import ComposableSolver
from nsgablack.core.evaluation_feedback import OptimizationFeedbackBatch, feedback_constraint_violation
from nsgablack.core.state.incumbent import IncumbentState
from nsgablack.core.state.run_progress import RunProgressState
from nsgablack.core.state.step_outcome import StepOutcome
from nsgablack.core.semantic_extension import SolverSemanticExtension


@dataclass(frozen=True)
class _MLModelHandle:
    """Process-local model handle; never an authority for evaluation feedback."""

    model: Any
    candidate_token: str | None = None


@dataclass
class _PendingArtifactFinalization:
    session: ArtifactFinalizationSession
    report: dict[str, Any]
    best_model_ref: DataRef | None
    best_state_ref: DataRef | None
    history_ref: DataRef | None
    report_ref: DataRef | None


class _MLProjectionCheckpointState:
    """Persist ML evidence projections without another Solver state owner."""

    schema = "mlblack.projection_checkpoint/v1"
    checkpoint_required = True

    def __init__(self, solver: "LearningSolver") -> None:
        self.solver = solver

    def get_state(self) -> Mapping[str, Any]:
        return {"schema": self.schema, **self.solver._export_ml_projection_state()}

    def set_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get("schema") != self.schema:
            raise ValueError("unsupported ML projection checkpoint schema")
        # The shared plugin restores Pool and incumbent before components.
        self.solver._restore_ml_projection_state(state)


class MLRepresentationBridge:
    """Project an ML ``ModelRepresentation`` onto NSGABlack's numeric pipeline.

    The bridge never defines a model layout. It only converts the shared
    ``UnknownState`` protocol at the repository boundary; the wrapped ML Codec
    remains the sole owner of init/repair/decode semantics.
    """

    name = "ml_representation_bridge"
    context_requires = ("candidate.unknown_state",)
    context_optional = ("backend.session", "resource.context", "resource_context")
    context_provides = ("candidate.model",)
    context_mutates = ("candidate.repaired_state",)
    context_cache = ()

    def __init__(
        self,
        representation: Any,
        *,
        initial_state: UnknownState | None = None,
        problem_validator: Any = None,
    ) -> None:
        if problem_validator is not None:
            if not callable(problem_validator):
                raise TypeError("problem_validator must be callable")
            inspect.signature(problem_validator).bind(object(), {})
        self.problem_validator = problem_validator
        self.representation = representation
        self._control: LearningSolver | None = None
        self._initial_state = initial_state
        self._default_metadata = (
            {} if initial_state is None else dict(initial_state.metadata)
        )

    representation_capabilities = RepresentationCapabilities(
        frozenset({"init", "repair", "repair_batch", "encode", "decode"})
    )

    @property
    def problem_repair(self):
        """Opt-in final-state validation over the existing ML repair owner."""
        return self if self.problem_validator is not None else None

    def validate(self, candidate: UnknownState, context: Mapping[str, Any]) -> bool:
        if self.problem_validator is None:
            raise TypeError("ML joint repair requires an explicit problem_validator")
        return self.problem_validator(candidate, self._ml_context(context))

    def encode(self, model: Any, context=None) -> UnknownState:
        encoded = self.representation.encode(model, self._ml_context(context))
        if not isinstance(encoded, UnknownState):
            raise TypeError("ML ModelRepresentation.encode(...) must return UnknownState")
        if self._control is not None:
            self._control._bind_ml_dimension(encoded.size)
        return encoded

    def bind(self, control: "LearningSolver") -> None:
        self._control = control

    def init(self, context: Mapping[str, Any]) -> UnknownState:
        state = self._initial_state
        self._initial_state = None
        if state is None:
            state = _initialize_representation(self.representation, self._ml_context(context))
        if not isinstance(state, UnknownState):
            raise TypeError(
                "ML ModelRepresentation.init(...) must return UnknownState; "
                f"got {type(state).__name__}"
            )
        self._default_metadata = dict(state.metadata)
        if self._control is not None:
            self._control._bind_ml_dimension(state.size)
        return state

    def repair(self, candidate: Any, context: Mapping[str, Any] | None = None) -> UnknownState:
        state = self._semantic_state(candidate, context)
        repaired = self.representation.repair(state, self._ml_context(context))
        if not isinstance(repaired, UnknownState):
            raise TypeError(
                "ML ModelRepresentation.repair(...) must return UnknownState; "
                f"got {type(repaired).__name__}"
            )
        return repaired

    def repair_batch(
        self,
        candidates: Sequence[Any],
        contexts: Sequence[Mapping[str, Any] | None] | None = None,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> list[UnknownState]:
        values = list(candidates)
        if contexts is None:
            context_values = [context] * len(values)
        else:
            context_values = list(contexts)
            if len(context_values) != len(values):
                raise ValueError("representation contexts must align with candidates")
        return [
            self.repair(candidate, context_values[index])
            for index, candidate in enumerate(values)
        ]

    def decode(self, candidate: Any, context: Mapping[str, Any] | None = None) -> Any:
        return self.representation.decode(
            self._semantic_state(candidate, context),
            self._ml_context(context),
        )

    def _semantic_state(
        self,
        candidate: Any,
        context: Mapping[str, Any] | None,
    ) -> UnknownState:
        if isinstance(candidate, UnknownState):
            return candidate
        control = self._control
        if control is None:
            return _unknown_state(candidate)
        raw_index = None if context is None else context.get(KEY_INDIVIDUAL_ID)
        state = control.semantic_candidate_state(
            candidate,
            candidate_index=None if raw_index is None else int(raw_index),
        )
        if not state.metadata and self._default_metadata:
            return UnknownState(
                values=state.as_array().copy(),
                metadata=dict(self._default_metadata),
            )
        return state

    def describe(self) -> Mapping[str, Any]:
        describe = getattr(self.representation, "describe", None)
        wrapped = dict(describe()) if callable(describe) else {
            "name": type(self.representation).__name__,
        }
        return {
            **wrapped,
            "control_projection": "mlblack.UnknownState<->nsgablack.ndarray",
        }

    def _ml_context(
        self,
        context: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        control = self._control
        if control is None:
            return dict(context or {})
        return control.enrich_ml_context(context)


class MLLearningProblemBridge(BlackBoxProblem):
    """Expose an ML ``LearningProblem`` as one rich-feedback black-box problem."""

    def __init__(
        self,
        semantic_problem: Any,
        representation_bridge: MLRepresentationBridge,
        *,
        dimension: int,
    ) -> None:
        super().__init__(
            name=str(getattr(semantic_problem, "name", "ml_learning_problem")),
            dimension=int(dimension),
            bounds={f"x{index}": (-np.inf, np.inf) for index in range(int(dimension))},
            objectives=tuple(
                f"objective_{index}"
                for index in range(_problem_objective_count(semantic_problem))
            ),
        )
        self.semantic_problem = semantic_problem
        self.representation_bridge = representation_bridge
        self._control: LearningSolver | None = None

    def bind(self, control: "LearningSolver") -> None:
        self._control = control

    def evaluate(self, candidate: Any, context: Mapping[str, Any] | None = None) -> Feedback:
        control = self._require_control()
        raw_index = None if context is None else context.get(KEY_INDIVIDUAL_ID)
        candidate_index = None if raw_index is None else int(raw_index)
        state = control.semantic_candidate_state(
            candidate,
            candidate_index=candidate_index,
        )
        provenance = control.candidate_provenance_for(
            candidate,
            candidate_index=candidate_index,
        )
        ml_context = control.enrich_ml_context(
            context,
            **{
                "evaluation.mode": "train",
                "candidate.unknown_state": state,
            },
        )
        decoded_model = self.representation_bridge.representation.decode(state, ml_context)
        model = self.semantic_problem.prepare_model_for_evaluation(
            decoded_model,
            state,
            ml_context,
        )
        ml_context["candidate.model"] = model
        feedback = self.semantic_problem.evaluate(model, state, ml_context)
        if not isinstance(feedback, Feedback):
            raise TypeError(
                "ML LearningProblem.evaluate(...) must return blackbase.Feedback; "
                f"got {type(feedback).__name__}"
            )
        feedback = control.adjust_feedback_with_biases(
            (state,),
            (feedback,),
            ml_context,
        )[0]
        control._record_ml_model_handle(
            model,
            candidate_token=(None if provenance is None else provenance.candidate_token),
        )
        return feedback

    def describe(self) -> Mapping[str, Any]:
        describe = getattr(self.semantic_problem, "describe", None)
        semantic = dict(describe()) if callable(describe) else {"name": self.name}
        return {
            **semantic,
            "optimization_control_plane": "nsgablack.ComposableSolver",
        }

    def __getattr__(self, name: str) -> Any:
        # Preserve public semantic/provider surfaces such as problem_id,
        # gateway, build_model_artifact, and data without copying them.
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.semantic_problem, name)

    def _require_control(self) -> "LearningSolver":
        if self._control is None:
            raise RuntimeError("MLLearningProblemBridge is not bound to a control plane")
        return self._control


class _MLSemantics(SolverSemanticExtension):
    name = "ml_projection"

    def prepare_run(self, solver):
        solver._prepare_ml_run()

    def setup(self, solver):
        solver._setup_ml_semantics()

    def project_context(self, solver, context):
        projected = solver.enrich_ml_context(context)
        # Router/task contexts are checkpointable data. The process-local
        # backend is injected only at ML representation/problem boundaries.
        projected.pop("backend.session", None)
        return projected

    def checkpoint_components(self, solver):
        return solver._ml_checkpoint_components()

    def should_execute_step(self, solver, step_index):
        return solver._ml_completion_allows_step()

    def cleanup_components(self, solver):
        return {
            "provider_trajectory": solver._release_provider_trajectory,
            "compute_backend": solver.compute_backend_session.close,
        }

    def before_step(self, solver):
        solver._step_ml_model_handles = []

    def after_step_committed(self, solver, outcome):
        solver._synchronize_ml_projection(record_history=True)


class LearningSolver(ComposableSolver):
    """ML-friendly facade whose only optimization control is ComposableSolver.

    ``fit`` and ``TrainerResult`` are semantic projections. They do not run a
    second lifecycle, select another best state, or maintain another snapshot
    authority.
    """

    control_plane = "nsgablack.ComposableSolver"

    def __init__(
        self,
        *,
        problem: Any,
        representation: Any,
        adapter: Any,
        run_name: str = "learning_solver_run",
        resource_context: Mapping[str, Any] | ResourceContext | None = None,
        compute_backend: str | Mapping[str, Any] | ComputeBackendSpec | ComputeBackendSession | None = None,
        artifact_provider: ArtifactProvider | None = None,
        result_inline_limit_bytes: int = 64 * 1024,
        problem_validator: Any = None,
        **solver_kwargs: Any,
    ) -> None:
        self.semantic_problem = problem
        self.model_representation = representation
        self.run_name = str(run_name)
        self._resource_context_explicit = resource_context is not None
        resource = _resource_context_with_namespace(resource_context, self.run_name)
        self._compute_backend_request = (
            compute_backend.spec
            if isinstance(compute_backend, ComputeBackendSession)
            else compute_backend
        )
        self.compute_backend_session = (
            compute_backend
            if isinstance(compute_backend, ComputeBackendSession)
            and resource_context is None
            else _build_compute_backend_session(
                self._compute_backend_request,
                resource,
                explicit=resource_context is not None,
            )
        )
        self.biases: list[Any] = []
        representation_bridge = MLRepresentationBridge(
            representation,
            problem_validator=problem_validator,
        )
        problem_bridge = MLLearningProblemBridge(
            problem,
            representation_bridge,
            # Zero means a deferred layout, never an empty sampled candidate.
            # Neural layouts resolve in setup; custom representations may bind
            # their dimension when the first runtime candidate is initialized.
            dimension=int(getattr(representation, "dimension", 0) or 0),
        )
        super().__init__(
            problem_bridge,
            adapter=adapter,
            representation_pipeline=representation_bridge,
            resource_context=resource,
            **solver_kwargs,
        )
        self._resource_context_explicit = resource_context is not None
        self.register_semantic_extension(_MLSemantics())
        self.finalization_hooks = FinalizationHooks(
            prepare_result=self.finalize_run_result,
            prepare=self.prepare_finalization_after_teardown,
            freeze=self.freeze_finalization_after_prepare,
            abort=self.abort_finalization_after_teardown,
        )
        representation_bridge.bind(self)
        problem_bridge.bind(self)

        self.artifact_provider: ArtifactProvider
        self.set_artifact_provider(
            artifact_provider or CaseRuntimeArtifactProvider()
        )
        self.best_state: UnknownState | None = None
        self.best_model: Any | None = None
        self.best_feedback: Feedback | None = None
        self.best_score: float | None = None
        self.best_model_ref: DataRef | None = None
        self.result_artifact_refs: dict[str, DataRef] = {}
        self.last_evaluated_population: tuple[UnknownState, ...] = ()
        self.last_evaluated_feedback: tuple[Feedback, ...] = ()
        self.feedback: tuple[Feedback, ...] = ()
        self.history: list[dict[str, Any]] = []
        self._step_ml_model_handles: list[_MLModelHandle] = []
        self._best_feedback_binding: dict[str, Any] | None = None
        self._completion_policy: Any | None = None
        self._last_state_release: StateReleaseResult | None = None
        self.result_inline_limit_bytes = max(1024, int(result_inline_limit_bytes))
        self._pending_finalization_report: dict[str, Any] | None = None
        self._finalized_report: dict[str, Any] | None = None
        self._finalized_trainer_result: TrainerResult | None = None
        self._best_state_ref: DataRef | None = None
        self._history_ref: DataRef | None = None
        self._report_ref: DataRef | None = None
        self.run_session.finalizing = False
        self._pending_artifact_finalization: _PendingArtifactFinalization | None = None
        self._ml_projection_checkpoint = _MLProjectionCheckpointState(self)

    def _bind_ml_dimension(self, dimension: int) -> None:
        """Bind a declared/runtime layout without sampling a throwaway model."""
        size = int(dimension)
        if size <= 0:
            raise ValueError("ML representation must define at least one trainable value")
        if self.dimension not in (0, size):
            raise ValueError("ML candidate dimension differs from the configured model layout")
        self.dimension = self.problem.dimension = size
        self.problem.variables = [f"x{index}" for index in range(size)]
        self.problem.bounds = {name: (-np.inf, np.inf) for name in self.problem.variables}
        self.var_bounds = self.problem.bounds

    @property
    def step_index(self) -> int:
        return int(self.generation)

    @property
    def representation(self) -> Any:
        return self.model_representation

    @property
    def context(self) -> Mapping[str, Any]:
        snapshot = getattr(self.context_store, "snapshot", None)
        return dict(snapshot()) if callable(snapshot) else {}

    def enrich_ml_context(
        self,
        context: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        ctx = dict(context or {})
        resource = self.resource_context.as_dict()
        ctx.update(
            {
                "run_name": self.run_name,
                "step": int(self.generation),
                "trajectory_id": str(self.run_session.run_id or self.run_name),
                "resource_context": resource,
                "resource.context": resource,
                **self.resource_context.context_items(prefix="resource"),
                **self.compute_backend_session.context_items(),
            }
        )
        ctx.update(extra)
        for bias in self.biases:
            project = getattr(bias, "project_context", None)
            if callable(project):
                projected = project(self, ctx)
                if not isinstance(projected, Mapping):
                    raise TypeError(
                        f"ML bias {type(bias).__name__}.project_context(...) "
                        "must return a Mapping"
                    )
                ctx = dict(projected)
        return ctx

    def init_candidate(
        self,
        context: Mapping[str, Any] | None = None,
        *,
        initializer: Any = None,
    ) -> UnknownState:
        candidate = super().init_candidate(
            self.enrich_ml_context(context),
            initializer=initializer,
        )
        state = self.semantic_candidate_state(candidate)
        provenance = self.candidate_provenance_for(candidate)
        if provenance is not None:
            self.bind_candidate_provenance((state,), (provenance,), activate=False)
        return state

    def set_initial_state(self, state: UnknownState | Sequence[float] | np.ndarray) -> "LearningSolver":
        """Replace the next run's representation-owned initial candidate.

        Standard Case input binding happens after construction.  This method
        updates the already-created representation bridge without creating a
        second warm-start or lifecycle authority.
        """

        candidate = _unknown_state(state)
        self._bind_ml_dimension(candidate.size)
        bridge = self.representation_pipeline
        if not isinstance(bridge, MLRepresentationBridge):
            raise TypeError("LearningSolver representation bridge is unavailable")
        bridge._initial_state = candidate
        return self

    def evaluate_individual(
        self,
        candidate: Any,
        individual_id: int | None = None,
    ) -> Feedback:
        # Public ML projection. The NSG population loop calls its internal
        # evaluation helper directly and therefore still receives the native
        # (objectives, violation) control-plane tuple exactly once.
        state = _unknown_state(candidate)
        self._bind_ml_dimension(state.size)
        row = _state_array(state, owner="evaluate_individual")
        provenance = self.prepare_candidate_provenance((state,))
        self.bind_candidate_provenance((row,), provenance, activate=False)
        super().evaluate_individual(row, individual_id=individual_id)
        feedback = getattr(self, "_last_individual_feedback", None)
        if not isinstance(feedback, Feedback):
            raise RuntimeError("NSG evaluation did not retain rich Feedback")
        return feedback

    def _setup_ml_semantics(self) -> None:
        if self.adapter is None:
            raise ValueError("LearningSolver requires an Adapter before fit()")
        requirements = tuple(
            dict.fromkeys(
                (
                    *tuple(getattr(self.model_representation, "backend_requires", ()) or ()),
                    *tuple(getattr(self.semantic_problem, "backend_requires", ()) or ()),
                    *tuple(getattr(self.adapter, "backend_requires", ()) or ()),
                )
            )
        )
        # Provider-backed problems own their execution session and intentionally
        # declare no local requirements. Direct semantic problems use this
        # already-authorized session.
        if requirements:
            self.require_compute_backend(requirements, consumer="LearningSolver.setup")
        setup = getattr(self.model_representation, "setup", None)
        if callable(setup):
            setup(self, self.enrich_ml_context(self.build_context()))
        dimension = int(getattr(self.model_representation, "dimension", 0) or 0)
        if dimension:
            self._bind_ml_dimension(dimension)

    def _prepare_ml_run(self) -> None:
        reset_policy = getattr(self._completion_policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        self.best_state = None
        self.best_model = None
        self.best_feedback = None
        self.best_score = None
        self.best_model_ref = None
        self.result_artifact_refs = {}
        self.last_evaluated_population = ()
        self.last_evaluated_feedback = ()
        self.feedback = ()
        self._step_ml_model_handles = []
        self._best_feedback_binding = None
        self._last_state_release = None
        self._pending_finalization_report = None
        self._prepared_model_artifact = None
        self._finalized_report = None
        self._finalized_trainer_result = None
        self._best_state_ref = None
        self._history_ref = None
        self._report_ref = None
        self.run_session.finalizing = False
        self._pending_artifact_finalization = None

    def require_compute_backend(
        self,
        requirements: Sequence[str],
        *,
        consumer: str = "",
    ) -> Any:
        """Resolve one already-authorized ML execution backend.

        This is the ML semantic backend surface. It does not allocate a device
        or create another scheduler: the session is derived
        from the Project L0 ``ResourceContext`` owned by the NSG control plane.
        """

        return self.compute_backend_session.ensure(
            tuple(str(item) for item in requirements),
            consumer=consumer or "LearningSolver.require_compute_backend",
        )

    def _release_provider_trajectory(self) -> StateReleaseResult | None:
        provider = getattr(self, "evaluation_provider", None)
        gateway = getattr(self.semantic_problem, "gateway", None)
        release = getattr(gateway, "release", None)
        if provider is None or not callable(getattr(provider, "release", None)):
            return None
        if not callable(release):
            raise RuntimeError("stateful evaluation Provider has no release gateway")
        trajectory_id = str(self.run_session.run_id or self.run_name)
        result = release(
            StateReleaseRequest(
                provider_id=str(provider.spec.provider_id),
                scope_id=str(self.resource_context.namespace or ""),
                trajectory_id=trajectory_id,
                metadata={"owner": "LearningSolver.teardown"},
            ),
            self.resource_context,
        )
        self._last_state_release = result
        return result

    def _ml_checkpoint_components(self) -> Mapping[str, Any]:
        components: dict[str, Any] = {
            "ml_projection": self._ml_projection_checkpoint,
        }
        provider = getattr(self, "evaluation_provider", None)
        if provider is not None:
            components["evaluation_provider"] = provider
        components["model_representation"] = self.model_representation
        semantic_state = getattr(self.semantic_problem, "get_state", None)
        semantic_restore = getattr(self.semantic_problem, "set_state", None)
        if callable(semantic_state) and callable(semantic_restore):
            components["learning_problem"] = self.semantic_problem
        policy_state = getattr(self._completion_policy, "get_state", None)
        policy_restore = getattr(self._completion_policy, "set_state", None)
        if callable(policy_state) and callable(policy_restore):
            components["completion_policy"] = self._completion_policy
        return components

    def _ml_completion_allows_step(self) -> bool:
        if self._completion_policy is not None and self._completion_policy.is_complete(
            step=int(self.run_progress_steps),
            elapsed=float(self.run_progress_elapsed_seconds),
            ctx={"trainer": self, "run_name": self.run_name},
        ):
            self.request_stop("completion_policy")
            return False
        return True

    def fit(self, max_steps: int | None = None) -> TrainerResult:
        """Run the unified Solver lifecycle for the configured step budget.

        An explicit argument is a per-call override.  A no-argument Case run
        follows the Solver-owned ``max_steps`` value instead of introducing a
        second ML-only default budget.
        """

        requested_steps = max(
            0,
            int(self.max_steps if max_steps is None else max_steps),
        )
        run_limit = self.run_session.step_limit(requested_steps)
        ComposableSolver.run(self, max_steps=run_limit)
        if self._finalized_trainer_result is None:
            raise RuntimeError("ML run completed without a finalized TrainerResult")
        return self._finalized_trainer_result

    def finalize_run_result(self, raw_result: Mapping[str, Any]) -> dict[str, Any]:
        """Prepare ML semantics without publishing successful Artifacts."""

        self._synchronize_ml_projection(record_history=False)
        report = self.build_report(raw_result)
        # Capture the derived payload while local backend services are alive.
        # Publication/serialization still occurs only after successful teardown.
        self._prepared_model_artifact = None
        if self.case_runtime is not None and self.best_model is not None:
            from mlblack.core.artifact_provider import materialize_model_artifact
            self._prepared_model_artifact = (self.best_model, materialize_model_artifact(self, self.best_model))
        self._pending_finalization_report = dict(report)
        # on_solver_finish is a pre-teardown finishing notification.  It may
        # inspect semantic output, but no durable success Artifact exists yet.
        return {
            "status": str(raw_result.get("status", "ok")),
            "termination_reason": str(raw_result.get("termination_reason", "")),
            "report": report,
            "trainer_result_ready": False,
            "artifact_refs": {},
        }

    def prepare_finalization_after_teardown(
        self,
        raw_result: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Stage all success Artifacts after teardown without exposing refs."""

        if not self.run_session.finalizing:
            raise RuntimeError(
                "ML result finalization is owned by the Solver lifecycle"
            )
        if self._pending_finalization_report is None:
            raise RuntimeError("ML finalization plan is unavailable after teardown")
        if self._pending_artifact_finalization is not None:
            raise RuntimeError("ML Artifact finalization is already active")
        report = dict(self._pending_finalization_report)
        report["provider_state_release"] = (
            None
            if self._last_state_release is None
            else self._last_state_release.as_dict()
        )
        begin = getattr(self.artifact_provider, "begin_finalization", None)
        if not callable(begin):
            raise TypeError(
                "artifact provider must implement begin_finalization(control, ...)"
            )
        session = begin(self, label="mlblack_result_finalization")
        try:
            best_model_ref = session.publish_best_model()
            best_state_ref = self._stage_large_result_field(
                session,
                "trainer_best_state",
                self.best_state,
                kind="trainer_state",
                path="trainer_result.best_state",
            )
            history_ref = self._stage_large_result_field(
                session,
                "trainer_history",
                tuple(self.history),
                kind="trainer_history",
                path="trainer_result.history",
            )
            report_ref = self._stage_large_result_field(
                session,
                "trainer_report",
                dict(report),
                kind="trainer_report",
                path="trainer_result.report",
            )
        except BaseException:
            session.abort("staging_failed")
            raise
        self._pending_artifact_finalization = _PendingArtifactFinalization(
            session=session,
            report=report,
            best_model_ref=best_model_ref,
            best_state_ref=best_state_ref,
            history_ref=history_ref,
            report_ref=report_ref,
        )
        return {
            "status": str(raw_result.get("status", "ok")),
            "termination_reason": str(raw_result.get("termination_reason", "")),
            "report": dict(report),
            "trainer_result_ready": False,
            "artifact_refs": {},
        }

    @staticmethod
    def _prepared_ref(
        provisional: DataRef | None,
        prepared: Mapping[str, DataRef],
    ) -> DataRef | None:
        if provisional is None:
            return None
        for ref in prepared.values():
            if ref.uri == provisional.uri:
                return ref
        raise RuntimeError(
            "Artifact finalization prepare omitted one of its staged refs"
        )

    def freeze_finalization_after_prepare(
        self,
        raw_result: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Freeze staged refs after every strict prepare hook passes."""

        if not self.run_session.finalizing:
            raise RuntimeError(
                "ML result finalization is owned by the Solver lifecycle"
            )
        pending = self._pending_artifact_finalization
        if pending is None:
            raise RuntimeError("ML Artifact finalization was not prepared")
        prepared = dict(pending.session.prepare())
        best_model_ref = self._prepared_ref(pending.best_model_ref, prepared)
        best_state_ref = self._prepared_ref(pending.best_state_ref, prepared)
        history_ref = self._prepared_ref(pending.history_ref, prepared)
        report_ref = self._prepared_ref(pending.report_ref, prepared)
        result_refs = dict(prepared)
        if best_model_ref is not None:
            result_refs["best_model"] = best_model_ref
        self.best_model_ref = best_model_ref
        self._best_state_ref = best_state_ref
        self._history_ref = history_ref
        self._report_ref = report_ref
        self.result_artifact_refs = result_refs
        self._finalized_report = dict(pending.report)
        self._finalized_trainer_result = self._build_trainer_result(
            self._finalized_report
        )
        self._pending_finalization_report = None
        self._pending_artifact_finalization = None
        return {
            "status": str(raw_result.get("status", "ok")),
            "termination_reason": str(raw_result.get("termination_reason", "")),
            "report": dict(self._finalized_report),
            "trainer_result_ready": True,
            "artifact_refs": dict(self.result_artifact_refs),
        }

    def abort_finalization_after_teardown(self, error: BaseException) -> None:
        pending = self._pending_artifact_finalization
        if pending is not None:
            pending.session.abort(
                f"{type(error).__name__}: {error}"
            )
        self._pending_artifact_finalization = None
        self._finalized_report = None
        self._finalized_trainer_result = None
        self.best_model_ref = None
        self._best_state_ref = None
        self._history_ref = None
        self._report_ref = None
        self.result_artifact_refs = {}

    def finalize_after_teardown(self, raw_result: Mapping[str, Any]) -> Mapping[str, Any]:
        del raw_result
        raise RuntimeError(
            "direct ML Artifact publication is disabled; use the unified Solver lifecycle"
        )

    def _stage_large_result_field(
        self,
        session: ArtifactFinalizationSession,
        name: str,
        value: Any,
        *,
        kind: str,
        path: str,
    ) -> DataRef | None:
        if value is None:
            return None
        encoded = encode_shared_value(value, path=path)
        size = len(
            json.dumps(
                encoded,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if size <= self.result_inline_limit_bytes:
            return None
        ref = session.publish_payload(name, value, kind=kind)
        if ref is None:
            # Standalone fits have no Case artifact authority.  They return the
            # in-process TrainerResult object and therefore keep the field
            # inline; a Case execution never fabricates an unresolved ref.
            return None
        return ref

    def _build_trainer_result(self, report: Mapping[str, Any]) -> TrainerResult:
        return TrainerResult(
            best_state=self.best_state,
            best_model=self.best_model,
            best_objectives=(
                None
                if self.best_feedback is None
                else np.asarray(self.best_feedback.objectives, dtype=float)
            ),
            best_feedback=self.best_feedback,
            history=tuple(self.history),
            report=report,
            best_model_ref=self.best_model_ref,
            best_state_ref=self._best_state_ref,
            history_ref=self._history_ref,
            report_ref=self._report_ref,
            artifact_refs=self.result_artifact_refs,
        )

    def run(self, max_steps: int | None = None) -> TrainerResult:
        return self.fit(max_steps=max_steps)

    def export_case_result(self, raw_output: Any) -> TrainerResult:
        if isinstance(raw_output, TrainerResult):
            return raw_output
        if (
            isinstance(raw_output, Mapping)
            and str(raw_output.get("protocol_type", "")) == "blackbase.trainer_result"
        ):
            return TrainerResult.from_dict(raw_output)
        return self._build_trainer_result(
            self.build_report(raw_output if isinstance(raw_output, Mapping) else None)
        )

    def evaluate(
        self,
        candidate: UnknownState | np.ndarray,
        *,
        resource_context: Mapping[str, Any] | ResourceContext | None = None,
        max_steps: int | None = None,
        **_: Any,
    ) -> TrainerResult:
        if resource_context is not None:
            self.set_resource_context(resource_context)
        self.set_warm_start(_unknown_state(candidate))
        return self.fit(max_steps=int(max_steps or 1))

    def add_capability(self, capability: Capability) -> "LearningSolver":
        if not isinstance(capability, Capability):
            raise TypeError("capability must be an mlblack Capability")
        self.add_plugin(capability)
        return self

    def add_bias(self, bias: Any) -> "LearningSolver":
        """Attach one ML preference projection without creating another loop."""

        if not callable(getattr(bias, "adjust_feedback", None)) and not callable(
            getattr(bias, "project_context", None)
        ):
            raise TypeError(
                "ML bias must provide adjust_feedback(...) or project_context(...)"
            )
        self.biases.append(bias)
        return self

    def adjust_feedback_with_biases(
        self,
        states: Sequence[UnknownState],
        feedback: Sequence[Feedback],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[Feedback, ...]:
        adjusted = tuple(feedback)
        state_values = tuple(states)
        ctx = dict(context or {})
        for bias in self.biases:
            adjust = getattr(bias, "adjust_feedback", None)
            if callable(adjust):
                measured = adjusted
                adjusted = tuple(adjust(self, state_values, adjusted, ctx))
                if len(adjusted) != len(state_values):
                    raise ValueError(
                        f"ML bias {type(bias).__name__} changed feedback batch size"
                    )
                if not all(isinstance(item, Feedback) for item in adjusted):
                    raise TypeError(
                        f"ML bias {type(bias).__name__} must return Feedback values"
                    )
                adjusted = tuple(preserve_feedback_evidence(
                    raw, projected, raw_violation=feedback_constraint_violation(raw),
                    projected_violation=feedback_constraint_violation(projected),
                    reason=f"bias:{type(bias).__name__}",
                ) for raw, projected in zip(measured, adjusted))
        return adjusted

    def set_completion_policy(self, policy: Any) -> "LearningSolver":
        if policy is not None and not callable(getattr(policy, "is_complete", None)):
            raise TypeError("completion policy must provide is_complete(...)")
        if policy is not None:
            has_get_state = callable(getattr(policy, "get_state", None))
            has_set_state = callable(getattr(policy, "set_state", None))
            if has_get_state != has_set_state:
                raise TypeError(
                    "stateful completion policy must provide both get_state() and set_state()"
                )
        self._completion_policy = policy
        return self

    def set_artifact_provider(self, provider: ArtifactProvider) -> "LearningSolver":
        if not callable(getattr(provider, "begin_finalization", None)):
            raise TypeError(
                "artifact provider must implement begin_finalization(control, ...)"
            )
        if not callable(getattr(provider, "publish_best_model", None)):
            raise TypeError("artifact provider must implement publish_best_model(control)")
        if not callable(getattr(provider, "publish_payload", None)):
            raise TypeError("artifact provider must implement publish_payload(control, ...)")
        self.artifact_provider = provider
        return self

    def set_resource_context(
        self,
        context: Mapping[str, Any] | ResourceContext | None,
    ) -> "LearningSolver":
        previous = getattr(self, "compute_backend_session", None)
        explicit = context is not None
        resource = _resource_context_with_namespace(context, self.run_name)
        super().set_resource_context(resource)
        self._resource_context_explicit = explicit
        self.compute_backend_session = _build_compute_backend_session(
            self._compute_backend_request,
            self.resource_context,
            explicit=explicit,
        )
        if previous is not None and previous is not self.compute_backend_session:
            previous.close()
        return self

    def set_best_model_ref(
        self,
        ref: DataRef | Mapping[str, Any] | None,
    ) -> "LearningSolver":
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
    ) -> "LearningSolver":
        item = ref if isinstance(ref, DataRef) else DataRef.from_dict(ref)
        self.result_artifact_refs[str(name)] = item
        return self

    def publish_best_model_artifact(self) -> DataRef | None:
        raise RuntimeError(
            "direct best-model publication is disabled; successful Artifacts "
            "are committed by the unified Solver finalization transaction"
        )

    def build_report(self, raw_result: Mapping[str, Any] | None = None) -> dict[str, Any]:
        adapter = self.adapter
        report = {
            "run_name": self.run_name,
            "status": "finished",
            "steps": int(len(self.history)),
            "best_score": self.best_score,
            "best_state": (
                None if self.best_state is None else self.best_state.as_array().tolist()
            ),
            "best_metrics": (
                {} if self.best_feedback is None else dict(self.best_feedback.metrics)
            ),
            "representation": dict(self.model_representation.describe()),
            "problem": dict(self.problem.describe()),
            "adapter": None
            if adapter is None
            else {
                "name": str(getattr(adapter, "name", type(adapter).__name__)),
                "state": dict(adapter.get_state()),
            },
            "biases": [
                dict(bias.describe())
                if callable(getattr(bias, "describe", None))
                else {"name": type(bias).__name__}
                for bias in self.biases
            ],
            "resources": self.resource_context.as_dict(),
            "compute_backend": self.compute_backend_session.as_dict(),
            "control_plane": self.control_plane,
            "optimization_control_plane": self.control_plane,
            "optimization_runtime": dict(raw_result or self.last_result or {}),
            "provider_state_release": (
                None
                if self._last_state_release is None
                else self._last_state_release.as_dict()
            ),
        }
        report["state_signature"] = build_trainer_state(self).signature
        return report

    def get_state(self) -> Mapping[str, Any]:
        incumbent = self.get_incumbent()
        provider = getattr(self, "evaluation_provider", None)
        provider_state = getattr(provider, "get_state", None)
        return {
            "state_version": 7,
            "control_plane": self.control_plane,
            "dimension": int(self.dimension),
            "solution_pool": self.solution_pool.get_state(),
            "candidate_sequence": int(self._candidate_sequence),
            "proposal_sequence": int(self._proposal_sequence),
            "rng_state": self.get_rng_state(),
            "run_name": self.run_name,
            "generation": int(self.generation),
            "evaluation_count": int(self.evaluation_count),
            "active_run_id": self.run_session.run_id,
            "incumbent": None if incumbent is None else incumbent.as_dict(),
            "adapter": None
            if self.adapter is None
            else {
                "name": str(getattr(self.adapter, "name", type(self.adapter).__name__)),
                "state": dict(self.adapter.get_state()),
            },
            "provider": None if not callable(provider_state) else dict(provider_state()),
            "run_progress": self.export_run_progress_state(),
            "completion_policy": (
                None
                if not callable(getattr(self._completion_policy, "get_state", None))
                else {
                    "class": (
                        f"{type(self._completion_policy).__module__}."
                        f"{type(self._completion_policy).__qualname__}"
                    ),
                    "state": dict(self._completion_policy.get_state()),
                }
            ),
            **self._export_ml_projection_state(),
            "history": tuple(dict(row) for row in self.history),
            "resource_context": self.resource_context.as_dict(),
            "compute_backend": self.compute_backend_session.as_dict(),
        }

    def _export_ml_projection_state(self) -> dict[str, Any]:
        """One field definition for facade and formal component checkpoints."""
        return copy.deepcopy({
            "dimension": int(self.dimension),
            "best_feedback": (
                None if self.best_feedback is None else self.best_feedback.as_dict()
            ),
            "best_feedback_binding": self._best_feedback_binding,
            "best_model_ref": (
                None if self.best_model_ref is None else self.best_model_ref.as_dict()
            ),
            "result_artifact_refs": {
                name: ref.as_dict() for name, ref in self.result_artifact_refs.items()
            },
        })

    def _restore_ml_projection_state(self, state: Mapping[str, Any]) -> None:
        """Restore saved evidence only; never initialize, decode or evaluate."""
        raw_feedback = state.get("best_feedback")
        if raw_feedback is not None and not isinstance(raw_feedback, Mapping):
            raise TypeError("ML projection best_feedback must be a Mapping")
        feedback = None if raw_feedback is None else Feedback.from_dict(raw_feedback)
        raw_binding = state.get("best_feedback_binding")
        if raw_binding is not None and not isinstance(raw_binding, Mapping):
            raise TypeError("ML projection feedback binding must be a Mapping")
        binding = None if raw_binding is None else copy.deepcopy(dict(raw_binding))
        incumbent = self.get_incumbent()
        dimension = int(state.get("dimension", 0) or 0)
        if dimension < 0:
            raise ValueError("ML projection dimension must be nonnegative")
        if not dimension and incumbent is not None:
            dimension = np.asarray(incumbent.candidate).size
        if dimension and self.dimension not in (0, dimension):
            raise ValueError("ML candidate dimension differs from the configured model layout")
        if incumbent is not None or feedback is not None or binding is not None:
            bound_evaluation = None if binding is None else binding.get("evaluation_id")
            if (
                incumbent is None or feedback is None or binding is None
                or incumbent.candidate_token is None
                or binding.get("candidate_token") != incumbent.candidate_token
                or (
                    bound_evaluation is not None and incumbent.evaluation_id is not None
                    and str(bound_evaluation) != str(incumbent.evaluation_id)
                )
            ):
                raise ValueError(
                    "restored best_feedback is not bound to the authoritative incumbent"
                )
        raw_ref = state.get("best_model_ref")
        if raw_ref is not None and not isinstance(raw_ref, Mapping):
            raise TypeError("ML projection best_model_ref must be a Mapping")
        best_ref = None if raw_ref is None else DataRef.from_dict(raw_ref)
        raw_refs = state.get("result_artifact_refs", {})
        if not isinstance(raw_refs, Mapping):
            raise TypeError("ML projection result_artifact_refs must be a Mapping")
        refs = {}
        for name, raw in raw_refs.items():
            if not isinstance(raw, Mapping):
                raise TypeError("ML projection artifact reference must be a Mapping")
            refs[str(name)] = DataRef.from_dict(raw)
        if best_ref is not None:
            refs.setdefault("best_model", best_ref)
        best_state = None
        if incumbent is not None:
            metadata = incumbent.metadata.get("candidate.semantic_metadata", {})
            decoded = decode_shared_value(dict(metadata)) if isinstance(metadata, Mapping) else {}
            best_state = UnknownState(
                values=np.asarray(incumbent.candidate, dtype=float).copy(),
                metadata=dict(decoded) if isinstance(decoded, Mapping) else {},
            )
        # Commit detached values after all validation; live handles are never
        # checkpoint state, and model reconstruction is deferred to projection.
        if dimension:
            self._bind_ml_dimension(dimension)
        self.best_feedback = feedback
        self._best_feedback_binding = binding
        self.best_state = best_state
        self.best_score = None if incumbent is None else float(incumbent.score)
        self.best_model = None
        self._step_ml_model_handles = []
        self.best_model_ref = best_ref
        self.result_artifact_refs = refs

    def set_state(self, state: Mapping[str, Any]) -> "LearningSolver":
        payload = self._migrate_learning_state(state)
        self.run_session.submit_restore(
            lambda: self._apply_state_now(payload),
            generation=max(0, int(payload.get("generation", 0) or 0)),
            source="LearningSolver.set_state",
        )
        return self

    @staticmethod
    def _migrate_learning_state(state: Mapping[str, Any]) -> dict[str, Any]:
        payload = copy.deepcopy(dict(state or {}))
        version = int(payload.get("state_version", 3) or 3)
        if version == 7:
            if not isinstance(payload.get("solution_pool"), Mapping):
                raise ValueError("LearningSolver v7 state requires its solution_pool closure")
            return payload
        if version not in {3, 4, 5, 6}:
            raise ValueError(
                f"unsupported LearningSolver state version: {version}"
            )
        if not isinstance(payload.get("solution_pool"), Mapping):
            raise ValueError(
                "legacy LearningSolver state has no solution_pool value closure; "
                "restore a complete shared checkpoint instead"
            )
        if version == 3:
            payload.setdefault(
                "run_progress",
                {
                    "schema": "nsgablack.run_progress/v1",
                    "steps_completed": max(
                        0,
                        int(payload.get("generation", 0) or 0),
                    ),
                    "elapsed_seconds": 0.0,
                    "deadline_remaining_seconds": None,
                    "run_id": payload.get("active_run_id"),
                },
            )
            payload.setdefault("completion_policy", None)
        payload["run_progress"] = RunProgressState.from_dict(
            payload.get("run_progress", {})
        ).as_dict()
        payload.setdefault("best_feedback_binding", None)
        payload["state_version"] = 7
        payload["migrated_from_state_version"] = version
        return payload

    def _apply_state_now(self, state: Mapping[str, Any]) -> "LearningSolver":
        if str(state.get("control_plane", self.control_plane)) != self.control_plane:
            raise ValueError("trainer state belongs to another control plane")
        dimension = int(state.get("dimension", 0) or 0)
        if not dimension and isinstance(state.get("incumbent"), Mapping):
            dimension = np.asarray(state["incumbent"]["candidate"]).size
        if dimension:
            self._bind_ml_dimension(dimension)
        self.run_name = str(state.get("run_name", self.run_name))
        resource = state.get("resource_context")
        if isinstance(resource, Mapping):
            self.set_resource_context(resource)
        pool_payload = state.get("solution_pool")
        if isinstance(pool_payload, Mapping):
            # Restore immutable values before any Adapter resolves exact refs.
            self.solution_pool.set_state(pool_payload)
        adapter_payload = state.get("adapter")
        if self.adapter is not None and isinstance(adapter_payload, Mapping):
            adapter_state = adapter_payload.get("state", adapter_payload)
            if isinstance(adapter_state, Mapping):
                self.adapter.set_state(adapter_state)
        if self.adapter is not None:
            roots = self.adapter.get_solution_pool_roots(self)
            if roots is not None:
                self.solution_pool.read(tuple(roots))
        self._candidate_sequence = max(0, int(state.get("candidate_sequence", 0)))
        self._proposal_sequence = max(0, int(state.get("proposal_sequence", 0)))
        if isinstance(state.get("rng_state"), Mapping):
            self.set_rng_state(dict(state["rng_state"]))
        provider = getattr(self, "evaluation_provider", None)
        provider_payload = state.get("provider")
        provider_setter = getattr(provider, "set_state", None)
        if callable(provider_setter) and isinstance(provider_payload, Mapping):
            provider_setter(provider_payload)
        incumbent_payload = state.get("incumbent")
        if isinstance(incumbent_payload, Mapping):
            self.set_incumbent(IncumbentState.from_dict(incumbent_payload))
        generation = max(0, int(state.get("generation", 0) or 0))
        self.set_generation(generation)
        self.evaluation_count = max(
            0,
            int(state.get("evaluation_count", self.evaluation_count) or 0),
        )
        active_run_id = state.get("active_run_id")
        if active_run_id:
            self.run_session.run_id = str(active_run_id)
        self.restore_run_progress_state(state.get("run_progress"))
        policy_payload = state.get("completion_policy")
        if isinstance(policy_payload, Mapping):
            if self._completion_policy is None:
                raise ValueError(
                    "trainer state requires a configured completion policy"
                )
            expected_class = (
                f"{type(self._completion_policy).__module__}."
                f"{type(self._completion_policy).__qualname__}"
            )
            if str(policy_payload.get("class", expected_class)) != expected_class:
                raise ValueError("completion policy checkpoint identity mismatch")
            policy_setter = getattr(self._completion_policy, "set_state", None)
            if not callable(policy_setter):
                raise ValueError(
                    "configured completion policy cannot restore checkpoint state"
                )
            policy_setter(dict(policy_payload.get("state", {}) or {}))
        self._restore_ml_projection_state(state)
        self.history = [dict(row) for row in state.get("history", ())]
        self.run_session.mark_restored(generation)
        self._synchronize_ml_projection(record_history=False)
        return self

    def _record_ml_model_handle(
        self,
        model: Any,
        *,
        candidate_token: str | None = None,
    ) -> None:
        self._step_ml_model_handles.append(
            _MLModelHandle(
                model=model,
                candidate_token=candidate_token,
            )
        )

    def _synchronize_ml_projection(self, *, record_history: bool) -> None:
        envelope = self.get_last_evaluation_event()
        candidate_batch: CandidateBatch | None = None
        feedback_batch: OptimizationFeedbackBatch | None = None
        if envelope is not None:
            if not envelope.semantic_complete:
                raise RuntimeError(
                    "ML projection requires a semantically complete evaluation event"
                )
            if envelope.candidate_codec != "blackbase.candidate_batch/v1":
                raise RuntimeError(
                    f"unsupported ML candidate event codec: {envelope.candidate_codec}"
                )
            if envelope.feedback_codec != "nsgablack.optimization_feedback_batch/v1":
                raise RuntimeError(
                    f"unsupported ML feedback event codec: {envelope.feedback_codec}"
                )
            candidate_batch = CandidateBatch.from_dict(envelope.candidate_payload)
            feedback_batch = OptimizationFeedbackBatch.from_dict(
                envelope.feedback_payload
            )
            if len(candidate_batch.semantic_states) != feedback_batch.candidate_count:
                raise RuntimeError("canonical ML evaluation event is cardinality-inconsistent")

        self.last_evaluated_population = (
            () if candidate_batch is None else tuple(candidate_batch.semantic_states)
        )
        self.last_evaluated_feedback = (
            () if feedback_batch is None else tuple(feedback_batch.items)
        )
        self.feedback = self.last_evaluated_feedback
        incumbent = self.get_incumbent()
        if incumbent is None:
            return
        semantic_metadata = incumbent.metadata.get("candidate.semantic_metadata", {})
        decoded_metadata = (
            decode_shared_value(dict(semantic_metadata))
            if isinstance(semantic_metadata, Mapping)
            else {}
        )
        matching_index: int | None = None
        if candidate_batch is not None and incumbent.candidate_token is not None:
            try:
                matching_index = candidate_batch.candidate_tokens.index(
                    incumbent.candidate_token
                )
            except ValueError:
                matching_index = None
        state = (
            candidate_batch.semantic_states[matching_index]
            if candidate_batch is not None and matching_index is not None
            else UnknownState(
                values=np.asarray(incumbent.candidate, dtype=float).copy(),
                metadata=(
                    dict(decoded_metadata)
                    if isinstance(decoded_metadata, Mapping)
                    else {}
                ),
            )
        )
        changed = (
            self.best_state is None
            or not self._feedback_binding_matches(incumbent)
            or self.best_state.as_dict()["metadata"] != state.as_dict()["metadata"]
            or not self.model_representation.equivalent(self.best_state, state)
        )
        self.best_state = state
        self.best_score = float(incumbent.score)
        matching_handle = next(
            (
                item for item in self._step_ml_model_handles
                if item.candidate_token == incumbent.candidate_token
            ),
            None,
        )
        if matching_index is not None and feedback_batch is not None:
            self.best_feedback = feedback_batch.items[matching_index]
            self._best_feedback_binding = {
                "candidate_token": incumbent.candidate_token,
                "evaluation_id": incumbent.evaluation_id,
                "event_id": None if envelope is None else envelope.event_id,
            }
        elif not self._feedback_binding_matches(incumbent):
            self.best_feedback = None
            self._best_feedback_binding = None
            raise RuntimeError(
                "authoritative ML incumbent has no token-matched canonical feedback"
            )
        if changed or self.best_model is None:
            if matching_handle is not None:
                self.best_model = matching_handle.model
            else:
                # Projection consumes saved evidence. It must never train,
                # evaluate or run an inner Case to reconstruct a best model.
                context = self.enrich_ml_context(self.build_context())
                decoded_model = self.model_representation.decode(state, context)
                self.best_model = self.semantic_problem.restore_model_for_evaluation(
                    decoded_model, state, self.best_feedback, context,
                )
            if changed:
                self.set_best_model_ref(None)
        if not record_history:
            return
        row = {
            "step": int(self.generation),
            "num_candidates": len(self.last_evaluated_population),
            "global_best_score": self.best_score,
            "metrics": (
                {}
                if self.best_feedback is None
                else dict(self.best_feedback.metrics)
            ),
            "metrics_candidate_token": incumbent.candidate_token,
            "metrics_evaluation_id": incumbent.evaluation_id,
            "control_plane": self.control_plane,
        }
        if not self.history or self.history[-1] != row:
            self.history.append(row)

    def _feedback_binding_matches(self, incumbent: IncumbentState | None) -> bool:
        if incumbent is None or self._best_feedback_binding is None:
            return False
        token = incumbent.candidate_token
        if token is None or self._best_feedback_binding.get("candidate_token") != token:
            return False
        bound_evaluation = self._best_feedback_binding.get("evaluation_id")
        return (
            bound_evaluation is None
            or incumbent.evaluation_id is None
            or str(bound_evaluation) == str(incumbent.evaluation_id)
        )


def build_learning_solver(
    *,
    problem: Any,
    representation: Any,
    adapter: Any,
    **kwargs: Any,
) -> LearningSolver:
    """Canonical assembly helper for Adapter-driven ML tasks."""

    return LearningSolver(
        problem=problem,
        representation=representation,
        adapter=adapter,
        **kwargs,
    )


def _initialize_representation(
    representation: Any,
    context: Mapping[str, Any],
) -> UnknownState:
    state = representation.init(dict(context))
    if not isinstance(state, UnknownState):
        raise TypeError(
            "ML ModelRepresentation.init(...) must return blackbase.UnknownState; "
            f"got {type(state).__name__}"
        )
    if state.size <= 0:
        raise ValueError("ML representation must define at least one trainable value")
    return state


def _problem_objective_count(problem: Any) -> int:
    getter = getattr(problem, "get_num_objectives", None)
    count = int(getter()) if callable(getter) else int(getattr(problem, "objective_count", 1))
    if count <= 0:
        raise ValueError("ML LearningProblem must declare a positive objective count")
    return count


def _resource_context_with_namespace(
    value: Mapping[str, Any] | ResourceContext | None,
    run_name: str,
) -> ResourceContext:
    """Every stateful Provider trajectory has an owned, non-empty scope."""

    resource = coerce_resource_context(value)
    if resource.namespace:
        return resource
    payload = resource.as_dict()
    payload["namespace"] = (
        f"mlblack.{str(run_name or 'run')}.{uuid4().hex}"
    )
    metadata = dict(payload.get("metadata", {}) or {})
    metadata["namespace_source"] = "mlblack.standalone_scope"
    payload["metadata"] = metadata
    return ResourceContext.from_mapping(payload)


def _build_compute_backend_session(
    request: Any,
    resource: ResourceContext,
    *,
    explicit: bool,
) -> ComputeBackendSession:
    spec = ComputeBackendSpec.from_value(
        request,
        resource_context=resource.as_dict(),
    )
    if explicit:
        granted_backend = str(resource.compute_backend or "auto").strip().lower()
        spec = ComputeBackendSpec(
            name=(
                spec.name
                if granted_backend in {"", "auto", "cpu", "cuda", "gpu", "tpu"}
                else granted_backend
            ),
            device=str(resource.device),
            device_policy=spec.device_policy,
            metadata={
                **dict(spec.metadata),
                "resource_namespace": str(resource.namespace),
            },
        )
    return ComputeBackendSession(spec)


def _unknown_state(value: Any) -> UnknownState:
    if isinstance(value, UnknownState):
        return value
    return UnknownState(values=np.asarray(value, dtype=float).reshape(-1))


def _state_array(value: Any, *, owner: str) -> np.ndarray:
    if not isinstance(value, UnknownState):
        raise TypeError(f"{owner} must return UnknownState; got {type(value).__name__}")
    return np.asarray(value.as_array(), dtype=float).reshape(-1).copy()


__all__ = [
    "LearningSolver",
    "MLLearningProblemBridge",
    "MLRepresentationBridge",
    "build_learning_solver",
]
