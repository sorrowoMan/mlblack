"""BlackBase Evaluation Provider for Torch-backed ML problem semantics.

The provider owns backend dispatch, autograd and device execution.  It does not
choose SGD/Adam/AdamW or a learning rate; those decisions belong to the
optimization Adapter.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from blackbase.evaluation import (
    EvaluationBinding,
    EvaluationProviderContractError,
    EvaluationProviderSpec,
    EvaluationRequest,
    EvaluationResult,
    StateMaterializationRequest,
    StateMaterializationResult,
    StateReleaseRequest,
    StateReleaseResult,
    StateTransitionMethodSpec,
    StateTransitionRequest,
    StateTransitionResult,
    StateVersionConflict,
)
from blackbase.resources import ResourceRequirement
from blackbase.state_ref import StateRef
from blackbase.types import Feedback, UnknownState

from mlblack.backends.torch_neural.backend import TorchNeuralBackend
from mlblack.core.backend_session import ComputeBackendSession, ComputeBackendSpec
from mlblack.pipeline.datasets import NumericBatch, NumericBatchSchedule


@dataclass(frozen=True)
class TorchEvaluationProviderConfig:
    provider_id: str | None = None
    priority: int = 0
    supported_devices: tuple[str, ...] = ("cpu", "gpu", "mps")
    preferred_devices: tuple[str, ...] = ("gpu", "mps", "cpu")
    random_seed: int = 42
    publish_state_refs: bool = False
    inline_gradients: bool = True
    max_released_state_tombstones: int = 4096
    max_live_evaluation_requests: int = 1


@dataclass
class _TorchStateRecord:
    tensor: Any
    state_kind: str
    scope_id: str
    trajectory_id: str
    device: str
    version: int = 0
    evaluation_request_id: str = ""
    semantic_metadata: Mapping[str, Any] = field(default_factory=dict)


class TorchEvaluationProvider:
    """Evaluate one configured ML Problem/Representation pair with Torch."""

    def __init__(
        self,
        problem: Any,
        representation: Any,
        *,
        data_schedule: NumericBatchSchedule | None = None,
        config: TorchEvaluationProviderConfig | None = None,
    ) -> None:
        self.problem = problem
        self.representation = representation
        self.data_schedule = data_schedule
        self.config = config or TorchEvaluationProviderConfig()
        self.problem_id = evaluation_problem_id(problem)
        self.representation_id = evaluation_representation_id(representation)
        self.route = self._resolve_route()
        self.request_capabilities = self._request_capabilities()
        provider_id = self.config.provider_id or (
            "mlblack.torch."
            + _identifier_fragment(self.problem_id)
            + "."
            + _identifier_fragment(self.representation_id)
            + "/v1"
        )
        backend_capabilities = TorchNeuralBackend().contract().provides
        capabilities = tuple(
            dict.fromkeys((*backend_capabilities, *self.request_capabilities))
        )
        self.spec = EvaluationProviderSpec(
            provider_id=provider_id,
            problem_ids=(self.problem_id,),
            capabilities=capabilities,
            resource_requirement=ResourceRequirement(
                threads=1,
                gpus=0,
                resource_backend="any",
            ),
            supported_devices=self.config.supported_devices,
            preferred_devices=self.config.preferred_devices,
            compute_backend="torch",
            modes=("train", "evaluate", "validate"),
            priority=int(self.config.priority),
            state_kinds=(
                "model_parameters",
                "optimizer_slot.m",
                "optimizer_slot.v",
            ),
            materialization_targets=("unknown_state",),
            transition_methods=(
                StateTransitionMethodSpec(
                    method_id="gradient.sgd",
                    required_operands=("gradient",),
                    operand_state_kinds={"gradient": ("gradient",)},
                    inline_operands=("gradient",),
                    required_parameters=("learning_rate",),
                    optional_parameters=(
                        "min_learning_rate",
                        "weight_decay",
                        "max_gradient_norm",
                    ),
                ),
                StateTransitionMethodSpec(
                    method_id="gradient.adam",
                    required_operands=("gradient",),
                    optional_operands=("first_moment", "second_moment"),
                    operand_state_kinds={"gradient": ("gradient",)},
                    inline_operands=("gradient", "first_moment", "second_moment"),
                    required_parameters=("learning_rate",),
                    optional_parameters=(
                        "min_learning_rate",
                        "weight_decay",
                        "max_gradient_norm",
                        "beta1",
                        "beta2",
                        "epsilon",
                    ),
                    optional_slots=("m", "v"),
                    result_slots=("m", "v"),
                    slot_state_kinds={
                        "m": ("optimizer_slot.m",),
                        "v": ("optimizer_slot.v",),
                    },
                    result_slot_state_kinds={
                        "m": ("optimizer_slot.m",),
                        "v": ("optimizer_slot.v",),
                    },
                ),
                StateTransitionMethodSpec(
                    method_id="gradient.adamw",
                    required_operands=("gradient",),
                    optional_operands=("first_moment", "second_moment"),
                    operand_state_kinds={"gradient": ("gradient",)},
                    inline_operands=("gradient", "first_moment", "second_moment"),
                    required_parameters=("learning_rate",),
                    optional_parameters=(
                        "min_learning_rate",
                        "weight_decay",
                        "max_gradient_norm",
                        "beta1",
                        "beta2",
                        "epsilon",
                    ),
                    optional_slots=("m", "v"),
                    result_slots=("m", "v"),
                    slot_state_kinds={
                        "m": ("optimizer_slot.m",),
                        "v": ("optimizer_slot.v",),
                    },
                    result_slot_state_kinds={
                        "m": ("optimizer_slot.m",),
                        "v": ("optimizer_slot.v",),
                    },
                ),
            ),
            metadata={
                "framework": "mlblack",
                "route": self.route,
                "representation_id": self.representation_id,
                "algorithm_owner": "adapter",
                "resource_authority": "project_l0",
            },
        )
        self._lock = RLock()
        self._evaluation_count = 0
        self._states: dict[str, _TorchStateRecord] = {}
        self._released_versions: dict[str, int] = {}
        self._evaluation_request_order: list[str] = []
        self._provider_scope_id = f"provider:{self.spec.provider_id}:{uuid4().hex}"

    def evaluate(
        self,
        request: EvaluationRequest,
        binding: EvaluationBinding,
    ) -> EvaluationResult:
        if request.problem_id != self.problem_id:
            raise EvaluationProviderContractError(
                f"provider '{self.spec.provider_id}' is configured for "
                f"problem '{self.problem_id}', got '{request.problem_id}'"
            )
        if binding.provider_id != self.spec.provider_id:
            raise EvaluationProviderContractError(
                "TorchEvaluationProvider received a binding owned by another provider"
            )

        if bool(self.config.publish_state_refs):
            self._begin_evaluation_request(request.request_id)
        batch = self._next_batch()
        feedback: list[Feedback] = []
        result_states: list[UnknownState | StateRef] = []
        for index, state in enumerate(request.states):
            if isinstance(state, StateRef):
                state = self.materialize_state(state)
            if not isinstance(state, UnknownState):
                raise TypeError(
                    "TorchEvaluationProvider currently requires UnknownState inputs; "
                    f"got {type(state).__name__} at index {index}"
                )
            context, session = self._execution_context(request, binding)
            try:
                item = self._evaluate_neural_graph(state, context, session, batch)
            finally:
                session.close()
            if bool(self.config.publish_state_refs):
                item, state_ref = self._publish_evaluation_refs(
                    state,
                    item,
                    binding,
                    candidate_index=index,
                    trajectory_id=str(
                        request.payload.get("trajectory_id", request.request_id)
                    ),
                )
                result_states.append(state_ref)
            else:
                result_states.append(state)
            feedback.append(item)

        with self._lock:
            self._evaluation_count += len(feedback)
            evaluation_count = int(self._evaluation_count)
        return EvaluationResult(
            request_id=request.request_id,
            feedback=tuple(feedback),
            result_states=tuple(result_states),
            metadata={
                "provider_id": self.spec.provider_id,
                "problem_id": self.problem_id,
                "route": self.route,
                "evaluation_count": evaluation_count,
            },
        )

    def get_state(self) -> Mapping[str, Any]:
        with self._lock:
            count = int(self._evaluation_count)
            live_state_count = int(len(self._states))
            released_state_count = int(len(self._released_versions))
        return {
            "provider_id": self.spec.provider_id,
            "problem_id": self.problem_id,
            "representation_id": self.representation_id,
            "route": self.route,
            "evaluation_count": count,
            "live_state_count": live_state_count,
            "released_state_count": released_state_count,
            "publish_state_refs": bool(self.config.publish_state_refs),
            "data_schedule": (
                None
                if self.data_schedule is None
                else dict(self.data_schedule.get_state())
            ),
        }

    def checkpoint_identity(self) -> Mapping[str, Any]:
        schedule = self.data_schedule
        schedule_identity = None
        if schedule is not None:
            schedule_identity = {
                "class": (
                    f"{type(schedule).__module__}.{type(schedule).__qualname__}"
                ),
                "n_train": int(schedule.data.X_train.shape[0]),
                "config": {
                    "batch_size": int(schedule.config.batch_size),
                    "shuffle": bool(schedule.config.shuffle),
                    "drop_last": bool(schedule.config.drop_last),
                    "seed": int(schedule.config.seed),
                },
            }
        return {
            "provider_id": self.spec.provider_id,
            "problem_id": self.problem_id,
            "representation_id": self.representation_id,
            "route": self.route,
            "config": asdict(self.config),
            "data_schedule": schedule_identity,
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        if str(state.get("provider_id", self.spec.provider_id)) != self.spec.provider_id:
            raise ValueError("provider checkpoint belongs to another provider")
        if str(state.get("problem_id", self.problem_id)) != self.problem_id:
            raise ValueError("provider checkpoint belongs to another problem")
        if str(state.get("representation_id", self.representation_id)) != self.representation_id:
            raise ValueError("provider checkpoint belongs to another representation")
        if str(state.get("route", self.route)) != self.route:
            raise ValueError("provider checkpoint route does not match current route")
        schedule_state = state.get("data_schedule")
        if schedule_state is not None:
            if self.data_schedule is None:
                raise ValueError("provider checkpoint requires a data schedule")
            self.data_schedule.set_state(dict(schedule_state))
        with self._lock:
            self._evaluation_count = int(state.get("evaluation_count", 0) or 0)
            # StateRef is explicitly live/process-local.  A logical checkpoint
            # cannot claim to restore device buffers it did not persist.
            self._states.clear()
            self._released_versions.clear()
            self._evaluation_request_order.clear()

    def transition(
        self,
        request: StateTransitionRequest,
        binding: EvaluationBinding,
    ) -> StateTransitionResult:
        """Execute a declared first-order update inside the Provider."""

        if binding.provider_id != self.spec.provider_id:
            raise EvaluationProviderContractError(
                "TorchEvaluationProvider received a transition binding owned by another provider"
            )
        with self._lock:
            parameter_record = self._require_record(request.state_ref, "model_parameters")
            gradient = self._resolve_gradient_operand(request.operands.get("gradient"))
            parameter = parameter_record.tensor.detach().clone()
            gradient = gradient.to(device=parameter.device, dtype=parameter.dtype)
            if tuple(gradient.shape) != tuple(parameter.shape):
                raise ValueError(
                    "gradient state shape must match model parameter state: "
                    f"gradient={tuple(gradient.shape)}, parameters={tuple(parameter.shape)}"
                )
            if not bool(gradient.isfinite().all().item()):
                raise ValueError("gradient state must contain only finite values")

            parameters = dict(request.parameters)
            learning_rate = max(
                float(parameters.get("min_learning_rate", 1e-12) or 1e-12),
                float(parameters["learning_rate"]),
            )
            weight_decay = float(parameters.get("weight_decay", 0.0) or 0.0)
            max_norm = parameters.get("max_gradient_norm")
            gradient_norm = float(gradient.norm().detach().cpu().item())
            if max_norm is not None and gradient_norm > float(max_norm) and gradient_norm > 0.0:
                gradient = gradient * (float(max_norm) / gradient_norm)
                gradient_norm = float(max_norm)

            next_slots: dict[str, tuple[Any, StateRef | None]] = {}
            method_id = request.method_id
            if method_id == "gradient.sgd":
                effective = gradient + (weight_decay * parameter) if weight_decay > 0.0 else gradient
                successor_tensor = parameter - (learning_rate * effective)
            else:
                beta1 = float(parameters.get("beta1", 0.9) or 0.9)
                beta2 = float(parameters.get("beta2", 0.999) or 0.999)
                epsilon = float(parameters.get("epsilon", 1e-8) or 1e-8)
                effective = gradient
                if method_id == "gradient.adam" and weight_decay > 0.0:
                    effective = gradient + (weight_decay * parameter)
                m_ref = request.slot_refs.get("m")
                v_ref = request.slot_refs.get("v")
                m = self._slot_tensor(
                    m_ref,
                    parameter,
                    "m",
                    initial=request.operands.get("first_moment"),
                )
                v = self._slot_tensor(
                    v_ref,
                    parameter,
                    "v",
                    initial=request.operands.get("second_moment"),
                )
                m_next = (beta1 * m) + ((1.0 - beta1) * effective)
                v_next = (beta2 * v) + ((1.0 - beta2) * (effective ** 2))
                time_index = int(request.step_index) + 1
                m_hat = m_next / (1.0 - (beta1 ** time_index))
                v_hat = v_next / (1.0 - (beta2 ** time_index))
                delta = learning_rate * m_hat / (v_hat.sqrt() + epsilon)
                if method_id == "gradient.adamw" and weight_decay > 0.0:
                    delta = delta + (learning_rate * weight_decay * parameter)
                successor_tensor = parameter - delta
                next_slots = {
                    "m": (m_next, m_ref),
                    "v": (v_next, v_ref),
                }

            successor_ref = request.state_ref.next_version(
                metadata={
                    **dict(request.state_ref.metadata),
                    "last_method_id": method_id,
                    "last_step_index": int(request.step_index),
                }
            )
            next_state_map = dict(self._states)
            next_state_map[request.state_ref.state_id] = _TorchStateRecord(
                tensor=successor_tensor.detach().clone(),
                state_kind=parameter_record.state_kind,
                scope_id=parameter_record.scope_id,
                trajectory_id=parameter_record.trajectory_id,
                device=parameter_record.device,
                version=int(parameter_record.version) + 1,
                evaluation_request_id=parameter_record.evaluation_request_id,
                semantic_metadata=dict(parameter_record.semantic_metadata),
            )
            slot_refs: dict[str, StateRef] = {}
            for name, (tensor, previous_ref) in next_slots.items():
                state_kind = f"optimizer_slot.{name}"
                if previous_ref is None:
                    state_id = f"slot-{name}-{uuid4().hex}"
                    next_state_map[state_id] = _TorchStateRecord(
                        tensor=tensor.detach().clone(),
                        state_kind=state_kind,
                        scope_id=successor_ref.scope_id,
                        trajectory_id=successor_ref.trajectory_id,
                        device=successor_ref.device,
                        semantic_metadata={},
                    )
                    slot_refs[name] = StateRef(
                        provider_id=self.spec.provider_id,
                        state_id=state_id,
                        state_kind=state_kind,
                        scope_id=successor_ref.scope_id,
                        trajectory_id=successor_ref.trajectory_id,
                        device=successor_ref.device,
                    )
                else:
                    previous = self._require_record(previous_ref, state_kind)
                    next_state_map[previous_ref.state_id] = _TorchStateRecord(
                        tensor=tensor.detach().clone(),
                        state_kind=previous.state_kind,
                        scope_id=previous.scope_id,
                        trajectory_id=previous.trajectory_id,
                        device=previous.device,
                        version=int(previous.version) + 1,
                        evaluation_request_id=previous.evaluation_request_id,
                        semantic_metadata=dict(previous.semantic_metadata),
                    )
                    slot_refs[name] = previous_ref.next_version()
            released_gradient: tuple[str, int] | None = None
            if isinstance(request.operands.get("gradient"), StateRef):
                gradient_ref = request.operands["gradient"]
                gradient_record = next_state_map.pop(gradient_ref.state_id, None)
                if gradient_record is not None:
                    released_gradient = (
                        gradient_ref.state_id,
                        int(gradient_record.version),
                    )
            # Publish the complete parameter/slot transaction with one map swap.
            self._states = next_state_map
            if released_gradient is not None:
                self._remember_released(*released_gradient)
        return StateTransitionResult(
            request_id=request.request_id,
            method_id=method_id,
            status="applied",
            state_ref=successor_ref,
            slot_refs=slot_refs,
            metrics={
                "gradient_norm": gradient_norm,
                "learning_rate": learning_rate,
            },
            metadata={"provider_state_update": True},
        )

    def materialize_state(self, state_ref: StateRef) -> UnknownState:
        """Explicitly export one numeric process-local state."""

        with self._lock:
            record = self._require_record(state_ref, state_ref.state_kind)
            values = record.tensor.detach().cpu().numpy().astype(float, copy=True)
        return UnknownState(
            values=values,
            metadata={
                **dict(record.semantic_metadata),
                "provider_state_kind": record.state_kind,
            },
        )

    def materialize(
        self,
        request: StateMaterializationRequest,
        binding: EvaluationBinding,
    ) -> StateMaterializationResult:
        if binding.provider_id != self.spec.provider_id:
            raise EvaluationProviderContractError(
                "TorchEvaluationProvider received a materialization binding owned by another provider"
            )
        if request.target != "unknown_state":
            raise ValueError(
                "TorchEvaluationProvider currently materializes only unknown_state"
            )
        # Validation, export and optional release are one version-fenced
        # transaction.  Splitting export and release across lock acquisitions
        # would allow a concurrent transition to be deleted after an older
        # value had already been copied.
        with self._lock:
            record = self._require_record(
                request.state_ref,
                request.state_ref.state_kind,
            )
            value = UnknownState(
                values=record.tensor.detach().cpu().numpy().astype(float, copy=True),
                metadata={
                    **dict(record.semantic_metadata),
                    "provider_state_kind": record.state_kind,
                },
            )
            result = StateMaterializationResult(
                request_id=request.request_id,
                state_ref=request.state_ref,
                target=request.target,
                value=value,
                metadata={
                    "provider_state_export": True,
                    "released": bool(request.release_after),
                },
            )
            if request.release_after:
                self._states.pop(request.state_ref.state_id)
                self._remember_released(request.state_ref.state_id, record.version)
            return result

    def release(
        self,
        request: StateReleaseRequest,
        binding: EvaluationBinding,
    ) -> StateReleaseResult:
        """Atomically tear down one Provider-owned scope/trajectory."""

        if binding.provider_id != self.spec.provider_id:
            raise EvaluationProviderContractError(
                "TorchEvaluationProvider received a release binding owned by another provider"
            )
        kinds = set(request.state_kinds)
        with self._lock:
            released: list[tuple[str, int]] = []
            next_state_map = dict(self._states)
            for state_id, record in tuple(self._states.items()):
                if request.scope_id and record.scope_id != request.scope_id:
                    continue
                if request.trajectory_id and record.trajectory_id != request.trajectory_id:
                    continue
                if kinds and record.state_kind.lower() not in kinds:
                    continue
                next_state_map.pop(state_id, None)
                released.append((state_id, int(record.version)))
            self._states = next_state_map
            for state_id, version in released:
                self._remember_released(state_id, version)
        return StateReleaseResult(
            request_id=request.request_id,
            provider_id=self.spec.provider_id,
            status="released" if released else "not_found",
            released_count=len(released),
            released_state_ids=tuple(state_id for state_id, _ in released[:64]),
            metadata={
                "scope_id": request.scope_id,
                "trajectory_id": request.trajectory_id,
                "released_state_id_sample_truncated": len(released) > 64,
            },
        )

    def _publish_evaluation_refs(
        self,
        state: UnknownState,
        feedback: Feedback,
        binding: EvaluationBinding,
        *,
        candidate_index: int,
        trajectory_id: str,
    ) -> tuple[Feedback, StateRef]:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("TorchEvaluationProvider requires torch") from exc
        device_name = _torch_device_name(binding.device)
        scope_id = str(
            binding.resource_context.namespace
            or self._provider_scope_id
        )
        parameter_id = f"parameters-{uuid4().hex}"
        gradient_id = f"gradient-{uuid4().hex}"
        parameter = torch.as_tensor(
            np.asarray(state.as_array(), dtype=np.float32),
            dtype=torch.float32,
            device=torch.device(device_name),
        ).reshape(-1)
        gradient_values = feedback.gradients
        if gradient_values is None:
            raise ValueError("publish_state_refs requires inline provider gradients")
        gradient = torch.as_tensor(
            np.asarray(gradient_values, dtype=np.float32),
            dtype=torch.float32,
            device=torch.device(device_name),
        ).reshape(-1)
        state_ref = StateRef(
            provider_id=self.spec.provider_id,
            state_id=parameter_id,
            state_kind="model_parameters",
            scope_id=scope_id,
            trajectory_id=str(trajectory_id),
            device=binding.device,
            metadata={"candidate_index": int(candidate_index)},
        )
        gradient_ref = StateRef(
            provider_id=self.spec.provider_id,
            state_id=gradient_id,
            state_kind="gradient",
            scope_id=scope_id,
            trajectory_id=str(trajectory_id),
            device=binding.device,
            metadata={"candidate_index": int(candidate_index)},
        )
        with self._lock:
            self._states[parameter_id] = _TorchStateRecord(
                tensor=parameter.detach().clone(),
                state_kind="model_parameters",
                scope_id=scope_id,
                trajectory_id=str(trajectory_id),
                device=binding.device,
                evaluation_request_id=str(binding.request_id),
                semantic_metadata=dict(state.metadata),
            )
            self._states[gradient_id] = _TorchStateRecord(
                tensor=gradient.detach().clone(),
                state_kind="gradient",
                scope_id=scope_id,
                trajectory_id=str(trajectory_id),
                device=binding.device,
                evaluation_request_id=str(binding.request_id),
                semantic_metadata={},
            )
        return (
            Feedback(
                objectives=np.array(feedback.objectives, dtype=float, copy=True),
                constraints=np.array(feedback.constraints, dtype=float, copy=True),
                gradients=(
                    np.array(feedback.gradients, dtype=float, copy=True)
                    if bool(self.config.inline_gradients)
                    else None
                ),
                gradient_ref=gradient_ref,
                loss=feedback.loss,
                metrics=dict(feedback.metrics),
                residuals=(
                    None
                    if feedback.residuals is None
                    else np.array(feedback.residuals, dtype=float, copy=True)
                ),
                signals=dict(feedback.signals),
                info=dict(feedback.info),
            ),
            state_ref,
        )

    def _require_record(self, state_ref: StateRef, expected_kind: str) -> _TorchStateRecord:
        if state_ref.provider_id != self.spec.provider_id:
            raise ValueError("StateRef belongs to another provider")
        record = self._states.get(state_ref.state_id)
        if record is None:
            released_version = self._released_versions.get(state_ref.state_id)
            if released_version is not None:
                raise StateVersionConflict(
                    state_ref,
                    actual_version=released_version,
                )
            raise KeyError(f"unknown or released provider state: {state_ref.state_id}")
        if record.state_kind != expected_kind or state_ref.state_kind != expected_kind:
            raise ValueError(
                f"StateRef kind mismatch: expected {expected_kind}, got {state_ref.state_kind}"
            )
        if (
            record.scope_id != state_ref.scope_id
            or record.trajectory_id != state_ref.trajectory_id
            or record.device != state_ref.device
        ):
            raise ValueError("StateRef scope/trajectory/device does not match provider state")
        if int(record.version) != int(state_ref.version):
            raise StateVersionConflict(state_ref, actual_version=record.version)
        return record

    def _remember_released(self, state_id: str, version: int) -> None:
        self._released_versions[str(state_id)] = int(version)
        limit = max(1, int(self.config.max_released_state_tombstones))
        while len(self._released_versions) > limit:
            oldest = next(iter(self._released_versions))
            self._released_versions.pop(oldest, None)

    def _begin_evaluation_request(self, request_id: str) -> None:
        """Keep ephemeral parameter/gradient refs bounded by request count."""

        with self._lock:
            normalized = str(request_id)
            self._evaluation_request_order.append(normalized)
            limit = max(1, int(self.config.max_live_evaluation_requests))
            while len(self._evaluation_request_order) > limit:
                expired = self._evaluation_request_order.pop(0)
                for state_id, record in tuple(self._states.items()):
                    if record.evaluation_request_id != expired:
                        continue
                    self._states.pop(state_id, None)
                    self._remember_released(state_id, record.version)

    def _resolve_gradient_operand(self, value: Any) -> Any:
        if isinstance(value, StateRef):
            return self._require_record(value, "gradient").tensor.detach().clone()
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("TorchEvaluationProvider requires torch") from exc
        return torch.as_tensor(value, dtype=torch.float32).reshape(-1)

    def _slot_tensor(
        self,
        state_ref: StateRef | None,
        parameter: Any,
        name: str,
        *,
        initial: Any = None,
    ) -> Any:
        if state_ref is None:
            if initial is None:
                return parameter.new_zeros(parameter.shape)
            try:
                import torch
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("TorchEvaluationProvider requires torch") from exc
            value = torch.as_tensor(
                np.array(initial, dtype=float, copy=True),
                dtype=parameter.dtype,
                device=parameter.device,
            ).reshape(-1)
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"initial optimizer slot '{name}' must match parameter shape"
                )
            return value.detach().clone()
        return self._require_record(state_ref, f"optimizer_slot.{name}").tensor.detach().clone()

    def _resolve_route(self) -> str:
        if callable(getattr(self.problem, "compute_backend_loss", None)):
            return "neural_graph"
        raise TypeError(
            "TorchEvaluationProvider requires a neural Problem exposing "
            "compute_backend_loss(...)"
        )

    def _request_capabilities(self) -> tuple[str, ...]:
        common = (
            "autograd.backward",
            "autograd.zero_grad",
            "autograd.gradients.flat_export",
            "tensor.device",
        )
        return tuple(
            dict.fromkeys(
                (
                    *common,
                    *tuple(getattr(self.representation, "backend_requires", ()) or ()),
                    *tuple(getattr(self.problem, "backend_requires", ()) or ()),
                )
            )
        )

    def _next_batch(self) -> NumericBatch | None:
        schedule = self.data_schedule
        if schedule is None:
            return None
        with self._lock:
            return schedule.next_train()

    def _execution_context(
        self,
        request: EvaluationRequest,
        binding: EvaluationBinding,
    ) -> tuple[dict[str, Any], ComputeBackendSession]:
        device = _torch_device_name(binding.device)
        session = ComputeBackendSession(
            ComputeBackendSpec(
                name="torch",
                device=device,
                device_policy="strict",
                metadata={
                    "provider_id": self.spec.provider_id,
                    "binding_id": binding.binding_id,
                },
            )
        )
        resource_context = binding.resource_context.as_dict()
        context = {
            **dict(request.payload),
            "evaluation.mode": request.mode,
            "evaluation.request_id": request.request_id,
            "evaluation.binding_id": binding.binding_id,
            "resource_context": resource_context,
            "resource": resource_context,
            **binding.resource_context.context_items(prefix="resource"),
            **session.context_items(),
            "resource.device": device,
        }
        return context, session

    def _evaluate_neural_graph(
        self,
        state: UnknownState,
        context: Mapping[str, Any],
        session: ComputeBackendSession,
        batch: NumericBatch | None = None,
    ) -> Feedback:
        context = dict(context)
        if batch is not None:
            context["data.batch"] = batch
        requirements = tuple(
            dict.fromkeys(
                (
                    *self.request_capabilities,
                    *tuple(getattr(self.representation, "backend_requires", ()) or ()),
                    *tuple(getattr(self.problem, "backend_requires", ()) or ()),
                )
            )
        )
        backend = session.ensure(requirements, consumer=self.spec.provider_id)
        model = self.representation.decode(state, context)
        device = backend.tensor.device(context, strict=True)
        backend.autograd.train(model, device=device)
        backend.autograd.zero_grad(model)
        evaluation = self.problem.compute_backend_loss(
            model,
            state,
            context,
            differentiable=True,
        )
        loss = getattr(evaluation, "loss", None)
        if loss is None:
            raise ValueError(
                "problem.compute_backend_loss(...) must return a backend-native loss"
            )
        backend.autograd.backward(loss)
        gradient = np.asarray(backend.autograd.flat_grads(model), dtype=float).reshape(-1)
        semantic = evaluation.as_feedback()
        return _feedback_with_gradient(
            semantic,
            gradient,
            provider_id=self.spec.provider_id,
            device=str(device),
            backend_loss=float(getattr(evaluation, "loss_value")),
            gradient_norm=float(np.linalg.norm(gradient)),
            batch_indices=() if batch is None else tuple(batch.indices),
            batch_metadata={} if batch is None else dict(batch.metadata),
        )


def evaluation_problem_id(problem: Any) -> str:
    explicit = str(getattr(problem, "evaluation_problem_id", "") or "").strip().lower()
    if explicit:
        return explicit
    name = str(getattr(problem, "name", type(problem).__name__) or "problem").strip().lower()
    return f"mlblack.{_identifier_fragment(name)}/v1"


def evaluation_representation_id(representation: Any) -> str:
    """Stable identity for the exact representation bound to one Provider."""

    explicit = str(
        getattr(representation, "evaluation_representation_id", "") or ""
    ).strip().lower()
    if explicit:
        return explicit
    describe = getattr(representation, "describe", None)
    description = dict(describe()) if callable(describe) else {}
    payload = {
        "module": str(type(representation).__module__),
        "class": str(type(representation).__qualname__),
        "description": description,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: (
            value.as_dict()
            if callable(getattr(value, "as_dict", None))
            else repr(value)
        ),
    ).encode("utf-8")
    return "representation-" + hashlib.sha256(encoded).hexdigest()[:20]


def _identifier_fragment(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in {".", "-", "_"} else "-"
        for character in str(value).strip().lower()
    )
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized.strip("-.") or "problem"


def _torch_device_name(device: str) -> str:
    value = str(device or "cpu").strip().lower()
    if value == "gpu":
        return "cuda"
    if value.startswith("gpu:"):
        return "cuda:" + value.split(":", 1)[1]
    return value


def _feedback_with_gradient(
    feedback: Feedback,
    gradient: np.ndarray,
    *,
    provider_id: str,
    device: str,
    backend_loss: float,
    gradient_norm: float,
    batch_indices: Sequence[int],
    batch_metadata: Mapping[str, Any],
) -> Feedback:
    metrics = dict(feedback.metrics or {})
    metrics.update(
        {
            "provider.backend_loss": float(backend_loss),
            "provider.gradient_norm": float(gradient_norm),
            "provider.batch_size": int(len(tuple(batch_indices))),
        }
    )
    signals = dict(feedback.signals or {})
    signals.update(
        {
            "evaluation_provider": str(provider_id),
            "compute_backend": "torch",
            "device": str(device),
            "batch_indices": tuple(int(value) for value in batch_indices),
            "batch": dict(batch_metadata),
        }
    )
    return Feedback(
        objectives=np.array(feedback.objectives, dtype=float, copy=True),
        constraints=np.array(feedback.constraints, dtype=float, copy=True),
        gradients=np.array(gradient, dtype=float, copy=True),
        gradient_ref=feedback.gradient_ref,
        loss=None if feedback.loss is None else float(feedback.loss),
        metrics=metrics,
        residuals=(
            None
            if feedback.residuals is None
            else np.array(feedback.residuals, dtype=float, copy=True)
        ),
        signals=signals,
        info=dict(feedback.info or {}),
    )


__all__ = [
    "TorchEvaluationProvider",
    "TorchEvaluationProviderConfig",
    "evaluation_problem_id",
    "evaluation_representation_id",
]
