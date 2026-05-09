from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from core.common.contracts import ProcessedDataset, SampleDataset, SurrogateArtifact
from core.execution import ExecutionResourceRequest, detect_local_execution_offer, normalize_execution_device_token
from training.capabilities import TrainerCapabilities, coerce_trainer_capabilities
from training.compatibility import require_training_setup
from training.init import TrainingInit
from training.lineage import TrainingLineage
from training.result import FitResult
from training.signatures import attach_signature_to_artifact, build_task_signature
from training.task import TrainTask


class BaseSurrogateTrainer(ABC):
    """Unified trainer contract for pluggable surrogate backends.

    Legacy trainer implementations can keep implementing `fit(data) -> artifact`.
    The newer training-control plane should call `fit_task(...)`, which bridges
    the legacy data-only contract into a structured FitResult.
    """

    name = "base_trainer"

    @abstractmethod
    def fit(self, data: ProcessedDataset | SampleDataset) -> SurrogateArtifact:
        ...

    def capabilities(self) -> TrainerCapabilities | Mapping[str, Any]:
        """Describes trainer behavior for registry/UI inspection."""
        return TrainerCapabilities(
            metadata={
                "trainer_name": str(getattr(self, "name", type(self).__name__)),
            }
        )

    def execution_resource_request(self) -> ExecutionResourceRequest | Mapping[str, Any]:
        """Describe the total resources consumed by one trainer run.

        Control-plane orchestration can aggregate these requests across concurrent
        runs without embedding trainer-family-specific heuristics.
        """
        return ExecutionResourceRequest(
            threads=1,
            backend="serial",
            label=str(getattr(self, "name", type(self).__name__)),
        )

    def execution_resource_requests(self) -> Sequence[ExecutionResourceRequest | Mapping[str, Any]]:
        """Return resource request components for this trainer run.

        Override this when a trainer composes multiple planes that should be
        budgeted separately, for example trainer + evaluation/problem helpers.
        """
        return (self.execution_resource_request(),)

    def portfolio_execution_resource_requests(
        self,
        *,
        run_spec: Any | None = None,
        model_spec: Any | None = None,
    ) -> Sequence[ExecutionResourceRequest | Mapping[str, Any]]:
        """Return per-run resource components for portfolio scheduling.

        Trainers can override this when portfolio-time resource accounting
        depends on model selection or scenario-level helpers that are only known
        after the flow spec is assembled.
        """
        _ = (run_spec, model_spec)
        return self.execution_resource_requests()

    @staticmethod
    def resolve_execution_device_tokens(requested: str | int | None) -> tuple[str, ...]:
        if requested is None:
            return tuple()
        raw = str(requested).strip().lower()
        if raw in {"", "none", "cpu"}:
            return tuple()
        if raw == "auto":
            offer = detect_local_execution_offer()
            if offer.cuda_devices:
                return (str(offer.cuda_devices[0]),)
            if offer.mps_devices:
                return (str(offer.mps_devices[0]),)
            return tuple()
        token = normalize_execution_device_token(requested)
        return tuple() if str(token) == "cpu" else (str(token),)

    def fit_task(
        self,
        task: TrainTask,
        init: TrainingInit | None = None,
    ) -> FitResult:
        init_eff = init or TrainingInit()
        caps = coerce_trainer_capabilities(self.capabilities())
        task_signature = build_task_signature(task, trainer=self)
        verdict = require_training_setup(
            caps,
            init_eff,
            trainer_name=str(getattr(self, "name", type(self).__name__)),
            current_signature=task_signature,
        )
        artifact = self.fit(task.data)
        attach_signature_to_artifact(artifact, task_signature)
        lineage = TrainingLineage(
            mode=str(init_eff.mode),
            trainer_name=str(getattr(self, "name", type(self).__name__)),
            parent_artifact_id=(
                None
                if init_eff.parent_artifact is None
                else str(getattr(init_eff.parent_artifact, "artifact_id", type(init_eff.parent_artifact).__name__))
            ),
            parent_state_trainer=(
                None
                if init_eff.parent_state is None
                else str(getattr(init_eff.parent_state, "trainer_name", type(init_eff.parent_state).__name__))
            ),
            metadata={
                "task_id": str(task.task_id),
                "task_metadata": dict(task.metadata),
                "task_signature": task_signature.as_dict(),
            },
        )
        return FitResult(
            artifact=artifact,
            trainer_state=None,
            report={
                "training_mode": str(init_eff.mode),
                "trainer_capabilities": caps.as_dict(),
                "task_signature": task_signature.as_dict(),
                "compatibility": verdict.metadata,
                "compatibility_warnings": list(verdict.warnings),
            },
            lineage=lineage,
        )
