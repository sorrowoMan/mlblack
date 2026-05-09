from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from bias import BaseTrainingBias, FitContext, NoOpBias
from numericizer import BaseNumericizer, DefaultNumericizer, ModalityEncoder, TargetCodec
from pipeline import BasePipeline, IdentityPipeline

from core.artifacts.artifact import LinearSurrogateArtifact
from core.common.base_trainer import BaseSurrogateTrainer
from core.common.contracts import ProcessedDataset, SampleDataset
from core.linear.trainer_family import LinearTrainerFamilySpec, build_ridge_family_spec, coerce_linear_family_spec
from training import (
    FitResult,
    TrainTask,
    TrainerState,
    TrainingInit,
    TrainingLineage,
    attach_signature_to_artifact,
    build_task_signature,
    clone_pickled_trainer_payload,
    coerce_trainer_capabilities,
    coerce_training_signature,
    load_pickled_trainer_state_file,
    require_training_setup,
    save_pickled_trainer_state_file,
)


@dataclass(frozen=True)
class RidgeTrainerConfig:
    l2: float = 1.0
    ood_z_threshold: float = 4.0
    artifact_id: str = "ridge_surrogate_v1"
    family_spec: LinearTrainerFamilySpec | Mapping[str, Any] | None = None


class RidgeSurrogateTrainer(BaseSurrogateTrainer):
    """Core trainer that depends on pipeline, bias, and numericizer layers.

    Supports both:
    - ProcessedDataset: traditional numeric matrix training
    - SampleDataset: object-first multi-modal samples converted by numericizer
    """

    name = "ridge"

    def __init__(
        self,
        config: RidgeTrainerConfig | None = None,
        *,
        pipeline: BasePipeline | None = None,
        biases: Sequence[BaseTrainingBias] | None = None,
        numericizer: BaseNumericizer | None = None,
        modality_encoders: Mapping[str, ModalityEncoder] | None = None,
        target_codecs: Mapping[str, TargetCodec] | None = None,
        target_codec: str | None = None,
        categorical_unknown: str = "error",
    ) -> None:
        self.config = config or RidgeTrainerConfig()
        self.linear_family_spec = self._resolve_family_spec(self.config)
        self.pipeline = pipeline or IdentityPipeline()
        self.biases = list(biases) if biases else [NoOpBias()]
        try:
            setattr(self, "linear_family_metadata", self.linear_family_spec.description_dict())
        except Exception:
            pass

        if numericizer is not None and (
            modality_encoders is not None or target_codecs is not None or target_codec is not None
        ):
            raise ValueError("Provide either numericizer or encoder/codec options, not both")

        if numericizer is not None:
            self.numericizer = numericizer
        else:
            self.numericizer = DefaultNumericizer(
                modality_encoders=modality_encoders,
                target_codecs=target_codecs,
                target_codec=target_codec,
                categorical_unknown=categorical_unknown,
            )

    def _normalize_data(self, data: ProcessedDataset | SampleDataset) -> ProcessedDataset:
        return self.numericizer.to_processed(data)

    @staticmethod
    def _resolve_family_spec(config: RidgeTrainerConfig | None) -> LinearTrainerFamilySpec:
        cfg = config or RidgeTrainerConfig()
        raw = getattr(cfg, "family_spec", None)
        if raw is not None:
            return coerce_linear_family_spec(raw, trainer_key="ridge")
        return build_ridge_family_spec(
            trainer_key="ridge",
            l2=float(cfg.l2),
        )

    def _current_family_payload(self) -> dict[str, object]:
        return self.linear_family_spec.description_dict()

    def _current_family_signature(self) -> str | None:
        return self.linear_family_spec.family_signature()

    def _assert_parent_family_compatible(self, payload: Mapping[str, object] | None) -> None:
        if payload is None:
            return
        current_sig = self._current_family_signature()
        parent_sig = payload.get("linear_family_signature")
        if current_sig is None or parent_sig is None:
            return
        if str(current_sig) != str(parent_sig):
            raise ValueError(
                f"{self.name} continuation rejected because linear family components changed "
                f"(current={current_sig}, parent={parent_sig})"
            )

    def _validate_family_surface(self) -> None:
        family = self.linear_family_spec
        if str(family.backend.solver_kind) != "ridge":
            raise ValueError("ridge trainer requires linear_family backend.solver_kind='ridge'")
        if str(family.regularization.penalty) != "l2":
            raise ValueError("ridge trainer currently supports only linear_family regularization.penalty='l2'")
        if str(family.task_head.task) != "point" or str(family.task_head.objective_family) != "regression":
            raise ValueError("ridge trainer currently supports only point/regression task_head")
        if not bool(family.function_class.fit_intercept):
            raise ValueError("ridge trainer currently requires linear_family function_class.fit_intercept=True")

    def capabilities(self) -> dict[str, object]:
        family = self.linear_family_spec
        return {
            "supports_fresh": True,
            "supports_resume": bool(family.backend.supports_resume),
            "supports_warm_start": bool(family.backend.supports_warm_start),
            "supports_incremental": bool(family.backend.supports_incremental),
            "supports_recalibration": False,
            "name": self.name,
            "model_family": "linear",
            "backend": str(family.backend.runtime_backend),
            "nonlinear": False,
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "sample_weight": True,
                "target_codec": True,
                "linear_family_spec": True,
            },
            "artifacts": {
                "type": "LinearSurrogateArtifact",
                "uncertainty": str(family.task_head.uncertainty_mode),
                "ood_validity": True,
            },
            "runtime": {
                "trainer_state": bool(family.backend.trainer_state_enabled),
                "resume_semantics": str(family.backend.continuation_mode),
                "warm_start_semantics": str(family.backend.continuation_mode),
                "incremental_semantics": str(family.backend.continuation_mode),
            },
            "linear_family": family.description_dict(),
        }

    @staticmethod
    def _clone_payload_cpu(value: object) -> object:
        return clone_pickled_trainer_payload(value)

    @staticmethod
    def _clone_state_cpu(state: Mapping[str, object]) -> dict[str, object]:
        return dict(clone_pickled_trainer_payload(dict(state)))

    @classmethod
    def save_trainer_state(cls, path: str | Path, state: TrainerState) -> str:
        return save_pickled_trainer_state_file(
            path,
            trainer_name=str(getattr(state, "trainer_name", cls.name)),
            payload=dict(getattr(state, "payload", {})),
            metadata=dict(getattr(state, "metadata", {})),
        )

    @classmethod
    def load_trainer_state(cls, path: str | Path) -> TrainerState:
        resume_path = Path(path).resolve()
        payload = load_pickled_trainer_state_file(resume_path)
        signature = coerce_training_signature(payload.get("training_signature"))
        return TrainerState(
            trainer_name=str(payload.get("trainer_name", cls.name)),
            payload=dict(payload),
            schema_signature=signature.schema_signature,
            feature_signature=signature.feature_signature,
            target_signature=signature.target_signature,
            objective_signature=signature.objective_signature,
            pipeline_signature=signature.pipeline_signature,
            numericizer_signature=signature.numericizer_signature,
            regime_signature=signature.regime_signature,
            symbolic_family_signature=signature.symbolic_family_signature,
            metadata={
                "resume_source": str(resume_path),
                "epoch_done": 0,
                "training_signature": signature.as_dict(),
                "linear_family_signature": payload.get("linear_family_signature"),
            },
        )

    @staticmethod
    def _payload_from_artifact(artifact: object) -> dict[str, object] | None:
        if not isinstance(artifact, LinearSurrogateArtifact):
            return None
        metadata = dict(getattr(artifact, "metadata", {}) or {})
        return {
            "schema_version": 1,
            "trainer_name": "ridge",
            "coef": np.asarray(artifact.coef, dtype=float),
            "intercept": np.asarray(artifact.intercept, dtype=float),
            "x_mean": np.asarray(artifact.x_mean, dtype=float),
            "x_std": np.asarray(artifact.x_std, dtype=float),
            "residual_std": np.asarray(artifact.residual_std, dtype=float),
            "feature_names": tuple(str(v) for v in artifact.feature_names),
            "target_names": tuple(str(v) for v in artifact.target_names),
            "l2": float(metadata.get("l2", 0.0)),
            "l2_effective": float(metadata.get("l2_effective", 0.0)),
            "linear_family": metadata.get("linear_family"),
            "linear_family_signature": metadata.get("linear_family_signature"),
            "training_signature": metadata.get("training_signature"),
        }

    @staticmethod
    def _solve_ridge(
        *,
        Xb: np.ndarray,
        Yb: np.ndarray,
        l2_eff: float,
        sample_weight: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, d = Xb.shape
        Xa = np.hstack([Xb, np.ones((n, 1), dtype=float)])

        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=float).reshape(-1)
            if w.shape[0] != n:
                raise ValueError("sample_weight length mismatch")
            sw = np.sqrt(np.maximum(w, 0.0)).reshape(-1, 1)
            Xa_fit = Xa * sw
            Y_fit = Yb * sw
        else:
            Xa_fit = Xa
            Y_fit = Yb

        reg = np.eye(d + 1, dtype=float) * float(l2_eff)
        reg[d, d] = 0.0
        coef_all = np.linalg.solve(Xa_fit.T @ Xa_fit + reg, Xa_fit.T @ Y_fit)
        coef = np.asarray(coef_all[:d, :], dtype=float)
        intercept = np.asarray(coef_all[d, :], dtype=float)
        pred = np.asarray(Xb @ coef + intercept, dtype=float)
        return coef, intercept, pred

    def _fit_internal(
        self,
        data: ProcessedDataset | SampleDataset,
        *,
        init: TrainingInit | None = None,
        training_signature: Mapping[str, object] | None = None,
    ) -> tuple[LinearSurrogateArtifact, TrainerState | None]:
        self._validate_family_surface()

        init_eff = init or TrainingInit()
        mode = str(init_eff.mode).strip().lower() or "fresh"
        parent_payload: dict[str, object] | None = None
        parent_source: str | None = None
        parent_kind: str | None = None
        if init_eff.parent_state is not None:
            parent_payload = self._clone_state_cpu(getattr(init_eff.parent_state, "payload", {}))
            parent_source = str(
                dict(getattr(init_eff.parent_state, "metadata", {}) or {}).get("resume_source")
                or getattr(init_eff.parent_state, "trainer_name", type(init_eff.parent_state).__name__)
            )
            parent_kind = "trainer_state"
        elif init_eff.parent_artifact is not None:
            parent_payload = self._payload_from_artifact(init_eff.parent_artifact)
            parent_source = str(getattr(init_eff.parent_artifact, "artifact_id", type(init_eff.parent_artifact).__name__))
            parent_kind = "artifact"

        self._assert_parent_family_compatible(parent_payload)

        normalized = self._normalize_data(data)

        X = np.asarray(normalized.X_train, dtype=float)
        Y = np.asarray(normalized.y_train, dtype=float)
        if X.ndim != 2:
            raise ValueError("X_train must be 2D")
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y.ndim != 2:
            raise ValueError("y_train must be 1D or 2D")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X_train and y_train row count mismatch")

        Xp = self.pipeline.fit_transform(X, Y)

        context = FitContext()
        Xb = np.asarray(Xp, dtype=float)
        Yb = np.asarray(Y, dtype=float)
        for bias in self.biases:
            Xb, Yb = bias.apply(Xb, Yb, context)

        n, d = Xb.shape
        m = Yb.shape[1]

        family_payload = self._current_family_payload()
        family_signature = self._current_family_signature()
        l2_value = float(self.linear_family_spec.regularization.l2)
        l2_eff = float(l2_value) * float(context.l2_multiplier)
        coef, intercept, pred = self._solve_ridge(
            Xb=Xb,
            Yb=Yb,
            l2_eff=l2_eff,
            sample_weight=context.sample_weight,
        )
        residual = Yb - pred
        residual_std = np.std(residual, axis=0, ddof=1) + 1e-8

        if normalized.feature_names is not None and len(tuple(normalized.feature_names)) == d:
            feature_names = tuple(normalized.feature_names)
        else:
            feature_names = tuple(f"x{i}" for i in range(d))

        target_names = (
            tuple(normalized.target_names)
            if normalized.target_names is not None
            else tuple(f"y{i}" for i in range(m))
        )

        metadata = {
            "trainer": "RidgeSurrogateTrainer",
            "l2": float(l2_value),
            "l2_effective": float(l2_eff),
            "n_train": int(n),
            "feature_dim": int(d),
            "target_dim": int(m),
            "pipeline": str(getattr(self.pipeline, "name", "identity")),
            "biases": [str(getattr(b, "name", type(b).__name__)) for b in self.biases],
            "fit_context": dict(context.metadata),
            "data_protocol": str((normalized.metadata or {}).get("input_protocol", "processed_dataset")),
            "data_metadata": dict(normalized.metadata or {}),
            "numericizer": str(getattr(self.numericizer, "name", type(self.numericizer).__name__)),
            "linear_family": family_payload,
            "linear_family_signature": family_signature,
            "resume": {
                "enabled": bool(mode in {"resume", "warm_start", "incremental"} and parent_payload is not None),
                "mode": str(mode),
                "strategy": "closed_form_refit",
                "from": parent_source,
                "parent_kind": parent_kind,
            },
            "training_init": {
                "mode": str(mode),
                "parent_source": parent_source,
                "parent_kind": parent_kind,
            },
        }

        artifact = LinearSurrogateArtifact(
            artifact_id=str(self.config.artifact_id),
            coef=np.asarray(coef, dtype=float),
            intercept=np.asarray(intercept, dtype=float),
            x_mean=np.mean(Xb, axis=0),
            x_std=np.std(Xb, axis=0) + 1e-8,
            residual_std=np.asarray(residual_std, dtype=float),
            feature_names=feature_names,
            target_names=target_names,
            pipeline_name=str(getattr(self.pipeline, "name", "identity")),
            pipeline_state=dict(getattr(self.pipeline, "state_dict")()),
            ood_z_threshold=float(self.config.ood_z_threshold),
            metadata=metadata,
        )
        if training_signature is None:
            return artifact, None

        signature_obj = coerce_training_signature(training_signature)
        payload = {
            "schema_version": 1,
            "trainer_name": str(self.name),
            "coef": np.asarray(coef, dtype=float),
            "intercept": np.asarray(intercept, dtype=float),
            "x_mean": np.mean(Xb, axis=0),
            "x_std": np.std(Xb, axis=0) + 1e-8,
            "residual_std": np.asarray(residual_std, dtype=float),
            "feature_names": tuple(str(v) for v in feature_names),
            "target_names": tuple(str(v) for v in target_names),
            "l2": float(l2_value),
            "l2_effective": float(l2_eff),
            "linear_family": family_payload,
            "linear_family_signature": family_signature,
            "training_signature": signature_obj.as_dict(),
        }
        trainer_state = TrainerState(
            trainer_name=str(self.name),
            payload=payload,
            schema_signature=signature_obj.schema_signature,
            feature_signature=signature_obj.feature_signature,
            target_signature=signature_obj.target_signature,
            objective_signature=signature_obj.objective_signature,
            pipeline_signature=signature_obj.pipeline_signature,
            numericizer_signature=signature_obj.numericizer_signature,
            regime_signature=signature_obj.regime_signature,
            symbolic_family_signature=signature_obj.symbolic_family_signature,
            metadata={
                "resume_source": parent_source if mode == "resume" else None,
                "training_signature": signature_obj.as_dict(),
                "continuation_strategy": "closed_form_refit",
                "linear_family_signature": family_signature,
            },
        )
        return artifact, trainer_state

    def fit(self, data: ProcessedDataset | SampleDataset) -> LinearSurrogateArtifact:
        artifact, _ = self._fit_internal(data)
        return artifact

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
        artifact, trainer_state = self._fit_internal(
            task.data,
            init=init_eff,
            training_signature=task_signature.as_dict(),
        )
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
            trainer_state=trainer_state,
            report={
                "training_mode": str(init_eff.mode),
                "trainer_capabilities": caps.as_dict(),
                "task_signature": task_signature.as_dict(),
                "compatibility": verdict.metadata,
                "compatibility_warnings": list(verdict.warnings),
            },
            lineage=lineage,
        )
