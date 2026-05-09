from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from bias import BaseTrainingBias, FitContext, NoOpBias
from numericizer import BaseNumericizer, DefaultNumericizer, ModalityEncoder, TargetCodec
from pipeline import BasePipeline, IdentityPipeline

from core.common.base_trainer import BaseSurrogateTrainer
from core.common.contracts import ProcessedDataset, SampleDataset
from core.execution import ExecutionResourceRequest, detect_local_execution_offer
from core.artifacts.xgboost_artifact import XGBoostMultiOutputModelWrapper, XGBoostSurrogateArtifact
from core.mechanisms.runtime import (
    MechanismRuntimeStack,
    MechanismRuntimeState,
    RuntimeMechanismSpec,
    build_runtime_mechanisms,
)
from core.tree_boosting.trainer_family import (
    TreeBoostingTrainerFamilySpec,
    build_xgboost_family_spec,
    coerce_tree_boosting_family_spec,
)
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

try:
    from sklearn.multioutput import MultiOutputRegressor
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "xgboost and scikit-learn are required for XGBoostSurrogateTrainer. Install them before using xgboost."
    ) from exc


@dataclass(frozen=True)
class XGBoostTrainerConfig:
    artifact_id: str = "xgboost_surrogate_v1"
    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_weight: float = 1.0
    gamma: float = 0.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    objective: str = "reg:squarederror"
    tree_method: str = "hist"
    n_jobs: int = -1
    random_seed: int = 42
    verbosity: int = 0
    resume_training_from: str | None = None
    ood_z_threshold: float = 4.0
    family_spec: TreeBoostingTrainerFamilySpec | Mapping[str, Any] | None = None
    mechanisms: Sequence[RuntimeMechanismSpec | Mapping[str, Any] | str] = field(
        default_factory=lambda: ({"key": "aggregation.ensemble_summary", "params": {}},)
    )


class XGBoostSurrogateTrainer(BaseSurrogateTrainer):
    """XGBoost trainer with the same layering contract as other trainers."""

    name = "xgboost"

    def __init__(
        self,
        config: XGBoostTrainerConfig | None = None,
        *,
        pipeline: BasePipeline | None = None,
        biases: Sequence[BaseTrainingBias] | None = None,
        numericizer: BaseNumericizer | None = None,
        modality_encoders: Mapping[str, ModalityEncoder] | None = None,
        target_codecs: Mapping[str, TargetCodec] | None = None,
        target_codec: str | None = None,
        categorical_unknown: str = "error",
    ) -> None:
        self.config = config or XGBoostTrainerConfig()
        self.tree_boosting_family_spec = self._resolve_family_spec(self.config)
        self.pipeline = pipeline or IdentityPipeline()
        self.biases = list(biases) if biases else [NoOpBias()]
        self.mechanism_stack: MechanismRuntimeStack = build_runtime_mechanisms(self.config.mechanisms)
        try:
            setattr(self, "tree_boosting_family_metadata", self.tree_boosting_family_spec.description_dict())
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

    @staticmethod
    def _resolve_family_spec(config: XGBoostTrainerConfig | None) -> TreeBoostingTrainerFamilySpec:
        cfg = config or XGBoostTrainerConfig()
        raw = getattr(cfg, "family_spec", None)
        if raw is not None:
            return coerce_tree_boosting_family_spec(raw, trainer_key="xgboost")
        return build_xgboost_family_spec(
            trainer_key="xgboost",
            n_estimators=int(cfg.n_estimators),
            max_depth=int(cfg.max_depth),
            learning_rate=float(cfg.learning_rate),
            subsample=float(cfg.subsample),
            colsample_bytree=float(cfg.colsample_bytree),
            min_child_weight=float(cfg.min_child_weight),
            gamma=float(cfg.gamma),
            reg_lambda=float(cfg.reg_lambda),
            reg_alpha=float(cfg.reg_alpha),
            objective=str(cfg.objective),
            tree_method=str(cfg.tree_method),
            n_jobs=int(cfg.n_jobs),
            random_seed=int(cfg.random_seed),
            verbosity=int(cfg.verbosity),
        )

    def _current_family_payload(self) -> dict[str, object]:
        return self.tree_boosting_family_spec.description_dict()

    def _current_family_signature(self) -> str | None:
        return self.tree_boosting_family_spec.family_signature()

    def _assert_parent_family_compatible(self, payload: Mapping[str, object] | None) -> None:
        if payload is None:
            return
        current_sig = self._current_family_signature()
        parent_sig = payload.get("tree_boosting_family_signature")
        if current_sig is None or parent_sig is None:
            return
        if str(current_sig) != str(parent_sig):
            raise ValueError(
                f"{self.name} continuation rejected because tree boosting family components changed "
                f"(current={current_sig}, parent={parent_sig})"
            )

    def _validate_family_surface(self) -> None:
        family = self.tree_boosting_family_spec
        if str(family.backend.backend) != "xgboost":
            raise ValueError("xgboost trainer requires tree_boosting_family backend.backend='xgboost'")
        if str(family.backend.booster) != "gbtree":
            raise ValueError("xgboost trainer currently supports only tree_boosting_family backend.booster='gbtree'")
        if str(family.task_head.task) != "point" or str(family.task_head.objective_family) != "regression":
            raise ValueError("xgboost trainer currently supports only point/regression task_head")

    def capabilities(self) -> dict[str, object]:
        family = self.tree_boosting_family_spec
        return {
            "supports_fresh": True,
            "supports_resume": bool(family.backend.supports_resume),
            "supports_warm_start": bool(family.backend.supports_warm_start),
            "supports_incremental": bool(family.backend.supports_incremental),
            "supports_recalibration": False,
            "name": self.name,
            "model_family": "tree_boosting",
            "backend": str(family.backend.backend),
            "nonlinear": True,
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "sample_weight": True,
                "target_codec": True,
                "runtime_mechanism_interface": True,
                "tree_boosting_family_spec": True,
            },
            "artifacts": {
                "type": "XGBoostSurrogateArtifact",
                "uncertainty": str(family.task_head.uncertainty_mode),
                "ood_validity": True,
            },
            "runtime": {
                "requires": ["xgboost", "sklearn"],
                "tree_method": str(family.boosting.tree_method),
                "trainer_state": bool(family.backend.trainer_state_enabled),
                "resume_from_trainer_state": True,
                "continuation_via": str(family.backend.continuation_mode),
                "active_runtime_mechanisms": self.mechanism_stack.summaries(),
            },
            "tree_boosting_family": family.description_dict(),
        }

    def execution_resource_request(self) -> ExecutionResourceRequest:
        requested_jobs = int(self.tree_boosting_family_spec.execution.n_jobs)
        if requested_jobs <= 0:
            requested_jobs = int(detect_local_execution_offer().threads)
        return ExecutionResourceRequest(
            threads=max(1, int(requested_jobs)),
            backend="serial",
            label=str(self.name),
            metadata={
                "backend_family": "xgboost",
                "n_jobs": int(self.tree_boosting_family_spec.execution.n_jobs),
                "tree_method": str(self.tree_boosting_family_spec.boosting.tree_method),
            },
        )

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
                "num_boosted_rounds": payload.get("num_boosted_rounds"),
                "tree_boosting_family_signature": payload.get("tree_boosting_family_signature"),
                "input_feature_indices": payload.get("input_feature_indices"),
            },
        )

    @staticmethod
    def _payload_from_artifact(artifact: object) -> dict[str, object] | None:
        if not isinstance(artifact, XGBoostSurrogateArtifact):
            return None
        metadata = dict(getattr(artifact, "metadata", {}) or {})
        return {
            "schema_version": 1,
            "trainer_name": "xgboost",
            "model": artifact.model,
            "x_mean": np.asarray(artifact.x_mean, dtype=float),
            "x_std": np.asarray(artifact.x_std, dtype=float),
            "residual_std": np.asarray(artifact.residual_std, dtype=float),
            "feature_names": tuple(str(v) for v in artifact.feature_names),
            "target_names": tuple(str(v) for v in artifact.target_names),
            "input_feature_indices": (
                None
                if getattr(artifact, "input_feature_indices", None) is None
                else tuple(int(v) for v in tuple(artifact.input_feature_indices))
            ),
            "num_boosted_rounds": metadata.get("model", {}).get("num_boosted_rounds"),
            "tree_boosting_family": metadata.get("tree_boosting_family"),
            "tree_boosting_family_signature": metadata.get("tree_boosting_family_signature"),
            "training_signature": metadata.get("training_signature"),
            "runtime_mechanisms": metadata.get("runtime_mechanisms"),
        }

    def _make_base_model(self) -> XGBRegressor:
        family = self.tree_boosting_family_spec
        return XGBRegressor(
            n_estimators=int(family.boosting.n_estimators),
            max_depth=int(family.regularization.max_depth),
            learning_rate=float(family.boosting.learning_rate),
            subsample=float(family.sampling.subsample),
            colsample_bytree=float(family.sampling.colsample_bytree),
            min_child_weight=float(family.regularization.min_child_weight),
            gamma=float(family.regularization.gamma),
            reg_lambda=float(family.regularization.reg_lambda),
            reg_alpha=float(family.regularization.reg_alpha),
            objective=str(family.boosting.objective),
            tree_method=str(family.boosting.tree_method),
            n_jobs=int(family.execution.n_jobs),
            random_state=int(family.execution.random_seed),
            verbosity=int(family.boosting.verbosity),
            booster=str(family.backend.booster),
        )

    @staticmethod
    def _num_boosted_rounds(model: Any) -> int | list[int] | None:
        if model is None:
            return None
        if isinstance(model, XGBoostMultiOutputModelWrapper):
            return [int(est.get_booster().num_boosted_rounds()) for est in model.estimators]
        if isinstance(model, MultiOutputRegressor):
            return [int(est.get_booster().num_boosted_rounds()) for est in getattr(model, "estimators_", [])]
        if hasattr(model, "get_booster"):
            return int(model.get_booster().num_boosted_rounds())
        return None

    @staticmethod
    def _extract_parent_model(payload: Mapping[str, object] | None) -> Any | None:
        if payload is None:
            return None
        return payload.get("model")

    @staticmethod
    def _input_feature_dim(payload: Mapping[str, object] | None, parent_model: Any | None) -> int | None:
        if payload is not None:
            input_feature_indices = payload.get("input_feature_indices")
            if input_feature_indices is not None:
                return int(len(tuple(input_feature_indices)))
        if parent_model is None:
            return None
        value = getattr(parent_model, "n_features_in_", None)
        if value is None:
            return None
        return int(value)

    def _fit_single_output(
        self,
        Xb: np.ndarray,
        y: np.ndarray,
        *,
        fit_kwargs: Mapping[str, object],
        parent_model: Any | None,
    ) -> tuple[XGBRegressor, np.ndarray]:
        model = self._make_base_model()
        extra_kwargs = dict(fit_kwargs)
        if parent_model is not None:
            extra_kwargs["xgb_model"] = parent_model
        model.fit(Xb, y, **extra_kwargs)
        pred = np.asarray(model.predict(Xb), dtype=float).reshape(-1, 1)
        return model, pred

    def _fit_multi_output(
        self,
        Xb: np.ndarray,
        Yb: np.ndarray,
        *,
        fit_kwargs: Mapping[str, object],
        parent_model: Any | None,
    ) -> tuple[XGBoostMultiOutputModelWrapper, np.ndarray]:
        parent_estimators: tuple[Any, ...] = tuple()
        if isinstance(parent_model, XGBoostMultiOutputModelWrapper):
            parent_estimators = tuple(parent_model.estimators)
        elif isinstance(parent_model, MultiOutputRegressor):
            parent_estimators = tuple(getattr(parent_model, "estimators_", []) or ())

        preds: list[np.ndarray] = []
        estimators: list[XGBRegressor] = []
        for idx in range(int(Yb.shape[1])):
            est = self._make_base_model()
            extra_kwargs = dict(fit_kwargs)
            if idx < len(parent_estimators):
                extra_kwargs["xgb_model"] = parent_estimators[idx]
            est.fit(Xb, Yb[:, idx], **extra_kwargs)
            estimators.append(est)
            preds.append(np.asarray(est.predict(Xb), dtype=float).reshape(-1, 1))
        return XGBoostMultiOutputModelWrapper(tuple(estimators)), np.asarray(np.concatenate(preds, axis=1), dtype=float)

    def _fit_internal(
        self,
        data: ProcessedDataset | SampleDataset,
        *,
        init: TrainingInit | None = None,
        training_signature: Mapping[str, object] | None = None,
    ) -> tuple[XGBoostSurrogateArtifact, TrainerState | None]:
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

        if parent_payload is not None and self.config.resume_training_from:
            raise ValueError(
                "resume source is ambiguous: both training_init parent payload and config.resume_training_from were provided"
            )
        if parent_payload is None and self.config.resume_training_from:
            resume_path = Path(str(self.config.resume_training_from)).resolve()
            parent_payload = load_pickled_trainer_state_file(resume_path)
            parent_source = str(resume_path)
            parent_kind = "trainer_state_file"
            if mode == "fresh":
                mode = "resume"

        if mode == "resume" and parent_payload is None:
            raise ValueError("resume mode requires parent trainer_state payload")

        self._assert_parent_family_compatible(parent_payload)

        normalized = self.numericizer.to_processed(data)

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

        if normalized.feature_names is not None and len(tuple(normalized.feature_names)) == Xb.shape[1]:
            feature_names = tuple(str(v) for v in normalized.feature_names)
        else:
            feature_names = tuple(f"x{i}" for i in range(Xb.shape[1]))
        target_names = (
            tuple(str(v) for v in normalized.target_names)
            if normalized.target_names is not None
            else tuple(f"y{i}" for i in range(Yb.shape[1]))
        )

        signal_parent_model = self._extract_parent_model(parent_payload)
        continuation_parent_model = signal_parent_model if mode in {"resume", "warm_start", "incremental"} else None
        runtime_state = MechanismRuntimeState(
            trainer_key=str(self.name),
            family_key="tree_boosting",
            X=np.asarray(Xb, dtype=float),
            Y=np.asarray(Yb, dtype=float),
            feature_names=feature_names,
            target_names=target_names,
            sample_weight=(
                None
                if context.sample_weight is None
                else np.asarray(context.sample_weight, dtype=float).reshape(-1)
            ),
            parent_model=signal_parent_model,
            parent_payload=parent_payload,
            metadata={
                "mode": str(mode),
                "active_components": self.mechanism_stack.summaries(),
            },
        )
        self.mechanism_stack.run_pre_fit(runtime_state)

        X_fit = np.asarray(runtime_state.X, dtype=float)
        Y_fit = np.asarray(runtime_state.Y, dtype=float)
        if Y_fit.ndim == 1:
            Y_fit = Y_fit.reshape(-1, 1)
        n, d = X_fit.shape
        m = Y_fit.shape[1]
        if n <= 0 or d <= 0:
            raise ValueError("runtime mechanisms produced empty training data")
        if runtime_state.sample_weight is not None and np.asarray(runtime_state.sample_weight).reshape(-1).shape[0] != n:
            raise ValueError("sample_weight length mismatch after runtime mechanisms")
        if continuation_parent_model is not None:
            expected_d = self._input_feature_dim(parent_payload, continuation_parent_model)
            if expected_d is not None and expected_d != d:
                raise ValueError(
                    f"{self.name} continuation rejected because runtime feature view changed "
                    f"(current_dim={d}, parent_dim={expected_d})"
                )

        fit_kwargs: dict[str, object] = {}
        if runtime_state.sample_weight is not None:
            fit_kwargs["sample_weight"] = np.maximum(np.asarray(runtime_state.sample_weight, dtype=float).reshape(-1), 0.0)

        rounds_before = self._num_boosted_rounds(continuation_parent_model)
        if m == 1:
            model, pred = self._fit_single_output(
                X_fit,
                Y_fit[:, 0],
                fit_kwargs=fit_kwargs,
                parent_model=continuation_parent_model,
            )
        else:
            model, pred = self._fit_multi_output(
                X_fit,
                Y_fit,
                fit_kwargs=fit_kwargs,
                parent_model=continuation_parent_model,
            )

        residual = Y_fit - pred
        residual_std = np.std(residual, axis=0, ddof=1) + 1e-8

        feature_names = tuple(str(v) for v in runtime_state.feature_names) if runtime_state.feature_names else feature_names
        target_names = tuple(str(v) for v in runtime_state.target_names) if runtime_state.target_names else target_names
        rounds_after = self._num_boosted_rounds(model)
        input_feature_indices = (
            None if runtime_state.feature_indices is None else tuple(int(v) for v in np.asarray(runtime_state.feature_indices, dtype=int))
        )
        family_payload = self._current_family_payload()
        family_signature = self._current_family_signature()

        metadata = {
            "trainer": "XGBoostSurrogateTrainer",
            "n_train": int(n),
            "feature_dim": int(d),
            "target_dim": int(m),
            "pipeline": str(getattr(self.pipeline, "name", "identity")),
            "biases": [str(getattr(b, "name", type(b).__name__)) for b in self.biases],
            "fit_context": dict(context.metadata),
            "data_protocol": str((normalized.metadata or {}).get("input_protocol", "processed_dataset")),
            "data_metadata": dict(normalized.metadata or {}),
            "numericizer": str(getattr(self.numericizer, "name", type(self.numericizer).__name__)),
            "runtime_mechanisms": {
                "active_components": self.mechanism_stack.summaries(),
            },
            "tree_boosting_family": family_payload,
            "tree_boosting_family_signature": family_signature,
            "model": {
                "backend": str(self.tree_boosting_family_spec.backend.backend),
                "booster": str(self.tree_boosting_family_spec.backend.booster),
                "n_estimators": int(self.tree_boosting_family_spec.boosting.n_estimators),
                "max_depth": int(self.tree_boosting_family_spec.regularization.max_depth),
                "learning_rate": float(self.tree_boosting_family_spec.boosting.learning_rate),
                "subsample": float(self.tree_boosting_family_spec.sampling.subsample),
                "colsample_bytree": float(self.tree_boosting_family_spec.sampling.colsample_bytree),
                "min_child_weight": float(self.tree_boosting_family_spec.regularization.min_child_weight),
                "gamma": float(self.tree_boosting_family_spec.regularization.gamma),
                "reg_lambda": float(self.tree_boosting_family_spec.regularization.reg_lambda),
                "reg_alpha": float(self.tree_boosting_family_spec.regularization.reg_alpha),
                "objective": str(self.tree_boosting_family_spec.boosting.objective),
                "tree_method": str(self.tree_boosting_family_spec.boosting.tree_method),
                "num_boosted_rounds": rounds_after,
            },
            "resume": {
                "enabled": bool(mode in {"resume", "warm_start", "incremental"} and parent_payload is not None),
                "mode": str(mode),
                "strategy": str(self.tree_boosting_family_spec.backend.continuation_mode),
                "from": parent_source,
                "parent_kind": parent_kind,
                "num_boosted_rounds_before": rounds_before,
                "num_boosted_rounds_after": rounds_after,
            },
            "training_init": {
                "mode": str(mode),
                "parent_source": parent_source,
                "parent_kind": parent_kind,
            },
        }
        self.mechanism_stack.run_post_fit(runtime_state, model=model, artifact_metadata=metadata)

        artifact = XGBoostSurrogateArtifact(
            artifact_id=str(self.config.artifact_id),
            model=model,
            x_mean=np.mean(X_fit, axis=0),
            x_std=np.std(X_fit, axis=0) + 1e-8,
            residual_std=np.asarray(residual_std, dtype=float),
            feature_names=feature_names,
            target_names=target_names,
            pipeline_name=str(getattr(self.pipeline, "name", "identity")),
            pipeline_state=dict(getattr(self.pipeline, "state_dict")()),
            ood_z_threshold=float(self.config.ood_z_threshold),
            input_feature_indices=input_feature_indices,
            metadata=metadata,
        )
        if training_signature is None:
            return artifact, None

        signature_obj = coerce_training_signature(training_signature)
        payload = {
            "schema_version": 1,
            "trainer_name": str(self.name),
            "model": model,
            "x_mean": np.mean(X_fit, axis=0),
            "x_std": np.std(X_fit, axis=0) + 1e-8,
            "residual_std": np.asarray(residual_std, dtype=float),
            "feature_names": tuple(str(v) for v in feature_names),
            "target_names": tuple(str(v) for v in target_names),
            "input_feature_indices": input_feature_indices,
            "num_boosted_rounds": rounds_after,
            "tree_boosting_family": family_payload,
            "tree_boosting_family_signature": family_signature,
            "training_signature": signature_obj.as_dict(),
            "runtime_mechanisms": metadata.get("runtime_mechanisms"),
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
                "continuation_strategy": str(self.tree_boosting_family_spec.backend.continuation_mode),
                "num_boosted_rounds": rounds_after,
                "tree_boosting_family_signature": family_signature,
                "input_feature_indices": input_feature_indices,
            },
        )
        return artifact, trainer_state

    def fit(self, data: ProcessedDataset | SampleDataset) -> XGBoostSurrogateArtifact:
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
                "runtime_mechanisms": self.mechanism_stack.summaries(),
            },
            lineage=lineage,
        )
