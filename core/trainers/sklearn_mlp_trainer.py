from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from bias import BaseTrainingBias, FitContext, NoOpBias
from numericizer import BaseNumericizer, DefaultNumericizer, ModalityEncoder, TargetCodec
from pipeline import BasePipeline, IdentityPipeline

from core.common.base_trainer import BaseSurrogateTrainer
from core.common.contracts import ProcessedDataset, SampleDataset
from core.artifacts.sklearn_mlp_artifact import SklearnMLPSurrogateArtifact
from core.mechanisms.runtime import (
    MechanismRuntimeStack,
    MechanismRuntimeState,
    RuntimeMechanismSpec,
    build_runtime_mechanisms,
)
from core.neural.trainer_family import (
    NeuralTrainerFamilySpec,
    build_sklearn_mlp_family_spec,
    coerce_neural_family_spec,
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
    from sklearn.neural_network import MLPRegressor
except Exception as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "scikit-learn is required for SklearnMLPSurrogateTrainer. Install sklearn before using sklearn_mlp."
    ) from exc


@dataclass(frozen=True)
class SklearnMLPTrainerConfig:
    artifact_id: str = "sklearn_mlp_surrogate_v1"
    hidden_layer_sizes: Sequence[int] = (128, 64)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 1e-4
    batch_size: int | str = "auto"
    learning_rate_init: float = 1e-3
    max_iter: int = 300
    tol: float = 1e-4
    n_iter_no_change: int = 20
    validation_fraction: float = 0.15
    early_stopping: bool = True
    random_seed: int = 42
    ood_z_threshold: float = 4.0
    verbose: bool = False
    mechanisms: Sequence[RuntimeMechanismSpec | Mapping[str, Any] | str] = field(
        default_factory=lambda: ({"key": "aggregation.ensemble_summary", "params": {}},)
    )


class SklearnMLPSurrogateTrainer(BaseSurrogateTrainer):
    """Sklearn MLP trainer with the same training-control contract as other trainers."""

    name = "sklearn_mlp"

    def __init__(
        self,
        config: SklearnMLPTrainerConfig | None = None,
        *,
        pipeline: BasePipeline | None = None,
        biases: Sequence[BaseTrainingBias] | None = None,
        numericizer: BaseNumericizer | None = None,
        modality_encoders: Mapping[str, ModalityEncoder] | None = None,
        target_codecs: Mapping[str, TargetCodec] | None = None,
        target_codec: str | None = None,
        categorical_unknown: str = "error",
    ) -> None:
        self.config = config or SklearnMLPTrainerConfig()
        self.pipeline = pipeline or IdentityPipeline()
        self.biases = list(biases) if biases else [NoOpBias()]
        self.mechanism_stack: MechanismRuntimeStack = build_runtime_mechanisms(self.config.mechanisms)

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

    def capabilities(self) -> dict[str, object]:
        family = self._current_family_spec()
        return {
            "supports_fresh": True,
            "supports_resume": False,
            "supports_warm_start": True,
            "supports_incremental": False,
            "supports_recalibration": False,
            "name": self.name,
            "model_family": "neural_network",
            "backend": "scikit-learn",
            "nonlinear": True,
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "sample_weight": False,
                "target_codec": True,
                "neural_family_spec": True,
                "runtime_mechanism_interface": True,
            },
            "artifacts": {
                "type": "SklearnMLPSurrogateArtifact",
                "uncertainty": "residual_std",
                "ood_validity": True,
            },
            "runtime": {
                "requires": "sklearn",
                "early_stopping": bool(self.config.early_stopping),
                "trainer_state": True,
                "save_load_trainer_state": True,
                "warm_start_via": "sklearn_estimator_reuse",
                "active_runtime_mechanisms": self.mechanism_stack.summaries(),
            },
            "neural_family": family.description_dict(),
        }

    @staticmethod
    def _clone_payload_cpu(value: object) -> object:
        return clone_pickled_trainer_payload(value)

    @staticmethod
    def _clone_model(value: Any) -> Any:
        return pickle.loads(pickle.dumps(value))

    @classmethod
    def _clone_state_cpu(cls, state: Mapping[str, object]) -> dict[str, object]:
        cloned = dict(clone_pickled_trainer_payload(dict(state)))
        if cloned.get("model") is not None:
            cloned["model"] = cls._clone_model(cloned["model"])
        return cloned

    def _current_family_spec(self) -> NeuralTrainerFamilySpec:
        family = getattr(self, "neural_family_spec", None)
        if isinstance(family, NeuralTrainerFamilySpec):
            return family
        family = coerce_neural_family_spec(
            build_sklearn_mlp_family_spec(
                trainer_key=str(self.name),
                hidden_layers=tuple(int(v) for v in tuple(self.config.hidden_layer_sizes)),
                activation=str(self.config.activation),
                solver=str(self.config.solver),
                alpha=float(self.config.alpha),
                learning_rate_init=float(self.config.learning_rate_init),
                max_iter=int(self.config.max_iter),
                tol=float(self.config.tol),
                n_iter_no_change=int(self.config.n_iter_no_change),
                validation_fraction=float(self.config.validation_fraction),
                early_stopping=bool(self.config.early_stopping),
                batch_size=self.config.batch_size,
                random_seed=int(self.config.random_seed),
                metadata={"preset_kind": "sklearn_backend"},
            ).as_dict(),
            trainer_key=str(self.name),
        )
        try:
            setattr(self, "neural_family_spec", family)
            setattr(self, "neural_family_metadata", family.description_dict())
        except Exception:
            pass
        return family

    def _current_family_payload(self) -> dict[str, object]:
        return self._current_family_spec().description_dict()

    def _current_family_signature(self) -> str | None:
        return self._current_family_spec().family_signature()

    def _assert_parent_family_compatible(self, payload: Mapping[str, object] | None) -> None:
        if payload is None:
            return
        current_sig = self._current_family_signature()
        parent_sig = payload.get("neural_family_signature")
        if current_sig is None or parent_sig is None:
            return
        if str(current_sig) != str(parent_sig):
            raise ValueError(
                f"{self.name} continuation rejected because neural family components changed "
                f"(current={current_sig}, parent={parent_sig})"
            )

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
                "neural_family_signature": payload.get("neural_family_signature"),
                "input_feature_indices": payload.get("input_feature_indices"),
            },
        )

    def _config_model_spec(self) -> dict[str, object]:
        return {
            "hidden_layer_sizes": tuple(int(h) for h in self.config.hidden_layer_sizes),
            "activation": str(self.config.activation),
            "solver": str(self.config.solver),
            "alpha": float(self.config.alpha),
            "batch_size": self.config.batch_size,
            "learning_rate_init": float(self.config.learning_rate_init),
            "max_iter": int(self.config.max_iter),
            "tol": float(self.config.tol),
            "n_iter_no_change": int(self.config.n_iter_no_change),
            "validation_fraction": float(self.config.validation_fraction),
            "early_stopping": bool(self.config.early_stopping),
            "random_seed": int(self.config.random_seed),
            "verbose": bool(self.config.verbose),
        }

    @staticmethod
    def _payload_model_spec(payload: Mapping[str, object] | None) -> dict[str, object] | None:
        if payload is None:
            return None
        model_spec = payload.get("model_spec")
        if isinstance(model_spec, Mapping):
            return {str(k): v for k, v in dict(model_spec).items()}

        legacy_model = payload.get("model")
        if legacy_model is not None and hasattr(legacy_model, "get_params"):
            raw = dict(legacy_model.get_params(deep=False))
            return {
                "hidden_layer_sizes": tuple(int(h) for h in tuple(raw.get("hidden_layer_sizes", tuple()))),
                "activation": str(raw.get("activation", "")),
                "solver": str(raw.get("solver", "")),
            }
        return None

    def _validate_parent_model_spec(self, payload: Mapping[str, object]) -> None:
        parent_spec = self._payload_model_spec(payload)
        if parent_spec is None:
            return

        current_spec = self._config_model_spec()
        critical_keys = ("hidden_layer_sizes", "activation", "solver")
        mismatches = [
            f"{key}: current={current_spec.get(key)} parent={parent_spec.get(key)}"
            for key in critical_keys
            if key in parent_spec and parent_spec.get(key) != current_spec.get(key)
        ]
        if mismatches:
            joined = ", ".join(mismatches)
            raise ValueError(
                "sklearn_mlp warm_start requires compatible parent model spec; mismatched fields: "
                f"{joined}"
            )

    @staticmethod
    def _payload_from_artifact(artifact: object) -> dict[str, object] | None:
        if not isinstance(artifact, SklearnMLPSurrogateArtifact):
            return None
        metadata = dict(getattr(artifact, "metadata", {}) or {})
        model_spec = metadata.get("model_spec")
        if not isinstance(model_spec, Mapping):
            raw_model = dict(metadata.get("model", {}) or {})
            model_spec = {
                "hidden_layer_sizes": tuple(int(h) for h in tuple(raw_model.get("hidden_layer_sizes", tuple()))),
                "activation": str(raw_model.get("activation", "")),
                "solver": str(raw_model.get("solver", "")),
                "alpha": float(raw_model.get("alpha", 0.0)),
                "learning_rate_init": float(raw_model.get("learning_rate_init", 0.0)),
                "max_iter": int(raw_model.get("max_iter", 0)),
            }
        return {
            "schema_version": 1,
            "trainer_name": "sklearn_mlp",
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
            "model_spec": dict(model_spec),
            "neural_family": metadata.get("neural_family"),
            "neural_family_signature": metadata.get("neural_family_signature"),
            "training_signature": metadata.get("training_signature"),
            "runtime_mechanisms": metadata.get("runtime_mechanisms"),
        }

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

    def _build_model(
        self,
        *,
        warm_start: bool,
        parent_payload: Mapping[str, object] | None = None,
    ) -> MLPRegressor:
        params = {
            "hidden_layer_sizes": tuple(int(h) for h in self.config.hidden_layer_sizes),
            "activation": str(self.config.activation),
            "solver": str(self.config.solver),
            "alpha": float(self.config.alpha),
            "batch_size": self.config.batch_size,
            "learning_rate_init": float(self.config.learning_rate_init),
            "max_iter": int(self.config.max_iter),
            "tol": float(self.config.tol),
            "n_iter_no_change": int(self.config.n_iter_no_change),
            "validation_fraction": float(self.config.validation_fraction),
            "early_stopping": bool(self.config.early_stopping),
            "random_state": int(self.config.random_seed),
            "verbose": bool(self.config.verbose),
            "warm_start": bool(warm_start),
        }
        if parent_payload is None:
            return MLPRegressor(**params)

        self._validate_parent_model_spec(parent_payload)
        parent_model = parent_payload.get("model")
        if parent_model is None:
            raise ValueError("sklearn_mlp warm_start requires parent model payload")
        model = self._clone_model(parent_model)
        model.set_params(**params)
        return model

    def _fit_internal(
        self,
        data: ProcessedDataset | SampleDataset,
        *,
        init: TrainingInit | None = None,
        training_signature: Mapping[str, object] | None = None,
    ) -> tuple[SklearnMLPSurrogateArtifact, TrainerState | None]:
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

        signal_parent_model = self._extract_parent_model(parent_payload)
        warm_parent_payload = parent_payload if mode == "warm_start" else None
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

        if normalized.feature_names is not None and len(tuple(normalized.feature_names)) == Xb.shape[1]:
            feature_names = tuple(str(v) for v in normalized.feature_names)
        else:
            feature_names = tuple(f"x{i}" for i in range(Xb.shape[1]))
        target_names = (
            tuple(str(v) for v in normalized.target_names)
            if normalized.target_names is not None
            else tuple(f"y{i}" for i in range(Yb.shape[1]))
        )

        runtime_state = MechanismRuntimeState(
            trainer_key=str(self.name),
            family_key="neural",
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
        feature_names = tuple(str(v) for v in runtime_state.feature_names) if runtime_state.feature_names else feature_names
        target_names = tuple(str(v) for v in runtime_state.target_names) if runtime_state.target_names else target_names
        input_feature_indices = (
            None if runtime_state.feature_indices is None else tuple(int(v) for v in np.asarray(runtime_state.feature_indices, dtype=int))
        )

        if warm_parent_payload is not None:
            self._assert_parent_family_compatible(warm_parent_payload)
            expected_d = self._input_feature_dim(warm_parent_payload, signal_parent_model)
            if expected_d is not None and expected_d != d:
                raise ValueError(
                    f"{self.name} warm_start rejected because runtime feature view changed "
                    f"(current_dim={d}, parent_dim={expected_d})"
                )

        if runtime_state.sample_weight is not None:
            sample_weight = np.asarray(runtime_state.sample_weight, dtype=float).reshape(-1)
            warning_msg = (
                "sample_weight from bias/context/runtime mechanisms is not supported by sklearn_mlp trainer and will be ignored"
            )
            warnings.warn(warning_msg, RuntimeWarning)
            warn_list = list(context.metadata.get("warnings", []))
            warn_list.append(warning_msg)
            context.metadata["warnings"] = warn_list
            context.metadata["sample_weight_ignored"] = True
            context.metadata["sample_weight_size"] = int(sample_weight.shape[0])
            runtime_state.sample_weight = None

        y_fit: np.ndarray
        if m == 1:
            y_fit = Y_fit[:, 0]
        else:
            y_fit = Y_fit

        parent_n_iter = None
        if warm_parent_payload is not None and warm_parent_payload.get("model") is not None:
            parent_n_iter = getattr(warm_parent_payload["model"], "n_iter_", None)

        model = self._build_model(
            warm_start=bool(warm_parent_payload is not None),
            parent_payload=warm_parent_payload,
        )
        model.fit(X_fit, y_fit)

        pred = np.asarray(model.predict(X_fit), dtype=float)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)

        residual = Y_fit - pred
        residual_std = np.std(residual, axis=0, ddof=1) + 1e-8

        model_spec = self._config_model_spec()
        family_payload = self._current_family_payload()
        family_signature = self._current_family_signature()
        loss_curve = [float(v) for v in list(getattr(model, "loss_curve_", []) or [])]
        validation_scores = [float(v) for v in list(getattr(model, "validation_scores_", []) or [])]
        n_iter_done = int(getattr(model, "n_iter_", 0) or 0)
        max_iter_cfg = int(self.config.max_iter)
        reached_max_iter = bool(n_iter_done >= max_iter_cfg)
        training_diagnostics = {
            "early_stopping": bool(self.config.early_stopping),
            "n_iter": n_iter_done,
            "max_iter": max_iter_cfg,
            "reached_max_iter": reached_max_iter,
            "stopped_by": "max_iter" if reached_max_iter else "early_stopping_or_tolerance",
            "loss": float(getattr(model, "loss_", float("nan"))),
            "best_loss": (
                None
                if getattr(model, "best_loss_", None) is None
                else float(getattr(model, "best_loss_"))
            ),
            "best_validation_score": (
                None
                if getattr(model, "best_validation_score_", None) is None
                else float(getattr(model, "best_validation_score_"))
            ),
            "loss_curve": loss_curve,
            "loss_curve_length": int(len(loss_curve)),
            "validation_scores": validation_scores,
            "validation_curve_length": int(len(validation_scores)),
            "tol": float(self.config.tol),
            "n_iter_no_change": int(self.config.n_iter_no_change),
            "validation_fraction": float(self.config.validation_fraction),
            "batch_size": self.config.batch_size,
            "learning_rate_init": float(self.config.learning_rate_init),
        }
        metadata = {
            "trainer": "SklearnMLPSurrogateTrainer",
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
            "neural_family": family_payload,
            "neural_family_signature": family_signature,
            "model_spec": dict(model_spec),
            "model": {
                "hidden_layer_sizes": [int(h) for h in self.config.hidden_layer_sizes],
                "activation": str(self.config.activation),
                "solver": str(self.config.solver),
                "alpha": float(self.config.alpha),
                "learning_rate_init": float(self.config.learning_rate_init),
                "max_iter": int(self.config.max_iter),
                "n_iter_": int(getattr(model, "n_iter_", 0)),
                "loss_": float(getattr(model, "loss_", float("nan"))),
            },
            "training_diagnostics": training_diagnostics,
            "resume": {
                "enabled": bool(mode == "warm_start" and warm_parent_payload is not None),
                "mode": str(mode),
                "strategy": "sklearn_estimator_reuse",
                "from": parent_source,
                "parent_kind": parent_kind,
                "n_iter_before": None if parent_n_iter is None else int(parent_n_iter),
                "n_iter_after": int(getattr(model, "n_iter_", 0)),
            },
            "training_init": {
                "mode": str(mode),
                "parent_source": parent_source,
                "parent_kind": parent_kind,
            },
        }
        self.mechanism_stack.run_post_fit(runtime_state, model=model, artifact_metadata=metadata)

        artifact = SklearnMLPSurrogateArtifact(
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
            "model": self._clone_model(model),
            "x_mean": np.mean(X_fit, axis=0),
            "x_std": np.std(X_fit, axis=0) + 1e-8,
            "residual_std": np.asarray(residual_std, dtype=float),
            "feature_names": tuple(str(v) for v in feature_names),
            "target_names": tuple(str(v) for v in target_names),
            "input_feature_indices": input_feature_indices,
            "model_spec": dict(model_spec),
            "neural_family": family_payload,
            "neural_family_signature": family_signature,
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
                "resume_source": parent_source if mode == "warm_start" else None,
                "input_feature_indices": input_feature_indices,
                "training_signature": signature_obj.as_dict(),
                "neural_family_signature": family_signature,
                "continuation_strategy": "sklearn_estimator_reuse",
            },
        )
        return artifact, trainer_state

    def fit(self, data: ProcessedDataset | SampleDataset) -> SklearnMLPSurrogateArtifact:
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
