from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from bias import BaseTrainingBias, FitContext, NoOpBias
from numericizer import BaseNumericizer, DefaultNumericizer, ModalityEncoder, TargetCodec
from pipeline import BasePipeline, IdentityPipeline

from core.artifacts.tree_ensemble_artifact import TreeEnsembleSurrogateArtifact
from core.common.base_trainer import BaseSurrogateTrainer
from core.common.contracts import ProcessedDataset, SampleDataset
from core.execution import ExecutionResourceRequest, detect_local_execution_offer
from core.mechanisms.runtime import (
    MechanismRuntimeStack,
    MechanismRuntimeState,
    RuntimeMechanismSpec,
    build_runtime_mechanisms,
)
from core.tree.trainer_family import TreeTrainerFamilySpec, coerce_tree_family_spec
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
    from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
except Exception as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "scikit-learn is required for tree ensemble trainers. Install scikit-learn before using tree ensemble presets."
    ) from exc


@dataclass(frozen=True)
class TreeEnsembleTrainerConfig:
    artifact_id: str = "tree_ensemble_surrogate_v1"
    family_spec: TreeTrainerFamilySpec = field(default_factory=TreeTrainerFamilySpec)
    resume_training_from: str | None = None
    ood_z_threshold: float = 4.0
    mechanisms: Sequence[RuntimeMechanismSpec | Mapping[str, Any] | str] = field(
        default_factory=lambda: ({"key": "aggregation.ensemble_summary", "params": {}},)
    )


class SklearnTreeEnsembleSurrogateTrainer(BaseSurrogateTrainer):
    name = "tree_ensemble"

    def __init__(
        self,
        config: TreeEnsembleTrainerConfig | None = None,
        *,
        pipeline: BasePipeline | None = None,
        biases: Sequence[BaseTrainingBias] | None = None,
        numericizer: BaseNumericizer | None = None,
        modality_encoders: Mapping[str, ModalityEncoder] | None = None,
        target_codecs: Mapping[str, TargetCodec] | None = None,
        target_codec: str | None = None,
        categorical_unknown: str = "error",
    ) -> None:
        self.config = config or TreeEnsembleTrainerConfig()
        self.tree_family_spec = coerce_tree_family_spec(self.config.family_spec, trainer_key=self.name)
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

    def capabilities(self) -> dict[str, object]:
        family = self.tree_family_spec
        ensemble = family.ensemble
        return {
            "supports_fresh": True,
            "supports_resume": bool(ensemble.supports_resume),
            "supports_warm_start": bool(ensemble.supports_warm_start),
            "supports_incremental": bool(ensemble.supports_incremental),
            "supports_recalibration": False,
            "name": self.name,
            "model_family": "tree_ensemble",
            "backend": str(ensemble.backend),
            "nonlinear": True,
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "sample_weight": True,
                "target_codec": True,
                "tree_family_spec": True,
                "runtime_mechanism_interface": True,
            },
            "artifacts": {
                "type": "TreeEnsembleSurrogateArtifact",
                "uncertainty": str(family.task_head.uncertainty_mode),
                "ood_validity": True,
            },
            "runtime": {
                "requires": "sklearn",
                "trainer_state": bool(ensemble.trainer_state_enabled),
                "resume_semantics": self._resume_semantics(),
                "warm_start_semantics": self._resume_semantics(),
                "incremental_semantics": self._resume_semantics(),
                "active_runtime_mechanisms": self.mechanism_stack.summaries(),
            },
            "tree_family": family.description_dict(),
        }

    def execution_resource_request(self) -> ExecutionResourceRequest:
        requested_jobs = int(self.tree_family_spec.ensemble.n_jobs)
        if requested_jobs <= 0:
            requested_jobs = int(detect_local_execution_offer().threads)
        return ExecutionResourceRequest(
            threads=max(1, int(requested_jobs)),
            backend="serial",
            label=str(self.name),
            metadata={
                "backend_family": "sklearn_tree_ensemble",
                "ensemble_kind": str(self.tree_family_spec.ensemble.ensemble_kind),
                "n_jobs": int(self.tree_family_spec.ensemble.n_jobs),
            },
        )

    @staticmethod
    def _clone_payload_cpu(value: object) -> object:
        return clone_pickled_trainer_payload(value)

    @staticmethod
    def _clone_state_cpu(state: Mapping[str, object]) -> dict[str, object]:
        return dict(clone_pickled_trainer_payload(dict(state)))

    @staticmethod
    def _clone_model_cpu(model: Any) -> Any:
        return pickle.loads(pickle.dumps(model))

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
                "trained_n_estimators": payload.get("trained_n_estimators"),
                "training_signature": signature.as_dict(),
                "tree_family_signature": payload.get("tree_family_signature"),
                "input_feature_indices": payload.get("input_feature_indices"),
            },
        )

    @classmethod
    def _payload_from_artifact(cls, artifact: object) -> dict[str, object] | None:
        if not isinstance(artifact, TreeEnsembleSurrogateArtifact):
            return None
        metadata = dict(getattr(artifact, "metadata", {}) or {})
        return {
            "schema_version": 1,
            "trainer_name": str(getattr(cls, "name", "tree_ensemble")),
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
            "trained_n_estimators": metadata.get("model", {}).get("trained_n_estimators"),
            "tree_family": metadata.get("tree_family"),
            "tree_family_signature": metadata.get("tree_family_signature"),
            "training_signature": metadata.get("training_signature"),
            "runtime_mechanisms": metadata.get("runtime_mechanisms"),
        }

    def _ensemble_kind(self) -> str:
        return str(self.tree_family_spec.ensemble.ensemble_kind).strip().lower()

    def _resume_semantics(self) -> str:
        if self._ensemble_kind() == "adaboost":
            return "fresh_only"
        return "sklearn_warm_start_append"

    def _validate_family_surface(self) -> None:
        family = self.tree_family_spec
        ensemble_kind = self._ensemble_kind()
        if str(family.task_head.task) != "point" or str(family.task_head.objective_family) != "regression":
            raise ValueError("tree ensemble trainers currently support only point/regression task_head")
        if family.sampling.class_weight is not None:
            raise ValueError("tree ensemble regression presets do not support class_weight yet")
        if ensemble_kind in {"random_forest", "extra_trees"}:
            if not bool(family.sampling.bootstrap) and family.sampling.max_samples is not None:
                raise ValueError(f"max_samples requires bootstrap=True for {ensemble_kind}")
            if bool(family.ensemble.oob_score) and not bool(family.sampling.bootstrap):
                raise ValueError(f"oob_score requires bootstrap=True for {ensemble_kind}")
        if ensemble_kind == "random_forest" and str(family.splitter.splitter) != "best":
            raise ValueError("random_forest currently supports only splitter='best'")
        if ensemble_kind == "extra_trees" and str(family.splitter.splitter) not in {"best", "random"}:
            raise ValueError("extra_trees currently supports splitter='best' or 'random'")
        if ensemble_kind == "bagging":
            if bool(family.ensemble.oob_score) and not bool(family.sampling.bootstrap):
                raise ValueError("bagging oob_score requires bootstrap=True")
        if ensemble_kind == "adaboost":
            if bool(family.ensemble.oob_score):
                raise ValueError("adaboost does not support oob_score")
            if family.sampling.max_samples not in {None, 1.0}:
                raise ValueError("adaboost internal max_samples is not supported; use runtime sampling mechanisms instead")
            if family.sampling.bootstrap_features:
                raise ValueError("adaboost does not support bootstrap_features")

    def _build_base_tree(self, *, include_max_features: bool) -> DecisionTreeRegressor:
        family = self.tree_family_spec
        return DecisionTreeRegressor(
            criterion=str(family.splitter.criterion),
            splitter=str(family.splitter.splitter),
            max_depth=family.regularization.max_depth,
            min_samples_split=family.regularization.min_samples_split,
            min_samples_leaf=family.regularization.min_samples_leaf,
            min_weight_fraction_leaf=float(family.regularization.min_weight_fraction_leaf),
            max_features=family.sampling.max_features if include_max_features else None,
            max_leaf_nodes=family.regularization.max_leaf_nodes,
            min_impurity_decrease=float(family.splitter.min_impurity_decrease),
            ccp_alpha=float(family.regularization.ccp_alpha),
            random_state=int(family.ensemble.random_seed),
        )

    def _build_model(self, *, n_estimators: int, warm_start: bool) -> Any:
        family = self.tree_family_spec
        ensemble_kind = self._ensemble_kind()
        if ensemble_kind == "random_forest":
            return RandomForestRegressor(
                n_estimators=max(1, int(n_estimators)),
                criterion=str(family.splitter.criterion),
                max_depth=family.regularization.max_depth,
                min_samples_split=family.regularization.min_samples_split,
                min_samples_leaf=family.regularization.min_samples_leaf,
                min_weight_fraction_leaf=float(family.regularization.min_weight_fraction_leaf),
                max_features=family.sampling.max_features,
                max_leaf_nodes=family.regularization.max_leaf_nodes,
                min_impurity_decrease=float(family.splitter.min_impurity_decrease),
                bootstrap=bool(family.sampling.bootstrap),
                oob_score=bool(family.ensemble.oob_score),
                n_jobs=int(family.ensemble.n_jobs),
                random_state=int(family.ensemble.random_seed),
                warm_start=bool(warm_start),
                ccp_alpha=float(family.regularization.ccp_alpha),
                max_samples=family.sampling.max_samples,
            )
        if ensemble_kind == "extra_trees":
            return ExtraTreesRegressor(
                n_estimators=max(1, int(n_estimators)),
                criterion=str(family.splitter.criterion),
                max_depth=family.regularization.max_depth,
                min_samples_split=family.regularization.min_samples_split,
                min_samples_leaf=family.regularization.min_samples_leaf,
                min_weight_fraction_leaf=float(family.regularization.min_weight_fraction_leaf),
                max_features=family.sampling.max_features,
                max_leaf_nodes=family.regularization.max_leaf_nodes,
                min_impurity_decrease=float(family.splitter.min_impurity_decrease),
                bootstrap=bool(family.sampling.bootstrap),
                oob_score=bool(family.ensemble.oob_score),
                n_jobs=int(family.ensemble.n_jobs),
                random_state=int(family.ensemble.random_seed),
                warm_start=bool(warm_start),
                ccp_alpha=float(family.regularization.ccp_alpha),
                max_samples=family.sampling.max_samples,
            )
        if ensemble_kind == "bagging":
            return BaggingRegressor(
                estimator=self._build_base_tree(include_max_features=False),
                n_estimators=max(1, int(n_estimators)),
                max_samples=1.0 if family.sampling.max_samples is None else family.sampling.max_samples,
                max_features=1.0 if family.sampling.max_features is None else family.sampling.max_features,
                bootstrap=bool(family.sampling.bootstrap),
                bootstrap_features=bool(family.sampling.bootstrap_features),
                oob_score=bool(family.ensemble.oob_score),
                warm_start=bool(warm_start),
                n_jobs=int(family.ensemble.n_jobs),
                random_state=int(family.ensemble.random_seed),
            )
        if ensemble_kind == "adaboost":
            return AdaBoostRegressor(
                estimator=self._build_base_tree(include_max_features=True),
                n_estimators=max(1, int(n_estimators)),
                learning_rate=float(family.ensemble.learning_rate),
                loss=str(family.ensemble.loss),
                random_state=int(family.ensemble.random_seed),
            )
        raise ValueError(f"unsupported tree ensemble kind: {ensemble_kind}")

    @staticmethod
    def _extract_parent_model(payload: Mapping[str, object] | None) -> Any | None:
        if payload is None:
            return None
        return payload.get("model")

    def _current_family_signature(self) -> str | None:
        return self.tree_family_spec.family_signature()

    def _assert_parent_family_compatible(self, payload: Mapping[str, object] | None) -> None:
        if payload is None:
            return
        current_sig = self._current_family_signature()
        parent_sig = payload.get("tree_family_signature")
        if parent_sig is None or current_sig is None:
            return
        if str(parent_sig) != str(current_sig):
            raise ValueError(
                f"{self.name} continuation rejected because tree family components changed "
                f"(current={current_sig}, parent={parent_sig})"
            )

    def _continuation_supported(self) -> bool:
        return self._ensemble_kind() in {"random_forest", "extra_trees", "bagging"}

    def _fit_internal(
        self,
        data: ProcessedDataset | SampleDataset,
        *,
        init: TrainingInit | None = None,
        training_signature: Mapping[str, object] | None = None,
    ) -> tuple[TreeEnsembleSurrogateArtifact, TrainerState | None]:
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
        continuation_parent_model = (
            signal_parent_model if mode in {"resume", "warm_start", "incremental"} else None
        )
        runtime_state = MechanismRuntimeState(
            trainer_key=str(self.name),
            family_key=self._ensemble_kind(),
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
        if mode in {"resume", "warm_start", "incremental"} and continuation_parent_model is not None and hasattr(continuation_parent_model, "n_features_in_"):
            expected_d = int(getattr(continuation_parent_model, "n_features_in_"))
            if expected_d != d:
                raise ValueError(
                    f"{self.name} continuation rejected because runtime feature view changed "
                    f"(current_dim={d}, parent_dim={expected_d})"
                )

        fit_kwargs: dict[str, object] = {}
        if runtime_state.sample_weight is not None:
            fit_kwargs["sample_weight"] = np.maximum(np.asarray(runtime_state.sample_weight, dtype=float).reshape(-1), 0.0)

        trees_before = (
            int(len(getattr(continuation_parent_model, "estimators_", ()) or ()))
            if continuation_parent_model is not None
            else 0
        )
        if continuation_parent_model is not None:
            if not self._continuation_supported():
                raise ValueError(f"{self.name} does not support resume/warm_start continuation")
            target_n_estimators = trees_before + int(self.tree_family_spec.ensemble.n_estimators)
            model = self._clone_model_cpu(continuation_parent_model)
            model.set_params(
                n_estimators=max(1, int(target_n_estimators)),
                warm_start=True,
            )
            if hasattr(model, "n_jobs"):
                model.set_params(n_jobs=int(self.tree_family_spec.ensemble.n_jobs))
            if hasattr(model, "oob_score"):
                model.set_params(oob_score=bool(self.tree_family_spec.ensemble.oob_score))
        else:
            target_n_estimators = int(self.tree_family_spec.ensemble.n_estimators)
            model = self._build_model(
                n_estimators=target_n_estimators,
                warm_start=bool(self.tree_family_spec.ensemble.warm_start_enabled),
            )

        fit_y = Y_fit[:, 0] if m == 1 else Y_fit
        model.fit(X_fit, fit_y, **fit_kwargs)
        pred = np.asarray(model.predict(X_fit), dtype=float)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)

        residual = Y_fit - pred
        residual_std = np.std(residual, axis=0, ddof=1) + 1e-8

        trees_after = int(len(getattr(model, "estimators_", ()) or ()))
        family_signature = self._current_family_signature()
        family_payload = self.tree_family_spec.as_dict()
        input_feature_indices = (
            None if runtime_state.feature_indices is None else tuple(int(v) for v in np.asarray(runtime_state.feature_indices, dtype=int))
        )
        oob_score_value = getattr(model, "oob_score_", None)

        metadata = {
            "trainer": str(type(self).__name__),
            "n_train": int(n),
            "feature_dim": int(d),
            "target_dim": int(m),
            "pipeline": str(getattr(self.pipeline, "name", "identity")),
            "biases": [str(getattr(b, "name", type(b).__name__)) for b in self.biases],
            "fit_context": dict(context.metadata),
            "data_protocol": str((normalized.metadata or {}).get("input_protocol", "processed_dataset")),
            "data_metadata": dict(normalized.metadata or {}),
            "numericizer": str(getattr(self.numericizer, "name", type(self.numericizer).__name__)),
            "tree_family": family_payload,
            "tree_family_signature": family_signature,
            "runtime_mechanisms": {
                "active_components": self.mechanism_stack.summaries(),
            },
            "model": {
                "ensemble_kind": str(self.tree_family_spec.ensemble.ensemble_kind),
                "trained_n_estimators": int(trees_after),
                "requested_n_estimators": int(self.tree_family_spec.ensemble.n_estimators),
                "n_estimators_before": int(trees_before),
                "n_estimators_target": int(target_n_estimators),
                "criterion": str(self.tree_family_spec.splitter.criterion),
                "max_depth": self.tree_family_spec.regularization.max_depth,
                "max_features": self.tree_family_spec.sampling.max_features,
                "bootstrap": bool(self.tree_family_spec.sampling.bootstrap),
                "bootstrap_features": bool(self.tree_family_spec.sampling.bootstrap_features),
                "max_samples": self.tree_family_spec.sampling.max_samples,
                "oob_score_enabled": bool(self.tree_family_spec.ensemble.oob_score),
                "oob_score": None if oob_score_value is None else float(oob_score_value),
                "learning_rate": float(self.tree_family_spec.ensemble.learning_rate),
                "loss": str(self.tree_family_spec.ensemble.loss),
            },
            "resume": {
                "enabled": bool(mode in {"resume", "warm_start", "incremental"} and parent_payload is not None),
                "mode": str(mode),
                "strategy": self._resume_semantics(),
                "from": parent_source,
                "parent_kind": parent_kind,
                "n_estimators_before": int(trees_before),
                "n_estimators_after": int(trees_after),
            },
            "training_init": {
                "mode": str(mode),
                "parent_source": parent_source,
                "parent_kind": parent_kind,
            },
        }
        self.mechanism_stack.run_post_fit(runtime_state, model=model, artifact_metadata=metadata)

        artifact = TreeEnsembleSurrogateArtifact(
            artifact_id=str(self.config.artifact_id),
            model=model,
            x_mean=np.mean(X_fit, axis=0),
            x_std=np.std(X_fit, axis=0) + 1e-8,
            residual_std=np.asarray(residual_std, dtype=float),
            feature_names=tuple(str(v) for v in runtime_state.feature_names) if runtime_state.feature_names else feature_names,
            target_names=tuple(str(v) for v in runtime_state.target_names) if runtime_state.target_names else target_names,
            pipeline_name=str(getattr(self.pipeline, "name", "identity")),
            pipeline_state=dict(getattr(self.pipeline, "state_dict")()),
            ood_z_threshold=float(self.config.ood_z_threshold),
            model_family="tree_ensemble",
            ensemble_kind=str(self.tree_family_spec.ensemble.ensemble_kind),
            uncertainty_mode=str(self.tree_family_spec.task_head.uncertainty_mode),
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
            "feature_names": tuple(str(v) for v in artifact.feature_names),
            "target_names": tuple(str(v) for v in artifact.target_names),
            "input_feature_indices": input_feature_indices,
            "trained_n_estimators": int(trees_after),
            "tree_family": family_payload,
            "tree_family_signature": family_signature,
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
                "trained_n_estimators": int(trees_after),
                "training_signature": signature_obj.as_dict(),
                "tree_family_signature": family_signature,
                "input_feature_indices": input_feature_indices,
                "continuation_strategy": self._resume_semantics(),
            },
        )
        return artifact, trainer_state

    def fit(self, data: ProcessedDataset | SampleDataset) -> TreeEnsembleSurrogateArtifact:
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


__all__ = [
    "SklearnTreeEnsembleSurrogateTrainer",
    "TreeEnsembleTrainerConfig",
]
