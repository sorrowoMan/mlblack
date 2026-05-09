from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from bias import BaseTrainingBias, FitContext, NoOpBias
from numericizer import BaseNumericizer, DefaultNumericizer, ModalityEncoder, TargetCodec
from pipeline import BasePipeline, IdentityPipeline

from core.common.base_trainer import BaseSurrogateTrainer
from core.common.batch_stream import BatchStreamSpec, create_torch_batch_stream
from core.common.contracts import ProcessedDataset, SampleDataset
from core.common.hypothesis_space import TorchModuleHypothesisSpace
from core.common.loss_objective import create_regression_objective
from core.common.param_optimizer import OptimizerSpec, create_torch_optimizer
from core.artifacts.torch_artifact import TorchMLPSurrogateArtifact
from core.execution import ExecutionResourceRequest
from core.mechanisms.runtime import (
    MechanismRuntimeStack,
    MechanismRuntimeState,
    RuntimeMechanismSpec,
    build_runtime_mechanisms,
)
from core.neural.trainer_family import (
    NeuralTrainerFamilySpec,
    build_torch_mlp_family_spec,
    coerce_neural_family_spec,
)
from core.models.torch_model import TorchMLPRegressor
from core.common.trainer_shared import prepare_training_data, resolve_torch_device, set_torch_seed, split_train_val_indices
from training import (
    FitResult,
    TrainTask,
    TrainerState,
    TrainingInit,
    TrainingLineage,
    attach_signature_to_artifact,
    build_task_signature,
    coerce_trainer_capabilities,
    coerce_training_signature,
    require_training_setup,
)

try:
    import torch
except Exception as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "PyTorch is required for TorchMLPSurrogateTrainer. Install torch before using mlp_torch."
    ) from exc


@dataclass(frozen=True)
class TorchMLPTrainerConfig:
    artifact_id: str = "torch_mlp_surrogate_v1"
    hidden_dims: Sequence[int] = (128, 64)
    activation: str = "relu"
    dropout: float = 0.0
    epochs: int = 120
    batch_size: int = 64
    batch_shuffle: bool = True
    batch_drop_last: bool = False
    batch_num_workers: int = 0
    batch_pin_memory: bool = False
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # adamw | adam | sgd | rmsprop
    optimizer_params: Mapping[str, Any] = field(default_factory=dict)
    objective: str = "mse"
    val_ratio: float = 0.15
    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-6
    random_seed: int = 42
    device: str = "auto"  # auto | cpu | cuda | cuda:<index>
    checkpoint_dir: str | None = None
    checkpoint_every_epochs: int = 0
    resume_training_from: str | None = None
    ood_z_threshold: float = 4.0
    verbose: bool = False
    mechanisms: Sequence[RuntimeMechanismSpec | Mapping[str, Any] | str] = field(
        default_factory=lambda: ({"key": "aggregation.ensemble_summary", "params": {}},)
    )


class _TorchRuntimePredictor:
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
        dropout: float,
        model_state: Mapping[str, Any],
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.activation = str(activation)
        self.dropout = float(dropout)
        self.n_features_in_ = int(input_dim)
        self._model = TorchMLPRegressor(
            int(input_dim),
            int(output_dim),
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
        )
        state = {k: v.detach().cpu() if hasattr(v, "detach") else v for k, v in dict(model_state).items()}
        self._model.load_state_dict(state, strict=True)
        self._model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        xt = torch.as_tensor(x, dtype=torch.float32)
        with torch.no_grad():
            pred = self._model(xt).detach().cpu().numpy()
        return np.asarray(pred, dtype=float)

    def gradient_norm(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        y = np.asarray(Y, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("gradient_norm requires matching X/Y rows")

        norms: list[float] = []
        self._model.eval()
        with torch.enable_grad():
            for idx in range(int(x.shape[0])):
                xt = torch.as_tensor(x[idx : idx + 1], dtype=torch.float32)
                yt = torch.as_tensor(y[idx : idx + 1], dtype=torch.float32)
                self._model.zero_grad(set_to_none=True)
                pred = self._model(xt)
                loss = torch.mean((pred - yt) ** 2)
                loss.backward()
                sq_sum = 0.0
                for param in self._model.parameters():
                    if param.grad is None:
                        continue
                    sq_sum += float(torch.sum(param.grad.detach() ** 2).cpu())
                norms.append(float(np.sqrt(max(sq_sum, 0.0))))
        return np.asarray(norms, dtype=float)


class TorchMLPSurrogateTrainer(BaseSurrogateTrainer):
    """Torch MLP trainer with explicit Model/Loss/Optimizer/DataLoader components."""

    name = "mlp_torch"

    def __init__(
        self,
        config: TorchMLPTrainerConfig | None = None,
        *,
        pipeline: BasePipeline | None = None,
        biases: Sequence[BaseTrainingBias] | None = None,
        numericizer: BaseNumericizer | None = None,
        modality_encoders: Mapping[str, ModalityEncoder] | None = None,
        target_codecs: Mapping[str, TargetCodec] | None = None,
        target_codec: str | None = None,
        categorical_unknown: str = "error",
    ) -> None:
        self.config = config or TorchMLPTrainerConfig()
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
        family = self._current_family_spec()
        return {
            "supports_fresh": True,
            "supports_resume": True,
            "supports_warm_start": False,
            "supports_incremental": False,
            "supports_recalibration": False,
            "name": self.name,
            "model_family": "neural_network",
            "backend": "pytorch",
            "nonlinear": True,
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "sample_weight": True,
                "target_codec": True,
                "neural_family_spec": True,
                "runtime_mechanism_interface": True,
            },
            "artifacts": {
                "type": "TorchMLPSurrogateArtifact",
                "uncertainty": "residual_std",
                "ood_validity": True,
            },
            "runtime": {
                "requires_torch": True,
                "device": "auto|cpu|cuda|cuda:<index>",
                "early_stop": True,
                "objective": "mse",
                "optimizer": "adamw|adam|sgd|rmsprop",
                "epoch_checkpoint_resume": True,
                "active_runtime_mechanisms": self.mechanism_stack.summaries(),
            },
            "neural_family": family.description_dict(),
        }

    def execution_resource_request(self) -> ExecutionResourceRequest:
        total_threads = 1 + max(0, int(self.config.batch_num_workers))
        return ExecutionResourceRequest(
            threads=int(total_threads),
            backend="serial",
            label=str(self.name),
            device_tokens=self.resolve_execution_device_tokens(self.config.device),
            metadata={
                "backend_family": "pytorch",
                "batch_num_workers": int(self.config.batch_num_workers),
                "requested_device": str(self.config.device),
            },
        )

    def _resolve_device(self) -> torch.device:
        return resolve_torch_device(torch, str(self.config.device))

    def _split_indices(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        return split_train_val_indices(
            int(n),
            val_ratio=float(self.config.val_ratio),
            seed=int(self.config.random_seed),
            min_no_val_below=10,
        )

    @staticmethod
    def _clone_payload_cpu(value: Any) -> Any:
        if hasattr(value, "detach"):
            return value.detach().cpu().clone()
        if isinstance(value, np.ndarray):
            return np.asarray(value).copy()
        if isinstance(value, Mapping):
            return {str(k): TorchMLPSurrogateTrainer._clone_payload_cpu(v) for k, v in dict(value).items()}
        if isinstance(value, tuple):
            return tuple(TorchMLPSurrogateTrainer._clone_payload_cpu(v) for v in value)
        if isinstance(value, list):
            return [TorchMLPSurrogateTrainer._clone_payload_cpu(v) for v in value]
        return value

    @staticmethod
    def _clone_state_cpu(state: Mapping[str, Any]) -> dict[str, Any]:
        return dict(TorchMLPSurrogateTrainer._clone_payload_cpu(dict(state)))

    def _current_family_spec(self) -> NeuralTrainerFamilySpec:
        family = getattr(self, "neural_family_spec", None)
        if isinstance(family, NeuralTrainerFamilySpec):
            return family
        payload = build_torch_mlp_family_spec(
            trainer_key=str(self.name),
            hidden_layers=tuple(int(v) for v in tuple(self.config.hidden_dims)),
            activation=str(self.config.activation),
            dropout=float(self.config.dropout),
            optimizer=str(self.config.optimizer),
            objective=str(self.config.objective),
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
            epochs=int(self.config.epochs),
            batch_size=int(self.config.batch_size),
            shuffle=bool(self.config.batch_shuffle),
            drop_last=bool(self.config.batch_drop_last),
            num_workers=int(self.config.batch_num_workers),
            pin_memory=bool(self.config.batch_pin_memory),
            val_ratio=float(self.config.val_ratio),
            early_stopping=True,
            early_stop_patience=int(self.config.early_stop_patience),
            early_stop_min_delta=float(self.config.early_stop_min_delta),
            random_seed=int(self.config.random_seed),
            metadata={"preset_kind": "torch_backend"},
        ).as_dict()
        payload["optimization"] = {
            **dict(payload.get("optimization", {}) or {}),
            "optimizer_params": dict(self.config.optimizer_params),
        }
        family = coerce_neural_family_spec(payload, trainer_key=str(self.name))
        try:
            setattr(self, "neural_family_spec", family)
            setattr(self, "neural_family_metadata", family.description_dict())
        except Exception:
            pass
        return family

    def _current_family_payload(self) -> dict[str, Any]:
        return self._current_family_spec().description_dict()

    def _current_family_signature(self) -> str | None:
        return self._current_family_spec().family_signature()

    def _assert_parent_family_compatible(self, payload: Mapping[str, Any] | None) -> None:
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

    def _build_epoch_payload(
        self,
        *,
        epoch_done: int,
        d: int,
        m: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        model_state: Mapping[str, Any],
        optimizer_state: Mapping[str, Any],
        best_state: Mapping[str, Any],
        best_metric: float,
        best_epoch: int,
        patience: int,
        training_signature: Mapping[str, Any] | None = None,
        input_feature_indices: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        family_payload = self._current_family_payload()
        family_signature = self._current_family_signature()
        payload: dict[str, Any] = {
            "schema_version": 1,
            "trainer_name": str(self.name),
            "epoch_done": int(epoch_done),
            "input_dim": int(d),
            "output_dim": int(m),
            "hidden_dims": tuple(int(h) for h in self.config.hidden_dims),
            "activation": str(self.config.activation),
            "dropout": float(self.config.dropout),
            "train_idx": np.asarray(train_idx, dtype=int),
            "val_idx": np.asarray(val_idx, dtype=int),
            "model_state": self._clone_state_cpu(model_state),
            "optimizer_state": self._clone_payload_cpu(dict(optimizer_state)),
            "best_state": self._clone_state_cpu(best_state),
            "best_metric": float(best_metric),
            "best_epoch": int(best_epoch),
            "patience": int(patience),
            "input_feature_indices": (
                None if input_feature_indices is None else tuple(int(v) for v in tuple(input_feature_indices))
            ),
            "runtime_mechanisms": {
                "active_components": self.mechanism_stack.summaries(),
            },
            "neural_family": family_payload,
            "neural_family_signature": family_signature,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "training_signature": None if training_signature is None else dict(training_signature),
        }
        if torch.cuda.is_available():
            payload["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return dict(self._clone_payload_cpu(payload))

    def _save_epoch_checkpoint(
        self,
        *,
        path: Path,
        epoch_done: int,
        d: int,
        m: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        model_state: Mapping[str, Any],
        optimizer_state: Mapping[str, Any],
        best_state: Mapping[str, Any],
        best_metric: float,
        best_epoch: int,
        patience: int,
        training_signature: Mapping[str, Any] | None = None,
        input_feature_indices: Sequence[int] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._build_epoch_payload(
            epoch_done=int(epoch_done),
            d=int(d),
            m=int(m),
            train_idx=np.asarray(train_idx, dtype=int),
            val_idx=np.asarray(val_idx, dtype=int),
            model_state=model_state,
            optimizer_state=optimizer_state,
            best_state=best_state,
            best_metric=float(best_metric),
            best_epoch=int(best_epoch),
            patience=int(patience),
            training_signature=training_signature,
            input_feature_indices=input_feature_indices,
        )
        torch.save(payload, path)

    @staticmethod
    def _load_epoch_checkpoint(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"resume checkpoint must be dict payload: {path}")
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError(f"unsupported checkpoint schema_version in: {path}")
        return payload

    @classmethod
    def load_trainer_state(cls, path: str | Path) -> TrainerState:
        resume_path = Path(path).resolve()
        payload = cls._load_epoch_checkpoint(resume_path)
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
                "epoch_done": int(payload.get("epoch_done", 0)),
                "training_signature": signature.as_dict(),
                "neural_family_signature": payload.get("neural_family_signature"),
                "input_feature_indices": payload.get("input_feature_indices"),
            },
        )

    @staticmethod
    def _payload_from_artifact(artifact: object) -> dict[str, Any] | None:
        if not isinstance(artifact, TorchMLPSurrogateArtifact):
            return None
        metadata = dict(getattr(artifact, "metadata", {}) or {})
        return {
            "schema_version": 1,
            "trainer_name": "mlp_torch",
            "input_dim": int(artifact.input_dim),
            "output_dim": int(artifact.output_dim),
            "hidden_dims": tuple(int(h) for h in artifact.hidden_dims),
            "activation": str(artifact.activation),
            "dropout": float(artifact.dropout),
            "model_state": dict(artifact.model_state),
            "input_feature_indices": (
                None
                if getattr(artifact, "input_feature_indices", None) is None
                else tuple(int(v) for v in tuple(artifact.input_feature_indices))
            ),
            "x_mean": np.asarray(artifact.x_mean, dtype=float),
            "x_std": np.asarray(artifact.x_std, dtype=float),
            "residual_std": np.asarray(artifact.residual_std, dtype=float),
            "feature_names": tuple(str(v) for v in artifact.feature_names),
            "target_names": tuple(str(v) for v in artifact.target_names),
            "training_signature": metadata.get("training_signature"),
            "neural_family": metadata.get("neural_family"),
            "neural_family_signature": metadata.get("neural_family_signature"),
            "runtime_mechanisms": metadata.get("runtime_mechanisms"),
        }

    @classmethod
    def _extract_parent_model(cls, payload: Mapping[str, Any] | None) -> Any | None:
        if payload is None:
            return None
        model_state = payload.get("model_state")
        if not isinstance(model_state, Mapping):
            return None
        input_dim = payload.get("input_dim")
        output_dim = payload.get("output_dim")
        hidden_dims = payload.get("hidden_dims", tuple())
        activation = payload.get("activation", "relu")
        dropout = payload.get("dropout", 0.0)
        if input_dim is None or output_dim is None:
            return None
        return _TorchRuntimePredictor(
            input_dim=int(input_dim),
            output_dim=int(output_dim),
            hidden_dims=tuple(int(v) for v in tuple(hidden_dims)),
            activation=str(activation),
            dropout=float(dropout),
            model_state=cls._clone_state_cpu(dict(model_state)),
        )

    def _fit_internal(
        self,
        data: ProcessedDataset | SampleDataset,
        *,
        init: TrainingInit | None = None,
        training_signature: Mapping[str, Any] | None = None,
    ) -> tuple[TorchMLPSurrogateArtifact, TrainerState | None]:
        init_eff = init or TrainingInit()
        mode = str(init_eff.mode).strip().lower() or "fresh"
        parent_payload: dict[str, Any] | None = None
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
            payload = self._payload_from_artifact(init_eff.parent_artifact)
            parent_payload = None if payload is None else self._clone_state_cpu(payload)
            parent_source = str(getattr(init_eff.parent_artifact, "artifact_id", type(init_eff.parent_artifact).__name__))
            parent_kind = "artifact"

        if parent_payload is not None and self.config.resume_training_from:
            raise ValueError("resume source is ambiguous: both training_init.parent_state and config.resume_training_from were provided")
        if parent_payload is None and self.config.resume_training_from:
            resume_path = Path(str(self.config.resume_training_from)).resolve()
            parent_payload = self._load_epoch_checkpoint(resume_path)
            parent_source = str(resume_path)
            parent_kind = "trainer_state_file"
            if mode == "fresh":
                mode = "resume"

        if mode == "resume" and parent_payload is None:
            raise ValueError("resume mode requires parent trainer_state payload")
        if mode == "resume":
            self._assert_parent_family_compatible(parent_payload)

        prepared = prepare_training_data(
            data=data,
            numericizer=self.numericizer,
            pipeline=self.pipeline,
            biases=self.biases,
            fit_context_cls=FitContext,
        )
        normalized = prepared.normalized
        context = prepared.context
        Xb = prepared.X
        Yb = prepared.Y
        n, d, m = int(prepared.n), int(prepared.d), int(prepared.m)
        feature_names = prepared.feature_names
        target_names = prepared.target_names

        signal_parent_model = self._extract_parent_model(parent_payload)
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
        n, d, m = int(X_fit.shape[0]), int(X_fit.shape[1]), int(Y_fit.shape[1])
        if n <= 0 or d <= 0:
            raise ValueError("runtime mechanisms produced empty training data")
        if runtime_state.sample_weight is not None and np.asarray(runtime_state.sample_weight).reshape(-1).shape[0] != n:
            raise ValueError("sample_weight length mismatch after runtime mechanisms")
        feature_names = tuple(str(v) for v in runtime_state.feature_names) if runtime_state.feature_names else feature_names
        target_names = tuple(str(v) for v in runtime_state.target_names) if runtime_state.target_names else target_names
        input_feature_indices = (
            None if runtime_state.feature_indices is None else tuple(int(v) for v in np.asarray(runtime_state.feature_indices, dtype=int))
        )

        device = self._resolve_device()
        set_torch_seed(torch, int(self.config.random_seed))

        train_idx, val_idx = self._split_indices(n)

        X_train = torch.as_tensor(X_fit[train_idx], dtype=torch.float32)
        Y_train = torch.as_tensor(Y_fit[train_idx], dtype=torch.float32)

        sw = runtime_state.sample_weight
        if sw is not None:
            w_arr = np.asarray(sw, dtype=float).reshape(-1)
            if w_arr.shape[0] != n:
                raise ValueError("sample_weight length mismatch")
            w_train = torch.as_tensor(np.maximum(w_arr[train_idx], 0.0), dtype=torch.float32)
        else:
            w_train = None

        batch_spec = BatchStreamSpec(
            batch_size=int(max(1, self.config.batch_size)),
            shuffle=bool(self.config.batch_shuffle),
            drop_last=bool(self.config.batch_drop_last),
            num_workers=int(max(0, self.config.batch_num_workers)),
            pin_memory=bool(self.config.batch_pin_memory),
        )
        if w_train is None:
            loader = create_torch_batch_stream((X_train, Y_train), spec=batch_spec)
        else:
            loader = create_torch_batch_stream((X_train, Y_train, w_train), spec=batch_spec)

        model = TorchMLPRegressor(
            int(d),
            int(m),
            hidden_dims=tuple(int(h) for h in self.config.hidden_dims),
            activation=str(self.config.activation),
            dropout=float(self.config.dropout),
        ).to(device)
        hypothesis = TorchModuleHypothesisSpace(
            module=model,
            family="mlp_torch",
            name="torch_module",
        )
        objective = create_regression_objective(str(self.config.objective))

        optimizer_spec = OptimizerSpec(
            key=str(self.config.optimizer),
            params=dict(self.config.optimizer_params),
        )
        optimizer = create_torch_optimizer(
            hypothesis.parameters(),
            spec=optimizer_spec,
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
        )

        has_val = len(val_idx) > 0
        if has_val:
            X_val = torch.as_tensor(X_fit[val_idx], dtype=torch.float32, device=device)
            Y_val = torch.as_tensor(Y_fit[val_idx], dtype=torch.float32, device=device)

        checkpoint_dir: Path | None = None
        latest_checkpoint_path: Path | None = None
        checkpoint_every = int(max(0, int(self.config.checkpoint_every_epochs)))
        if self.config.checkpoint_dir:
            checkpoint_dir = Path(str(self.config.checkpoint_dir)).resolve()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            latest_checkpoint_path = checkpoint_dir / "latest.pt"

        best_metric = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in hypothesis.module.state_dict().items()}
        patience = 0
        best_epoch = 0
        start_epoch = 1
        resumed_epoch = 0
        resumed_from: str | None = None
        latest_saved_checkpoint: str | None = None

        payload = None if parent_payload is None else dict(parent_payload)
        if payload is not None and mode == "resume":
            if int(payload.get("schema_version", 0)) != 1:
                raise ValueError("unsupported in-memory trainer_state schema_version")
            resumed_from = str(parent_source or "trainer_state")

        if payload is not None and mode == "resume":
            if int(payload.get("input_dim", -1)) != int(d) or int(payload.get("output_dim", -1)) != int(m):
                raise ValueError(
                    f"resume checkpoint dim mismatch: ck=({payload.get('input_dim')},{payload.get('output_dim')}) "
                    f"current=({d},{m})"
                )
            parent_feature_indices = payload.get("input_feature_indices")
            parent_feature_indices = (
                None
                if parent_feature_indices is None
                else tuple(int(v) for v in tuple(parent_feature_indices))
            )
            if parent_feature_indices != input_feature_indices:
                raise ValueError(
                    "resume checkpoint input_feature_indices mismatch: "
                    f"ck={parent_feature_indices} current={input_feature_indices}"
                )

            ck_hidden = tuple(int(x) for x in payload.get("hidden_dims", ()))
            cur_hidden = tuple(int(x) for x in self.config.hidden_dims)
            if ck_hidden and ck_hidden != cur_hidden:
                raise ValueError(f"resume checkpoint hidden_dims mismatch: ck={ck_hidden}, current={cur_hidden}")

            ck_train_idx = np.asarray(payload.get("train_idx", np.asarray([], dtype=int)), dtype=int).reshape(-1)
            ck_val_idx = np.asarray(payload.get("val_idx", np.asarray([], dtype=int)), dtype=int).reshape(-1)
            if ck_train_idx.size > 0 and not np.array_equal(ck_train_idx, np.asarray(train_idx, dtype=int)):
                raise ValueError("resume checkpoint train split mismatch")
            if ck_val_idx.size > 0 and not np.array_equal(ck_val_idx, np.asarray(val_idx, dtype=int)):
                raise ValueError("resume checkpoint val split mismatch")

            hypothesis.module.load_state_dict(dict(payload["model_state"]), strict=True)
            optimizer.load_state_dict(dict(payload["optimizer_state"]))

            best_state = self._clone_state_cpu(payload.get("best_state", hypothesis.module.state_dict()))
            best_metric = float(payload.get("best_metric", float("inf")))
            best_epoch = int(payload.get("best_epoch", 0))
            patience = int(payload.get("patience", 0))
            resumed_epoch = int(payload.get("epoch_done", 0))
            start_epoch = max(1, resumed_epoch + 1)

            if "python_random_state" in payload:
                random.setstate(payload["python_random_state"])
            if "numpy_random_state" in payload:
                np.random.set_state(payload["numpy_random_state"])
            if "torch_rng_state" in payload:
                torch.set_rng_state(payload["torch_rng_state"])
            if "torch_cuda_rng_state_all" in payload and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state_all"])

        last_completed_epoch = max(0, int(start_epoch) - 1)

        for epoch in range(int(start_epoch), int(self.config.epochs) + 1):
            hypothesis.module.train()
            epoch_loss_sum = 0.0
            epoch_count = 0

            for batch in loader:
                if len(batch) == 3:
                    xb_cpu, yb_cpu, wb_cpu = batch
                else:
                    xb_cpu, yb_cpu = batch
                    wb_cpu = None

                xb = xb_cpu.to(device)
                yb = yb_cpu.to(device)

                optimizer.zero_grad(set_to_none=True)
                pred = hypothesis.forward(xb)

                if wb_cpu is None:
                    loss = objective.loss(pred, yb)
                else:
                    wb = wb_cpu.to(device)
                    loss = objective.loss(pred, yb, sample_weight=wb)

                loss.backward()
                optimizer.step()

                batch_size = int(xb.shape[0])
                epoch_loss_sum += float(loss.detach().cpu()) * batch_size
                epoch_count += batch_size

            train_loss = epoch_loss_sum / max(1, epoch_count)

            if has_val:
                hypothesis.module.eval()
                with torch.no_grad():
                    val_pred = hypothesis.forward(X_val)
                    val_loss = float(objective.loss(val_pred, Y_val).detach().cpu())
                monitor = val_loss
            else:
                monitor = train_loss

            should_stop = False
            if monitor + float(self.config.early_stop_min_delta) < best_metric:
                best_metric = monitor
                best_state = {k: v.detach().cpu().clone() for k, v in hypothesis.module.state_dict().items()}
                best_epoch = epoch
                patience = 0
            else:
                patience += 1
                if patience >= int(self.config.early_stop_patience):
                    should_stop = True

            if latest_checkpoint_path is not None:
                self._save_epoch_checkpoint(
                    path=latest_checkpoint_path,
                    epoch_done=int(epoch),
                    d=int(d),
                    m=int(m),
                    train_idx=np.asarray(train_idx, dtype=int),
                    val_idx=np.asarray(val_idx, dtype=int),
                    model_state=hypothesis.module.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    best_state=best_state,
                    best_metric=float(best_metric),
                    best_epoch=int(best_epoch),
                    patience=int(patience),
                    training_signature=training_signature,
                    input_feature_indices=input_feature_indices,
                )
                latest_saved_checkpoint = str(latest_checkpoint_path)
                if checkpoint_dir is not None and checkpoint_every > 0 and int(epoch) % int(checkpoint_every) == 0:
                    epoch_path = checkpoint_dir / f"epoch_{int(epoch):04d}.pt"
                    self._save_epoch_checkpoint(
                        path=epoch_path,
                        epoch_done=int(epoch),
                        d=int(d),
                        m=int(m),
                        train_idx=np.asarray(train_idx, dtype=int),
                        val_idx=np.asarray(val_idx, dtype=int),
                        model_state=hypothesis.module.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        best_state=best_state,
                        best_metric=float(best_metric),
                        best_epoch=int(best_epoch),
                        patience=int(patience),
                        training_signature=training_signature,
                        input_feature_indices=input_feature_indices,
                    )

            if bool(self.config.verbose):
                if has_val:
                    print(f"[mlp] epoch={epoch} train_loss={train_loss:.6f} val_loss={monitor:.6f}")
                else:
                    print(f"[mlp] epoch={epoch} train_loss={train_loss:.6f}")
            last_completed_epoch = int(epoch)
            if should_stop:
                break

        trainer_state_payload = self._build_epoch_payload(
            epoch_done=int(last_completed_epoch),
            d=int(d),
            m=int(m),
            train_idx=np.asarray(train_idx, dtype=int),
            val_idx=np.asarray(val_idx, dtype=int),
            model_state=hypothesis.module.state_dict(),
            optimizer_state=optimizer.state_dict(),
            best_state=best_state,
            best_metric=float(best_metric),
            best_epoch=int(best_epoch),
            patience=int(patience),
            training_signature=training_signature,
            input_feature_indices=input_feature_indices,
        )

        hypothesis.module.load_state_dict(best_state, strict=True)
        hypothesis.module.eval()

        with torch.no_grad():
            full_pred = hypothesis.forward(
                torch.as_tensor(X_fit, dtype=torch.float32, device=device)
            ).detach().cpu().numpy()

        residual = Y_fit - np.asarray(full_pred, dtype=float)
        residual_std = np.std(residual, axis=0, ddof=1) + 1e-8
        family_payload = self._current_family_payload()
        family_signature = self._current_family_signature()

        metadata = {
            "trainer": "TorchMLPSurrogateTrainer",
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
            "model": {
                "hidden_dims": [int(h) for h in self.config.hidden_dims],
                "activation": str(self.config.activation),
                "dropout": float(self.config.dropout),
                "lr": float(self.config.lr),
                "weight_decay": float(self.config.weight_decay),
                "optimizer": str(self.config.optimizer),
                "optimizer_params": dict(self.config.optimizer_params),
                "objective": str(self.config.objective),
                "epochs": int(self.config.epochs),
                "start_epoch": int(start_epoch),
                "last_completed_epoch": int(last_completed_epoch),
                "best_epoch": int(best_epoch),
                "best_monitor": float(best_metric),
            },
            "training_components": {
                "hypothesis_space": "torch_module",
                "objective": str(getattr(objective, "name", self.config.objective)),
                "optimizer": str(self.config.optimizer),
                "batch_stream": {
                    "batch_size": int(batch_spec.batch_size),
                    "shuffle": bool(batch_spec.shuffle),
                    "drop_last": bool(batch_spec.drop_last),
                    "num_workers": int(batch_spec.num_workers),
                    "pin_memory": bool(batch_spec.pin_memory),
                },
            },
            "device": str(device),
            "checkpointing": {
                "enabled": bool(latest_checkpoint_path is not None),
                "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
                "checkpoint_every_epochs": int(checkpoint_every),
                "latest_checkpoint": latest_saved_checkpoint,
            },
            "resume": {
                "enabled": bool(mode == "resume" and resumed_from is not None),
                "mode": str(mode),
                "from": resumed_from,
                "parent_kind": parent_kind,
                "resume_epoch": int(resumed_epoch),
                "start_epoch": int(start_epoch),
            },
        }
        self.mechanism_stack.run_post_fit(runtime_state, model=hypothesis.module, artifact_metadata=metadata)

        artifact = TorchMLPSurrogateArtifact(
            artifact_id=str(self.config.artifact_id),
            input_dim=int(d),
            output_dim=int(m),
            hidden_dims=tuple(int(h) for h in self.config.hidden_dims),
            activation=str(self.config.activation),
            dropout=float(self.config.dropout),
            model_state=best_state,
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
        signature_obj = coerce_training_signature(training_signature)
        trainer_state = TrainerState(
            trainer_name=str(self.name),
            payload=trainer_state_payload,
            schema_signature=signature_obj.schema_signature,
            feature_signature=signature_obj.feature_signature,
            target_signature=signature_obj.target_signature,
            objective_signature=signature_obj.objective_signature,
            pipeline_signature=signature_obj.pipeline_signature,
            numericizer_signature=signature_obj.numericizer_signature,
            regime_signature=signature_obj.regime_signature,
            symbolic_family_signature=signature_obj.symbolic_family_signature,
            metadata={
                "resume_source": resumed_from,
                "latest_checkpoint": latest_saved_checkpoint,
                "epoch_done": int(last_completed_epoch),
                "best_epoch": int(best_epoch),
                "input_feature_indices": input_feature_indices,
                "training_signature": signature_obj.as_dict(),
                "neural_family_signature": family_signature,
            },
        )
        return artifact, trainer_state

    def fit(self, data: ProcessedDataset | SampleDataset) -> TorchMLPSurrogateArtifact:
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
