from __future__ import annotations

import random
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from bias import BaseTrainingBias, FitContext, NoOpBias
from numericizer import BaseNumericizer, DefaultNumericizer, ModalityEncoder, TargetCodec
from pipeline import BasePipeline, IdentityPipeline

from core.artifacts.symbolic_artifact import SymbolicSurrogateArtifact
from core.common.base_trainer import BaseSurrogateTrainer
from core.common.batch_stream import BatchStreamSpec, create_torch_batch_stream
from core.common.contracts import ProcessedDataset, SampleDataset
from core.common.hypothesis_space import TorchModuleHypothesisSpace
from core.common.loss_objective import create_regression_objective
from core.common.param_optimizer import OptimizerSpec, create_torch_optimizer
from core.execution import ExecutionResourceRequest
from core.common.trainer_shared import (
    prepare_training_data,
    resolve_torch_device,
    set_torch_seed,
    split_train_val_indices,
)
from core.models.symbolic_torch_model import SymbolicTorchRegressor
from core.symbolic.structure_metadata import (
    build_assembler_budget_payload,
    build_basis_overlap_report,
    build_basis_semantics_payload,
    build_basis_term_rows,
)
from core.symbolic.symbolic_dsl import default_genome, default_genome_v2
from core.symbolic.trainer_state_io import (
    clone_symbolic_payload_cpu,
    load_symbolic_trainer_state_file,
    save_symbolic_trainer_state_file,
)
from core.symbolic.trainer_family import SymbolicStructureEngineSpec, coerce_symbolic_structure_engine_spec
from core.trainers.symbolic_stagewise_trainer import SymbolicStagewiseSurrogateTrainer, SymbolicStagewiseTrainerConfig
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
        "PyTorch is required for SymbolicTorchSurrogateTrainer. Install torch before using symbolic_torch."
    ) from exc


@dataclass(frozen=True)
class SymbolicTorchTrainerConfig:
    artifact_id: str = "symbolic_torch_surrogate_v2"
    version: str = "v2"  # v1 | v2

    # Optional explicit genome override. If provided, version strategy is skipped.
    genome: Sequence[Mapping[str, Any]] | None = None

    # Unified symbolic structure entry.
    structure_engine: SymbolicStructureEngineSpec | Mapping[str, Any] | None = None
    structure_engine_params: Mapping[str, Any] = field(default_factory=dict)

    # v1 base library
    library_ops: Sequence[str] = ("identity", "square", "sin", "cos")

    # v2 base + expansion controls
    v2_continuous_ops: Sequence[str] = ("identity", "sin", "cos")
    v2_binary_ops: Sequence[str] = ("identity",)
    v2_include_interactions: bool = True
    v2_max_interactions: int = 20
    v2_topk_features: int = 6
    v2_include_hinge: bool = True
    v2_hinge_quantiles: Sequence[float] = (0.25, 0.5, 0.75)

    # training
    epochs: int = 220
    batch_size: int = 128
    batch_shuffle: bool = True
    batch_drop_last: bool = False
    batch_num_workers: int = 0
    batch_pin_memory: bool = False
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # adamw | adam | sgd | rmsprop
    optimizer_params: Mapping[str, Any] = field(default_factory=dict)
    objective: str = "mse"
    l1_readout: float = 0.0
    l1_params: float = 0.0
    val_ratio: float = 0.15
    early_stop_patience: int = 25
    early_stop_min_delta: float = 1e-6
    random_seed: int = 42
    device: str = "auto"  # auto | cpu | cuda | cuda:<index>
    checkpoint_dir: str | None = None
    checkpoint_every_epochs: int = 0
    resume_training_from: str | None = None
    ood_z_threshold: float = 4.0
    epsilon: float = 1e-6
    verbose: bool = False


class SymbolicTorchSurrogateTrainer(BaseSurrogateTrainer):
    """Symbolic-torch hybrid trainer with unified structure engine and training-control hooks."""

    name = "symbolic_torch"

    def __init__(
        self,
        config: SymbolicTorchTrainerConfig | None = None,
        *,
        pipeline: BasePipeline | None = None,
        biases: Sequence[BaseTrainingBias] | None = None,
        numericizer: BaseNumericizer | None = None,
        modality_encoders: Mapping[str, ModalityEncoder] | None = None,
        target_codecs: Mapping[str, TargetCodec] | None = None,
        target_codec: str | None = None,
        categorical_unknown: str = "error",
    ) -> None:
        self.config = config or SymbolicTorchTrainerConfig()
        self.pipeline = pipeline or IdentityPipeline()
        self.biases = list(biases) if biases else [NoOpBias()]

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
        return {
            "supports_fresh": True,
            "supports_resume": True,
            "supports_warm_start": True,
            "supports_incremental": True,
            "supports_recalibration": False,
            "name": self.name,
            "model_family": "symbolic_hybrid",
            "backend": "pytorch",
            "nonlinear": True,
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "sample_weight": True,
                "target_codec": True,
                "dynamic_function_genome": True,
                "v2_interaction_terms": True,
                "v2_piecewise_hinge_terms": True,
                "stagewise_structure_search": True,
                "trainer_state": True,
            },
            "artifacts": {
                "type": "SymbolicSurrogateArtifact",
                "uncertainty": "residual_std",
                "ood_validity": True,
                "expression_export": True,
            },
            "runtime": {
                "requires_torch": True,
                "device": "auto|cpu|cuda|cuda:<index>",
                "early_stop": True,
                "objective": "mse",
                "optimizer": "adamw|adam|sgd|rmsprop",
                "resume_from_trainer_state": True,
            },
        }

    def execution_resource_request(self) -> ExecutionResourceRequest:
        total_threads = 1 + max(0, int(self.config.batch_num_workers))
        return ExecutionResourceRequest(
            threads=int(total_threads),
            backend="serial",
            label=str(self.name),
            device_tokens=self.resolve_execution_device_tokens(self.config.device),
            metadata={
                "backend_family": "pytorch_symbolic",
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
        return clone_symbolic_payload_cpu(value)

    @staticmethod
    def _clone_state_cpu(state: Mapping[str, Any]) -> dict[str, Any]:
        return dict(SymbolicTorchSurrogateTrainer._clone_payload_cpu(dict(state)))

    @staticmethod
    def _copy_genome(genome: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
        return tuple(dict(term) for term in tuple(genome))

    @classmethod
    def _load_state_payload(cls, path: str | Path) -> dict[str, Any]:
        return load_symbolic_trainer_state_file(path)

    @classmethod
    def save_trainer_state(cls, path: str | Path, state: TrainerState) -> str:
        return save_symbolic_trainer_state_file(
            path,
            trainer_name=str(getattr(state, "trainer_name", cls.name)),
            payload=dict(getattr(state, "payload", {})),
            metadata=dict(getattr(state, "metadata", {})),
        )

    def _save_epoch_checkpoint(
        self,
        *,
        path: Path,
        payload: Mapping[str, Any],
    ) -> None:
        save_symbolic_trainer_state_file(
            path,
            trainer_name=str(self.name),
            payload=dict(payload),
            metadata={"checkpoint_kind": "epoch"},
        )

    @classmethod
    def load_trainer_state(cls, path: str | Path) -> TrainerState:
        resume_path = Path(path).resolve()
        payload = cls._load_state_payload(resume_path)
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
            },
        )

    def _resolve_structure_engine(self) -> SymbolicStructureEngineSpec:
        if self.config.genome is not None:
            default = SymbolicStructureEngineSpec(
                structure_mode="explicit_genome",
                search_driver="none",
                dynamic_pool_enabled=False,
            )
        else:
            default = SymbolicStructureEngineSpec(
                structure_mode="seed_library",
                search_driver="local_seed_builder",
                dynamic_pool_enabled=False,
            )
        return coerce_symbolic_structure_engine_spec(self.config.structure_engine, default=default)

    def _structure_engine_params(self) -> dict[str, Any]:
        return dict(self.config.structure_engine_params or {})

    def _stagewise_search_genome(
        self,
        *,
        X_basis: np.ndarray,
        Y_basis: np.ndarray,
        feature_names: Sequence[str],
        target_names: Sequence[str],
        metadata: Mapping[str, Any],
        seed: int,
        engine: SymbolicStructureEngineSpec,
    ) -> tuple[Sequence[Mapping[str, Any]] | None, dict[str, Any]]:
        raw_cfg = self._structure_engine_params()
        allowed_keys = {f.name for f in fields(SymbolicStagewiseTrainerConfig)}
        cfg = {k: v for k, v in raw_cfg.items() if k in allowed_keys}
        ignored = sorted(set(raw_cfg.keys()) - allowed_keys)

        if "artifact_id" not in cfg:
            cfg["artifact_id"] = f"{self.config.artifact_id}_stagewise_structure"
        if "random_seed" not in cfg:
            cfg["random_seed"] = int(seed)
        if "search_path_memory_enabled" not in cfg:
            cfg["search_path_memory_enabled"] = True
        if "search_path_memory_namespace" not in cfg:
            cfg["search_path_memory_namespace"] = "symbolic_torch"

        try:
            stage_cfg = SymbolicStagewiseTrainerConfig(**cfg)
            stage_trainer = SymbolicStagewiseSurrogateTrainer(
                config=stage_cfg,
                pipeline=IdentityPipeline(),
                biases=[NoOpBias()],
                numericizer=self.numericizer,
            )
            warm_ds = ProcessedDataset(
                X_train=np.asarray(X_basis, dtype=float),
                y_train=np.asarray(Y_basis, dtype=float),
                feature_names=tuple(feature_names),
                target_names=tuple(target_names),
                metadata=dict(metadata),
            )
            warm_artifact = stage_trainer.fit(warm_ds)
            genome = tuple(warm_artifact.genome)
            if len(genome) == 0:
                return None, {
                    "status": "empty_genome",
                    "ignored_config_keys": ignored,
                    "engine_spec": engine.as_dict(),
                }
            return genome, {
                "status": "ok",
                "ignored_config_keys": ignored,
                "terms": int(len(genome)),
                "path_memory_enabled": bool(stage_cfg.search_path_memory_enabled),
                "path_memory_namespace": str(stage_cfg.search_path_memory_namespace),
                "path_memory_db_path": str(stage_cfg.search_path_memory_db_path),
                "seed": int(seed),
                "engine_spec": engine.as_dict(),
            }
        except Exception as exc:
            return None, {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "ignored_config_keys": ignored,
                "seed": int(seed),
                "engine_spec": engine.as_dict(),
            }

    def _seed_library_genome(
        self,
        X_basis: np.ndarray,
        Y_basis: np.ndarray,
        *,
        engine: SymbolicStructureEngineSpec,
    ) -> tuple[Sequence[Mapping[str, Any]], dict[str, Any]]:
        version = str(self.config.version or "v2").strip().lower()
        if version in {"v1", "1", "legacy"}:
            return (
                default_genome(int(X_basis.shape[1]), ops=tuple(str(o) for o in self.config.library_ops)),
                {
                    "status": "seed_library",
                    "seed_library_version": "v1",
                    "engine_spec": engine.as_dict(),
                },
            )

        y_for_score: np.ndarray | None
        if Y_basis.ndim == 2 and int(Y_basis.shape[1]) > 0:
            y_for_score = Y_basis[:, 0]
        else:
            y_for_score = None

        return (
            default_genome_v2(
                X_basis,
                y=y_for_score,
                continuous_ops=tuple(str(o) for o in self.config.v2_continuous_ops),
                binary_ops=tuple(str(o) for o in self.config.v2_binary_ops),
                include_interactions=bool(self.config.v2_include_interactions),
                max_interactions=int(self.config.v2_max_interactions),
                topk_features=int(self.config.v2_topk_features),
                include_hinge=bool(self.config.v2_include_hinge),
                hinge_quantiles=tuple(float(q) for q in self.config.v2_hinge_quantiles),
            ),
            {
                "status": "seed_library",
                "seed_library_version": "v2",
                "engine_spec": engine.as_dict(),
            },
        )

    def _build_genome(
        self,
        X_basis: np.ndarray,
        Y_basis: np.ndarray,
        *,
        feature_names: Sequence[str],
        target_names: Sequence[str],
        metadata: Mapping[str, Any],
        seed: int,
    ) -> tuple[Sequence[Mapping[str, Any]], dict[str, Any]]:
        engine = self._resolve_structure_engine()
        if self.config.genome is not None:
            return tuple(self.config.genome), {
                "status": "explicit",
                "engine_spec": engine.as_dict(),
            }

        mode = str(engine.structure_mode).strip().lower()
        if mode == "explicit_genome":
            raise ValueError("structure_engine=explicit_genome requires config.genome to be provided")

        if mode in {"stagewise_search", "stagewise_warmup_then_seed_library"}:
            stagewise_genome, stagewise_info = self._stagewise_search_genome(
                X_basis=X_basis,
                Y_basis=Y_basis,
                feature_names=feature_names,
                target_names=target_names,
                metadata=metadata,
                seed=int(seed),
                engine=engine,
            )
            if stagewise_genome is not None:
                return tuple(stagewise_genome), dict(stagewise_info)
            if mode == "stagewise_search":
                raise RuntimeError(
                    "structure_engine 'stagewise_search' failed to produce a genome: "
                    f"{stagewise_info.get('error', stagewise_info.get('status', 'unknown_error'))}"
                )
            seed_genome, seed_info = self._seed_library_genome(X_basis, Y_basis, engine=engine)
            merged = dict(seed_info)
            merged["fallback_from"] = dict(stagewise_info)
            merged["status"] = "stagewise_fallback_to_seed_library"
            return tuple(seed_genome), merged

        if mode in {"seed_library", "local_seed_builder", "default_genome"}:
            return self._seed_library_genome(X_basis, Y_basis, engine=engine)

        raise ValueError(f"unsupported structure_engine mode for symbolic_torch: {engine.structure_mode}")

    def _apply_parent_artifact_payload(
        self,
        model: SymbolicTorchRegressor,
        payload: Mapping[str, Any],
    ) -> None:
        parameter_values = payload.get("parameter_values")
        if isinstance(parameter_values, Mapping):
            for name, value in parameter_values.items():
                key = str(name)
                if key in model.param_table:
                    with torch.no_grad():
                        model.param_table[key].copy_(torch.tensor(float(value), dtype=torch.float32))

        readout_weight = payload.get("readout_weight")
        if readout_weight is not None:
            weight = torch.as_tensor(np.asarray(readout_weight, dtype=float).T, dtype=torch.float32)
            with torch.no_grad():
                model.readout.weight.copy_(weight)

        readout_bias = payload.get("readout_bias")
        if readout_bias is not None:
            bias = torch.as_tensor(np.asarray(readout_bias, dtype=float).reshape(-1), dtype=torch.float32)
            with torch.no_grad():
                model.readout.bias.copy_(bias)

    @staticmethod
    def _trainer_state_payload_from_artifact(artifact: Any) -> dict[str, Any]:
        if not isinstance(artifact, SymbolicSurrogateArtifact):
            raise TypeError(
                "symbolic_torch warm_start/incremental parent_artifact must be SymbolicSurrogateArtifact"
            )
        metadata = dict(getattr(artifact, "metadata", {}) or {})
        return {
            "schema_version": 1,
            "trainer_name": "symbolic_torch",
            "epoch_done": 0,
            "input_dim": int(np.asarray(getattr(artifact, "x_mean")).shape[0]),
            "output_dim": int(np.asarray(getattr(artifact, "readout_bias")).shape[0]),
            "genome": SymbolicTorchSurrogateTrainer._copy_genome(artifact.genome),
            "parameter_values": dict(artifact.parameter_values),
            "readout_weight": np.asarray(artifact.readout_weight, dtype=float),
            "readout_bias": np.asarray(artifact.readout_bias, dtype=float),
            "training_signature": dict(metadata.get("training_signature", {})),
        }

    def _build_trainer_state_payload(
        self,
        *,
        epoch_done: int,
        d: int,
        m: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        genome: Sequence[Mapping[str, Any]],
        model_state: Mapping[str, Any],
        optimizer_state: Mapping[str, Any],
        best_state: Mapping[str, Any],
        best_metric: float,
        best_epoch: int,
        patience: int,
        readout_weight: np.ndarray,
        readout_bias: np.ndarray,
        parameter_values: Mapping[str, float],
        training_signature: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "trainer_name": str(self.name),
            "epoch_done": int(epoch_done),
            "input_dim": int(d),
            "output_dim": int(m),
            "train_idx": np.asarray(train_idx, dtype=int),
            "val_idx": np.asarray(val_idx, dtype=int),
            "genome": self._copy_genome(genome),
            "model_state": self._clone_state_cpu(model_state),
            "optimizer_state": self._clone_payload_cpu(dict(optimizer_state)),
            "best_state": self._clone_state_cpu(best_state),
            "best_metric": float(best_metric),
            "best_epoch": int(best_epoch),
            "patience": int(patience),
            "readout_weight": np.asarray(readout_weight, dtype=float),
            "readout_bias": np.asarray(readout_bias, dtype=float),
            "parameter_values": dict(parameter_values),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "training_signature": None if training_signature is None else dict(training_signature),
        }
        if torch.cuda.is_available():
            payload["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return dict(self._clone_payload_cpu(payload))

    def _fit_internal(
        self,
        data: ProcessedDataset | SampleDataset,
        *,
        init: TrainingInit | None = None,
        training_signature: Mapping[str, Any] | None = None,
    ) -> tuple[SymbolicSurrogateArtifact, TrainerState | None]:
        init_eff = init or TrainingInit()
        mode = str(init_eff.mode)

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

        device = self._resolve_device()
        set_torch_seed(torch, int(self.config.random_seed))

        train_idx, val_idx = self._split_indices(n)

        X_train = torch.as_tensor(Xb[train_idx], dtype=torch.float32)
        Y_train = torch.as_tensor(Yb[train_idx], dtype=torch.float32)

        sw = context.sample_weight
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

        checkpoint_dir: Path | None = None
        latest_checkpoint_path: Path | None = None
        checkpoint_every = int(max(0, int(self.config.checkpoint_every_epochs)))
        if self.config.checkpoint_dir:
            checkpoint_dir = Path(str(self.config.checkpoint_dir)).resolve()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            latest_checkpoint_path = checkpoint_dir / "latest.pt"

        parent_payload: dict[str, Any] | None = None
        parent_source: str | None = None
        latest_saved_checkpoint: str | None = None
        if init_eff.parent_state is not None:
            parent_payload = dict(getattr(init_eff.parent_state, "payload", {}))
            parent_source = str(getattr(init_eff.parent_state, "trainer_name", "trainer_state"))
        elif init_eff.parent_artifact is not None:
            parent_payload = self._trainer_state_payload_from_artifact(init_eff.parent_artifact)
            parent_source = str(getattr(init_eff.parent_artifact, "artifact_id", type(init_eff.parent_artifact).__name__))

        if parent_payload is not None and self.config.resume_training_from:
            raise ValueError(
                "resume source is ambiguous: both training_init parent payload and config.resume_training_from were provided"
            )
        if parent_payload is None and self.config.resume_training_from:
            resume_path = Path(str(self.config.resume_training_from)).resolve()
            parent_payload = self._load_state_payload(resume_path)
            parent_source = str(resume_path)
            if mode == "fresh":
                mode = "resume"

        if mode == "resume" and parent_payload is None:
            raise ValueError("resume mode requires parent trainer_state payload")

        if parent_payload is not None:
            payload_in_dim = parent_payload.get("input_dim")
            payload_out_dim = parent_payload.get("output_dim")
            if payload_in_dim is not None and int(payload_in_dim) != int(d):
                raise ValueError(
                    f"parent symbolic payload input_dim mismatch: parent={payload_in_dim} current={d}"
                )
            if payload_out_dim is not None and int(payload_out_dim) != int(m):
                raise ValueError(
                    f"parent symbolic payload output_dim mismatch: parent={payload_out_dim} current={m}"
                )
            if mode == "resume":
                ck_train_idx = np.asarray(parent_payload.get("train_idx", np.asarray([], dtype=int)), dtype=int).reshape(-1)
                ck_val_idx = np.asarray(parent_payload.get("val_idx", np.asarray([], dtype=int)), dtype=int).reshape(-1)
                if ck_train_idx.size > 0 and not np.array_equal(ck_train_idx, np.asarray(train_idx, dtype=int)):
                    raise ValueError("resume trainer_state train split mismatch")
                if ck_val_idx.size > 0 and not np.array_equal(ck_val_idx, np.asarray(val_idx, dtype=int)):
                    raise ValueError("resume trainer_state val split mismatch")

        if parent_payload is not None and parent_payload.get("genome") is not None:
            genome = self._copy_genome(parent_payload.get("genome", ()))
            genome_info = {
                "status": "from_parent_payload",
                "source": parent_source,
            }
        else:
            genome, genome_info = self._build_genome(
                Xb,
                Yb,
                feature_names=feature_names,
                target_names=target_names,
                metadata=dict(normalized.metadata or {}),
                seed=int(self.config.random_seed),
            )
            genome = self._copy_genome(genome)

        model = SymbolicTorchRegressor(
            input_dim=int(d),
            output_dim=int(m),
            genome=genome,
            epsilon=float(self.config.epsilon),
        ).to(device)
        hypothesis = TorchModuleHypothesisSpace(
            module=model,
            family="symbolic_torch",
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
            X_val = torch.as_tensor(Xb[val_idx], dtype=torch.float32, device=device)
            Y_val = torch.as_tensor(Yb[val_idx], dtype=torch.float32, device=device)

        best_metric = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in hypothesis.module.state_dict().items()}
        patience = 0
        best_epoch = 0
        start_epoch = 1
        resumed_epoch = 0
        parent_training_source = None if parent_source is None else str(parent_source)

        if parent_payload is not None:
            if parent_payload.get("model_state") is not None:
                hypothesis.module.load_state_dict(dict(parent_payload["model_state"]), strict=True)
            else:
                self._apply_parent_artifact_payload(hypothesis.module, parent_payload)

            best_state = {k: v.detach().cpu().clone() for k, v in hypothesis.module.state_dict().items()}

            if mode == "resume":
                if parent_payload.get("optimizer_state") is None:
                    raise ValueError("resume mode requires optimizer_state in parent trainer_state payload")
                optimizer.load_state_dict(dict(parent_payload["optimizer_state"]))
                best_state = self._clone_state_cpu(parent_payload.get("best_state", hypothesis.module.state_dict()))
                best_metric = float(parent_payload.get("best_metric", float("inf")))
                best_epoch = int(parent_payload.get("best_epoch", 0))
                patience = int(parent_payload.get("patience", 0))
                resumed_epoch = int(parent_payload.get("epoch_done", 0))
                start_epoch = max(1, resumed_epoch + 1)
                if "python_random_state" in parent_payload:
                    random.setstate(parent_payload["python_random_state"])
                if "numpy_random_state" in parent_payload:
                    np.random.set_state(parent_payload["numpy_random_state"])
                if "torch_rng_state" in parent_payload:
                    torch.set_rng_state(parent_payload["torch_rng_state"])
                if "torch_cuda_rng_state_all" in parent_payload and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(parent_payload["torch_cuda_rng_state_all"])

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

                if float(self.config.l1_readout) > 0.0:
                    loss = loss + float(self.config.l1_readout) * torch.mean(
                        torch.abs(hypothesis.module.readout.weight)
                    )

                if float(self.config.l1_params) > 0.0 and len(hypothesis.module.param_table) > 0:
                    l1_parts = [torch.mean(torch.abs(p)) for _, p in hypothesis.module.param_table.items()]
                    if l1_parts:
                        l1_total = torch.mean(torch.stack(l1_parts))
                        loss = loss + float(self.config.l1_params) * l1_total

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

            if monitor + float(self.config.early_stop_min_delta) < best_metric:
                best_metric = monitor
                best_state = {k: v.detach().cpu().clone() for k, v in hypothesis.module.state_dict().items()}
                best_epoch = epoch
                patience = 0
            else:
                patience += 1
                if patience >= int(self.config.early_stop_patience):
                    last_completed_epoch = int(epoch)
                    break

            if latest_checkpoint_path is not None:
                readout_w_t, readout_b_t = hypothesis.module.export_readout()
                checkpoint_payload = self._build_trainer_state_payload(
                    epoch_done=int(epoch),
                    d=int(d),
                    m=int(m),
                    train_idx=np.asarray(train_idx, dtype=int),
                    val_idx=np.asarray(val_idx, dtype=int),
                    genome=hypothesis.module.genome,
                    model_state=hypothesis.module.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    best_state=best_state,
                    best_metric=float(best_metric),
                    best_epoch=int(best_epoch),
                    patience=int(patience),
                    readout_weight=np.asarray(readout_w_t.numpy(), dtype=float),
                    readout_bias=np.asarray(readout_b_t.numpy(), dtype=float),
                    parameter_values=dict(hypothesis.module.export_parameter_values()),
                    training_signature=training_signature,
                )
                self._save_epoch_checkpoint(path=latest_checkpoint_path, payload=checkpoint_payload)
                latest_saved_checkpoint = str(latest_checkpoint_path)
                if checkpoint_dir is not None and checkpoint_every > 0 and int(epoch) % int(checkpoint_every) == 0:
                    epoch_path = checkpoint_dir / f"epoch_{int(epoch):04d}.pt"
                    self._save_epoch_checkpoint(path=epoch_path, payload=checkpoint_payload)

            if bool(self.config.verbose):
                if has_val:
                    print(f"[symbolic] epoch={epoch} train_loss={train_loss:.6f} val_loss={monitor:.6f}")
                else:
                    print(f"[symbolic] epoch={epoch} train_loss={train_loss:.6f}")
            last_completed_epoch = int(epoch)

        hypothesis.module.load_state_dict(best_state, strict=True)
        hypothesis.module.eval()

        with torch.no_grad():
            full_pred = hypothesis.forward(
                torch.as_tensor(Xb, dtype=torch.float32, device=device)
            ).detach().cpu().numpy()

        residual = Yb - np.asarray(full_pred, dtype=float)
        residual_std = np.std(residual, axis=0, ddof=1) + 1e-8

        readout_w_t, readout_b_t = hypothesis.module.export_readout()
        readout_weight = np.asarray(readout_w_t.numpy(), dtype=float)
        readout_bias = np.asarray(readout_b_t.numpy(), dtype=float)
        param_values = dict(hypothesis.module.export_parameter_values())

        expressions = hypothesis.module.expression_strings(with_values=True)

        term_names = tuple(str(t.get("name", "")) for t in hypothesis.module.genome)
        n_interaction_terms = int(sum(1 for nname in term_names if "*" in nname))
        n_hinge_terms = int(sum(1 for nname in term_names if nname.startswith("relu(")))
        structure_engine_payload = self._resolve_structure_engine().as_dict()
        selected_basis_rows = build_basis_term_rows(
            hypothesis.module.genome,
            feature_names=feature_names,
            parameter_values=param_values,
            expression_strings=expressions,
            scope="global",
        )
        basis_semantics = build_basis_semantics_payload(
            selected_basis_rows,
            source="symbolic_torch.final_genome",
            basis_scope="global",
            extra={
                "parameter_backend": "torch",
                "task": "point",
                "structure_mode": str(structure_engine_payload.get("structure_mode", "")),
            },
        )
        basis_overlap_report = build_basis_overlap_report(
            selected_basis_rows,
            source="symbolic_torch.final_genome",
            extra={
                "parameter_backend": "torch",
                "task": "point",
            },
        )
        assembler_budget = build_assembler_budget_payload(
            source="symbolic_torch.training_config",
            assembler_mode="budgeted_symbolic_regression",
            output_expression_count=int(max(1, len(tuple(expressions)))),
            selected_basis_count=int(len(selected_basis_rows)),
            budget_axes={
                "epochs": int(self.config.epochs),
                "batch_size": int(max(1, self.config.batch_size)),
                "max_interactions": int(self.config.v2_max_interactions),
                "topk_features": int(self.config.v2_topk_features),
                "max_piecewise_branches": 1,
            },
            extra={
                "last_completed_epoch": int(last_completed_epoch),
                "target_dim": int(m),
                "trainable_symbolic_params": int(len(hypothesis.module.param_table)),
            },
        )

        trainer_state_payload = self._build_trainer_state_payload(
            epoch_done=int(last_completed_epoch),
            d=int(d),
            m=int(m),
            train_idx=np.asarray(train_idx, dtype=int),
            val_idx=np.asarray(val_idx, dtype=int),
            genome=hypothesis.module.genome,
            model_state=hypothesis.module.state_dict(),
            optimizer_state=optimizer.state_dict(),
            best_state=best_state,
            best_metric=float(best_metric),
            best_epoch=int(best_epoch),
            patience=int(patience),
            readout_weight=readout_weight,
            readout_bias=readout_bias,
            parameter_values=param_values,
            training_signature=training_signature,
        )

        metadata = {
            "trainer": "SymbolicTorchSurrogateTrainer",
            "n_train": int(n),
            "feature_dim": int(d),
            "target_dim": int(m),
            "pipeline": str(getattr(self.pipeline, "name", "identity")),
            "biases": [str(getattr(b, "name", type(b).__name__)) for b in self.biases],
            "fit_context": dict(context.metadata),
            "data_protocol": str((normalized.metadata or {}).get("input_protocol", "processed_dataset")),
            "data_metadata": dict(normalized.metadata or {}),
            "numericizer": str(getattr(self.numericizer, "name", type(self.numericizer).__name__)),
            "model": {
                "version": str(self.config.version),
                "library_ops": [str(o) for o in self.config.library_ops],
                "v2_continuous_ops": [str(o) for o in self.config.v2_continuous_ops],
                "v2_binary_ops": [str(o) for o in self.config.v2_binary_ops],
                "v2_include_interactions": bool(self.config.v2_include_interactions),
                "v2_max_interactions": int(self.config.v2_max_interactions),
                "v2_topk_features": int(self.config.v2_topk_features),
                "v2_include_hinge": bool(self.config.v2_include_hinge),
                "v2_hinge_quantiles": [float(q) for q in self.config.v2_hinge_quantiles],
                "epochs": int(self.config.epochs),
                "start_epoch": int(start_epoch),
                "last_completed_epoch": int(last_completed_epoch),
                "best_epoch": int(best_epoch),
                "best_monitor": float(best_metric),
                "lr": float(self.config.lr),
                "weight_decay": float(self.config.weight_decay),
                "optimizer": str(self.config.optimizer),
                "optimizer_params": dict(self.config.optimizer_params),
                "objective": str(self.config.objective),
                "l1_readout": float(self.config.l1_readout),
                "l1_params": float(self.config.l1_params),
                "terms": int(readout_weight.shape[0]),
                "interaction_terms": int(n_interaction_terms),
                "hinge_terms": int(n_hinge_terms),
                "trainable_symbolic_params": int(len(hypothesis.module.param_table)),
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
            "symbolic": {
                "genome": list(hypothesis.module.genome),
                "term_names": list(term_names),
                "parameter_values": dict(param_values),
                "expressions": list(expressions),
                "structure_engine": dict(structure_engine_payload),
                "genome_build": dict(genome_info),
            },
            "device": str(device),
            "checkpointing": {
                "enabled": bool(latest_checkpoint_path is not None),
                "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
                "checkpoint_every_epochs": int(checkpoint_every),
                "latest_checkpoint": latest_saved_checkpoint,
            },
            "resume": {
                "enabled": bool(mode == "resume" and parent_payload is not None),
                "from": parent_training_source,
                "resume_epoch": int(resumed_epoch),
                "start_epoch": int(start_epoch),
            },
            "training_init": {
                "mode": mode,
                "parent_source": parent_training_source,
            },
        }
        metadata["selected_basis"] = list(selected_basis_rows)
        metadata["basis_semantics"] = dict(basis_semantics)
        metadata["basis_overlap_report"] = dict(basis_overlap_report)
        metadata["assembler_budget"] = dict(assembler_budget)
        metadata["symbolic"].update(
            {
                "selected_basis": list(selected_basis_rows),
                "basis_semantics": dict(basis_semantics),
                "basis_overlap_report": dict(basis_overlap_report),
                "assembler_budget": dict(assembler_budget),
            }
        )
        family_spec = getattr(self, "symbolic_family_spec", None)
        if family_spec is not None and hasattr(family_spec, "as_dict"):
            metadata["symbolic_family"] = family_spec.as_dict()

        artifact = SymbolicSurrogateArtifact(
            artifact_id=str(self.config.artifact_id),
            genome=tuple(hypothesis.module.genome),
            parameter_values=param_values,
            readout_weight=readout_weight,
            readout_bias=readout_bias,
            x_mean=np.mean(Xb, axis=0),
            x_std=np.std(Xb, axis=0) + 1e-8,
            residual_std=np.asarray(residual_std, dtype=float),
            feature_names=feature_names,
            target_names=target_names,
            pipeline_name=str(getattr(self.pipeline, "name", "identity")),
            pipeline_state=dict(getattr(self.pipeline, "state_dict")()),
            ood_z_threshold=float(self.config.ood_z_threshold),
            epsilon=float(self.config.epsilon),
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
                "resume_source": parent_training_source if mode == "resume" else None,
                "epoch_done": int(last_completed_epoch),
                "best_epoch": int(best_epoch),
                "training_signature": signature_obj.as_dict(),
            },
        )
        return artifact, trainer_state

    def fit(self, data: ProcessedDataset | SampleDataset) -> SymbolicSurrogateArtifact:
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
