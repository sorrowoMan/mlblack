from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, fields
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
from core.common.loss_objective import create_quantile_objective
from core.common.param_optimizer import OptimizerSpec, create_torch_optimizer
from core.execution import ExecutionResourceRequest
from core.common.trainer_shared import (
    prepare_training_data,
    resolve_torch_device,
    set_torch_seed,
    split_train_val_indices,
)
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
from core.artifacts.piecewise_symbolic_interval_artifact import PiecewiseSymbolicIntervalSurrogateArtifact
from core.artifacts.symbolic_interval_artifact import SymbolicIntervalSurrogateArtifact
from core.models.symbolic_torch_model import SymbolicTorchRegressor
from core.trainers.symbolic_stagewise_trainer import SymbolicStagewiseSurrogateTrainer, SymbolicStagewiseTrainerConfig
from training import (
    FitResult,
    InnerRuntimeDispatcher,
    InnerRuntimeErrorPayload,
    InnerRuntimeFinishPayload,
    InnerRuntimeRoundPayload,
    InnerRuntimeStartPayload,
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
        "PyTorch is required for SymbolicTorchIntervalTrainer. Install torch before using symbolic_torch_interval."
    ) from exc


def _as_2d_float(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError("array must be 1D or 2D")
    return out


def _canonical_gate_key(bits: Sequence[int]) -> str:
    return "|".join(str(int(v)) for v in bits)


def _metadata_basis_groups(value: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(value, Mapping):
        return {
            str(key): [dict(row) for row in tuple(rows) if isinstance(row, Mapping)]
            for key, rows in dict(value).items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {"global": [dict(row) for row in tuple(value) if isinstance(row, Mapping)]}
    return {}


def _count_expression_strings(value: Any) -> int:
    if isinstance(value, str):
        return 1 if value.strip() else 0
    if isinstance(value, Mapping):
        return sum(_count_expression_strings(item) for item in dict(value).values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return sum(_count_expression_strings(item) for item in tuple(value))
    return 0


@dataclass(frozen=True)
class SymbolicTorchIntervalTrainerConfig:
    artifact_id: str = "symbolic_torch_interval_surrogate_v1"
    version: str = "v2"  # v1 | v2

    # quantiles
    lower_quantile: float = 0.1
    upper_quantile: float = 0.9

    # Optional explicit genome override for both heads.
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

    # Optional stagewise structure warmup (supports path-memory through stagewise config).
    stagewise_warmup_enabled: bool = False
    stagewise_warmup_params: Mapping[str, Any] = field(default_factory=dict)

    # Optional gate-piecewise training.
    gate_piecewise_enabled: bool = False
    gate_feature_names: Sequence[str] = (
        "gate_feature_0",
        "gate_feature_1",
        "gate_feature_2",
        "gate_feature_3",
    )
    gate_threshold: float = 0.5
    gate_min_leaf: int = 96
    gate_max_local_models: int = 8
    gate_blend_kappa: float = 512.0

    # regularization
    order_penalty: float = 8.0
    width_penalty: float = 0.0
    l1_readout: float = 0.0
    l1_params: float = 0.0

    # calibration
    conformal_calibration: bool = True
    conformal_level: float | None = None  # defaults to (upper-lower)

    # training
    epochs: int = 260
    batch_size: int = 128
    batch_shuffle: bool = True
    batch_drop_last: bool = False
    batch_num_workers: int = 0
    batch_pin_memory: bool = False
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # adamw | adam | sgd | rmsprop
    optimizer_params: Mapping[str, Any] = field(default_factory=dict)
    quantile_objective: str = "pinball"
    val_ratio: float = 0.15
    early_stop_patience: int = 30
    early_stop_min_delta: float = 1e-6
    random_seed: int = 42
    device: str = "auto"  # auto | cpu | cuda | cuda:<index>
    checkpoint_dir: str | None = None
    checkpoint_every_epochs: int = 0
    resume_training_from: str | None = None
    ood_z_threshold: float = 4.0
    epsilon: float = 1e-6
    verbose: bool = False


class SymbolicTorchIntervalTrainer(BaseSurrogateTrainer):
    """Dynamic symbolic interval trainer using dual quantile heads."""

    name = "symbolic_torch_interval"

    def __init__(
        self,
        config: SymbolicTorchIntervalTrainerConfig | None = None,
        *,
        pipeline: BasePipeline | None = None,
        biases: Sequence[BaseTrainingBias] | None = None,
        numericizer: BaseNumericizer | None = None,
        modality_encoders: Mapping[str, ModalityEncoder] | None = None,
        target_codecs: Mapping[str, TargetCodec] | None = None,
        target_codec: str | None = None,
        categorical_unknown: str = "error",
    ) -> None:
        self.config = config or SymbolicTorchIntervalTrainerConfig()
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
            "model_family": "symbolic_interval",
            "backend": "pytorch",
            "nonlinear": True,
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "sample_weight": True,
                "target_codec": True,
                "dynamic_function_genome": True,
                "interval_output": True,
                "conformal_calibration": True,
                "stagewise_warmup": True,
                "gate_piecewise": True,
            },
            "artifacts": {
                "type": "SymbolicIntervalSurrogateArtifact|PiecewiseSymbolicIntervalSurrogateArtifact",
                "predict": "center",
                "predict_interval": True,
                "expression_export": True,
            },
            "runtime": {
                "requires_torch": True,
                "device": "auto|cpu|cuda|cuda:<index>",
                "early_stop": True,
                "objective": "pinball",
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
                "backend_family": "pytorch_symbolic_interval",
                "batch_num_workers": int(self.config.batch_num_workers),
                "requested_device": str(self.config.device),
            },
        )

    def _resolve_device(self) -> torch.device:
        return resolve_torch_device(torch, str(self.config.device))

    def _split_indices(self, n: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        return split_train_val_indices(
            int(n),
            val_ratio=float(self.config.val_ratio),
            seed=int(seed),
            min_no_val_below=10,
        )

    @staticmethod
    def _clone_payload_cpu(value: Any) -> Any:
        return clone_symbolic_payload_cpu(value)

    @staticmethod
    def _clone_state_cpu(state: Mapping[str, Any]) -> dict[str, Any]:
        return dict(SymbolicTorchIntervalTrainer._clone_payload_cpu(dict(state)))

    @staticmethod
    def _copy_genome(genome: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
        return tuple(dict(term) for term in tuple(genome))

    @staticmethod
    def _extend_inner_runtime_context(
        base_context: Mapping[str, Any] | None,
        *,
        suffix: str,
        runtime_key: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged = {} if base_context is None else dict(base_context)
        base_run_id = str(merged.get("run_id", merged.get("task_id", "symbolic_interval")))
        suffix_text = str(suffix).strip()
        merged["run_id"] = f"{base_run_id}:{suffix_text}" if suffix_text else base_run_id
        merged.setdefault("task_id", base_run_id)
        merged["runtime_key"] = str(runtime_key)
        merged["trainer_name"] = str(merged.get("trainer_name", "symbolic_torch_interval"))
        if extra:
            for key, value in dict(extra).items():
                if value is not None:
                    merged[str(key)] = value
        return merged

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

    @classmethod
    def load_trainer_state(cls, path: str | Path) -> TrainerState:
        resume_path = Path(path).resolve()
        payload = cls._load_state_payload(resume_path)
        embedded = payload.get("training_signature")
        if not isinstance(embedded, Mapping) and isinstance(payload.get("global_state"), Mapping):
            embedded = dict(payload.get("global_state", {})).get("training_signature")
        signature = coerce_training_signature(embedded)
        epoch_done = int(payload.get("epoch_done", 0))
        if epoch_done <= 0 and isinstance(payload.get("global_state"), Mapping):
            epoch_done = int(dict(payload.get("global_state", {})).get("epoch_done", 0))
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
                "epoch_done": int(epoch_done),
                "training_signature": signature.as_dict(),
            },
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

    @staticmethod
    def _jsonable_manifest(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Mapping):
            return {
                str(k): SymbolicTorchIntervalTrainer._jsonable_manifest(v)
                for k, v in dict(value).items()
            }
        if isinstance(value, (list, tuple)):
            return [SymbolicTorchIntervalTrainer._jsonable_manifest(v) for v in value]
        return str(value)

    @classmethod
    def _write_aggregate_manifest(cls, path: Path, manifest: Mapping[str, Any]) -> str:
        out_path = Path(path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(cls._jsonable_manifest(dict(manifest)), fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        return str(out_path)

    @staticmethod
    def _core_payload_from_parent(
        parent_payload: Mapping[str, Any] | None,
        *,
        regime_key: str | None = None,
    ) -> dict[str, Any] | None:
        if parent_payload is None:
            return None
        raw = dict(parent_payload)
        mode = str(raw.get("mode", "")).strip().lower()
        if mode == "piecewise":
            if regime_key is not None:
                local_map = raw.get("local_states", {})
                if isinstance(local_map, Mapping) and regime_key in local_map and isinstance(local_map[regime_key], Mapping):
                    return dict(local_map[regime_key])
            global_state = raw.get("global_state")
            if isinstance(global_state, Mapping):
                return dict(global_state)
            return None
        return raw

    @staticmethod
    def _core_payload_from_interval_artifact(
        artifact: SymbolicIntervalSurrogateArtifact,
    ) -> dict[str, Any]:
        metadata = dict(getattr(artifact, "metadata", {}) or {})
        return {
            "schema_version": 1,
            "trainer_name": "symbolic_torch_interval",
            "epoch_done": int(metadata.get("model", {}).get("last_completed_epoch", 0) or 0),
            "input_dim": int(np.asarray(getattr(artifact, "x_mean")).shape[0]),
            "output_dim": int(np.asarray(getattr(artifact, "readout_bias_low")).shape[0]),
            "genome": SymbolicTorchIntervalTrainer._copy_genome(artifact.genome_low),
            "readout_weight_low": np.asarray(artifact.readout_weight_low, dtype=float),
            "readout_bias_low": np.asarray(artifact.readout_bias_low, dtype=float),
            "readout_weight_high": np.asarray(artifact.readout_weight_high, dtype=float),
            "readout_bias_high": np.asarray(artifact.readout_bias_high, dtype=float),
            "parameter_values_low": dict(artifact.parameter_values_low),
            "parameter_values_high": dict(artifact.parameter_values_high),
            "calibration_margin": np.asarray(artifact.calibration_margin, dtype=float),
            "training_signature": dict(metadata.get("training_signature", {})),
        }

    @classmethod
    def _trainer_state_payload_from_artifact(
        cls,
        artifact: SymbolicIntervalSurrogateArtifact | PiecewiseSymbolicIntervalSurrogateArtifact,
    ) -> dict[str, Any]:
        if isinstance(artifact, PiecewiseSymbolicIntervalSurrogateArtifact):
            global_state = cls._core_payload_from_interval_artifact(artifact.global_artifact)
            local_states = {
                str(key): cls._core_payload_from_interval_artifact(local_art)
                for key, local_art in dict(artifact.local_artifacts).items()
            }
            metadata = dict(getattr(artifact, "metadata", {}) or {})
            return {
                "schema_version": 1,
                "trainer_name": "symbolic_torch_interval",
                "mode": "piecewise",
                "global_state": global_state,
                "local_states": local_states,
                "gate_piecewise": dict(metadata.get("gate_piecewise", {})),
                "training_signature": dict(metadata.get("training_signature", {})),
            }
        if isinstance(artifact, SymbolicIntervalSurrogateArtifact):
            payload = cls._core_payload_from_interval_artifact(artifact)
            payload["mode"] = "global"
            return payload
        raise TypeError(
            "symbolic_torch_interval parent_artifact must be SymbolicIntervalSurrogateArtifact or "
            "PiecewiseSymbolicIntervalSurrogateArtifact"
        )

    @staticmethod
    def _interval_metrics(y_true: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict[str, float]:
        yt = np.asarray(y_true, dtype=float)
        lo = np.asarray(low, dtype=float)
        hi = np.asarray(high, dtype=float)

        if yt.ndim == 1:
            yt = yt.reshape(-1, 1)
        if lo.ndim == 1:
            lo = lo.reshape(-1, 1)
        if hi.ndim == 1:
            hi = hi.reshape(-1, 1)

        inside = (yt >= lo) & (yt <= hi)
        width = np.maximum(hi - lo, 0.0)
        return {
            "coverage": float(np.mean(inside)),
            "mean_width": float(np.mean(width)),
        }

    @staticmethod
    def _apply_margin(low: np.ndarray, high: np.ndarray, margin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo = np.asarray(low, dtype=float)
        hi = np.asarray(high, dtype=float)
        m = np.asarray(margin, dtype=float).reshape(1, -1)
        if lo.ndim == 1:
            lo = lo.reshape(-1, 1)
        if hi.ndim == 1:
            hi = hi.reshape(-1, 1)
        if m.shape[1] == 1 and lo.shape[1] > 1:
            m = np.repeat(m, lo.shape[1], axis=1)
        return lo - m, hi + m

    def _resolve_structure_engine(self) -> SymbolicStructureEngineSpec:
        if self.config.genome is not None:
            default = SymbolicStructureEngineSpec(
                structure_mode="explicit_genome",
                search_driver="none",
                dynamic_pool_enabled=False,
            )
        elif bool(self.config.stagewise_warmup_enabled):
            default = SymbolicStructureEngineSpec(
                structure_mode="stagewise_warmup_then_seed_library",
                search_driver="nsgablack_warmup",
                dynamic_pool_enabled=True,
            )
        else:
            default = SymbolicStructureEngineSpec(
                structure_mode="seed_library",
                search_driver="local_seed_builder",
                dynamic_pool_enabled=False,
            )
        return coerce_symbolic_structure_engine_spec(self.config.structure_engine, default=default)

    def _structure_engine_params(self) -> dict[str, Any]:
        raw = dict(self.config.structure_engine_params or {})
        if raw:
            return raw
        return dict(self.config.stagewise_warmup_params or {})

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
            cfg["artifact_id"] = f"{self.config.artifact_id}_stagewise_warmup"
        if "random_seed" not in cfg:
            cfg["random_seed"] = int(seed)
        if "search_path_memory_enabled" not in cfg:
            cfg["search_path_memory_enabled"] = True
        if "search_path_memory_namespace" not in cfg:
            cfg["search_path_memory_namespace"] = "interval_warmup"

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
        if mode in {"explicit_genome"}:
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

        raise ValueError(f"unsupported structure_engine mode for symbolic_torch_interval: {engine.structure_mode}")

    @staticmethod
    def _composite_interval_loss(
        *,
        pred_low: Any,
        pred_high: Any,
        target: Any,
        objective_low: Any,
        objective_high: Any,
        sample_weight: Any | None,
        order_penalty: float,
        width_penalty: float,
        l1_readout: float,
        l1_params: float,
        model_low: SymbolicTorchRegressor,
        model_high: SymbolicTorchRegressor,
        torch_module: Any,
    ) -> Any:
        loss_low = objective_low.loss(pred_low, target, sample_weight=sample_weight)
        loss_high = objective_high.loss(pred_high, target, sample_weight=sample_weight)

        loss = loss_low + loss_high

        if float(order_penalty) > 0.0:
            order_violation = torch_module.relu(pred_low - pred_high)
            loss = loss + float(order_penalty) * torch_module.mean(order_violation)

        if float(width_penalty) > 0.0:
            width = torch_module.relu(pred_high - pred_low)
            loss = loss + float(width_penalty) * torch_module.mean(width)

        if float(l1_readout) > 0.0:
            loss = loss + float(l1_readout) * (
                torch_module.mean(torch_module.abs(model_low.readout.weight))
                + torch_module.mean(torch_module.abs(model_high.readout.weight))
            )

        if float(l1_params) > 0.0:
            l1_parts = []
            if len(model_low.param_table) > 0:
                l1_parts.extend(torch_module.mean(torch_module.abs(p)) for _, p in model_low.param_table.items())
            if len(model_high.param_table) > 0:
                l1_parts.extend(torch_module.mean(torch_module.abs(p)) for _, p in model_high.param_table.items())
            if l1_parts:
                l1_total = torch_module.mean(torch_module.stack(list(l1_parts)))
                loss = loss + float(l1_params) * l1_total

        return loss

    def _fit_interval_core(
        self,
        *,
        X_basis: np.ndarray,
        Y_basis: np.ndarray,
        sample_weight: np.ndarray | None,
        feature_names: Sequence[str],
        target_names: Sequence[str],
        metadata: Mapping[str, Any],
        seed: int,
        run_tag: str,
        parent_payload: Mapping[str, Any] | None = None,
        training_signature: Mapping[str, Any] | None = None,
        checkpoint_dir: Path | None = None,
        inner_runtime_dispatcher: InnerRuntimeDispatcher | None = None,
        inner_runtime_context: Mapping[str, Any] | None = None,
    ) -> tuple[SymbolicIntervalSurrogateArtifact, dict[str, Any]]:
        Xb = _as_2d_float(np.asarray(X_basis, dtype=float))
        Yb = _as_2d_float(np.asarray(Y_basis, dtype=float))

        if Xb.shape[0] != Yb.shape[0]:
            raise ValueError("X_basis and Y_basis row mismatch")

        n = int(Xb.shape[0])
        d = int(Xb.shape[1])
        m = int(Yb.shape[1])

        if n <= 0:
            raise ValueError("empty training data")

        runtime_context = self._extend_inner_runtime_context(
            inner_runtime_context,
            suffix=str(run_tag),
            runtime_key="symbolic_interval_core",
            extra={
                "run_tag": str(run_tag),
                "seed": int(seed),
                "feature_dim": int(d),
                "target_dim": int(m),
            },
        )
        runtime_metadata = {
            "task": "interval",
            "epochs": int(self.config.epochs),
            "batch_size": int(max(1, self.config.batch_size)),
            "checkpoint_enabled": bool(checkpoint_dir is not None),
        }
        last_epoch_for_error: int | None = None

        try:
            device = self._resolve_device()
            runtime_context["device"] = str(device)
            set_torch_seed(torch, int(seed))

            train_idx, val_idx = self._split_indices(n, seed=int(seed))

            X_train = torch.as_tensor(Xb[train_idx], dtype=torch.float32)
            Y_train = torch.as_tensor(Yb[train_idx], dtype=torch.float32)

            sw_train = None
            if sample_weight is not None:
                sw = np.asarray(sample_weight, dtype=float).reshape(-1)
                if sw.shape[0] != n:
                    raise ValueError("sample_weight length mismatch")
                sw = np.maximum(sw, 0.0)
                sw_train = torch.as_tensor(sw[train_idx], dtype=torch.float32)

            batch_spec = BatchStreamSpec(
                batch_size=int(max(1, self.config.batch_size)),
                shuffle=bool(self.config.batch_shuffle),
                drop_last=bool(self.config.batch_drop_last),
                num_workers=int(max(0, self.config.batch_num_workers)),
                pin_memory=bool(self.config.batch_pin_memory),
            )
            if sw_train is None:
                loader = create_torch_batch_stream((X_train, Y_train), spec=batch_spec)
            else:
                loader = create_torch_batch_stream((X_train, Y_train, sw_train), spec=batch_spec)

            latest_checkpoint_path: Path | None = None
            latest_saved_checkpoint: str | None = None
            checkpoint_every = int(max(0, int(self.config.checkpoint_every_epochs)))
            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                latest_checkpoint_path = checkpoint_dir / "latest.pt"

            if parent_payload is not None and parent_payload.get("genome") is not None:
                genome = self._copy_genome(parent_payload.get("genome", ()))
                structure_info = {
                    "status": "from_parent_payload",
                    "source": str(parent_payload.get("trainer_name", "trainer_state")),
                }
            else:
                genome, structure_info = self._build_genome(
                    Xb,
                    Yb,
                    feature_names=feature_names,
                    target_names=target_names,
                    metadata=metadata,
                    seed=int(seed),
                )
                genome = self._copy_genome(genome)

            model_low = SymbolicTorchRegressor(
                input_dim=int(d),
                output_dim=int(m),
                genome=genome,
                epsilon=float(self.config.epsilon),
            ).to(device)
            model_high = SymbolicTorchRegressor(
                input_dim=int(d),
                output_dim=int(m),
                genome=genome,
                epsilon=float(self.config.epsilon),
            ).to(device)

            hypothesis_low = TorchModuleHypothesisSpace(
                module=model_low,
                family="symbolic_torch_interval",
                name="low_head",
            )
            hypothesis_high = TorchModuleHypothesisSpace(
                module=model_high,
                family="symbolic_torch_interval",
                name="high_head",
            )

            objective_low = create_quantile_objective(
                str(self.config.quantile_objective),
                quantile=float(self.config.lower_quantile),
            )
            objective_high = create_quantile_objective(
                str(self.config.quantile_objective),
                quantile=float(self.config.upper_quantile),
            )

            optimizer_spec = OptimizerSpec(
                key=str(self.config.optimizer),
                params=dict(self.config.optimizer_params),
            )
            opt_low = create_torch_optimizer(
                hypothesis_low.parameters(),
                spec=optimizer_spec,
                lr=float(self.config.lr),
                weight_decay=float(self.config.weight_decay),
            )
            opt_high = create_torch_optimizer(
                hypothesis_high.parameters(),
                spec=optimizer_spec,
                lr=float(self.config.lr),
                weight_decay=float(self.config.weight_decay),
            )

            has_val = len(val_idx) > 0
            if has_val:
                X_val = torch.as_tensor(Xb[val_idx], dtype=torch.float32, device=device)
                Y_val = torch.as_tensor(Yb[val_idx], dtype=torch.float32, device=device)

            best_metric = float("inf")
            best_state_low = {k: v.detach().cpu().clone() for k, v in model_low.state_dict().items()}
            best_state_high = {k: v.detach().cpu().clone() for k, v in model_high.state_dict().items()}
            best_epoch = 0
            patience = 0
            start_epoch = 1
            resumed_epoch = 0
            resumed_from: str | None = None

            if parent_payload is not None:
                if parent_payload.get("input_dim") is not None and int(parent_payload.get("input_dim")) != int(d):
                    raise ValueError(
                        f"parent interval payload input_dim mismatch: parent={parent_payload.get('input_dim')} current={d}"
                    )
                if parent_payload.get("output_dim") is not None and int(parent_payload.get("output_dim")) != int(m):
                    raise ValueError(
                        f"parent interval payload output_dim mismatch: parent={parent_payload.get('output_dim')} current={m}"
                    )

                if parent_payload.get("model_state_low") is not None:
                    model_low.load_state_dict(dict(parent_payload["model_state_low"]), strict=True)
                else:
                    if parent_payload.get("readout_weight_low") is not None:
                        with torch.no_grad():
                            model_low.readout.weight.copy_(
                                torch.as_tensor(
                                    np.asarray(parent_payload["readout_weight_low"], dtype=float).T,
                                    dtype=torch.float32,
                                )
                            )
                            model_low.readout.bias.copy_(
                                torch.as_tensor(
                                    np.asarray(parent_payload["readout_bias_low"], dtype=float).reshape(-1),
                                    dtype=torch.float32,
                                )
                            )
                    if isinstance(parent_payload.get("parameter_values_low"), Mapping):
                        for name, value in dict(parent_payload["parameter_values_low"]).items():
                            if str(name) in model_low.param_table:
                                with torch.no_grad():
                                    model_low.param_table[str(name)].copy_(torch.tensor(float(value), dtype=torch.float32))

                if parent_payload.get("model_state_high") is not None:
                    model_high.load_state_dict(dict(parent_payload["model_state_high"]), strict=True)
                else:
                    if parent_payload.get("readout_weight_high") is not None:
                        with torch.no_grad():
                            model_high.readout.weight.copy_(
                                torch.as_tensor(
                                    np.asarray(parent_payload["readout_weight_high"], dtype=float).T,
                                    dtype=torch.float32,
                                )
                            )
                            model_high.readout.bias.copy_(
                                torch.as_tensor(
                                    np.asarray(parent_payload["readout_bias_high"], dtype=float).reshape(-1),
                                    dtype=torch.float32,
                                )
                            )
                    if isinstance(parent_payload.get("parameter_values_high"), Mapping):
                        for name, value in dict(parent_payload["parameter_values_high"]).items():
                            if str(name) in model_high.param_table:
                                with torch.no_grad():
                                    model_high.param_table[str(name)].copy_(
                                        torch.tensor(float(value), dtype=torch.float32)
                                    )

                best_state_low = {k: v.detach().cpu().clone() for k, v in model_low.state_dict().items()}
                best_state_high = {k: v.detach().cpu().clone() for k, v in model_high.state_dict().items()}

                if parent_payload.get("optimizer_state_low") is not None and parent_payload.get("optimizer_state_high") is not None:
                    ck_train_idx = np.asarray(parent_payload.get("train_idx", np.asarray([], dtype=int)), dtype=int).reshape(-1)
                    ck_val_idx = np.asarray(parent_payload.get("val_idx", np.asarray([], dtype=int)), dtype=int).reshape(-1)
                    if ck_train_idx.size > 0 and not np.array_equal(ck_train_idx, np.asarray(train_idx, dtype=int)):
                        raise ValueError("resume trainer_state train split mismatch")
                    if ck_val_idx.size > 0 and not np.array_equal(ck_val_idx, np.asarray(val_idx, dtype=int)):
                        raise ValueError("resume trainer_state val split mismatch")
                    opt_low.load_state_dict(dict(parent_payload["optimizer_state_low"]))
                    opt_high.load_state_dict(dict(parent_payload["optimizer_state_high"]))
                    best_state_low = self._clone_state_cpu(parent_payload.get("best_state_low", model_low.state_dict()))
                    best_state_high = self._clone_state_cpu(parent_payload.get("best_state_high", model_high.state_dict()))
                    best_metric = float(parent_payload.get("best_metric", float("inf")))
                    best_epoch = int(parent_payload.get("best_epoch", 0))
                    patience = int(parent_payload.get("patience", 0))
                    resumed_epoch = int(parent_payload.get("epoch_done", 0))
                    start_epoch = max(1, resumed_epoch + 1)
                    resumed_from = str(parent_payload.get("resume_source", parent_payload.get("trainer_name", "trainer_state")))
                    if "python_random_state" in parent_payload:
                        random.setstate(parent_payload["python_random_state"])
                    if "numpy_random_state" in parent_payload:
                        np.random.set_state(parent_payload["numpy_random_state"])
                    if "torch_rng_state" in parent_payload:
                        torch.set_rng_state(parent_payload["torch_rng_state"])
                    if "torch_cuda_rng_state_all" in parent_payload and torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(parent_payload["torch_cuda_rng_state_all"])

            last_completed_epoch = max(0, int(start_epoch) - 1)
            total_rounds = max(0, int(self.config.epochs) - int(start_epoch) + 1)
            if inner_runtime_dispatcher is not None and inner_runtime_dispatcher.enabled:
                inner_runtime_dispatcher.emit_start(
                    InnerRuntimeStartPayload(
                        run_id=str(runtime_context.get("run_id", runtime_context.get("task_id", "symbolic_interval"))),
                        runtime_key=str(runtime_context.get("runtime_key", "symbolic_interval_core")),
                        trainer_name=str(runtime_context.get("trainer_name", self.name)),
                        total_rounds=int(total_rounds),
                        input_shape=(int(n), int(d)),
                        feature_names=tuple(str(v) for v in tuple(feature_names)),
                        seed_terms=int(len(tuple(genome))),
                        context=runtime_context,
                        metadata={
                            **dict(runtime_metadata),
                            "start_epoch": int(start_epoch),
                            "resumed_epoch": int(resumed_epoch),
                            "device": str(device),
                        },
                    )
                )

            for epoch in range(int(start_epoch), int(self.config.epochs) + 1):
                last_epoch_for_error = int(epoch)
                model_low.train()
                model_high.train()
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
                    wb = None if wb_cpu is None else wb_cpu.to(device)

                    opt_low.zero_grad(set_to_none=True)
                    opt_high.zero_grad(set_to_none=True)

                    pred_low = model_low(xb)
                    pred_high = model_high(xb)

                    loss = self._composite_interval_loss(
                        pred_low=pred_low,
                        pred_high=pred_high,
                        target=yb,
                        objective_low=objective_low,
                        objective_high=objective_high,
                        sample_weight=wb,
                        order_penalty=float(self.config.order_penalty),
                        width_penalty=float(self.config.width_penalty),
                        l1_readout=float(self.config.l1_readout),
                        l1_params=float(self.config.l1_params),
                        model_low=model_low,
                        model_high=model_high,
                        torch_module=torch,
                    )

                    loss.backward()
                    opt_low.step()
                    opt_high.step()

                    bs = int(xb.shape[0])
                    epoch_loss_sum += float(loss.detach().cpu()) * bs
                    epoch_count += bs

                train_loss = epoch_loss_sum / max(1, epoch_count)

                if has_val:
                    model_low.eval()
                    model_high.eval()
                    with torch.no_grad():
                        val_low = model_low(X_val)
                        val_high = model_high(X_val)
                        val_loss = self._composite_interval_loss(
                            pred_low=val_low,
                            pred_high=val_high,
                            target=Y_val,
                            objective_low=objective_low,
                            objective_high=objective_high,
                            sample_weight=None,
                            order_penalty=float(self.config.order_penalty),
                            width_penalty=float(self.config.width_penalty),
                            l1_readout=float(self.config.l1_readout),
                            l1_params=float(self.config.l1_params),
                            model_low=model_low,
                            model_high=model_high,
                            torch_module=torch,
                        )
                        monitor = float(val_loss.detach().cpu())
                else:
                    monitor = float(train_loss)

                should_stop = False
                if monitor + float(self.config.early_stop_min_delta) < best_metric:
                    best_metric = monitor
                    best_state_low = {k: v.detach().cpu().clone() for k, v in model_low.state_dict().items()}
                    best_state_high = {k: v.detach().cpu().clone() for k, v in model_high.state_dict().items()}
                    best_epoch = int(epoch)
                    patience = 0
                else:
                    patience += 1
                    if patience >= int(self.config.early_stop_patience):
                        should_stop = True

                if latest_checkpoint_path is not None:
                    readout_w_low_t, readout_b_low_t = model_low.export_readout()
                    readout_w_high_t, readout_b_high_t = model_high.export_readout()
                    checkpoint_payload = {
                        "schema_version": 1,
                        "trainer_name": str(self.name),
                        "epoch_done": int(epoch),
                        "input_dim": int(d),
                        "output_dim": int(m),
                        "train_idx": np.asarray(train_idx, dtype=int),
                        "val_idx": np.asarray(val_idx, dtype=int),
                        "genome": self._copy_genome(genome),
                        "model_state_low": self._clone_state_cpu(model_low.state_dict()),
                        "model_state_high": self._clone_state_cpu(model_high.state_dict()),
                        "optimizer_state_low": self._clone_payload_cpu(dict(opt_low.state_dict())),
                        "optimizer_state_high": self._clone_payload_cpu(dict(opt_high.state_dict())),
                        "best_state_low": self._clone_state_cpu(best_state_low),
                        "best_state_high": self._clone_state_cpu(best_state_high),
                        "best_metric": float(best_metric),
                        "best_epoch": int(best_epoch),
                        "patience": int(patience),
                        "readout_weight_low": np.asarray(readout_w_low_t.numpy(), dtype=float),
                        "readout_bias_low": np.asarray(readout_b_low_t.numpy(), dtype=float),
                        "readout_weight_high": np.asarray(readout_w_high_t.numpy(), dtype=float),
                        "readout_bias_high": np.asarray(readout_b_high_t.numpy(), dtype=float),
                        "parameter_values_low": dict(model_low.export_parameter_values()),
                        "parameter_values_high": dict(model_high.export_parameter_values()),
                        "python_random_state": random.getstate(),
                        "numpy_random_state": np.random.get_state(),
                        "torch_rng_state": torch.get_rng_state(),
                        "run_tag": str(run_tag),
                        "training_signature": None if training_signature is None else dict(training_signature),
                    }
                    if torch.cuda.is_available():
                        checkpoint_payload["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
                    self._save_epoch_checkpoint(path=latest_checkpoint_path, payload=checkpoint_payload)
                    latest_saved_checkpoint = str(latest_checkpoint_path)
                    if checkpoint_dir is not None and checkpoint_every > 0 and int(epoch) % int(checkpoint_every) == 0:
                        epoch_path = checkpoint_dir / f"epoch_{int(epoch):04d}.pt"
                        self._save_epoch_checkpoint(path=epoch_path, payload=checkpoint_payload)

                if inner_runtime_dispatcher is not None and inner_runtime_dispatcher.enabled:
                    inner_runtime_dispatcher.emit_round_end(
                        InnerRuntimeRoundPayload(
                            run_id=str(runtime_context.get("run_id", runtime_context.get("task_id", "symbolic_interval"))),
                            runtime_key=str(runtime_context.get("runtime_key", "symbolic_interval_core")),
                            trainer_name=str(runtime_context.get("trainer_name", self.name)),
                            round_index=int(epoch - int(start_epoch) + 1),
                            total_rounds=int(total_rounds),
                            genome_size=int(len(tuple(genome))),
                            history_entry={
                                "epoch": int(epoch),
                                "train_loss": float(train_loss),
                                "monitor": float(monitor),
                                "best_metric": float(best_metric),
                                "best_epoch": int(best_epoch),
                                "patience": int(patience),
                                "checkpoint": latest_saved_checkpoint,
                                "stopped": bool(should_stop),
                            },
                            context=runtime_context,
                            metadata={**dict(runtime_metadata), "device": str(device)},
                        )
                    )

                if bool(self.config.verbose):
                    if has_val:
                        print(f"[symbolic_interval:{run_tag}] epoch={epoch} train_loss={train_loss:.6f} val_loss={monitor:.6f}")
                    else:
                        print(f"[symbolic_interval:{run_tag}] epoch={epoch} train_loss={train_loss:.6f}")
                last_completed_epoch = int(epoch)
                if should_stop:
                    break

            model_low.load_state_dict(best_state_low, strict=True)
            model_high.load_state_dict(best_state_high, strict=True)
            model_low.eval()
            model_high.eval()

            with torch.no_grad():
                full_low = model_low(torch.as_tensor(Xb, dtype=torch.float32, device=device)).detach().cpu().numpy()
                full_high = model_high(torch.as_tensor(Xb, dtype=torch.float32, device=device)).detach().cpu().numpy()
            low_raw = np.asarray(full_low, dtype=float)
            high_raw = np.asarray(full_high, dtype=float)
            if low_raw.ndim == 1:
                low_raw = low_raw.reshape(-1, 1)
            if high_raw.ndim == 1:
                high_raw = high_raw.reshape(-1, 1)

            lo = np.minimum(low_raw, high_raw)
            hi = np.maximum(low_raw, high_raw)

            target_coverage = float(np.clip(float(self.config.upper_quantile) - float(self.config.lower_quantile), 0.0, 1.0))
            conformal_level = (
                float(self.config.conformal_level)
                if self.config.conformal_level is not None
                else target_coverage
            )
            conformal_level = float(np.clip(conformal_level, 0.0, 1.0))

            margin = np.zeros((m,), dtype=float)
            if bool(self.config.conformal_calibration):
                miss_low = lo - Yb
                miss_high = Yb - hi
                nonconf = np.maximum(np.maximum(miss_low, miss_high), 0.0)
                for j in range(m):
                    q = float(np.quantile(nonconf[:, j], conformal_level)) if n > 0 else 0.0
                    margin[j] = float(max(0.0, q))

            lo_cal, hi_cal = self._apply_margin(lo, hi, margin)
            interval_metrics = self._interval_metrics(Yb, lo_cal, hi_cal)

            center_pred = 0.5 * (lo_cal + hi_cal)
            residual = Yb - center_pred
            residual_std = np.std(residual, axis=0, ddof=1) + 1e-8

            readout_w_low_t, readout_b_low_t = model_low.export_readout()
            readout_w_high_t, readout_b_high_t = model_high.export_readout()

            readout_weight_low = np.asarray(readout_w_low_t.numpy(), dtype=float)
            readout_bias_low = np.asarray(readout_b_low_t.numpy(), dtype=float)
            readout_weight_high = np.asarray(readout_w_high_t.numpy(), dtype=float)
            readout_bias_high = np.asarray(readout_b_high_t.numpy(), dtype=float)

            param_values_low = dict(model_low.export_parameter_values())
            param_values_high = dict(model_high.export_parameter_values())

            trainer_state_payload = {
                "schema_version": 1,
                "trainer_name": str(self.name),
                "epoch_done": int(last_completed_epoch),
                "input_dim": int(d),
                "output_dim": int(m),
                "train_idx": np.asarray(train_idx, dtype=int),
                "val_idx": np.asarray(val_idx, dtype=int),
                "genome": self._copy_genome(genome),
                "model_state_low": self._clone_state_cpu(model_low.state_dict()),
                "model_state_high": self._clone_state_cpu(model_high.state_dict()),
                "optimizer_state_low": self._clone_payload_cpu(dict(opt_low.state_dict())),
                "optimizer_state_high": self._clone_payload_cpu(dict(opt_high.state_dict())),
                "best_state_low": self._clone_state_cpu(best_state_low),
                "best_state_high": self._clone_state_cpu(best_state_high),
                "best_metric": float(best_metric),
                "best_epoch": int(best_epoch),
                "patience": int(patience),
                "readout_weight_low": np.asarray(readout_weight_low, dtype=float),
                "readout_bias_low": np.asarray(readout_bias_low, dtype=float),
                "readout_weight_high": np.asarray(readout_weight_high, dtype=float),
                "readout_bias_high": np.asarray(readout_bias_high, dtype=float),
                "parameter_values_low": dict(param_values_low),
                "parameter_values_high": dict(param_values_high),
                "calibration_margin": np.asarray(margin, dtype=float).reshape(-1),
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "run_tag": str(run_tag),
                "training_signature": None if training_signature is None else dict(training_signature),
            }
            if torch.cuda.is_available():
                trainer_state_payload["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

            term_names = tuple(str(t.get("name", "")) for t in genome)
            n_interaction_terms = int(sum(1 for name in term_names if "*" in name))
            n_hinge_terms = int(sum(1 for name in term_names if name.startswith("relu(")))

            expressions_low = model_low.expression_strings(with_values=True)
            expressions_high = model_high.expression_strings(with_values=True)
            selected_basis = {
                "low": build_basis_term_rows(
                    genome,
                    feature_names=feature_names,
                    parameter_values=param_values_low,
                    expression_strings=expressions_low,
                    scope="low",
                ),
                "high": build_basis_term_rows(
                    genome,
                    feature_names=feature_names,
                    parameter_values=param_values_high,
                    expression_strings=expressions_high,
                    scope="high",
                ),
            }
            basis_semantics = build_basis_semantics_payload(
                selected_basis,
                source="symbolic_torch_interval.dual_head_genome",
                basis_scope="global",
                extra={
                    "task": "interval",
                    "parameter_backend": "torch",
                    "head_groups": ["low", "high"],
                },
            )
            basis_overlap_report = build_basis_overlap_report(
                selected_basis,
                source="symbolic_torch_interval.dual_head_genome",
                extra={
                    "task": "interval",
                    "parameter_backend": "torch",
                    "head_groups": ["low", "high"],
                },
            )
            assembler_budget = build_assembler_budget_payload(
                source="symbolic_torch_interval.training_config",
                assembler_mode="dual_head_budgeted_symbolic_regression",
                output_expression_count=int(len(tuple(expressions_low)) + len(tuple(expressions_high))),
                selected_basis_count=int(sum(len(rows) for rows in selected_basis.values())),
                budget_axes={
                    "epochs": int(self.config.epochs),
                    "batch_size": int(max(1, self.config.batch_size)),
                    "max_interactions": int(self.config.v2_max_interactions),
                    "topk_features": int(self.config.v2_topk_features),
                    "gate_max_local_models": 1,
                },
                extra={
                    "last_completed_epoch": int(last_completed_epoch),
                    "target_dim": int(m),
                    "trainable_symbolic_params": int(len(model_low.param_table) + len(model_high.param_table)),
                },
            )

            art_meta = {
                "trainer": "SymbolicTorchIntervalTrainer",
                "n_train": int(n),
                "feature_dim": int(d),
                "target_dim": int(m),
                "pipeline": str(getattr(self.pipeline, "name", "identity")),
                "biases": [str(getattr(b, "name", type(b).__name__)) for b in self.biases],
                "data_protocol": str((metadata or {}).get("input_protocol", "processed_dataset")),
                "data_metadata": dict(metadata or {}),
                "numericizer": str(getattr(self.numericizer, "name", type(self.numericizer).__name__)),
                "interval": {
                    "lower_quantile": float(self.config.lower_quantile),
                    "upper_quantile": float(self.config.upper_quantile),
                    "target_coverage": float(target_coverage),
                    "conformal_calibration": bool(self.config.conformal_calibration),
                    "conformal_level": float(conformal_level),
                    "calibration_margin": [float(v) for v in np.asarray(margin, dtype=float).reshape(-1).tolist()],
                    "train_coverage": float(interval_metrics["coverage"]),
                    "train_mean_width": float(interval_metrics["mean_width"]),
                },
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
                    "quantile_objective": str(self.config.quantile_objective),
                    "order_penalty": float(self.config.order_penalty),
                    "width_penalty": float(self.config.width_penalty),
                    "l1_readout": float(self.config.l1_readout),
                    "l1_params": float(self.config.l1_params),
                    "terms": int(len(term_names)),
                    "interaction_terms": int(n_interaction_terms),
                    "hinge_terms": int(n_hinge_terms),
                    "trainable_symbolic_params": int(len(model_low.param_table) + len(model_high.param_table)),
                },
                "training_components": {
                    "hypothesis_space": "torch_module(low+high)",
                    "objective_low": str(getattr(objective_low, "name", self.config.quantile_objective)),
                    "objective_high": str(getattr(objective_high, "name", self.config.quantile_objective)),
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
                    "genome_low": list(genome),
                    "genome_high": list(genome),
                    "term_names": list(term_names),
                    "parameter_values_low": dict(param_values_low),
                    "parameter_values_high": dict(param_values_high),
                    "expressions_low": list(expressions_low),
                    "expressions_high": list(expressions_high),
                    "structure_engine": dict(structure_info),
                },
                "device": str(device),
                "run_tag": str(run_tag),
                "checkpointing": {
                    "enabled": bool(latest_checkpoint_path is not None),
                    "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
                    "checkpoint_every_epochs": int(checkpoint_every),
                    "latest_checkpoint": latest_saved_checkpoint,
                },
                "resume": {
                    "enabled": bool(parent_payload is not None and resumed_epoch > 0),
                    "from": resumed_from,
                    "resume_epoch": int(resumed_epoch),
                    "start_epoch": int(start_epoch),
                },
            }
            art_meta["selected_basis"] = {str(key): list(rows) for key, rows in selected_basis.items()}
            art_meta["basis_semantics"] = dict(basis_semantics)
            art_meta["basis_overlap_report"] = dict(basis_overlap_report)
            art_meta["assembler_budget"] = dict(assembler_budget)
            art_meta["symbolic"].update(
                {
                    "selected_basis": {str(key): list(rows) for key, rows in selected_basis.items()},
                    "basis_semantics": dict(basis_semantics),
                    "basis_overlap_report": dict(basis_overlap_report),
                    "assembler_budget": dict(assembler_budget),
                }
            )
            family_spec = getattr(self, "symbolic_family_spec", None)
            if family_spec is not None and hasattr(family_spec, "as_dict"):
                art_meta["symbolic_family"] = family_spec.as_dict()

            artifact = SymbolicIntervalSurrogateArtifact(
                artifact_id=str(self.config.artifact_id),
                lower_quantile=float(self.config.lower_quantile),
                upper_quantile=float(self.config.upper_quantile),
                genome_low=tuple(genome),
                parameter_values_low=param_values_low,
                readout_weight_low=readout_weight_low,
                readout_bias_low=readout_bias_low,
                genome_high=tuple(genome),
                parameter_values_high=param_values_high,
                readout_weight_high=readout_weight_high,
                readout_bias_high=readout_bias_high,
                x_mean=np.mean(Xb, axis=0),
                x_std=np.std(Xb, axis=0) + 1e-8,
                residual_std=np.asarray(residual_std, dtype=float),
                calibration_margin=np.asarray(margin, dtype=float).reshape(-1),
                feature_names=tuple(feature_names),
                target_names=tuple(target_names),
                pipeline_name=str(getattr(self.pipeline, "name", "identity")),
                pipeline_state=dict(getattr(self.pipeline, "state_dict")()),
                ood_z_threshold=float(self.config.ood_z_threshold),
                epsilon=float(self.config.epsilon),
                metadata=art_meta,
            )
            if inner_runtime_dispatcher is not None and inner_runtime_dispatcher.enabled:
                completed_rounds = (
                    int(max(0, last_completed_epoch - int(start_epoch) + 1))
                    if last_completed_epoch >= int(start_epoch)
                    else 0
                )
                inner_runtime_dispatcher.emit_finish(
                    InnerRuntimeFinishPayload(
                        run_id=str(runtime_context.get("run_id", runtime_context.get("task_id", "symbolic_interval"))),
                        runtime_key=str(runtime_context.get("runtime_key", "symbolic_interval_core")),
                        trainer_name=str(runtime_context.get("trainer_name", self.name)),
                        total_rounds=int(total_rounds),
                        completed_rounds=int(completed_rounds),
                        genome_size=int(len(tuple(genome))),
                        final_metrics={
                            "best_metric": float(best_metric),
                            "best_epoch": int(best_epoch),
                            "coverage": float(interval_metrics["coverage"]),
                            "mean_width": float(interval_metrics["mean_width"]),
                        },
                        context=runtime_context,
                        metadata={**dict(runtime_metadata), "device": str(device)},
                    )
                )
            return artifact, trainer_state_payload
        except Exception as exc:
            if inner_runtime_dispatcher is not None and inner_runtime_dispatcher.enabled:
                inner_runtime_dispatcher.emit_error(
                    InnerRuntimeErrorPayload(
                        run_id=str(runtime_context.get("run_id", runtime_context.get("task_id", "symbolic_interval"))),
                        runtime_key=str(runtime_context.get("runtime_key", "symbolic_interval_core")),
                        trainer_name=str(runtime_context.get("trainer_name", self.name)),
                        error=f"{type(exc).__name__}: {exc}",
                        round_index=last_epoch_for_error,
                        context=runtime_context,
                        metadata=runtime_metadata,
                    )
                )
            raise

    def _gate_indices(self, feature_names: Sequence[str]) -> tuple[int, ...]:
        names = tuple(str(v) for v in feature_names)
        out: list[int] = []
        for g in self.config.gate_feature_names:
            key = str(g)
            if key in names:
                out.append(int(names.index(key)))
        return tuple(out)

    def _gate_keys(self, X_gate: np.ndarray, gate_indices: Sequence[int]) -> tuple[str, ...]:
        x = _as_2d_float(np.asarray(X_gate, dtype=float))
        idx = [int(i) for i in gate_indices]
        if len(idx) == 0:
            return tuple("GLOBAL" for _ in range(int(x.shape[0])))
        g = np.asarray(x[:, idx], dtype=float)
        th = float(self.config.gate_threshold)
        out: list[str] = []
        for row in g:
            bits = tuple(int(float(v) > th) for v in row.tolist())
            out.append(_canonical_gate_key(bits))
        return tuple(out)

    def _regime_index(
        self,
        X_gate: np.ndarray,
        gate_indices: Sequence[int],
    ) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, int]]:
        keys = np.asarray(self._gate_keys(X_gate, gate_indices), dtype=object)

        all_counts: dict[str, int] = {}
        for k in keys.tolist():
            ks = str(k)
            all_counts[ks] = int(all_counts.get(ks, 0) + 1)

        min_leaf = int(max(1, self.config.gate_min_leaf))
        max_models = int(max(1, self.config.gate_max_local_models))

        eligible = [(k, c) for k, c in all_counts.items() if int(c) >= min_leaf]
        eligible.sort(key=lambda item: (-int(item[1]), str(item[0])))

        chosen = eligible[:max_models]
        chosen_keys = {str(k) for k, _ in chosen}

        skipped: dict[str, int] = {}
        for k, c in all_counts.items():
            if str(k) not in chosen_keys:
                skipped[str(k)] = int(c)

        selected: dict[str, np.ndarray] = {}
        for key, _ in chosen:
            mask = keys == str(key)
            idx = np.where(mask)[0]
            if idx.size <= 0:
                continue
            selected[str(key)] = np.asarray(idx, dtype=int)

        return selected, all_counts, skipped

    @staticmethod
    def _seed_from_key(base_seed: int, key: str) -> int:
        offset = int(np.sum(np.frombuffer(str(key).encode("utf-8"), dtype=np.uint8)))
        return int(base_seed + offset)

    def _fit_internal(
        self,
        data: ProcessedDataset | SampleDataset,
        *,
        init: TrainingInit | None = None,
        training_signature: Mapping[str, Any] | None = None,
    ) -> tuple[SymbolicIntervalSurrogateArtifact | PiecewiseSymbolicIntervalSurrogateArtifact, TrainerState | None]:
        init_eff = init or TrainingInit()
        mode = str(init_eff.mode)
        training_signature_payload = {} if training_signature is None else dict(training_signature)
        training_signature_meta = dict(training_signature_payload.get("metadata", {}) or {})
        inner_runtime_hooks = tuple(init_eff.inner_runtime_hooks)
        inner_runtime_dispatcher = (
            InnerRuntimeDispatcher.from_hooks(inner_runtime_hooks) if inner_runtime_hooks else None
        )

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
        n = int(prepared.n)
        d = int(prepared.d)
        feature_names = prepared.feature_names
        target_names = prepared.target_names

        if n <= 0:
            raise ValueError("empty training dataset")

        sw = None
        if context.sample_weight is not None:
            sw = np.asarray(context.sample_weight, dtype=float).reshape(-1)
            if sw.shape[0] != n:
                raise ValueError("sample_weight length mismatch")

        base_data_meta = dict(normalized.metadata or {})
        base_data_meta.setdefault(
            "input_protocol",
            str((normalized.metadata or {}).get("input_protocol", "processed_dataset")),
        )
        inner_runtime_context = {
            "task_id": str(training_signature_meta.get("task_id", "train_task")),
            "run_id": str(training_signature_meta.get("task_id", "train_task")),
            "trainer_name": str(getattr(self, "name", type(self).__name__)),
            "training_mode": str(mode),
            "artifact_id": str(self.config.artifact_id),
        }

        checkpoint_root: Path | None = None
        if self.config.checkpoint_dir:
            checkpoint_root = Path(str(self.config.checkpoint_dir)).resolve()
            checkpoint_root.mkdir(parents=True, exist_ok=True)

        parent_payload: dict[str, Any] | None = None
        parent_source: str | None = None
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

        global_parent_payload = self._core_payload_from_parent(parent_payload, regime_key=None)
        global_artifact, global_state_payload = self._fit_interval_core(
            X_basis=Xb,
            Y_basis=Yb,
            sample_weight=sw,
            feature_names=feature_names,
            target_names=target_names,
            metadata=base_data_meta,
            seed=int(self.config.random_seed),
            run_tag="global",
            parent_payload=global_parent_payload,
            training_signature=training_signature,
            checkpoint_dir=None if checkpoint_root is None else checkpoint_root / "global",
            inner_runtime_dispatcher=inner_runtime_dispatcher,
            inner_runtime_context=self._extend_inner_runtime_context(
                inner_runtime_context,
                suffix="global",
                runtime_key="symbolic_interval_core",
                extra={"scope": "global"},
            ),
        )

        if not bool(self.config.gate_piecewise_enabled):
            global_artifact.metadata.setdefault("training_init", {"mode": mode, "parent_source": parent_source})
            top_payload = dict(global_state_payload)
            top_payload["mode"] = "global"
            top_payload["training_signature"] = None if training_signature is None else dict(training_signature)
            signature_obj = coerce_training_signature(training_signature)
            trainer_state = TrainerState(
                trainer_name=str(self.name),
                payload=top_payload,
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
                    "epoch_done": int(top_payload.get("epoch_done", 0)),
                    "training_signature": signature_obj.as_dict(),
                },
            )
            if checkpoint_root is not None:
                final_state_path = checkpoint_root / "trainer_state" / "latest.pt"
                self.save_trainer_state(final_state_path, trainer_state)
                global_artifact.metadata.setdefault("trainer_state_checkpoint", str(final_state_path))
            return global_artifact, trainer_state

        X_gate = _as_2d_float(np.asarray(normalized.X_train, dtype=float))
        if X_gate.shape[0] != n or X_gate.shape[1] != d:
            X_gate = np.asarray(Xb, dtype=float)

        gate_idx = self._gate_indices(feature_names)
        if len(gate_idx) == 0:
            global_artifact.metadata["gate_piecewise"] = {
                "enabled": True,
                "status": "no_gate_features_matched",
                "gate_feature_names": [str(v) for v in self.config.gate_feature_names],
                "feature_names": [str(v) for v in feature_names],
            }
            signature_obj = coerce_training_signature(training_signature)
            top_payload = {
                "schema_version": 1,
                "trainer_name": str(self.name),
                "mode": "global",
                **dict(global_state_payload),
                "training_signature": None if training_signature is None else dict(training_signature),
            }
            trainer_state = TrainerState(
                trainer_name=str(self.name),
                payload=top_payload,
                schema_signature=signature_obj.schema_signature,
                feature_signature=signature_obj.feature_signature,
                target_signature=signature_obj.target_signature,
                objective_signature=signature_obj.objective_signature,
                pipeline_signature=signature_obj.pipeline_signature,
                numericizer_signature=signature_obj.numericizer_signature,
                regime_signature=signature_obj.regime_signature,
                symbolic_family_signature=signature_obj.symbolic_family_signature,
                metadata={"training_signature": signature_obj.as_dict(), "epoch_done": int(top_payload.get("epoch_done", 0))},
            )
            return global_artifact, trainer_state

        selected, all_counts, skipped = self._regime_index(X_gate, gate_idx)
        if len(selected) == 0:
            global_artifact.metadata["gate_piecewise"] = {
                "enabled": True,
                "status": "no_regime_selected",
                "gate_feature_names": [str(v) for v in self.config.gate_feature_names],
                "gate_indices": [int(v) for v in gate_idx],
                "counts_all": {str(k): int(v) for k, v in all_counts.items()},
                "gate_min_leaf": int(self.config.gate_min_leaf),
                "gate_max_local_models": int(self.config.gate_max_local_models),
            }
            signature_obj = coerce_training_signature(training_signature)
            top_payload = {
                "schema_version": 1,
                "trainer_name": str(self.name),
                "mode": "global",
                **dict(global_state_payload),
                "training_signature": None if training_signature is None else dict(training_signature),
            }
            trainer_state = TrainerState(
                trainer_name=str(self.name),
                payload=top_payload,
                schema_signature=signature_obj.schema_signature,
                feature_signature=signature_obj.feature_signature,
                target_signature=signature_obj.target_signature,
                objective_signature=signature_obj.objective_signature,
                pipeline_signature=signature_obj.pipeline_signature,
                numericizer_signature=signature_obj.numericizer_signature,
                regime_signature=signature_obj.regime_signature,
                symbolic_family_signature=signature_obj.symbolic_family_signature,
                metadata={"training_signature": signature_obj.as_dict(), "epoch_done": int(top_payload.get("epoch_done", 0))},
            )
            return global_artifact, trainer_state

        local_artifacts: dict[str, SymbolicIntervalSurrogateArtifact] = {}
        local_state_payloads: dict[str, dict[str, Any]] = {}
        local_counts: dict[str, int] = {}
        failed: dict[str, str] = {}

        piecewise_runtime_context = self._extend_inner_runtime_context(
            inner_runtime_context,
            suffix="piecewise",
            runtime_key="symbolic_interval_piecewise",
            extra={"scope": "piecewise", "gate_feature_count": int(len(gate_idx))},
        )
        selected_keys = [str(key) for key in selected.keys() if int(np.asarray(selected[key], dtype=int).size) > 0]
        if inner_runtime_dispatcher is not None and inner_runtime_dispatcher.enabled:
            inner_runtime_dispatcher.emit_start(
                InnerRuntimeStartPayload(
                    run_id=str(piecewise_runtime_context.get("run_id", piecewise_runtime_context.get("task_id", "symbolic_interval"))),
                    runtime_key=str(piecewise_runtime_context.get("runtime_key", "symbolic_interval_piecewise")),
                    trainer_name=str(piecewise_runtime_context.get("trainer_name", self.name)),
                    total_rounds=int(len(selected_keys)),
                    input_shape=(int(n), int(d)),
                    feature_names=tuple(str(v) for v in tuple(feature_names)),
                    seed_terms=int(len(selected_keys)),
                    context=piecewise_runtime_context,
                    metadata={
                        "task": "interval_piecewise",
                        "gate_indices": [int(v) for v in gate_idx],
                        "selected_regimes": list(selected_keys),
                    },
                )
            )

        piecewise_round = 0
        for key, idx in selected.items():
            if int(idx.size) <= 0:
                continue
            piecewise_round += 1
            try:
                local_meta = dict(base_data_meta)
                local_meta["gate_regime_key"] = str(key)
                local_meta["gate_regime_size"] = int(idx.size)

                local_parent_payload = self._core_payload_from_parent(parent_payload, regime_key=str(key))
                local_art, local_state = self._fit_interval_core(
                    X_basis=np.asarray(Xb[idx, :], dtype=float),
                    Y_basis=np.asarray(Yb[idx, :], dtype=float),
                    sample_weight=None if sw is None else np.asarray(sw[idx], dtype=float),
                    feature_names=feature_names,
                    target_names=target_names,
                    metadata=local_meta,
                    seed=self._seed_from_key(int(self.config.random_seed), str(key)),
                    run_tag=f"regime_{key}",
                    parent_payload=local_parent_payload,
                    training_signature=training_signature,
                    checkpoint_dir=(
                        None
                        if checkpoint_root is None
                        else checkpoint_root / "local" / f"regime_{str(key).replace('|', '_')}"
                    ),
                    inner_runtime_dispatcher=inner_runtime_dispatcher,
                    inner_runtime_context=self._extend_inner_runtime_context(
                        piecewise_runtime_context,
                        suffix=f"regime_{str(key)}",
                        runtime_key="symbolic_interval_core",
                        extra={"scope": "local_regime", "regime_key": str(key)},
                    ),
                )
                local_artifacts[str(key)] = local_art
                local_state_payloads[str(key)] = dict(local_state)
                local_counts[str(key)] = int(idx.size)
                if inner_runtime_dispatcher is not None and inner_runtime_dispatcher.enabled:
                    inner_runtime_dispatcher.emit_round_end(
                        InnerRuntimeRoundPayload(
                            run_id=str(piecewise_runtime_context.get("run_id", piecewise_runtime_context.get("task_id", "symbolic_interval"))),
                            runtime_key=str(piecewise_runtime_context.get("runtime_key", "symbolic_interval_piecewise")),
                            trainer_name=str(piecewise_runtime_context.get("trainer_name", self.name)),
                            round_index=int(piecewise_round),
                            total_rounds=int(len(selected_keys)),
                            genome_size=int(len(local_state.get("genome", ()))),
                            history_entry={
                                "regime_key": str(key),
                                "status": "ok",
                                "train_size": int(idx.size),
                                "epoch_done": int(local_state.get("epoch_done", 0)),
                            },
                            context=piecewise_runtime_context,
                            metadata={"task": "interval_piecewise"},
                        )
                    )
            except Exception as exc:
                failed[str(key)] = f"{type(exc).__name__}: {exc}"
                if inner_runtime_dispatcher is not None and inner_runtime_dispatcher.enabled:
                    inner_runtime_dispatcher.emit_round_end(
                        InnerRuntimeRoundPayload(
                            run_id=str(piecewise_runtime_context.get("run_id", piecewise_runtime_context.get("task_id", "symbolic_interval"))),
                            runtime_key=str(piecewise_runtime_context.get("runtime_key", "symbolic_interval_piecewise")),
                            trainer_name=str(piecewise_runtime_context.get("trainer_name", self.name)),
                            round_index=int(piecewise_round),
                            total_rounds=int(len(selected_keys)),
                            genome_size=0,
                            history_entry={
                                "regime_key": str(key),
                                "status": "failed",
                                "train_size": int(idx.size),
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                            context=piecewise_runtime_context,
                            metadata={"task": "interval_piecewise"},
                        )
                    )

        gate_meta = {
            "enabled": True,
            "status": "ok" if len(local_artifacts) > 0 else "all_regimes_failed",
            "gate_feature_names": [str(v) for v in self.config.gate_feature_names],
            "gate_indices": [int(v) for v in gate_idx],
            "gate_threshold": float(self.config.gate_threshold),
            "gate_min_leaf": int(self.config.gate_min_leaf),
            "gate_max_local_models": int(self.config.gate_max_local_models),
            "blend_kappa": float(self.config.gate_blend_kappa),
            "counts_all": {str(k): int(v) for k, v in all_counts.items()},
            "counts_selected": {str(k): int(v) for k, v in local_counts.items()},
            "counts_skipped": {str(k): int(v) for k, v in skipped.items()},
            "failed": dict(failed),
        }

        if len(local_artifacts) == 0:
            global_artifact.metadata["gate_piecewise"] = dict(gate_meta)
            if inner_runtime_dispatcher is not None and inner_runtime_dispatcher.enabled:
                inner_runtime_dispatcher.emit_finish(
                    InnerRuntimeFinishPayload(
                        run_id=str(piecewise_runtime_context.get("run_id", piecewise_runtime_context.get("task_id", "symbolic_interval"))),
                        runtime_key=str(piecewise_runtime_context.get("runtime_key", "symbolic_interval_piecewise")),
                        trainer_name=str(piecewise_runtime_context.get("trainer_name", self.name)),
                        total_rounds=int(len(selected_keys)),
                        completed_rounds=int(piecewise_round),
                        genome_size=0,
                        final_metrics={"selected_local_models": 0, "failed_regimes": int(len(failed))},
                        context=piecewise_runtime_context,
                        metadata={"task": "interval_piecewise"},
                    )
                )
            signature_obj = coerce_training_signature(training_signature)
            top_payload = {
                "schema_version": 1,
                "trainer_name": str(self.name),
                "mode": "global",
                **dict(global_state_payload),
                "training_signature": None if training_signature is None else dict(training_signature),
            }
            trainer_state = TrainerState(
                trainer_name=str(self.name),
                payload=top_payload,
                schema_signature=signature_obj.schema_signature,
                feature_signature=signature_obj.feature_signature,
                target_signature=signature_obj.target_signature,
                objective_signature=signature_obj.objective_signature,
                pipeline_signature=signature_obj.pipeline_signature,
                numericizer_signature=signature_obj.numericizer_signature,
                regime_signature=signature_obj.regime_signature,
                symbolic_family_signature=signature_obj.symbolic_family_signature,
                metadata={"training_signature": signature_obj.as_dict(), "epoch_done": int(top_payload.get("epoch_done", 0))},
            )
            return global_artifact, trainer_state

        global_artifact.metadata["gate_piecewise"] = dict(gate_meta)
        final_state_path = None if checkpoint_root is None else (checkpoint_root / "trainer_state" / "latest.pt")
        aggregate_manifest = {
            "schema_version": 1,
            "kind": "piecewise_interval_aggregate_checkpoint",
            "mode": "piecewise",
            "checkpoint_root": None if checkpoint_root is None else str(checkpoint_root),
            "trainer_state_checkpoint": None if final_state_path is None else str(final_state_path),
            "global_checkpoint": (
                None
                if checkpoint_root is None
                else str((checkpoint_root / "global" / "latest.pt").resolve())
            ),
            "gate_feature_names": [str(v) for v in self.config.gate_feature_names],
            "gate_indices": [int(v) for v in gate_idx],
            "gate_threshold": float(self.config.gate_threshold),
            "gate_min_leaf": int(self.config.gate_min_leaf),
            "gate_max_local_models": int(self.config.gate_max_local_models),
            "blend_kappa": float(self.config.gate_blend_kappa),
            "selected_regime_keys": [str(k) for k in sorted(local_state_payloads.keys())],
            "failed_regimes": dict(failed),
            "counts_all": {str(k): int(v) for k, v in all_counts.items()},
            "counts_selected": {str(k): int(v) for k, v in local_counts.items()},
            "counts_skipped": {str(k): int(v) for k, v in skipped.items()},
            "local_regimes": {
                str(key): {
                    "count": int(local_counts.get(str(key), 0)),
                    "epoch_done": int(dict(local_state_payloads.get(str(key), {})).get("epoch_done", 0)),
                    "state_mode": str(dict(local_state_payloads.get(str(key), {})).get("mode", "global")),
                    "checkpoint": (
                        None
                        if checkpoint_root is None
                        else str(
                            (
                                checkpoint_root
                                / "local"
                                / f"regime_{str(key).replace('|', '_')}"
                                / "latest.pt"
                            ).resolve()
                        )
                    ),
                }
                for key in sorted(local_state_payloads.keys())
            },
            "training_signature": None if training_signature is None else dict(training_signature),
        }
        if checkpoint_root is not None:
            aggregate_manifest_path = (checkpoint_root / "aggregate_manifest.json").resolve()
            aggregate_manifest["aggregate_manifest_path"] = str(aggregate_manifest_path)
            self._write_aggregate_manifest(aggregate_manifest_path, aggregate_manifest)

        aggregate_selected_basis: dict[str, list[dict[str, Any]]] = {}
        for group_name, rows in _metadata_basis_groups(global_artifact.metadata.get("selected_basis")).items():
            aggregate_selected_basis[f"global:{str(group_name)}"] = list(rows)
        for key, local_artifact in sorted(local_artifacts.items(), key=lambda item: item[0]):
            local_groups = _metadata_basis_groups(local_artifact.metadata.get("selected_basis"))
            for group_name, rows in local_groups.items():
                aggregate_selected_basis[f"regime:{str(key)}:{str(group_name)}"] = list(rows)
        piecewise_basis_semantics = build_basis_semantics_payload(
            aggregate_selected_basis,
            source="symbolic_torch_interval.piecewise_aggregate",
            basis_scope="global+local",
            extra={
                "task": "interval_piecewise",
                "parameter_backend": "torch",
                "selected_regime_keys": [str(k) for k in sorted(local_artifacts.keys())],
                "head_groups": ["low", "high"],
            },
        )
        piecewise_basis_overlap_report = build_basis_overlap_report(
            aggregate_selected_basis,
            source="symbolic_torch_interval.piecewise_aggregate",
            extra={
                "task": "interval_piecewise",
                "parameter_backend": "torch",
                "selected_regime_keys": [str(k) for k in sorted(local_artifacts.keys())],
            },
        )
        piecewise_output_expression_count = _count_expression_strings(global_artifact.expressions())
        for local_artifact in local_artifacts.values():
            piecewise_output_expression_count += _count_expression_strings(local_artifact.expressions())
        piecewise_assembler_budget = build_assembler_budget_payload(
            source="symbolic_torch_interval.piecewise_aggregate",
            assembler_mode="piecewise_budgeted_symbolic_regression",
            output_expression_count=int(piecewise_output_expression_count),
            selected_basis_count=int(sum(len(rows) for rows in aggregate_selected_basis.values())),
            budget_axes={
                "epochs": int(self.config.epochs),
                "batch_size": int(max(1, self.config.batch_size)),
                "global_models": 1,
                "local_models": int(len(local_artifacts)),
                "gate_max_local_models": int(self.config.gate_max_local_models),
            },
            uses_piecewise_gate=True,
            extra={
                "failed_regime_count": int(len(failed)),
                "selected_regime_count": int(len(local_artifacts)),
            },
        )
        piecewise_metadata = {
            "trainer": "SymbolicTorchIntervalTrainer",
            "gate_piecewise": dict(gate_meta),
            "aggregate_manifest": dict(aggregate_manifest),
            "data_metadata": dict(base_data_meta),
            "training_init": {"mode": mode, "parent_source": parent_source},
            "selected_basis": {str(key): list(rows) for key, rows in aggregate_selected_basis.items()},
            "basis_semantics": dict(piecewise_basis_semantics),
            "basis_overlap_report": dict(piecewise_basis_overlap_report),
            "assembler_budget": dict(piecewise_assembler_budget),
            "symbolic": {
                "selected_basis": {str(key): list(rows) for key, rows in aggregate_selected_basis.items()},
                "basis_semantics": dict(piecewise_basis_semantics),
                "basis_overlap_report": dict(piecewise_basis_overlap_report),
                "assembler_budget": dict(piecewise_assembler_budget),
                "structure_engine": self._resolve_structure_engine().as_dict(),
                "selected_regime_keys": [str(k) for k in sorted(local_artifacts.keys())],
            },
        }
        family_spec = getattr(self, "symbolic_family_spec", None)
        if family_spec is not None and hasattr(family_spec, "as_dict"):
            piecewise_metadata["symbolic_family"] = family_spec.as_dict()

        piecewise_artifact = PiecewiseSymbolicIntervalSurrogateArtifact(
            artifact_id=f"{self.config.artifact_id}_piecewise",
            global_artifact=global_artifact,
            local_artifacts=local_artifacts,
            gate_feature_names=tuple(str(v) for v in self.config.gate_feature_names),
            blend_kappa=float(self.config.gate_blend_kappa),
            regime_counts={str(k): int(v) for k, v in local_counts.items()},
            feature_names=tuple(feature_names),
            target_names=tuple(target_names),
            pipeline_name=str(getattr(self.pipeline, "name", "identity")),
            pipeline_state=dict(getattr(self.pipeline, "state_dict")()),
            ood_z_threshold=float(self.config.ood_z_threshold),
            metadata=piecewise_metadata,
        )
        if inner_runtime_dispatcher is not None and inner_runtime_dispatcher.enabled:
            inner_runtime_dispatcher.emit_finish(
                InnerRuntimeFinishPayload(
                    run_id=str(piecewise_runtime_context.get("run_id", piecewise_runtime_context.get("task_id", "symbolic_interval"))),
                    runtime_key=str(piecewise_runtime_context.get("runtime_key", "symbolic_interval_piecewise")),
                    trainer_name=str(piecewise_runtime_context.get("trainer_name", self.name)),
                    total_rounds=int(len(selected_keys)),
                    completed_rounds=int(piecewise_round),
                    genome_size=int(len(local_artifacts)),
                    final_metrics={
                        "selected_local_models": int(len(local_artifacts)),
                        "failed_regimes": int(len(failed)),
                    },
                    context=piecewise_runtime_context,
                    metadata={"task": "interval_piecewise"},
                )
            )
        signature_obj = coerce_training_signature(training_signature)
        top_payload = {
            "schema_version": 1,
            "trainer_name": str(self.name),
            "mode": "piecewise",
            "global_state": dict(global_state_payload),
            "local_states": {str(k): dict(v) for k, v in local_state_payloads.items()},
            "gate_piecewise": dict(gate_meta),
            "aggregate_manifest": dict(aggregate_manifest),
            "training_signature": None if training_signature is None else dict(training_signature),
        }
        trainer_state = TrainerState(
            trainer_name=str(self.name),
            payload=top_payload,
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
                "epoch_done": int(global_state_payload.get("epoch_done", 0)),
                "training_signature": signature_obj.as_dict(),
            },
        )
        if checkpoint_root is not None and final_state_path is not None:
            self.save_trainer_state(final_state_path, trainer_state)
            piecewise_artifact.metadata.setdefault("trainer_state_checkpoint", str(final_state_path))
        return piecewise_artifact, trainer_state

    def fit(
        self,
        data: ProcessedDataset | SampleDataset,
    ) -> SymbolicIntervalSurrogateArtifact | PiecewiseSymbolicIntervalSurrogateArtifact:
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


__all__ = [
    "SymbolicTorchIntervalTrainerConfig",
    "SymbolicTorchIntervalTrainer",
]
