from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from bias import BaseTrainingBias, NoOpBias
from core.artifacts.symbolic_interval_artifact import SymbolicIntervalSurrogateArtifact
from core.common.base_trainer import BaseSurrogateTrainer
from core.common.contracts import ProcessedDataset, SampleDataset
from core.execution import ExecutionResourceRequest
from core.trainers.symbolic_orthogonal_trainer import (
    SymbolicOrthogonalSurrogateTrainer,
    SymbolicOrthogonalTrainerConfig,
)
from core.trainers.symbolic_torch_interval_trainer import (
    SymbolicTorchIntervalTrainer,
    SymbolicTorchIntervalTrainerConfig,
)
from numericizer import BaseNumericizer, DefaultNumericizer, ModalityEncoder, TargetCodec
from pipeline import BasePipeline, IdentityPipeline


@dataclass(frozen=True)
class SymbolicOrthogonalIntervalTrainerConfig:
    """Discover an orthogonal symbolic basis, then train native quantile heads on it.

    This is intentionally a composition layer:
    1. `SymbolicOrthogonalSurrogateTrainer` decides the symbolic genome/basis objects.
    2. `SymbolicTorchIntervalTrainer` receives that genome as an explicit basis and
       trains the lower/upper quantile heads.
    """

    artifact_id: str = "symbolic_orthogonal_interval_surrogate_v1"
    orthogonal_config: SymbolicOrthogonalTrainerConfig = field(default_factory=SymbolicOrthogonalTrainerConfig)
    interval_config: SymbolicTorchIntervalTrainerConfig = field(default_factory=SymbolicTorchIntervalTrainerConfig)
    basis_artifact_id_suffix: str = "_basis_discovery"
    metadata_protocol: str = "OrthogonalBasisToNativeQuantileHead"


class SymbolicOrthogonalIntervalSurrogateTrainer(BaseSurrogateTrainer):
    name = "symbolic_orthogonal_interval"

    def __init__(
        self,
        config: SymbolicOrthogonalIntervalTrainerConfig | None = None,
        *,
        pipeline: BasePipeline | None = None,
        biases: Sequence[BaseTrainingBias] | None = None,
        numericizer: BaseNumericizer | None = None,
        modality_encoders: Mapping[str, ModalityEncoder] | None = None,
        target_codecs: Mapping[str, TargetCodec] | None = None,
        target_codec: str | None = None,
        categorical_unknown: str = "error",
    ) -> None:
        self.config = config or SymbolicOrthogonalIntervalTrainerConfig()
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
            "supports_resume": False,
            "supports_warm_start": False,
            "supports_incremental": False,
            "supports_recalibration": False,
            "name": self.name,
            "model_family": "symbolic_interval",
            "backend": "orthogonal_basis+pytorch_quantile",
            "nonlinear": True,
            "supports": {
                "processed_dataset": True,
                "sample_dataset": True,
                "interval_output": True,
                "orthogonal_basis_discovery": True,
                "native_quantile_heads": True,
                "explicit_genome_binding": True,
            },
            "artifacts": {
                "type": "SymbolicIntervalSurrogateArtifact",
                "predict": "center",
                "predict_interval": True,
                "expression_export": True,
            },
        }

    def execution_resource_request(self) -> ExecutionResourceRequest:
        return ExecutionResourceRequest(
            threads=1,
            backend="serial",
            label=str(self.name),
            metadata={
                "planes": ("orthogonal_basis_discovery", "native_quantile_interval_head"),
            },
        )

    @staticmethod
    def _copy_genome(genome: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
        return tuple(dict(term) for term in tuple(genome) if isinstance(term, Mapping))

    @staticmethod
    def _attach_orthogonal_metadata(
        artifact: SymbolicIntervalSurrogateArtifact,
        *,
        basis_artifact: Any,
        protocol: str,
    ) -> None:
        basis_meta = dict(getattr(basis_artifact, "metadata", {}) or {})
        basis_schema = dict(basis_meta.get("symbolic_artifact_schema", {}) or {})
        basis_symbolic = dict(basis_meta.get("symbolic", {}) or {})

        artifact.metadata["orthogonal_interval_protocol"] = str(protocol)
        artifact.metadata["orthogonal_basis_artifact_id"] = str(getattr(basis_artifact, "artifact_id", ""))
        artifact.metadata["orthogonal_basis_genome"] = [
            dict(term) for term in tuple(getattr(basis_artifact, "genome", ()) or ()) if isinstance(term, Mapping)
        ]
        artifact.metadata["orthogonal_basis_metadata"] = {
            "train_metrics": dict(basis_meta.get("train_metrics", {}) or {}),
            "search": dict(basis_meta.get("search", {}) or {}),
            "structure_head": basis_meta.get("structure_head"),
            "prediction_head": basis_meta.get("prediction_head"),
            "search_input_space": basis_meta.get("search_input_space"),
            "basis_binding_mode": basis_meta.get("basis_binding_mode"),
            "escape_policy": basis_meta.get("escape_policy"),
        }
        if basis_schema:
            artifact.metadata["orthogonal_basis_artifact_schema"] = dict(basis_schema)
        if basis_symbolic:
            artifact.metadata["orthogonal_basis_symbolic"] = dict(basis_symbolic)

        symbolic = dict(artifact.metadata.get("symbolic", {}) or {})
        for key in (
            "basis_context",
            "basis_semantics",
            "basis_overlap_report",
            "residual_complementarity_report",
            "semantic_dedup_report",
            "orthogonal_outer_basis_genome",
            "inner_symbolic_search",
            "orthogonal_search_objective",
            "basis_object_gradient_pool",
        ):
            value = basis_meta.get(key, basis_symbolic.get(key))
            if value is not None:
                symbolic[key] = value
                symbolic[f"orthogonal_{key}"] = value
        symbolic["orthogonal_interval_protocol"] = str(protocol)
        artifact.metadata["symbolic"] = symbolic

    def fit(self, data: ProcessedDataset | SampleDataset) -> SymbolicIntervalSurrogateArtifact:
        basis_artifact_id = str(self.config.artifact_id) + str(self.config.basis_artifact_id_suffix)
        orthogonal_config = replace(
            self.config.orthogonal_config,
            artifact_id=basis_artifact_id,
        )
        basis_trainer = SymbolicOrthogonalSurrogateTrainer(
            config=orthogonal_config,
            pipeline=self.pipeline,
            biases=self.biases,
            numericizer=self.numericizer,
        )
        basis_artifact = basis_trainer.fit(data)
        genome = self._copy_genome(tuple(getattr(basis_artifact, "genome", ()) or ()))
        if not genome:
            raise RuntimeError("Orthogonal basis discovery produced an empty genome; cannot train interval heads.")

        interval_config = replace(
            self.config.interval_config,
            artifact_id=str(self.config.artifact_id),
            genome=genome,
        )
        interval_trainer = SymbolicTorchIntervalTrainer(
            config=interval_config,
            pipeline=self.pipeline,
            biases=self.biases,
            numericizer=self.numericizer,
        )
        interval_artifact = interval_trainer.fit(data)
        self._attach_orthogonal_metadata(
            interval_artifact,
            basis_artifact=basis_artifact,
            protocol=str(self.config.metadata_protocol),
        )
        return interval_artifact


__all__ = [
    "SymbolicOrthogonalIntervalTrainerConfig",
    "SymbolicOrthogonalIntervalSurrogateTrainer",
]
