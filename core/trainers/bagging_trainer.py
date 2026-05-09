from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from bias import BaseTrainingBias
from numericizer import BaseNumericizer, ModalityEncoder, TargetCodec
from pipeline import BasePipeline

from core.tree.trainer_family import TreeTrainerFamilySpec, build_bagging_family_spec
from .tree_ensemble_trainer import SklearnTreeEnsembleSurrogateTrainer, TreeEnsembleTrainerConfig


@dataclass(frozen=True)
class BaggingTrainerConfig(TreeEnsembleTrainerConfig):
    artifact_id: str = "bagging_surrogate_v1"
    family_spec: TreeTrainerFamilySpec = field(
        default_factory=lambda: build_bagging_family_spec(trainer_key="bagging")
    )


class BaggingSurrogateTrainer(SklearnTreeEnsembleSurrogateTrainer):
    name = "bagging"

    def __init__(
        self,
        config: BaggingTrainerConfig | None = None,
        *,
        pipeline: BasePipeline | None = None,
        biases: Sequence[BaseTrainingBias] | None = None,
        numericizer: BaseNumericizer | None = None,
        modality_encoders: Mapping[str, ModalityEncoder] | None = None,
        target_codecs: Mapping[str, TargetCodec] | None = None,
        target_codec: str | None = None,
        categorical_unknown: str = "error",
    ) -> None:
        super().__init__(
            config=config or BaggingTrainerConfig(),
            pipeline=pipeline,
            biases=biases,
            numericizer=numericizer,
            modality_encoders=modality_encoders,
            target_codecs=target_codecs,
            target_codec=target_codec,
            categorical_unknown=categorical_unknown,
        )


__all__ = [
    "BaggingTrainerConfig",
    "BaggingSurrogateTrainer",
]
