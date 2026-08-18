from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from blackbase.contracts import ComponentContract
from mlblack.core.head import BaseDecoder, HeadBlock, OutputHead
from mlblack.models import BinaryLogisticProbabilityModel, SoftmaxProbabilityModel, TemperatureCalibratedProbabilityModel


@dataclass(frozen=True)
class BinaryLogisticHead(OutputHead):
    """Binary probability head: one base logit model -> sigmoid probabilities."""

    temperature: float = 1.0
    threshold: float = 0.5
    classes: Sequence[Any] = (0, 1)

    name = "binary_logistic"
    output_kind = "probability"
    context_requires = ('base_decoder', 'candidate.unknown_state')
    context_optional = ()
    context_provides = ('candidate.probability_model', 'model.predict_proba', 'model.predict')
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: base_decoder, candidate.unknown_state; provides candidate.probability_model, model.predict_proba, model.predict.'
    contract = ComponentContract(
        name=name,
        requires=("base_decoder", "candidate.unknown_state"),
        provides=("candidate.probability_model", "model.predict_proba", "model.predict"),
        supports_batch=True,
        supports_resume=True,
        metadata={"head": "probability", "mode": "binary_logistic"},
    )

    def parameter_size(self, base_dimension: int) -> int:
        return int(base_dimension)

    def blocks(self, base_dimension: int) -> tuple[HeadBlock, ...]:
        return (HeadBlock("logit", 0, int(base_dimension)),)

    def decode(
        self,
        values: np.ndarray,
        *,
        base_dimension: int,
        base_decode: BaseDecoder,
        context: Mapping[str, Any] | None = None,
    ) -> BinaryLogisticProbabilityModel:
        arr = self.repair_values(values, base_dimension=base_dimension)
        logit_model = base_decode(arr[: int(base_dimension)], {**dict(context or {}), "head.block": "logit"})
        return BinaryLogisticProbabilityModel(
            logit_model=logit_model,
            temperature=float(self.temperature),
            threshold=float(self.threshold),
            classes_=tuple(self.classes),
            metadata={"head": self.name},
        )


@dataclass(frozen=True)
class SoftmaxHead(OutputHead):
    """Multiclass probability head: one base logit model per class."""

    n_classes: int = 2
    temperature: float = 1.0
    classes: Sequence[Any] = tuple()

    name = "softmax"
    output_kind = "probability"
    context_requires = ('base_decoder', 'candidate.unknown_state')
    context_optional = ()
    context_provides = ('candidate.probability_model', 'model.predict_proba', 'model.predict')
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: base_decoder, candidate.unknown_state; provides candidate.probability_model, model.predict_proba, model.predict.'
    contract = ComponentContract(
        name=name,
        requires=("base_decoder", "candidate.unknown_state"),
        provides=("candidate.probability_model", "model.predict_proba", "model.predict"),
        supports_batch=True,
        supports_resume=True,
        metadata={"head": "probability", "mode": "softmax"},
    )

    def parameter_size(self, base_dimension: int) -> int:
        return int(base_dimension) * max(2, int(self.n_classes))

    def blocks(self, base_dimension: int) -> tuple[HeadBlock, ...]:
        dim = int(base_dimension)
        return tuple(HeadBlock(f"class_{idx}", idx * dim, (idx + 1) * dim) for idx in range(max(2, int(self.n_classes))))

    def decode(
        self,
        values: np.ndarray,
        *,
        base_dimension: int,
        base_decode: BaseDecoder,
        context: Mapping[str, Any] | None = None,
    ) -> SoftmaxProbabilityModel:
        ctx = dict(context or {})
        arr = self.repair_values(values, base_dimension=base_dimension)
        models = []
        for block in self.blocks(base_dimension):
            models.append(base_decode(block.values(arr), {**ctx, "head.block": block.name}))
        return SoftmaxProbabilityModel(
            logit_models=tuple(models),
            temperature=float(self.temperature),
            classes_=tuple(self.classes) or tuple(range(len(models))),
            metadata={"head": self.name},
        )


@dataclass(frozen=True)
class ProbabilityCalibrationHead(OutputHead):
    """Base probability/logit model plus one temperature parameter."""

    min_temperature: float = 1e-3
    max_temperature: float = 100.0

    name = "probability_calibration"
    output_kind = "probability"
    context_requires = ('base_decoder', 'candidate.unknown_state')
    context_optional = ()
    context_provides = ('candidate.probability_model', 'model.predict_proba', 'model.predict')
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: base_decoder, candidate.unknown_state; provides candidate.probability_model, model.predict_proba, model.predict.'
    contract = ComponentContract(
        name=name,
        requires=("base_decoder", "candidate.unknown_state"),
        provides=("candidate.probability_model", "model.predict_proba", "model.predict"),
        supports_batch=True,
        supports_resume=True,
        metadata={"head": "probability", "mode": "temperature_calibration"},
    )

    def parameter_size(self, base_dimension: int) -> int:
        return int(base_dimension) + 1

    def blocks(self, base_dimension: int) -> tuple[HeadBlock, ...]:
        dim = int(base_dimension)
        return (HeadBlock("base", 0, dim), HeadBlock("temperature", dim, dim + 1))

    def decode(
        self,
        values: np.ndarray,
        *,
        base_dimension: int,
        base_decode: BaseDecoder,
        context: Mapping[str, Any] | None = None,
    ) -> TemperatureCalibratedProbabilityModel:
        ctx = dict(context or {})
        arr = self.repair_values(values, base_dimension=base_dimension)
        base_block, temp_block = self.blocks(base_dimension)
        base_model = base_decode(base_block.values(arr), {**ctx, "head.block": "base"})
        raw_temp = float(temp_block.values(arr)[0])
        temperature = float(np.log1p(np.exp(raw_temp)) + float(self.min_temperature))
        temperature = float(np.clip(temperature, float(self.min_temperature), float(self.max_temperature)))
        return TemperatureCalibratedProbabilityModel(
            base_model=base_model,
            temperature=temperature,
            metadata={"head": self.name, "raw_temperature": raw_temp},
        )
