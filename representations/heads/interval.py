from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from mlblack.core.contracts import ComponentContract
from mlblack.core.head import BaseDecoder, HeadBlock, OutputHead
from mlblack.models import CenterRadiusIntervalModel, IntervalPredictionModel


@dataclass(frozen=True)
class TwoModelIntervalHead(OutputHead):
    """Interval head: lower base model + upper base model."""

    enforce_order: bool = True

    name = "interval_two_model"
    output_kind = "interval"
    context_requires = ('base_decoder', 'candidate.unknown_state')
    context_optional = ()
    context_provides = ('candidate.interval_model', 'model.predict_interval')
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: base_decoder, candidate.unknown_state; provides candidate.interval_model, model.predict_interval.'
    contract = ComponentContract(
        name=name,
        requires=("base_decoder", "candidate.unknown_state"),
        provides=("candidate.interval_model", "model.predict_interval"),
        supports_batch=True,
        supports_resume=True,
        metadata={"head": "interval", "mode": "two_model"},
    )

    def parameter_size(self, base_dimension: int) -> int:
        return 2 * int(base_dimension)

    def blocks(self, base_dimension: int) -> tuple[HeadBlock, ...]:
        dim = int(base_dimension)
        return (
            HeadBlock("lower", 0, dim),
            HeadBlock("upper", dim, 2 * dim),
        )

    def decode(
        self,
        values: np.ndarray,
        *,
        base_dimension: int,
        base_decode: BaseDecoder,
        context: Mapping[str, Any] | None = None,
    ) -> IntervalPredictionModel:
        ctx = dict(context or {})
        arr = self.repair_values(values, base_dimension=base_dimension)
        lower_block, upper_block = self.blocks(base_dimension)
        lower_model = base_decode(lower_block.values(arr), {**ctx, "head.block": "lower"})
        upper_model = base_decode(upper_block.values(arr), {**ctx, "head.block": "upper"})
        return IntervalPredictionModel(
            lower_model=lower_model,
            upper_model=upper_model,
            enforce_order=bool(self.enforce_order),
            metadata={"head": self.name, "mode": "two_model"},
        )


@dataclass(frozen=True)
class CenterRadiusIntervalHead(OutputHead):
    """Interval head: center base model + positive radius base model."""

    radius_transform: str = "softplus"

    name = "interval_center_radius"
    output_kind = "interval"
    context_requires = ('base_decoder', 'candidate.unknown_state')
    context_optional = ()
    context_provides = ('candidate.interval_model', 'model.predict_interval')
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: base_decoder, candidate.unknown_state; provides candidate.interval_model, model.predict_interval.'
    contract = ComponentContract(
        name=name,
        requires=("base_decoder", "candidate.unknown_state"),
        provides=("candidate.interval_model", "model.predict_interval"),
        supports_batch=True,
        supports_resume=True,
        metadata={"head": "interval", "mode": "center_radius"},
    )

    def parameter_size(self, base_dimension: int) -> int:
        return 2 * int(base_dimension)

    def blocks(self, base_dimension: int) -> tuple[HeadBlock, ...]:
        dim = int(base_dimension)
        return (
            HeadBlock("center", 0, dim),
            HeadBlock("radius", dim, 2 * dim),
        )

    def decode(
        self,
        values: np.ndarray,
        *,
        base_dimension: int,
        base_decode: BaseDecoder,
        context: Mapping[str, Any] | None = None,
    ) -> CenterRadiusIntervalModel:
        ctx = dict(context or {})
        arr = self.repair_values(values, base_dimension=base_dimension)
        center_block, radius_block = self.blocks(base_dimension)
        center_model = base_decode(center_block.values(arr), {**ctx, "head.block": "center"})
        radius_model = base_decode(radius_block.values(arr), {**ctx, "head.block": "radius"})
        return CenterRadiusIntervalModel(
            center_model=center_model,
            radius_model=radius_model,
            radius_transform=str(self.radius_transform),
            metadata={"head": self.name, "mode": "center_radius"},
        )


IntervalHead = TwoModelIntervalHead

