from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mlblack.core.contracts import ComponentContract
from mlblack.core.head import BaseDecoder, HeadBlock, OutputHead


class PointHead(OutputHead):
    """Point-output head: one base model, direct predict output."""

    name = "point"
    output_kind = "point"
    context_requires = ('base_decoder', 'candidate.unknown_state')
    context_optional = ()
    context_provides = ('candidate.model', 'model.predict')
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: base_decoder, candidate.unknown_state; provides candidate.model, model.predict.'
    contract = ComponentContract(
        name=name,
        requires=("base_decoder", "candidate.unknown_state"),
        provides=("candidate.model", "model.predict"),
        supports_batch=True,
        supports_resume=True,
        metadata={"head": "point"},
    )

    def parameter_size(self, base_dimension: int) -> int:
        return int(base_dimension)

    def blocks(self, base_dimension: int) -> tuple[HeadBlock, ...]:
        return (HeadBlock("point", 0, int(base_dimension)),)

    def decode(
        self,
        values: np.ndarray,
        *,
        base_dimension: int,
        base_decode: BaseDecoder,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        arr = self.repair_values(values, base_dimension=base_dimension)
        return base_decode(arr[: int(base_dimension)], dict(context or {}))

