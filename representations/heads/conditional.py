from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from blackbase.contracts import ComponentContract
from mlblack.core.head import BaseDecoder, HeadBlock, OutputHead
from mlblack.models import PiecewiseModel, Router, ThresholdRouter


@dataclass(frozen=True)
class PiecewiseHead(OutputHead):
    """Decoder-side branch composition head.

    It allocates one base-parameter block per branch and returns a PiecewiseModel.
    """

    n_branches: int = 2
    router: Router | None = None
    default_branch: int = 0

    name = "piecewise"
    output_kind = "piecewise"
    context_requires = ('base_decoder', 'candidate.unknown_state', 'router')
    context_optional = ()
    context_provides = ('candidate.model', 'model.predict', 'model.route')
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: base_decoder, candidate.unknown_state, router; provides candidate.model, model.predict, model.route.'
    contract = ComponentContract(
        name=name,
        requires=("base_decoder", "candidate.unknown_state", "router"),
        provides=("candidate.model", "model.predict", "model.route"),
        supports_batch=True,
        supports_resume=True,
        metadata={"head": "piecewise"},
    )

    def parameter_size(self, base_dimension: int) -> int:
        return max(1, int(self.n_branches)) * int(base_dimension)

    def blocks(self, base_dimension: int) -> tuple[HeadBlock, ...]:
        dim = int(base_dimension)
        return tuple(HeadBlock(f"branch_{idx}", idx * dim, (idx + 1) * dim) for idx in range(max(1, int(self.n_branches))))

    def decode(
        self,
        values: np.ndarray,
        *,
        base_dimension: int,
        base_decode: BaseDecoder,
        context: Mapping[str, Any] | None = None,
    ) -> PiecewiseModel:
        ctx = dict(context or {})
        arr = self.repair_values(values, base_dimension=base_dimension)
        branch_models = [base_decode(block.values(arr), {**ctx, "head.block": block.name}) for block in self.blocks(base_dimension)]
        router = self.router or ThresholdRouter(feature_index=0, thresholds=(0.0,))
        return PiecewiseModel(
            router=router,
            branch_models=tuple(branch_models),
            default_branch=int(self.default_branch),
            metadata={"head": self.name, "n_branches": len(branch_models)},
        )
