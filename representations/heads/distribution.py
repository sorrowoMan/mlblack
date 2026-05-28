from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from mlblack.core.contracts import ComponentContract
from mlblack.core.head import BaseDecoder, HeadBlock, OutputHead
from mlblack.models import NormalDistributionModel, PoissonDistributionModel, NegativeBinomialDistributionModel


@dataclass(frozen=True)
class NormalHead(OutputHead):
    sigma_transform: str = "softplus"

    name = "normal_distribution"
    output_kind = "distribution"
    context_requires = ('base_decoder', 'candidate.unknown_state')
    context_optional = ()
    context_provides = ('candidate.distribution_model', 'model.predict', 'model.predict_params')
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: base_decoder, candidate.unknown_state; provides candidate.distribution_model, model.predict, model.predict_params.'
    contract = ComponentContract(
        name=name,
        requires=("base_decoder", "candidate.unknown_state"),
        provides=("candidate.distribution_model", "model.predict", "model.predict_params"),
        supports_batch=True,
        supports_resume=True,
        metadata={"head": "distribution", "family": "normal"},
    )

    def parameter_size(self, base_dimension: int) -> int:
        return 2 * int(base_dimension)

    def blocks(self, base_dimension: int) -> tuple[HeadBlock, ...]:
        dim = int(base_dimension)
        return (
            HeadBlock("mu", 0, dim),
            HeadBlock("log_sigma", dim, 2 * dim),
        )

    def decode(
        self,
        values: np.ndarray,
        *,
        base_dimension: int,
        base_decode: BaseDecoder,
        context: Mapping[str, Any] | None = None,
    ) -> NormalDistributionModel:
        ctx = dict(context or {})
        arr = self.repair_values(values, base_dimension=base_dimension)
        mu_block, sigma_block = self.blocks(base_dimension)
        mu_model = base_decode(mu_block.values(arr), {**ctx, "head.block": "mu"})
        sigma_model = base_decode(sigma_block.values(arr), {**ctx, "head.block": "log_sigma"})
        return NormalDistributionModel(
            mu_model=mu_model,
            sigma_model=sigma_model,
            sigma_transform=str(self.sigma_transform),
            metadata={"head": self.name, "family": "normal"},
        )


@dataclass(frozen=True)
class PoissonHead(OutputHead):
    rate_transform: str = "softplus"

    name = "poisson_distribution"
    output_kind = "distribution"
    context_requires = ('base_decoder', 'candidate.unknown_state')
    context_optional = ()
    context_provides = ('candidate.distribution_model', 'model.predict', 'model.predict_params')
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: base_decoder, candidate.unknown_state; provides candidate.distribution_model, model.predict, model.predict_params.'
    contract = ComponentContract(
        name=name,
        requires=("base_decoder", "candidate.unknown_state"),
        provides=("candidate.distribution_model", "model.predict", "model.predict_params"),
        supports_batch=True,
        supports_resume=True,
        metadata={"head": "distribution", "family": "poisson"},
    )

    def parameter_size(self, base_dimension: int) -> int:
        return int(base_dimension)

    def blocks(self, base_dimension: int) -> tuple[HeadBlock, ...]:
        dim = int(base_dimension)
        return (HeadBlock("log_rate", 0, dim),)

    def decode(
        self,
        values: np.ndarray,
        *,
        base_dimension: int,
        base_decode: BaseDecoder,
        context: Mapping[str, Any] | None = None,
    ) -> PoissonDistributionModel:
        ctx = dict(context or {})
        arr = self.repair_values(values, base_dimension=base_dimension)
        block = self.blocks(base_dimension)[0]
        rate_model = base_decode(block.values(arr), {**ctx, "head.block": "log_rate"})
        return PoissonDistributionModel(
            rate_model=rate_model,
            rate_transform=str(self.rate_transform),
            metadata={"head": self.name, "family": "poisson"},
        )


@dataclass(frozen=True)
class NegativeBinomialHead(OutputHead):
    alpha_transform: str = "softplus"

    name = "negative_binomial_distribution"
    output_kind = "distribution"
    context_requires = ('base_decoder', 'candidate.unknown_state')
    context_optional = ()
    context_provides = ('candidate.distribution_model', 'model.predict', 'model.predict_params')
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: base_decoder, candidate.unknown_state; provides candidate.distribution_model, model.predict, model.predict_params.'
    contract = ComponentContract(
        name=name,
        requires=("base_decoder", "candidate.unknown_state"),
        provides=("candidate.distribution_model", "model.predict", "model.predict_params"),
        supports_batch=True,
        supports_resume=True,
        metadata={"head": "distribution", "family": "negative_binomial"},
    )

    def parameter_size(self, base_dimension: int) -> int:
        return 2 * int(base_dimension)

    def blocks(self, base_dimension: int) -> tuple[HeadBlock, ...]:
        dim = int(base_dimension)
        return (
            HeadBlock("log_mu", 0, dim),
            HeadBlock("log_alpha", dim, 2 * dim),
        )

    def decode(
        self,
        values: np.ndarray,
        *,
        base_dimension: int,
        base_decode: BaseDecoder,
        context: Mapping[str, Any] | None = None,
    ) -> NegativeBinomialDistributionModel:
        ctx = dict(context or {})
        arr = self.repair_values(values, base_dimension=base_dimension)
        mu_block, alpha_block = self.blocks(base_dimension)
        mu_model = base_decode(mu_block.values(arr), {**ctx, "head.block": "log_mu"})
        alpha_model = base_decode(alpha_block.values(arr), {**ctx, "head.block": "log_alpha"})
        return NegativeBinomialDistributionModel(
            mu_model=mu_model,
            alpha_model=alpha_model,
            alpha_transform=str(self.alpha_transform),
            metadata={"head": self.name, "family": "negative_binomial"},
        )
