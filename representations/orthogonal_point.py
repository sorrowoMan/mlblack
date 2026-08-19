from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from blackbase.contracts import ComponentContract
from mlblack.core.head import OutputHead
from mlblack.core.representation import ModelRepresentation
from mlblack.core.types import UnknownState
from mlblack.representations.heads import PointHead
from mlblack.models import OrthogonalFeatureMap, OrthogonalLinearPointModel
from mlblack.representations.codecs import OrthogonalLinearCodecConfig, OrthogonalLinearPointCodec


@dataclass(frozen=True)
class OrthogonalPointConfig:
    include_raw: bool = True
    include_square: bool = True
    include_interactions: bool = True
    max_components: int | None = None
    energy_threshold: float | None = 0.999
    init_scale: float = 0.01
    random_seed: int = 42


class OrthogonalPointLinearRepresentation(ModelRepresentation):
    """Unknown vector -> orthogonal linear model with pluggable output head.

    Point head values are [intercept, w0, w1, ...]. Interval heads allocate
    multiple orthogonal-linear base blocks.
    """

    name = "orthogonal_point_linear"
    context_requires = ('candidate.unknown_state', 'orthogonal_feature_map')
    context_optional = ()
    context_provides = ('candidate.output', 'model.predict')
    context_mutates = ('candidate.repaired_state',)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: candidate.unknown_state, orthogonal_feature_map; provides candidate.output, model.predict; mutates candidate.repaired_state.'
    contract = ComponentContract(
        name=name,
        requires=("candidate.unknown_state", "orthogonal_feature_map"),
        provides=("candidate.output", "model.predict"),
        mutates=("candidate.repaired_state",),
        supports_gradient=None,
        supports_batch=True,
        supports_resume=True,
        metadata={"family": "linear", "head": "point", "feature_space": "orthogonal"},
    )

    def __init__(
        self,
        feature_map: OrthogonalFeatureMap,
        config: OrthogonalPointConfig | None = None,
        head: OutputHead | None = None,
    ) -> None:
        self.feature_map = feature_map
        self.config = config or OrthogonalPointConfig()
        self.codec = OrthogonalLinearPointCodec(
            feature_map,
            OrthogonalLinearCodecConfig(
                include_raw=bool(self.config.include_raw),
                include_square=bool(self.config.include_square),
                include_interactions=bool(self.config.include_interactions),
                max_components=self.config.max_components,
                energy_threshold=self.config.energy_threshold,
                init_scale=float(self.config.init_scale),
                random_seed=int(self.config.random_seed),
                representation_name=self.name,
            ),
        )
        self.head = head or PointHead()
        self.base_dimension = int(self.codec.base_dimension)
        self.dimension = int(self.head.parameter_size(self.base_dimension))

    @classmethod
    def from_data(
        cls,
        X: np.ndarray,
        *,
        feature_names: tuple[str, ...] | None = None,
        config: OrthogonalPointConfig | None = None,
        head: OutputHead | None = None,
    ) -> "OrthogonalPointLinearRepresentation":
        cfg = config or OrthogonalPointConfig()
        fmap = OrthogonalFeatureMap.fit(
            X,
            feature_names=feature_names,
            include_raw=cfg.include_raw,
            include_square=cfg.include_square,
            include_interactions=cfg.include_interactions,
            max_components=cfg.max_components,
            energy_threshold=cfg.energy_threshold,
        )
        return cls(fmap, config=cfg, head=head)

    def init(self, context: Mapping[str, Any]) -> UnknownState:
        _ = context
        values = np.zeros(self.dimension, dtype=float)
        base_values = self.codec.init_values()
        values[: base_values.shape[0]] = base_values
        return UnknownState(values=np.asarray(values, dtype=float), metadata={"source": "orthogonal_point_init"})

    def encode(self, model: Any, context: Mapping[str, Any] | None = None) -> UnknownState:
        _ = context
        if self.head.output_kind != "point":
            raise NotImplementedError("encoding is only implemented for point head")
        if not isinstance(model, OrthogonalLinearPointModel):
            raise TypeError("OrthogonalPointLinearRepresentation can only encode OrthogonalLinearPointModel")
        values = self.codec.encode(model)
        return UnknownState(values=values, metadata={"source": "encoded_model"})

    def decode(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> Any:
        ctx = dict(context or {})
        arr = np.asarray(state.values, dtype=float).reshape(-1)
        if arr.shape[0] != self.dimension:
            raise ValueError(f"state dimension {arr.shape[0]} does not match representation dimension {self.dimension}")
        return self.head.decode(
            arr,
            base_dimension=self.base_dimension,
            base_decode=self._decode_base,
            context=ctx,
        )

    def _decode_base(self, values: np.ndarray, context: Mapping[str, Any] | None = None) -> OrthogonalLinearPointModel:
        ctx = dict(context or {})
        return self.codec.decode(values, ctx)

    def repair(self, state: UnknownState, context: Mapping[str, Any] | None = None) -> UnknownState:
        _ = context
        arr = self.head.repair_values(state.values, base_dimension=self.base_dimension)
        return state.with_values(arr)

    def get_contract(self) -> ComponentContract:
        return self.contract.merged(self.head.get_contract(), name=self.name)

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "dimension": int(self.dimension),
            "base_dimension": int(self.base_dimension),
            "codec": self.codec.describe(),
            "head": self.head.describe(base_dimension=self.base_dimension),
        }


