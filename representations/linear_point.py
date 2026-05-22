from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.core.contracts import ComponentContract
from mlblack.core.head import OutputHead
from mlblack.core.representation import ModelRepresentation
from mlblack.core.types import UnknownState
from mlblack.representations.heads import PointHead
from mlblack.models import LinearPointModel
from mlblack.representations.codecs import LinearCodecConfig, LinearPointCodec


@dataclass(frozen=True)
class LinearPointConfig:
    n_features: int
    feature_names: tuple[str, ...] = tuple()
    init_scale: float = 0.01
    random_seed: int = 42


class LinearPointRepresentation(ModelRepresentation):
    """Unknown vector -> raw linear model with pluggable output head.

    Point head values are [intercept, w0, w1, ...]. Interval heads allocate
    multiple base-linear blocks and return interval prediction objects.
    """

    name = "linear_point"
    context_requires = ('candidate.unknown_state',)
    context_optional = ()
    context_provides = ('candidate.output', 'model.predict')
    context_mutates = ('candidate.repaired_state',)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'Reads context fields: candidate.unknown_state; provides candidate.output, model.predict; mutates candidate.repaired_state.'
    contract = ComponentContract(
        name=name,
        requires=("candidate.unknown_state",),
        provides=("candidate.output", "model.predict"),
        mutates=("candidate.repaired_state",),
        supports_gradient=None,
        supports_batch=True,
        supports_resume=True,
        metadata={"family": "linear", "head": "point", "feature_space": "raw"},
    )

    def __init__(self, config: LinearPointConfig, head: OutputHead | None = None) -> None:
        self.config = config
        self.codec = LinearPointCodec(
            LinearCodecConfig(
                n_features=int(config.n_features),
                feature_names=tuple(config.feature_names),
                init_scale=float(config.init_scale),
                random_seed=int(config.random_seed),
                representation_name=self.name,
            )
        )
        self.head = head or PointHead()
        self.base_dimension = int(self.codec.base_dimension)
        self.dimension = int(self.head.parameter_size(self.base_dimension))

    @classmethod
    def from_data(
        cls,
        X: np.ndarray,
        *,
        feature_names: Sequence[str] | None = None,
        config: LinearPointConfig | None = None,
        head: OutputHead | None = None,
    ) -> "LinearPointRepresentation":
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D")
        cfg = config or LinearPointConfig(
            n_features=int(X_arr.shape[1]),
            feature_names=tuple(feature_names or tuple(f"x{i}" for i in range(X_arr.shape[1]))),
        )
        return cls(cfg, head=head)

    def init(self, context: Mapping[str, Any]) -> UnknownState:
        _ = context
        values = np.zeros(self.dimension, dtype=float)
        base_values = self.codec.init_values()
        values[: base_values.shape[0]] = base_values
        return UnknownState(values=np.asarray(values, dtype=float), metadata={"source": "linear_point_init"})

    def encode(self, model: Any, context: Mapping[str, Any] | None = None) -> UnknownState:
        _ = context
        if self.head.output_kind != "point":
            raise NotImplementedError("encoding is only implemented for point head")
        if not isinstance(model, LinearPointModel):
            raise TypeError("LinearPointRepresentation can only encode LinearPointModel")
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

    def _decode_base(self, values: np.ndarray, context: Mapping[str, Any] | None = None) -> LinearPointModel:
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


