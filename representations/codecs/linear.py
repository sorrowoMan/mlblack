from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.models import LinearPointModel, OrthogonalFeatureMap, OrthogonalLinearPointModel


@dataclass(frozen=True)
class LinearCodecConfig:
    n_features: int
    feature_names: tuple[str, ...] = tuple()
    init_scale: float = 0.01
    random_seed: int = 42
    representation_name: str = "linear"


class LinearPointCodec:
    """Raw linear parameter codec used by headed representations."""

    def __init__(self, config: LinearCodecConfig) -> None:
        self.config = config
        self.base_dimension = 1 + int(config.n_features)
        self._rng = np.random.default_rng(int(config.random_seed))

    def init_values(self) -> np.ndarray:
        values = self._rng.normal(loc=0.0, scale=float(self.config.init_scale), size=self.base_dimension)
        values[0] = 0.0
        return np.asarray(values, dtype=float)

    def encode(self, model: LinearPointModel) -> np.ndarray:
        return np.concatenate([[float(model.intercept)], np.asarray(model.weights, dtype=float).reshape(-1)])

    def decode(self, values: np.ndarray, context: Mapping[str, Any] | None = None) -> LinearPointModel:
        ctx = dict(context or {})
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.shape[0] != self.base_dimension:
            raise ValueError(f"base state dimension {arr.shape[0]} does not match {self.base_dimension}")
        return LinearPointModel(
            intercept=float(arr[0]),
            weights=np.asarray(arr[1:], dtype=float),
            feature_names=tuple(self.config.feature_names),
            metadata={"representation": self.config.representation_name, "head_block": ctx.get("head.block")},
        )

    def describe(self) -> dict[str, Any]:
        return {
            "codec": "linear_point",
            "base_dimension": int(self.base_dimension),
            "n_features": int(self.config.n_features),
            "feature_names": tuple(self.config.feature_names),
        }


@dataclass(frozen=True)
class OrthogonalLinearCodecConfig:
    include_raw: bool = True
    include_square: bool = True
    include_interactions: bool = True
    max_components: int | None = None
    energy_threshold: float | None = 0.999
    init_scale: float = 0.01
    random_seed: int = 42
    representation_name: str = "orthogonal_point_linear"


class OrthogonalLinearPointCodec:
    """Orthogonal-feature linear parameter codec."""

    def __init__(self, feature_map: OrthogonalFeatureMap, config: OrthogonalLinearCodecConfig) -> None:
        self.feature_map = feature_map
        self.config = config
        self.base_dimension = 1 + int(feature_map.output_dim)
        self._rng = np.random.default_rng(int(config.random_seed))

    @classmethod
    def from_data(
        cls,
        X: np.ndarray,
        *,
        feature_names: Sequence[str] | None = None,
        config: OrthogonalLinearCodecConfig | None = None,
    ) -> "OrthogonalLinearPointCodec":
        cfg = config or OrthogonalLinearCodecConfig()
        fmap = OrthogonalFeatureMap.fit(
            X,
            feature_names=feature_names,
            include_raw=cfg.include_raw,
            include_square=cfg.include_square,
            include_interactions=cfg.include_interactions,
            max_components=cfg.max_components,
            energy_threshold=cfg.energy_threshold,
        )
        return cls(fmap, cfg)

    def init_values(self) -> np.ndarray:
        values = self._rng.normal(loc=0.0, scale=float(self.config.init_scale), size=self.base_dimension)
        values[0] = 0.0
        return np.asarray(values, dtype=float)

    def encode(self, model: OrthogonalLinearPointModel) -> np.ndarray:
        return np.concatenate([[float(model.intercept)], np.asarray(model.weights, dtype=float).reshape(-1)])

    def decode(self, values: np.ndarray, context: Mapping[str, Any] | None = None) -> OrthogonalLinearPointModel:
        ctx = dict(context or {})
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.shape[0] != self.base_dimension:
            raise ValueError(f"base state dimension {arr.shape[0]} does not match {self.base_dimension}")
        return OrthogonalLinearPointModel(
            feature_map=self.feature_map,
            intercept=float(arr[0]),
            weights=np.asarray(arr[1:], dtype=float),
            metadata={
                "representation": self.config.representation_name,
                "feature_map": dict(self.feature_map.metadata),
                "head_block": ctx.get("head.block"),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "codec": "orthogonal_linear_point",
            "base_dimension": int(self.base_dimension),
            "feature_map": dict(self.feature_map.metadata),
        }

