from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.core.backend_session import get_compute_backend_from_context
from mlblack.models import NumpyMLPPointModel, mlp_parameter_shapes, split_mlp_parameters

from .specs import NeuralBlockSpec, NeuralGraphSpec, NeuralHeadSpec


@dataclass(frozen=True)
class ParameterLayout:
    """Flat parameter layout for a decoded neural graph."""

    shapes: tuple[tuple[int, ...], ...]
    names: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_size(self) -> int:
        return int(sum(np.prod(shape) for shape in self.shapes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "shapes": tuple(tuple(int(v) for v in shape) for shape in self.shapes),
            "names": tuple(str(name) for name in self.names),
            "total_size": int(self.total_size),
            "metadata": dict(self.metadata),
        }


class NeuralGraphCodec:
    """Decode a flat parameter state into a neural model from NeuralGraphSpec.

    The codec is a backend-dispatch facade. Pure numpy MLP remains local. Other
    routes are lowered by the selected compute backend, for example the torch
    backend's neural lowering capability.
    """

    def __init__(
        self,
        spec: NeuralGraphSpec | Mapping[str, Any],
        *,
        init_scale: float = 0.02,
        random_seed: int = 42,
        representation_name: str = "neural_graph",
    ) -> None:
        self.spec = NeuralGraphSpec.from_value(spec)
        self.init_scale = float(init_scale)
        self.random_seed = int(random_seed)
        self.representation_name = str(representation_name)
        self._rng = np.random.default_rng(self.random_seed)
        self._layout_cache: dict[str, ParameterLayout] = {}
        self._backend_contract_cache: dict[str, Mapping[str, Any]] = {}

    @classmethod
    def mlp(
        cls,
        *,
        input_dim: int,
        hidden_layers: Sequence[int] = (64, 32),
        output_dim: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
        init_scale: float = 0.02,
        random_seed: int = 42,
        representation_name: str = "neural_graph_mlp",
    ) -> "NeuralGraphCodec":
        return cls(
            NeuralGraphSpec.mlp(
                input_dim=int(input_dim),
                hidden_layers=tuple(int(v) for v in hidden_layers),
                output_dim=int(output_dim),
                activation=str(activation),
                dropout=float(dropout),
                name=representation_name,
            ),
            init_scale=init_scale,
            random_seed=random_seed,
            representation_name=representation_name,
        )

    @property
    def layout(self) -> ParameterLayout:
        return self.parameter_layout()

    @property
    def base_dimension(self) -> int:
        return int(self.parameter_layout().total_size)

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        return self.parameter_layout().shapes

    def parameter_layout(self, context: Mapping[str, Any] | None = None) -> ParameterLayout:
        route = self._route()
        if route == "mlp":
            if context is None:
                for key, layout in self._layout_cache.items():
                    if key.startswith("backend:"):
                        return layout
            backend = self._optional_backend(
                context,
                ("parameters.layout", "neural.lowering.mlp"),
                consumer="NeuralGraphCodec.parameter_layout[mlp]",
            )
            if backend is not None:
                return self._backend_parameter_layout(backend, route)
            key = "local:mlp"
            if key not in self._layout_cache:
                self._layout_cache[key] = self._build_mlp_parameter_layout()
            return self._layout_cache[key]
        if context is None and self._layout_cache:
            return next(iter(self._layout_cache.values()))
        backend = self._backend(context, ("parameters.layout",), consumer=f"NeuralGraphCodec.parameter_layout[{route}]")
        return self._backend_parameter_layout(backend, route)

    def _backend_parameter_layout(self, backend: Any, route: str) -> ParameterLayout:
        key = f"backend:{backend.contract().name}:{route}"
        if key not in self._layout_cache:
            shapes, names = backend.parameter_layout(self.spec)
            self._layout_cache[key] = ParameterLayout(
                shapes=tuple(shapes),
                names=tuple(names),
                metadata={"route": route, "graph_name": self.spec.name, "backend": backend.contract().name},
            )
            self._backend_contract_cache[key] = backend.contract().as_dict()
        return self._layout_cache[key]

    def init_values(self, context: Mapping[str, Any] | None = None) -> np.ndarray:
        if self._route() == "mlp":
            backend = self._optional_backend(
                context,
                ("parameters.init", "neural.lowering.mlp"),
                consumer="NeuralGraphCodec.init_values[mlp]",
            )
            if backend is not None:
                return backend.initial_values(self.spec, random_seed=self.random_seed)
            return self._rng.normal(loc=0.0, scale=self.init_scale, size=self.parameter_layout().total_size)
        backend = self._backend(context, ("parameters.init",), consumer="NeuralGraphCodec.init_values")
        return backend.initial_values(self.spec, random_seed=self.random_seed)

    def decode(self, values: np.ndarray, context: Mapping[str, Any] | None = None) -> Any:
        route = self._route()
        if route == "mlp":
            backend = self._optional_backend(
                context,
                ("neural.lowering.mlp",),
                consumer="NeuralGraphCodec.decode[mlp]",
            )
            if backend is not None:
                return backend.decode_neural_graph(
                    np.asarray(values, dtype=float),
                    self.spec,
                    random_seed=self.random_seed,
                    context=context,
                )
            return self._decode_mlp(values, context)
        backend = self._backend(context, ("neural.lowering",), consumer=f"NeuralGraphCodec.decode[{route}]")
        return backend.decode_neural_graph(np.asarray(values, dtype=float), self.spec, random_seed=self.random_seed, context=context)

    def describe(self, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        route = self._route()
        layout: ParameterLayout | None = None
        backend_contract: Mapping[str, Any] | None = None
        backend_name = "local" if route == "mlp" else "unresolved"
        if route == "mlp" or context is not None or self._layout_cache:
            layout = self.parameter_layout(context)
            backend_name = str(layout.metadata.get("backend", "local"))
            if route != "mlp" and context is not None:
                backend = self._backend(context, (), consumer="NeuralGraphCodec.describe")
                backend_contract = backend.contract().as_dict()
            elif route != "mlp":
                backend_contract = self._backend_contract_cache.get(f"backend:{backend_name}:{route}")
            elif context is not None and backend_name != "local":
                backend = self._optional_backend(context, ("neural.lowering.mlp",), consumer="NeuralGraphCodec.describe[mlp]")
                backend_contract = None if backend is None else backend.contract().as_dict()
        return {
            "codec": "neural_graph",
            "route": route,
            "backend": backend_name,
            "backend_contract": backend_contract,
            "base_dimension": None if layout is None else int(layout.total_size),
            "layout": None if layout is None else layout.as_dict(),
            "spec": self.spec.as_dict(),
            "init_scale": float(self.init_scale),
            "random_seed": int(self.random_seed),
        }

    def _route(self) -> str:
        blocks = self.spec.block_specs()
        if len(blocks) == 1 and str(blocks[0].kind).lower() in {"mlp", "mlp_block", "feed_forward"}:
            return "mlp"
        return str(self.spec.metadata.get("route", "unknown"))

    def _backend(self, context: Mapping[str, Any] | None, requirements: tuple[str, ...], *, consumer: str) -> Any:
        if not context or context.get("backend.session") is None:
            raise ValueError(
                f"{consumer} requires Trainer/L0 compute backend context. "
                "Pass trainer.build_context() or run through Trainer.setup()/fit(); "
                "component-local backend fallback is disabled for neural graph routes."
            )
        return get_compute_backend_from_context(context, requirements, consumer=consumer)

    def _optional_backend(self, context: Mapping[str, Any] | None, requirements: tuple[str, ...], *, consumer: str) -> Any | None:
        if not context or context.get("backend.session") is None:
            return None
        backend = get_compute_backend_from_context(context, (), consumer=consumer)
        if backend.contract().missing(tuple(str(item) for item in requirements)):
            return None
        return get_compute_backend_from_context(context, requirements, consumer=consumer)

    def _build_mlp_parameter_layout(self) -> ParameterLayout:
        input_dim, hidden_layers, output_dim, _activation = _mlp_parts(self.spec)
        shapes = mlp_parameter_shapes(input_dim, hidden_layers, output_dim)
        names: list[str] = []
        layer_count = len(shapes) // 2
        for idx in range(layer_count):
            names.append(f"mlp.layers.{idx}.weight")
            names.append(f"mlp.layers.{idx}.bias")
        return ParameterLayout(
            shapes=tuple(shapes),
            names=tuple(names),
            metadata={"route": "mlp", "input_dim": int(input_dim), "output_dim": int(output_dim)},
        )

    def _decode_mlp(self, values: np.ndarray, context: Mapping[str, Any] | None = None) -> NumpyMLPPointModel:
        ctx = dict(context or {})
        input_dim, hidden_layers, output_dim, activation = _mlp_parts(self.spec)
        weights, biases = split_mlp_parameters(
            np.asarray(values, dtype=float),
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
        )
        return NumpyMLPPointModel(
            weights=weights,
            biases=biases,
            activation=activation,
            metadata={
                "representation": self.representation_name,
                "codec": "neural_graph",
                "route": "mlp",
                "graph_name": self.spec.name,
                "head_block": ctx.get("head.block"),
            },
        )


def _mlp_parts(spec: NeuralGraphSpec) -> tuple[int, tuple[int, ...], int, str]:
    input_cfg = dict(spec.input)
    input_dim = int(input_cfg.get("input_dim", input_cfg.get("dimension", 0)))
    if input_dim <= 0:
        raise ValueError("MLP NeuralGraphSpec requires input.input_dim")
    blocks = spec.block_specs()
    if len(blocks) != 1:
        raise ValueError("MLP NeuralGraphSpec expects exactly one block")
    block: NeuralBlockSpec = blocks[0]
    params = dict(block.params)
    hidden_layers = tuple(int(v) for v in params.get("hidden_layers", (64, 32)))
    activation = str(params.get("activation", "relu"))
    heads = spec.head_specs()
    if heads:
        head: NeuralHeadSpec = heads[0]
        output_dim = int(dict(head.params).get("output_dim", 1))
    else:
        output_dim = 1
    return input_dim, hidden_layers, output_dim, activation


__all__ = ["NeuralGraphCodec", "ParameterLayout"]
