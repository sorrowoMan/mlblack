from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from core.symbolic.symbolic_dsl import (
    ParameterSpec,
    collect_parameter_specs,
    evaluate_genome_torch,
    genome_to_strings,
    normalize_genome,
)

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "PyTorch is required for symbolic torch model. Install torch before using symbolic_torch trainer."
    ) from exc


class SymbolicTorchRegressor(nn.Module):
    """Torch regressor over a symbolic basis genome."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        genome: Sequence[Mapping[str, Any]],
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()

        if int(input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        if int(output_dim) <= 0:
            raise ValueError("output_dim must be positive")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.epsilon = float(epsilon)

        self.genome = normalize_genome(genome, input_dim=self.input_dim)
        self.term_names = tuple(str(t["name"]) for t in self.genome)

        specs = collect_parameter_specs(self.genome)
        self._parameter_specs = tuple(specs)

        self.param_table = nn.ParameterDict()
        self._fixed_param_map: Dict[str, str] = {}

        for spec in specs:
            if bool(spec.trainable):
                self.param_table[spec.name] = nn.Parameter(torch.tensor(float(spec.init), dtype=torch.float32))
            else:
                buf_name = f"_fixed_param_{len(self._fixed_param_map)}"
                self.register_buffer(buf_name, torch.tensor(float(spec.init), dtype=torch.float32), persistent=True)
                self._fixed_param_map[spec.name] = buf_name

        self.readout = nn.Linear(int(len(self.genome)), int(self.output_dim), bias=True)

    @property
    def parameter_specs(self) -> tuple[ParameterSpec, ...]:
        return self._parameter_specs

    def _parameter_values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, p in self.param_table.items():
            values[str(name)] = p
        for name, buf_name in self._fixed_param_map.items():
            values[str(name)] = getattr(self, str(buf_name))
        return values

    def basis(self, X):
        return evaluate_genome_torch(
            self.genome,
            X,
            param_values=self._parameter_values(),
            eps=float(self.epsilon),
        )

    def forward(self, X):
        phi = self.basis(X)
        return self.readout(phi)

    def export_parameter_values(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, p in self.param_table.items():
            out[str(name)] = float(p.detach().cpu().item())
        for name, buf_name in self._fixed_param_map.items():
            out[str(name)] = float(getattr(self, str(buf_name)).detach().cpu().item())
        return out

    def export_readout(self) -> tuple[Any, Any]:
        # torch linear weight is (M, T); export as (T, M)
        w = self.readout.weight.detach().cpu().T.contiguous()
        b = self.readout.bias.detach().cpu().contiguous()
        return w, b

    def expression_strings(self, *, with_values: bool = True) -> tuple[str, ...]:
        vals = self.export_parameter_values() if bool(with_values) else None
        return genome_to_strings(self.genome, param_values=vals)
