from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.core.contracts import ComponentContract, ContractMixin
from mlblack.pipeline.data import NumericDataView
from mlblack.presets import build_tiny_transformer_classification_trainer, build_tiny_transformer_lm_trainer
from mlblack.representations import NeuralGraphSpec

try:  # optional integration dependency
    from nsgablack.core.base import BlackBoxProblem as _BlackBoxProblem
except Exception:  # pragma: no cover

    class _BlackBoxProblem:  # type: ignore[no-redef]
        def __init__(self, *, name: str, dimension: int, bounds: Mapping[str, Sequence[float]], objectives: Sequence[str]) -> None:
            self.name = name
            self.dimension = int(dimension)
            self.bounds = dict(bounds)
            self.objectives = tuple(objectives)


@dataclass(frozen=True)
class TransformerSpecSearchConfig:
    """Outer-search surface for nsgablack.

    nsgablack searches this compact vector. mlblack decodes it into a
    NeuralGraphSpec and runs an inner trainer for the task metric.
    """

    task: str = "classification"  # classification | language_modeling
    vocab_size: int = 16
    max_length: int = 8
    num_classes: int = 2
    hidden_dim_choices: tuple[int, ...] = (8, 16, 32)
    num_layer_choices: tuple[int, ...] = (1, 2)
    num_head_choices: tuple[int, ...] = (1, 2, 4)
    ffn_ratio_choices: tuple[float, ...] = (2.0, 4.0)
    norm_choices: tuple[str, ...] = ("layer_norm", "rms_norm")
    position_choices: tuple[str, ...] = ("learned", "rope")
    ffn_kind_choices: tuple[str, ...] = ("mlp", "swiglu")
    lora_rank_choices: tuple[int, ...] = (0, 2, 4)
    inner_steps: int = 2
    learning_rate: float = 1e-2
    complexity_weight: float = 1e-5
    resource_context: Mapping[str, Any] = field(default_factory=dict)
    objective_names: tuple[str, ...] = ("loss", "complexity")


@dataclass(frozen=True)
class TransformerSpecEvaluationRecord:
    outer_candidate: tuple[float, ...]
    graph_spec: Mapping[str, Any]
    objectives: tuple[float, ...]
    metrics: Mapping[str, Any]
    report: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "outer_candidate": list(self.outer_candidate),
            "graph_spec": dict(self.graph_spec),
            "objectives": list(self.objectives),
            "metrics": dict(self.metrics),
            "report": dict(self.report),
        }


class TransformerSpecSearchSpace:
    dimension = 8

    def __init__(self, config: TransformerSpecSearchConfig | None = None) -> None:
        self.config = config or TransformerSpecSearchConfig()

    def bounds(self) -> dict[str, tuple[float, float]]:
        return {f"x{i}": (0.0, 1.0) for i in range(self.dimension)}

    def decode(self, x: Sequence[float] | np.ndarray) -> NeuralGraphSpec:
        values = np.asarray(x, dtype=float).reshape(-1)
        if values.shape[0] < self.dimension:
            values = np.pad(values, (0, self.dimension - values.shape[0]))
        hidden_dim = int(_choice(values[0], self.config.hidden_dim_choices))
        num_layers = int(_choice(values[1], self.config.num_layer_choices))
        num_heads = int(_choice(values[2], _valid_heads(hidden_dim, self.config.num_head_choices)))
        ffn_ratio = float(_choice(values[3], self.config.ffn_ratio_choices))
        norm = str(_choice(values[4], self.config.norm_choices))
        position = str(_choice(values[5], self.config.position_choices))
        ffn_kind = str(_choice(values[6], self.config.ffn_kind_choices))
        lora_rank = int(_choice(values[7], self.config.lora_rank_choices))
        task = str(self.config.task).lower()
        if task in {"classification", "classifier"}:
            heads = ({"kind": "classification", "name": "classification", "params": {"num_classes": int(self.config.num_classes)}},)
        elif task in {"language_modeling", "lm", "causal_lm"}:
            heads = ({"kind": "language_modeling", "name": "lm", "params": {"vocab_size": int(self.config.vocab_size)}},)
        else:
            raise ValueError(f"unsupported TransformerSpec task: {self.config.task}")
        lora = (
            {"enabled": True, "rank": lora_rank, "alpha": 2 * lora_rank, "targets": ("attention.q", "attention.v")}
            if lora_rank > 0
            else {}
        )
        return NeuralGraphSpec.tiny_transformer(
            vocab_size=int(self.config.vocab_size),
            max_length=int(self.config.max_length),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_expansion_ratio=ffn_ratio,
            ffn_kind=ffn_kind,
            norm=norm,
            position_encoding=position,
            lora=lora,
            heads=heads,
            name="outer_searched_tiny_transformer",
        )


class TransformerSpecSearchProblem(_BlackBoxProblem, ContractMixin):
    """nsgablack-facing problem for outer TransformerSpec search."""

    name = "transformer_spec_search_problem"
    context_requires = ("data.X_train", "data.y_train", "resource.context")
    context_optional = ("data.X_valid", "data.y_valid", "resource.lease")
    context_provides = ("feedback.objectives", "feedback.metrics", "neural.transformer_spec")
    context_mutates = ("stage.audit",)
    context_cache = ("task.fitted_model_ref",)
    requires_metrics = ("loss",)
    metrics_fallback = "strict"
    context_notes = "Outer nsgablack searches TransformerSpec; inner mlblack fits parameters and returns task metrics."
    contract = ComponentContract(
        name=name,
        requires=("data.X_train", "data.y_train", "resource.context"),
        optional=("data.X_valid", "data.y_valid", "resource.lease"),
        provides=("feedback.objectives", "feedback.metrics", "neural.transformer_spec"),
        mutates=("stage.audit",),
        cache=("task.fitted_model_ref",),
        supports_batch=False,
        supports_resume=True,
        metadata={"integration": "nsgablack_neural", "outer": "transformer_spec_search"},
    )

    def __init__(
        self,
        data: NumericDataView,
        *,
        config: TransformerSpecSearchConfig | None = None,
        resource_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config or TransformerSpecSearchConfig()
        self.data = data
        self.resource_context = dict(resource_context or self.config.resource_context)
        self.search_space = TransformerSpecSearchSpace(self.config)
        self.last_record: TransformerSpecEvaluationRecord | None = None
        super().__init__(
            name=self.name,
            dimension=self.search_space.dimension,
            bounds=self.search_space.bounds(),
            objectives=tuple(self.config.objective_names),
        )

    def decode_spec(self, x: Sequence[float] | np.ndarray) -> NeuralGraphSpec:
        return self.search_space.decode(x)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.evaluate_detailed(x).objectives, dtype=float)

    def evaluate_detailed(self, x: Sequence[float] | np.ndarray) -> TransformerSpecEvaluationRecord:
        candidate = tuple(float(v) for v in np.asarray(x, dtype=float).reshape(-1))
        spec = self.decode_spec(candidate)
        task = str(self.config.task).lower()
        common = {
            "vocab_size": int(self.config.vocab_size),
            "max_length": int(self.config.max_length),
            "hidden_dim": int(spec.input["hidden_dim"]),
            "num_layers": int(spec.block_specs()[0].repeat),
            "num_heads": int(spec.block_specs()[0].mechanism_specs()["attention"].params["num_heads"]),
            "ffn_expansion_ratio": float(spec.block_specs()[0].mechanism_specs()["ffn"].params["expansion_ratio"]),
            "ffn_kind": str(spec.block_specs()[0].mechanism_specs()["ffn"].kind),
            "norm": str(spec.block_specs()[0].norm.get("kind", "layer_norm")),
            "position_encoding": str(spec.input.get("position_encoding", "learned")),
            "lora": dict(spec.parameterization.get("lora", {}) or {}),
            "learning_rate": float(self.config.learning_rate),
            "run_name": "inner_transformer_spec_fit",
        }
        if task in {"classification", "classifier"}:
            trainer = build_tiny_transformer_classification_trainer(
                self.data,
                num_classes=int(self.config.num_classes),
                **common,
            )
        else:
            trainer = build_tiny_transformer_lm_trainer(self.data, **common)
        if self.resource_context:
            trainer.set_resource_context(self.resource_context)
        result = trainer.fit(max_steps=int(self.config.inner_steps))
        if result.best_feedback is None:
            raise RuntimeError("inner TransformerSpec fit produced no feedback")
        loss = float(result.best_feedback.loss if result.best_feedback.loss is not None else result.best_feedback.scalar_score())
        complexity = float(_parameter_count_from_report(result.report)) * float(self.config.complexity_weight)
        objectives = (loss, complexity)
        record = TransformerSpecEvaluationRecord(
            outer_candidate=candidate,
            graph_spec=spec.as_dict(),
            objectives=objectives,
            metrics=dict(result.best_feedback.metrics),
            report=dict(result.report),
        )
        self.last_record = record
        return record


def _choice(value: float, choices: Sequence[Any]) -> Any:
    if not choices:
        raise ValueError("choice list is empty")
    idx = int(abs(float(value)) * len(choices)) % len(choices)
    return tuple(choices)[idx]


def _valid_heads(hidden_dim: int, choices: Sequence[int]) -> tuple[int, ...]:
    valid = tuple(int(v) for v in choices if int(v) > 0 and int(hidden_dim) % int(v) == 0)
    return valid or (1,)


def _parameter_count_from_report(report: Mapping[str, Any]) -> int:
    representation = dict(report.get("representation", {}) or {})
    layout = dict(dict(representation.get("codec", {}) or {}).get("layout", {}) or {})
    return int(layout.get("total_size", 0) or 0)


__all__ = [
    "TransformerSpecEvaluationRecord",
    "TransformerSpecSearchConfig",
    "TransformerSpecSearchProblem",
    "TransformerSpecSearchSpace",
]
