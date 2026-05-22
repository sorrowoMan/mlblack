from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PretrainedTokenizerBridgeConfig:
    model_name_or_path: str
    max_length: int = 128
    padding: str = "max_length"
    truncation: bool = True
    return_attention_mask: bool = True
    kwargs: Mapping[str, Any] = field(default_factory=dict)


class PretrainedTokenizerBridge:
    """Lazy bridge to HuggingFace tokenizers.

    The bridge is optional and does not make transformers a hard mlblack
    dependency. It is a numericizer surface: text -> token IDs/masks.
    """

    name = "pretrained_tokenizer_bridge"
    context_requires = ("data.raw_rows",)
    context_optional = ("resource.context",)
    context_provides = ("text.token_ids", "pretrained.tokenizer")
    context_mutates = ()
    context_cache = ("pretrained.tokenizer",)
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Loads a pretrained tokenizer lazily and maps text rows to token ids."

    def __init__(self, config: PretrainedTokenizerBridgeConfig) -> None:
        self.config = config
        self._tokenizer: Any | None = None

    def load(self) -> Any:
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("PretrainedTokenizerBridge requires optional dependency 'transformers'") from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path, **dict(self.config.kwargs))
        return self._tokenizer

    def encode(self, texts: Sequence[str]) -> Mapping[str, np.ndarray]:
        tokenizer = self.load()
        encoded = tokenizer(
            list(str(text) for text in texts),
            max_length=int(self.config.max_length),
            padding=str(self.config.padding),
            truncation=bool(self.config.truncation),
            return_attention_mask=bool(self.config.return_attention_mask),
            return_tensors="np",
        )
        return {str(key): np.asarray(value) for key, value in dict(encoded).items()}

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_name_or_path": self.config.model_name_or_path,
            "max_length": int(self.config.max_length),
            "padding": self.config.padding,
            "truncation": bool(self.config.truncation),
            "loaded": self._tokenizer is not None,
        }


@dataclass(frozen=True)
class PretrainedModelBridgeConfig:
    model_name_or_path: str
    task: str = "base"  # base | causal_lm | sequence_classification
    device: str = "cpu"
    kwargs: Mapping[str, Any] = field(default_factory=dict)


class PretrainedModelBridge:
    """Lazy pretrained-model loader for external-model/component use cases."""

    name = "pretrained_model_bridge"
    context_requires = ("pretrained.tokenizer",)
    context_optional = ("resource.device", "resource.context")
    context_provides = ("pretrained.model",)
    context_mutates = ()
    context_cache = ("pretrained.model",)
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Loads a pretrained transformer model lazily; does not define a training framework."

    def __init__(self, config: PretrainedModelBridgeConfig) -> None:
        self.config = config
        self._model: Any | None = None

    def load(self) -> Any:
        if self._model is None:
            try:
                from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("PretrainedModelBridge requires optional dependency 'transformers'") from exc
            task = str(self.config.task).lower()
            kwargs = dict(self.config.kwargs)
            if task in {"causal_lm", "lm", "language_model"}:
                model = AutoModelForCausalLM.from_pretrained(self.config.model_name_or_path, **kwargs)
            elif task in {"sequence_classification", "classification", "classifier"}:
                model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name_or_path, **kwargs)
            elif task in {"base", "embedding", "encoder"}:
                model = AutoModel.from_pretrained(self.config.model_name_or_path, **kwargs)
            else:
                raise ValueError(f"unsupported pretrained model task: {self.config.task}")
            try:
                model.to(str(self.config.device))
            except Exception:
                if str(self.config.device).startswith("cuda"):
                    raise
            self._model = model
        return self._model

    def forward(self, **inputs: Any) -> Any:
        model = self.load()
        return model(**inputs)

    __call__ = forward

    def generate(self, **inputs: Any) -> Any:
        model = self.load()
        generate = getattr(model, "generate", None)
        if not callable(generate):
            raise TypeError(f"loaded pretrained model for task {self.config.task!r} does not expose generate(...)")
        return generate(**inputs)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_name_or_path": self.config.model_name_or_path,
            "task": self.config.task,
            "device": self.config.device,
            "loaded": self._model is not None,
        }


@dataclass(frozen=True)
class PretrainedCheckpointMappingConfig:
    """Shape-safe parameter mapper config.

    name_map is target_name -> source_name. If a target name is not listed, the
    mapper first tries the same name, then optional unique shape matching.
    """

    name_map: Mapping[str, str] = field(default_factory=dict)
    source_prefixes: Sequence[str] = field(default_factory=tuple)
    target_prefixes: Sequence[str] = field(default_factory=tuple)
    name_replacements: Sequence[tuple[str, str]] = field(default_factory=tuple)
    allow_shape_match: bool = False
    strict_shapes: bool = True
    missing_policy: str = "keep_target"  # keep_target | zero | error


@dataclass(frozen=True)
class PretrainedCheckpointMappingReport:
    matched: tuple[tuple[str, str], ...]
    missing: tuple[str, ...]
    shape_mismatch: tuple[tuple[str, str, tuple[int, ...], tuple[int, ...]], ...]
    total_target_parameters: int
    mapped_target_parameters: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def mapped_fraction(self) -> float:
        if self.total_target_parameters <= 0:
            return 0.0
        return float(self.mapped_target_parameters) / float(self.total_target_parameters)

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched": [list(item) for item in self.matched],
            "missing": list(self.missing),
            "shape_mismatch": [
                {
                    "target": target,
                    "source": source,
                    "target_shape": list(target_shape),
                    "source_shape": list(source_shape),
                }
                for target, source, target_shape, source_shape in self.shape_mismatch
            ],
            "total_target_parameters": int(self.total_target_parameters),
            "mapped_target_parameters": int(self.mapped_target_parameters),
            "mapped_fraction": float(self.mapped_fraction),
            "metadata": dict(self.metadata),
        }


class PretrainedCheckpointMapper:
    """Map an external checkpoint state_dict into an mlblack neural graph.

    This is deliberately explicit and conservative. It does not guess semantic
    architecture equivalence; callers provide a target->source name map or opt
    into unique shape matching for smoke/prototype use.
    """

    name = "pretrained_checkpoint_mapper"
    context_requires = ("pretrained.model", "candidate.model")
    context_optional = ("pretrained.checkpoint_map",)
    context_provides = ("candidate.unknown_state", "pretrained.checkpoint_report")
    context_mutates = ("candidate.model",)
    context_cache = ("pretrained.checkpoint_report",)
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Maps external checkpoint tensors into a neural graph target module with shape-safe reporting."

    def __init__(self, config: PretrainedCheckpointMappingConfig | None = None) -> None:
        self.config = config or PretrainedCheckpointMappingConfig()
        self.last_report: PretrainedCheckpointMappingReport | None = None

    def map_state_dict(self, source_state: Mapping[str, Any], target_model: Any) -> PretrainedCheckpointMappingReport:
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PretrainedCheckpointMapper requires optional dependency 'torch'") from exc

        target_state = target_model.state_dict()
        source = {str(key): value for key, value in dict(source_state).items()}
        source_lookup = _source_key_lookup(
            source,
            source_prefixes=tuple(str(item) for item in self.config.source_prefixes),
            replacements=tuple((str(old), str(new)) for old, new in self.config.name_replacements),
        )
        shape_index = _unique_shape_index(source) if bool(self.config.allow_shape_match) else {}
        matched: list[tuple[str, str]] = []
        missing: list[str] = []
        shape_mismatch: list[tuple[str, str, tuple[int, ...], tuple[int, ...]]] = []
        mapped_params = 0

        next_state = {}
        for target_name, target_tensor in target_state.items():
            source_name = ""
            source_tensor = None
            for candidate_name in _candidate_source_names(
                str(target_name),
                name_map=self.config.name_map,
                target_prefixes=tuple(str(item) for item in self.config.target_prefixes),
                replacements=tuple((str(old), str(new)) for old, new in self.config.name_replacements),
            ):
                actual_source_name = source_lookup.get(candidate_name)
                if actual_source_name is not None:
                    source_name = actual_source_name
                    source_tensor = source.get(actual_source_name)
                    break
            if source_tensor is None and bool(self.config.allow_shape_match):
                source_name = shape_index.get(_shape_tuple(target_tensor), "")
                source_tensor = source.get(source_name) if source_name else None
            if source_tensor is None:
                missing.append(str(target_name))
                if str(self.config.missing_policy).lower() == "error":
                    raise KeyError(f"checkpoint is missing target parameter: {target_name}")
                if str(self.config.missing_policy).lower() == "zero":
                    next_state[target_name] = torch.zeros_like(target_tensor)
                else:
                    next_state[target_name] = target_tensor
                continue
            source_tensor = torch.as_tensor(source_tensor, dtype=target_tensor.dtype, device=target_tensor.device)
            if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                shape_mismatch.append((str(target_name), str(source_name), _shape_tuple(target_tensor), _shape_tuple(source_tensor)))
                if bool(self.config.strict_shapes):
                    next_state[target_name] = target_tensor
                    continue
                try:
                    source_tensor = source_tensor.reshape_as(target_tensor)
                except Exception:
                    next_state[target_name] = target_tensor
                    continue
            next_state[target_name] = source_tensor
            matched.append((str(target_name), str(source_name)))
            mapped_params += int(target_tensor.numel())

        target_model.load_state_dict(next_state, strict=True)
        total_params = int(sum(int(tensor.numel()) for tensor in target_state.values()))
        report = PretrainedCheckpointMappingReport(
            matched=tuple(matched),
            missing=tuple(missing),
            shape_mismatch=tuple(shape_mismatch),
            total_target_parameters=total_params,
            mapped_target_parameters=int(mapped_params),
            metadata={
                "allow_shape_match": bool(self.config.allow_shape_match),
                "strict_shapes": bool(self.config.strict_shapes),
                "source_prefixes": tuple(str(item) for item in self.config.source_prefixes),
                "target_prefixes": tuple(str(item) for item in self.config.target_prefixes),
                "name_replacements": tuple((str(old), str(new)) for old, new in self.config.name_replacements),
            },
        )
        self.last_report = report
        return report

    def map_bridge(self, bridge: PretrainedModelBridge, target_model: Any) -> PretrainedCheckpointMappingReport:
        source_model = bridge.load()
        state_dict = source_model.state_dict() if hasattr(source_model, "state_dict") else {}
        return self.map_state_dict(state_dict, target_model)

    def flat_values_from_model(self, target_model: Any) -> np.ndarray:
        rows = [param.detach().cpu().numpy().reshape(-1).astype(float) for param in target_model.parameters()]
        return np.concatenate(rows) if rows else np.zeros(0, dtype=float)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in getattr(value, "shape", ()))


def _candidate_source_names(
    target_name: str,
    *,
    name_map: Mapping[str, str],
    target_prefixes: tuple[str, ...],
    replacements: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    explicit = str(name_map.get(target_name, ""))
    candidates = [explicit] if explicit else []
    candidates.append(str(target_name))
    candidates.append(_normalize_key(str(target_name), prefixes=target_prefixes, replacements=replacements))
    seen: set[str] = set()
    out: list[str] = []
    for candidate in candidates:
        key = str(candidate)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return tuple(out)


def _source_key_lookup(
    source_state: Mapping[str, Any],
    *,
    source_prefixes: tuple[str, ...],
    replacements: tuple[tuple[str, str], ...],
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for source_name in source_state.keys():
        raw = str(source_name)
        lookup.setdefault(raw, raw)
        lookup.setdefault(_normalize_key(raw, prefixes=source_prefixes, replacements=replacements), raw)
    return lookup


def _normalize_key(name: str, *, prefixes: tuple[str, ...], replacements: tuple[tuple[str, str], ...]) -> str:
    value = str(name)
    for prefix in prefixes:
        if prefix and value.startswith(prefix):
            value = value[len(prefix) :]
    for old, new in replacements:
        if old:
            value = value.replace(old, new)
    return value


def _unique_shape_index(source_state: Mapping[str, Any]) -> dict[tuple[int, ...], str]:
    grouped: dict[tuple[int, ...], list[str]] = {}
    for name, tensor in source_state.items():
        grouped.setdefault(_shape_tuple(tensor), []).append(str(name))
    return {shape: names[0] for shape, names in grouped.items() if len(names) == 1}


__all__ = [
    "PretrainedCheckpointMapper",
    "PretrainedCheckpointMappingConfig",
    "PretrainedCheckpointMappingReport",
    "PretrainedModelBridge",
    "PretrainedModelBridgeConfig",
    "PretrainedTokenizerBridge",
    "PretrainedTokenizerBridgeConfig",
]
