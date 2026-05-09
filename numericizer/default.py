from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from core.common.contracts import Cell, ProcessedDataset, Sample, SampleDataset

from .base import BaseNumericizer
from .plan import NumericizationPlan
from .target_codec import (
    BaseTargetCodec,
    TargetCodec,
    clone_target_codec,
    default_target_codecs,
    infer_target_codec_key,
)

ModalityEncoder = Callable[[Any], np.ndarray]


_CATEGORICAL_MODALITIES = {
    "categorical",
    "category",
    "label",
    "enum",
    "class",
}


def _normalize_modality(modality: str | None) -> str:
    return str(modality or "value").strip().lower()


def _is_categorical_modality(modality: str) -> bool:
    return _normalize_modality(modality) in _CATEGORICAL_MODALITIES


def _to_hashable_scalar(value: Any, *, key: str) -> Any:
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.ndim == 0:
            value = arr.item()
        else:
            raise ValueError(f"{key}: categorical value must be scalar, got ndarray ndim={arr.ndim}")

    if isinstance(value, (list, tuple, dict, set)):
        raise ValueError(f"{key}: categorical value must be scalar+hashable, got {type(value).__name__}")

    try:
        hash(value)
    except Exception as exc:
        raise ValueError(f"{key}: categorical value must be hashable, got {type(value).__name__}") from exc

    return value


def _build_one_hot_state(values: Sequence[Any], *, key: str, unknown: str = "error") -> dict[str, Any]:
    vocab: list[Any] = []
    seen = set()
    for i, raw in enumerate(values):
        if raw is None:
            raise ValueError(f"{key}: categorical value is None at index {i}")
        value = _to_hashable_scalar(raw, key=key)
        if value not in seen:
            seen.add(value)
            vocab.append(value)

    if not vocab:
        raise ValueError(f"{key}: categorical vocab must not be empty")

    idx = {v: i for i, v in enumerate(vocab)}
    return {
        "encoding": "one_hot",
        "unknown": str(unknown).strip().lower(),
        "vocab": tuple(vocab),
        "index_map": idx,
    }


def _encode_one_hot(value: Any, *, state: Mapping[str, Any], key: str, sample_id: str) -> np.ndarray:
    vocab = tuple(state.get("vocab", ()))
    index_map = dict(state.get("index_map", {}))
    unknown = str(state.get("unknown", "error")).strip().lower()

    if not vocab:
        raise ValueError(f"{key}: one_hot state vocab is empty")

    scalar = _to_hashable_scalar(value, key=key)
    idx = index_map.get(scalar)
    if idx is None:
        if unknown == "zero":
            return np.zeros((len(vocab),), dtype=float)
        raise ValueError(
            f"Unknown category '{scalar}' for feature '{key}' (sample_id={sample_id}). "
            f"Known vocab={list(vocab)}"
        )

    out = np.zeros((len(vocab),), dtype=float)
    out[int(idx)] = 1.0
    return out


def _encode_numeric_payload(payload: Any) -> np.ndarray:
    arr = np.asarray(payload, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def _encode_scalar_payload(payload: Any) -> np.ndarray:
    arr = np.asarray(payload, dtype=float).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"scalar modality expects exactly 1 value, got {arr.size}")
    return arr


_DEFAULT_MODALITY_ENCODERS: dict[str, ModalityEncoder] = {
    "value": _encode_numeric_payload,
    "numeric": _encode_numeric_payload,
    "integer": _encode_numeric_payload,
    "boolean": _encode_numeric_payload,
    "scalar": _encode_scalar_payload,
    "vector": _encode_numeric_payload,
    "matrix": _encode_numeric_payload,
    "tensor": _encode_numeric_payload,
}


def _build_modality_encoders(custom: Mapping[str, ModalityEncoder] | None) -> dict[str, ModalityEncoder]:
    enc = dict(_DEFAULT_MODALITY_ENCODERS)
    if custom:
        for key, fn in custom.items():
            k = str(key).strip().lower()
            if not k:
                raise ValueError("modality encoder key must not be empty")
            if not callable(fn):
                raise TypeError(f"encoder for modality '{key}' must be callable")
            enc[k] = fn
    return enc


def _build_target_codecs(custom: Mapping[str, BaseTargetCodec] | None) -> dict[str, BaseTargetCodec]:
    codecs = default_target_codecs()
    if custom:
        for key, codec in custom.items():
            k = str(key).strip().lower()
            if not k:
                raise ValueError("target codec key must not be empty")
            if not isinstance(codec, BaseTargetCodec):
                raise TypeError(
                    f"target codec for key '{key}' must be BaseTargetCodec, got {type(codec).__name__}"
                )
            codecs[k] = clone_target_codec(codec)
    return codecs


def _expand_names(base: str, size: int) -> tuple[str, ...]:
    if size <= 1:
        return (base,)
    return tuple(f"{base}[{i}]" for i in range(size))


def _extract_target_raw(sample: Sample, target_key: str) -> tuple[Any, str]:
    # 1) sample-level label
    if target_key in sample.labels:
        return sample.labels[target_key], "sample.labels"

    # 2) cell-level labels (possibly multiple cells contribute)
    vals: list[Any] = []
    for cell_name in sorted(sample.cells.keys()):
        cell = sample.cells[cell_name]
        if target_key in cell.labels:
            vals.append(cell.labels[target_key])
    if vals:
        if len(vals) == 1:
            return vals[0], "cell.labels"
        return vals, "cell.labels.multi"

    # 3) dedicated target cell payload
    if target_key in sample.cells:
        return sample.cells[target_key].payload, "cell.payload"

    raise ValueError(
        f"Sample '{sample.sample_id}' has no target '{target_key}'. "
        "Expected sample.labels[target], cell.labels[target], or a target cell payload."
    )


class DefaultNumericizer(BaseNumericizer):
    """Strong-typed numericizer: unknown modality raises error.

    Workflow:
    1) fit(data): infer feature layout + fit target codec
    2) transform_features / transform_targets: apply the frozen encoding plan
    3) from_sample_dataset(data): fit + transform in one call (trainer default)

    Notes:
    - categorical features use built-in one-hot by default.
    - to override categorical encoding, register a custom modality encoder for that modality key.
    """

    name = "default_numericizer"

    def __init__(
        self,
        *,
        modality_encoders: Mapping[str, ModalityEncoder] | None = None,
        target_codecs: Mapping[str, TargetCodec] | None = None,
        target_codec: str | None = None,
        categorical_unknown: str = "error",  # error | zero
    ) -> None:
        self.modality_encoders = _build_modality_encoders(modality_encoders)
        self.target_codecs = _build_target_codecs(target_codecs)
        self.target_codec = None if target_codec is None else str(target_codec).strip().lower()
        self.categorical_unknown = str(categorical_unknown).strip().lower()
        if self.categorical_unknown not in {"error", "zero"}:
            raise ValueError("categorical_unknown must be 'error' or 'zero'")

        self._plan: NumericizationPlan | None = None
        self._fitted_target_codec: BaseTargetCodec | None = None
        self._fit_description: str | None = None
        self._target_sources: tuple[str, ...] | None = None

    @property
    def plan(self) -> NumericizationPlan | None:
        return self._plan

    def _require_fitted(self) -> tuple[NumericizationPlan, BaseTargetCodec]:
        if self._plan is None or self._fitted_target_codec is None:
            raise RuntimeError("DefaultNumericizer is not fitted. Call fit(data) first.")
        return self._plan, self._fitted_target_codec

    def _use_builtin_categorical(self, modality: str) -> bool:
        m = _normalize_modality(modality)
        return _is_categorical_modality(m) and m not in self.modality_encoders

    def _encode_cell(self, cell: Cell) -> np.ndarray:
        modality = _normalize_modality(cell.modality)
        encoder = self.modality_encoders.get(modality)
        if encoder is None:
            raise ValueError(
                f"No encoder for modality '{modality}' in cell '{cell.name}'. "
                "Register a custom modality encoder."
            )

        try:
            vec = np.asarray(encoder(cell.payload), dtype=float).reshape(-1)
        except Exception as exc:
            raise ValueError(
                f"Failed to encode cell '{cell.name}' (modality='{modality}') with registered encoder"
            ) from exc

        if vec.size == 0:
            raise ValueError(f"Cell '{cell.name}' encoded to empty vector")
        return vec

    def _resolve_target_codec(self, raw_targets: list[Any], data: SampleDataset) -> tuple[BaseTargetCodec, str]:
        if self.target_codec is None:
            key = infer_target_codec_key(raw_targets)
        else:
            key = self.target_codec

        codec = self.target_codecs.get(str(key).strip().lower())
        if codec is None:
            available = ", ".join(sorted(self.target_codecs.keys()))
            raise ValueError(f"Unknown target codec '{key}'. Available: [{available}]")

        fitted = clone_target_codec(codec).fit(
            raw_targets,
            target_key=str(data.target_key),
            target_names=data.target_names,
        )
        return fitted, str(key)

    def _select_feature_keys(self, data: SampleDataset, samples: Sequence[Sample]) -> tuple[str, ...]:
        if data.feature_cell_keys is not None:
            keys = [str(k) for k in data.feature_cell_keys]
        else:
            first = samples[0]
            keys = [k for k in sorted(first.cells.keys()) if k != str(data.target_key)]

        if not keys:
            raise ValueError("No feature cells selected for SampleDataset")
        return tuple(keys)

    def fit(self, data: SampleDataset) -> "DefaultNumericizer":
        samples = list(data.samples)
        if not samples:
            raise ValueError("SampleDataset.samples must not be empty")

        feature_keys = self._select_feature_keys(data, samples)

        feature_sizes: dict[str, int] = {}
        feature_modalities: dict[str, str] = {}
        feature_states: dict[str, dict[str, Any]] = {}

        categorical_values: dict[str, list[Any]] = {}

        raw_targets: list[Any] = []
        target_sources: list[str] = []

        for i, sample in enumerate(samples):
            for key in feature_keys:
                cell = sample.cells.get(key)
                if cell is None:
                    raise ValueError(f"Sample '{sample.sample_id}' missing feature cell '{key}'")

                modality = _normalize_modality(cell.modality)

                if i == 0:
                    feature_modalities[key] = modality
                    if self._use_builtin_categorical(modality):
                        categorical_values[key] = []
                else:
                    expected = feature_modalities[key]
                    if self._use_builtin_categorical(expected):
                        if not _is_categorical_modality(modality):
                            raise ValueError(
                                f"Feature modality mismatch on cell '{key}': expected categorical-like '{expected}', "
                                f"got '{modality}' (sample_id={sample.sample_id})"
                            )
                    elif modality != expected:
                        raise ValueError(
                            f"Feature modality mismatch on cell '{key}': expected '{expected}', "
                            f"got '{modality}' (sample_id={sample.sample_id})"
                        )

                if self._use_builtin_categorical(modality):
                    categorical_values[key].append(cell.payload)
                else:
                    vec = self._encode_cell(cell)
                    if i == 0:
                        feature_sizes[key] = int(vec.size)
                        feature_states[key] = {}
                    else:
                        expected_size = int(feature_sizes[key])
                        if int(vec.size) != expected_size:
                            raise ValueError(
                                f"Feature size mismatch on cell '{key}': expected {expected_size}, got {vec.size} "
                                f"(sample_id={sample.sample_id})"
                            )

            raw_target, target_source = _extract_target_raw(sample, str(data.target_key))
            if raw_target is None:
                raise ValueError(
                    f"Target '{data.target_key}' is None for sample '{sample.sample_id}'. "
                    "Training target values must be present."
                )
            raw_targets.append(raw_target)
            target_sources.append(target_source)

        for key in feature_keys:
            modality = feature_modalities[key]
            if self._use_builtin_categorical(modality):
                state = _build_one_hot_state(
                    categorical_values[key],
                    key=key,
                    unknown=self.categorical_unknown,
                )
                feature_states[key] = state
                feature_sizes[key] = int(len(tuple(state.get("vocab", ()))))

        feature_names: list[str] = []
        for key in feature_keys:
            modality = feature_modalities[key]
            if self._use_builtin_categorical(modality):
                vocab = tuple(feature_states[key].get("vocab", ()))
                feature_names.extend(tuple(f"cell.{key}=={str(v)}" for v in vocab))
            else:
                feature_names.extend(_expand_names(f"cell.{key}", int(feature_sizes[key])))

        fitted_codec, target_codec_key = self._resolve_target_codec(raw_targets, data)

        target_names = fitted_codec.target_names
        if target_names is None:
            target_names = tuple(data.target_names or (str(data.target_key),))

        self._plan = NumericizationPlan(
            feature_keys=tuple(feature_keys),
            feature_sizes=dict(feature_sizes),
            feature_names=tuple(feature_names),
            feature_modalities=dict(feature_modalities),
            feature_states={k: dict(v) for k, v in feature_states.items()},
            target_key=str(data.target_key),
            target_names=tuple(target_names),
            target_codec_key=str(target_codec_key),
            target_codec_state=dict(fitted_codec.metadata()),
        )
        self._fitted_target_codec = fitted_codec
        self._fit_description = data.description
        self._target_sources = tuple(target_sources)

        return self

    def transform_features(self, samples: Sequence[Sample]) -> np.ndarray:
        plan, _ = self._require_fitted()

        rows: list[np.ndarray] = []
        for sample in samples:
            chunks: list[np.ndarray] = []
            for key in plan.feature_keys:
                cell = sample.cells.get(key)
                if cell is None:
                    raise ValueError(f"Sample '{sample.sample_id}' missing feature cell '{key}'")

                modality = _normalize_modality(cell.modality)
                expected = str(plan.feature_modalities[key])

                if self._use_builtin_categorical(expected):
                    if not _is_categorical_modality(modality):
                        raise ValueError(
                            f"Feature modality mismatch on cell '{key}': expected categorical-like '{expected}', "
                            f"got '{modality}' (sample_id={sample.sample_id})"
                        )
                    state = dict(plan.feature_states.get(key, {}))
                    vec = _encode_one_hot(
                        cell.payload,
                        state=state,
                        key=key,
                        sample_id=sample.sample_id,
                    )
                else:
                    if modality != expected:
                        raise ValueError(
                            f"Feature modality mismatch on cell '{key}': expected '{expected}', "
                            f"got '{modality}' (sample_id={sample.sample_id})"
                        )
                    vec = self._encode_cell(cell)

                expected_size = int(plan.feature_sizes[key])
                if int(vec.size) != expected_size:
                    raise ValueError(
                        f"Feature size mismatch on cell '{key}': expected {expected_size}, got {vec.size} "
                        f"(sample_id={sample.sample_id})"
                    )

                chunks.append(vec)

            if not chunks:
                raise ValueError(f"No feature chunks encoded for sample '{sample.sample_id}'")
            rows.append(np.concatenate(chunks, axis=0))

        if not rows:
            raise ValueError("transform_features received empty samples")

        return np.asarray(np.vstack(rows), dtype=float)

    def transform_targets(self, samples: Sequence[Sample]) -> np.ndarray:
        plan, codec = self._require_fitted()

        rows: list[np.ndarray] = []
        expected_dim = int(codec.output_dim)

        for sample in samples:
            raw_target, _ = _extract_target_raw(sample, str(plan.target_key))
            if raw_target is None:
                raise ValueError(
                    f"Target '{plan.target_key}' is None for sample '{sample.sample_id}'. "
                    "Target encoding requires non-null values."
                )

            vec = np.asarray(codec.encode(raw_target), dtype=float).reshape(-1)
            if int(vec.size) != expected_dim:
                raise ValueError(
                    f"Target size mismatch: expected {expected_dim}, got {vec.size} "
                    f"(sample_id={sample.sample_id})"
                )
            rows.append(vec)

        if not rows:
            raise ValueError("transform_targets received empty samples")

        return np.asarray(np.vstack(rows), dtype=float)

    def encode_features_only(self, samples: Sequence[Sample]) -> np.ndarray:
        """Alias for inference-time feature numericization."""
        return self.transform_features(samples)

    def from_sample_dataset(self, data: SampleDataset) -> ProcessedDataset:
        self.fit(data)
        plan, _ = self._require_fitted()

        samples = list(data.samples)
        X = self.transform_features(samples)
        Y = self.transform_targets(samples)

        target_sources = self._target_sources or tuple()
        target_source = "unknown"
        if target_sources:
            target_source = target_sources[0] if len(set(target_sources)) == 1 else "mixed"

        return ProcessedDataset(
            X_train=np.asarray(X, dtype=float),
            y_train=np.asarray(Y, dtype=float),
            feature_names=tuple(plan.feature_names),
            target_names=tuple(plan.target_names),
            metadata={
                "input_protocol": "sample_dataset",
                "sample_count": int(len(samples)),
                "numericizer": self.name,
                "description": data.description,
                "fit_description": self._fit_description,
                "encoding_plan": dict(plan.to_metadata()),
                "feature_cell_keys": tuple(plan.feature_keys),
                "target_key": str(plan.target_key),
                "target_source": str(target_source),
                "target_sources": tuple(target_sources),
                "target_codec": str(plan.target_codec_key),
                "target_codec_state": dict(plan.target_codec_state),
            },
        )
