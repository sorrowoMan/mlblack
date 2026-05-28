from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from mlblack.pipeline.data_views import NumericDataView
from mlblack.pipeline.numericizer.text import VocabularyTokenizer, VocabularyTokenizerConfig


@dataclass(frozen=True)
class DatasetStreamConfig:
    batch_size: int = 32
    shuffle: bool = False
    drop_last: bool = False
    repeat: int = 1
    seed: int = 42


@dataclass(frozen=True)
class RowBatch:
    rows: tuple[Mapping[str, Any], ...]
    indices: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(len(self.rows))


class RowDatasetStream:
    """Lightweight row streaming surface for local or external datasets."""

    def __init__(self, rows: Iterable[Mapping[str, Any]], config: DatasetStreamConfig | None = None) -> None:
        self.rows = tuple(dict(row) for row in rows)
        self.config = config or DatasetStreamConfig()

    @classmethod
    def from_huggingface(
        cls,
        dataset: Any,
        *,
        columns: Sequence[str] | None = None,
        config: DatasetStreamConfig | None = None,
    ) -> "RowDatasetStream":
        names = tuple(str(col) for col in columns) if columns is not None else None
        rows = []
        for row in dataset:
            payload = dict(row)
            if names is not None:
                payload = {name: payload[name] for name in names}
            rows.append(payload)
        return cls(rows, config=config)

    def iter_batches(self) -> Iterator[RowBatch]:
        n_rows = len(self.rows)
        batch_size = max(1, int(self.config.batch_size))
        repeat = max(1, int(self.config.repeat))
        rng = np.random.default_rng(int(self.config.seed))
        base = np.arange(n_rows)
        for epoch in range(repeat):
            indices = base.copy()
            if bool(self.config.shuffle):
                rng.shuffle(indices)
            for start in range(0, n_rows, batch_size):
                batch_idx = indices[start : start + batch_size]
                if bool(self.config.drop_last) and batch_idx.shape[0] < batch_size:
                    continue
                yield RowBatch(
                    rows=tuple(self.rows[int(idx)] for idx in batch_idx),
                    indices=tuple(int(idx) for idx in batch_idx),
                    metadata={"epoch": int(epoch), "batch_size": int(batch_idx.shape[0]), "source": "row_dataset_stream"},
                )

    def describe(self) -> dict[str, Any]:
        keys = tuple(sorted({str(key) for row in self.rows for key in row.keys()}))
        return {
            "name": "row_dataset_stream",
            "n_rows": int(len(self.rows)),
            "columns": keys,
            "batch_size": int(self.config.batch_size),
            "shuffle": bool(self.config.shuffle),
            "drop_last": bool(self.config.drop_last),
            "repeat": int(self.config.repeat),
        }


@dataclass(frozen=True)
class NumericBatch:
    X: np.ndarray
    y: np.ndarray
    indices: tuple[int, ...]
    split: str = "train"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_data_view(self, *, feature_names: Sequence[str] | None = None, target_name: str = "target") -> NumericDataView:
        return NumericDataView(
            X_train=self.X,
            y_train=self.y,
            feature_names=feature_names,
            target_name=target_name,
            metadata={**dict(self.metadata), "batch_indices": tuple(self.indices), "split": self.split},
        )


class NumericBatchLoader:
    """Batch iterator over a NumericDataView.

    The loader is deliberately not a runtime scheduler. It only creates stable
    data slices that a problem or adapter can consume inside one trainer.
    """

    def __init__(self, data: NumericDataView, config: DatasetStreamConfig | None = None) -> None:
        self.data = data
        self.config = config or DatasetStreamConfig()

    def iter_train(self) -> Iterator[NumericBatch]:
        yield from self._iter_arrays(self.data.X_train, self.data.y_train, split="train")

    def iter_valid(self) -> Iterator[NumericBatch]:
        if self.data.X_valid is None or self.data.y_valid is None:
            return
        yield from self._iter_arrays(self.data.X_valid, self.data.y_valid, split="valid")

    def _iter_arrays(self, X: np.ndarray, y: np.ndarray, *, split: str) -> Iterator[NumericBatch]:
        n_rows = int(X.shape[0])
        batch_size = max(1, int(self.config.batch_size))
        repeat = max(1, int(self.config.repeat))
        rng = np.random.default_rng(int(self.config.seed))
        base = np.arange(n_rows)
        for epoch in range(repeat):
            indices = base.copy()
            if bool(self.config.shuffle):
                rng.shuffle(indices)
            for start in range(0, n_rows, batch_size):
                batch_idx = indices[start : start + batch_size]
                if bool(self.config.drop_last) and batch_idx.shape[0] < batch_size:
                    continue
                yield NumericBatch(
                    X=np.asarray(X[batch_idx], dtype=float),
                    y=np.asarray(y[batch_idx], dtype=float).reshape(-1),
                    indices=tuple(int(idx) for idx in batch_idx),
                    split=split,
                    metadata={"epoch": int(epoch), "batch_size": int(batch_idx.shape[0])},
                )

    def describe(self) -> dict[str, Any]:
        return {
            "name": "numeric_batch_loader",
            "n_train": int(self.data.X_train.shape[0]),
            "n_valid": 0 if self.data.X_valid is None else int(self.data.X_valid.shape[0]),
            "batch_size": int(self.config.batch_size),
            "shuffle": bool(self.config.shuffle),
            "drop_last": bool(self.config.drop_last),
            "repeat": int(self.config.repeat),
        }


@dataclass(frozen=True)
class TokenizedTextDatasetConfig:
    text_column: str = "text"
    target_column: str | None = "label"
    max_length: int = 128
    valid_ratio: float = 0.0
    seed: int = 42
    tokenizer_metadata_key: str = "tokenizer"


class TokenizedTextDatasetBuilder:
    """Build token-id NumericDataView objects from text datasets."""

    def __init__(
        self,
        tokenizer: Any | None = None,
        config: TokenizedTextDatasetConfig | None = None,
    ) -> None:
        self.config = config or TokenizedTextDatasetConfig()
        self.tokenizer = tokenizer or VocabularyTokenizer(
            VocabularyTokenizerConfig(max_length=int(self.config.max_length), add_eos=True)
        )

    def from_rows(self, rows: Iterable[Mapping[str, Any]]) -> NumericDataView:
        payload = tuple(dict(row) for row in rows)
        texts = tuple(str(row.get(self.config.text_column, "")) for row in payload)
        labels = self._labels_from_rows(payload)
        token_payload = self._encode_texts(texts)
        return self._build_data_view(token_payload, labels)

    def from_huggingface(self, dataset: Any) -> NumericDataView:
        return self.from_rows(dict(row) for row in dataset)

    def _labels_from_rows(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        target_column = self.config.target_column
        if target_column is None:
            return np.zeros(len(rows), dtype=float)
        return np.asarray([row.get(target_column, 0.0) for row in rows], dtype=float).reshape(-1)

    def _encode_texts(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        if isinstance(self.tokenizer, VocabularyTokenizer):
            token_ids = self.tokenizer.fit_transform(texts, max_length=int(self.config.max_length))
            return {"input_ids": np.asarray(token_ids, dtype=np.int64)}
        if callable(self.tokenizer):
            encoded = self.tokenizer(
                list(texts),
                max_length=int(self.config.max_length),
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            return {str(key): np.asarray(value) for key, value in dict(encoded).items()}
        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            encoded = encode(tuple(texts))
            if isinstance(encoded, Mapping):
                return {str(key): np.asarray(value) for key, value in dict(encoded).items()}
        raise TypeError("tokenizer must be VocabularyTokenizer, PretrainedTokenizerBridge, or a HuggingFace-style callable")

    def _build_data_view(self, token_payload: Mapping[str, np.ndarray], labels: np.ndarray) -> NumericDataView:
        input_ids = np.asarray(token_payload.get("input_ids"), dtype=float)
        if input_ids.ndim != 2:
            raise ValueError("tokenizer must produce 2D input_ids")
        labels = np.asarray(labels, dtype=float).reshape(-1)
        if input_ids.shape[0] != labels.shape[0]:
            raise ValueError("input_ids and labels row counts differ")
        n_rows = int(input_ids.shape[0])
        valid_ratio = max(0.0, min(0.95, float(self.config.valid_ratio)))
        if valid_ratio > 0.0 and n_rows > 1:
            rng = np.random.default_rng(int(self.config.seed))
            indices = np.arange(n_rows)
            rng.shuffle(indices)
            n_valid = max(1, int(round(valid_ratio * n_rows)))
            valid_idx = indices[:n_valid]
            train_idx = indices[n_valid:]
        else:
            train_idx = np.arange(n_rows)
            valid_idx = np.zeros(0, dtype=int)

        metadata = {
            "data_kind": "tokenized_text",
            "text_column": self.config.text_column,
            "target_column": self.config.target_column,
            "max_length": int(self.config.max_length),
            self.config.tokenizer_metadata_key: self._tokenizer_description(),
        }
        for key, value in token_payload.items():
            if str(key) == "input_ids":
                continue
            arr = np.asarray(value)
            metadata[f"{key}_train"] = arr[train_idx]
            if valid_idx.size:
                metadata[f"{key}_valid"] = arr[valid_idx]

        return NumericDataView(
            X_train=input_ids[train_idx],
            y_train=labels[train_idx],
            X_valid=None if not valid_idx.size else input_ids[valid_idx],
            y_valid=None if not valid_idx.size else labels[valid_idx],
            feature_names=tuple(f"token_{idx}" for idx in range(input_ids.shape[1])),
            target_name=str(self.config.target_column or "target"),
            metadata=metadata,
        )

    def _tokenizer_description(self) -> Mapping[str, Any]:
        describe = getattr(self.tokenizer, "describe", None)
        if callable(describe):
            return dict(describe())
        return {"name": type(self.tokenizer).__name__}

    def describe(self) -> dict[str, Any]:
        return {
            "name": "tokenized_text_dataset_builder",
            "text_column": self.config.text_column,
            "target_column": self.config.target_column,
            "max_length": int(self.config.max_length),
            "valid_ratio": float(self.config.valid_ratio),
            "tokenizer": self._tokenizer_description(),
        }


__all__ = [
    "DatasetStreamConfig",
    "NumericBatch",
    "NumericBatchLoader",
    "RowBatch",
    "RowDatasetStream",
    "TokenizedTextDatasetBuilder",
    "TokenizedTextDatasetConfig",
]
