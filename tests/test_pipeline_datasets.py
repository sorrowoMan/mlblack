from __future__ import annotations

import numpy as np
import pytest

from mlblack.pipeline import (
    DatasetStreamConfig,
    NumericBatchLoader,
    NumericDataView,
    RowDatasetStream,
    TokenizedTextDatasetBuilder,
    TokenizedTextDatasetConfig,
    VocabularyTokenizer,
    VocabularyTokenizerConfig,
)


def test_row_dataset_stream_batches_rows_deterministically() -> None:
    rows = [{"text": f"row {idx}", "label": idx % 2} for idx in range(5)]
    stream = RowDatasetStream(rows, DatasetStreamConfig(batch_size=2, shuffle=False, repeat=1))

    batches = tuple(stream.iter_batches())
    assert [batch.indices for batch in batches] == [(0, 1), (2, 3), (4,)]
    assert batches[0].rows[0]["text"] == "row 0"
    assert stream.describe()["n_rows"] == 5


def test_numeric_batch_loader_slices_numeric_data_view() -> None:
    data = NumericDataView(
        X_train=np.arange(12, dtype=float).reshape(6, 2),
        y_train=np.arange(6, dtype=float),
    )
    loader = NumericBatchLoader(data, DatasetStreamConfig(batch_size=4, shuffle=False, drop_last=False))

    batches = tuple(loader.iter_train())
    assert [batch.X.shape[0] for batch in batches] == [4, 2]
    assert batches[0].indices == (0, 1, 2, 3)
    assert batches[0].to_data_view().X_train.shape == (4, 2)


def test_tokenized_text_dataset_builder_with_local_tokenizer() -> None:
    rows = [
        {"text": "cat sat", "label": 0},
        {"text": "dog sat", "label": 1},
        {"text": "cat ran", "label": 0},
        {"text": "dog ran", "label": 1},
    ]
    builder = TokenizedTextDatasetBuilder(
        VocabularyTokenizer(VocabularyTokenizerConfig(max_length=5, add_eos=True)),
        TokenizedTextDatasetConfig(text_column="text", target_column="label", max_length=5, valid_ratio=0.25, seed=1),
    )

    data = builder.from_rows(rows)
    assert data.X_train.shape[1] == 5
    assert data.X_valid is not None
    assert data.metadata["data_kind"] == "tokenized_text"
    assert data.metadata["tokenizer"]["name"] == "vocabulary_tokenizer"


def test_tokenized_text_dataset_builder_from_huggingface_dataset_if_installed() -> None:
    datasets = pytest.importorskip("datasets")
    dataset = datasets.Dataset.from_dict({"text": ["a b", "b c", "c d"], "label": [0, 1, 1]})
    builder = TokenizedTextDatasetBuilder(config=TokenizedTextDatasetConfig(max_length=4, valid_ratio=0.0))

    data = builder.from_huggingface(dataset)
    assert data.X_train.shape == (3, 4)
    assert data.y_train.tolist() == [0.0, 1.0, 1.0]


def test_tokenized_text_dataset_builder_accepts_huggingface_style_callable() -> None:
    class _Tokenizer:
        def __call__(self, texts, *, max_length, padding, truncation, return_tensors):
            assert padding == "max_length"
            assert truncation is True
            assert return_tensors == "np"
            return {
                "input_ids": np.ones((len(texts), max_length), dtype=np.int64),
                "attention_mask": np.ones((len(texts), max_length), dtype=np.int64),
            }

    builder = TokenizedTextDatasetBuilder(
        _Tokenizer(),
        TokenizedTextDatasetConfig(max_length=3, target_column=None),
    )

    data = builder.from_rows([{"text": "hello"}, {"text": "world"}])
    assert data.X_train.shape == (2, 3)
    assert data.metadata["attention_mask_train"].shape == (2, 3)
