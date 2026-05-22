from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class VocabularyTokenizerConfig:
    max_length: int = 32
    lowercase: bool = True
    split_pattern: str = r"\w+|[^\w\s]"
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    add_bos: bool = False
    add_eos: bool = False
    min_frequency: int = 1


class VocabularyTokenizer:
    """Small local tokenizer/numericizer for transformer smoke tasks.

    This is intentionally not a pretrained tokenizer. It is the framework-local
    text -> token-id numericizer used when we need deterministic tests without a
    transformers dependency.
    """

    name = "vocabulary_tokenizer"
    context_requires = ("data.raw_rows",)
    context_optional = ("data.schema",)
    context_provides = ("text.token_ids", "tokenizer.vocab")
    context_mutates = ("pipeline.fit_state",)
    context_cache = ("tokenizer.vocab",)
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Fits a local vocabulary and maps text rows to padded token-id arrays."

    def __init__(self, config: VocabularyTokenizerConfig | None = None) -> None:
        self.config = config or VocabularyTokenizerConfig()
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: tuple[str, ...] = tuple()

    def fit(self, texts: Iterable[str]) -> "VocabularyTokenizer":
        counts: dict[str, int] = {}
        for text in texts:
            for token in self.tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        specials = [self.config.pad_token, self.config.unk_token, self.config.bos_token, self.config.eos_token]
        vocab = {token: idx for idx, token in enumerate(specials)}
        for token in sorted(token for token, count in counts.items() if count >= int(self.config.min_frequency)):
            if token not in vocab:
                vocab[token] = len(vocab)
        self.vocab = vocab
        self.inverse_vocab = tuple(token for token, _idx in sorted(vocab.items(), key=lambda pair: pair[1]))
        return self

    def tokenize(self, text: str) -> tuple[str, ...]:
        value = str(text)
        if bool(self.config.lowercase):
            value = value.lower()
        return tuple(match.group(0) for match in re.finditer(str(self.config.split_pattern), value))

    def encode(self, text: str, *, max_length: int | None = None) -> np.ndarray:
        if not self.vocab:
            raise ValueError("VocabularyTokenizer must be fit before encode()")
        tokens = list(self.tokenize(text))
        if bool(self.config.add_bos):
            tokens.insert(0, self.config.bos_token)
        if bool(self.config.add_eos):
            tokens.append(self.config.eos_token)
        limit = int(max_length or self.config.max_length)
        ids = [self.vocab.get(token, self.vocab[self.config.unk_token]) for token in tokens[:limit]]
        if len(ids) < limit:
            ids.extend([self.vocab[self.config.pad_token]] * (limit - len(ids)))
        return np.asarray(ids, dtype=np.int64)

    def transform(self, texts: Iterable[str], *, max_length: int | None = None) -> np.ndarray:
        return np.vstack([self.encode(text, max_length=max_length) for text in texts]).astype(np.int64)

    def fit_transform(self, texts: Iterable[str], *, max_length: int | None = None) -> np.ndarray:
        rows = tuple(str(text) for text in texts)
        self.fit(rows)
        return self.transform(rows, max_length=max_length)

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "max_length": int(self.config.max_length),
            "vocab_size": int(len(self.vocab)),
            "lowercase": bool(self.config.lowercase),
            "add_bos": bool(self.config.add_bos),
            "add_eos": bool(self.config.add_eos),
        }

    def as_dict(self) -> dict[str, object]:
        return {"config": self.config.__dict__.copy(), "vocab": dict(self.vocab)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "VocabularyTokenizer":
        config_payload = dict(payload.get("config", {}) or {})
        tokenizer = cls(VocabularyTokenizerConfig(**config_payload))
        tokenizer.vocab = {str(key): int(value) for key, value in dict(payload.get("vocab", {}) or {}).items()}
        tokenizer.inverse_vocab = tuple(token for token, _idx in sorted(tokenizer.vocab.items(), key=lambda pair: pair[1]))
        return tokenizer


__all__ = ["VocabularyTokenizer", "VocabularyTokenizerConfig"]
