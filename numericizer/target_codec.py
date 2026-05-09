from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Mapping, Sequence

import numpy as np


class TargetCodecError(ValueError):
    pass


class BaseTargetCodec(ABC):
    """Codec for encoding semantic targets into numeric model-ready tensors."""

    name = "base_target_codec"
    task_type = "unknown"

    def __init__(self) -> None:
        self._output_dim: int | None = None
        self._target_names: tuple[str, ...] | None = None

    @abstractmethod
    def fit(
        self,
        values: Sequence[Any],
        *,
        target_key: str,
        target_names: Sequence[str] | None = None,
    ) -> "BaseTargetCodec":
        ...

    @abstractmethod
    def encode(self, value: Any) -> np.ndarray:
        ...

    def decode(self, value: np.ndarray) -> Any:
        return np.asarray(value, dtype=float)

    @property
    def output_dim(self) -> int:
        if self._output_dim is None:
            raise TargetCodecError(f"Codec '{self.name}' is not fitted")
        return int(self._output_dim)

    @property
    def target_names(self) -> tuple[str, ...] | None:
        return self._target_names

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": str(self.name),
            "task_type": str(self.task_type),
            "output_dim": None if self._output_dim is None else int(self._output_dim),
            "target_names": None if self._target_names is None else list(self._target_names),
        }


class NumericTargetCodec(BaseTargetCodec):
    name = "numeric"
    task_type = "regression"

    def fit(
        self,
        values: Sequence[Any],
        *,
        target_key: str,
        target_names: Sequence[str] | None = None,
    ) -> "NumericTargetCodec":
        if not values:
            raise TargetCodecError("NumericTargetCodec.fit expects non-empty values")

        dim: int | None = None
        for value in values:
            vec = _as_numeric_vector(value)
            if dim is None:
                dim = int(vec.size)
            elif int(vec.size) != int(dim):
                raise TargetCodecError(
                    f"Numeric target size mismatch: expected {dim}, got {vec.size}"
                )

        if dim is None:
            dim = 1

        if target_names is not None:
            names = tuple(str(n) for n in target_names)
            if len(names) != int(dim):
                raise TargetCodecError(
                    f"target_names size mismatch: len(target_names)={len(names)} vs dim={dim}"
                )
            self._target_names = names
        else:
            if int(dim) == 1:
                self._target_names = (str(target_key),)
            else:
                self._target_names = tuple(f"{target_key}[{i}]" for i in range(int(dim)))

        self._output_dim = int(dim)
        return self

    def encode(self, value: Any) -> np.ndarray:
        vec = _as_numeric_vector(value)
        if self._output_dim is None:
            raise TargetCodecError(f"Codec '{self.name}' is not fitted")
        if int(vec.size) != int(self._output_dim):
            raise TargetCodecError(
                f"Numeric target size mismatch at encode: expected {self._output_dim}, got {vec.size}"
            )
        return vec

    def decode(self, value: np.ndarray) -> Any:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if self.output_dim == 1:
            return float(arr[0])
        return arr


class BinaryTargetCodec(BaseTargetCodec):
    name = "binary"
    task_type = "classification"

    _TRUE_SET = {True, 1, "1", "true", "yes", "y", "t", "on"}
    _FALSE_SET = {False, 0, "0", "false", "no", "n", "f", "off"}

    def fit(
        self,
        values: Sequence[Any],
        *,
        target_key: str,
        target_names: Sequence[str] | None = None,
    ) -> "BinaryTargetCodec":
        if not values:
            raise TargetCodecError("BinaryTargetCodec.fit expects non-empty values")

        for value in values:
            _ = self._to_binary(value)

        self._output_dim = 1
        if target_names is not None:
            names = tuple(str(n) for n in target_names)
            if len(names) != 1:
                raise TargetCodecError("Binary target expects exactly 1 target name")
            self._target_names = names
        else:
            self._target_names = (str(target_key),)
        return self

    def _to_binary(self, value: Any) -> int:
        if isinstance(value, (np.bool_, bool)):
            return int(bool(value))

        if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            return int(value)

        if isinstance(value, str):
            v = value.strip().lower()
            if v in self._TRUE_SET:
                return 1
            if v in self._FALSE_SET:
                return 0

        if value in self._TRUE_SET:
            return 1
        if value in self._FALSE_SET:
            return 0

        raise TargetCodecError(f"BinaryTargetCodec cannot parse value '{value}'")

    def encode(self, value: Any) -> np.ndarray:
        return np.asarray([float(self._to_binary(value))], dtype=float)

    def decode(self, value: np.ndarray) -> Any:
        arr = np.asarray(value, dtype=float).reshape(-1)
        return int(arr[0] >= 0.5)


class CategoricalTargetCodec(BaseTargetCodec):
    name = "categorical"
    task_type = "classification"

    def __init__(self, *, vocab: Sequence[Any] | None = None) -> None:
        super().__init__()
        self.vocab = None if vocab is None else tuple(vocab)
        self._class_to_index: Dict[Any, int] = {}
        self._index_to_class: Dict[int, Any] = {}

    def fit(
        self,
        values: Sequence[Any],
        *,
        target_key: str,
        target_names: Sequence[str] | None = None,
    ) -> "CategoricalTargetCodec":
        if not values:
            raise TargetCodecError("CategoricalTargetCodec.fit expects non-empty values")

        classes: list[Any]
        if self.vocab is not None:
            classes = list(self.vocab)
        else:
            classes = []
            seen = set()
            for value in values:
                if isinstance(value, (list, tuple, dict, np.ndarray)):
                    raise TargetCodecError(
                        f"Categorical target expects scalar labels, got {type(value).__name__}"
                    )
                try:
                    key = value if isinstance(value, (str, int, float, bool)) else value
                    if key not in seen:
                        seen.add(key)
                        classes.append(value)
                except TypeError as exc:
                    raise TargetCodecError(
                        f"Categorical target label must be hashable, got {type(value).__name__}"
                    ) from exc

        if len(classes) < 2:
            raise TargetCodecError("Categorical target expects at least 2 classes")

        self._class_to_index = {c: i for i, c in enumerate(classes)}
        self._index_to_class = {i: c for c, i in self._class_to_index.items()}

        for value in values:
            if value not in self._class_to_index:
                raise TargetCodecError(f"Unknown class '{value}' for categorical target")

        self._output_dim = 1
        if target_names is not None:
            names = tuple(str(n) for n in target_names)
            if len(names) != 1:
                raise TargetCodecError("Categorical target expects exactly 1 target name")
            self._target_names = names
        else:
            self._target_names = (str(target_key),)

        return self

    def encode(self, value: Any) -> np.ndarray:
        if value not in self._class_to_index:
            raise TargetCodecError(f"Unknown class '{value}' for categorical target")
        idx = int(self._class_to_index[value])
        return np.asarray([float(idx)], dtype=float)

    def decode(self, value: np.ndarray) -> Any:
        idx = int(np.asarray(value, dtype=float).reshape(-1)[0])
        if idx not in self._index_to_class:
            raise TargetCodecError(f"Unknown class index '{idx}' for categorical target")
        return self._index_to_class[idx]

    def metadata(self) -> Dict[str, Any]:
        out = super().metadata()
        out.update(
            {
                "classes": [self._index_to_class[i] for i in sorted(self._index_to_class.keys())],
            }
        )
        return out


def _as_numeric_vector(value: Any) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except Exception as exc:
        raise TargetCodecError(f"Numeric target conversion failed for type '{type(value).__name__}'") from exc

    arr = arr.reshape(-1)
    if arr.size == 0:
        raise TargetCodecError("Numeric target must not be empty")
    if not np.all(np.isfinite(arr)):
        raise TargetCodecError("Numeric target contains non-finite values")
    return arr


def clone_target_codec(codec: BaseTargetCodec) -> BaseTargetCodec:
    if not isinstance(codec, BaseTargetCodec):
        raise TypeError(f"codec must be BaseTargetCodec, got {type(codec).__name__}")
    return deepcopy(codec)


def default_target_codecs() -> Dict[str, BaseTargetCodec]:
    return {
        "numeric": NumericTargetCodec(),
        "binary": BinaryTargetCodec(),
        "categorical": CategoricalTargetCodec(),
    }


def infer_target_codec_key(values: Sequence[Any]) -> str:
    first: Any | None = None
    for value in values:
        if value is not None:
            first = value
            break

    if first is None:
        return "numeric"

    if isinstance(first, (bool, np.bool_)):
        return "binary"

    if isinstance(first, str):
        key = first.strip().lower()
        if key in BinaryTargetCodec._TRUE_SET or key in BinaryTargetCodec._FALSE_SET:
            return "binary"
        return "categorical"

    if isinstance(first, (int, float, np.integer, np.floating)):
        return "numeric"

    try:
        arr = np.asarray(first, dtype=float)
    except Exception:
        return "categorical"

    if arr.size == 1 and isinstance(first, (list, tuple, np.ndarray)):
        return "numeric"

    if np.issubdtype(arr.dtype, np.number):
        return "numeric"

    return "categorical"


TargetCodec = BaseTargetCodec
