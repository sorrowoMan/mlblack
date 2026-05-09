from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class ProcessedDataset:
    """Container for already-processed ML data.

    Note:
    - mlblack assumes upstream data processing is done.
    - This object only carries finalized splits for training/validation/testing.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray | None = None
    y_valid: np.ndarray | None = None
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    feature_names: Sequence[str] | None = None
    target_names: Sequence[str] | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Cell:
    """One cell in a sample; payload can be any modality object."""

    name: str
    payload: Any
    modality: str = "value"
    labels: Mapping[str, Any] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Sample:
    """One sample composed of multiple heterogeneous cells."""

    sample_id: str
    cells: Mapping[str, Cell]
    labels: Mapping[str, Any] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SampleDataset:
    """Object-first dataset protocol for multi-modal surrogate training."""

    samples: Sequence[Sample]
    target_key: str = "target"
    feature_cell_keys: Sequence[str] | None = None
    target_names: Sequence[str] | None = None
    description: str | None = None


class SurrogateArtifact(Protocol):
    """Runtime contract consumed by optimization frameworks."""

    artifact_id: str
    feature_names: Sequence[str]
    target_names: Sequence[str]
    metadata: Dict[str, Any]

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def uncertainty(self, X: np.ndarray) -> np.ndarray: ...

    def validity(self, X: np.ndarray) -> np.ndarray: ...

    def save(self, out_dir: str) -> None: ...
