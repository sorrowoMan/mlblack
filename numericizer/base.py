from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from core.common.contracts import ProcessedDataset, Sample, SampleDataset


class BaseNumericizer(ABC):
    """Converts object-first samples into numeric training matrices."""

    name = "base_numericizer"

    @abstractmethod
    def from_sample_dataset(self, data: SampleDataset) -> ProcessedDataset:
        ...

    def fit(self, data: SampleDataset) -> "BaseNumericizer":
        _ = data
        return self

    def transform_features(self, samples: Sequence[Sample]) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} does not implement transform_features")

    def transform_targets(self, samples: Sequence[Sample]) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} does not implement transform_targets")

    def to_processed(self, data: ProcessedDataset | SampleDataset) -> ProcessedDataset:
        if isinstance(data, ProcessedDataset):
            return data
        if isinstance(data, SampleDataset):
            return self.from_sample_dataset(data)
        raise TypeError("data must be ProcessedDataset or SampleDataset")
