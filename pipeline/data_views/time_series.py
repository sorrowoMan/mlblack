from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TimeSeriesDataView:
    """Ordered univariate time-series data with optional exogenous features."""

    y: Sequence[float] | np.ndarray
    time_index: Sequence[Any] = tuple()
    exogenous: Sequence[Sequence[float]] | np.ndarray | None = None
    exogenous_names: Sequence[str] = tuple()
    target_name: str = "target"
    frequency: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        y_arr = np.asarray(self.y, dtype=float).reshape(-1)
        if y_arr.size < 2:
            raise ValueError("TimeSeriesDataView requires at least two observations")
        object.__setattr__(self, "y", y_arr)
        if self.time_index:
            if len(tuple(self.time_index)) != y_arr.shape[0]:
                raise ValueError("time_index length must match y")
            object.__setattr__(self, "time_index", tuple(self.time_index))
        else:
            object.__setattr__(self, "time_index", tuple(range(y_arr.shape[0])))
        if self.exogenous is not None:
            exog = np.asarray(self.exogenous, dtype=float)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            if exog.ndim != 2:
                raise ValueError("exogenous must be 2D")
            if exog.shape[0] != y_arr.shape[0]:
                raise ValueError("exogenous row count must match y")
            object.__setattr__(self, "exogenous", exog)
            names = tuple(str(name) for name in self.exogenous_names) or tuple(f"exog{i}" for i in range(exog.shape[1]))
            if len(names) != exog.shape[1]:
                raise ValueError("exogenous_names length must match exogenous columns")
            object.__setattr__(self, "exogenous_names", names)
        else:
            object.__setattr__(self, "exogenous_names", tuple())

    @classmethod
    def from_values(
        cls,
        y: Sequence[float] | np.ndarray,
        *,
        time_index: Sequence[Any] = tuple(),
        exogenous: Sequence[Sequence[float]] | np.ndarray | None = None,
        exogenous_names: Sequence[str] = tuple(),
        target_name: str = "target",
        frequency: str = "",
        sort_by_time: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> "TimeSeriesDataView":
        if not sort_by_time or not time_index:
            return cls(
                y=y,
                time_index=time_index,
                exogenous=exogenous,
                exogenous_names=exogenous_names,
                target_name=target_name,
                frequency=frequency,
                metadata=dict(metadata or {}),
            )
        order = np.argsort(np.asarray(tuple(time_index)))
        y_arr = np.asarray(y, dtype=float).reshape(-1)[order]
        exog_arr = None if exogenous is None else np.asarray(exogenous, dtype=float)[order]
        time_sorted = tuple(tuple(time_index)[int(idx)] for idx in order)
        return cls(
            y=y_arr,
            time_index=time_sorted,
            exogenous=exog_arr,
            exogenous_names=exogenous_names,
            target_name=target_name,
            frequency=frequency,
            metadata={**dict(metadata or {}), "sorted_by_time": True},
        )

    @property
    def n_obs(self) -> int:
        return int(np.asarray(self.y).shape[0])

    @property
    def has_exogenous(self) -> bool:
        return self.exogenous is not None

    def tail_target(self, size: int) -> np.ndarray:
        count = int(size)
        if count <= 0 or count >= self.n_obs:
            raise ValueError("tail size must be between 1 and n_obs - 1")
        return np.asarray(self.y, dtype=float)[-count:]

    def history_before_tail(self, size: int) -> np.ndarray:
        count = int(size)
        if count <= 0 or count >= self.n_obs:
            raise ValueError("tail size must be between 1 and n_obs - 1")
        return np.asarray(self.y, dtype=float)[:-count]

    def describe(self) -> dict[str, Any]:
        return {
            "name": "time_series_data_view",
            "n_obs": int(self.n_obs),
            "target_name": str(self.target_name),
            "frequency": str(self.frequency),
            "has_exogenous": bool(self.has_exogenous),
            "exogenous_names": tuple(str(name) for name in self.exogenous_names),
            "metadata": dict(self.metadata),
        }
