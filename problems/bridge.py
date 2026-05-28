from __future__ import annotations

from typing import Any, Mapping, Sequence

from mlblack.pipeline.data_views import NumericDataView
from .proxy import MLBlackTrainingProxy, SpecMapper


def build_training_proxy(
    data: NumericDataView,
    *,
    trainer_spec: Mapping[str, Any] | None = None,
    max_steps: int = 100,
    mapper: SpecMapper | None = None,
) -> MLBlackTrainingProxy:
    return MLBlackTrainingProxy(
        data=data,
        base_trainer_spec=trainer_spec,
        max_steps=max_steps,
        mapper=mapper,
    )


def result_to_outer_tuple(result: Any) -> tuple[Any, Any]:
    payload = result.as_dict() if hasattr(result, "as_dict") else dict(result)
    return payload.get("objectives", []), payload.get("constraints", [])

