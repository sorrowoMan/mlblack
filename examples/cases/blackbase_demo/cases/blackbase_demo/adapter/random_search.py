"""用于 substrate 演示的最小随机搜索策略。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from mlblack.core.adapter import OptimizerAdapter
from mlblack.core.types import Feedback, UnknownState


class DemoRandomSearchAdapter(OptimizerAdapter):
    """从 pipeline 初始化或扰动候选，不接管 Trainer 生命周期。"""

    name = "demo_random_search"

    def __init__(self, batch_size: int = 4):
        self.batch_size = max(1, int(batch_size))

    def propose(self, control: Any, context: Mapping[str, Any]) -> Sequence[UnknownState]:
        representation = control.representation_pipeline
        if control.best_state is None:
            return tuple(representation.init(context) for _ in range(self.batch_size))
        return tuple(
            representation.mutate(control.best_state, context)
            for _ in range(self.batch_size)
        )

    def update(
        self,
        control: Any,
        candidates: Sequence[UnknownState],
        feedback: Sequence[Feedback],
        context: Mapping[str, Any],
    ) -> None:
        del control, candidates, feedback, context

    def get_state(self) -> Mapping[str, Any]:
        return {"batch_size": self.batch_size}
