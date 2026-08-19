"""用于 substrate 演示的最小随机搜索策略。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from nsgablack.adapters import AlgorithmAdapter


class DemoRandomSearchAdapter(AlgorithmAdapter):
    """从 pipeline 初始化或扰动候选，不接管 Trainer 生命周期。"""

    name = "demo_random_search"

    def __init__(self, batch_size: int = 4):
        super().__init__(name=self.name)
        self.batch_size = max(1, int(batch_size))

    def propose(self, control: Any, context: Mapping[str, Any]) -> Sequence[Any]:
        return tuple(control.init_population(self.batch_size, context))

    def update(
        self,
        control: Any,
        candidates: Sequence[Any],
        feedback: Any,
        context: Mapping[str, Any],
    ) -> None:
        del control, candidates, feedback, context

    def get_state(self) -> Mapping[str, Any]:
        return {"batch_size": self.batch_size}
