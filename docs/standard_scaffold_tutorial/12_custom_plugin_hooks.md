# 12. 自定义 Plugin：10 钩子完整实战（mlblack）

mlblack capability 与 nsgablack plugin lifecycle 是统一映射关系。  
本章给你一个可以直接照抄的 10 钩子样例。

## 1. 先创建插件文件

```powershell
python -m nsgablack project add-component --case my_trainer --kind plugin --name trainer_audit_plugin
```

## 2. 10 钩子清单（统一）

1. `on_solver_init`
2. `on_population_init`
3. `on_generation_start`
4. `on_evaluate_start`
5. `on_evaluate_end`
6. `on_step`
7. `on_generation_end`
8. `on_solver_finish`
9. `on_error`
10. `on_context_build`

在 mlblack 语义下，可以把 generation/step 看成训练 step 生命周期映射。

## 3. 可运行样例（全钩子）

```python
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from nsgablack.plugins.base import Plugin


class TrainerAuditPlugin(Plugin):
    def __init__(self, name: str = "trainer_audit_plugin"):
        super().__init__(name=name, priority=80)
        self.events = []
        self._t0 = None

    def _log(self, event: str, payload: Optional[Dict[str, Any]] = None):
        self.events.append({"ts": time.time(), "event": event, "payload": dict(payload or {})})

    def on_solver_init(self, solver):
        self._t0 = time.time()
        self._log("on_solver_init", {"solver": type(solver).__name__})

    def on_population_init(self, population, objectives, violations):
        self._log("on_population_init", {"n": 0 if population is None else len(population)})

    def on_generation_start(self, generation: int):
        self._log("on_generation_start", {"generation": generation})

    def on_evaluate_start(self, candidate, context=None):
        self._log("on_evaluate_start")

    def on_evaluate_end(self, candidate, feedback, context=None):
        self._log("on_evaluate_end", {"feedback_type": type(feedback).__name__})

    def on_step(self, solver, generation: int):
        self._log("on_step", {"generation": generation})

    def on_generation_end(self, generation: int):
        self._log("on_generation_end", {"generation": generation})

    def on_solver_finish(self, result):
        elapsed = 0.0 if self._t0 is None else time.time() - self._t0
        self._log("on_solver_finish", {"elapsed_s": elapsed})

    def on_error(self, error: BaseException, context=None):
        self._log("on_error", {"error": f"{type(error).__name__}: {error}"})

    def on_context_build(self, context):
        out = dict(context or {})
        out["plugin.trainer_audit.events"] = len(self.events)
        return out
```

## 4. 挂载

```python
trainer = ...
trainer.add_plugin(TrainerAuditPlugin())
```

## 5. 推荐插件实战组合

1. `TrainerAuditPlugin`：生命周期审计  
2. 资源审计插件：记录 ResourceContext 生效值  
3. artifact/report 插件：输出训练产物索引  
4. checkpoint 插件：恢复能力

## 6. 必须遵守

- plugin 只扩展能力，不改训练语义定义
- short-circuit 评估时输出形状必须合法
- 大对象用 snapshot/artifact ref，不塞 context
