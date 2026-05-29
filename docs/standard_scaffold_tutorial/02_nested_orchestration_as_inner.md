# 02. mlblack 作为内层的嵌套编排

本文档是 [nsgablack 嵌套编排标准规范](https://github.com/../07_nested_orchestration_standard.md) 的 mlblack 侧补充。讲解 mlblack Trainer 如何作为嵌套优化的内层被外层 nsgablack Solver 调用。

**核心**：mlblack Trainer 在嵌套架构中地位与 nsgablack Solver **完全对等**。外层不关心内层是 solver 还是 trainer——只调用 `build_solver()`。

---

## 1. build_solver() 契约

内层必须暴露一个 `build_solver()` 函数，接受以下标准参数：

```python
def build_solver(
    hyperparams: dict | None = None,       # 外层解码的超参
    data: tuple | None = None,             # (X_train, y_train) — 独立运行时可 None
    resource_context: dict | None = None,  # 外层注入的资源授权
    budget: int = 100,                     # 最大训练步数
):
    """Canonical assembly entry for inner trainer."""
```

返回一个已装配完毕的 `ComposableTrainer`（或等价对象），可直接 `.fit()` / `.run()`。

## 2. 内层自包含规则

内层必须能独立运行，不依赖外层：

```python
# 独立运行（内层 case 目录内）：
trainer = build_solver(data=my_data, budget=50)
result = trainer.fit()
print(result.best_score)

# 被外层调用（外层 Problem.evaluate() 内）：
trainer = build_solver(hyperparams={"lr": 0.001}, data=shared_data, budget=30)
result = trainer.fit()
return float(result.best_score)  # → 外层 objective
```

**禁止**：在内层 `build_solver()` 中 import 外层代码。

## 3. Problem 的 evaluate() 契约

mlblack 的 `LearningProblem.evaluate()` 签名：

```python
def evaluate(self, model, state=None, context=None) -> Feedback:
    """
    Args:
        model: decoded model (from representation.decode())
        state: raw unknown state vector
        context: mutable context dict
    Returns:
        Feedback(objectives, gradients, loss, metrics, constraints)
    """
```

外层不直接调 `problem.evaluate()`——外层调 `trainer.fit()`，trainer 内部调 problem。

## 4. 完整示例：超参搜索

### 4.1 项目结构

```text
nested_ml/
  project_config.py
  run_project.py
  cases/
    outer_hpo/              # nsgablack — 搜索超参
      build_solver.py
      problem/hpo_problem.py
    inner_linear/            # mlblack — 训练线性模型
      build_solver.py        # build_trainer.py 别名可用
      problem/regression_problem.py
      pipeline/linear_rep.py
```

### 4.2 内层 build_solver()

`cases/inner_linear/build_solver.py`：

```python
import numpy as np
from mlblack.core.trainer import ComposableTrainer
from mlblack.adapters.gradient_descent import GradientDescentAdapter
from problem.regression_problem import RegressionProblem
from pipeline.linear_rep import LinearRepresentation


def build_solver(hyperparams=None, data=None, resource_context=None, budget=100):
    hp = hyperparams or {}
    lr = hp.get("learning_rate", 0.01)
    l2 = hp.get("l2_weight", 0.0)

    if data is None:
        X = np.random.randn(100, 5)
        y = X @ np.array([1.0, 2.0, -1.0, 0.5, -0.5]) + np.random.normal(0, 0.1, 100)
        data = (X, y)

    problem = RegressionProblem(data=data)
    representation = LinearRepresentation(n_features=data[0].shape[1])
    adapter = GradientDescentAdapter(learning_rate=lr)

    trainer = ComposableTrainer(
        problem=problem,
        representation=representation,
        adapter=adapter,
        run_name="inner_linear",
    )

    if l2 > 0:
        from mlblack.bias import StateL2Bias
        trainer.add_bias(StateL2Bias(weight=l2))

    if resource_context:
        trainer.set_resource_context(resource_context)

    return trainer
```

### 4.3 外层 Problem

`cases/outer_hpo/problem/hpo_problem.py`：

```python
import sys, os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from cases.inner_linear.build_solver import build_solver as build_inner


class HPOProblem:
    name = "hpo"
    dimension = 2
    objectives = ("val_loss",)

    def __init__(self, data_path=None, inner_budget=30):
        self.X = np.random.randn(200, 5)
        self.y = self.X @ np.array([1, 2, -1, 0.5, -0.5])
        self.inner_budget = inner_budget

    def evaluate(self, x):
        x = np.asarray(x, dtype=float).ravel()
        hp = {"learning_rate": 10 ** float(x[0]), "l2_weight": 10 ** float(x[1])}
        try:
            trainer = build_inner(hyperparams=hp, data=(self.X, self.y), budget=self.inner_budget)
            result = trainer.fit(max_steps=self.inner_budget)
            return np.array([float(result.best_score or 1e10)], dtype=float)
        except Exception:
            return np.array([1e10], dtype=float)

    def evaluate_constraints(self, x):
        return np.zeros(0, dtype=float)
```

### 4.4 外层 build_solver()

`cases/outer_hpo/build_solver.py`：

```python
import numpy as np
from nsgablack.core import ComposableSolver
from nsgablack.adapters import RandomSearchAdapter
from nsgablack.representation import RepresentationPipeline, UniformInitializer, ClipRepair
from problem.hpo_problem import HPOProblem


def build_solver():
    problem = HPOProblem(inner_budget=30)

    pipeline = RepresentationPipeline(
        initializer=UniformInitializer(low=np.array([-4.0, -4.0]), high=np.array([-1.0, 1.0])),
        repair=ClipRepair(low=np.array([-4.0, -4.0]), high=np.array([-1.0, 1.0])),
    )

    adapter = RandomSearchAdapter(n_points=50)

    solver = ComposableSolver(
        problem=problem,
        representation_pipeline=pipeline,
        adapter=adapter,
    )
    return solver
```

### 4.5 编排配置

`project_config.py`：

```python
STAGES = [
    {"name": "hpo_stage", "cases": ["outer_hpo"], "policy": "run_all_in_parallel"},
]
GROUPS = {"default": {"stages": ["hpo_stage"]}}
```

### 4.6 运行

```powershell
# 从项目根启动
python run_project.py
```

## 5. 多 Trainer 被外层编排

外层可以并行调用多个不同的 trainer：

```python
class MultiModelHPOProblem:
    def evaluate(self, x):
        hp = self._decode(x)
        losses = {}
        for model in ("linear", "mlp", "xgboost"):
            trainer = build_inner(model_type=model, hyperparams=hp[model])
            result = trainer.fit()
            losses[model] = float(result.best_score or 1e10)
        return np.array(list(losses.values()), dtype=float)
```

或者用 `project_config.py` 的 STAGES 并行编排多个 trainer case：

```python
STAGES = [
    {"name": "train_all", "cases": ["trainer_a", "trainer_b"], "policy": "run_all_in_parallel"},
]
```

编排不区分 solver 和 trainer。

## 6. 检查清单

- [ ] 内层独立可运行
- [ ] `build_solver()` 接受 `hyperparams`/`data`/`resource_context`/`budget`
- [ ] 内层不 import 外层
- [ ] 外层 Problem.evaluate() 中正确调用内层 `build_solver()` + `.fit()`
- [ ] ResourceContext 显式从外层传入内层
- [ ] 从项目根 `python run_project.py` 成功
