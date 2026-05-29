# 01. 创建并运行第一个 mlblack 标准项目

mlblack 和 nsgablack 使用完全一致的标准脚手架。唯一的差异：mlblack 的 pipeline 多了 model-level encode/decode（Codec/Head/ModelRepresentation）。

**关键**：和 nsgablack 一样，正式运行入口在项目顶层 `run_project.py`，不在 case 内。

---

## 1. 创建项目

```powershell
python -m nsgablack project new my_ml_project
cd my_ml_project
python -m nsgablack project add-case my_trainer --type trainer
```

> `--type trainer` 和 `--type solver` 生成完全相同的目录结构。差异仅在 catalog 注册的 `kind` 字段。

## 2. 最小训练任务

编辑 `cases/my_trainer/build_solver.py`：

```python
"""My first mlblack trainer."""
import numpy as np
from mlblack.core.trainer import ComposableTrainer
from mlblack.adapters.gradient_descent import GradientDescentAdapter
from problem.regression_problem import RegressionProblem
from pipeline.linear_rep import LinearRepresentation


def build_solver():
    # 1. 准备数据
    X = np.linspace(-1.0, 1.0, 80).reshape(-1, 1)
    y = 1.0 + 2.0 * X[:, 0] + np.random.normal(0, 0.1, 80)

    # 2. 创建 Problem
    problem = RegressionProblem(data=(X, y))

    # 3. 创建 Representation（model encode/decode — mlblack 独有语义）
    representation = LinearRepresentation(n_features=1)

    # 4. 创建 Adapter
    adapter = GradientDescentAdapter(learning_rate=0.05)

    # 5. 装配 Trainer
    trainer = ComposableTrainer(
        problem=problem,
        representation=representation,
        adapter=adapter,
        run_name="first_linear",
    )
    return trainer
```

## 3. Problem 定义

`cases/my_trainer/problem/regression_problem.py`：

```python
"""Supervised regression with MSE loss."""
import numpy as np
from mlblack.core.problem import LearningProblem
from mlblack.core.types import Feedback


class RegressionProblem(LearningProblem):
    """Minimize MSE between prediction and target."""

    def __init__(self, data, *, name="regression"):
        self.X = np.asarray(data[0], dtype=float)
        self.y = np.asarray(data[1], dtype=float).ravel()
        super().__init__(name=name)

    def evaluate(self, model, state=None, context=None):
        pred = np.asarray(model, dtype=float).ravel()
        if len(pred) != len(self.y):
            pred = np.full_like(self.y, pred[0] if len(pred) else 0.0)
        residuals = pred - self.y
        mse = float(np.mean(residuals ** 2))
        return Feedback(
            objectives=np.array([mse]),
            gradients=residuals,
            loss=mse,
            metrics={"mse": mse},
        )
```

## 4. Representation 定义

`cases/my_trainer/pipeline/linear_rep.py`：

```python
"""Linear model encode/decode (mlblack pipeline component)."""
import numpy as np
from mlblack.core.representation import ModelRepresentation


class LinearRepresentation(ModelRepresentation):
    """Encode/decode a linear coefficient vector."""

    def __init__(self, n_features=1, *, name="linear"):
        self.n_features = max(1, int(n_features))
        self.dimension = n_features
        super().__init__(name=name)

    def init(self, rng=None):
        rng = rng or np.random.default_rng()
        return rng.normal(0.0, 0.1, size=(self.n_features,))

    def encode(self, coefficients):
        return np.asarray(coefficients, dtype=float).ravel()

    def decode(self, encoded):
        return np.asarray(encoded, dtype=float).ravel()

    def repair(self, state):
        # No constraints on linear coefficients
        return np.asarray(state, dtype=float)
```

## 5. 验证装配

```powershell
cd cases/my_trainer
python run_solver.py --check
# → [check] assembly ok | problem=RegressionProblem | pipeline=LinearRepresentation | adapter=GradientDescentAdapter
```

## 6. 训练

```powershell
python run_solver.py
```

输出类似：

```text
best_score: 0.0234
representation: LinearRepresentation(n_features=1)
adapter: GradientDescentAdapter(lr=0.05)
```

## 7. 从项目根运行

```powershell
cd ../..  # 回到 my_ml_project/
python run_project.py
```

## 8. mlblack 独有语义速查

| 组件 | nsgablack | mlblack |
|---|---|---|
| Pipeline 中的 encode/decode | 搜索空间向量 ↔ 候选解 | **模型参数 ↔ 向量**（Codec/Head） |
| Adapter | 搜索策略（NSGA2, VNS...） | 梯度下降策略（GD, Adam, backprop） |
| Problem | 目标/约束评估 | 损失/梯度/指标评估（LearningProblem） |
| Plugin | 统一 10 钩子体系 | 统一 10 钩子体系（同 nsgablack） |

## 9. 常见错误

| 现象 | 原因 |
|---|---|
| `adapter requires gradients` | Problem 的 Feedback 没提供 `gradients` |
| `representation has no decode_candidate` | Representation 没实现 `decode()` |
| `trainer.fit() shape mismatch` | X/y 维度不匹配，或 representation 输出维度不对 |
