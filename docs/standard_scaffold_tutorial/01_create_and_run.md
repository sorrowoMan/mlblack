# 01. 创建并运行第一个标准项目

这一章从最小可运行项目开始。目标不是教某个模型，而是固定标准脚手架：数据、pipeline、trainer spec、fit、report、artifact 每一步都可审计。

## 1. 运行位置

```powershell
Set-Location "C:\Users\hp\Desktop\新建文件夹 (2)"
```

本教程默认直接使用本地源码。正式 case 的 runner 应带 `_bootstrap.py`，不要依赖用户手工设置 `PYTHONPATH`。

## 2. 最小回归任务

```python
import numpy as np

from mlblack.assembly import build_pipeline, build_trainer
from mlblack.pipeline.data_views import train_valid_split

X = np.linspace(-1.0, 1.0, 80).reshape(-1, 1)
y = 1.0 + 2.0 * X[:, 0]

raw = train_valid_split(
    X,
    y,
    valid_ratio=0.2,
    feature_names=("x",),
)

pipeline = build_pipeline({
    "name": "first_pipeline",
    "components": [
        "zscore",
        "feature_space",
    ],
})
data = pipeline.fit_transform(raw)

trainer = build_trainer(
    {
        "preset": "orthogonal_linear_point",
        "run_name": "first_linear_run",
        "params": {
            "learning_rate": 0.05,
            "l2": 0.0,
        },
        "capabilities": ["resource_audit"],
        "resource_context": {
            "device": "cpu",
            "threads": 1,
            "namespace": "tutorial.first",
        },
    },
    data=data,
)

result = trainer.fit(max_steps=40)
print(result.report["best_score"])
print(result.report["representation"])
print(result.report["problem"])
print(result.report["adapter"])
```

## 3. 这段代码的职责拆解

| 步骤 | 对象 | 说明 |
| --- | --- | --- |
| 1 | `train_valid_split` | 生成 `NumericDataView` |
| 2 | `build_pipeline` | 声明数据处理链 |
| 3 | `pipeline.fit_transform` | 输出准备好的数据视图 |
| 4 | `build_trainer` | 只构造一个 inner trainer |
| 5 | `trainer.fit` | 单 trainer 内部优化 |
| 6 | `result.report` | 审计模型、问题、adapter、资源、contract |

关键点：

```text
Problem 吃数据。
Representation 解码模型。
Adapter 更新候选。
Trainer 只跑一个 inner training lifecycle。
```

## 4. 不要把外层编排塞进 spec

错误示例：

```python
spec = {
    "preset": "orthogonal_linear_point",
    "workflow": {
        "stages": ["baseline", "residual"],
    },
    "runtime": {
        "backend": "thread",
    },
    "resource_request": {
        "gpu": 1,
    },
}
```

正确拆法：

```text
nsgablack case config:
  stages / groups / parallel / resource lease / solver budget

mlblack trainer spec:
  preset / params / resource_context / capabilities / biases
```

## 5. 第一个 artifact

```python
from mlblack.core import ArtifactBuilder

bundle = ArtifactBuilder().build(trainer, result)
print(bundle.describe())
bundle.save("runs/first_linear/artifact_bundle")
```

Artifact 不是日志。它是可复现边界：

```text
model_artifact:
  保存 best_model 的类型、family、head、metadata

trainer_state:
  保存恢复/回放所需状态摘要

run_report:
  保存 metrics、components、resources、metadata
```

## 6. 手工装配一个 trainer

如果你在开发新组件，可以绕过 preset，直接装配：

```python
from mlblack.adapters import GradientDescentAdapter, GradientDescentConfig
from mlblack.core import Trainer
from mlblack.pipeline.data_views import NumericDataView
from mlblack.problems import SupervisedRegressionProblem
from mlblack.representations import OrthogonalPointLinearRepresentation

representation = OrthogonalPointLinearRepresentation.from_data(
    data.X_train,
    feature_names=data.effective_feature_names,
)
problem = SupervisedRegressionProblem(data)
adapter = GradientDescentAdapter(GradientDescentConfig(learning_rate=0.05))

trainer = Trainer(
    problem=problem,
    representation=representation,
    adapter=adapter,
    run_name="manual_linear",
)
result = trainer.fit(max_steps=40)
```

这个写法适合单元测试和新组件 smoke，但正式 case 仍建议把装配逻辑放进 `build_trainer.py` 或 `build_case.py`。

## 7. 标准 case 目录

正式 case 不要写成一个很长的脚本。建议结构：

```text
examples/cases/<case>/
  README.md
  build_case.py or build_solver.py
  run_case.py or run_solver.py
  config/
    case_config.py
  problem/
    data.py
    factories.py
  pipeline/
    representation.py
    transforms.py
  reporting/
    report_writer.py
```

职责：

| 文件 | 职责 |
| --- | --- |
| `run_case.py` | CLI 薄入口，只解析参数和调用 builder |
| `build_case.py` | 组装 trainer/problem/model composition surface |
| `build_solver.py` | 组装 nsgablack outer solver |
| `config/` | 可复现配置，不散落 magic number |
| `problem/` | 数据、problem、task factory |
| `pipeline/` | data transform、outer representation adapter |
| `reporting/` | summary、artifact、dashboard 输出 |

## 8. 单 trainer 与复杂 case 的分界线

| 如果你只需要 | 用 |
| --- | --- |
| 一个模型训练一次 | `build_trainer` |
| 一个模型 spec 搜索 | `build_trainer` + estimator/neural/symbolic representation |
| 一个模型产出 artifact | `ArtifactBuilder` |
| 多阶段训练 | `nsgablack` outer stage + mlblack inner trainers |
| 多模型并行训练 | `nsgablack` group/parallel + mlblack trainer specs |
| 组合多个已训练模型 | `IntegratedPredictionModel` |
| 根据上阶段模型生成下阶段数据 | `ModelConditionedTargetComponent` |

## 9. 运行现有 smoke

```powershell
python examples\orthogonal_point_demo.py
python -m pytest -q tests\test_model_integration.py
```

如果改到神经图后端：

```powershell
python -m pytest -q tests\test_neural_graph_codec.py
```

如果改到跨框架/符号：

```powershell
python examples\cases\symbolic_orthogonal_nested\run_solver.py --check
```

## 10. 常见错误

### 10.1 unknown preset

检查 `mlblack/assembly/builders.py` 的 `_build_preset_trainer(...)` 是否注册。

### 10.2 adapter requires gradients

`GradientDescentAdapter` 需要 `feedback.gradients`。如果 problem 无梯度，换 random/search adapter，或实现 problem-owned gradient hook。

### 10.3 backend capability missing

后端不支持某能力时应该报错。不要在 component 内偷偷换 backend。

### 10.4 report 缺 components/contracts

说明装配绕过了标准 `Trainer` surface。正式 case 必须能看到：

```text
representation
problem
adapter
resources
compute_backend
contracts
state_signature
```
