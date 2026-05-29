# mlblack 标准脚手架教程

`mlblack` 和 `nsgablack` 共享同一个内核。**Solver = Trainer**，两者的标准脚手架完全一致，差异仅在 catalog 注册语义。`mlblack` 是 `nsgablack` 的 ML 语义特化——本质是 pipeline 中多了 model-level encode/decode。

## 0. 一句话总览

```text
nsgablack + mlblack 统一架构:
  共享同一套标准脚手架（project/scaffold/case_template）
  共享同一个 Plugin 体系（plugins/base.py, 10 个钩子超集）
  Solver 和 Trainer 是同一抽象层级

mlblack 的独有语义:
  pipeline 内的 model-level encode/decode（Codec/Head/ModelRepresentation）
  ComposableTrainer（等价于 ComposableSolver + representation pipeline）
  ML 语义组件：DataView、Spec、Problem、Adapter（梯度下降系）

nsgablack 的独有语义:
  多目标/Pareto、外层搜索、多策略编排、L0 资源调度
```

不要把二者混在一起：

```text
错误方向:
  在 mlblack 里新增 HybridTrainer / Workflow / StageRunner / RuntimeBackend / ResourceAllocator。
  在 mlblack 里维护独立的脚手架系统（scaffold_legacy.py 已删除）。

正确方向:
  在 mlblack 里新增可组合 ML 语义组件，通过统一的 Plugin 体系挂载能力。
  脚手架统一走 `nsgablack.project.scaffold`。
```

## 1. 教程结构

| 章节 | 主题 | 解决的问题 |
| --- | --- | --- |
| [00_assembly_api_reference.md](00_assembly_api_reference.md) | 架构地图与 API 速查 | 每层是什么、常用 API 是什么、哪些字段禁止放在 mlblack |
| [01_create_and_run.md](01_create_and_run.md) | 第一个标准项目 | 从 `NumericDataView` 到 `build_trainer`、`fit`、`report`、ResourceContext 注入 |
| **[02_nested_orchestration_as_inner.md](02_nested_orchestration_as_inner.md)** | **mlblack 作为内层的嵌套编排** | **mlblack 怎么被 nsgablack 外层调用、`build_trainer()` 签名、资源上下文传递、SerialTrainer 多阶段** |
| [03_model_composition_and_io_contract.md](03_model_composition_and_io_contract.md) | 模型整合与 I/O contract | 残差、stacking、多模态、late fusion 怎么严谨表达、artifact 跨 stage 流转 |
| **[04_nsgablack_orchestration_and_resource_layers.md](04_nsgablack_orchestration_and_resource_layers.md)** | **nsgablack 编排与统一的 L0 资源层** | **嵌套编排标准、ResourceContext 垂直流、Stage/Group/SerialTrainer、符号学习、多 Trainer 并行** |
| [05_symbolic_nested_case.md](05_symbolic_nested_case.md) | 符号 nested case | 符号学习为什么是 outer structure search + inner fitting、outer/inner 文件结构、资源授权流 |
| [06_validation_catalog_artifacts.md](06_validation_catalog_artifacts.md) | 验收、catalog、artifact | doctor、context contract、artifact viewer、catalog dashboard、跨层 artifact 审计 |
| [07_benchmark_dashboard_resource.md](07_benchmark_dashboard_resource.md) | benchmark、dashboard、资源审计 | benchmark suite、实验查询、资源上下文、性能报告、L0 资源使用报表 |
| [08_complex_pattern_catalog.md](08_complex_pattern_catalog.md) | 复杂组合模式目录 | 把可实现的复杂组合模式系统列出来，说明怎么落层、L0 资源需求 |

## 2. 标准心智模型

### 2.1 单 inner trainer

```text
raw data
  -> numericizer / DataPipeline
  -> NumericDataView
  -> build_trainer(spec, data)
  -> Trainer.fit(...)
  -> TrainerResult / ArtifactBundle / RunReport
```

适合：线性模型、树模型、boosting spec、MLP、小 Transformer/CNN/GNN smoke、单个符号表达式参数拟合。

### 2.2 多阶段复杂训练

```text
nsgablack outer stage 1
  -> mlblack inner trainer/problem
  -> model/artifact/report

nsgablack outer stage 2
  -> consumes previous model/artifact/report
  -> mlblack data transform / next trainer / problem
  -> next model/artifact/report

mlblack IntegratedPredictionModel
  -> combines fitted component models for final inference/evaluation
```

适合：残差、boosting、stacking、多模态融合、专家模型、router + branch、symbolic nested learning。

### 2.3 后端与编排的区别

```text
backend:
  一个 inner trainer 内部怎么执行 tensor / autograd / optimizer / loss。
  例：torch, jax, tensorflow, numpy。

orchestration:
  多个 solver/trainer/stage 怎么排序、并行、分资源、失败恢复。
  归 nsgablack。
```

## 3. 关键边界表

| 需求 | 归属 | 例子 |
| --- | --- | --- |
| 一个 unknown state 怎么变成模型 | `mlblack.representations` / codec / head | linear, symbolic, neural graph |
| 一个模型怎么吃数据并打分 | `mlblack.problems` | supervised regression, classification, LM, DPO |
| 一个模型参数怎么更新 | `mlblack.adapters` + backend | GD, functional backprop, torch backprop |
| 一个训练产物怎么描述和保存 | `mlblack.core.artifacts` | model artifact, trainer state, run report |
| 多个模型预测怎么合成 | `mlblack.models.composition` | additive, mean, late fusion |
| 下一阶段 target 怎么由旧模型生成 | `mlblack.pipeline.model_conditioning` | residual target, stacking feature |
| 多个 trainer/stage 怎么跑 | `nsgablack` | serial, group, parallel, event |
| 设备/线程/预算怎么授权 | `nsgablack` L0，`mlblack` 被动消费 | ResourceContext |

## 4. 当前正式能力面

### 4.1 模型表示

```text
linear / orthogonal linear
symbolic expression / symbolic basis set
estimator spec: tree / boosting / sklearn MLP / xgboost
neural graph: MLP / tiny Transformer / CNN / GNN
```

### 4.2 输出语义

```text
point
interval
probability / softmax
piecewise / routed branch
symbolic basis / expression set
embedding / ranking / preference / LM heads in neural graph route
integrated prediction model for composed fitted models
```

### 4.3 后端

```text
numpy:
  CPU ndarray, MLP lowering, MSE, artifact summary

jax:
  functional gradient, MLP lowering, SGD helper

tensorflow:
  GradientTape functional gradient, MLP lowering, SGD helper

torch:
  module backward, optimizer.step, tiny Transformer/CNN/GNN, audit artifacts
```

后端能力不足必须 fail-fast，不允许静默换后端。

## 5. 标准项目落点

所有 case 使用统一的脚手架模板（与 nsgablack 完全一致）：

```text
examples/cases/<case>/
  build_solver.py          # canonical 装配入口
  build_trainer.py         # 别名: from .build_solver import build_solver as build_trainer
  run_solver.py            # CLI 薄入口
  run_trainer.py           # 别名
  config.py                # 组件注册聚合
  problem/
  pipeline/                # encode/decode/init/mutate/repair + data
  adapter/
  bias/
  plugins/                 # 统一能力层（替代 legacy capabilities/）
  evaluation/
  runtime/
  solver/
```

多层嵌套项目（Project / Case / Scaffold 三层结构）：

```text
<project_root>/
  project_config.py        # 跨 case 编排
  run_project.py           # 顶层入口
  cases/
    outer_solver/          # nsgablack 外层
    inner_trainer/         # mlblack 内层（结构完全一致）
```

规则：runner 是薄入口，真实装配逻辑进 `build_solver.py`、`problem/`、`pipeline/`、`plugins/`。`build_trainer.py` 是 `build_solver.py` 的别名，`run_trainer.py` 是 `run_solver.py` 的别名。`representation/` 不作为独立目录，编解码器是 `pipeline/` 的内部组件。

## 6. 读者应避免的误区

```text
误区 1: 多模态要写 MultiModalTrainer。
正确: 多个 branch trainer/model + IntegratedPredictionModel + nsgablack 外层编排。

误区 2: 残差要写 ResidualWorkflow。
正确: ModelConditionedTargetComponent 生成 residual target，nsgablack 排下一阶段 trainer。

误区 3: 后端选择应该写在 codec 里。
正确: trainer 指定 compute_backend，codec 按 backend session lowering。

误区 4: adapter 可以自己读 X/y。
正确: problem 是唯一稳定吃数据的位置，adapter 消费 feedback。

误区 5: mlblack 要自己管理 GPU lease。
正确: nsgablack L0 授权 ResourceContext，mlblack 被动读取和审计。
```

## 7. 最小验证命令

```powershell
Set-Location "C:\Users\hp\Desktop\新建文件夹 (2)"
python -m pytest -q tests
python -c "from mlblack.project import run_project_doctor, format_doctor_report; print(format_doctor_report(run_project_doctor('.', strict=True)))"
python examples\orthogonal_point_demo.py
```

跨框架或符号相关修改再跑：

```powershell
python examples\cases\symbolic_orthogonal_nested\run_solver.py --check
```
