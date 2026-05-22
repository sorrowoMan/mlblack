# mlblack 标准脚手架教程

这套教程按当前架构重新整理：`mlblack` 是 `nsgablack` 的 ML 特化层，不是第二套 workflow/runtime/L0。读者应先理解边界，再学习单 trainer，再学习复杂模型组合，最后把多阶段、并行、资源和外层搜索交给 `nsgablack` 标准脚手架。

## 0. 一句话总览

```text
nsgablack:
  负责外层优化、阶段、组、serial、parallel、event、resource lease、solver fanout。

mlblack:
  负责数据视图、模型表示、codec/head、problem/evaluation、inner fitting、模型整合语义、artifact/report。
```

不要把二者混在一起：

```text
错误方向:
  在 mlblack 里新增 HybridTrainer / Workflow / StageRunner / RuntimeBackend / ResourceAllocator。

正确方向:
  在 mlblack 里新增可组合 ML 语义组件，然后由 nsgablack 编排这些组件的训练顺序和资源。
```

## 1. 教程结构

| 章节 | 主题 | 解决的问题 |
| --- | --- | --- |
| [00_assembly_api_reference.md](00_assembly_api_reference.md) | 架构地图与 API 速查 | 每层是什么、常用 API 是什么、哪些字段禁止放在 mlblack |
| [01_create_and_run.md](01_create_and_run.md) | 第一个标准项目 | 从 `NumericDataView` 到 `build_trainer`、`fit`、`report` |
| [02_component_configuration.md](02_component_configuration.md) | 组件配置拆解 | data / representation / codec / head / problem / adapter / backend / artifact 怎样配 |
| [03_model_composition_and_io_contract.md](03_model_composition_and_io_contract.md) | 模型整合与 I/O contract | 残差、stacking、多模态、late fusion 怎么严谨表达 |
| [04_nsgablack_orchestration_patterns.md](04_nsgablack_orchestration_patterns.md) | nsgablack 外层编排模式 | stage、serial、group、多 solver、resource context 怎么接 mlblack |
| [05_symbolic_nested_case.md](05_symbolic_nested_case.md) | 符号 nested case | 符号学习为什么是 outer structure search + inner fitting |
| [06_validation_catalog_artifacts.md](06_validation_catalog_artifacts.md) | 验收、catalog、artifact | doctor、context contract、artifact viewer、catalog dashboard |
| [07_benchmark_dashboard_resource.md](07_benchmark_dashboard_resource.md) | benchmark、dashboard、资源审计 | benchmark suite、实验查询、资源上下文、性能报告 |
| [08_complex_pattern_catalog.md](08_complex_pattern_catalog.md) | 复杂组合模式目录 | 把可实现的复杂组合模式系统列出来，说明怎么落层 |

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

正式 case：

```text
examples/cases/<case>/
  README.md
  build_solver.py or build_case.py
  run_solver.py or run_case.py
  config/
  problem/
  pipeline/
  reporting/
```

跨框架 case：

```text
examples/cross_framework/<case>/
```

benchmark：

```text
examples/benchmarks/<benchmark>.py
examples/benchmarks/<benchmark>/
```

规则：runner 是薄入口，真实装配逻辑进 `build_*`、`problem/`、`pipeline/`、`reporting/`。

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
