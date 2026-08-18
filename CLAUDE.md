# CLAUDE.md

## 位置

`C:\Users\hp\Desktop\mlblack` — 优化优先的 ML 框架。

## 常用命令

```
python -m compileall -q mlblack
python -m pytest tests/ -x -q
python -m mlblack catalog list --kind <kind>
python -m mlblack catalog search <query> --show-import
python -m mlblack catalog show <key>

```

## Catalog 驱动（强制）

所有阶段**绝不要扫描源文件**，用 catalog。可用 kind：`adapter` `problem` `representation` `codec` `head` `preset` `model` `provider` `capability` `bias` `pipeline` `data_view` `assembly` `trainer` `core`。`--profile framework-core` 排除 examples。

## 架构

| 层 | 组件 | 目录 |
|---|---|---|
| 控制平面 | Trainer | `core/` |
| 策略 | OptimizerAdapter | `adapters/` |
| 表示 | ModelRepresentation + Codec + Head | `representations/` |
| 评估 | LearningProblem | `problems/` |
| 能力 | Capability | `capabilities/` |
| 软引导 | OptimizationBias | `bias/` |
| 数据 | Pipeline / DataView / ModelConditionedTarget | `pipeline/` |

组合能力：
- 完整 Trainer/Solver 子任务通过 BlackBase `CaseStageRunner` 与 `CaseRunRequest` 编排；
- 仅共享一个 Case 生命周期的轻量值流使用 `DataPipeline`；
- `DataPipeline`：有序数据变换链（fit → transform）
- `ModelConditionedTargetComponent`：基于已训练模型的 target 变换（残差学习）
- `IntegratedPredictionModel`：多模型推理期集成

Project/Case 编排属于 BlackBase，不属于 mlblack 或 nsgablack 的私有能力；
nsgablack 只负责优化搜索语义，mlblack 只负责学习语义。

## 规则

- Adapter 不直接读数据；Problem 是数据唯一消费方
- 大对象 → ArtifactBundle，不进 context
- 新增组件必须显式登记到 `catalog/entries/<kind>.toml`；源码扫描只用于 Doctor 覆盖率审计
- 每个组件声明 `ComponentContract` 或 `context_requires/provides/mutates`
