---
name: mlblack-dev
description: 在 mlblack 优化优先 ML 框架中开发。新增 Problem、Adapter、Representation、Codec、Head、Capability、时序模型路由、预设时使用。教会 agent 使用 catalog 驱动组件发现，而非扫描源码。
metadata:
  short-description: mlblack 组件开发
  project: mlblack
  framework: optimization-first ML
---

# mlblack 开发

## 第一步：用 catalog 了解组件

**不要扫描源码来理解组件。** catalog 有自动发现的契约信息。

```bash
cd C:\Users\hp\Desktop\mlblack

# 按 kind 列出组件
python -m mlblack catalog list --kind problem
python -m mlblack catalog list --kind adapter
python -m mlblack catalog list --kind codec
python -m mlblack catalog list --kind preset

# 按关键词搜索
python -m mlblack catalog search <关键词> --show-import

# 查看完整契约（requires/provides/mutates/import_path）
python -m mlblack catalog show <key>

# framework-core 口径（排除 example/doc）
python -m mlblack catalog list --kind problem --profile framework-core
```

可用的 kind：`adapter`、`problem`、`representation`、`codec`、`head`、`preset`、`model`、`provider`、`capability`、`bias`、`pipeline`、`data_view`、`assembly`、`trainer`、`core`。

## 架构（正交分层）

| 层 | mlblack | nsgablack | 不能做的事 |
|---|---|---|---|
| 控制平面 | Trainer | Solver | — |
| 策略 | OptimizerAdapter | Adapter | 直接读数据 |
| 表示 | ModelRepresentation + Codec + Head | Representation | — |
| 评估 | LearningProblem | Problem | 唯一稳定吃数据的层 |
| 能力 | Capability | Plugin | 改变优化语义 |
| 软引导 | OptimizationBias | Bias | 替代硬约束 |

## 硬规则

- `mlblack` 不拥有：workflow、runtime、L0 resource、并行调度、stage/group/event 编排。这些归 `nsgablack`。
- 每个组件必须有 `context_requires/provides/mutates` class attr 或 `ComponentContract`。
- 大对象进 SnapshotStore/ArtifactBundle，不进 context。
- 新增组件后验证 catalog 能发现：`python -m mlblack catalog search <关键词>`
- 如果 auto-discovery 发现不了（如 `@classmethod` 工厂），加到 `catalog/registry.py` 的 `_default_entries()`。

## 设计与规划（EnterPlanMode 时也必须遵守）

**规划阶段同样禁止扫描源码。** 使用 catalog 做组件调研：

```bash
# 1. 了解现有同类组件 — 不读源码，用 catalog
python -m mlblack catalog list --kind <kind> --profile framework-core

# 2. 查看具体契约 — 不读源码，用 catalog show
python -m mlblack catalog show <key>

# 3. 搜索相关实现 — 不 grep 源码，用 catalog search
python -m mlblack catalog search <关键词> --show-import
```

**Explore agent 的 prompt 必须指示它用 catalog 命令，禁止扫描完整源文件。**
调研目的不是理解每一行代码，而是找到：已有接口签名、契约（requires/provides）、import_path、以及可复用的模式。

## 新增组件检查清单

1. 选对层：Problem / Adapter / Representation / Codec / Head / Capability / Bias / Pipeline
2. 加 `context_requires/provides/mutates` + `ComponentContract`
3. 更新父级 `__init__.py` 和 `__all__`
4. 验证 catalog：`python -m mlblack catalog search <关键词>`
5. 缺失则加到 `catalog/registry.py` `_default_entries()`
6. 加 preset builder 到 `presets/`
7. 加 assembly 入口到 `assembly/builders.py`
8. 跑测试：`python -m pytest tests/ -x -q`
9. 跑语法检查：`python -m compileall -q mlblack`

## 常用命令

```powershell
python -m pytest tests/ -x -q                           # 测试
python -m compileall -q mlblack                         # 语法检查
python -m mlblack catalog list --kind <kind>            # 组件发现
python -m mlblack catalog show <key>                    # 契约详情
```
