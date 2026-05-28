---
name: unified-dev
description: 跨框架 nsgablack + mlblack 协作开发。构建跨两框架的功能时使用：嵌套优化、符号学习外搜索、神经架构搜索、多阶段训练管线、或任何组合外优化（nsgablack）与内 ML 训练（mlblack）的编排场景。
metadata:
  short-description: nsgablack + mlblack 跨框架开发
  projects: nsgablack, mlblack
---

# 统一框架栈开发（nsgablack + mlblack）

## 两个仓库，一个框架栈

| | nsgablack | mlblack |
|---|---|---|
| 路径 | `C:\Users\hp\Desktop\nsgablack` | `C:\Users\hp\Desktop\mlblack` |
| 职责 | 外层编排 | 内层 ML 语义 |
| Catalog | `python -m nsgablack catalog ...` | `python -m mlblack catalog ...` |

## 职责边界

如果问题回答"怎么调度/并行/编排/分配资源" → **nsgablack**
如果问题回答"怎么解码/评估/训练/表示 ML 模型" → **mlblack**

## 通信面

```
nsgablack outer solver
  → ResourceLease → ResourceContext JSON
  → component_overrides / inner task payload
  → mlblack inner training (Trainer / proxy)
  → Artifact / result payload → 返回 nsgablack
```

## 集成包

`mlblack/integrations/nsgablack_*/` — nsgablack 侧接口。这些包可以 import nsgablack。mlblack 核心不得 import nsgablack。

## 关键跨框架模式

1. **嵌套优化**：nsgablack 外层 solver 搜索超参；每个候选启动 mlblack 内层 Trainer
2. **符号学习**：nsgablack Stage 1 选基函数项；mlblack 拟合内层符号参数；nsgablack Stage 2 组合函数池
3. **神经架构搜索**：nsgablack 搜索 NeuralGraphSpec 参数；mlblack 训练并返回 Feedback

## 组件发现（两仓库）

```bash
# mlblack — 有哪些 ML 组件？
python -m mlblack catalog search <关键词> --show-import
python -m mlblack catalog show <key>

# nsgablack — 有哪些编排组件？
python -m nsgablack catalog search <关键词> --profile framework-core --show-import
python -m nsgablack catalog show <key> --profile framework-core
```

## 设计与规划（跨框架场景）

**规划阶段禁止扫描源码。** 跨框架设计时，先用两个仓库的 catalog 了解各自组件，再设计通信面：

```bash
# mlblack 侧
python -m mlblack catalog list --kind <kind>
python -m mlblack catalog show <key>

# nsgablack 侧
python -m nsgablack catalog list --kind <kind> --profile framework-core
python -m nsgablack catalog show <key> --profile framework-core
```

## 规则

- 绝不让 mlblack 拥有 workflow/runtime/L0/stage/parallel
- 绝不让 nsgablack 硬编码 mlblack 模型内部细节
- 跨框架案例落 `examples/cases/<case>/`（nsgablack）+ 正确的 mlblack assembly
- 只通过正式 surface 通信：ResourceContext、component_overrides、artifact payload
