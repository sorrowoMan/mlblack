# Memory — 跨会话持久知识

## 框架组合规则（强制）

### 核心原则：组件组合 > 从头手写

所有案例应优先复用框架现有组件。只在以下情况自定义：
- **Problem**：领域特定的损失函数/评估逻辑（框架不可能穷举）
- **Representation/Codec/Head**：仅当现有 codec/head 不匹配时

### 禁止手写的内容
- Adapter：框架有 23+6 个，覆盖 DE/SA/VNS/NSGA2/3/SPEA2/MOEAD/GD/Backprop...
- Bias：框架有 66 个，覆盖图约束/收敛/多样性/局部精修/生产约束...
- 只在框架组件确实不满足需求时才自定义

## nsgablack 组件速查 (catalog profile: framework-core)

| 类别 | 数量 | 关键组件 |
|---|---|---|
| Adapter | 23 | de, sa, vns, nsga2, nsga3, spea2, moead, pattern_search, trust_region*, astar, strategy_chain, strategy_router, gradient_descent, async_event_driven |
| Bias | 66 | graph: tsp_constraint, hamiltonian_path, coloring, connectivity, matching, max_flow, shortest_path, sparsity, community, tree |
| | | algorithmic: cmaes, pso, tabu, levy, convergence*, diversity*, crowding, niche |
| | | local: gradient_descent, line_search, nelder_mead, newton, quasi_newton, trust_region |
| | | domain: constraint, feasibility, dynamic_penalty, production*, scheduling, risk, safety |

## mlblack 组件速查 (catalog profile: default/framework-core)

| 类别 | 数量 | 关键组件 |
|---|---|---|
| Adapter | 6 | gradient_descent, torch_backprop, neural_graph_backprop, functional_backprop, random_search, estimator_spec_search |
| Problem | 19 | supervised_regression/classification, temporal_neural_*, tiny_cnn/gnn/transformer_*, symbolic_*, piecewise_regression |
| Representation | 13 | orthogonal_linear, numpy_mlp, neural_graph, piecewise, symbolic_*, baseline_forecast |
| Codec | 16 | linear_point, orthogonal_linear_point, numpy_mlp, neural_graph, temporal_*(7), tabular_tabnet, symbolic_* |
| Head | 12 | point, interval, softmax, logistic, piecewise, normal, poisson, negbinomial, calibration, symbolic_basis_set |

## 案例设计模式

### 标准组合模式
```text
Problem (自定义，必须) 
  + Representation/Codec/Head (优先复用)
  + Adapter (复用框架，不手写)
  + Bias (复用框架组合，不手写)
  = 完整案例
```

### 组合案例示例
- **TSP**: adapter.sa + bias.graph_tsp_constraint + bias.graph_hamiltonian_constraint + 自定义 Problem
- **图着色**: adapter.de + bias.graph_coloring + bias.graph_coloring_constraint + 自定义 Problem
- **因果发现**: adapter.de + bias.graph_sparsity + bias.graph_connectivity + bias.constraint + 自定义 Problem
- **GMM**: adapter.de → adapter.vns (chain) + bias.local_nelder_mead + 自定义 Problem
- **t-SNE**: adapter.gradient_descent + representation.linear_point + head.point + 自定义 Problem (KL divergence)
- **矩阵分解**: adapter.gradient_descent + head.point + 自定义 Problem (sparse MSE)

## 关键约定
1. **开工前先查 catalog**：`python -m mlblack catalog search <关键词>` — 必须首先执行，避免重复造轮子
2. 查看组件详情：`python -m mlblack catalog show <key>`
3. 检查 companion：很多 adapter 有配套 bias/plugin 建议
4. **Catalog 唯一事实源**：框架条目与 Case 条目都使用 `catalog/entries/<kind>.toml`；数据库/UI 只是其物化视图。

5. nsgablack 新项目用 `python -m nsgablack project new <project_name> --force`

6. mlblack 新项目用 `python -m mlblack project new <project_name> --force`

7. nsgablack 侧 Catalog 注册：编辑对应 `catalog/entries/<kind>.toml` 分片。
8. 不要删除脚手架生成的模板文件（config.py, assembly.py 等），只添加新文件

## 常用命令
```powershell
# nsgablack
python -m nsgablack catalog list --kind <kind> --profile framework-core
python -m nsgablack catalog search <query> --profile framework-core --show-import
python -m nsgablack catalog show <key> --profile framework-core
python -m nsgablack project new <project_name> --force
python -m nsgablack project doctor --path . --strict --format problem

# mlblack
cd C:\Users\hp\Desktop\mlblack
python -m mlblack catalog list --kind <kind>
python -m mlblack catalog search <query> --show-import
python -m mlblack catalog show <key>
python -m mlblack project new <project_name> --force
python -m compileall -q .
```

## README.md 标准格式（强制）

每个案例必须按此模板写 README.md：

```markdown
# name（中文标题）

一句话：这个 case 验证什么。

## 是否使用 mlblack / nsgablack

明确标注框架归属：纯 nsgablack / 纯 mlblack / 双框架 hybrid。

## 这个 case 验证什么

详细分解：候选编码 → 评估逻辑 → 搜索策略 → 能力信号。

## 搜索向量

| 变量 | 含义 | 范围 |
|---|---|---|
| ... | ... | ... |

## 目标和指标

| 目标 | 方向 | 含义 |
|---|---|---|
| ... | minimize/maximize | ... |

## 组件组合（必须）

| 层 | 组件 | 来源 |
|---|---|---|
| Problem | 自定义 XXXProblem | 自定义 |
| Representation | XXX | 框架 repr.xxx |
| Adapter | XXX | 框架 adapter.xxx |
| Bias | XXX（可选） | 框架 bias.xxx |

## 效果对比（必须）

| Method | 指标1 | 指标2 | Time | vs baseline |
|---|---|---|---|---|
| 传统实现 (sklearn/...) | ... | ... | ... | baseline |
| nsgablack/mlblack | ... | ... | ... | ...× |

效果对比表格必须包含真实跑出来的数字，和传统实现的对比。

## 结构

| 路径 | 作用 |
|---|---|
| ... | ... |

## 运行和验证

```powershell
python build_solver.py ...
```
```

## 已完成案例索引
| 案例 | 路径 | 框架 | 备注 |
|---|---|---|---|
| GMM EM vs DE | nsgablack/.../gmm_em_vs_de/ | 🟦 nsgablack | DE→VNS 策略链，vs sklearn EM |
| 因果发现 | nsgablack/.../causal_discovery/ | 🟦 nsgablack | bias.callable(Kahn) + bias.graph_sparsity |
| t-SNE | mlblack/.../tsne_umap/ | 🟩 mlblack | adapter.gradient_descent，KL 散度梯度 |
| 异常检测 | nsgablack/.../anomaly_detection/ | 🟦 nsgablack | adapter.de + bias.constraint |
| TSP/VRP | nsgablack/.../tsp_vrp/ | 🟦 nsgablack | bias.graph_tsp_constraint + repr.permutation |
| 矩阵分解 | mlblack/.../matrix_factorization/ | 🟩 mlblack | adapter.gradient_descent，GD vs SVD，15x |
| ARIMA 阶数搜索 | nsgablack/.../arima_order_search/ | 🟦 nsgablack | DE vs 网格搜索 (p,d,q) |
| Granger 因果 | mlblack/.../granger_causality/ | 🟩 mlblack | L1 稀疏 VAR(1) 恢复，因果边检测 |
| 交通拥堵预测 | mlblack/.../traffic_congestion/ | 🟩 mlblack | XGBoost + GD 符号回归，真实交通数据 |
| 线程池手递手 | nsgablack/.../pooled_backend_handoff/ | 🟦 nsgablack | L0 PoolScheduler + L4 CoptBackend，存储做线程交接 |
