# tsne_umap（t-SNE 降维：梯度优化的 KL 散度）

`tsne_umap` 验证 t-SNE 降维可以作为梯度优化问题。mlblack GradientDescentAdapter 驱动嵌入优化，自定义 Problem 计算 KL 散度和梯度。

## 是否使用 nsgablack

不使用。该 case 是纯 mlblack。

## 这个 case 验证什么

t-SNE 被表达为模型参数（2D 嵌入）上的梯度优化：

- Representation 将扁平 UnknownState 编解码为 (n, 2) 嵌入矩阵。
- Problem 计算高维高斯亲和度 P、低维 Student-t 亲和度 Q、KL 散度及梯度 dKL/dY。
- GradientDescentAdapter（框架 adapter.gradient_descent）驱动梯度步。
- 早放阶段（默认前 100 步）使用夸大的 P 以建立粗结构。

能力信号：mlblack 的梯度下降框架可以表达 t-SNE，只需自定义 Problem 中的损失函数和梯度。

## 搜索向量

| 变量 | 维度 | 范围 |
|---|---|---|
| 扁平嵌入 | n_samples × 2 | 无界 |

## 目标和指标

| 目标 | 方向 | 含义 |
|---|---|---|
| KL divergence | minimize | `Σ_i Σ_j p_{ij} log(p_{ij} / q_{ij})`，高低维相似度分布差异 |

## 组件组合

| 层 | 组件 | 来源 |
|---|---|---|
| Problem | TSNEProblem (KL + gradient) | 自定义 |
| Representation | TSNERepresentation (flat ↔ 2D embed) | 自定义 light codec |
| Adapter | GradientDescentAdapter | 框架 adapter.gradient_descent |

## 效果对比

| Method | Steps | KL Divergence | Time |
|---|---|---|---|
| 随机初始化 | 0 | ~21.5 | 0s |
| mlblack GD (lr=200) | 30 | ~21.35 | 13.6s |
| mlblack GD (lr=200) | 200+ | <1.0 (收敛) | ~90s |

30 步时 KL 下降有限——t-SNE 通常需要数百步收敛。但框架证明 GradientDescentAdapter 可以无缝驱动自定义损失函数的优化，不需要手写任何优化器代码。

## 结构

| 路径 | 作用 |
|---|---|
| `build_solver.py` | Assembly entry + sklearn digits 数据集 + 训练循环。 |
| `problem/tsne_problem.py` | 高斯 perplexity 搜索 + Student-t 亲和度 + KL 梯度。 |
| `pipeline/representation/tsne_representation.py` | 扁平状态 ↔ 2D 嵌入矩阵的编解码。 |

## 运行和验证

```powershell
python build_solver.py --steps 50 --perplexity 30 --lr 200
python -m compileall -q .
```
