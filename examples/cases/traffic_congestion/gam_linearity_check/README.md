# GAM Linearity Check（GAM 线性假设检验）

Diagnostic tool to validate whether linear model assumptions hold for the traffic CI prediction task.

## 是否使用 mlblack / nsgablack

纯 sklearn 诊断工具，不属于 mlblack/nnsgablack 训练管线。不注册 catalog 组件。

## 这个 case 验证什么

通过 B-spline 基展开的 GAM（广义加性模型）与线性回归对比，检测：

1. 各特征的线性系数 vs GAM 偏依赖曲线是否一致
2. 是否存在 GAM 显著优于线性回归的非线性模式
3. 哪些特征表现出显著非线性，需考虑非线性建模

## 搜索向量

不适用（无搜索）。本工具做 comparison，不做 optimization。

## 目标和指标

| 目标 | 方向 | 含义 |
|---|---|---|
| RMSE | minimize | 训练集拟合误差 |
| GAM span / Linear span | ratio | 非线性程度指标 |

## 组件组合

| 层 | 组件 | 来源 |
|---|---|---|
| Problem | 无（纯诊断） | 不适用 |
| Representation | 无 | 不适用 |
| Adapter | RidgeCV | sklearn |
| Model | B-spline + Linear | sklearn |

## 效果对比

| Method | Train RMSE | Delta | Nonlin Features | Notes |
|---|---|---|---|---|
| Linear Regression | * | baseline | 0 | 假设所有特征线性 |
| GAM (B-spline + Ridge) | * | *% | * | 自动检测非线性 |

*实际数值取决于数据集和超参设置。

## 结构

| 路径 | 作用 |
|---|---|
| build_trainer.py | 主诊断脚本 |
| START_HERE.md | 快速开始说明 |
| README.md | 本文档 |
| assembly/ catalog/ adapter/ ... | 保留的脚手架目录（本工具未使用） |

## 运行和验证

```powershell
python build_trainer.py
python build_trainer.py --n-knots 8 --degree 4
```
