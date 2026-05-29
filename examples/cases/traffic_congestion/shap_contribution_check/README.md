# shap_contribution_check（SHAP 特征贡献一致性检查）

一句话：验证交通拥堵指数 CI 预测中，XGBoost 非线性模型与 Linear 线性模型的特征重要性
排序是否一致，确保归因分析的模型范式稳健性。

## 是否使用 mlblack / nsgablack

纯分析脚本，不依赖 mlblack/nsgablack 框架。使用 sklearn、xgboost、shap 直接比较。

## 这个 case 验证什么

特征重要性排序的跨范式稳定性：

- **Linear**: 使用 `|coef|` 作为特征重要性，假设特征独立且关系线性。
- **XGBoost (gain)**: 基于树分裂的增益加权重要性，捕捉非线性交互。
- **SHAP (TreeSHAP)**: 博弈论归因值，对 XGBoost 近似精确。
- **Permutation Importance**: 通过打乱特征列测量模型性能下降，与模型无关。

若四种方法的 top-k 排名高度一致，说明线性归因结论在非线性场景下也可靠。
若差异大，说明特征交互/非线性效应显著，线性简化可能误导。

## 搜索向量

不适用（纯分析，非搜索优化案例）。

## 目标和指标

| 指标 | 含义 |
|---|---|
| Spearman rank correlation | 方法间重要性排序的一致性 |
| Top-k agreement | top-k 特征被多少方法同时选中 |
| Average agreement score | 平均一致性得分（满分 = 方法数） |

## 组件组合

| 层 | 组件 | 来源 |
|---|---|---|
| N/A | sklearn LinearRegression | sklearn |
| N/A | xgboost XGBRegressor | xgboost |
| N/A | shap TreeExplainer | shap |
| N/A | sklearn permutation_importance | sklearn |

## 效果对比

| 比较对 | Spearman Rank Correlation | 解读 |
|---|---|---|
| Linear vs XGBoost Gain | 0.479 | 弱相关性，非线性效应明显 |
| Linear vs SHAP | 0.575 | 中等相关性 |
| Linear vs Permutation | 0.579 | 中等相关性 |
| XGBoost vs SHAP | 0.692 | 较强，gain 和 SHAP 接近 |
| XGBoost vs Permutation | 0.687 | 较强 |
| SHAP vs Permutation | 0.972 | 高度一致，两者等价 |

| 指标 | 值 |
|---|---|
| Average top-k agreement | 0.5 / 4 methods |
| Conclusion | DIFFER — 线性特征重要性不能替代非线性归因 |

## 结构

| 路径 | 作用 |
|---|---|
| build_solver.py | 主分析入口：训练、SHAP、排列重要性、排名对比 |
| START_HERE.md | 快速起步指南 |
| README.md | 本文件 |
| assembly/ | mlblack 脚手架（本 case 不使用） |
| problem/ adapter/ bias/ plugins/ | mlblack 脚手架目录 |
| pipeline/representation/ pipeline/ catalog/ | mlblack 脚手架目录 |

## 运行和验证

```powershell
python build_solver.py
python build_solver.py --n-estimators 300 --top-k 15
python -m compileall -q .
```
