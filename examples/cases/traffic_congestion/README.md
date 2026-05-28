# traffic_congestion（交通拥堵指数预测）

基于动态符号回归的高速公路拥堵临界阈值发现与可解释预测。1717 条 G4 广深高速宝安段真实日度交通数据。

## 论文叙事

```
ARIMAX（线性归因，不够好）
  → GAM 非线性检验：滞后特征存在 1.8x 非线性放大
  → SHAP 范式检验：线性 vs XGBoost 特征重要性相关性仅 0.48
  → Granger 因果检验：CI→Wind 反向因果显著
  → 结论：线性方法不足，必须符号回归
    → 机理重构：符号回归自主涌现占有率 20% 临界阈值
    → 前瞻预测：符号模型区间覆盖 PICP=95% vs XGBoost=22%
```

## 案例

### 第一幕：归因与诊断 — 为什么线性不够

| # | 案例 | 类型 | 关键结果 |
|---|---|---|---|
| 1 | [arimax_factor_attribution](./arimax_factor_attribution/) | ARIMAX 因子归因 | Weather 组 AIC 影响 HIGH，但系数贡献仅 0.4%——线性模型无法表达非线性效应 |
| 2 | [gam_linearity_check](./gam_linearity_check/) | GAM 非线性诊断 | GAM B-spline 检测到 8 个非线性特征，ci_lag3 的 GAM 响应幅度是线性回归的 1.8 倍 |
| 3 | [shap_contribution_check](./shap_contribution_check/) | SHAP 跨范式检验 | 线性 vs XGBoost 特征重要性 Spearman ρ=0.48（弱相关），SHAP vs Permutation ρ=0.97 |
| 4 | [granger_causality_check](./granger_causality_check/) | Granger 因果关系 | CI→Wind (lag=1, F=19.2, p≈0)，反向因果显著 |

**诊断结论**：ARIMAX 线性系数无法可靠归因。滞后变量存在非线性放大效应，特征重要性在不同模型范式下不一致。必须引入能自主发现非线性结构的方法。

### 第二幕：符号回归 — 从数据自主涌现物理规律

| # | 案例 | 类型 | 关键结果 |
|---|---|---|---|
| 5 | [symbolic_regression](./symbolic_regression/) | 机理重构（同日） | 符号模型 RMSE=0.07 vs XGBoost=1.04，涌现占有率 20% 临界阈值 |
| 6 | [xgboost_baseline](./xgboost_baseline/) | XGBoost 基线 | Train RMSE=5.01, Valid R²=0.85，但区间覆盖率仅 22% |

**发现**：在没有任何交通工程先验知识注入的前提下，符号回归自主涌现出以分段线性整流函数为核心的算子结构。占有率整流项阈值经多次独立演化后自主收敛至 0.20 附近，与交通流理论"临界密度"吻合。

## 数据

`data/` 源自 `C:\Users\hp\Desktop\work\final_pipeline_package_20260402`：

| 文件 | 用途 |
|---|---|
| `ci_interval_opt_table_no_flow_speed_occ.csv` | 机理重构：同日特征，无滞后 |
| `ci_interval_opt_table_no_flow_speed_occ_lag.csv` | 前瞻预测：仅滞后特征，无流量泄露 |
| `ci_formula_meta.txt` | CI 公式元数据 |

## 拥堵指数

```
CI = 100 * (0.7 * clip(1 - v/v_ff, 0, 1) + 0.3 * clip((occ - q10)/(q90 - q10), 0, 1))
```

## 运行

```powershell
# 诊断
cd arimax_factor_attribution && python build_trainer.py
cd ../gam_linearity_check && python build_trainer.py
cd ../shap_contribution_check && python build_trainer.py
cd ../granger_causality_check && python build_trainer.py

# 机理重构
cd ../symbolic_regression && python build_trainer.py --steps 200

# XGBoost 基线
cd ../xgboost_baseline && python build_trainer.py
```
