# START_HERE

## Granger 因果检验 (Traffic CI)

诊断交通拥堵指数 CI 与外部因素（天气、AQI、节假日、风力）之间是否存在统计显著的 Granger 因果关系。

## 1) 运行诊断

```powershell
python examples\cases\traffic_congestion\run_project.py --check --build-check
python examples\cases\traffic_congestion\cases\granger_causality_check\run_solver.py --maxlag 7
```

参数：
- `--maxlag`：最大滞后阶数，默认 7
- `--alpha`：显著性阈值，默认 0.05

## 2) 依赖

- pandas, numpy, scikit-learn（必需）
- statsmodels（推荐，用于 Granger 检验）
- scipy（降级备选，仅交叉相关分析）

安装缺失依赖：

```powershell
pip install statsmodels scipy
```

## 3) 输出解读

- **Pairwise 表**：每个外部因素 → CI 的 Granger F-test
  - p < 0.05 表示该因素显著有助于预测 CI
- **Reverse 表**：CI → 因素的逆向检验
  - 用于排除伪因果关系
- **VAR 模型**：多变量联合检验，输出最优滞后阶数和 AIC

## 4) 关键文件

| 路径 | 作用 |
|---|---|
| build_solver.py | Granger 因果检验主脚本 |
| ../data/ci_interval_opt_table_no_flow_speed_occ_lag.csv | 输入数据（无流量/速度/占有率泄漏） |
