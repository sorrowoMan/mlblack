# AGENTS.md (nowcasting_work_ci)

## 1) 目标与默认策略

本目录的 AI 执行默认目标是**区间预测优先**，不是点预测优先。  
默认优化口径：

1. 首要：压 `PINAW`
2. 次要：压 `Interval Score`
3. 约束：`PICP` 贴近目标覆盖（由 `alpha` 决定）

若用户未明确指定，默认目标覆盖采用：

- `alpha=0.20`（目标 `PICP≈0.80`）

---

## 2) 默认运行配置（AI 必须优先使用）

当用户说“跑一版 / go / 拉预算 / 看效果”且未给细节时，默认启用：

- `--strict4-branch-mode`（四分支时序/节假日状态建模）
- `--batched-eval 1`
- `--reinvest-search 1`
- `--dynamic-pool-enabled 1`
- `--graph-cache-enabled 1 --graph-cache-backend sqlite`
- `--drop-same-day-flow-speed-occ 1`（前瞻评估口径）
- `--lag-feature-enabled 1 --lag-orders 1,2,3 --lag-sources ci,total_flow,avg_speed,avg_occ`
- `--lag-cross-enabled 1`
- `--temporal-pack-enabled 1`（rolling/momentum/cross/ratio 默认开）
- `--regime-pack-enabled 1`（volatility/shock/ci_regime 默认开）

---

## 3) 预算档位约定

默认预算分三档，除非用户指定：

1. 小预算：`pop=32, gen=25`
2. 中预算：`pop=96, gen=80`
3. 高预算：`pop=256, gen=150`

用户说“拉满”默认使用高预算档。  
用户说“先看趋势”默认使用中预算档。

---

## 4) 选解规则约定

保持当前项目规则：

- 外层三目标：`coverage_error + PINAW + interval_score`
- 终选：先看 `coverage_error` 是否过阈值，再比较 `PINAW/IS`

若用户强调“PINAW 最重要”，可放宽 coverage 阈值但必须在汇报中说明 `PICP` 变化。

---

## 5) AI 汇报格式（每次跑完必须给）

至少给出：

1. 运行配置（`alpha/calib_ratio/pop/gen/seed`）
2. `PICP / PINAW / IS / RMSE`
3. 与上一版对比（升/降）
4. `summary.json` 绝对路径

禁止只报 RMSE 不报区间指标。

---

## 6) A/B 对照约定

用户要求“看某机制是否有效”时，AI 必须：

1. 固定 seed 与预算
2. 只改一个机制开关
3. 输出并排对照结果

不允许多变量同时变化后给结论。

---

## 7) 目录边界

- `nsgablack` 主干不改
- 优先在 `mlblack_side` 做特征与评估桥接增强
- `run.py` 作为统一外层入口


---

## 8) Example Scaffold Rule

If you add a new example/demo/benchmark runner in this subtree, it must use the standard scaffold / formal assembly path.

- Reuse the official workflow / CLI / assembly entrypoints first.
- Do not build a private one-off runtime path only for the example.
- Examples should mirror the real product assembly surface, not bypass it.

