# 07. Benchmark、Dashboard 与资源审计

这一章讲工程化运行面：如何跑 benchmark、如何看 dashboard、如何审计资源上下文，以及如何区分 benchmark runner 和正式 case。

## 1. Benchmark 第一原则

Benchmark 不应复制 case 装配逻辑。

```text
正确:
  benchmark runner 调用正式 case surface，多次运行并聚合结果。

错误:
  benchmark runner 重新实现 stage1/stage2 problem、artifact schema、resource handling。
```

## 2. Benchmark 输出结构

推荐：

```text
examples/benchmarks/runs/<benchmark_id>/
  benchmark_summary.json
  benchmark_dashboard.html
  runs/
    <suite_id_0>/...
    <suite_id_1>/...
```

`benchmark_summary.json` 至少包含：

```text
benchmark_id
case
protocol
repeats
seeds
config projection
resource context
per-run scores
aggregate statistics
artifact refs
failure records
```

## 3. 指标聚合

推荐记录：

| 指标 | 用途 |
| --- | --- |
| mean / std | 稳定性 |
| min / max | 极端情况 |
| median | 抗异常 |
| success rate | 可用性 |
| failure kind count | 失败归因 |
| wall time | 性能 |
| artifact write time | 工程开销 |

## 4. 资源审计

`ResourceAuditPlugin` 只记录外部注入的资源上下文。

```python
spec = {
    "preset": "orthogonal_linear_point",
    "resource_context": {
        "device": "cpu",
        "threads": 1,
        "namespace": "benchmark.run.0",
    },
    "plugins": ["resource_audit"],
}
```

Report 中应该看到：

```text
resources.device
resources.threads
resources.namespace
compute_backend.resolved_name
```

## 5. L0 边界

| 能力 | 归属 |
| --- | --- |
| outer solver fanout | nsgablack |
| worker pool | nsgablack |
| GPU/CPU lease | nsgablack |
| backend/thread/process scheduling | nsgablack |
| inner compute backend | mlblack trainer/backend |
| resource audit | mlblack capability |

`mlblack` 可以遵守 `ResourceContext`，不能自己授权资源。

## 6. Dashboard 类型

| dashboard | 来源 | 展示 |
| --- | --- | --- |
| catalog | `mlblack.catalog` | 组件、contract、tags |
| backend matrix | `mlblack.catalog.backend_dashboard` | backend capability |
| artifact viewer | `mlblack.catalog.artifacts` | artifact schema |
| experiment | `mlblack.catalog.experiment` | run records |
| benchmark | benchmark runner/reporting | 多 run 聚合 |

## 7. Artifact viewer 最低要求

普通 model artifact：

```text
model_type
family/head
representation summary
problem summary
adapter summary
metrics
resources
```

integrated model artifact：

```text
component names
component model types
component artifact refs
PredictionIntegrationSpec
PredictionIOContract
weights
final metrics
```

symbolic artifact：

```text
final expression
canonical expression
truth recovery
family recovery
phase equivalence
lineage
```

neural graph artifact：

```text
graph spec
parameter layout
audit maps / summaries
backend
optimizer state summary
```

## 8. Smoke 验证矩阵

基础：

```powershell
python -m pytest -q tests
python -c "from mlblack.project import run_project_doctor, format_doctor_report; print(format_doctor_report(run_project_doctor('.', strict=True)))"
```

单 trainer：

```powershell
python examples\orthogonal_point_demo.py
```

组合模型：

```powershell
python -m pytest -q tests\test_model_integration.py
```

神经图：

```powershell
python -m pytest -q tests\test_neural_graph_codec.py
```

符号 nested：

```powershell
python examples\cases\symbolic_orthogonal_nested\run_solver.py --check
python -m pytest -q tests\test_symbolic_nsgablack_integration.py
```

## 9. Benchmark 分类

| 类型 | 目的 | 规模 |
| --- | --- | --- |
| smoke benchmark | 能跑通 | 1-2 runs，小 steps |
| regression benchmark | 防退化 | 固定 seeds，中等 steps |
| stress benchmark | 找瓶颈 | 大 population/steps |
| backend benchmark | 比较后端 | numpy/jax/tf/torch matrix |
| composition benchmark | 比较组合模式 | baseline/residual/stacking/fusion |
| symbolic recovery benchmark | 看 truth/family/phase recovery | synthetic truth set |

## 10. 组合模型 benchmark 示例

```text
case: residual_vs_baseline
runs:
  baseline linear
  baseline orthogonal
  main + residual
  main + residual + residual2
  stacking base_pred feature
metrics:
  train/valid mse
  complexity
  component count
  total fit time
  artifact size
```

关键：benchmark runner 只组合正式 case surface，不重新写训练逻辑。

## 11. Backend benchmark 示例

```text
same MLP spec:
  numpy inference/loss
  jax functional grad
  tensorflow GradientTape
  torch backprop

record:
  capability support
  fit status
  loss after N steps
  wall time
  failure reason
```

不要为了比较而让 backend 假装支持不存在的 capability。

## 12. 报告字段建议

正式 benchmark report：

```json
{
  "benchmark_id": "...",
  "protocol": "mlblack.benchmark.v1",
  "case": "...",
  "runs": [],
  "aggregate": {
    "valid.mse.mean": 0.0,
    "valid.mse.std": 0.0,
    "success_rate": 1.0
  },
  "resources": {},
  "artifacts": {},
  "failures": []
}
```

## 13. 发布前建议

```text
[ ] tests 全量通过
[ ] doctor ok
[ ] catalog/dashboard 可导出
[ ] artifact viewer 能展示新增 artifact
[ ] benchmark small 至少一轮成功
[ ] failure records 可读
[ ] 资源上下文可审计
```
