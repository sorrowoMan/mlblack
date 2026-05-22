# 06. 验收、Catalog 与 Artifact

这一章讲怎么判断一个组件或 case 是否符合架构。重点不是“能不能跑”，而是能不能审计、复现、查询和长期演进。

## 1. 最小验收命令

```powershell
python -m pytest -q tests
python -c "from mlblack.project import run_project_doctor, format_doctor_report; print(format_doctor_report(run_project_doctor('.', strict=True)))"
```

常见专项：

```powershell
python -m pytest -q tests\test_model_integration.py
python -m pytest -q tests\test_neural_graph_codec.py
python -m pytest -q tests\test_symbolic_nsgablack_integration.py
```

## 2. Doctor 检查什么

Doctor 关注：

```text
context contract key 是否注册
component contract 是否可解析
是否重新出现 workflow/runtime/resource_request 等禁用口径
core/resource 是否仍为 passive ResourceContext
single-trainer assembly 是否没有外层编排字段
catalog 是否能解析组件
```

Doctor 不是性能测试，也不能替代 case smoke。

## 3. Context contract

组件应声明：

```python
context_requires = (...)
context_optional = (...)
context_provides = (...)
context_mutates = (...)
context_cache = (...)
requires_metrics = (...)
metrics_fallback = "strict"
context_notes = "..."
```

规则：

```text
requires:
  必须存在，否则组件不能正确运行。

optional:
  有则使用，没有也能运行。

provides:
  组件完成后对外提供什么。

mutates:
  会修改的状态边界。

cache:
  可复用但非语义必需的缓存。
```

## 4. Catalog

Catalog 的目标是组件发现，不是 workflow 执行。

应能查到：

```text
trainer
adapter
representation
problem
pipeline
backend capability
dashboard
artifact viewer
model composition component
```

推荐新增组件时加 catalog entry：

```text
key: model.integrated_prediction
kind: model
tags: composition, integration, residual, stacking
summary: Combines named fitted model predictions without owning training orchestration.
```

## 5. Artifact 边界

Artifact 应回答：

```text
这个模型/表达式是什么？
怎么得到的？
用了哪些组件？
数据和资源上下文是什么？
核心指标是什么？
能否恢复/重放？
```

不要把 artifact 写成 adapter side effect。应由 trainer result、problem build artifact hook、ArtifactBuilder 或 reporting layer 统一生成。

## 6. Artifact 类型

| artifact | 必备信息 |
| --- | --- |
| model | model type, family, head, representation |
| integrated_model | component refs, integration spec, I/O contract |
| neural_graph | graph spec, parameter layout, audit maps |
| tree/xgboost | estimator params, fitted state summary |
| symbolic_model | expression, canonical payload, recovery report |
| trainer_state | adapter/trainer state signature |
| run_report | metrics, resources, components |

## 7. Snapshot vs Context

Context 只放轻量字段：

```text
run_name
step
resource.*
best_score
artifact_ref
snapshot_key
small metrics
```

大对象进 snapshot/artifact：

```text
population
full history
model object
large arrays
trace
attention maps
symbolic graph cache
```

## 8. Dashboard

Dashboard 是查看面，不是执行面。

| dashboard | 用途 |
| --- | --- |
| catalog dashboard | 看组件和 contract |
| backend matrix | 看 backend capability |
| artifact viewer | 看 typed artifact |
| experiment dashboard | 查 runs/metrics |
| benchmark dashboard | 多 run 聚合 |

## 9. Case summary 验收

正式 case summary 至少包含：

```text
suite_id
protocol
config
stage reports
effective resource context
component reports
artifact refs
best record
failure records
runtime summary
```

组合模型 case 还应包含：

```text
component model names
component artifact refs
PredictionIOContract
PredictionIntegrationSpec
integration metrics
```

符号 case 还应包含：

```text
canonical key
truth recovery
family recovery
phase equivalence
basis artifact lineage
```

## 10. 禁止回归检查

每次大改后，人工检查是否出现：

```text
mlblack.workflow
mlblack.runtime
ResourceRequest
LocalResourceAllocator
StageRunner
HybridTrainer
MultiModalWorkflow
ResidualWorkflow
adapter directly reads X/y
problem directly chooses backend by get_backend("torch")
codec silently switches backend
```

出现这些通常说明边界开始倒退。

## 11. 新组件 PR 检查表

```text
[ ] 归属层是否明确
[ ] 是否声明 context contract
[ ] 是否避免 adapter 直接读数据
[ ] 是否避免大对象写 context
[ ] 是否有 describe()
[ ] 是否能进 catalog 或文档索引
[ ] 是否有至少一个 smoke/unit test
[ ] 是否能被 ArtifactBuilder 或 reporting surface 描述
[ ] 是否没有新增 mlblack-owned workflow/runtime/L0
```

## 12. 推荐验证矩阵

| 改动 | 最小验证 |
| --- | --- |
| pipeline/data | `tests/test_pipeline_datasets.py` + doctor |
| model composition | `tests/test_model_integration.py` |
| neural graph | `tests/test_neural_graph_codec.py` |
| backend | `tests/test_compute_backend_session.py` + backend matrix |
| symbolic | `tests/test_symbolic_nsgablack_integration.py` |
| docs only | rg links + optional doctor |
| cross-framework case | case `--check` |
