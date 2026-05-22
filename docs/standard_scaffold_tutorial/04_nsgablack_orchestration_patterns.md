# 04. nsgablack 外层编排模式

这一章说明：当任务超过单 trainer 时，应该怎样把 `mlblack` 组件交给 `nsgablack` 编排。这里不实现 nsgablack API 细节，而是固定跨框架设计口径。

## 1. 边界

```text
mlblack:
  inner trainer / model semantic / data transform / artifact surface

nsgablack:
  stage / group / serial / parallel / resource lease / outer solver / outer adapter
```

如果一个机制回答“多个训练任务怎么排”，它属于 `nsgablack`。

如果一个机制回答“单个训练任务怎么建模和评估”，它属于 `mlblack`。

## 2. 标准跨框架流

```text
nsgablack outer adapter.propose
  -> outer representation.decode
  -> mlblack-facing task spec
  -> inject ResourceContext
  -> mlblack build_trainer / problem / proxy
  -> inner fit/evaluate
  -> metrics/artifact/report projection
  -> nsgablack outer adapter.update
```

## 3. ResourceContext

ResourceContext 是外层授权结果。

```python
resource_context = {
    "device": "cpu",
    "threads": 1,
    "namespace": "suite.demo/stage1/candidate_0007",
    "budget": {"inner_steps": 20},
}

trainer.set_resource_context(resource_context)
```

`mlblack` 不做：

```text
LocalResourceAllocator
GPULeaseStore
ThreadPoolRuntime
ParallelBackendSelector
```

## 4. Stage 模式

### 4.1 串行阶段

```text
stage 1 trains base model
stage 2 trains residual model
stage 3 composes final model
```

落层：

| 内容 | 归属 |
| --- | --- |
| stage order | nsgablack |
| base trainer spec | mlblack |
| residual target transform | mlblack pipeline component |
| final model composition | mlblack model semantic |
| stage report | nsgablack summary + mlblack artifacts |

### 4.2 并行分支

```text
stage group:
  train tabular model
  train image model
  train text model

next stage:
  compose models with I/O contract
```

落层：

```text
nsgablack group/parallel:
  runs branch tasks

mlblack:
  each branch trainer + final IntegratedPredictionModel
```

### 4.3 Outer structure search

```text
nsgablack candidate:
  chooses model family, feature subset, graph spec, symbolic structure, fusion weights

mlblack inner:
  trains/evaluates the chosen structure
```

例：

```text
outer candidate = [model_family_id, hidden_dim_id, fusion_weight, feature_mask]
outer problem decodes candidate
inner mlblack trainers run
outer objectives = [valid_loss, complexity, runtime]
```

## 5. Group 模式

Group 不等于融合模型。Group 是运行策略；融合模型是最终预测语义。

```text
nsgablack group:
  多个候选/任务同时评估。

IntegratedPredictionModel:
  多个已训练模型在 predict 时合并输出。
```

这两个概念不能混。

## 6. Serial adapter 模式

如果要先用粗搜索，再精修：

```text
phase 1:
  random / evolutionary search over model structure

phase 2:
  gradient/backprop fine-tuning

phase 3:
  local repair / robustness check
```

归属：

```text
phase scheduling:
  nsgablack serial strategy / stage controller

inner training mechanics:
  mlblack adapter/problem/representation
```

## 7. 多 problem 模式

`nsgablack` 可以让 outer candidate 同时面对多个 objective/problem：

```text
accuracy problem
latency problem
stability problem
fairness/constraint problem
```

`mlblack` 提供每个 inner evaluation 的 metrics：

```text
valid.mse
complexity.nonzero
artifact parameter count
backend runtime summary
resource context
```

Outer problem 再投影成多目标向量。

## 8. 失败策略

跨框架失败必须可审计。

| 失败 | 推荐处理 |
| --- | --- |
| inner trainer 数值发散 | 返回 penalty objective + violation + error summary |
| backend capability missing | strict 抛错或 outer failure record |
| invalid symbolic expression | guard reject or protected evaluation penalty |
| artifact write failed | strict 抛错，soft 记录缺失 artifact |
| resource context 不满足 | 不私自换设备，返回 failure record |
| shape/I/O contract mismatch | fail-fast |

不要吞异常并返回正常指标。

## 9. Payload 设计

Outer -> inner：

```json
{
  "candidate_id": "stage1:0007",
  "stage": "base_model_search",
  "decoded_task": {
    "preset": "orthogonal_linear_point",
    "params": {"learning_rate": 0.05}
  },
  "resource_context": {
    "device": "cpu",
    "threads": 1,
    "namespace": "demo/stage1/0007"
  },
  "lineage": {
    "suite_id": "demo",
    "outer_generation": 3
  }
}
```

Inner -> outer：

```json
{
  "objectives": [0.02, 3.0],
  "violations": [0.0],
  "metrics": {
    "valid.mse": 0.02,
    "complexity.nonzero": 3
  },
  "artifact_ref": "artifact:model:...",
  "report_ref": "snapshot:report:..."
}
```

大对象走 snapshot/artifact ref，不塞进 payload。

## 10. 复杂组合怎么落 nsgablack

### 10.1 残差

```text
stage1: train main
stage2: build residual target using main artifact
stage3: train residual
stage4: compose integrated model
```

### 10.2 多模态

```text
group stage:
  train text branch
  train image branch
  train tabular branch
fusion stage:
  build IntegratedPredictionModel with I/O contract
```

### 10.3 Stacking

```text
stage1 group:
  train base learners
stage2 data transform:
  append base predictions
stage3:
  train meta learner
```

### 10.4 Symbolic nested

```text
stage1 outer:
  search basis structure
inner:
  fit symbolic parameters
stage2 outer:
  search task expression over basis artifact
inner:
  fit task parameters
```

## 11. 标准 case 文件组织

```text
examples/cases/<case>/
  build_solver.py
  run_solver.py
  config/case_config.py
  problem/data.py
  problem/factories.py
  pipeline/representation.py
  reporting/report_writer.py
```

`build_solver.py` 可以 import `nsgablack`，但 `mlblack` 主干不应依赖 `nsgablack`。

## 12. 检查点

一个跨框架 case 合格时，summary 必须包含：

```text
suite_id
protocol
stage configs
outer solver configs
effective ResourceContext
component reports
artifact refs
failure records
best record
```

如果 summary 只能看到最终 score，看不到每阶段组件和资源，这个 case 不合格。
