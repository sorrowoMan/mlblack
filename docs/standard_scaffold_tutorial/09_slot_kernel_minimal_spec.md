# 09. Slot Kernel 最小规范（mlblack 版）

本章是 mlblack 的中文实战手册，重点是：

1. 明确和 nsgablack 的 pipeline 语义差异  
2. 保持共享 slot kernel 编排契约  
3. 给出可直接运行的多组示例

## 0. 先把差异说清楚

两边都用一个 `pipeline/main.py` + slot spec，但语义不同：

- `nsgablack`：搜索表示流（`init/mutate/repair/encode/decode`）
- `mlblack`：训练语义流（`transform/codec/head`）

所以：

- 编排统一（slot kernel）
- 组件语义不同（搜索 vs codec/head）

---

## 1. 从空项目开始（trainer case）

```powershell
python -m nsgablack project new demo_slot_kernel_ml
cd demo_slot_kernel_ml
python -m nsgablack project add-case my_trainer --type trainer --framework mlblack
```

---

## 2. 用 CLI 创建 pipeline 内部组件

```powershell
python -m nsgablack project add-component --case my_trainer --kind pipeline --slot main --name main
python -m nsgablack project add-component --case my_trainer --kind pipeline --slot transform --name zscore_transform
python -m nsgablack project add-component --case my_trainer --kind pipeline --slot codec --name linear_codec
python -m nsgablack project add-component --case my_trainer --kind pipeline --slot head --name point_head
```

---

## 3. `pipeline/main.py` 装配范式（可抄）

```python
from typing import Any, Mapping
from mlblack.pipeline import PipelineSpec, build_pipeline_kernel


def build_pipeline(*, resource_context: Mapping[str, Any] | None = None, component_overrides: Mapping[str, Any] | None = None):
    del resource_context
    overrides = dict(component_overrides or {})
    registry = dict(overrides.get("pipeline_operators", {}) or {})
    spec = PipelineSpec.from_value(
        overrides.get("pipeline_spec", {"key": "trainer_default", "slots": ()})
    )
    kernel = build_pipeline_kernel(spec, operator_registry=registry, transform_slot="transform")
    return kernel.data_pipeline
```

---

## 4. 三种模式实战示例

### 4.1 serial：标准单链训练流

```python
pipeline_spec = {
    "key": "train_serial_v1",
    "slots": (
        {"slot": "transform", "mode": "serial", "operators": ("zscore_transform", "feature_build")},
        {"slot": "codec", "mode": "serial", "operators": ("linear_codec",)},
        {"slot": "head", "mode": "serial", "method": "predict", "operators": ("point_head",)},
    ),
}
```

### 4.2 parallel：多特征分支融合

```python
pipeline_spec = {
    "key": "train_parallel_feature_v1",
    "slots": (
        {
            "slot": "transform",
            "mode": "parallel",
            "operators": ("trend_branch", "seasonal_branch"),
            "merge": "mean",
        },
        {"slot": "codec", "mode": "serial", "operators": ("linear_codec",)},
    ),
}
```

### 4.3 router：按任务类型切 head

```python
pipeline_spec = {
    "key": "train_router_head_v1",
    "slots": (
        {"slot": "transform", "mode": "serial", "operators": ("zscore_transform",)},
        {"slot": "codec", "mode": "serial", "operators": ("linear_codec",)},
        {
            "slot": "head",
            "mode": "router",
            "method": "predict",
            "selector_key": "task_kind",
            "routes": {
                "point": "point_head",
                "interval": "interval_head",
                "prob": "probability_head",
            },
            "default_operator": "point_head",
        },
    ),
}
```

---

## 5. operator registry 示例

```python
pipeline_operators = {
    "zscore_transform": ZScoreTransform(),
    "feature_build": FeatureSpaceBuild(),
    "trend_branch": TrendBranchTransform(),
    "seasonal_branch": SeasonalBranchTransform(),
    "linear_codec": LinearCodec(),
    "point_head": PointHead(),
    "interval_head": IntervalHead(),
    "probability_head": ProbabilityHead(),
}
```

---

## 6. 重点：`method` 字段为什么必须掌握

mlblack 的 head 往往不是 `decode()`，而是 `predict()` 或 `forward()`。  
因此要显式指定：

```python
{"slot": "head", "method": "predict", ...}
```

否则会出现“slot 存在但方法名不匹配”的隐性错误。

---

## 7. 可直接执行的验证

```powershell
python run_project.py --check --build-check
python -m mlblack project doctor --path . --strict
```

建议额外做三类单测：

1. serial transform 链
2. parallel merge 行为
3. head router 路由行为

---

## 8. 常见错误

1. `head` 未设置 `method` 导致调用不到 `predict`
2. parallel 分支输出结构不一致导致 merge 失败
3. router `selector_key` 和 context 字段不匹配
4. 把编排写进 trainer 私有 runner（违反统一 substrate 口径）

---

## 9. 本章结论

mlblack 的 pipeline 确实和 nsgablack 不同（它是 codec/head 语义流），  
但两者应该共享同一个 slot kernel 编排契约，这样才能做到：

- 框架统一
- 语义清晰
- 可嵌套编排
- 可审计运行
