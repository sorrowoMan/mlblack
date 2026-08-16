# 13. Pipeline 编排与组件设计（mlblack 详细版）

本章聚焦 mlblack 的 pipeline 设计实践：它是 codec/head 语义流，不是搜索表示流。

## 1. 结构标准

Case 级单入口：

```text
pipeline/main.py
```

内部算子：

```text
pipeline/operators/transform/*.py
pipeline/operators/codec/*.py
pipeline/operators/head/*.py
pipeline/operators/custom/*.py
```

## 2. 语义层分工

- `transform`：DataView/特征/目标变换
- `codec`：模型参数状态编码解码
- `head`：输出语义（point/interval/probability/other）

不要把这些直接混进 trainer 主循环。

## 3. 编排模式实践

### serial

标准训练链：

```text
transform -> codec -> head
```

### parallel

特征并行分支：

```text
branch_a(transform) + branch_b(transform) -> merge -> codec -> head
```

### router

按任务或场景切换：

```text
task_kind=point -> point_head
task_kind=interval -> interval_head
task_kind=prob -> probability_head
```

## 4. 组件设计建议

每个 operator 文件最好满足：

- 单一职责
- 稳定输入输出
- 明确 method（例如 `predict` / `forward`）
- 可单测

## 5. 与 nsgablack 的统一点

统一点：

- 同一 slot kernel contract
- 同一 project/case/scaffold/L0 substrate

差异点：

- nsgablack 强调搜索表示
- mlblack 强调 codec/head 训练语义

## 6. 可运行检查清单

- [ ] `pipeline/main.py` 是唯一主入口
- [ ] slot spec 和 registry 一致
- [ ] head slot 的 `method` 明确
- [ ] `--check --build-check` 通过
- [ ] doctor 严格模式通过
