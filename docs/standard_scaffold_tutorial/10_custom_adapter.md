# 10. 自定义优化 Adapter（统一栈实战）

MLBlack 不定义私有优化 Adapter。自定义策略实现 `nsgablack.AlgorithmAdapter`
协议，可以放在具体 Case 或独立策略包中；MLBlack 只提供数据、模型、Problem 与
Provider 语义。

## 1. Adapter 边界

Adapter 可以负责：

- 参数更新策略
- 候选状态更新策略
- 与 feedback 对齐的优化步

Adapter 不应该负责：

- Project stage 编排
- 全局资源调度
- 训练数据读取与清洗主流程

---

## 2. 创建文件

```powershell
python -m nsgablack project add-component --case my_trainer --kind adapter --name my_trainer_adapter
```

---

## 3. 最小骨架（可运行思路）

```python
from nsgablack.adapters import AlgorithmAdapter


class MyOptimizationAdapter(AlgorithmAdapter):
    def propose(self, solver, context):
        # 返回当前需要评估/更新的状态候选
        ...

    def update(self, solver, candidates, feedback, context):
        # 使用反馈更新内部状态
        ...
```

如果你是梯度类训练，可把 propose 理解为“当前参数态”，update 理解为“一次优化步”。

---

## 4. 挂载示例

```python
trainer = ...
trainer.set_adapter(MyOptimizationAdapter(...))
```

---

## 5. 建议做的三层验证

1. `--check --build-check`：装配面正常  
2. `project doctor`：结构面正常  
3. 小步训练 smoke：更新面正常

```powershell
python run_project.py --check --build-check
python -m mlblack project doctor --path . --strict
```

---

## 6. 常见坑

1. Adapter 直接读取原始训练数据  
   修复：数据语义由 pipeline/problem 暴露给 trainer，不在 adapter 私读。

2. Adapter 内部自建资源池  
   修复：只消费 `resource_context`。

3. 把多 Case 编排逻辑塞进 Adapter  
   修复：上移到 Project substrate。
