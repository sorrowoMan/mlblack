# 线性模型的统一优化切片

这个切片用最简单的线性回归说明 MLBlack 与 NSGABlack 的统一视图。模型为：

```text
y_hat = intercept + X @ weights
```

当前 Solver 允许算法决定的变量是截距与权重，因此正式 Candidate 是：

```text
[intercept, weight_0, weight_1, ...]
```

运行链路如下：

```text
UnknownState Candidate
  -> LinearPointCodec.decode
  -> LinearPointModel
  -> SupervisedRegressionProblem.evaluate
  -> Feedback(objectives, constraints, gradients)
  -> nsgablack gradient.sgd / gradient.adam / gradient.adamw Adapter
  -> 下一组 UnknownState Candidate
```

## 每一层的职责

- `LinearPointCodec` 固定候选坐标与模型参数之间的双向映射，并公开版本化参数布局。
- `LinearPointModel` 只表达线性模型语义、预测行为和稳定的模型信封，不选择优化算法。
- `SupervisedRegressionProblem` 持有数据、目标、正则项和解析梯度语义。
- NSGABlack 的梯度 Adapter 只消费 Candidate 坐标系中的梯度，不读取训练数据，也不依赖 ML backend。
- `Trainer` 负责生命周期、评价入口、最佳 Candidate 和最终模型选择。
- `ArtifactBuilder` 或 Case Runtime Artifact Provider 将最佳模型转成可复用结果；协议不会伪造不存在的外部引用。

公共装配入口：

```python
from mlblack.presets import build_linear_point_trainer

trainer = build_linear_point_trainer(
    data,
    method="gradient.adam",
    learning_rate=0.05,
)
result = trainer.fit(max_steps=100)
```

这里的 `method` 是稳定方法标识。用户不需要知道 Adapter 来自哪个仓库；Catalog/装配层会把它解析成 NSGABlack 的优化策略，而模型、数据、评价和 Artifact 语义仍属于 MLBlack。

## 向神经网络推广

神经网络不会改变这条主链，只会把简单布局：

```text
intercept -> offset 0
weights   -> offset 1...
```

扩展为具名张量布局：

```text
layer1.weight -> offset ... + shape ...
layer1.bias   -> offset ... + shape ...
layer2.weight -> offset ... + shape ...
```

Codec 仍负责坐标映射；Evaluation Provider 负责在 Torch、JAX 等后端上构造运行模型并计算前向/反向；Adapter 仍只选择更新机制。
