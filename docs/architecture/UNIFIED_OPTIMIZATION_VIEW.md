# MLBlack 统一优化视图

MLBlack 不再以“每个模型族拥有一个训练 Adapter”为规范视图。公共装配先解析稳定
方法标识，再把优化策略、ML 语义和计算执行分别交给所属层：

```text
TrainingConfig / preset / Case override
  -> stable method ID
  -> nsgablack Adapter
       只拥有搜索或更新机制
  -> mlblack Representation + Problem
       拥有结构、Codec、loss、metric 与输出语义
  -> optional mlblack Evaluation Provider
       拥有 autograd、第三方 fit、设备 kernel 与活计算状态
  -> BlackBase Project L0 / EvaluationGateway / StateRef
       拥有资源授权、绑定、版本栅栏和跨边界协议
```

## 当前稳定方法

- `gradient.sgd`、`gradient.adam`、`gradient.adamw`：由
  `GradientOptimizerAdapter` 执行。梯度可以由解析 Problem 直接返回，也可以由
  Torch Provider 通过 BlackBase Gateway 返回。
- `search.random_gaussian`：由 `GaussianSearchAdapter` 执行。它只扰动数值候选，
  因而可以搜索区间头参数、分类模型参数、时序配置、树/Boosting estimator spec
  或其他 Codec 编码。

用户仍可使用 ML 词汇：

```python
trainer = build_trainer(
    {"preset": "mlp_regression", "params": {"optimizer": "adam"}},
    data=data,
)
```

也可显式装配但不暴露仓库边界：

```python
adapter = build_optimization_adapter(
    "search.random_gaussian",
    population_size=16,
    mutation_scale=0.2,
)
```

## 为什么仍然拆层

统一的是求解视图，不是职责。Adapter 不读取训练集、不解释概率输出、不持有模型
Artifact，也不选择 GPU；Problem/Provider 不决定 Adam、随机搜索或多目标策略；L0
只授权资源而不实现 loss 或优化公式。由此同一个算法可以跨 ML 与运筹问题复用，
同一个 ML Problem 也可以无缝替换梯度、黑盒、多目标或嵌套搜索机制。

优化策略全部来自 `nsgablack.adapters`；MLBlack 不再定义 GradientDescent、RandomSearch 或 EstimatorSpecSearch Adapter，
各后端 backprop Adapter 只作为迁移兼容面保留。新 preset、Case 与 Catalog 规范入口
必须使用稳定方法装配。
