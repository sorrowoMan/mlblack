# 统一梯度训练路径

这条路径把“优化方法”和“ML 计算语义”分开，但不要求用户理解仓库边界。

```text
optimizer="adam"
  -> stable method: gradient.adam
  -> mlblack LearningSolver facade
       只提供 fit / TrainerResult / Artifact 的 ML 词汇投影
  -> nsgablack ComposableSolver
       唯一拥有 lifecycle、incumbent、budget、cancellation、snapshot
  -> nsgablack GradientOptimizerAdapter
       决定学习率、moment 规则、weight decay、step
  -> mlblack ProviderBackedLearningProblem
       保留 objective / loss / metric / output 语义
  -> BlackBase EvaluationGateway
       按 problem_id + capability + L0 grant 绑定 Provider
  -> mlblack TorchEvaluationProvider
       执行 representation lowering、loss、autograd、CPU/GPU kernel
       发布参数 StateRef 与 Feedback.gradient_ref
  -> BlackBase StateTransitionRequest
       Provider 版本栅栏执行 gradient.sgd/adam/adamw
       参数、梯度与 optimizer slots 保持在设备侧
  -> BlackBase StateMaterializationRequest
       导出 UnknownState，成功后释放本轮活参数引用
```

## 所有权

- `NumericBatchSchedule` 拥有 epoch、shuffle、batch cursor，并可独立保存/恢复。
- `nsgablack.ComposableSolver` 是唯一优化控制面；`fit()` 不是第二套循环。
- `LearningSolver` 只把 ML Problem/Representation/Provider 投影到该控制面，并把权威 incumbent 投影为 `TrainerResult`。
- ML `Problem` 拥有训练/验证目标、loss、metric 和输出语义。
- `TorchEvaluationProvider` 拥有 Torch/autograd/device 执行、活参数、梯度和 optimizer slot，但不能选择 Adam。
- `GradientOptimizerAdapter` 拥有 SGD/Adam/AdamW 策略和参数选择，但不能读取数据、持有 Tensor 或申请 GPU。
- Project L0 的 `ResourceContext` 是设备和资源的唯一授权；`device="cuda"` 配置不能自行制造 GPU grant。
- BlackBase 拥有 `StateRef`、transition、materialization 和绑定审计协议，但不实现任何优化公式或 ML loss。

## 状态与恢复语义

`StateRef` 是 Provider 进程内活状态，不是 Artifact，也不是 checkpoint 数据。每次
transition 都必须匹配当前版本；旧版本会触发 `StateVersionConflict`。Adapter 取得
successor 后，通过 materialization 导出可保存的数值状态，并可要求 Provider 释放
本轮参数引用。Adam/AdamW 的 `m/v` slot 由 Provider 持有并逐步版本化；Solver teardown
再通过 BlackBase `StateReleaseRequest` 按 scope + trajectory 原子释放剩余 slot。

checkpoint 不会序列化 CUDA/Torch 对象，也不会声称能够恢复旧进程的 `StateRef`。
恢复后控制面从已物化的 `UnknownState` 和 Adapter 的数值 moment 影子继续；第一次
新 Provider transition 会用 moment 影子 reseed 新的 live slot，随后继续同一设备执行
路径，不会永久降级为本地 shadow 更新。
当前统一 Builder 在启用 StateRef 时要求 `inline_gradients=True`，用于维护可精确恢复的
数值 optimizer 影子；配置为仅返回 `gradient_ref` 会在装配期明确失败，而不是生成
一份丢失 Adam moment 的伪完整 checkpoint。

## 用户入口

原有 ML 词汇继续成立：

```python
trainer = build_mlp_regression_trainer(
    data,
    optimizer="adam",
    learning_rate=1e-3,
    batch_size=64,
)
```

公共入口只有 `build_mlp_regression_trainer()`。预设不装配 ML 私有 backprop Adapter，而是解析为：

```text
adam
  -> gradient.adam
  -> GradientOptimizerAdapter
  -> ComposableSolver（唯一控制面）
  -> TorchEvaluationProvider
  -> BlackBase EvaluationGateway / Project L0
```

需要显式组合时使用：

```python
trainer = build_gradient_trainer(
    problem=problem,
    representation=representation,
    method="gradient.adam",
    compute_backend="torch",
)
```

当 Problem 自己提供解析梯度，不需要 Torch/autograd Provider 时仍使用同一个稳定方法：

```python
trainer = build_gradient_trainer(
    problem=analytic_problem,
    representation=representation,
    method="gradient.sgd",
    compute_backend="problem",
)
```

此时 `gradient.sgd` 的策略仍归 nsgablack；区别只是梯度由 ML Problem 直接产生，
不创建虚假的 Provider 或设备状态。

稳定公共概念是 method 与 compute backend 分离。Torch 使用正式 Evaluation Provider；
JAX/TensorFlow 使用 `FunctionalGradientLearningProblem`；解析梯度可走
`compute_backend="problem"`。

## 当前迁移边界

正式 Torch 路径现已覆盖：

- `NeuralGraphSpec.mlp` + `NeuralGraphCodec` + Torch MLP lowering；
- Tiny Transformer 分类、语言模型和 DPO；
- Tiny CNN 分类/对比学习、Tiny GNN；
- LSTM、TCN、Temporal Transformer、N-BEATS、DeepAR、PatchTST、TFT；
- TabNet 表格分类与回归。

MLBlack 不再包含私有 Trainer 控制循环或优化 Adapter。Torch、JAX、TensorFlow、
解析梯度与第三方 estimator 都通过 ML Problem/Provider 生成反馈，统一交给
NSGABlack Adapter 和 ComposableSolver 生命周期。
