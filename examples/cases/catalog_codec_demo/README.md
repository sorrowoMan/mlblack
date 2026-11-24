# Catalog：直接线性 Codec 的 CPU 装配示例

这是一个正式 Project → Case → Scaffold，只有 `linear_point` 一个 Case。
运行 `python examples/cases/catalog_codec_demo/run_project.py --no-record`。
要求已安装或可导入 `blackbase`、`nsgablack`、`mlblack` 和 NumPy，不需要 Torch、GPU 或模型 API。

数据由 `pipeline/main.py` 生成：固定 seed=17，24 行、2 个特征，
目标为 `1 + 2*x0 - 0.5*x1`。`LinearPointRepresentation` 内部实际使用
`LinearPointCodec` 和 `PointHead`，而不是正交特征 Codec。
`SupervisedRegressionProblem` 定义原始回归评估；`gradient.sgd` 由
`nsgablack` 的 GradientOptimizerAdapter 执行。LearningSolver 消费 Project L0
发放的 CPU grant，最多提交3步。Case 目录中的四个入口保持正式/别名关系。

这个小运行用于验证组件装配与真实求值，3步结果不代表模型收敛或算法优越性。
它没有装配 NeuralGraphCodec，也不能用来声称神经网络后端已验证。
