# linear_point

规范入口是 `build_solver.py` / `run_solver.py`；trainer 文件只是薄别名。
装配 NumericDataView、LinearPointRepresentation（LinearPointCodec + PointHead）、
SupervisedRegressionProblem、GradientOptimizerAdapter 和 LearningSolver。
固定 seed=17、24条合成样本、2个特征、3步、学习率0.05；CPU授权来自上层Project。
没有额外Bias、Plugin、远程服务或Torch依赖。结果由标准TrainerResult与Project报告交付。
