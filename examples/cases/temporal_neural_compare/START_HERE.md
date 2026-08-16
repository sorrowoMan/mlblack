# 开始：时序神经模型 Project 比较

先检查七个独立 Trainer Case 的实际装配和资源上下文：

```powershell
python examples/cases/temporal_neural_compare/run_project.py --check --build-check
```

正式运行：

```powershell
python examples/cases/temporal_neural_compare/run_project.py
```

单 Case 调试示例：

```powershell
python examples/cases/temporal_neural_compare/cases/temporal_lstm/run_solver.py --check
python examples/cases/temporal_neural_compare/cases/temporal_lstm/run_solver.py --steps 5
```
