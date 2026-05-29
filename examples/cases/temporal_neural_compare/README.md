# temporal_neural_compare

Compare 7 temporal neural forecasting models on synthetic sine wave data.

## 是否使用 mlblack / nsgablack

Pure mlblack -- zero custom ML components.

## 这个 case 验证什么

Demonstrates pure framework preset composition. All 7 temporal neural architectures (LSTM, TCN, Transformer, N-BEATS, DeepAR, PatchTST, TFT) are compared on a common synthetic forecasting task using only existing framework presets, adapter, problem, and representation components.

## 搜索向量

| 变量 | 含义 | 范围 |
|---|---|---|
| Neural graph weights/biases | Model parameters | float, network-dependent |

## 目标和指标

| 目标 | 方向 | 含义 |
|---|---|---|
| MSE (via NeuralGraphBackprop) | minimize | Forecast error on sin(0.1*t) + noise |

## 组件组合

| 层 | 组件 | 来源 |
|---|---|---|
| Problem | TemporalNeuralForecastingProblem | framework problem.temporal_neural_forecasting |
| Representation | NeuralGraphRepresentation (7 variants) | framework preset (via NeuralGraphSpec) |
| Adapter | NeuralGraphBackpropAdapter | framework adapter.neural_graph_backprop |
| Bias | (none) | -- |

## 效果对比

| Model | RMSE | Time (s) | vs baseline |
|---|---|---|---|
| LSTM | TBD | TBD | TBD |
| TCN | TBD | TBD | TBD |
| Transformer | TBD | TBD | TBD |
| N-BEATS | TBD | TBD | TBD |
| DeepAR | TBD | TBD | TBD |
| PatchTST | TBD | TBD | TBD |
| TFT | TBD | TBD | TBD |

## 结构

| 路径 | 作用 |
|---|---|
| build_solver.py | Assembly: imports all 7 presets, runs comparison |
| pipeline/data_generator.py | Synthetic sin+noise data with lag features |
| run_solver.py | CLI entrypoint |

## 运行和验证

```powershell
python run_solver.py
```
