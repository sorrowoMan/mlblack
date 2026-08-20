# temporal_neural_compare

这是一个正式的 Project 级时序神经模型比较示例。

比较任务已拆成七个独立 Trainer Case：

- `temporal_lstm`
- `temporal_tcn`
- `temporal_transformer`
- `temporal_nbeats`
- `temporal_deepar`
- `temporal_patchtst`
- `temporal_tft`

每个 Case 独立装配一个 `LearningSolver`，共享同一套确定性正弦序列数据协议。模型顺序、资源授权和运行编排由本 Project 的 `project_config.py` 管理，不再由单个 Case 返回 Trainer 字典。

旧的单 LSTM 兼容 Case 已删除；Project 只暴露上述七个真实 Trainer Case。

验证装配及实际 L0 grant：

```powershell
python examples/cases/temporal_neural_compare/run_project.py --check --build-check
```

执行七个模型：

```powershell
python examples/cases/temporal_neural_compare/run_project.py
```
