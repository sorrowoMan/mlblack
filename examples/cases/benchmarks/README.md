# 神经训练基准 Project

本 Project 将四种神经训练路径拆成四个独立 Trainer Case：

- Tiny CNN 图像分类
- Tiny GNN 图分类
- Tiny CNN 对比学习
- Tiny Transformer 语言建模

每个 Case 只装配一个真实 `LearningSolver`；并行、CPU grant、运行耗时和失败信封由
BlackBase Project/L0 统一管理。不存在循环构造多个 Trainer 的兼容 Runner。

```powershell
python examples\cases\benchmarks\run_project.py --check --build-check
python examples\cases\benchmarks\run_project.py
```

可重复运行 Project，并直接比较每次 manifest 中各 Case 的 wall time 与 TrainerResult。
