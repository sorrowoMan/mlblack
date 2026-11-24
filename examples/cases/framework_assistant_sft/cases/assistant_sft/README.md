# assistant_sft

唯一装配入口是 `build_solver.py`，复用 `mlblack.integrations.hf_sft`；数据格式与完整运行步骤见上层 Project 的 [README](../../README.md)。本 Case 默认不下载模型，实际执行前需要显式设置本地基座和训练数据路径。
