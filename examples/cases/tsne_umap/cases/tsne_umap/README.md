# t-SNE / UMAP Style Embedding

这个 Case 把降维嵌入表达成 `mlblack` 的 ML 语义任务。它本身不需要 `nsgablack`；只有当外层 Project 要搜索 perplexity、learning rate、初始化、neighborhood policy 等 tradeoff 时，才需要优化搜索 Case。

## 边界

| 层 | 责任 |
| --- | --- |
| `problem/` | KL divergence 或 embedding-quality objective / feedback |
| `pipeline/` | 数据准备和 model-state encode/decode helper |
| `adapter/` | gradient-based 或 black-box update strategy |
| `build_solver.py` | canonical Case assembly entry |

旧 `pipeline/representation/` 引用只是 compatibility remnants。新的编码逻辑应放在 Case `pipeline/` 或框架语义模块。

## 运行

```powershell
python examples\cases\tsne_umap\run_project.py --check --build-check
python examples\cases\tsne_umap\cases\tsne_umap\run_solver.py --steps 50 --perplexity 30 --lr 200
python -m compileall -q examples\cases\tsne_umap
```
