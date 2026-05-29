# START_HERE

## 1) 这个 case 验证什么
`tsne_umap` 验证 t-SNE 降维可以作为梯度优化问题。

- mlblack 将高维数据嵌入到 2D 空间。
- Problem 计算 KL 散度并回传梯度。
- Adapter 使用框架内置的梯度下降。

## 2) 运行
```powershell
Set-Location "C:\Users\hp\Desktop\mlblack\examples\cases\tsne_umap"
python build_solver.py --steps 50
```

## 3) 关键指标
| 指标 | 含义 |
|---|---|
| KL divergence | 高维亲和度 P 与低维亲和度 Q 之间的 KL 散度，越低越好。 |

## 4) 预期信号
KL 散度从 ~22（随机初始化）开始下降，经过数百步收敛到低值。
