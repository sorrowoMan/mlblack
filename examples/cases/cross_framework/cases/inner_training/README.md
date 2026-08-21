# Inner Training Case

这是 `cross_framework` 项目的内层 Trainer Case。它通过规范 `build_solver.py` 装配正交线性点预测训练器，供外层 Case 以共享 `CaseRunRequest` 调用，并继承父级资源、预算、取消与 lineage。

```powershell
python run_solver.py --check
```
