# MLBlack + Blackbase 统一 substrate 演示

这是一个标准 `Project -> Case -> Scaffold` 示例。顶层 `run_project.py` 负责阶段编排和 L0 资源授权，`cases/blackbase_demo/` 是可独立运行的 Trainer Case。

```powershell
python run_project.py --check --build-check
python cases\blackbase_demo\run_solver.py --steps 3
```

运行时会通过统一 Project runner 展示实际生效的资源上下文、组件和命名空间，便于审计“声明了什么”和“实际使用了什么”。
