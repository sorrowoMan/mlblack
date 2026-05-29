# 04. nsgablack 编排与 L0 资源层的统一设计

这一章重新定位 `mlblack` 和 `nsgablack` 在嵌套架构中的角色，建立统一的 L0 资源层和编排规范。**关键变化**：用统一的嵌套编排标准代替分割式设计。

## 1. 统一的嵌套编排模型

所有多任务/多阶段/多模型场景都遵循同一个模式：

```text
外层（nsgablack Solver）
  ├─ 职责：搜索空间编码、阶段编排、并行/串行调度、资源授权、多目标管理
  └─ 评估：Problem.evaluate(x) 短路调用内层
       ├─ 解码外层向量为内层参数
       ├─ 注入 ResourceContext（L0 资源授权）
       └─ 调用 build_trainer / build_solver / ...
  
内层脚手架（mlblack Trainer / 嵌套 nsgablack Solver / 自定义优化器）
  ├─ 职责：单层优化、模型拟合、参数搜索、数据评估
  ├─ 被动消费：ResourceContext
  └─ 产出：Feedback / Artifact / Report
```

**这是统一的，不管内层是什么**。详见 [nsgablack 嵌套编排标准](../../nsgablack/docs/standard_scaffold_tutorial/07_nested_orchestration_standard.md)。

---

## 2. 一个完整的例子：超参优化

### 2.1 标准文件结构

```text
examples/cases/mlblack_hyperopt/
├─ outer/                              # 外层（nsgablack）
│  ├─ problem/
│  │  └─ hyperopt_problem.py           # 调用内层训练，返回目标
│  ├─ pipeline/
│  │  └─ param_encoding.py             # 超参向量编码/解码
│  ├─ adapter/
│  │  └─ config.py
│  ├─ solver/
│  │  └─ config.py
│  ├─ build_solver.py
│  └─ run_solver.py
│
├─ inner/                              # 内层（mlblack）
│  ├─ problem/
│  │  └─ training_problem.py           # 数据加载、模型评估
│  ├─ pipeline/
│  │  └─ data_pipeline.py              # 数据预处理
│  ├─ pipeline/
│  │  └─ representation.py              # 模型参数编码（mlblack pipeline 组件）
│  ├─ adapter/
│  │  └─ config.py                     # 梯度下降/搜索策略
│  ├─ build_solver.py                  # canonical 装配入口
│  ├─ build_trainer.py                 # 别名
│  └─ run_solver.py                    # canonical CLI 入口
│
├─ build_solver.py                     # 顶层入口
├─ run_solver.py
└─ README.md
```

### 2.2 内层独立脚手架

`inner/build_solver.py`（canonical entry, `build_trainer.py` 是别名）：

```python
def build_solver(
    hyperparams: dict | None = None,
    data: tuple | None = None,
    resource_context: dict | None = None,  # ← 从外层注入
    budget: int = 100,
):
    """
    构造 mlblack Trainer（Solver=Trainer 统一抽象）。
    
    Args:
        hyperparams: 外层解码的超参字典（JSON-compatible）
        data: (X_train, y_train)
        resource_context: 外层授权的资源（device/threads/namespace/budget）
        budget: 最多多少训练步
    
    Returns:
        ComposableTrainer，已装配完毕，可直接 .fit()
    """
    
    # ① 创建问题
    from problem.training_problem import MLTrainingProblem
    problem = MLTrainingProblem(data=data)
    
    # ② 创建表示
    from representations.model_representation import MLRepresentation
    representation = MLRepresentation(hyperparams)
    
    # ③ 创建优化器
    from mlblack.adapters import GradientDescentAdapter
    adapter = GradientDescentAdapter(
        learning_rate=hyperparams.get("learning_rate", 0.001)
    )
    
    # ④ 创建 Trainer
    from mlblack.core.trainer import ComposableTrainer
    trainer = ComposableTrainer(
        problem=problem,
        representation=representation,
    )
    trainer.set_adapter(adapter)
    
    # ⑤ 注入资源上下文（关键）
    if resource_context:
        trainer.set_resource_context(resource_context)
        # → Trainer 会把 device/threads/budget 传给 ComputeBackendSession
    
    trainer.set_max_steps(budget)
    
    return trainer
```

### 2.3 外层调用内层

`outer/problem/hyperopt_problem.py`：

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../inner'))

from nsgablack.core.problem import BlackBoxProblem
from nsgablack.core.types import Feedback
from build_solver import build_solver

class HyperoptProblem(BlackBoxProblem):
    def __init__(self, data=None, resource_context=None):
        self.data = data
        self.resource_context = resource_context  # 外层授权
    
    def evaluate(self, x):
        """
        外层候选向量 x → 内层训练 → 返回目标
        """
        try:
            # ① 解码超参
            hyperparams = {
                "learning_rate": 10 ** x[0],          # x[0] in [-5, 0]
                "batch_size": int(2 ** x[1]),         # x[1] in [3, 6]
                "num_layers": int(x[2]),              # x[2] in [1, 5]
                "hidden_dim": int(x[3] * 256),        # x[3] in [0.25, 2]
            }
            
            # ② 调用内层构造器（关键）
            trainer = build_solver(
                hyperparams=hyperparams,
                data=self.data,
                resource_context=self.resource_context,  # ← 传资源授权
                budget=100,
            )
            
            # ③ 内层训练
            result = trainer.fit()
            
            # ④ 提取外层目标
            return Feedback(
                objectives=[result.best_loss],
                constraints=[],
                metrics={
                    "valid_loss": float(result.best_loss),
                    "num_params": int(hyperparams["num_layers"] * hyperparams["hidden_dim"]),
                },
            )
        
        except Exception as e:
            return Feedback(
                objectives=[1e10],
                constraints=[],
            )
```

`outer/build_solver.py`：

```python
from nsgablack.core.composable_solver import ComposableSolver
from nsgablack.adapters.nsga2 import NSGA2Adapter

from problem.hyperopt_problem import HyperoptProblem

def build_solver(
    data=None,
    resource_context=None,  # 外层的 L0 资源授权
):
    """
    构造外层 nsgablack 求解器。
    """
    
    # ① 创建外层问题（调用内层 trainer）
    problem = HyperoptProblem(
        data=data,
        resource_context=resource_context,  # → 传给内层
    )
    
    # ② 创建求解器
    solver = ComposableSolver(
        problem=problem,
        search_space_dims=4,
        search_space_bounds=[
            (-5, 0),      # learning_rate log scale
            (3, 6),       # batch_size log scale
            (1, 5),       # num_layers
            (0.25, 2),    # hidden_dim scale
        ],
    )
    
    # ③ 设置外层适配器（多目标优化）
    adapter = NSGA2Adapter(population_size=20)
    solver.set_adapter(adapter)
    
    return solver
```

顶层 `build_solver.py`：

```python
from nsgablack.core.resources import ResourceContext
from outer.build_solver import build_solver

# 加载数据
data = load_my_data()

# ① L0 资源授权（关键）
resource_context = {
    "compute_backend": "torch_cuda:0",   # 哪个 GPU
    "parallelism": 2,                     # 最多并行多少内层任务
    "memory_limit_mb": 4096,              # 内存预算
    "threads": 4,                         # CPU 线程
}

# ② 外层求解器
solver = build_solver(
    data=data,
    resource_context=resource_context,
)

# ③ 运行
result = solver.evolve(generations=10)
```

---

## 3. L0 资源层的统一设计

### 3.1 资源流

```text
nsgablack L0 authorize (外层)
  ├─ allocation: who gets what (GPU id, thread count, memory)
  ├─ lease: temporal resource lock
  ├─ scheduling: parallel fanout + fairness
  └─ monitoring: resource usage + violations

       ↓ ResourceContext JSON ↓

mlblack L0 consume (内层)
  ├─ PassiveContext.read(device, threads, budget)
  ├─ ComputeBackendSession.create(backend_name, device, ...)
  ├─ audit: 记录实际消费
  └─ fail-fast if invalid or insufficient
```

### 3.2 ResourceContext 字段

```python
resource_context = {
    # 计算后端选择
    "compute_backend": "torch_cuda:0",       # torch / jax / tensorflow / numpy
    "device": "cuda:0",                      # device pin (if applicable)
    
    # 并发控制
    "parallelism": 2,                        # 当前任务最多并行几个子任务
    "threads": 4,                            # CPU 线程数
    
    # 预算
    "memory_limit_mb": 4096,                 # 内存上限
    "inner_steps_budget": 100,               # 单次 inner trainer 最多步数
    "total_time_seconds": 3600,              # 总时间预算
    
    # 名空间/审计
    "namespace": "suite.demo/stage1/candidate_0007",
    "audit_every_n_steps": 5,
    
    # 可选：失败策略
    "on_resource_violation": "strict",       # strict / soft / warning
}
```

### 3.3 mlblack 怎样消费资源

```python
# inner/build_solver.py

def build_solver(hyperparams, data, resource_context, budget):
    from mlblack.core import ComputeBackendSpec
    
    # 被动读取资源授权
    if resource_context:
        device = resource_context.get("device", "cpu")
        threads = resource_context.get("threads", 1)
        memory_mb = resource_context.get("memory_limit_mb", 1024)
        backend_name = resource_context.get("compute_backend", "numpy")
    else:
        device, threads, memory_mb, backend_name = "cpu", 1, 1024, "numpy"
    
    # 创建计算后端会话
    compute_backend = ComputeBackendSpec(
        name=backend_name.split("_")[0],  # torch / jax / tensorflow / numpy
        device=device,
        threads=threads,
        memory_limit_mb=memory_mb,
    )
    
    trainer = ComposableTrainer(
        problem=problem,
        representation=representation,
        compute_backend=compute_backend,
    )
    
    # 关键：不要自己分配资源
    # 不要写：
    #   os.environ["CUDA_VISIBLE_DEVICES"] = device
    #   torch.cuda.set_device(...)
    # 而是让后端会话根据授权的 device 去初始化
    
    return trainer
```

---

## 4. 多阶段编排：Stage 模式

### 4.1 串行阶段

这种情况下每个 stage 都有一个独立的 `mlblack Trainer`：

```text
nsgablack stage 1:
  └─ mlblack trainer 1 (base model)
  
↓ artifact flow: model → residual target

nsgablack stage 2:
  └─ mlblack trainer 2 (residual model)

↓ artifact flow: models → composition

nsgablack stage 3:
  └─ mlblack model composition (final prediction)
```

`outer/problem/stage1_problem.py`：

```python
class BaseModelProblem(BlackBoxProblem):
    def evaluate(self, x):
        hyperparams = decode_base_hyperparams(x)
        trainer = build_base_trainer(hyperparams, self.data, self.resource_context)
        result = trainer.fit()
        
        # 返回模型 artifact ref
        artifact_ref = result.model_artifact_ref
        
        return Feedback(
            objectives=[result.best_loss],
            artifacts={"model": artifact_ref},
        )
```

`outer/problem/stage2_problem.py`：

```python
class ResidualModelProblem(BlackBoxProblem):
    def __init__(self, data, base_model_artifact_ref, resource_context):
        self.base_model_artifact_ref = base_model_artifact_ref
        self.resource_context = resource_context
    
    def evaluate(self, x):
        hyperparams = decode_residual_hyperparams(x)
        
        # 读取前一阶段 artifact
        base_model = read_artifact(self.base_model_artifact_ref)
        
        # 生成 residual target
        from mlblack.pipeline import ModelConditionedTargetComponent
        residual_data = ModelConditionedTargetComponent().build(
            self.data,
            reference_model=base_model,
        )
        
        # 训练残差模型
        trainer = build_residual_trainer(hyperparams, residual_data, self.resource_context)
        result = trainer.fit()
        
        return Feedback(
            objectives=[result.best_loss],
            artifacts={"residual_model": result.model_artifact_ref},
        )
```

`outer/build_solver.py`：

```python
def build_solver(data, resource_context):
    from nsgablack.core import SerialStageSolver
    
    solver = SerialStageSolver(
        stages=[
            {
                "name": "base",
                "problem": BaseModelProblem(data, resource_context),
                "solver": ComposableSolver(...),
            },
            {
                "name": "residual",
                "problem": ResidualModelProblem(data, ..., resource_context),
                "solver": ComposableSolver(...),
                "consume_artifacts_from": "base",
            },
        ],
    )
    
    return solver
```

### 4.2 并行分支

多个 trainer 同时训练（不同分支或候选）：

```text
nsgablack group:
  ├─ tabular trainer
  ├─ image trainer
  └─ text trainer
  
↓ artifact: three models

mlblack composition:
  └─ IntegratedPredictionModel (fusion)
```

`outer/problem/multimodal_problem.py`：

```python
class MultimodalProblem(BlackBoxProblem):
    def evaluate(self, x):
        # x 编码了三个 branch 的超参 + fusion weights
        tabular_params = decode_tabular_params(x[:4])
        image_params = decode_image_params(x[4:8])
        fusion_weight = x[8]
        
        # 并行训练三个 branch
        # （nsgablack group 负责并行调度）
        results = parallel_train([
            ("tabular", tabular_params, self.data["tabular"]),
            ("image", image_params, self.data["image"]),
            ("text", text_params, self.data["text"]),
        ], resource_context=self.resource_context)
        
        # 组合模型（mlblack 负责）
        from mlblack.models import PredictionIntegrationComponent
        final_model = PredictionIntegrationComponent.mean(
            component_order=("tabular", "image", "text"),
            weights={
                "tabular": fusion_weight,
                "image": fusion_weight,
                "text": 1 - 2*fusion_weight,
            },
        ).compose({
            "tabular": results["tabular"].best_model,
            "image": results["image"].best_model,
            "text": results["text"].best_model,
        })
        
        # 评估组合模型
        val_score = evaluate_integrated_model(final_model, self.data)
        
        return Feedback(
            objectives=[val_score],
            artifacts={"composed_model": final_model},
        )
```

---

## 5. 符号学习：外层搜索 + 内层拟合

符号学习是嵌套编排的典型应用。详见 [第 5 章](05_symbolic_nested_case.md)。

```text
nsgablack outer:
  搜索表达式结构、basis 选择、复杂度权衡

mlblack inner:
  拟合表达式参数、计算 residual/gradient
```

---

## 6. 多 Trainer 串联（SerialTrainer）

当一个 **内层** 需要多个顺序 stage 时，用 `mlblack.core.trainer.SerialTrainer`：

```python
# inner/build_solver.py

from mlblack.core.trainer import SerialTrainer

def build_multi_stage_trainer(hyperparams, data, resource_context, budget):
    """多阶段训练（都在内层，不涉及外层编排）"""
    
    stages = [
        StageSpec(
            name="pretrain",
            factory=lambda: build_pretrain_trainer(hyperparams, data, resource_context, budget//2),
            output_artifacts=["init_state"],
        ),
        StageSpec(
            name="finetune",
            factory=lambda: build_finetune_trainer(hyperparams, data, resource_context, budget//2),
            input_artifacts={"init_state": "pretrain.init_state"},
        ),
    ]
    
    return SerialTrainer(stages)
```

外层调用时完全相同：

```python
trainer = build_solver(...)
result = trainer.fit()
```

---

## 7. 检查清单

部署跨框架 case 时确保：

- [ ] 内层是完整的标准脚手架，能独立运行
- [ ] `build_solver()` 接收 JSON-compatible 参数和 `resource_context`
- [ ] 内层不 import 外层模块
- [ ] 外层 Problem.evaluate() 中完整地调用内层构造+运行+提取指标
- [ ] ResourceContext 正确从外层传入内层
- [ ] 大对象走 artifact/snapshot ref，不塞进 context
- [ ] 内层 Trainer 不自己分配资源（GPU、线程、内存）
- [ ] 如果外层是 SerialStageSolver，每个 stage 的 problem 都遵循相同的外/内分层
- [ ] 运行入口能够打印完整的资源上下文、启用组件、生效后端

---

## 参考

- nsgablack 嵌套编排标准：[07_nested_orchestration_standard.md](../../nsgablack/docs/standard_scaffold_tutorial/07_nested_orchestration_standard.md)
- mlblack L0 资源消费：[00_assembly_api_reference.md](00_assembly_api_reference.md)
- 符号学习示例：[05_symbolic_nested_case.md](05_symbolic_nested_case.md)
