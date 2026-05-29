# 05. Symbolic Nested Case 标准读法

这一章用符号学习说明标准跨框架结构：外层搜索结构，内层拟合参数。它遵循统一的嵌套编排标准（见 [04_nsgablack_orchestration_and_resource_layers.md](04_nsgablack_orchestration_and_resource_layers.md) 和 [nsgablack 嵌套编排标准](../../nsgablack/docs/standard_scaffold_tutorial/07_nested_orchestration_standard.md)）。

## 0. 项目文件结构（关键）

```text
examples/cases/symbolic_orthogonal_nested/
│
├─ outer/                                    # 外层（nsgablack）
│  ├─ problem/
│  │  ├─ __init__.py
│  │  ├─ stage1_basis_search.py             # 调用内层 basis fitting
│  │  └─ stage2_task_search.py              # 调用内层 task fitting
│  ├─ pipeline/
│  │  └─ basis_index_encoding.py            # 外层候选 → basis index
│  ├─ adapter/
│  │  └─ config.py
│  ├─ solver/
│  │  └─ config.py
│  ├─ build_solver.py
│  └─ run_solver.py
│
├─ inner/                                    # 内层（mlblack）
│  ├─ problem/
│  │  ├─ __init__.py
│  │  ├─ basis_fitting_problem.py           # 参数拟合
│  │  └─ task_fitting_problem.py            # 参数拟合
│  ├─ representations/
│  │  ├─ basis_representation.py            # symbolic codec
│  │  └─ task_representation.py
│  ├─ pipeline/
│  │  └─ data_pipeline.py
│  ├─ build_trainer.py
│  └─ run_trainer.py
│
├─ build_solver.py                           # 顶层入口
├─ run_solver.py
└─ README.md
```

**关键**：outer/ 和 inner/ 都是完整的独立脚手架。

## 1. 核心结构

```text
nsgablack outer:
  搜索表达式结构、basis 组合、任务表达式、复杂度权衡。

mlblack inner:
  表示符号表达式、拟合参数、计算 residual/gradient、产出 artifact。
```

两阶段示例：

```text
Stage 1:
  outer searches orthogonal symbolic basis
  inner fits basis parameters
  output OrthogonalBasisSetArtifact

Stage 2:
  outer searches task expression over basis artifact
  inner fits task expression parameters
  output SymbolicTaskArtifact
```

## 2. 为什么不是单 trainer

符号学习同时包含：

```text
结构搜索
参数拟合
表达式规范化
truth recovery
family/phase-equivalence scoring
orthogonality / condition / rank metrics
artifact lineage
```

其中结构搜索和阶段顺序属于 `nsgablack`；表达式、拟合、评分和 artifact 属于 `mlblack`。

## 3. Stage 1 数据流

```text
outer candidate
  -> FunctionPoolIndexSearchSpace.decode
  -> symbolic basis terms
  -> mlblack symbolic representation/problem
  -> inner parameter fitting
  -> orthogonality/stability/complexity metrics
  -> outer objectives
  -> OrthogonalBasisSetArtifact
```

Stage 1 关注：

| 指标 | 含义 |
| --- | --- |
| basis fit error | basis 对数据的解释能力 |
| orthogonality error | basis 之间是否冗余 |
| condition number | 数值稳定性 |
| rank | 有效维度 |
| complexity | 表达式复杂度 |
| family diversity | 函数族是否过度集中 |

## 4. Stage 2 数据流

```text
Stage 1 basis artifact
  -> basis-conditioned function pool
  -> outer candidate selects task terms
  -> mlblack fits task parameters
  -> task metrics / recovery scoring
  -> SymbolicTaskArtifact
```

Stage 2 关注：

| 指标 | 含义 |
| --- | --- |
| rmse / task loss | 任务拟合质量 |
| interval/probability/classification metric | head 对应指标 |
| canonical exact recovery | 是否恢复真表达式规范形式 |
| family recovery | 是否恢复同族表达式 |
| phase equivalence | 是否恢复相位等价结构 |
| complexity | 是否过度复杂 |

## 5. 关键 mlblack 组件

| 组件 | 位置 | 职责 |
| --- | --- | --- |
| `SymbolicExpressionModel` | `mlblack.models.symbolic` | 表达式模型 |
| `SymbolicExpressionCodec` | `mlblack.representations.codecs.symbolic` | genome 到 expression |
| `FixedSymbolicRegressionProblem` | `mlblack.problems.symbolic` | 固定表达式参数拟合 |
| `FunctionPoolPipeline` | `mlblack.pipeline.symbolic` | 函数池生成 |
| `DynamicFunctionPoolPipeline` | `mlblack.pipeline.symbolic` | residual/gradient 驱动扩池 |
| `SymbolicExpressionAuditProducer` | integration surface | canonical/truth/family/phase report |
| `OrthogonalBasisSetArtifact` | integration artifact | Stage 1 产物 |
| `SymbolicTaskArtifact` | integration artifact | Stage 2 产物 |

## 6. 关键 nsgablack 组件

| 组件 | 职责 |
| --- | --- |
| `EvolutionSolver` | outer lifecycle |
| `NSGA2Adapter` | multi-objective outer search |
| outer representation pipeline | index-coded candidate repair/mutate/decode |
| outer problem | 调用 mlblack inner surface 并投影 objectives |
| L0 resource context | 注入 inner trainer 资源 |
| snapshot/report plugin | 记录 stage 和 artifacts |

## 7. 正式 case 入口

```powershell
python examples\cases\symbolic_orthogonal_nested\run_solver.py --check
```

小规模 smoke：

```powershell
python examples\cases\symbolic_orthogonal_nested\run_solver.py `
  --stage1-generations 1 `
  --stage2-generations 1 `
  --stage1-pop-size 4 `
  --stage2-pop-size 4 `
  --stage1-inner-steps 2 `
  --stage2-inner-steps 2
```

输出应包含：

```text
summary.json
stage1 records
stage2 records
basis artifact
task artifact
artifact html/dashboard
```

## 8. 结构守卫和动态扩池

### 8.1 Structure guard

用于拒绝明显无效结构：

```text
duplicate expression
invalid pole / domain instability
value explosion
redundant family overuse
complexity over budget
```

归属：search policy / problem-facing guard，不是 adapter 内硬编码。

### 8.2 Dynamic pool

动态扩池来源：

```text
residual signal
gradient signal
family coverage
phase-equivalence miss
path memory
```

动态扩池本质是生成更多候选 term 的 pipeline/search policy，不是 mlblack workflow。

## 9. Truth recovery 与 canonical schema

Artifact 应分层记录：

```text
final_expression:
  原始最终表达式

canonical_expression:
  canonical key / canonical string / simplification payload

truth_contract_recovery:
  exact / approximate / normalized matching

family_recovery:
  sin / cos / polynomial / rational / exp/log 等族级评分

phase_equivalence_recovery:
  sin(ax+b) / cos(ax+b) 等相位等价评分

lineage:
  stage / candidate / basis artifact / inner fit report
```

## 10. 为什么并行 batch runtime 不在 mlblack

符号候选批量评估、branch evaluator 并行、solver fanout 这些是运行时编排：

```text
parallel workers
batch scheduling
resource lease
device/thread assignment
failure retry
```

归 `nsgablack` L0/runtime。`mlblack` 只保证单个 inner task 的 evaluation surface 是干净的。

## 11. 资源流：L0 层的垂直传递

```text
顶层 (build_solver.py)
  ├─ L0 ResourceContext 授权
  │  {
  │    "compute_backend": "torch_cuda:0",
  │    "parallelism": 2,
  │    "memory_limit_mb": 4096,
  │  }
  └─ 外层 nsgablack Solver

外层 Stage 1 (outer/problem/stage1_basis_search.py)
  ├─ 接收 ResourceContext
  ├─ 候选向量 → basis index
  ├─ 调用 inner build_trainer
  │  ✓ 传 basis spec
  │  ✓ 传 ResourceContext（关键）
  └─ 返回 basis fitting metrics

       ↓ artifact: OrthogonalBasisSetArtifact

内层 (inner/build_trainer.py - basis stage)
  ├─ 消费 ResourceContext
  │  device = context["compute_backend"]
  │  threads = context.get("threads", 1)
  ├─ 初始化 ComputeBackendSession
  ├─ 参数拟合（orthogonal basis）
  └─ 产出 basis artifact

外层 Stage 2 (outer/problem/stage2_task_search.py)
  ├─ 读取 basis artifact
  ├─ 候选向量 → task terms
  ├─ 调用 inner build_trainer
  │  ✓ 传 basis conditioned function pool
  │  ✓ 传 ResourceContext
  └─ 返回 task fitting metrics

内层 (inner/build_trainer.py - task stage)
  ├─ 消费 ResourceContext
  ├─ 参数拟合（task parameters）
  └─ 产出 task artifact
```

关键点：

1. **ResourceContext 不走 artifact**，只走 evaluate() 函数参数
2. **内层不自己分配资源**，被动读取外层授权
3. **每个 stage 的 inner 都独立构造器**（但消费相同的 ResourceContext）

## 12. 代码示例：外层 Stage 1

```python
# outer/problem/stage1_basis_search.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../inner'))

from nsgablack.core.problem import BlackBoxProblem
from nsgablack.core.types import Feedback
from build_trainer import build_trainer

class BasisSearchProblem(BlackBoxProblem):
    """
    外层搜索 basis 结构，内层拟合参数
    """
    
    def __init__(self, data, resource_context):
        self.data = data
        self.resource_context = resource_context
    
    def evaluate(self, x):
        """
        x: basis index candidate (encoded as numpy array)
        """
        try:
            # ① 解码外层向量为 basis spec
            basis_spec = self.decode_basis_spec(x)
            
            # ② 构造内层 trainer（关键：传 resource_context）
            trainer = build_trainer(
                basis_spec=basis_spec,
                data=self.data,
                resource_context=self.resource_context,  # ← 授权资源
                budget=200,
            )
            
            # ③ 内层训练（参数拟合）
            result = trainer.fit()
            
            # ④ 提取外层多目标
            return Feedback(
                objectives=[
                    result.fit_error,              # 拟合误差
                    result.orthogonality_error,    # 冗余度
                    basis_spec.complexity,         # 复杂度
                ],
                constraints=[
                    result.condition_number - 1e4,  # 数值稳定性
                ],
                artifacts={
                    "basis_artifact": result.basis_artifact_ref,
                },
            )
        
        except Exception as e:
            return Feedback(
                objectives=[1e10, 1e10, 1e10],
                constraints=[1e10],
            )
    
    def decode_basis_spec(self, x):
        """外层向量 → basis 规范"""
        # 例：选择函数族、多项式阶数、基数等
        return {
            "families": ["polynomial", "sin", "cos"],
            "max_polynomial_degree": int(x[0]) + 1,
            "basis_size": int(x[1]) + 5,
            "orthogonalization": "gram_schmidt",
        }
```

## 13. 代码示例：内层 Trainer（Basis Stage）

```python
# inner/build_trainer.py

def build_trainer(
    basis_spec,
    data,
    resource_context,  # ← 外层授权
    budget,
):
    """
    构造符号基函数拟合 trainer
    """
    
    from mlblack.representations import SymbolicBasisRepresentation
    from mlblack.problems import SymbolicBasisFittingProblem
    from mlblack.adapters import GradientDescentAdapter
    from mlblack.core.trainer import ComposableTrainer
    from mlblack.core import ComputeBackendSpec
    
    # ① 创建 problem（消费数据）
    problem = SymbolicBasisFittingProblem(data)
    
    # ② 创建 representation（basis codec）
    representation = SymbolicBasisRepresentation(basis_spec)
    
    # ③ 创建 adapter（参数优化）
    adapter = GradientDescentAdapter(
        learning_rate=0.01,
        batch_size=32,
    )
    
    # ④ 创建计算后端（关键：从 resource_context 读取授权）
    if resource_context:
        backend_name = resource_context.get("compute_backend", "numpy").split("_")[0]
        device = resource_context.get("device", "cpu")
        threads = resource_context.get("threads", 1)
    else:
        backend_name, device, threads = "numpy", "cpu", 1
    
    compute_backend = ComputeBackendSpec(
        name=backend_name,
        device=device,
        threads=threads,
    )
    
    # ⑤ 组装 trainer（不私自分配资源）
    trainer = ComposableTrainer(
        problem=problem,
        representation=representation,
        adapter=adapter,
        compute_backend=compute_backend,
    )
    
    # ⑥ 注入完整资源上下文供审计
    if resource_context:
        trainer.set_resource_context(resource_context)
    
    trainer.set_max_steps(budget)
    
    return trainer
```

## 14. 扩展方向

| 想增强 | 归属 |
| --- | --- |
| 更多 primitive grammar | `mlblack.pipeline.symbolic` |
| 更强 CAS canonicalization | `mlblack.models.symbolic_normalization` |
| 更多 recovery scoring | symbolic artifact/audit |
| 更多 outer structure search | nsgablack outer problem/representation |
| 并行 branch runtime | nsgablack L0 |
| 多阶段锁核精修 | nsgablack serial/stage + mlblack inner fitting |
