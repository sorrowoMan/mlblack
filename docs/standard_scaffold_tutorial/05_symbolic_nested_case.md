# 05. Symbolic Nested Case 标准读法

这一章用符号学习说明标准跨框架结构：外层搜索结构，内层拟合参数。它不是 `mlblack` 自建 symbolic workflow。

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

## 11. 扩展方向

| 想增强 | 归属 |
| --- | --- |
| 更多 primitive grammar | `mlblack.pipeline.symbolic` |
| 更强 CAS canonicalization | `mlblack.models.symbolic_normalization` |
| 更多 recovery scoring | symbolic artifact/audit |
| 更多 outer structure search | nsgablack outer problem/representation |
| 并行 branch runtime | nsgablack L0 |
| 多阶段锁核精修 | nsgablack serial/stage + mlblack inner fitting |
