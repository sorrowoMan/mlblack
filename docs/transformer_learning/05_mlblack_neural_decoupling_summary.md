# mlblack 神经网络解耦总结（含 Transformer）

## 1. 结论

是的，`mlblack` 已经完成了关键方向的解耦：

- 神经网络的语义定义（representation/codec/head/problem）与执行后端（torch/jax/tf）分离。
- Transformer 已进入统一神经图路线，不再是框架绑定的单脚本实现。
- 外层编排与内层训练边界清晰：`nsgablack` 负责编排与资源，`mlblack` 负责模型语义与拟合。

一句话：

```text
同一模型语义可以在不同执行后端落地；同一后端可以承载不同模型语义。
```

---

## 2. 当前已达成的解耦面

### 2.1 语义层与执行层解耦

主链路已不是“直接写某框架模型”，而是：

```text
NeuralGraphSpec / typed representation
  -> codec / decoder
  -> executable model
  -> head / problem evaluation
  -> adapter / optimizer feedback
```

含义：

- 模型结构语义不绑定具体后端实现。
- 后端只承诺执行能力，不定义模型业务语义。
- 同一语义结构可路由到不同 backend（在 capability 允许时）。

### 2.2 编排层与训练层解耦

- 外层：`nsgablack`（stage/group/serial/parallel/resource/budget）。
- 内层：`mlblack`（模型构建、数据评估、参数拟合、artifact）。

这使复杂流程（多阶段、多模型组合、嵌套优化）可维护。

### 2.3 能力插件与算法语义解耦

checkpoint、trace、report、backend probe 等能力不再混入模型语义代码。  
同一模型语义可在不同工程能力组合下复用。

---

## 3. Transformer 解耦状态

Transformer 当前可被视为一条声明式神经图路线，而非硬编码训练脚本：

1. Transformer 是 `NeuralGraph route/spec`，不是单一类实现。
2. token/embedding/attention/ffn/head 进入统一 codec 组合面。
3. loss/metric/constraint 仍经 problem 接口回流。
4. backend 只负责执行，不重定义 Transformer 语义。

因此：

```text
Transformer = 可编排模型语义
而不是
固定框架脚本。
```

---

## 4. 直接收益

- MLP/CNN/GNN/Transformer 可在同一语义框架对比。
- 多模型组合（主线+残差、多头路由、多阶段）可通过编排完成。
- 可在不改模型语义前提下切换 backend 与资源策略。
- artifact 与 benchmark 口径更统一，便于审计与复现。

---

## 5. 需要补强的 4 项（可执行版本）

下面不是方向口号，而是可落地规范。

### 5.1 Capability Contract 细化

**目标**  
把“能不能跑”从隐式经验变成显式契约与可审计报错。

**最小契约字段（建议）**

```text
backend_id
supports_forward
supports_backward
supports_higher_order_grad
supports_dynamic_shape
supports_mixed_precision
supports_distributed
dtype_matrix
device_matrix
failure_modes
```

**报错边界要求**

- 若 route 需要 `supports_backward=true`，但 backend 不支持，立即抛能力错误，不走隐式降级。
- 若配置了 mixed precision，但 backend/dtype 不满足，报错需包含“不支持项 + 当前配置 + 推荐替代”。
- 分布式能力不满足时，禁止 silent fallback 到单机且不提示。

**验收标准**

1. 生成统一 capability matrix 文档（backend x capability）。
2. 每个不支持路径都有确定错误码和错误文案。
3. 最少 1 套 contract 单元测试覆盖 capability gating。

---

### 5.2 FunctionalBackpropAdapter 完整化

**目标**  
统一梯度协议，消除 route/backend 特例分支。

**统一接口（建议）**

```text
forward(params, batch, context) -> outputs
loss_fn(outputs, batch, context) -> scalar loss
grad_fn(params, batch, context) -> grads (+ aux)
apply_update(params, grads, state) -> new_params, new_state
```

**约束**

- route 不允许直接依赖 `model.parameter_gradient` 之类特例入口。
- grads 的 shape/dtype/none-handling 必须协议化。
- 训练态和评估态切换必须在 adapter 层显式体现。

**验收标准**

1. torch/jax/tf 路线都通过同一 adapter 协议测试。
2. 删除或封存现有特例梯度分支。
3. 产出一份“梯度协议一致性”测试报告。

---

### 5.3 Benchmark 套件化

**目标**  
从 smoke 升级为可重复、可比较、可回归的基准系统。

**基准维度（建议）**

```text
task family: regression/classification/retrieval/symbolic
model family: mlp/cnn/gnn/transformer
backend: torch/jax/tf
resource profile: cpu/gpu/single/distributed-ready
metrics: quality + latency + memory + stability
```

**运行规范**

- 固定 seeds、数据切分、预算与资源上下文。
- 每项基准至少输出：mean/std、p50/p95 时延、峰值内存、失败率。
- 结果必须可落盘并可回放。

**验收标准**

1. 至少 1 套 nightly benchmark suite 可自动运行。
2. 任意 PR 可对比基线并给出回归判断。
3. benchmark artifact 可被 dashboard 直接消费。

---

### 5.4 资源上下文贯通

**目标**  
确保外层编排注入的资源约束真实传递到内层训练执行面。

**最小资源上下文（建议）**

```text
resource_namespace
worker_id
lease_id
device_tokens
threads
memory_limit_mb
runtime_backend
artifact_backend
```

**约束**

- 内层 trainer 不得越权申请未授予设备。
- 内层并发预算不得突破外层 grant。
- 所有训练产物要带 resource provenance（来源上下文）。

**验收标准**

1. runtime summary 显示 worker/lease/resource_context/artifact refs。
2. nested 场景可审计“外层授予 -> 内层消费”的完整链路。
3. 越权资源使用可稳定复现并被拦截。

---

## 6. 下一步落地顺序（建议）

1. 先做 capability contract 与错误边界（否则后续测试不稳定）。
2. 再收敛 FunctionalBackpropAdapter 协议。
3. 同步建设 benchmark suite。
4. 最后把资源上下文贯通到所有关键 case。

这样能先把“可运行”变成“可判定”，再把“可判定”变成“可持续演进”。

---

## 7. 对外表达模板

```text
mlblack has decoupled neural modeling semantics from execution backends.
Transformer is treated as a declarative neural-graph route under the same representation/codec/problem/adapter scaffold, rather than a framework-specific hardcoded training script.
Current focus is hardening capability contracts, unifying functional backprop protocol, suite-level benchmarking, and end-to-end resource-context propagation.
```

