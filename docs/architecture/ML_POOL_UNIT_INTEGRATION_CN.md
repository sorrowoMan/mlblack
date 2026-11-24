# ML 候选、解池与 Unit：统一装配的边界

本文把 `mlblack` 的模型语义放回当前 `blackbase + nsgablack + mlblack` 框架栈，回答三个问题：模型 Codec 与 Unit 为什么相似；ML 能否使用解池和搜索编排；装配 GA、Adam 等方法时，哪些结构共享，哪些状态必须独立。

本文中的代码片段只使用现有公共入口。GA → Adam 一节用于说明装配条件与交接边界，不将某个尚未列出运行证据的组合视为已经验收的完整 Project。

## 1. 从一个候选看全局

以线性模型为例，候选可以是：

```text
θ = [intercept, weight_0, weight_1]
```

同一个候选有两类不同的结构。

**模型结构**说明这几个数代表什么。`LinearPointCodec` 把第一项解释为截距，其余项解释为权重，并构造能预测的 `LinearPointModel`。神经网络的 Codec 还要知道层、张量形状和参数布局。`Head` 决定模型输出怎样解释，例如点预测、区间或概率。

**搜索结构**说明某个算法怎样围绕候选工作。GA 可以把一百组 θ 放进种群，维护成员、反馈与选择关系；Adam 可以围绕其中一组 θ 维护当前参数和一、二阶矩。它们的搜索 Unit 不同，θ 的模型含义仍可以相同。

```text
                     同一个候选 θ
                    /           \
        模型结构：如何解释          搜索结构：如何搜索
        Codec / Head              所属 Adapter 的 Unit
        参数 → 模型/输出           种群、轨迹、moment、邻域关系
                    \           /
                 Problem / Provider 求值
```

解池保存候选版本及引用。它不会因为一个候选被两个算法使用，就把两份可写搜索状态合在一起。

## 2. Codec 与 Unit 的相似处到底是什么

用户把它们联系起来是有依据的。当前 Codec 不只做字节转码：

- `LinearPointCodec` 提供参数初始化、参数布局和模型构造。
- `NeuralGraphCodec` 提供图规格、具名张量布局、布局缓存及 backend lowering 入口。
- 某些 Codec 保留从数据拟合的解释信息，例如正交特征映射。

这些能力与 Unit 的结构构建、结构校验、派生特征确实相近。共同原则可以复用：结构必须有明确声明，依赖候选版本的派生值不能过期，持久化应使用显式数据协议。

不过，现有对象不能直接互换：

- 模型 Codec 可以供多个搜索算法使用；算法 Unit 的运行实例由一个具体 Adapter 拥有。
- Codec 的参数布局可能随模型规格长期存在；种群成员或 Adam moment 随算法运行变化。
- 当前部分 Codec 自持初始化 RNG；`UnitBinding` 的 builder 要求无状态，随机选择和身份由 Adapter 提供。
- 模型 `decode()` 构造模型语义；`UnitBinding.dump()/restore()` 保存、恢复搜索结构。两者都叫 codec 时，应明确是哪一种。

因此，复用布局、校验和显式状态协议是自然的；共享一个可写运行对象则需要逐项证明所有权和生命周期一致。无需为了名称统一，再增加一套与现有对象并行的候选结构。

相关实现：

- [`core/representation.py`](../../core/representation.py)
- [`representations/linear_point.py`](../../representations/linear_point.py)
- [`representations/codecs/linear.py`](../../representations/codecs/linear.py)
- [`representations/codecs/neural/base.py`](../../representations/codecs/neural/base.py)
- nsgablack：`units/contracts.py`、`units/collections.py`、`adapters/unit_backed.py`

## 3. 一次 ML 搜索怎样走过这些层

ML 使用同一个候选求解控制面：

```text
Problem / ModelRepresentation / Provider 装配
  → 建立运行的 SolutionPool
  → 问题管线初始化候选，或读取已声明的池输入
  → Adapter 按需建立自己的 Unit
  → Adapter 提出新候选
  → 问题 repair / 显式联合修复
  → ModelRepresentation / Codec 解释模型
  → LearningProblem / Provider 计算原始反馈
  → Bias 生成选择视图
  → 接纳、Adapter 更新、Unit 更新
  → 原步骤事务提交候选与池引用
  → 提交后的 ML 结果投影
```

这是一轮运行的职责关系；不能把它当成所有组件各执行一次的固定线性流水线。例如 Codec 的模型构造可以由 Provider 的 lowering 路径执行，Provider transition 也可以保留设备上的活状态。

`LearningSolver.fit()` 使用 `nsgablack.ComposableSolver` 的运行循环。ML 层通过 `SolverSemanticExtension` 接入投影，通过 `FinalizationHooks` 接入结果发布。ML 不再维护第二套训练步骤事务。

一个独立 ML 方法仍可从现有 preset 开始：

```python
from mlblack.presets import build_linear_point_trainer

# data 使用该 preset 已支持的数据输入。
trainer = build_linear_point_trainer(
    data,
    method="gradient.adam",
    learning_rate=0.05,
)
result = trainer.fit(max_steps=100)
```

显式装配时，`build_learning_solver(problem=..., representation=..., adapter=...)` 接收同一层级的组件。`adapter` 可以是正式的搜索组合；ML 不应为组合再创建私有循环。

## 4. Pool、Unit 和 ctx 分别存什么

### 4.1 Pool：候选在哪，哪一版候选

`SolutionPool` 管理运行内的候选版本、来源、引用集合及投影视图。候选通过 `SolutionRef` 保留 token 与版本；数值相等不代表同一个候选身份。

子池是同一池中的具名引用集合，不复制完整模型或参数数组。一个精英池可以引用已提交的模型候选，一批精修任务也可以读取这些引用。退出子池解除的是成员关系；其他 Unit、任务或历史仍可能保留该候选。

对 ML 而言，入池前必须明确候选是什么：

- 搜索线性参数时，是参数加必要的语义 metadata。
- 搜索网络规格时，是可解释的网络规格候选。
- 搜索 estimator 配置时，是配置候选；训练后的 estimator 是该配置求值的产物，不能悄悄冒充原配置。

### 4.2 Unit：这个算法怎样使用候选

`PopulationUnit` 维护算法的成员和对齐反馈。`ParameterUnit` 维护当前参数及 optimizer moment 等必要状态。Unit 只在所属 Adapter 真正需要它时创建；装配了一个后期才启动的 Adam，不要求运行开始就创建它的 Unit。

同一个 θ 可以被 GA 和 Adam 引用，但 GA 的排名不会成为 Adam 的可写状态。Adam 把 θ 改成新参数 θ′ 时产生新提案，不能原位覆写池里 GA 仍持有的 θ。

### 4.3 ctx：这次调用发生在什么条件下

ctx 表达步骤、任务、阶段、资源引用和本次求值的轻量信息。它可以告诉算法“当前在精修阶段”，但不应成为存放完整模型、整批参数、全部历史或活 Tensor 的另一处权威。

训练数据和模型大对象由各自的语义组件、Snapshot 或 Artifact 管理；进入共享 context 的自定义键仍需注册。把数据塞进 ctx 不会自动得到 Pool 的身份、事务或恢复语义。

## 5. GA → Adam：怎样自然装配，怎样正确交接

先规定一个明确场景：GA 搜索固定布局的模型参数，选择部分已提交成员，再由 Adam 精修这些参数，最后把精修后的候选交回 GA 竞争。

```text
GA 搜索并提交成员
  → 选择精修来源的精确 SolutionRef
  → Adam 为每个起点建立自己的参数轨迹
  → 重新取得当前问题的目标与梯度
  → 提出并评估新参数
  → 交付本次任务最终选择的候选引用
  → GA 按自己的规则竞争替换
```

这个流程不需要把 GA 的 `PopulationUnit` 转成 Adam 的 `ParameterUnit`。交接的是候选；Adam 从候选建立自己需要的结构。

装配时必须明确以下条件。

1. **候选解释一致。** 两者使用的 Codec、模型规格及参数布局必须兼容。把网络结构搜索的向量直接当成固定网络的权重是不合法的交接。
2. **精修来源明确。** 选择 GA 最终成员、精英集合或指定视图，保留准确版本；不能从全部历史接纳记录中猜当前成员。
3. **梯度语义明确。** Adam 需要当前任务目标在候选坐标系中的梯度。已有 raw feedback 不保证包含接收方所需的梯度，也不自动替代重新求值。
4. **任务结果明确。** 交回的是精修后实际选择的候选 θ′ 及其证据；不能保留 θ 的身份却给它写入 θ′ 的分数。
5. **恢复闭包完整。** 池、子任务进度、每个算法的 Unit 及必要的 Provider 恢复值必须一起保存。只保存参数数组不能恢复一条已有 moment 的 Adam 轨迹。

当前可复用的外层机制是 `SearchPlanAdapter`、`StrategySpec`、`PoolTaskSpec`、原 Router 与步骤事务。只有执行顺序的 `serial(stage(...), stage(...))` 不自动建立候选交接；改变起点需要显式绑定来源和受支持的池任务模式。反复激活一个算法也不等于重放 setup。

`GradientOptimizerAdapter` 当前声明池输入模式为 `initialize`，并支持搜索重启。现有 `PoolTaskAdapter` 把 `refine` 请求映射为叶算法的 `initialize` 输入，并检查重启能力，因此可以复用这条机制逐起点精修。它不意味着 Gradient Adapter 还支持 population 竞争或任意投影；具体模式仍由正式任务包装检查。

特别注意：当前 Gradient Adapter 声明 `supports_objective_projection=False`。MOEA/D 等子视图只改变目标值时，原梯度不会自动变成该投影的梯度。要支持这种精修，必须同时定义梯度的变换与验证，不能仅以标量分数可读取就宣称支持。

外层自然编排、精修结果交付与限制以 nsgablack 的 `docs/architecture/NATURAL_SEARCH_PLANS_CN.md`、`adapters/pool_tasks.py` 为准。本文不新增 `pool.run(adam)`、`pool.to_model_unit()` 一类尚未存在的快捷 API。

## 6. Provider 活状态怎样与池共存

模型可能很大，Provider 可以把参数、梯度和 optimizer slot 保留在设备侧，通过 `StateRef` 寻址。这里有三个必须区分的东西：

- `SolutionRef` 指向运行解池中的候选 token 与版本。
- `StateRef` 指向 Provider 声明的可寻址范围内的活状态。
- Artifact 引用指向持久化的模型或报告产物。

池不能靠复制一个 `StateRef` 获得可跨进程恢复的模型，也不能把 backend session 放入 Unit checkpoint。正式 checkpoint 需要可保存的候选值和必要 optimizer 状态；恢复到新进程后，再经正式 Provider surface 建立新的活状态。

当前梯度路径通过 copy-on-write slot transition 保留前驱：失败或拒绝时清理本次后继，成功提交后才能回收不再需要的前驱。Adapter 拥有搜索决策与精确清理责任，Provider 拥有实际设备对象；清理失败必须保留可重试信息。

这也解释了为什么“把所有参数和模型都放进一个共享 Unit”会带来额外问题：设备活对象、可传输候选和持久产物有不同的有效范围。

详见 [统一梯度路径](UNIFIED_GRADIENT_PATH.md)。

## 7. repair、部分接纳和恢复不能跳过

模型 repair 负责模型候选合法性。若算法 Unit 还有结构不变量，使用显式共同修复策略协调，最终版本须同时满足声明的检查。Codec 的模型构造不能顺便隐式重写候选，而不给提案所有者对账。

评估前同一提案被 repair，使用 `ProposalRepair`；评估后预算或准入筛选，使用 `BatchDisposition`。它们保护不同的对应关系。ML metadata、token、参数行、梯度与 Unit pending 状态都要沿原提案索引对齐。

步骤失败可以恢复搜索状态，但已发生的求值和预算消耗仍然存在。恢复不得为了重建一个最佳模型投影，隐式重跑会训练、写库或启动内层 Case 的求值准备过程。

ML facade 的 `get_state()/set_state()` 与正式 checkpoint 都应覆盖实际恢复所需的解池闭包；带有精确池引用的 Adapter state 不能独立于对应池状态恢复。本文的职责说明不替代这两条路径各自的行为验收。

正式 checkpoint 通过必需的 `ml_projection` 组件保存最佳反馈、其 candidate token / evaluation 绑定、最佳模型 Artifact 引用与其他结果引用。该组件与 facade 使用同一组字段导出和恢复逻辑；`history` 继续由共享 checkpoint 的 Solver state 保存，不在组件里重复保存。恢复顺序为 Pool、通用 incumbent、ML 投影组件；组件仅恢复已验证的数据并清空进程内模型 handle，不初始化候选、不调用求值准备或训练。随后需要模型时，走 Problem 显式的无求值恢复规则。

旧正式 checkpoint 若缺少必需的 `ml_projection` 状态，会明确拒绝，不能从最后一个事件猜测较早的最佳反馈。若需要迁移，应从旧版本生成完整可验证的数据，或显式重新开始。

详见 [ML 联合修复](JOINT_REPAIR_CN.md)。

## 8. 同一 Case 和内外层 Case 的分界

GA 与 Adam 若搜索同一种模型参数、同一数据语义和同一 Problem，可以在同一 Case 内组合，共用候选解池和问题解释；算法各自拥有 Unit。

若外层搜索超参数，内层真正训练模型，则二者通常不是同一个候选问题：外层候选是配置，内层候选是模型参数。此时使用正式的内层 Case 请求、结果与 Artifact，不把两个独立运行的池直接别名成一个池。共享的是 Project / Case / L0 substrate 与明确交付协议。

同样，跨 Case 传递一个模型 Artifact，不表示目标 Case 可以直接继续旧 Provider 的 `StateRef` 或旧运行的 Unit。继续训练所需的参数、布局、数据语义与 optimizer 恢复策略都要显式匹配。

## 9. 实施时按什么顺序闭环

优先让现有对象共同工作，不按术语数量扩张 API。

1. 装配时读取已声明的维度、布局和能力，不通过提前生成候选或创建 Unit 猜接口。
2. 运行时在共享池与生命周期建立后初始化候选，Adapter 首次需要时才建 Unit。
3. 统一保留候选身份、raw feedback、池引用与 ML 结果投影的对应关系。
4. 分别验证普通候选、修复、部分接纳、失败回滚及正式恢复；验证项目应明确具体算法和 Provider，不能从单组件测试推断所有组合。
5. 用一个固定模型布局的黑盒搜索 → 梯度精修 Case 验证真实交接，再讨论需要额外语义的结构搜索或投影梯度。

实际验证结果应记录命令、范围和剩余限制。源码支持某个单独 API，不代表任意 Codec、Provider、Adapter 的排列都已获得组合保证。

## 10. 本轮正式 checkpoint 针对性验收

执行：

```powershell
python -m pytest tests/test_ml_projection_checkpoint.py -q
```

结果：4 项通过。用真实 `CheckpointResumePlugin` 写入并重新读取签名 checkpoint，源运行第一步找到最佳候选、第二步求值得到更差候选，确保最佳候选不在最后一次事件中。恢复后验证最佳反馈、绑定、历史、Artifact 引用与模型结果；候选初始化、求值准备和 Problem 求值均未重跑。另覆盖错误 token / evaluation 绑定拒绝，以及 facade 与正式组件字段一致。

另已运行 `tests/test_learning_pool_integration.py` 的固定模型参数组合：NSGA-II 搜索后发布 population 视图，Gradient 用普通池任务逐成员精修。连续执行和在 GA 阶段后恢复两种路径，均检查全部成员、精确版本、新 token、来源链及未改写的 raw feedback。该测试使用 SGD 参数更新；不将结果推广为任意优化器或 Provider 的组合保证。

普通 `build_context()` 只保留可传输的后端描述。`enrich_ml_context()` 在 Representation / Problem 的局部调用处补入活 backend session；Router 的任务上下文和步骤快照不复制会话。默认 GA 的数值算子仍接收数值草稿，模型 metadata 与父候选精确引用由控制面保存。模型含义不会因为一次默认交叉而丢失。

维度已声明的 Representation 在装配时无需 `init()`；依赖后端的布局可在 Representation 的 `setup()` 确定维度。旧自定义 Representation 若配合不提前使用几何的算法，可在首次运行初始化时绑定维度；直接提出候选或在 setup 中使用几何的 Adapter，必须在 Adapter.setup 前得到模型维度。例如 BO 的初始观测点宽度校验需要这个声明，不能靠试采一个候选猜维度。正式 checkpoint 同时保存已绑定维度，恢复自定义延迟布局时不重采初值。

ML 门面状态已升为 v7，携带 Pool 的可传输值闭包。v3–v6 状态只有同时带有完整 Pool 内容时才能显式迁移；旧状态缺少候选值与引用闭包时会拒绝，不能从参数数值猜造身份。具有拟合副作用的 Problem 若没有只读模型恢复规则，同样明确拒绝；当前没有通用的任意第三方已拟合模型读取服务。

Doctor 通过 `model-restore-contract-invalid` 检查恢复 hook 的可调用签名，不构造 Problem 或模型，也不声称能证明任意用户函数没有副作用。

运行预算仍遵循现有入口：预先 `set_state()` 的 ML fit 能根据已知恢复游标折算本次步数；`CheckpointResumePlugin(auto_resume=True)` 在 setup 才发现 checkpoint，本次验收给它总逻辑步数上限，例如已提交 2 步后继续 1 步使用 `fit(max_steps=3)`。该入口差异本轮未改动，不应把两种恢复方式的步数参数直接互换。

## 11. 相关回归与检查范围

- ML 初始化、Pool 引用、连续/断点 GA→梯度精修、只读模型重建、正式签名 checkpoint 和恢复签名检查：三个新增测试文件，22 项。
- ML 生命周期、Provider 梯度、联合修复、脚手架等已有相关回归分批通过；固定线性模型的结果 Artifact 回归也已执行。
- 优化侧种群引用 63 项、成员操作 58 项、成员事务与 Pool 创建/访问 39 项，覆盖数值算子、来源、部分接纳、回滚和 Provider COW。
- 三级嵌套与完整拓扑 Project：31 项通过，保留独立内外池及原始反馈。
- mlblack 严格 Doctor：0 error、0 warning；覆盖 13 个 Project、30 个 Case 的结构及组件声明。
- nsgablack 严格 Doctor，开启 `--catalog-audit --profile framework-core`：0 error、0 warning；582 项检查完成，106 项跳过，其中 78 项 Case build 未开启、27 项 Case registry 和 1 项框架 registry 按作用域跳过。该报告不能代替每个 Case 的实际装配执行；本轮 GA→梯度和三级嵌套由行为测试实际运行。


## 2026-11-16：当前控制面与 Artifact 生命周期验证

本轮针对当前 nsgablack 工作树执行 LearningPool、ML projection checkpoint、单控制面与 Artifact 测试。
mlblack 的 pytest pythonpath 默认可加载相邻桌面版 nsgablack；跨工作树验证必须指定目标配置
（`python -m pytest -c <目标 nsgablack>/pyproject.toml <mlblack测试文件>`），并明确检查实际加载路径。
不能把另一份源码的测试结果当作当前改动验收。

修正模型 Artifact 本地语义调用：普通 build_context 不携带 backend.session，
materialize_model_artifact 在本地使用 enrich_ml_context，不把活 session 写回共享 context。
最终派生模型描述在 finalize_run_result、后端仍存活时准备，保存进程内的 (model, artifact) 临时值；
真正序列化/发布仍在成功 teardown 之后由原 Artifact finalization transaction 完成。
发布阶段复用已准备的描述，禁止为补参数布局重新解析后端；若 best_model 已换成另一对象则拒绝，
不隐式重建描述。运行开始清空临时值，不把它放进 checkpoint 或 context。

该身份检查不证明任意外部代码没有原位修改模型；finish 阶段观察者仍须遵守只读契约。
回归用真实神经 Case runtime 在 backend.close 后禁止 ensure，验证最终发布仍成功。

本轮针对性测试：池/投影/控制/Artifact组20项及最终发布组20项通过（有重叠），ML Doctor完成且保留1条SFT示例管线入口建议。并非ML全部可选后端验收。
