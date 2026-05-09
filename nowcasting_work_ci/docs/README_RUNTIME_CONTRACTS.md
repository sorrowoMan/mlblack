# Runtime 契约与复现性执行稿

这份文档不讨论“还要不要继续拆文件”，只回答四件事：

1. 运行时每一层到底负责什么
2. 每个 stage 的输入输出和 `context` 键是什么
3. 哪些地方禁止直接做 I/O / 资源清理
4. 复现性和公共接口现在如何被代码化

## 1. 设计目标

当前 `nowcasting_work_ci` 已经不再是一个单文件脚本问题，真正的风险也不再是“文件太长”，而是下面这几类问题：

- stage 之间靠字符串约定传值，字段一变会联动多层
- 随机种子只在外层设了，内层 torch 拟合仍会漂
- 报表、缓存、图、资源关闭混在 runtime/problem/evaluation 逻辑里
- `summary`/`payload` 没有固定 schema，后面很容易再牵动多层

这一轮的目标就是把这些边界写死。

## 2. 层级契约

### 2.1 控制平面

目录：

- `workflow/orchestrator.py`
- `workflow/hook_bus.py`
- `nowcasting_work_ci/mlblack_side/runtime/stages.py`

职责：

- 只负责 stage 顺序、生命周期、hook 触发
- 只负责维护轻量 context
- 不负责模型训练细节
- 不负责报表写盘
- 不负责图缓存清理

禁止：

- 直接训练模型
- 直接写 `summary.json`
- 直接关闭 `graph_cache`
- 直接决定目标函数策略

### 2.2 runtime 装配层

目录：

- `runtime/config.py`
- `runtime/assembly.py`
- `runtime/build_runtime.py`
- `runtime/actions/*`

职责：

- 解析 CLI
- 装配数据、特征空间、outer search 所需运行时对象
- 调用 problem/evaluation/model
- 组装最终结果对象

禁止：

- 直接写报表
- 直接写图
- 直接关闭资源
- 在 stage 里散落 ad-hoc 文件 I/O

### 2.3 problem / evaluation / model 实现层

目录：

- `nowcasting_work_ci/mlblack_side/problem/*`
- `evaluation/*`
- `model/*`

职责：

- `problem`：决策解码、problem 合约、委托评估
- `evaluation`：fold 评估、区间构造、汇总策略
- `model`：具体拟合与预测实现

禁止：

- 直接触碰 runtime context
- 直接写报表或落盘
- 直接清理资源
- 依赖某个 stage 名字来运行

### 2.4 plugin / hook 副作用层

目录：

- `plugins/report_writer_plugin.py`
- `plugins/runtime_resource_plugin.py`
- `plugins/reproducibility_plugin.py`

职责：

- 报表写盘
- 运行资源清理
- 进程级随机态初始化

允许：

- 读 `context`
- 写文件
- 输出运行摘要
- 关闭 runtime 资源

禁止：

- 改 outer search 策略
- 改 problem 解码
- 改目标值语义

## 3. Stage 输入输出契约

代码源：

- `nowcasting_work_ci/mlblack_side/runtime/contracts.py`

### 3.1 `parse_cli`

输入：

- `argv`

输出：

- `args`
- `runtime_seed`

说明：

- 只把 CLI 解析成 `RuntimeCliConfig`
- 不读数据
- 不创建目录

### 3.2 `build_runtime`

输入：

- `args`

输出：

- `prepared`
- `out_root`
- `graph_cache_resource`

说明：

- 只做运行时装配
- 返回可供后续阶段消费的内存对象
- 不写 summary

### 3.3 `outer_search`

输入：

- `args`
- `prepared`

输出：

- `search`

说明：

- 只做外层搜索
- 搜索缓存留在内存 / graph cache 中
- 不做最终报表 I/O

### 3.4 `evaluate_final`

输入：

- `args`
- `prepared`
- `search`

输出：

- `comparison`

说明：

- 只做最终 symbolic vs xgboost 对照评估
- 不写文件

### 3.5 `assemble_result`

输入：

- `args`
- `prepared`
- `search`
- `comparison`

输出：

- `final_result`

说明：

- 只组装结果对象和 `report_payload`
- 不直接写 `summary.json`
- 最终写盘交给 `ReportWriterPlugin`

## 4. Context 键清单

代码源：

- `RuntimeContextKey`

当前固定键：

- `argv`
- `runtime_seed`
- `reproducibility`
- `started_at`
- `finished_at`
- `failed_at`
- `duration_sec`
- `stage_results`
- `args`
- `prepared`
- `search`
- `comparison`
- `final_result`
- `out_root`
- `graph_cache_resource`
- `summary_path`
- `failed_stage`
- `last_stage`
- `last_stage_result`

约束：

- runtime/actions 只能通过 `ctx_get / ctx_require / ctx_set` 访问关键键
- 不再鼓励到处手写 `"search"`、`"comparison"` 这种裸字符串

## 5. 公共 payload 契约

### 5.1 `SummaryReportPayload`

代码源：

- `nowcasting_work_ci/mlblack_side/runtime/contracts.py`

固定字段：

- `report`
- `out_root`
- `graph_cache_snapshot`
- `sym_rmse`
- `xgb_rmse`
- `sym_interval`
- `xgb_interval`
- `interval_alpha`

意义：

- `assemble_result` 只负责产出 payload
- `ReportWriterPlugin` 只消费这个 payload
- runtime 不再把活的 `graph_cache` 实例塞进报表 payload

这件事很关键，因为它把“内存资源”和“报表数据”彻底分开了。

### 5.2 `SearchStageResult / ComparisonStageResult / FinalStageResult`

当前 stage 之间不再只传松散 dict，而是传 runtime payload 对象：

- `SearchStageResult`
- `ComparisonStageResult`
- `FinalStageResult`

它们都实现了 `Mapping` 兼容接口：

- 老代码仍然可以 `result["result_summary"]`
- 新代码可以直接用属性访问，如 `comparison.sym_rmse`

这样做的目的不是“换一种写法”，而是把三段核心 stage 接口固定下来：

- `outer_search` 负责输出结构搜索结果对象
- `evaluate_final` 负责输出最终比较结果对象
- `assemble_result` 负责输出最终交付对象

后面如果字段变化，必须优先改 payload schema，而不是在 action 里偷偷加新键。

## 6. 副作用边界

### 6.1 允许直接 I/O 的层

只有 `plugins/*`。

当前已经明确插件化的副作用：

- `ReportWriterPlugin`：写 `summary.json` 并输出摘要日志
- `RuntimeResourcePlugin`：关闭 `graph_cache`
- `ReproducibilityPlugin`：初始化进程级随机态

### 6.2 明确禁止直接 I/O 的层

- `runtime/config.py`
- `runtime/assembly.py`
- `runtime/build_runtime.py`
- `runtime/stages.py`
- `runtime/actions/*`
- `problem/*`
- `evaluation/*`
- `model/*`

这意味着以后如果要补：

- 图表落盘
- docx 报告
- sqlite 运行日志
- checkpoint 导出

都应该优先以 plugin/hook 方式进入，而不是再塞回 problem/runtime。

## 7. 复现性路径

### 7.1 进程级复现性

由 `ReproducibilityPlugin` 在 `on_experiment_start` 统一负责：

- `random.seed`
- `numpy.random.seed`
- `torch.manual_seed`
- `torch.cuda.manual_seed_all`
- `torch` deterministic 配置

### 7.2 内层拟合复现性

路径：

- `runtime seed`
- `ProblemConfig.random_seed`
- `FitPredictCallbackConfig.random_seed`
- `model/interval_fit._three_layer_fit_predict(...)`

当前策略：

- 外层 experiment 启动时固定全局随机态
- 内层 symbolic torch 拟合时，根据 `base_seed + genome_signature` 派生局部 `fit_seed`
- 局部只重置 torch 初始化，不重置外层 `numpy/random`

原因：

- 这样可以稳定 smoke 指标
- 同时避免每次候选评估都把外层求解器随机流重置掉

## 8. 当前最重要的稳定接口

### 8.1 runtime 主入口

- `nowcasting_work_ci/run.py`
- `nowcasting_work_ci/mlblack_side/runtime/workflow.py::main`

### 8.2 stage builder

- `build_experiment_stages(...)`

### 8.3 problem 评估接口

- `ProblemEvaluationCallbacks.fit_predict(...)`
- `ProblemEvaluationCallbacks.build_interval_bounds(...)`
- `ProblemEvaluationCallbacks.summarize_fold(...)`

### 8.4 report 交接接口

- `SummaryReportPayload`

## 9. 当前这版相对之前真正改善了什么

不是“文件更多了”，而是下面四个方面更硬了：

1. stage 之间的输入输出不再靠口头约定
2. 报表与资源清理不再藏在 runtime/problem 内部
3. 随机种子从 experiment 开始一路贯穿到 inner symbolic torch fit
4. `report_payload` 已经有固定 schema，不会再轻易牵动多层

## 10. 下一步建议

如果继续按这条线推进，优先级建议是：

1. 给图表/报告补独立 plugin，而不是回写 runtime
2. 给 stage 结果对象再补 typed payload，而不只是 `Mapping[str, Any]`
3. 针对 smoke run 做同 seed 双跑一致性测试
4. 逐步把现存旧兼容脚本缩成真正薄 wrapper

补充：

- 现在已经有 `run_deterministic_smoke_regression.py`
- 默认会固定命令双跑并对比稳定 summary 字段
