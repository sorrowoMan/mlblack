# Runtime Plugin TODO（副作用清理）

## 目标
把 `problem_model.py / workflow_runtime.py` 中的副作用（写文件、报表输出、缓存关闭、打印）统一迁移到 `RuntimeHook` 插件平面，确保控制平面只做编排。

## 已落地
1. `ReportWriterPlugin(RuntimeHook)` 已新增：
   - 文件：`plugins/report_writer_plugin.py`
   - 监听：`on_experiment_finish`
   - 行为：调用 `write_summary_report(...)` 统一写 `summary.json` 并输出摘要日志
2. `workflow_runtime.main()` 已改为返回结构化结果：
   - `report_payload`（供插件消费）
   - `result_summary`（供控制平面/上层观察）
3. `workflow.main()` 默认注册 `ReportWriterPlugin`，旧命令 `run.py` 无需修改即可继续产出报告。

## 下一步待办
1. 拆分 `orchestrator` 的单个 `"evaluate"` 胖阶段：
   - `prepare -> evaluate -> aggregate -> finalize`
2. 增加第二个插件：
   - `CacheLifecyclePlugin`（统一关闭/刷新 graph cache，处理异常路径）
3. 把 `runner.py` 的旧式 `write_summary_report(...)` 路径迁移到同一插件机制，避免双轨逻辑。
4. 定义标准事件载荷契约：
   - `on_stage_end` 的 payload schema
   - `on_experiment_finish` 的 `result` schema
