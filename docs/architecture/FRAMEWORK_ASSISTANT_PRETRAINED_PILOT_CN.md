# 框架助手首次预训练基座微调：实测记录

执行日期：2026-10-25。此次用户重新授权执行后，完成了上次因 GPU 争用而暂停的真实训练。旧的 20261024 暂停记录保留，本次另存新目录。

**训练与产物发布成功，模型质量未通过。** Qwen3-0.6B 完成 24 个 LoRA optimizer steps，正式 Project 发布了可校验的权重包。五道保留题的平均目标 token 损失下降，但独立 Agent 逐题审阅仍为 0 道正确、1 道部分、4 道错误。该产物不能替换当前线上助手。

本文保留首轮原始记录。后续第二轮已改用带源码依据的 48 条训练课程和独立冻结的六题，完成固定 120 步及 plain／grounded 四组对照；新 LoRA 两组均 0 正确／3 部分／3 错误，仍未通过质量门。详见 [第二轮完善报告](../../../nsgablack/docs/development/FRAMEWORK_ASSISTANT_IMPROVEMENT_CN.md)，不能将两轮不同问题的计数当成同题提升率。

## 1. 此次究竟执行了什么

运行身份：`framework-assistant-cognition-20261025T003133`。

顺序固定为：

1. 原始基座回答五道保留题，并计算参考回答的 completion NLL。
2. 经标准 `framework_assistant_sft` Project 的 `assistant_sft` Case，执行一次完整有限 fit。
3. 经 Case finalization 原子发布 LoRA Artifact，校验发布收据，再通过 verified ArtifactBinding 读取实际 ZIP。
4. 重新加载同一基座与该 LoRA，回答同五题，按相同设置计算 NLL。
5. 校验前后输入与生成参数一致，独立审阅每题语义，退出实验进程并检查 GPU 占用。

整个实验耗时约 **119.84 秒**，包含训练前生成、训练、发布和训练后生成，不是单独训练耗时。没有追加第二轮训练或根据测试答案调参重跑。此次没有调用 DeepSeek，也没有比较其效果。

## 2. 模型、数据与训练设置

基座为本地缓存的 `Qwen/Qwen3-0.6B`，请求 revision 固定为：

`c1899de289a04d12100db370d81485cdf75e47ca`

由于加载本地目录，HF config 的 resolved revision 为 null，不能伪称 HF 运行时再次确认了远程 commit。产物另记录本地模型内容摘要 `2f6b385bc7611a08f7cce73bdf78d42e2314e77f62008510a703d763645b77c4`；Provider 在加载前后及 fit 完成后检查同一来源未变。

本次使用 `framework_cognition_only/v1` 短认知 profile，来源是已复核的 `assistant-training-reviewed-20261024`，完整语料版本为 `2026-10-24.2`。短语料位置在 nsgablack 的 `.pytest_cache/framework-assistant-20261024/cognition-reviewed/`。

- train 12 条；validation 3 条；test 5 条。
- validation 文件没有用于本次训练、调参或选择 checkpoint，也没有另报验证集分数。
- 12 条训练会话长 322–463 tokens，单遍合计 **1,431 个监督目标 tokens**。
- batch size 为 1，循环读固定顺序，24 steps 相当于两遍训练样本；没有随机打乱。
- max length 为 1,024；超长直接拒绝，本次没有截断任何训练会话。
- completion-only 标签；输入提示不计入目标损失。
- LoRA rank 8、alpha 16、dropout 0；target modules 为 `q_proj` 和 `v_proj`。
- AdamW，learning rate 0.0002，seed 42。
- 普通 LoRA，`load_in_4bit=False`；这不是 QLoRA / NF4 验收。

短语料 manifest 摘要为 `598450559754e6325a704e0d121db7fb733fd5a6aee92b722c29957eddb215bd`；train.jsonl 文件摘要为 `a78c8fedb7e6393db92136ba5c6dda72083694cd518a8adf2829972eb6f27fbc`。独立检查确认四个短语料文件摘要及 parent manifest 连接均匹配。

环境复用仓库隔离 venv：Torch 2.7.0+cu128、Transformers 5.8.0、PEFT 0.18.1。没有替换全局 Torch。

## 3. 框架的真实交接

正式装配继续使用 [标准 Project](../../examples/cases/framework_assistant_sft/README.md)。Project-level L0 授予 `cuda:0`、1 个 GPU 和 1 个线程，Provider 消费授权，没有绕过共享授权层。训练模型、tokenizer 和 optimizer 由 ML Provider 持有，不放入 ctx，也不当作 Unit 或搜索候选传递。

框架计数必须与训练步数分开：**24 个 optimizer steps 是一次完整 fit，只有 1 次 Solver 逻辑提交及 1 次正式求值**。控制面的步骤事务没有因此获得逐 optimizer step 回滚能力。

实际更新了 **112 个 adapter 张量**。可训练参数为 **1,146,880**，含 LoRA 的总参数为 597,196,800，可训练占比约 **0.192%**。平均训练 loss 为 4.446742；它来自训练过程中的不同样本，不能当成保留集质量分数。

## 4. 前后比较的范围与数字

两次生成都使用 greedy decoding：`do_sample=False`、`max_new_tokens=512`、`enable_thinking=False`。同一题的 system/user 提示逐项相同；参考答案不进入生成提示。十个生成结果均遇到 EOS，没有达到 512-token 上限，因此以下错误不能归因于回答被截断。

两套 completion NLL 对相同的 **719 个参考目标 tokens** 计算 token 加权平均：

- 原始基座：**5.386575**。
- 加载 LoRA 后：**4.560712**。
- 描述性下降约 **15.33%**。

这里的 NLL 使用训练同款默认 chat template，包括模板形成的 completion；生成提示则明确关闭 thinking。前后比较各自保持同模板、同标签和同生成设置，但 NLL 与自由生成不是完全相同的条件，也不是同一个质量指标。不能把损失下降叫作算法装配成功率提升。

短 profile 是概念回答，不是线上完整 JSON 装配协议。五道题没有进入这次训练分集，但主题曾参与历史设计讨论，不能称严格盲测。没有统计显著性、未见分布泛化或生产可用性的证据。

## 5. 逐题语义结果

另一 Agent 在查看模型回答前固定了判据：正确需要覆盖关键边界且无实质反向断言；部分表示方向正确但关键交接不完整；错误表示中心判断错误或明确推荐禁止操作。审阅者读过参考答案，属于 **Agent 语义审阅，未经过人审**；不用关键词命中或 NLL 自动判对。

### 5.1 BO 与 Bias：前后均错误

基座把完整贝叶斯优化视为 Bias，并把“当前页面未开放”扩大成“框架没有完整 BO 实现”。

微调后知道当前页面不能直接执行完整 BO，却仍说“只能用 Bayes 作为软 Bias”，没有区分风险软评分与模型/采集函数拥有下一批采样决策，且混入无关的 CMA-ES。正确片段不能抵消中心职责混淆。

### 5.2 每成员 sigma：错误变为部分

基座赞同 mutator 从全局 ctx 读取行号，还把 sigma 与接纳条件混为一谈。

微调后拒绝直接读行号，但反复说“需要重新计算”，没有交代精确候选身份、PopulationChange 的本地成员映射、member_fields 的维护或只读 UnitInputs。该回答不足以指导安全实现；这是唯一可见的方向性改善。

### 5.3 嵌套子池：前后均错误

两套都要求删除父视图时一起删除后代，并赞同按名称前缀推拓扑、按 token 追读最新版本及重复投影。

这些都与当前契约相反：应保持显式 view 身份和冻结拓扑，读取精确 SolutionRef 版本，父级删除不级联，本层新反馈的继承投影只应用一次。

### 5.4 跨 Case 与 L0：前后均错误

两套都赞同跨 Case 直接传递活池对象、由内层另发全局 lease。微调后还说外层与内层是“同一 Case”。

它没有守住运行身份、引用 scope 与唯一授权边界。正确路线应是 Project 编排、正式 payload / Artifact / Snapshot 引用和收据，并由 Project L0 唯一发放全局 grant。

### 5.5 证据恢复：前后均错误

两套都明确建议重新调用 Problem、接纳规则和 Adapter.update 来补齐 pending Event。

当前恢复只能按 durable evidence 幂等对账，不隐式重跑业务；已有裁决待核验或快照暂不可读需保留 deferred，确认没有持久处置时才走相应 abandoned 分支。模型没有表达这些边界。

总体为：基座正确 0／部分 0／错误 5；LoRA 后正确 0／部分 1／错误 4。训练后部分回答出现无信息重复。**不能据此报告模型已经理解框架，也不能把这组结果外推为所有微调都无效。**

## 6. 权重、收据与复核文件

实验目录：`.pytest_cache/framework-assistant-pretrained-20261025/`。

- [实验设置](../../.pytest_cache/framework-assistant-pretrained-20261025/experiment.json)、[最终状态](../../.pytest_cache/framework-assistant-pretrained-20261025/status.json)、events.jsonl。
- [训练前回答](../../.pytest_cache/framework-assistant-pretrained-20261025/before.json)与[训练后回答](../../.pytest_cache/framework-assistant-pretrained-20261025/after.json)，含逐题参考、NLL、生成长度、EOS 和截断标记。
- [数值总结](../../.pytest_cache/framework-assistant-pretrained-20261025/summary.json)、[发布收据验真](../../.pytest_cache/framework-assistant-pretrained-20261025/publication-verification.json)、postcheck.json 与 [独立复核](../../.pytest_cache/framework-assistant-pretrained-20261025/root-verification.json)。
- [语义审阅](../../.pytest_cache/framework-assistant-pretrained-20261025/semantic-review.json)，绑定两份回答文件与逐题回答摘要。
- adapter/ 中的真实增量权重、tokenizer、模板和 sft_manifest.json。

正式发布 ZIP 为 **6,407,936 bytes**，摘要 `sha256:2ca0b889c354f2de9fe65d9e9071c14b9522783adbe98c8a51ff99d78247c3a0`。物理文件及其发布收据都已验证，准确引用见 summary.json；没有用一个“tuned”标签冒充加载了权重。包中 adapter_model.safetensors 为 4,602,248 bytes，不包含基座全量权重。

这些运行目录是本机缓存和 Project 产物，不随源码文档自动分发。复现实验需保留相应数据、固定基座和环境。一次性脚本 `.pytest_cache/run_cognition_sft.py` 拒绝覆盖已有输出目录，不能直接反复运行当作新的独立实验。

## 7. 结束状态与后续判断

本次 Python 进程已退出，GPU 占用从实验前约 601 MiB 回到约 603 MiB；这是结束时实测，不是持续监控。没有留下后台训练，没有自动更新线上助手。

此次闭环证明真实数据可以经框架完成训练、发布、加载与同题对照；质量门仍未通过。12 条样本、两遍合计约 2,862 个监督 token 的学习量很小，但此次没有对模型规模、数据覆盖和训练时长做受控消融，不能把失败单独归因于某一个因素。

后续应先补足关键规则的正反例、用户纠正与完整任务实例，把当前源码事实通过检索交给模型，再固定更独立的验收题。是否扩大模型和训练预算应由新对照决定，不能只因 NLL 下降就继续加轮次或切换生产。
