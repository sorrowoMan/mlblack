# HF 会话 SFT 的责任与边界

完整可运行示例和配置解释见 [framework_assistant_sft Project](../../examples/cases/framework_assistant_sft/README.md)。

实现验收的 64 项定向测试、13 项回归及当时暂停训练的记录见 [首轮实现验收](HF_SFT_INITIAL_ACCEPTANCE_CN.md)。2026-10-25 用户重新授权后的真实预训练基座、24 步 LoRA 和前后质量对照见 [首次预训练微调记录](FRAMEWORK_ASSISTANT_PRETRAINED_PILOT_CN.md)：训练发布成功，语义质量未通过。

数据管线 `mlblack.pipeline.conversation_sft` 拥有 messages 输入校验、chat template 的调用和 label mask；它不会调度训练。`mlblack.integrations.hf_sft.HFSFTProvider` 拥有 HF 模型、PEFT 增量权重和有限 AdamW fit；它通过 BlackBase EvaluationGateway 消费 Project 已授予的 ResourceContext。

`HFSFTLearningProblem` 把一次完整 fit 的训练 loss 解释成评估反馈。`HFSFTRepresentation` 的一维候选只是 fit 请求，不假装它是全模型参数，也不将模型 Codec 与 Adapter Unit 合并。`LearningSolver`、`FixedCandidateAdapter` 复用正式共享控制面；Case 最终发布事务负责带收据的 adapter ZIP。

这一版的 optimizer step 属于外部有限 fit 内部执行，不能逐步回滚或恢复。失败的实际计算已经发生；未完成包不会发布。完成后 checkpoint 恢复只找已保存 bundle，不重训。真正 bitsandbytes NF4 是可选路径，未执行时必须标记未验证；不能把普通 LoRA 或 tiny 假量化称为已验证 QLoRA。

框架能力验证与模型质量验证分开：前者检查真权重变化、语料和模板对齐、取消、恢复与 Artifact；后者需要不泄漏的独立任务、同基座对照以及真实装配运行证据。

## 参数化实验与冻结资料

`python -m mlblack.integrations.hf_sft_experiment` 提供 `preflight`、`evaluate`、`train-eval` 三种有限执行方式，复用已有正式 Project，不生成第二套脚手架。模型、语料、输出目录与 run ID 都由调用方显式提供；安装包未包含示例时须用 `--project` 给出已有 Project。输出已存在直接拒绝，训练失败不自动重试。具体可执行参数与 JSON 配置示例见 Project README 第八节。

实验端只消费冻结资料，不实现检索或对答案的语义判分。普通语料每行有唯一 `id` 和 `messages`；grounding 另存 `{sample_id, split, context, source_refs}`，按 `(split, sample_id)` 精确绑定。grounded 训练显式派生 `prepared/train.jsonl`，原始文件不改写，参考答案不进入 generation prompt。数据、资料、执行源码、基座和模板均记录摘要并检查漂移；派生 manifest 还保留逐条来源证据。

CLI 的真实 tokenizer preflight 会检查所选文件、两种 prompt 和 completion 的 token 前缀；`--check-train` 加入训练数据，超长直接失败。`evaluate` 只评估基座及可选已有 adapter；没有正式收据的已有目录明确标成未验证发布权威。`train-eval` 先评估基座，通过原 Case 完成一次 fit、核验 Artifact 收据与字节，再评估新 adapter。同一批冻结问题可固定比较 base/plain、base/grounded、LoRA/plain、LoRA/grounded；验证集可只计算 NLL，只有显式 `--generate-split` 的集合才生成完整答案。CLI 不用 test 指标选 checkpoint，也不会据此追加步数。

## 模板、配置与恢复兼容

`HFSFTConfig.enable_thinking=None` 不向 tokenizer 传这个参数，保留旧模板默认行为；显式 True / False 同时用于训练 mask、评测 NLL 和 generation prompt。CLI 默认 `--thinking off`，不会发生训练一种思考模板、生成另一种模板的隐式差异。旧 checkpoint 缺字段时归一为 None，只能恢复到相同含义的配置，不能冒充显式 False。模板摘要是原始 UTF-8 字节的 SHA-256，与产物清单一致。

Case 的 `MLBLACK_SFT_CONFIG_PATH` 读取完整 JSON 对象，最终仍由 `HFSFTConfig` 严格校验；不会把布尔值、浮点数或字符串强转为整数步数。CLI 仅在本次调用范围设置和恢复配置文件路径及 Project device。CPU 机制验收覆盖旧 checkpoint 配置归一、精确 completion 边界、JSON 类型拒绝、原 Project 发布及 eval-only 不训练。

## 显存与运行资源

评测同样通过共享 Project L0 申请 lease 和 ResourceContext，按题验证授权仍有效，结束关闭 guard 并释放 lease。模型、optimizer 和大张量留在 Provider / 有限评测函数内，不进入共享 context。CUDA 评测在每题 NLL 后、生成前释放临时张量并清理授权设备的空闲缓存，题末再次清理，记录 allocated / reserved 字节及分段耗时。该操作缓解长序列全词表 logits 的缓存压力，不改变评估与生成的数值定义；不操作其他用户进程。

`gradient_checkpointing=True` 是训练时重算前向激活以节省显存，与持久 checkpoint、步骤事务或 optimizer 中途恢复无关。完整 fit 仍只支持完成后的 bundle 恢复。具体实验若在训练前因运行环境停止，要保留已完成评测和 `optimizer_steps_completed=0`；运行修复不能被描述为已经训练或已经验收质量。

2026-10-25 的固定 120-step Qwen3-0.6B 实验已完成正式训练、发布与四组评估，最终 ML 定向组合 76 项通过；新权重的六题语义审阅仍未通过。机制测试、损失和模型正确性分开记录，详见相邻 nsgablack 仓库的 [第二轮报告](../../../nsgablack/docs/development/FRAMEWORK_ASSISTANT_IMPROVEMENT_CN.md)。
