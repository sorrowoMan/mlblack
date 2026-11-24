# 框架助手的真实 HF / PEFT SFT Project

这个 Project 将**一次有限的外部 SFT fit**装进正式 ML Case。它复用共享 Solver / Trainer 生命周期，Provider 内部负责本次有限 fit 的优化器循环。实际调用是：

```text
Project L0 ResourceContext
  → LearningSolver / ComposableSolver
  → FixedCandidateAdapter 提出一个 fit 请求
  → HFSFTLearningProblem → EvaluationGateway → HFSFTProvider
  → Transformers 模型 + PEFT LoRA + 原生 Torch AdamW
  → 一条完整 fit 的 loss 反馈
  → Case 最终发布事务 → LoRA ZIP Artifact / PublicationReceipt
```

本实现没有依赖 TRL。虽然它可以与 TRL 生态共存，这一版执行的是明确的原生 Torch SFT 内核，不能描述成已经接入 `SFTTrainer`。

## 一、这解决什么问题

现有大型模型继续负责线上助手；这个 Project 用于构建和验证专门理解框架的增量权重。它支持真实本地预训练 HF causal LM、真实 PEFT LoRA、可选真正的 bitsandbytes NF4。它**不是**仓库旧 tiny QLoRA 演示的量化近似，也不会把全模型参数展开成优化候选。

单独跑通训练机制，不能证明模型已经理解框架。目录内 `pipeline/example.jsonl` 只展示格式；专属模型效果必须用独立任务评测并与同一基座比较。

## 二、最小依赖与惰性加载

```powershell
python -m venv --system-site-packages .venv-sft
.venv-sft/Scripts/python.exe -m pip install -e '.[sft]'
```

真正的 4-bit 路径另外安装 `.[sft_4bit]`，还需要支持的 bitsandbytes / CUDA 环境。普通 LoRA 不声称执行了 QLoRA。`load_in_4bit=True` 而没有明确 CUDA grant 会直接拒绝，不会偷偷退回 CPU 或假量化。

模块导入、`build_solver()`、Doctor build 和 preflight 不加载预训练模型，不下载权重，不占 GPU；只有运行评估才加载。默认 `local_files_only=True`，`trust_remote_code=False`。

## 三、数据与 label 规则

每行是一个对象：

```json
{"id":"pool-ownership-01","messages":[{"role":"system","content":"你是框架协作助手。"},{"role":"user","content":"Unit 淘汰一个成员，要删掉母池中的解吗？"},{"role":"assistant","content":"先解除这个 Unit 的成员关系。其他 Unit、子池或历史仍可能引用同一候选版本，不能因此删除全局候选。"}]}
```

要求：可选 system 开头；user / assistant 严格交替；至少一轮 user；最后是 assistant；内容为非空纯文本。id、family、来源等额外字段允许保留作审计，但不自动进入模型文本。工具消息和多模态消息目前不支持。

`label_mode="assistant"` 使用 tokenizer 自己的 `chat_template` 与 generation-mask，只训练 assistant 内容。模板没有 `{% generation %}` 或返回空 mask 就拒绝。`label_mode="completion"` 只训练最后一个 assistant continuation；系统验证“前面的会话 + generation prompt”在 token 级是完整会话的精确前缀，不匹配就拒绝。

两者都忽略 prompt 与 padding 的 label（`-100`），不偷偷为全会话计算 loss。所有样本**禁止静默截断**：真实 token 数超出 `max_length` 就报错，数据作者需要明确缩短或提高预算。纯文件 preflight 没加载 tokenizer，所以 `token_length_checked=False`，不能据此声称长 packet 适合 2k / 4k 上下文。

## 四、先预检，再正式执行

本例的默认模型路径并不存在，需要明确提供你准备好的本地 HF 目录和数据路径：

```powershell
$env:MLBLACK_SFT_MODEL_PATH = 'C:/models/your-causal-lm'
$env:MLBLACK_SFT_DATASET_PATH = 'C:/data/framework-cognition/train.jsonl'
python examples/cases/framework_assistant_sft/run_project.py --check --build-check
```

离线预检 API：

```python
from mlblack.integrations.hf_sft import HFSFTConfig, preflight_hf_sft

config = HFSFTConfig(
    model_name_or_path="C:/models/your-causal-lm",
    max_length=2048,
    optimizer_steps=20,
    label_mode="completion",
    target_modules=("q_proj", "v_proj"),
)
report = preflight_hf_sft(config, "C:/data/framework-cognition/train.jsonl")
print(report)
```

模型不同，LoRA target module 名称也可能不同；不要把 GPT-2 的 `c_attn` 用到所有模型。配置来自 Case `config.py`，正式 Project 默认只申请 1 CPU 线程。显式设置 `$env:MLBLACK_SFT_DEVICE='cuda:0'` 时，`project_config.py` 同时声明 Project 的 GPU offer 和本 Case request，之后由共享 L0 发 grant；Provider 不自己抢 GPU。还可设置 `MLBLACK_SFT_MAX_LENGTH`、`MLBLACK_SFT_OPTIMIZER_STEPS`、`MLBLACK_SFT_TARGET_MODULES`（逗号分隔）、`MLBLACK_SFT_REVISION`。设备 token 是项目资源配置，不是 Provider 越过授权自动选择的设备。

确认模型、模板、数据与资源后，用同一 Project 入口去掉 check 参数执行训练。标准 `run_solver.py` 也可作为独立 Case CLI；正式产物发布仍推荐 Project 路径。

## 五、训练与事务边界

- 一个候选是 1 维 fit 请求描述符，不是模型权重。base 模型、optimizer、张量和 tokenizer 只在 Provider 内。
- 一个 Solver generation = 一次完整有限 fit。`optimizer_steps=20` 不表示 Solver 进行了 20 代；预算中的 evaluation 也是一次完整 fit。
- 训练循环按固定种子、循环批次运行 AdamW，检查有限 loss、有限梯度和真实 LoRA 权重变化。它不是无限在线训练服务。
- `provider.cancel()` 是合作式取消，在模型加载前、每个 optimizer step 前后检查。取消或出错不发布部分模型，不能将已消耗计算回滚。
- 尚不支持一个 optimizer step 中途暂停、跨进程恢复 optimizer 状态或逐 optimizer step rollback；`get_state()` 在运行 fit 时明确拒绝。
- 完成后的 Provider checkpoint 保存已完成的 LoRA bundle，恢复模型投影只读消费原有 feedback 和已完成包，不再 fit。Provider state 含模型产物，可能较大，走正式 checkpoint / Snapshot / Artifact，不进共享 ctx。
- 对已完成 Solver 做纯恢复时使用 `target.set_state(source.get_state()); target.fit(max_steps=0)`。共享 LearningSolver 的 `fit(max_steps=n)` 在恢复后表示追加 n 个逻辑步骤；显式追加 `1` 会请求另一次完整 fit，不能把这种请求当成只读恢复。
- 真正模型 Artifact 只在共享 Case finalization 提交成功后有权威引用；Provider 私有内存里的完成包不是已发布 DataRef。

## 六、产物包含什么

ZIP 内含 PEFT `adapter_model.safetensors`、`adapter_config.json`、tokenizer 文件、会话模板和 `sft_manifest.json`，不包含基座全量权重。Manifest 保存：

- 基座路径、请求 revision、解析出的 commit（若模型提供）、本地基座内容摘要。
- 原始 JSONL 文件摘要、规范化会话摘要、同目录数据 manifest 摘要（存在时）。
- 实际模板及模板摘要、标签模式、token 长度、监督 token 总数。
- LoRA / 学习率 / 步数 / seed / 量化设置、依赖版本。
- 真正执行设备和 L0 grant、训练参数数、权重发生变化的张量数量、loss。
- `optimizer_resume=False` 与“只证明机制，未声称框架质量提升”的标记。

发布的 ZIP 可用 PEFT `PeftModel.from_pretrained(base, adapter_dir)` 恢复；解压路径需要应用侧检查 ZIP 文件名与目录边界。未来接线上推理必须通过独立评测与显式后端切换，本 Project 不会自动替换 DeepSeek。

## 七、验收

`tests/test_hf_sft.py` 的小模型是本地随机初始化 GPT-2，不联网下载，用真实 Transformers + PEFT 执行一步。它覆盖 label mask、padding、超长拒绝、CUDA 约束、取消不发布、LoRA 变化、adapter-only ZIP 和恢复不重训。它检验训练接缝，**不检验框架认知质量**。

实际 pretrained 训练、独立评测的命令和结果应另附实验记录，不能将这个机制测试改称已训练好专属模型。

## 八、参数化实验入口

此前缓存里的固定脚本只用于记录首轮实验。后续可以使用可复用入口，不需要修改脚本里的路径，也不会另造 Project：

```powershell
python -m mlblack.integrations.hf_sft_experiment --help
```

它有 `preflight`、`evaluate`、`train-eval` 三种模式。所有模式都要显式传 `--dataset`、`--model`、`--output`、`--run-id`；`--project` 默认当前仓库本 Project，若安装包没有示例目录，必须提供正式 Project 路径。输出目录已存在会直接拒绝，既不覆盖旧实验，也不自动重试训练。

例如先做严格分词预检（这里都是示例路径）：

```powershell
python -m mlblack.integrations.hf_sft_experiment preflight `
  --dataset C:/data/framework-cognition --model C:/models/Qwen3-0.6B `
  --output C:/experiments/preflight-001 --run-id preflight-001 `
  --project examples/cases/framework_assistant_sft `
  --check-train --training-prompt-mode grounded `
  --grounding-file C:/data/framework-cognition/grounding.jsonl `
  --grounding-file C:/data/frozen-acceptance/grounding.jsonl `
  --eval-split validation --eval-file test=C:/data/frozen-acceptance/test.jsonl `
  --prompt-modes plain grounded --generate-split test `
  --thinking off --max-length 2048 --optimizer-steps 120 --gradient-checkpointing
```

`preflight` 只加载实际 tokenizer，逐条验证所选评测文件、prompt 模式和 completion 精确边界；加 `--check-train` 时还验证训练文件。它不加载模型权重，保存所有长度，超限直接失败，不静默截断。它与前面的纯文件 `preflight_hf_sft()` 不同，成功时 `token_length_checked=True`。

`evaluate` 不训练。默认评估 base；指定 `--adapter` 可以另外评估已有 PEFT adapter 目录。只有 `--generate-split` 指定的 split 才生成自由回答，其他 split 只记录 NLL，因此可以只给 test 生成回答，并对 validation 计算指标。指定 `--prompt-modes plain grounded` 会对相同问题固定做两种输入，保留完整回答、prompt/reference hash、资料来源、EOS 和生成上限标志。已有 adapter 没有正式收据时只作为本地输入，不能冒称已经验证其发布权威。

`train-eval` 先评估 base，再通过原 Project 执行一次完整 fit，验真 Artifact 收据与物理内容，然后用相同问题和参数评估新 LoRA。两种 prompt 默认在训练前后都保留，形成 base/plain、base/grounded、LoRA/plain、LoRA/grounded 四组。训练由 `--training-prompt-mode` 单独决定；grounded 模式显式派生输出目录中的 `prepared/train.jsonl`，不会改原语料。可以用 `--after-prompt-modes` 明确指定新模型的评估输入，但不能据此改变或自动追加训练。

外部 grounding 文件每行使用：

```json
{"sample_id":"pool-ownership-01","split":"train","context":"经检索冻结的框架资料","source_refs":[{"source":{"path":"core/state/solution_pool.py","line":1,"sha256":"来源摘要"},"excerpt":"来源片段"}]}
```

原始语料每行需有稳定、唯一的 `id`。CLI 按 `(split, sample_id)` 严格绑定，缺资料和重复身份都拒绝，不按文字相似度猜。ML 执行端只消费冻结资料；检索属于上游框架知识工具，不在这里重做。资料进入 system 区，参考答案不会注入用户 prompt。`prepared/manifest.json` 保存原文件和 grounding 的摘要及每条引用。

新 CLI 的 `--thinking off` 是默认设置，它会同时传给训练分词、NLL 分词、generation prompt，并在 manifest 中记录。`HFSFTConfig.enable_thinking=None` 仍保留旧调用的模板默认行为；旧 checkpoint 缺这个字段时解释为 `None`，不能冒充新 `False`。`--gradient-checkpointing` 是训练时用计算换激活显存，不是保存 optimizer checkpoint 或提供中途恢复。

除了前面的简单环境变量，还能通过 `MLBLACK_SFT_CONFIG_PATH` 指向 JSON 对象，字段是 `HFSFTConfig` 的字段和额外的 `dataset_path`。例如：

```json
{"model_name_or_path":"C:/models/Qwen3-0.6B","dataset_path":"C:/data/framework-cognition/train.jsonl","max_length":2048,"optimizer_steps":120,"enable_thinking":false,"gradient_checkpointing":true,"target_modules":["q_proj","v_proj"]}
```

JSON 字段覆盖本例环境变量默认值，`build_solver(config=..., component_overrides=...)` 的显式值再覆盖 JSON。JSON 不做宽松类型转换：`optimizer_steps` 的 `true`、`1.5`、`"120"` 都拒绝；`enable_thinking` 只能是布尔值或 null。未知字段也不能静默忽略。实验 CLI 自动生成 `case-config.json`，在自己的调用范围临时指定这个路径和 Project device，退出时恢复环境，不改用户的持久配置。

训练和评测都经现有 Project L0 授权。评测申请独立有限 lease，结束时关闭 guard、释放 lease 和模型张量；训练仍由原 Case 的 ResourceContext 交给 Provider。评测逐条完成高序列 NLL 后释放临时张量和当前授权 CUDA 设备的空闲 allocator cache，再生成回答，题末再次清理。这样避免把全词表 logits 的历史缓存一直占在 8GB 显卡上；不会改变 loss、候选、模板、生成参数或监督内容。每题额外记录 NLL / generation 耗时，以及清理后的 allocated / reserved 字节数。缓存释放不等于回滚已执行计算，也不能保证任何模型都适合本机显存。

主要输出是 `experiment.json`、`base.json`、`after.json`、`summary.json`、逐题 partial 文件、`status.json`、`events.jsonl`，训练成功时另有 Project 结果和发布验证。`experiment.json` 冻结原数据、grounding、Project 配置、三个实际执行模块的内容摘要，以及本地基座和模板摘要。`chat_template_sha256` 对原始模板 UTF-8 字节计算，与 Provider 产物使用同一口径。执行前后检查来源是否漂移。异常保留 `failure.json`，不自动补跑。NLL、EOS 与完整回答都只是证据；语义正确性须在独立审阅中判断，不自动用关键词计分替代。

2026-10-25 已实际完成 Qwen3-0.6B 的 24 步 LoRA、正式 Artifact 发布及五题前后对照。训练机制成功，但独立 Agent 语义审阅为 0 正确／1 部分／4 错误，未替换线上助手；详见 [首轮真实训练记录](../../../docs/architecture/FRAMEWORK_ASSISTANT_PRETRAINED_PILOT_CN.md)。

随后用上述可复用 CLI 完成第二轮：48 条带依据训练、120 steps、六题 plain／grounded 前后对照。正式收据通过；两组新 LoRA 均为 0 正确／3 部分／3 错误，仍未通过质量门。运行、检索缺口和验证限制见相邻 nsgablack 仓库的 [第二轮报告](../../../../nsgablack/docs/development/FRAMEWORK_ASSISTANT_IMPROVEMENT_CN.md)。两轮问题不同，不能直接把计数作提升率。
