# HF / PEFT SFT 首轮实现验收

记录日期：2026-10-25（开始于 10 月 24 日，缓存目录保留起始日期）。

本记录保存正式训练接缝、数据与标签契约、可审计产物和 CPU 机制验收，以及用户当时决定暂缓真实训练的历史状态。2026-10-25 用户重新授权后的实际 24 步 LoRA、发布收据和失败的质量验收另见 [首次预训练微调记录](FRAMEWORK_ASSISTANT_PRETRAINED_PILOT_CN.md)。下文“尚未执行”均指该次实现验收截止时，不代表后续仍未训练。

## 1. 已完成的实现

- `pipeline/conversation_sft.py`：messages JSONL 校验、真实 chat template、assistant / completion 标签掩码、padding 忽略、超长拒绝。
- `integrations/hf_sft.py`：真实 Transformers / PEFT LoRA，原生 Torch AdamW；可选真实 bitsandbytes NF4 路径；完整 fit 是一次 EvaluationProvider 求值。
- `examples/cases/framework_assistant_sft`：薄 Project / Case / Scaffold，L0 授权、LearningSolver、FixedCandidateAdapter、EvaluationGateway 与 Artifact 最终发布事务复用原框架。
- `[sft]`、`[sft_4bit]` 可选依赖，以及 `integration.hf_sft` Catalog 条目。

模型、optimizer、tokenizer 驻留 Provider；候选是一维 fit 请求，不平坦化全模型参数。构造与 preflight 不加载、下载模型。默认只读本地文件。

## 2. 实际环境

验证用解释器是仓库内隔离环境：

```text
C:/Users/hp/Desktop/mlblack/.pytest_cache/hf-sft-venv/Scripts/python.exe
```

该 venv 通过 `--system-site-packages` 复用原有 Torch，不更换全局 Torch；PEFT 和 Accelerate 只安装在这个 venv。实测版本为 Torch `2.7.0+cu128`、Transformers `5.8.0`、PEFT `0.18.1`、Accelerate `1.15.0`。依赖声明允许的旧版本没有逐个跑矩阵，CUDA 模型参数使用兼容 4.51 / 5.x 的 `torch_dtype` 写法。

## 3. 测试与行为证据

从 mlblack 根目录执行：

```powershell
.pytest_cache/hf-sft-venv/Scripts/python.exe -m pytest tests/test_hf_sft.py -q --tb=short
python -m pytest tests/test_model_projection_restore.py tests/test_case_runtime_artifact_provider.py -q --tb=short
python examples/cases/framework_assistant_sft/run_project.py --check --build-check
python -c "from mlblack.project.doctor import run_project_doctor,format_doctor_report; print(format_doctor_report(run_project_doctor('examples/cases/framework_assistant_sft',strict=True)))"
python -m mlblack catalog list --kind integration
```

结果：

- HF SFT 定向测试 **64 通过**；其中真实本地随机 GPT-2 + PEFT 执行一步，LoRA 张量确实变化，训练参数少于总参数，没有联网下载模型。两个 GPT-2 Conv1D 的 PEFT `fan_in_fan_out` 提示为预期兼容提示。
- 既有 ML 模型恢复和 Artifact Provider 回归 **13 通过**，与上面文件无重复，共 77 项。
- 标准 Project `--check --build-check` 成功，运行面可见实际 FixedCandidateAdapter、HFSFTProvider 与 `resource_binding.current=True`。
- 新 Project strict Doctor 成功，0 error / 0 warning，3 info；没有把这个结果夸大成任意算法正确性的证明。
- Catalog 可发现 `integration.hf_sft`。

64 项覆盖：配置在加载前拒绝小数、bool、字符串伪整数及非有限学习率；只 seed CPU 和授权 CUDA 的 RNG；CPU 不初始化 CUDA；源数据或 manifest 替换时拒绝；真实标签与 padding；超长不截断；取消不发布；4-bit 没 CUDA 时拒绝；LoRA-only ZIP；正式 Case 的发布收据验真。

完整 Solver 恢复测试使用：

```python
checkpoint = source.get_state()
target.set_state(checkpoint)
target.fit(max_steps=0)
```

测试禁止目标 Provider `_fit` 被再次调用，并核对相同候选 token、已完成结果 ID、bundle bytes、generation 与 evaluation count。这里 `0` 是只读恢复预算；共享 LearningSolver 的 `fit(max_steps=1)` 在恢复后是**追加一次完整 fit**，不是只读恢复。

最终定向输出保留在 `.pytest_cache/hf-sft-tests-final.txt`。独立的只读恢复试验过程中曾把追加预算 `1` 误当作总预算，测试准确抓到了新 fit 请求；修正测试为 `0`，没有更改共享控制面、重训恢复结果或吞掉异常。

## 4. 已准备的预训练基座与未执行的工作

公共 `Qwen/Qwen3-0.6B` 基座已经下载，固定 revision：

```text
c1899de289a04d12100db370d81485cdf75e47ca
.pytest_cache/models/Qwen3-0.6B/model.safetensors
1,503,300,328 bytes
```

该目录同时有 config、tokenizer 与会话模板。最初 Hub Xet 下载未推进，停止了本任务下载进程后，通过固定 revision 的普通 HTTPS 完成；没有使用或配置新的模型服务 API key。

已准备一次 12 train / 3 validation / 5 test 的 `cognition_only` 试验脚本，它不等于线上 JSON 装配训练。完整生产 packet 与短认知 profile 必须分开；不能把几十 k 字符的完整 packet 静默截断成 1k 后声称已学会线上装配。

预训练试验停在基线评估，日志只完成第一题；baseline 未完成，**框架语料 optimizer steps = 0，未产出专属 LoRA**。当时本机用户应用正在占用 GPU，随后按用户要求停止本任务的 Python 进程。没有关闭其他应用。停止后本任务进程已不存在，显存由约 7.6 GB 降至约 5.3 GB；此数字只是当时观测，不是持续监控。

状态文件：

```text
.pytest_cache/framework-assistant-pretrained-20261024/status.json
.pytest_cache/framework-assistant-pretrained-run.log
```

只在用户之后再次决定执行时，才运行：

```powershell
.pytest_cache/hf-sft-venv/Scripts/python.exe -u .pytest_cache/run_cognition_sft.py
```

脚本会重新完成同题基线、正式 Project 24 步有限 fit 和同题训练后评估，使用共享 L0 给 CUDA 发 grant；不会自动替换线上 DeepSeek。上述命令当前只是重跑说明，本轮没有排队或后台执行。

## 5. 没有声称已证明的事

- 随机 tiny 的权重变化只证明机制，不证明模型理解框架。
- 尚未执行完整框架语料训练或前后质量对照；不能报告认知提升、装配成功率或微调优于 DeepSeek。
- bitsandbytes 4-bit 代码存在，但本轮没有验证真实 GPU QLoRA。
- 不支持中途 optimizer checkpoint、逐 optimizer step 回滚或恢复；完成的 fit checkpoint 才能恢复。

操作说明与所有配置项见 [Project README](../../examples/cases/framework_assistant_sft/README.md)。
