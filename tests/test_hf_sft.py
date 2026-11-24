from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from mlblack.integrations.hf_sft import (HFSFTConfig, HFSFTProvider,
    SFTCancelled, build_hf_sft_solver, preflight_hf_sft)
from mlblack.pipeline.conversation_sft import (collate_conversations,
    load_conversations, tokenize_conversation)


def _dataset(tmp_path):
    path = tmp_path / "train.jsonl"
    messages = [{"role": "system", "content": "hello"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer world"}]
    path.write_text(json.dumps({"messages": messages}), encoding="utf-8")
    return path, messages


def test_preflight_does_not_load_or_claim_token_fit(tmp_path, monkeypatch):
    path, _ = _dataset(tmp_path)
    report = preflight_hf_sft(HFSFTConfig(str(tmp_path / "absent")), path)
    assert report["model_loaded"] is False
    assert report["download_attempted"] is False
    assert report["token_length_checked"] is False
    assert report["ready_to_attempt"] is False
    assert report["records"] == 1


@pytest.mark.parametrize("field", ["max_length", "optimizer_steps", "batch_size", "lora_rank", "lora_alpha", "seed"])
@pytest.mark.parametrize("value", [True, "2", 1.5, 0.5, float("nan"), float("inf")])
def test_integer_config_fields_reject_nonintegers_before_loading(field, value):
    with pytest.raises(ValueError, match=field):
        HFSFTConfig("not-loaded", **{field: value})


@pytest.mark.parametrize("field,value", [
    ("learning_rate", True), ("learning_rate", "0.1"), ("learning_rate", float("nan")),
    ("learning_rate", float("inf")), ("learning_rate", 0),
    ("lora_dropout", True), ("lora_dropout", float("nan")), ("lora_dropout", 1),
    ("target_modules", "q_proj"), ("target_modules", ("",)),
    ("load_in_4bit", "false"), ("local_files_only", 1), ("gradient_checkpointing", None),
])
def test_other_invalid_config_fails_before_loading(field, value):
    with pytest.raises(ValueError, match=field):
        HFSFTConfig("not-loaded", **{field: value})


def test_cpu_seed_never_accesses_or_initializes_cuda():
    from mlblack.integrations.hf_sft import _seed_fit_rng
    seeds = []
    class CPUOnly:
        random = SimpleNamespace(default_generator=SimpleNamespace(manual_seed=seeds.append))
        @property
        def cuda(self):
            pytest.fail("CPU SFT touched CUDA")
        def manual_seed(self, seed):
            pytest.fail("global manual_seed would touch unrelated CUDA devices")
    _seed_fit_rng(CPUOnly(), 42, [])
    assert seeds == [42]


def test_seed_only_changes_granted_cuda_generator():
    from mlblack.integrations.hf_sft import _seed_fit_rng
    seeds = []
    generators = [SimpleNamespace(manual_seed=lambda value: seeds.append((0, value))),
                  SimpleNamespace(manual_seed=lambda value: pytest.fail("ungranted CUDA seeded"))]
    fake = SimpleNamespace(random=SimpleNamespace(default_generator=SimpleNamespace(
        manual_seed=lambda value: seeds.append(("cpu", value)))),
        cuda=SimpleNamespace(default_generators=generators))
    _seed_fit_rng(fake, 7, [0])
    assert seeds == [("cpu", 7), (0, 7)]


@pytest.mark.parametrize("source", ["dataset", "manifest"])
def test_provider_rejects_replaced_source_before_model_loading(tmp_path, source):
    dataset, _ = _dataset(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"version":1}', encoding="utf-8")
    provider = HFSFTProvider(HFSFTConfig("not-loaded"), dataset)
    frozen = provider.dataset_file_digest
    if source == "dataset":
        dataset.write_text(dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        manifest.write_text('{"version":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="changed after Provider construction"):
        provider._assert_dataset_unchanged()
    assert provider.dataset_file_digest == frozen


@pytest.mark.parametrize("messages", [[], [{"role": "user", "content": "q"}],
    [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}],
    [{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}],
    [{"role": "tool", "content": "a"}, {"role": "assistant", "content": "b"}]])
def test_reject_invalid_conversations(tmp_path, messages):
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps({"messages": messages}), encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_conversations(path)


def _tiny_model(tmp_path):
    pytest.importorskip("peft")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    tokens = ["<pad>", "<unk>", "<eos>", "<|system|>", "<|user|>", "<|assistant|>",
              "question", "answer", "hello", "world"]
    backend = Tokenizer(models.WordLevel({value: index for index, value in enumerate(tokens)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
        unk_token="<unk>", eos_token="<eos>")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ '<|' ~ message['role'] ~ '|> ' }}"
        "{% if message['role'] == 'assistant' %}{% generation %}"
        "{{ message['content'] ~ ' <eos> ' }}{% endgeneration %}"
        "{% else %}{{ message['content'] ~ ' <eos> ' }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|assistant|> ' }}{% endif %}")
    folder = tmp_path / "base"
    tokenizer.save_pretrained(folder)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokens), n_embd=16, n_layer=1,
        n_head=2, n_positions=64, bos_token_id=2, eos_token_id=2, pad_token_id=0))
    model.save_pretrained(folder, safe_serialization=True)
    return folder, tokenizer


def test_real_token_masks_padding_and_no_truncation(tmp_path):
    _, tokenizer = _tiny_model(tmp_path)
    _, messages = _dataset(tmp_path)
    for mode in ("assistant", "completion"):
        row = tokenize_conversation(tokenizer, messages, max_length=64, label_mode=mode)
        selected = [value for value in row["labels"] if value != -100]
        assert tokenizer.convert_tokens_to_ids("answer") in selected
        assert tokenizer.convert_tokens_to_ids("question") not in selected
        short = {key: value[:-1] for key, value in row.items()}
        batch = collate_conversations((row, short), pad_token_id=tokenizer.pad_token_id)
        assert batch["labels"][1, -1].item() == -100
        assert batch["attention_mask"][1, -1].item() == 0
    with pytest.raises(ValueError, match="truncation is forbidden"):
        tokenize_conversation(tokenizer, messages, max_length=2)
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] ~ ' ' }}{% endfor %}"
    with pytest.raises(ValueError, match="assistant mask unavailable"):
        tokenize_conversation(tokenizer, messages, max_length=64)


def test_true_peft_fit_through_learning_solver_and_read_only_restore(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    folder, _ = _tiny_model(tmp_path)
    dataset, _ = _dataset(tmp_path)
    cfg = HFSFTConfig(str(folder), max_length=64, optimizer_steps=1,
        learning_rate=0.02, target_modules=("c_attn",), lora_rank=2, lora_alpha=4)
    solver = build_hf_sft_solver(cfg, dataset,
        resource_context={"device": "cpu", "threads": 1, "namespace": "test.sft"})
    before_threads = torch.get_num_threads()
    result = solver.fit()
    assert solver.generation == 1
    assert solver.dimension == 1
    assert solver.best_score is not None
    assert torch.get_num_threads() == before_threads
    provider = solver.evaluation_provider
    model = solver.best_model
    output = provider.result(model["sft_result_id"])
    manifest = output["manifest"]
    assert manifest["changed_adapter_tensors"] > 0
    assert manifest["trainable_parameters"] < manifest["total_parameters"]
    assert manifest["optimizer_resume"] is False
    assert manifest["quantization"] == "none"
    assert manifest["dataset_sha256"] == preflight_hf_sft(cfg, dataset)["dataset_sha256"]
    archive = zipfile.ZipFile(io.BytesIO(output["bundle"]))
    assert "adapter_model.safetensors" in archive.namelist()
    assert "sft_manifest.json" in archive.namelist()
    assert "tokenizer.json" in archive.namelist()
    assert "model.safetensors" not in archive.namelist()
    checkpoint = provider.get_state()
    restored = HFSFTProvider(cfg, dataset)
    restored.set_state(checkpoint)
    assert restored.result(model["sft_result_id"])["bundle"] == output["bundle"]
    monkeypatch.setattr(restored, "_fit", lambda *_: pytest.fail("restore must not fit"))
    from mlblack.integrations.hf_sft import HFSFTLearningProblem
    problem = HFSFTLearningProblem(restored)
    assert problem.restore_model_for_evaluation({}, None, solver.best_feedback, {})["sft_result_id"] == model["sft_result_id"]
    # Exercise the complete deferred LearningSolver restore envelope, not only
    # Provider.set_state. The one completed logical fit must never run twice.
    target = build_hf_sft_solver(cfg, dataset,
        resource_context={"device": "cpu", "threads": 1, "namespace": "test.sft"})
    monkeypatch.setattr(target.evaluation_provider, "_fit", lambda *_: pytest.fail("Solver recovery must not fit"))
    previous = solver.get_state()
    target.set_state(previous)
    # LearningSolver's explicit max_steps is an additional per-call budget;
    # zero requests evidence projection only, without authorizing another fit.
    target.fit(max_steps=0)
    assert target.generation == solver.generation == 1
    assert target.evaluation_count == solver.evaluation_count
    assert target.best_model["sft_result_id"] == model["sft_result_id"]
    assert target.evaluation_provider.result(model["sft_result_id"])["bundle"] == output["bundle"]
    assert target.get_incumbent().candidate_token == solver.get_incumbent().candidate_token


def test_cancelled_provider_never_publishes_output(tmp_path):
    dataset, _ = _dataset(tmp_path)
    solver = build_hf_sft_solver(HFSFTConfig("absent"), dataset,
        resource_context={"device": "cpu", "threads": 1})
    solver.evaluation_provider.cancel()
    with pytest.raises(SFTCancelled):
        solver.fit()
    assert solver.evaluation_provider.get_state()["completed"] == {}


def test_4bit_cpu_rejects_before_loading_base_weights(tmp_path):
    folder, _ = _tiny_model(tmp_path)
    dataset, _ = _dataset(tmp_path)
    solver = build_hf_sft_solver(HFSFTConfig(str(folder), load_in_4bit=True,
        target_modules=("c_attn",)), dataset, resource_context={"device": "cpu", "threads": 1})
    with pytest.raises(ValueError, match="granted CUDA"):
        solver.fit()


def test_standard_case_publishes_adapter_with_real_receipt(tmp_path):
    from blackbase.project import CaseExecutor, CaseRunRequest, ExecutionControl
    from blackbase.resources import ArtifactPublisher
    folder, _ = _tiny_model(tmp_path)
    dataset, _ = _dataset(tmp_path)
    project = Path(__file__).resolve().parents[1] / "examples/cases/framework_assistant_sft"
    resource = {"device": "cpu", "threads": 1, "namespace": "sft.publication",
        "metadata": {"artifact_authority": {"backend": "filesystem",
            "namespace": "sft.publication", "root": str(tmp_path / "artifacts"),
            "allow_unsafe_serializers": False, "schema_version": 1}}}
    request = CaseRunRequest(project_name="framework_assistant_sft", stage_name="sft",
        case_name="assistant_sft", case_kind="trainer", mode="build",
        resource_context=resource, control=ExecutionControl.with_timeout(120),
        component_overrides={"model_name_or_path": str(folder), "dataset_path": str(dataset),
            "optimizer_steps": 1, "target_modules": ["c_attn"], "max_length": 64})
    result = CaseExecutor(project).execute(request)
    assert result.ok, result.as_dict()
    assert "best_model" in result.artifact_refs
    receipt = result.artifact_publications["best_model"]
    publisher = ArtifactPublisher.from_resource_context(resource,
        project_run_id=request.identity.project_run_id, case_run_id=request.identity.case_run_id)
    assert publisher.verify_publication(receipt)
    assert result.artifact_refs["best_model"].uri.endswith(".zip")
