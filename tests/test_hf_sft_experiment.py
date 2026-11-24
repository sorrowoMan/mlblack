from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
from uuid import uuid4
import pytest

from mlblack.integrations.hf_sft import HFSFTConfig, HFSFTProvider
from mlblack.integrations.hf_sft_experiment import (ExperimentRecorder, build_parser,
    case_environment, main, prepare_records, read_grounding, read_records)
from mlblack.pipeline.conversation_sft import tokenize_conversation


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _rows():
    return [{"id": "one", "messages": [{"role": "system", "content": "hello"},
        {"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}]


def test_grounding_is_matched_by_split_and_id_without_changing_reference(tmp_path):
    data, ground = tmp_path / "data.jsonl", tmp_path / "grounding.jsonl"
    _write_rows(data, _rows())
    _write_rows(ground, [{"sample_id": "one", "split": "test", "context": "facts",
                         "source_refs": [{"source": "file.py"}]}])
    original, grounding = read_records(data), read_grounding([ground])
    plain = prepare_records(original, "test", "plain", grounding)[0]
    selected = prepare_records(original, "test", "grounded", grounding)[0]
    assert "facts" not in plain["messages"][0]["content"]
    assert "facts" in selected["messages"][0]["content"]
    assert selected["messages"][-1] == original[0]["messages"][-1]
    assert selected["reference_sha256"] == plain["reference_sha256"]
    assert selected["prompt_sha256"] != plain["prompt_sha256"]
    with pytest.raises(ValueError, match="missing grounding"):
        prepare_records(original, "validation", "grounded", grounding)


def test_grounding_duplicate_and_id_alias_conflicts_fail(tmp_path):
    ground = tmp_path / "grounding.jsonl"
    row = {"sample_id": "one", "split": "test", "context": "facts"}
    _write_rows(ground, [row, row])
    with pytest.raises(ValueError, match="duplicate grounding"):
        read_grounding([ground])
    _write_rows(ground, [{**row, "id": "other"}])
    with pytest.raises(ValueError, match="aliases disagree"):
        read_grounding([ground])


@pytest.mark.parametrize("rows", [[{"messages": _rows()[0]["messages"]}], _rows() * 2])
def test_explicit_unique_sample_ids_are_required(tmp_path, rows):
    data = tmp_path / "data.jsonl"
    _write_rows(data, rows)
    with pytest.raises(ValueError, match="unique nonempty id"):
        read_records(data)


def test_recorder_rejects_overwrite_and_environment_is_restored(tmp_path, monkeypatch):
    ExperimentRecorder(tmp_path / "run", "first")
    with pytest.raises(FileExistsError):
        ExperimentRecorder(tmp_path / "run", "second")
    monkeypatch.setenv("MLBLACK_SFT_DEVICE", "original")
    monkeypatch.delenv("MLBLACK_SFT_CONFIG_PATH", raising=False)
    with case_environment(tmp_path / "config.json", "cpu"):
        assert os.environ["MLBLACK_SFT_DEVICE"] == "cpu"
        assert os.environ["MLBLACK_SFT_CONFIG_PATH"].endswith("config.json")
    assert os.environ["MLBLACK_SFT_DEVICE"] == "original"
    assert "MLBLACK_SFT_CONFIG_PATH" not in os.environ


def test_old_provider_config_keeps_implicit_thinking_semantics(tmp_path):
    data = tmp_path / "data.jsonl"
    _write_rows(data, _rows())
    source = HFSFTProvider(HFSFTConfig("absent"), data)
    state = source.get_state()
    state["config"].pop("enable_thinking")
    HFSFTProvider(HFSFTConfig("absent"), data).set_state(state)
    with pytest.raises(ValueError, match="config differs"):
        HFSFTProvider(HFSFTConfig("absent", enable_thinking=False), data).set_state(state)


def test_one_thinking_setting_reaches_both_completion_boundaries():
    calls = []
    class Tokenizer:
        chat_template = "example"
        def apply_chat_template(self, messages, **kwargs):
            calls.append(kwargs)
            return {"input_ids": [1, 2] if kwargs["add_generation_prompt"] else [1, 2, 3]}
    row = tokenize_conversation(Tokenizer(), _rows()[0]["messages"], max_length=20,
        label_mode="completion", enable_thinking=False)
    assert row["labels"] == [-100, -100, 3]
    assert len(calls) == 2 and all(call["enable_thinking"] is False for call in calls)


@pytest.mark.parametrize("field,value", [("optimizer_steps", True), ("optimizer_steps", 1.5),
    ("optimizer_steps", "1"), ("enable_thinking", 0)])
def test_project_json_configuration_preserves_strict_value_types(tmp_path, monkeypatch, field, value):
    path = Path(__file__).resolve().parents[1] / "examples/cases/framework_assistant_sft/cases/assistant_sft/config.py"
    spec = importlib.util.spec_from_file_location("sft_case_config_strict_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    override = tmp_path / "case.json"
    override.write_text(json.dumps({field: value}), encoding="utf-8")
    monkeypatch.setenv("MLBLACK_SFT_CONFIG_PATH", str(override))
    settings = module.settings()
    settings.pop("dataset_path")
    with pytest.raises(ValueError, match=field):
        HFSFTConfig(**settings)


def test_cpu_tiny_cli_preflight_train_evaluate_and_fixed_prompt_pairs(tmp_path):
    from test_hf_sft import _tiny_model
    model, _ = _tiny_model(tmp_path)
    data = tmp_path / "dataset"
    data.mkdir()
    _write_rows(data / "train.jsonl", _rows())
    _write_rows(data / "validation.jsonl", _rows())
    _write_rows(data / "test.jsonl", _rows())
    ground = data / "grounding.jsonl"
    _write_rows(ground, [{"sample_id": "one", "split": split, "context": "hello", "source_refs": []}
                         for split in ("train", "validation", "test")])
    common = ["--dataset", str(data), "--model", str(model),
        "--eval-split", "validation", "--eval-split", "test", "--generate-split", "test",
        "--prompt-modes", "plain", "grounded", "--training-prompt-mode", "grounded",
        "--grounding-file", str(ground), "--max-length", "64", "--max-new-tokens", "3",
        "--target-modules", "c_attn", "--optimizer-steps", "1", "--device", "cpu"]
    preflight = tmp_path / "preflight"
    assert main(["preflight", *common, "--check-train", "--output", str(preflight), "--run-id", "tiny-preflight"]) == 0
    assert json.loads((preflight / "status.json").read_text())["model_loaded"] is False
    run = tmp_path / "run"
    assert main(["train-eval", *common, "--output", str(run), "--run-id", "tiny-sft-cli-" + uuid4().hex[:10]]) == 0
    assert (run / "publication-verification.json").is_file()
    experiment = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    training_manifest = json.loads((run / "adapter/sft_manifest.json").read_text(encoding="utf-8"))
    assert experiment["chat_template_sha256"] == training_manifest["chat_template_sha256"]
    assert any(path.endswith("hf_sft_experiment.py") for path in experiment["source_fingerprints"])
    base = json.loads((run / "base.json").read_text(encoding="utf-8"))
    after = json.loads((run / "after.json").read_text(encoding="utf-8"))
    assert set(base["results"]) == set(after["results"]) == {
        "validation:plain", "validation:grounded", "test:plain", "test:grounded"}
    for key in base["results"]:
        before_row, after_row = base["results"][key]["answers"][0], after["results"][key]["answers"][0]
        assert before_row["prompt_sha256"] == after_row["prompt_sha256"]
        assert before_row["reference_sha256"] == after_row["reference_sha256"]
        assert base["results"][key]["enable_thinking"] is False
        assert before_row["generation_requested"] == key.startswith("test:")
        assert before_row["nll_seconds"] >= 0 and before_row["question_seconds"] >= 0
        assert before_row["memory_after_question"] is None
    evaluation = tmp_path / "evaluation"
    assert main(["evaluate", *common, "--adapter", str(run / "adapter"), "--output", str(evaluation),
                 "--run-id", "tiny-eval-only"]) == 0
    assert not (evaluation / "project-result.json").exists()
    assert json.loads((evaluation / "status.json").read_text())["optimizer_steps_completed"] == 0
