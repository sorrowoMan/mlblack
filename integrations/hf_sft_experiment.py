"""Small, explicit SFT experiment CLI over the existing formal Project.

No service, automatic training retries, retrieval implementation or semantic
grader is introduced. Grounding is supplied as frozen external data.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import time
import traceback
import zipfile

from mlblack.integrations.hf_sft import HFSFTConfig, _local_model_digest
from mlblack.pipeline.conversation_sft import load_conversations, tokenize_conversation


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def read_records(path):
    before = file_digest(path)
    conversations = load_conversations(path)
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if file_digest(path) != before:
        raise ValueError(f"dataset changed while reading: {path}")
    result, seen = [], set()
    for index, (row, messages) in enumerate(zip(rows, conversations)):
        identity = row.get("id")
        if not isinstance(identity, str) or not identity.strip() or identity in seen:
            raise ValueError(f"dataset requires unique nonempty id at row {index + 1}: {path}")
        seen.add(identity)
        result.append({"id": identity, "messages": list(messages)})
    return result


def read_grounding(paths):
    result = {}
    for path in paths:
        for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row.get("sample_id", row.get("id"))
            if row.get("id", identity) != identity:
                raise ValueError(f"grounding id aliases disagree: {path}:{number}")
            split, context = row.get("split"), row.get("context")
            if any(not isinstance(value, str) or not value.strip() for value in (identity, split, context)):
                raise ValueError(f"grounding requires sample_id, split and context: {path}:{number}")
            if not isinstance(row.get("source_refs", []), list):
                raise ValueError("grounding source_refs must be a list")
            key = (split, identity)
            if key in result:
                raise ValueError(f"duplicate grounding identity: {key}")
            result[key] = {"context": context, "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                "source_refs": row.get("source_refs", [])}
    return result


def prepare_records(records, split, mode, grounding):
    prepared = []
    for row in records:
        messages = [dict(message) for message in row["messages"]]
        evidence = None
        if mode == "grounded":
            evidence = grounding.get((split, row["id"]))
            if evidence is None:
                raise ValueError(f"missing grounding for split={split!r}, id={row['id']!r}")
            context = "本次任务的冻结框架资料（需按来源核对，不代表已经运行）：\n" + evidence["context"]
            if messages[0]["role"] == "system":
                messages[0]["content"] += "\n\n" + context
            else:
                messages.insert(0, {"role": "system", "content": context})
        prepared.append({"id": row["id"], "messages": messages,
            "prompt_sha256": value_digest(messages[:-1]),
            "reference_sha256": value_digest(messages[-1]), "grounding": evidence})
    return prepared


@contextmanager
def case_environment(config_path, device):
    updates = {"MLBLACK_SFT_CONFIG_PATH": str(config_path), "MLBLACK_SFT_DEVICE": device}
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_project_settings(project):
    name = "mlblack_sft_project_" + hashlib.sha256(str(project).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, Path(project) / "project_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExperimentRecorder:
    def __init__(self, output, run_id):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.run_id, self.started, self.phase = run_id, time.monotonic(), "initializing"

    def save(self, name, value):
        path = self.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    def status(self, phase, **details):
        self.phase = phase
        record = {"schema": "mlblack.sft_experiment_status/v1", "phase": phase,
            "run_id": self.run_id, "pid": os.getpid(), "time_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - self.started, "online_backend_changed": False, **details}
        self.save("status.json", record)
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)


def check_sources(fingerprints):
    for path, expected in fingerprints.items():
        if file_digest(path) != expected:
            raise ValueError(f"frozen experiment input changed: {path}")


def evaluate_model(config, project_settings, project, prepared, *, adapter=None,
                   max_new_tokens=512, generate_splits=(), progress=None):
    """Evaluate all fixed prompt variants with one model load and one L0 grant."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from blackbase.project import ProjectL0Runtime, load_project_runtime_config
    runtime = ProjectL0Runtime(load_project_runtime_config(project_settings), project_root=project)
    lease = runtime.acquire_case("heldout_evaluation", stage_name="evaluation")
    guard = runtime.start_lease_guard(lease)
    resource = runtime.resource_context(lease, case_name="heldout_evaluation", stage_name="evaluation")
    model = base = batch = prompt = generated = None
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(resource.threads)
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path or config.model_name_or_path,
            revision=config.tokenizer_revision or config.revision, local_files_only=True, trust_remote_code=False)
        dtype = torch.float32 if resource.device == "cpu" else (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        base = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, revision=config.revision,
            local_files_only=True, trust_remote_code=False, torch_dtype=dtype).to(resource.device)
        if adapter is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, str(adapter), local_files_only=True)
        else:
            model = base
        model.eval()
        eos_ids = model.generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [tokenizer.eos_token_id])
        template_options = {} if config.enable_thinking is None else {"enable_thinking": config.enable_thinking}
        result = {}
        def clear_question_memory():
            # A high-sequence NLL creates large temporary vocabulary logits.
            # Retaining that allocator cache across generation can force WDDM
            # paging on an 8GB laptop even though the live model is small.
            gc.collect()
            if resource.device.startswith("cuda"):
                with torch.cuda.device(resource.device):
                    torch.cuda.empty_cache()
                return {"allocated_bytes": torch.cuda.memory_allocated(resource.device),
                        "reserved_bytes": torch.cuda.memory_reserved(resource.device)}
            return None
        with torch.no_grad():
            for (split, mode), rows in prepared.items():
                answers, total_loss, total_tokens = [], 0.0, 0
                for row in rows:
                    question_started = time.monotonic()
                    runtime.assert_resource_context_current(resource.as_dict())
                    encoded = tokenize_conversation(tokenizer, row["messages"], max_length=config.max_length,
                        label_mode="completion", enable_thinking=config.enable_thinking)
                    count = sum(token != -100 for token in encoded["labels"][1:])
                    batch = {key: torch.tensor([values], device=resource.device) for key, values in encoded.items()}
                    nll = float(model(**batch).loss.cpu())
                    nll_seconds = time.monotonic() - question_started
                    if not __import__("math").isfinite(nll):
                        raise ValueError("nonfinite evaluation NLL")
                    batch = prompt = generated = None
                    memory_before_generation = clear_question_memory()
                    total_loss += nll * count
                    total_tokens += count
                    item = {"id": row["id"], "prompt": row["messages"][:-1],
                        "reference": row["messages"][-1]["content"], "prompt_sha256": row["prompt_sha256"],
                        "reference_sha256": row["reference_sha256"], "grounding": row["grounding"],
                        "completion_nll": nll, "supervised_tokens": count,
                        "generation_requested": split in generate_splits, "answer": None,
                        "generated_tokens": 0, "ended_eos": None, "hit_generation_limit": None,
                        "nll_seconds": nll_seconds, "generation_seconds": None,
                        "memory_before_generation": memory_before_generation}
                    if split in generate_splits:
                        generation_started = time.monotonic()
                        prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=True,
                            add_generation_prompt=True, return_dict=True, return_tensors="pt", **template_options)
                        prompt = {key: value.to(resource.device) for key, value in prompt.items()}
                        generated = model.generate(**prompt, do_sample=False, max_new_tokens=max_new_tokens,
                            pad_token_id=tokenizer.eos_token_id)
                        ids = generated[0, prompt["input_ids"].shape[1]:].tolist()
                        eos = bool(ids and ids[-1] in eos_ids)
                        item.update(answer=tokenizer.decode(ids, skip_special_tokens=True),
                            generated_tokens=len(ids), ended_eos=eos,
                            hit_generation_limit=len(ids) >= max_new_tokens and not eos,
                            generation_seconds=time.monotonic() - generation_started)
                    batch = prompt = generated = None
                    item["memory_after_question"] = clear_question_memory()
                    item["question_seconds"] = time.monotonic() - question_started
                    answers.append(item)
                    result[f"{split}:{mode}"] = {"split": split, "prompt_mode": mode, "answers": answers,
                        "token_weighted_completion_nll": total_loss / total_tokens,
                        "supervised_tokens": total_tokens, "enable_thinking": config.enable_thinking,
                        "generation": {"do_sample": False, "max_new_tokens": max_new_tokens, "eos_ids": eos_ids},
                        "semantic_quality": "unreviewed; NLL is not a correctness grade"}
                    if progress:
                        progress(split, mode, len(answers), len(rows), result)
        return {"results": result, "resource": resource.as_dict(), "adapter": None if adapter is None else str(adapter)}
    finally:
        model = base = batch = prompt = generated = None
        gc.collect()
        if resource.device.startswith("cuda"):
            torch.cuda.empty_cache()
        torch.set_num_threads(previous_threads)
        guard.close()
        runtime.release(lease)
        runtime.close()


def verified_adapter(run, output):
    from blackbase.resources import ArtifactBinding, ArtifactPublisher
    if not run.ok:
        raise RuntimeError(f"formal SFT Project failed, exit_code={run.exit_code}")
    case = run.case_results[0]
    ref, receipt = case.artifact_refs["best_model"], case.artifact_publications["best_model"]
    publisher = ArtifactPublisher.from_resource_context(case.request.resource_context,
        project_run_id=case.identity.project_run_id, case_run_id=case.identity.case_run_id)
    if publisher is None or not publisher.verify_publication(receipt):
        raise RuntimeError("adapter Artifact publication verification failed")
    payload = publisher.read_verified(ArtifactBinding(ref=ref, publication=receipt))
    folder = Path(output) / "adapter"
    folder.mkdir(exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            if Path(name).name != name or "\\" in name:
                raise ValueError("unexpected nested path in adapter ZIP")
        archive.extractall(folder)
    return folder, {"verified": True, "artifact": ref.as_dict(), "receipt": receipt.as_dict(),
        "bundle_sha256": hashlib.sha256(payload).hexdigest()}


def run_experiment(args):
    project = Path(args.project).resolve()
    if not (project / "project_config.py").is_file():
        raise ValueError("formal Project not found; pass --project explicitly")
    dataset = Path(args.dataset).resolve()
    model = str(Path(args.model).resolve())
    if not Path(model).is_dir():
        raise ValueError("--model must be an existing local HF directory")
    if Path(args.output).resolve().is_relative_to(Path(model)):
        raise ValueError("experiment output must not be inside the frozen base model directory")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not args.run_id.strip():
        raise ValueError("--run-id must not be empty")
    if args.adapter and args.mode != "evaluate":
        raise ValueError("--adapter is an evaluate-only input; it never silently resumes training")
    config = HFSFTConfig(model, revision=args.revision, max_length=args.max_length,
        optimizer_steps=args.optimizer_steps, learning_rate=args.learning_rate,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules=tuple(args.target_modules), label_mode="completion",
        enable_thinking={"off": False, "on": True, "default": None}[args.thinking],
        gradient_checkpointing=args.gradient_checkpointing, seed=args.seed)
    recorder = ExperimentRecorder(args.output, args.run_id)
    try:
        recorder.status("preflight", optimizer_steps_completed=0)
        splits = {split: dataset / f"{split}.jsonl" for split in args.eval_split}
        for value in args.eval_file:
            name, separator, path = value.partition("=")
            if not separator or not name or not path:
                raise ValueError("--eval-file must be SPLIT=PATH")
            splits[name] = Path(path).resolve()
        if not splits:
            raise ValueError("select at least one --eval-split or --eval-file")
        if not set(args.generate_split).issubset(splits):
            raise ValueError("--generate-split must name a selected evaluation split")
        source_paths = [*splits.values(), *map(Path, args.grounding_file), project / "project_config.py",
            Path(__file__), Path(__file__).with_name("hf_sft.py"),
            Path(__file__).resolve().parents[1] / "pipeline/conversation_sft.py"]
        train_path = Path(args.train_file).resolve() if args.train_file else dataset / "train.jsonl"
        training = args.mode == "train-eval" or args.check_train
        if training:
            source_paths.append(train_path)
        if (dataset / "manifest.json").is_file():
            source_paths.append(dataset / "manifest.json")
        fingerprints = {str(path.resolve()): file_digest(path) for path in source_paths}
        grounding = read_grounding(args.grounding_file)
        modes = tuple(args.prompt_modes)
        after_modes = tuple(args.after_prompt_modes or modes)
        prepared = {(split, mode): prepare_records(read_records(path), split, mode, grounding)
            for split, path in splits.items() for mode in modes}
        after_prepared = {(split, mode): prepare_records(read_records(path), split, mode, grounding)
            for split, path in splits.items() for mode in after_modes}
        train_rows = prepare_records(read_records(train_path), "train", args.training_prompt_mode, grounding) if training else []
        model_digest = _local_model_digest(model)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, revision=config.revision,
            local_files_only=True, trust_remote_code=False)
        lengths = {}
        for key, rows in {**prepared, **after_prepared, **({("train", args.training_prompt_mode): train_rows} if training else {})}.items():
            lengths[":".join(key)] = [len(tokenize_conversation(tokenizer, row["messages"],
                max_length=config.max_length, label_mode="completion", enable_thinking=config.enable_thinking)["input_ids"]) for row in rows]
        prepared_path = recorder.output / "prepared" / "train.jsonl"
        if training:
            prepared_path.parent.mkdir()
            prepared_path.write_text("".join(json.dumps({"id": row["id"], "messages": row["messages"]},
                ensure_ascii=False) + "\n" for row in train_rows), encoding="utf-8")
            recorder.save("prepared/manifest.json", {"schema": "mlblack.sft_prepared_dataset/v1",
                "source_fingerprints": fingerprints, "prompt_mode": args.training_prompt_mode,
                "grounding": {row["id"]: row["grounding"] for row in train_rows},
                "train_sha256": file_digest(prepared_path), "enable_thinking": config.enable_thinking})
        config_path = recorder.output / "case-config.json"
        recorder.save("case-config.json", {**asdict(config), "dataset_path": str(prepared_path)})
        check_sources(fingerprints)
        if _local_model_digest(model) != model_digest:
            raise ValueError("base model files changed during tokenizer preflight")
        tokenizer_template = getattr(tokenizer, "chat_template", None)
        recorder.save("experiment.json", {"schema": "mlblack.sft_experiment/v1", "arguments": vars(args),
            "config": asdict(config), "source_fingerprints": fingerprints, "local_base_sha256": model_digest,
            "token_lengths": lengths,
            "chat_template_sha256": hashlib.sha256(tokenizer_template.encode("utf-8")).hexdigest(),
            "online_backend_changed": False, "semantic_review": "separate; not automatically scored"})
        if args.mode == "preflight":
            recorder.status("preflight_completed", optimizer_steps_completed=0, model_loaded=False,
                token_length_checked=True)
            return 0
        with case_environment(config_path, args.device):
            settings = load_project_settings(project)
            def evaluate(label, source, rows):
                check_sources(fingerprints)
                if _local_model_digest(model) != model_digest:
                    raise ValueError("base model files changed")
                recorder.status(label, optimizer_steps_completed=(config.optimizer_steps if label == "after" else 0))
                def progress(split, mode, completed, total, partial):
                    recorder.save(label + ".partial.json", partial)
                    recorder.status(label, split=split, prompt_mode=mode, items_completed=completed, items_total=total)
                result = evaluate_model(config, settings, project, rows, adapter=source,
                    max_new_tokens=args.max_new_tokens, generate_splits=args.generate_split, progress=progress)
                check_sources(fingerprints)
                if _local_model_digest(model) != model_digest:
                    raise ValueError("base model files changed during evaluation")
                recorder.save(label + ".json", result)
                return result
            before = evaluate("base", None, prepared)
            summary = {"base": {key: value["token_weighted_completion_nll"] for key, value in before["results"].items()}}
            if args.mode == "evaluate" and args.adapter:
                adapter = Path(args.adapter).resolve()
                if not (adapter / "adapter_config.json").is_file():
                    raise ValueError("--adapter must be an existing PEFT adapter directory")
                adapter_digest = _local_model_digest(adapter)
                adapter_manifest_path = adapter / "sft_manifest.json"
                adapter_manifest = json.loads(adapter_manifest_path.read_text(encoding="utf-8")) if adapter_manifest_path.is_file() else None
                if adapter_manifest and adapter_manifest.get("local_base_sha256") != model_digest:
                    raise ValueError("existing adapter was trained against another base source digest")
                recorder.save("existing-adapter.json", {"directory": str(adapter), "sha256": adapter_digest,
                    "training_manifest": adapter_manifest, "authority_receipt_verified": False})
                result = evaluate("existing_adapter", adapter, after_prepared)
                if _local_model_digest(adapter) != adapter_digest:
                    raise ValueError("existing adapter files changed during evaluation")
                summary["existing_adapter"] = {key: value["token_weighted_completion_nll"] for key, value in result["results"].items()}
            if args.mode == "train-eval":
                from mlblack.project.project_runner import execute_project
                recorder.status("training", planned_optimizer_steps=config.optimizer_steps, optimizer_steps_completed=None)
                check_sources(fingerprints)
                run = execute_project(project, record=True, run_id=args.run_id)
                recorder.save("project-result.json", run.as_dict())
                adapter, verification = verified_adapter(run, recorder.output)
                recorder.save("publication-verification.json", verification)
                result = evaluate("after", adapter, after_prepared)
                summary["after"] = {key: value["token_weighted_completion_nll"] for key, value in result["results"].items()}
                summary["publication_verified"] = True
            recorder.save("summary.json", {"metrics": summary, "semantic_quality": "unreviewed",
                "meaning": "NLL and generation termination are observations, not semantic correctness grades"})
        recorder.status("completed", optimizer_steps_completed=config.optimizer_steps if args.mode == "train-eval" else 0)
        return 0
    except BaseException as exc:
        previous = recorder.phase
        recorder.save("failure.json", {"failed_phase": previous, "type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(), "automatic_retry": False})
        recorder.status("failed", failed_phase=previous, error_type=type(exc).__name__, error=str(exc))
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "evaluate", "train-eval"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    default_project = Path(__file__).resolve().parents[1] / "examples/cases/framework_assistant_sft"
    parser.add_argument("--project", default=str(default_project))
    parser.add_argument("--revision")
    parser.add_argument("--adapter")
    parser.add_argument("--train-file")
    parser.add_argument("--check-train", action="store_true")
    parser.add_argument("--eval-split", action="append", default=[])
    parser.add_argument("--eval-file", action="append", default=[])
    parser.add_argument("--generate-split", action="append", default=[])
    parser.add_argument("--grounding-file", action="append", default=[])
    parser.add_argument("--prompt-modes", nargs="+", choices=("plain", "grounded"), default=["plain"])
    parser.add_argument("--after-prompt-modes", nargs="+", choices=("plain", "grounded"))
    parser.add_argument("--training-prompt-mode", choices=("plain", "grounded"), default="plain")
    parser.add_argument("--thinking", choices=("off", "on", "default"), default="off")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--optimizer-steps", type=int, default=120)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    return run_experiment(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
