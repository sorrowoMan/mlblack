"""Optional, real Hugging Face + PEFT fit provider over the shared Solver.

One Solver evaluation owns one complete finite SFT fit. Optimizer steps are
internal third-party fit work, not individually rollbackable Solver generations.
The base model, optimizer, tokenizer and tensors never enter candidate vectors.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import io
import json
from numbers import Real
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
from typing import Any, Mapping
from uuid import uuid4
import zipfile

import numpy as np
from blackbase.evaluation import (EvaluationGateway, EvaluationProviderRegistry,
    EvaluationProviderSpec, EvaluationRequest, EvaluationResult)
from blackbase.resources import ArtifactSerializer, ResourceContext, ResourceRequirement
from mlblack.core import Feedback, LearningProblem, ModelRepresentation, UnknownState
from mlblack.core.artifact_provider import CaseRuntimeArtifactProvider
from mlblack.core.artifacts import TypedModelArtifact
from mlblack.pipeline.conversation_sft import (collate_conversations,
    conversation_digest, load_conversations, tokenize_conversation)


@dataclass(frozen=True)
class HFSFTConfig:
    model_name_or_path: str
    revision: str | None = None
    tokenizer_name_or_path: str | None = None
    tokenizer_revision: str | None = None
    max_length: int = 2048
    optimizer_steps: int = 20
    batch_size: int = 1
    learning_rate: float = 2e-4
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    label_mode: str = "assistant"
    load_in_4bit: bool = False
    local_files_only: bool = True
    gradient_checkpointing: bool = False
    enable_thinking: bool | None = None
    seed: int = 42

    def __post_init__(self):
        if not isinstance(self.model_name_or_path, str) or not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path is required")
        for field in ("max_length", "optimizer_steps", "batch_size", "lora_rank", "lora_alpha"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if self.label_mode not in {"assistant", "completion"}:
            raise ValueError("label_mode must be assistant or completion")
        if (isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, Real)
                or not np.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError("learning_rate must be finite and positive")
        if (isinstance(self.lora_dropout, bool) or not isinstance(self.lora_dropout, Real)
                or not np.isfinite(self.lora_dropout) or not 0 <= self.lora_dropout < 1):
            raise ValueError("lora_dropout must be finite in [0, 1)")
        if (not isinstance(self.target_modules, (tuple, list)) or not self.target_modules
                or any(not isinstance(item, str) or not item.strip() for item in self.target_modules)):
            raise ValueError("target_modules must be a nonempty sequence of module names")
        for field in ("load_in_4bit", "local_files_only", "gradient_checkpointing"):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"{field} must be a boolean")
        if self.enable_thinking is not None and not isinstance(self.enable_thinking, bool):
            raise ValueError("enable_thinking must be a boolean or None")
        for field in ("revision", "tokenizer_name_or_path", "tokenizer_revision"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field} must be a nonempty string or None")
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
        object.__setattr__(self, "lora_dropout", float(self.lora_dropout))


def preflight_hf_sft(config: HFSFTConfig, dataset_path) -> dict[str, Any]:
    """Validate files/dependency availability only; never import/load/download HF.

    Token-length fit cannot be claimed until the actual tokenizer is loaded.
    Missing optional packages and model files are reported, not auto-installed.
    """
    records = load_conversations(dataset_path)
    packages = ("torch", "transformers", "peft", "accelerate", "safetensors")
    if config.load_in_4bit:
        packages += ("bitsandbytes",)
    dependencies = {name: importlib.util.find_spec(name) is not None for name in packages}
    local_model = Path(config.model_name_or_path).is_dir()
    return {
        "schema": "mlblack.hf_sft_preflight/v1", "config": asdict(config),
        "records": len(records), "dataset_sha256": conversation_digest(records),
        "dependencies": dependencies, "local_model_directory_exists": local_model,
        "model_loaded": False, "download_attempted": False,
        "token_length_checked": False, "token_length_note": "requires actual tokenizer; no truncation permitted",
        "ready_to_attempt": all(dependencies.values()) and (local_model or not config.local_files_only),
    }


class SFTCancelled(RuntimeError):
    """Cooperative cancellation discards the complete uncommitted fit output."""


class HFSFTProvider:
    """A finite fit backend; no private Solver/Project lifecycle or global lease.

    Fresh fit work is isolated per evaluation. Failed/cancelled work is never
    published. get_state restores completed adapter bundles only; there is no
    mid-fit optimizer checkpoint/resume promise.
    """
    problem_id = "mlblack.hf_sft.fit/v1"
    request_capabilities = ("ml.sft.causal_lm.fit",)

    def __init__(self, config: HFSFTConfig, dataset_path):
        self.config, self.dataset_path = config, str(Path(dataset_path).resolve())
        self.dataset_file_digest = hashlib.sha256(Path(self.dataset_path).read_bytes()).hexdigest()
        self.dataset_manifest_digest = _sibling_manifest_digest(self.dataset_path)
        self.records = load_conversations(dataset_path)
        if hashlib.sha256(Path(self.dataset_path).read_bytes()).hexdigest() != self.dataset_file_digest:
            raise ValueError("SFT dataset changed while it was loaded")
        self.dataset_digest = conversation_digest(self.records)
        self.spec = EvaluationProviderSpec(
            provider_id="mlblack.hf_sft.peft/v1", problem_ids=(self.problem_id,),
            capabilities=self.request_capabilities,
            resource_requirement=ResourceRequirement(threads=1, resource_backend="any"),
            supported_devices=("cpu", "gpu"), preferred_devices=("gpu", "cpu"),
            compute_backend="torch", modes=("train",),
            metadata={"fit_boundary": "one_complete_fit", "optimizer_resume": False},
        )
        self._completed: dict[str, dict[str, Any]] = {}
        self._cancel = Event()
        self._fit_lock = Lock()
        self._runtime_checkpoint = None

    def cancel(self):
        self._cancel.set()

    def evaluate(self, request, binding):
        if request.problem_id != self.problem_id or binding.provider_id != self.spec.provider_id:
            raise ValueError("SFT request/binding owner mismatch")
        if request.mode != "train" or len(request.states) != 1:
            raise ValueError("SFT provider requires one train request")
        state = request.states[0]
        if not isinstance(state, UnknownState) or state.metadata.get("dataset_sha256") != self.dataset_digest:
            raise ValueError("SFT fit request is bound to another dataset")
        if not self._fit_lock.acquire(blocking=False):
            raise RuntimeError("SFT provider is already fitting")
        try:
            self._check_cancelled()
            self._assert_dataset_unchanged()
            payload = self._fit(binding)
            self._assert_dataset_unchanged()
            self._check_cancelled()
            result_id = uuid4().hex
            self._completed[result_id] = payload
            manifest = payload["manifest"]
            feedback = Feedback(
                objectives=[manifest["mean_training_loss"]], loss=manifest["mean_training_loss"],
                metrics={"training_loss": manifest["mean_training_loss"],
                         "optimizer_steps": self.config.optimizer_steps},
                info={"sft_result_id": result_id, "dataset_sha256": self.dataset_digest,
                      "fit_boundary": "one_complete_fit", "optimizer_resume": False},
            )
            return EvaluationResult(request_id=request.request_id, feedback=(feedback,), binding=binding)
        finally:
            self._fit_lock.release()

    def _check_cancelled(self):
        if self._runtime_checkpoint is not None:
            self._runtime_checkpoint()
        if self._cancel.is_set():
            raise SFTCancelled("SFT fit cancelled; no partial model is published")

    def _assert_dataset_unchanged(self):
        if (hashlib.sha256(Path(self.dataset_path).read_bytes()).hexdigest() != self.dataset_file_digest
                or _sibling_manifest_digest(self.dataset_path) != self.dataset_manifest_digest):
            raise ValueError("SFT dataset or source manifest changed after Provider construction")

    def _fit(self, binding):
        try:
            import torch
            import transformers
            import peft
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError("HFSFTProvider requires optional dependencies: pip install mlblack[sft]") from exc
        cfg = self.config
        resource = binding.resource_context
        device = str(binding.device)
        if device == "gpu":
            tokens = tuple(resource.grant.get("device_tokens", ()))
            if resource.device.startswith("cuda"):
                device = resource.device
            elif len(tokens) == 1 and str(tokens[0]).startswith("cuda"):
                device = str(tokens[0])
            else:
                raise ValueError("GPU fit requires an explicit granted cuda device token")
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError(f"unsupported granted SFT device: {device}")
        if cfg.load_in_4bit and not device.startswith("cuda"):
            raise ValueError("real bitsandbytes 4-bit SFT requires a granted CUDA device")
        # Tokenize and verify all supervision before loading expensive weights.
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_name_or_path or cfg.model_name_or_path,
            revision=cfg.tokenizer_revision or cfg.revision,
            local_files_only=cfg.local_files_only, trust_remote_code=False,
        )
        rows = [tokenize_conversation(tokenizer, row, max_length=cfg.max_length,
                 label_mode=cfg.label_mode, enable_thinking=cfg.enable_thinking) for row in self.records]
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer needs pad_token_id or eos_token_id")
            tokenizer.pad_token = tokenizer.eos_token
        kwargs = {"revision": cfg.revision, "local_files_only": cfg.local_files_only,
                  "trust_remote_code": False}
        if device.startswith("cuda") and not cfg.load_in_4bit:
            # torch_dtype remains accepted by Transformers 4.51 and 5.x;
            # dtype= is a newer spelling and cannot justify a 4.51 floor.
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs.update(quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
                device_map={"": device})
        # RNG state is restored when the finite backend call finishes.
        cuda_devices = [] if device == "cpu" else [torch.device(device).index or 0]
        previous_threads = torch.get_num_threads()
        model = optimizer = None
        base_source_digest = _local_model_digest(cfg.model_name_or_path)
        try:
            torch.set_num_threads(resource.threads)
            with torch.random.fork_rng(devices=cuda_devices):
                _seed_fit_rng(torch, cfg.seed, cuda_devices)
                base = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **kwargs)
                if _local_model_digest(cfg.model_name_or_path) != base_source_digest:
                    raise ValueError("local base model changed while loading")
                if cfg.load_in_4bit:
                    from peft import prepare_model_for_kbit_training
                    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=cfg.gradient_checkpointing)
                else:
                    base.to(device)
                model = get_peft_model(base, LoraConfig(
                    r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                    target_modules=list(cfg.target_modules), task_type=TaskType.CAUSAL_LM,
                    bias="none",
                ))
                if cfg.gradient_checkpointing:
                    model.gradient_checkpointing_enable()
                    model.enable_input_require_grads()
                if hasattr(model.config, "use_cache"):
                    model.config.use_cache = False
                parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
                if not parameters:
                    raise ValueError("LoRA has no trainable parameters")
                optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate)
                initial = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
                losses = []
                model.train()
                for step in range(cfg.optimizer_steps):
                    self._check_cancelled()
                    batch_rows = [rows[(step * cfg.batch_size + i) % len(rows)] for i in range(cfg.batch_size)]
                    batch = {key: value.to(device) for key, value in collate_conversations(
                        batch_rows, pad_token_id=tokenizer.pad_token_id, torch_module=torch).items()}
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(**batch).loss
                    if not torch.isfinite(loss):
                        raise ValueError("non-finite SFT loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                self._check_cancelled()
                changed = sum(not torch.equal(initial[name], p.detach().cpu())
                              for name, p in model.named_parameters() if p.requires_grad)
                if not changed:
                    raise RuntimeError("SFT completed without changing any LoRA weights")
                manifest = {
                    "schema": "mlblack.hf_sft_adapter/v1", "config": asdict(cfg),
                    "base_model": cfg.model_name_or_path,
                    "requested_revision": cfg.revision,
                    "resolved_revision": getattr(base.config, "_commit_hash", None),
                    "local_base_sha256": base_source_digest,
                    "dataset_sha256": self.dataset_digest, "records": len(rows),
                    "dataset_file_sha256": self.dataset_file_digest,
                    "dataset_manifest_sha256": self.dataset_manifest_digest,
                    "dataset_source_frozen_at": "provider_construction",
                    "token_lengths": [len(row["input_ids"]) for row in rows],
                    "supervised_tokens": sum(sum(v != -100 for v in row["labels"][1:]) for row in rows),
                    "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                    "chat_template": tokenizer.chat_template,
                    "mean_training_loss": float(np.mean(losses)),
                    "trainable_parameters": sum(p.numel() for p in parameters),
                    "total_parameters": sum(p.numel() for p in model.parameters()),
                    "changed_adapter_tensors": changed,
                    "quantization": "bitsandbytes_nf4" if cfg.load_in_4bit else "none",
                    "resource": resource.as_dict(), "execution_device": device,
                    "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
                    "fit_boundary": "one_complete_fit", "optimizer_resume": False,
                    "quality_claim": "mechanism_only; no framework quality evaluation implied",
                }
                with TemporaryDirectory(prefix="mlblack-sft-") as temporary:
                    folder = Path(temporary)
                    model.save_pretrained(folder, safe_serialization=True)
                    tokenizer.save_pretrained(folder)
                    (folder / "sft_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                    files = tuple(sorted(p for p in folder.iterdir() if p.is_file()))
                    if not (folder / "adapter_model.safetensors").is_file():
                        raise RuntimeError("PEFT did not export adapter-only safetensors")
                    output = io.BytesIO()
                    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                        for file in files:
                            archive.write(file, arcname=file.name)
                    if _local_model_digest(cfg.model_name_or_path) != base_source_digest:
                        raise ValueError("local base model changed during SFT fit")
                    return {"manifest": manifest, "bundle": output.getvalue()}
        finally:
            optimizer = model = None
            torch.set_num_threads(previous_threads)

    def result(self, result_id):
        if result_id not in self._completed:
            raise ValueError("completed SFT result is absent; restore requires its completed provider checkpoint")
        return self._completed[result_id]

    def get_state(self):
        if self._fit_lock.locked():
            raise RuntimeError("mid-fit SFT checkpoint is unsupported")
        return {"schema": "mlblack.hf_sft_provider/v1", "dataset_sha256": self.dataset_digest,
                "config": asdict(self.config), "completed": dict(self._completed)}

    def set_state(self, state):
        if self._fit_lock.locked():
            raise RuntimeError("cannot restore a running SFT provider")
        if state.get("schema") != "mlblack.hf_sft_provider/v1" or state.get("dataset_sha256") != self.dataset_digest:
            raise ValueError("SFT provider checkpoint/dataset mismatch")
        restored_config = dict(state.get("config") or {})
        # Older v1 snapshots used tokenizer defaults implicitly. Never reinterpret
        # those weights as an explicitly enabled/disabled thinking template.
        restored_config.setdefault("enable_thinking", None)
        if json.dumps(restored_config, sort_keys=True) != json.dumps(asdict(self.config), sort_keys=True):
            raise ValueError("SFT provider config differs from checkpoint")
        self._completed = dict(state["completed"])
        self._cancel.clear()


def _local_model_digest(path):
    folder = Path(path)
    if not folder.is_dir():
        return None
    digest = hashlib.sha256()
    for file in sorted(p for p in folder.rglob("*") if p.is_file() and ".cache" not in p.relative_to(folder).parts):
        digest.update(file.relative_to(folder).as_posix().encode())
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _seed_fit_rng(torch_module, seed, cuda_devices):
    """Seed only generators whose states are captured by this finite fit.

    torch.manual_seed seeds every visible CUDA device, including devices that
    this resource grant does not own. CPU-only fit must not initialize CUDA.
    """
    torch_module.random.default_generator.manual_seed(seed)
    for index in cuda_devices:
        torch_module.cuda.default_generators[index].manual_seed(seed)


def _sibling_manifest_digest(path):
    manifest = Path(path).parent / "manifest.json"
    return hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None


class HFSFTRepresentation(ModelRepresentation):
    """A scalar fit-request candidate; never a flattened model parameter vector."""
    name, dimension = "hf_sft_request", 1
    context_requires = ()
    context_optional = ()
    context_provides = ("candidate.model",)
    context_mutates = ()

    def __init__(self, dataset_digest):
        self.dataset_digest = dataset_digest

    def init(self, context):
        return UnknownState([0.0], metadata={"kind": "hf_sft_fit_request", "dataset_sha256": self.dataset_digest})

    def decode(self, state, context=None):
        if state.metadata.get("kind") != "hf_sft_fit_request" or state.metadata.get("dataset_sha256") != self.dataset_digest:
            raise ValueError("SFT candidate is not a request for the configured dataset")
        return {"kind": "hf_sft_fit_request", "dataset_sha256": self.dataset_digest}


class HFSFTLearningProblem(LearningProblem):
    name, backend_requires = "hf_sft_training_loss", ()
    context_requires = ("candidate.unknown_state", "resource.context")
    context_optional = ()
    thread_safe_evaluation = False

    def __init__(self, provider):
        self.provider = provider
        registry = EvaluationProviderRegistry()
        registry.register(provider)
        self.gateway = EvaluationGateway(registry)

    def evaluate(self, model, state, context):
        request = EvaluationRequest(problem_id=self.provider.problem_id, states=(state,),
            mode="train", capabilities=self.provider.request_capabilities)
        result = self.gateway.evaluate(request, ResourceContext.from_mapping(context["resource.context"]))
        feedback = result.feedback[0]
        model["sft_result_id"] = feedback.info["sft_result_id"]
        return feedback

    def restore_model_for_evaluation(self, model, state, feedback, context):
        result_id = feedback.info["sft_result_id"]
        self.provider.result(result_id)
        return {**model, "sft_result_id": result_id}

    def build_model_artifact(self, model, context=None):
        result = self.provider.result(model["sft_result_id"])
        return TypedModelArtifact(name="hf_sft_lora", model=result["bundle"],
            family="causal_lm", artifact_type="peft_lora_bundle",
            metadata={"manifest": result["manifest"]})


def sft_bundle_serializer(value):
    if not isinstance(value, bytes):
        raise TypeError("SFT model artifact must be a completed adapter bundle")
    return ArtifactSerializer(name="mlblack_hf_sft_zip", extension=".zip",
        media_type="application/zip", dump=lambda payload, path: Path(path).write_bytes(payload))


def build_hf_sft_solver(config: HFSFTConfig, dataset_path, *, resource_context=None):
    """Assemble lazily: build/Doctor do not import HF, allocate GPU or train."""
    from mlblack.integrations.nsgablack_control import build_learning_solver
    from nsgablack.adapters import FixedCandidateAdapter
    provider = HFSFTProvider(config, dataset_path)
    solver = build_learning_solver(
        problem=HFSFTLearningProblem(provider), representation=HFSFTRepresentation(provider.dataset_digest),
        adapter=FixedCandidateAdapter(), resource_context=resource_context,
        artifact_provider=CaseRuntimeArtifactProvider(serializer_selector=sft_bundle_serializer),
        run_name="framework_assistant_sft",
    )
    solver.evaluation_provider = provider
    # Bind the ordinary Case cancellation/deadline without copying runtime
    # services or live model objects into checkpointable request contexts.
    def runtime_checkpoint():
        runtime = getattr(solver, "case_runtime", None)
        if runtime is not None:
            runtime.checkpoint()
    provider._runtime_checkpoint = runtime_checkpoint
    solver.set_max_steps(1)
    return solver


__all__ = ["HFSFTConfig", "HFSFTProvider", "HFSFTRepresentation", "HFSFTLearningProblem",
    "SFTCancelled", "preflight_hf_sft", "build_hf_sft_solver", "sft_bundle_serializer"]
