from __future__ import annotations

from typing import Any

import numpy as np

from mlblack.backends.contracts import BackendCapabilityContract


class TorchLossesCapability:
    contract = BackendCapabilityContract(
        backend="torch",
        capability="losses",
        provides=(
            "loss.cross_entropy",
            "loss.lm_next_token",
            "loss.dpo",
            "loss.triplet",
            "loss.gaussian_nll",
            "metrics.classification",
        ),
        methods={
            "loss.cross_entropy": "classification_loss(model_or_output, labels, head_name) -> (loss, logits)",
            "loss.lm_next_token": "lm_loss(model, tokens, head_name) -> loss",
            "loss.dpo": "dpo_loss_and_metrics(model, chosen, rejected, head_name, beta, reference_model, prefix) -> (loss, metrics)",
            "loss.triplet": "triplet_loss(anchor, positive, negative, margin) -> (loss, metrics)",
            "loss.gaussian_nll": "gaussian_nll(output, target, head_name) -> (loss, mean, scale)",
            "metrics.classification": "classification_metrics(logits, labels, prefix) -> dict",
        },
        tensor_kinds=("torch.Tensor",),
        heads=("classification", "language_modeling", "preference_dpo", "retrieval"),
        supports_autograd=True,
        notes="Task losses and metrics over torch tensors for neural problems.",
    )

    def __init__(self, tensor: Any) -> None:
        self.tensor = tensor

    def torch(self) -> Any:
        return self.tensor.torch()

    def classification_loss(self, model: Any, input_ids: Any, labels: Any, head_name: str) -> tuple[Any, Any]:
        output = model(input_ids)
        return self.generic_classification_loss(output, labels, head_name)

    def generic_classification_loss(self, output: dict[str, Any], labels: Any, head_name: str) -> tuple[Any, Any]:
        torch = self.torch()
        logits = output.get("head_outputs", {}).get(head_name, output.get("logits"))
        if logits is None:
            raise ValueError(f"classification head output is missing: {head_name}")
        return torch.nn.functional.cross_entropy(logits, labels), logits

    def embedding_from_output(self, output: dict[str, Any], head_name: str) -> Any:
        embedding = output.get("head_outputs", {}).get(head_name, output.get("embeddings"))
        if embedding is None:
            raise ValueError(f"embedding/retrieval head output is missing: {head_name}")
        return embedding

    def classification_metrics(self, logits: Any, labels: Any, *, prefix: str) -> dict[str, float]:
        torch = self.torch()
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
        accuracy = float(torch.mean((pred == labels).to(dtype=torch.float32)).detach().cpu().item())
        loss = float(torch.nn.functional.cross_entropy(logits, labels).detach().cpu().item())
        return {f"{prefix}.accuracy": accuracy, f"{prefix}.error_rate": 1.0 - accuracy, f"{prefix}.log_loss": loss}

    def lm_loss(self, model: Any, tokens: Any, head_name: str) -> Any:
        torch = self.torch()
        if tokens.shape[1] < 2:
            raise ValueError("language modeling data must have sequence length >= 2")
        input_ids = tokens[:, :-1]
        targets = tokens[:, 1:]
        output = model(input_ids)
        logits = output.get("head_outputs", {}).get(head_name, output.get("logits"))
        if logits is None:
            raise ValueError(f"language-modeling head output is missing: {head_name}")
        vocab = int(logits.shape[-1])
        return torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))

    def sequence_log_probs(self, model: Any, tokens: Any, head_name: str) -> Any:
        torch = self.torch()
        if tokens.shape[1] < 2:
            raise ValueError("preference/DPO data must have sequence length >= 2")
        input_ids = tokens[:, :-1]
        targets = tokens[:, 1:]
        output = model(input_ids)
        logits = output.get("head_outputs", {}).get(head_name, output.get("logits"))
        if logits is None:
            raise ValueError(f"language-modeling head output is missing: {head_name}")
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        return torch.sum(gathered, dim=-1)

    def dpo_loss_and_metrics(
        self,
        model: Any,
        chosen: Any,
        rejected: Any,
        head_name: str,
        *,
        beta: float,
        reference_model: Any | None,
        prefix: str,
    ) -> tuple[Any, dict[str, float]]:
        torch = self.torch()
        chosen_logp = self.sequence_log_probs(model, chosen, head_name)
        rejected_logp = self.sequence_log_probs(model, rejected, head_name)
        policy_margin = chosen_logp - rejected_logp
        if reference_model is None:
            reference_margin = torch.zeros_like(policy_margin)
        else:
            with torch.no_grad():
                reference_margin = self.sequence_log_probs(reference_model, chosen, head_name) - self.sequence_log_probs(
                    reference_model,
                    rejected,
                    head_name,
                )
        logits = float(beta) * (policy_margin - reference_margin)
        loss = -torch.mean(torch.nn.functional.logsigmoid(logits))
        preference_accuracy = torch.mean((policy_margin > reference_margin).to(dtype=torch.float32))
        metrics = {
            f"{prefix}.dpo_loss": float(loss.detach().cpu().item()),
            f"{prefix}.preference_accuracy": float(preference_accuracy.detach().cpu().item()),
            f"{prefix}.policy_margin": float(torch.mean(policy_margin).detach().cpu().item()),
            f"{prefix}.reference_margin": float(torch.mean(reference_margin).detach().cpu().item()),
        }
        return loss, metrics

    def triplet_loss(self, anchor: Any, positive: Any, negative: Any, *, margin: float) -> tuple[Any, dict[str, float]]:
        torch = self.torch()
        pos_dist = torch.mean((anchor - positive) ** 2, dim=-1)
        neg_dist = torch.mean((anchor - negative) ** 2, dim=-1)
        loss = torch.mean(torch.relu(pos_dist - neg_dist + float(margin)))
        retrieval_accuracy = float(torch.mean((pos_dist < neg_dist).to(dtype=torch.float32)).detach().cpu().item())
        metrics = {
            "train.triplet_loss": float(loss.detach().cpu().item()),
            "train.retrieval_accuracy": retrieval_accuracy,
            "train.positive_distance": float(torch.mean(pos_dist).detach().cpu().item()),
            "train.negative_distance": float(torch.mean(neg_dist).detach().cpu().item()),
        }
        return loss, metrics

    def gaussian_nll(self, output: Any, target: Any, head_name: str) -> tuple[Any, Any, Any]:
        """Return differentiable Gaussian NLL and normalized distribution tensors."""
        torch = self.torch()
        head_output = output.get("head_outputs", {}).get(head_name, output) if isinstance(output, dict) else output
        if not isinstance(head_output, dict):
            raise ValueError(f"probabilistic head output must be a mapping; got {type(head_output).__name__}")
        mean = head_output.get("mu")
        log_scale = head_output.get("log_sigma")
        if mean is None or log_scale is None:
            raise ValueError(
                "probabilistic head output must contain 'mu' and 'log_sigma'; "
                f"got keys={list(head_output)}"
            )
        mean = mean.reshape(target.shape)
        log_scale = log_scale.reshape(target.shape)
        scale = torch.exp(log_scale).clamp(min=1e-4, max=1e6)
        normalizer = 0.5 * torch.log(torch.as_tensor(2.0 * torch.pi, dtype=mean.dtype, device=mean.device))
        loss = (normalizer + log_scale + 0.5 * ((target - mean) / scale) ** 2).mean()
        return loss, mean, scale

    def scalar(self, loss: Any) -> float:
        return float(loss.detach().cpu().item())

    def perplexity(self, cross_entropy: float) -> float:
        return float(np.exp(min(float(cross_entropy), 50.0)))


__all__ = ["TorchLossesCapability"]
