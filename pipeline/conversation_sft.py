"""Strict conversation input and explicit causal-LM label ownership.

No model dependencies are imported until tokenization is requested. This is a
data pipeline, not a Trainer or an optimization Adapter.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def load_conversations(path: str | Path) -> tuple[tuple[dict[str, str], ...], ...]:
    records = []
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                messages = row["messages"]
                if not isinstance(messages, list) or len(messages) < 2:
                    raise ValueError("messages must contain at least user and assistant")
                normalized = []
                for message in messages:
                    if not isinstance(message, Mapping):
                        raise ValueError("each message must be an object")
                    role, content = message.get("role"), message.get("content")
                    if role not in {"system", "user", "assistant"}:
                        raise ValueError("only system/user/assistant roles are supported")
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("message content must be a nonempty string")
                    normalized.append({"role": role, "content": content})
                roles = [item["role"] for item in normalized]
                if roles[-1] != "assistant" or "user" not in roles:
                    raise ValueError("conversation must include a user and end with assistant")
                # System is optional, but turns are strictly alternating. Tool
                # conversations require an explicit future data contract.
                turns = roles[1:] if roles[0] == "system" else roles
                if any(role != ("user" if i % 2 == 0 else "assistant")
                       for i, role in enumerate(turns)):
                    raise ValueError("user/assistant turns must alternate")
                records.append(tuple(normalized))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid conversation at line {line_number}: {exc}") from exc
    if not records:
        raise ValueError("conversation dataset is empty")
    return tuple(records)


def conversation_digest(records) -> str:
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tokenize_conversation(tokenizer: Any, messages, *, max_length: int,
                          label_mode: str = "assistant",
                          enable_thinking: bool | None = None) -> dict[str, list[int]]:
    """Tokenize without truncation; fail rather than silently drop supervision.

    assistant: template must declare Jinja generation blocks and return an
    assistant mask. completion: supervise only the final assistant continuation;
    the generation-prefix token IDs must be an exact prefix of the final text.
    """
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    if enable_thinking is not None and not isinstance(enable_thinking, bool):
        raise ValueError("enable_thinking must be boolean or None")
    template_options = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer.chat_template is required; no implicit template")
    if label_mode == "assistant":
        encoded = tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=False,
            return_dict=True, return_assistant_tokens_mask=True,
            **template_options,
        )
        ids = list(encoded["input_ids"])
        mask = encoded.get("assistant_masks")
        if mask is None or len(mask) != len(ids) or not any(mask):
            raise ValueError("assistant mask unavailable: chat template needs generation blocks")
        labels = [token if supervised else -100 for token, supervised in zip(ids, mask)]
    elif label_mode == "completion":
        ids = list(tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=False, return_dict=True,
            **template_options,
        )["input_ids"])
        prefix = list(tokenizer.apply_chat_template(
            list(messages[:-1]), tokenize=True, add_generation_prompt=True, return_dict=True,
            **template_options,
        )["input_ids"])
        if ids[:len(prefix)] != prefix or len(prefix) >= len(ids):
            raise ValueError("completion boundary is not an exact token prefix")
        labels = [-100] * len(prefix) + ids[len(prefix):]
    else:
        raise ValueError("label_mode must be assistant or completion")
    if len(ids) > int(max_length):
        raise ValueError(f"conversation has {len(ids)} tokens, exceeds max_length={max_length}; truncation is forbidden")
    if not any(value != -100 for value in labels[1:]):
        raise ValueError("conversation has no causally supervised target tokens")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def collate_conversations(rows, *, pad_token_id: int, torch_module=None):
    if torch_module is None:
        import torch as torch_module
    width = max(len(row["input_ids"]) for row in rows)
    return {
        key: torch_module.tensor([
            row[key] + [fill] * (width - len(row[key])) for row in rows
        ], dtype=torch_module.long)
        for key, fill in (("input_ids", int(pad_token_id)), ("attention_mask", 0), ("labels", -100))
    }


__all__ = ["load_conversations", "conversation_digest", "tokenize_conversation", "collate_conversations"]
