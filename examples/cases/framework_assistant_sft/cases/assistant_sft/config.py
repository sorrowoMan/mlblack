"""模型/语料路径必须显式配置；默认仅装配、预检，不下载权重。"""
import os
import json
from pathlib import Path


def settings():
    values = {
        "model_name_or_path": os.environ.get("MLBLACK_SFT_MODEL_PATH", ".models/framework-assistant-base"),
        "dataset_path": os.environ.get("MLBLACK_SFT_DATASET_PATH", str(Path(__file__).parent / "pipeline" / "example.jsonl")),
        "max_length": int(os.environ.get("MLBLACK_SFT_MAX_LENGTH", "2048")),
        "optimizer_steps": int(os.environ.get("MLBLACK_SFT_OPTIMIZER_STEPS", "20")), "batch_size": 1,
        "target_modules": tuple(os.environ.get("MLBLACK_SFT_TARGET_MODULES", "q_proj,v_proj").split(",")),
        "revision": os.environ.get("MLBLACK_SFT_REVISION") or None,
        "label_mode": "completion",
        "local_files_only": True,
    }
    config_path = os.environ.get("MLBLACK_SFT_CONFIG_PATH")
    if config_path:
        override = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(override, dict):
            raise ValueError("MLBLACK_SFT_CONFIG_PATH must contain a JSON object")
        values.update(override)
    return values
