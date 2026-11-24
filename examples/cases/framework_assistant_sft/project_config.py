"""CPU 默认授权；显式 MLBLACK_SFT_DEVICE 声明 Project GPU offer/request。"""
import os

_device = os.environ.get("MLBLACK_SFT_DEVICE", "cpu")
if _device != "cpu" and not (_device.startswith("cuda:") and _device[5:].isdigit()):
    raise ValueError("MLBLACK_SFT_DEVICE must be cpu or an explicit cuda:N token")
_tokens = [] if _device == "cpu" else [_device]
_request = {"threads": 1, "gpus": len(_tokens), "backend": "local", "device_tokens": _tokens}
PROJECT_NAME = "framework_assistant_sft"
L0 = {
    "namespace": "examples.framework_assistant_sft",
    "offer": dict(_request),
    "policy": {"mode": "strict", "cpu_oversubscribe": False},
    "default_request": dict(_request),
    "compute_backend": "auto", "execution_backend": "local",
}
STAGES = [{"name": "sft", "cases": ["assistant_sft"], "policy": "serial",
    "resource_requests": {"assistant_sft": dict(_request)},
    "timeout_seconds": 3600}]
GROUPS = {"default": {"stages": ["sft"]}}
