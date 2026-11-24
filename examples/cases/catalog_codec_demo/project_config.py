"""Catalog 的 CPU 小型线性回归 Project；资源授权由共享 L0 负责。"""

PROJECT_NAME = "catalog_codec_demo"
L0 = {
    "namespace": "examples.catalog_codec_demo",
    "offer": {"threads": 1, "gpus": 0, "backend": "local", "device_tokens": []},
    "policy": {"mode": "strict", "cpu_oversubscribe": False},
    "default_request": {"threads": 1, "gpus": 0, "backend": "local"},
    "compute_backend": "auto",
    "execution_backend": "local",
}
STAGES = [{
    "name": "linear",
    "cases": ["linear_point"],
    "policy": "serial",
    "resource_requests": {"linear_point": {"threads": 1, "gpus": 0, "backend": "local"}},
    "timeout_seconds": 60,
}]
GROUPS = {"default": {"stages": ["linear"]}}
