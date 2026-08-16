"""blackbase substrate 演示项目的阶段和 L0 资源声明。"""

L0 = {
    "namespace": "blackbase_demo",
    "offer": {"threads": 2, "gpus": 0, "backend": "local", "device_tokens": []},
    "policy": {"mode": "strict", "gpu_sharing": "exclusive", "cpu_oversubscribe": False},
    "default_request": {"threads": 1, "gpus": 0, "backend": "local"},
    "compute_backend": "auto",
    "execution_backend": "local",
}

STAGES = [
    {
        "name": "train",
        "cases": ["blackbase_demo"],
        "policy": "serial",
        "resource_requests": {
            "blackbase_demo": {"threads": 1, "gpus": 0, "backend": "local"}
        },
    }
]

GROUPS = {"default": {"stages": ["train"]}}
