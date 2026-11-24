"""唯一装配入口：可编辑数据入口 + 共享 HF SFT Provider。"""
from mlblack.integrations.hf_sft import HFSFTConfig, build_hf_sft_solver
from .config import settings


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    options = {**settings(), **dict(config or {}), **dict(component_overrides or {})}
    dataset = options.pop("dataset_path")
    return build_hf_sft_solver(HFSFTConfig(**options), dataset, resource_context=resource_context)
